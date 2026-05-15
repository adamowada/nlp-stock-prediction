"""Persistent Rich terminal menu for the research assistant."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Protocol

from rich.console import Console, Group
from rich.json import JSON
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from nlp_stock_prediction.app.codex_agent import (
    CodexAgentAdapter,
    CodexAgentError,
    CodexTurnRequest,
    CodexTurnResult,
)
from nlp_stock_prediction.app.evaluation import (
    EVALUATION_COMMAND_SPECS,
    EvaluationServiceProtocol,
    build_evaluation_service,
    run_evaluation_action,
)
from nlp_stock_prediction.app.reports import (
    load_report_for_entry,
    render_markdown_report,
    render_report_details,
    render_report_summary,
    report_table,
)
from nlp_stock_prediction.app.research import (
    ReportGenerator,
    ResearchRequest,
    run_research_request,
)
from nlp_stock_prediction.app.settings import AppMode, AppSettings
from nlp_stock_prediction.app.state import (
    AppState,
    CodexSessionRecord,
    ReportIndexEntry,
    load_app_state,
    refresh_report_index,
    report_entry_from_bundle,
    save_app_state,
    utc_now,
)
from nlp_stock_prediction.environment import load_local_dotenv
from nlp_stock_prediction.pipeline import generate_daily_report
from nlp_stock_prediction.storage import initialize_research_database

InputFunc = Callable[[str], str]
_MAX_CODEX_ACTIVITY_LINES = 14
_MAX_ACTIVITY_LINE_LENGTH = 120
_MAX_ACTIVITY_VALUE_LENGTH = 48
_CLEAR_SCROLLBACK_SEQUENCE = "\x1b[3J"
_SENSITIVE_ACTIVITY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
    "user_agent",
    "user-agent",
)


class EvaluationServiceFactory(Protocol):
    def __call__(
        self,
        repo_root: Path,
        database_path: Path,
        extra_write_roots: tuple[Path, ...] = (),
    ) -> EvaluationServiceProtocol: ...


def run_app(
    *,
    repo_root: Path | None = None,
    console: Console | None = None,
    input_func: InputFunc | None = None,
) -> int:
    load_local_dotenv()
    app = TerminalApp(
        repo_root=(repo_root or Path.cwd()).resolve(),
        console=console,
        input_func=input_func,
    )
    return app.run()


class TerminalApp:
    def __init__(
        self,
        *,
        repo_root: Path,
        console: Console | None = None,
        input_func: InputFunc | None = None,
        report_generator: ReportGenerator | None = None,
        evaluation_service_factory: EvaluationServiceFactory | None = None,
        codex_adapter: CodexAgentAdapter | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.console = console or Console()
        self.input_func = input_func or input
        self.report_generator = report_generator or generate_daily_report
        self.evaluation_service_factory = evaluation_service_factory or build_evaluation_service
        self.codex_adapter = codex_adapter or CodexAgentAdapter()
        self.state = refresh_report_index(repo_root, load_app_state(repo_root))
        save_app_state(self.repo_root, self.state)

    def run(self) -> int:
        self._clear_screen()
        while True:
            self.console.print(_home_panel(self.state))
            choice = self._ask("Choose", default="1").strip().lower()
            if choice in {"6", "exit", "q", "quit"}:
                self._save()
                return 0
            self._clear_screen()
            if choice in {"1", "research", "r"}:
                self.research_menu()
            elif choice in {"2", "reports"}:
                self.reports_menu()
            elif choice in {"3", "evaluation", "e"}:
                self.evaluation_menu()
            elif choice in {"4", "agent", "chat", "c"}:
                self.agent_chat_menu()
            elif choice in {"5", "settings", "s"}:
                self.settings_menu()
            else:
                self.console.print("[yellow]Choose a listed option.[/yellow]")

    def research_menu(self) -> None:
        request = ResearchRequest.from_settings(self.state.settings)
        self.console.print(_research_panel(request))
        try:
            choice = self._ask("1 Run defaults, 2 Configure, 3 Back", default="1").strip().lower()
            if choice in {"3", "b", "back", "q"}:
                self._clear_screen()
                return
            self._clear_screen()
            if choice in {"2", "configure"}:
                request = self._configure_research_request(request)
            elif choice not in {"1", "run", "default", "defaults"}:
                self.console.print("[yellow]Choose a listed option.[/yellow]")
                return
            request = self._ensure_research_symbol(request)
            request = self._resolve_research_request(request)
            self._clear_screen()
            with self.console.status("Running research...", spinner="dots"):
                bundle = run_research_request(
                    request,
                    report_generator=self.report_generator,
                )
        except (KeyError, ValueError, OSError) as exc:
            self._clear_screen()
            self.console.print(Panel(str(exc), title="Research blocked", border_style="red"))
            return
        entry = report_entry_from_bundle(self.repo_root, bundle)
        self.state = self.state.upsert_report(entry)
        self._save()
        render_report_summary(bundle.report, console=self.console, title="Research Complete")

    def reports_menu(self) -> None:
        self.state = refresh_report_index(self.repo_root, self.state)
        self._save()
        if not self.state.reports:
            self.console.print("[yellow]No reports found under reports/ yet.[/yellow]")
            return
        while True:
            self.console.print(report_table(self.state.reports))
            choice = self._ask("Report #, or B back", default="1").strip().lower()
            if choice in {"b", "back", "q"}:
                self._clear_screen()
                return
            self._clear_screen()
            selected = self._report_by_choice(choice)
            if selected is None:
                self.console.print("[yellow]Choose a valid report number.[/yellow]")
                continue
            self.state = self.state.select_report(selected.report_id)
            self._save()
            self.report_actions(selected)

    def report_actions(self, entry: ReportIndexEntry) -> None:
        while True:
            try:
                report = load_report_for_entry(self.repo_root, entry)
            except (OSError, ValueError) as exc:
                self.console.print(Panel(str(exc), title="Report unavailable", border_style="red"))
                return
            render_report_summary(report, console=self.console)
            choice = self._ask(
                "1 Full report, 2 Details, 3 Paths, 4 Chat, 5 Back",
                default="1",
            ).strip()
            if choice == "5":
                self._clear_screen()
                return
            self._clear_screen()
            if choice == "1":
                try:
                    render_markdown_report(
                        entry.resolve_markdown_path(self.repo_root),
                        console=self.console,
                    )
                    self._clear_screen()
                except OSError as exc:
                    self.console.print(
                        Panel(str(exc), title="Report unavailable", border_style="red")
                    )
            elif choice == "2":
                render_report_details(report, console=self.console)
            elif choice == "3":
                self._render_report_paths(entry)
            elif choice == "4":
                self.agent_chat_menu()
            else:
                self.console.print("[yellow]Choose a listed option.[/yellow]")

    def evaluation_menu(self) -> None:
        selected = self.state.selected_report()
        database_path = self._default_database_path(selected)
        if database_path is None:
            value = self._ask("Research database path", default="data/prediction-research.sqlite3")
            database_path = self._resolve_path(value)
            self._clear_screen()
        if not database_path.exists():
            self.console.print(
                Panel(
                    f"Database not found: {database_path}",
                    title="Evaluation blocked",
                    border_style="red",
                )
            )
            return
        table = Table(title="Evaluation Commands", expand=True)
        table.add_column("#", justify="right")
        table.add_column("Command")
        table.add_column("Description")
        for index, spec in enumerate(EVALUATION_COMMAND_SPECS, start=1):
            table.add_row(str(index), spec.name, spec.label)
        self.console.print(table)
        choice = self._ask("Command #, or B back", default="1").strip().lower()
        if choice in {"b", "back", "q"}:
            self._clear_screen()
            return
        self._clear_screen()
        try:
            index = int(choice)
        except ValueError:
            self.console.print("[yellow]Choose a valid command number.[/yellow]")
            return
        if index < 1 or index > len(EVALUATION_COMMAND_SPECS):
            self.console.print("[yellow]Choose a valid command number.[/yellow]")
            return
        spec = EVALUATION_COMMAND_SPECS[index - 1]
        values = self._collect_evaluation_values(spec.name, selected)
        self._clear_screen()
        try:
            service = self.evaluation_service_factory(
                self.repo_root,
                database_path,
                self._evaluation_extra_write_roots(selected, database_path),
            )
            result = run_evaluation_action(service, spec.name, values)
        except ValueError as exc:
            self.console.print(Panel(str(exc), title="Evaluation blocked", border_style="red"))
            return
        self.console.print(
            Panel(
                JSON(json.dumps(result, default=str, sort_keys=True)),
                title=f"{spec.name} result",
            )
        )

    def agent_chat_menu(self) -> None:
        health = self.codex_adapter.health(self.state.settings, repo_root=self.repo_root)
        if not (health.codex_available and health.mcp_available):
            self.console.print(
                Panel(health.message, title="Codex Agent Health", border_style="yellow")
            )
            return
        selected = self.state.selected_report()
        database_path = self._agent_chat_database_path(selected)
        if not database_path.exists():
            initialize_research_database(database_path)
        session_key = selected.report_id if selected is not None else "global"
        session = self.state.codex_session(session_key)
        self.console.print(Panel("Type /back to return to the app menu.", title="Codex Agent Chat"))
        while True:
            message = self._ask("You", default="/back")
            if message.strip().lower() in {"/back", "back", "exit", "quit"}:
                self._clear_screen()
                self._save()
                return
            self._clear_screen()
            try:
                result = self._send_codex_turn_with_activity(
                    self._codex_turn_request(
                        message=message,
                        selected=selected,
                        database_path=database_path,
                        session_key=session_key,
                        session=session,
                    )
                )
            except CodexAgentError as exc:
                if session is not None and _looks_like_stale_codex_session(str(exc)):
                    self.console.print(
                        "[yellow]Stored Codex session could not be resumed; "
                        "starting fresh.[/yellow]"
                    )
                    session = None
                    try:
                        result = self._send_codex_turn_with_activity(
                            self._codex_turn_request(
                                message=message,
                                selected=selected,
                                database_path=database_path,
                                session_key=session_key,
                                session=None,
                            )
                        )
                    except CodexAgentError as retry_exc:
                        self.console.print(
                            Panel(str(retry_exc), title="Codex turn failed", border_style="red")
                        )
                        continue
                else:
                    self.console.print(
                        Panel(str(exc), title="Codex turn failed", border_style="red")
                    )
                    continue
            session = CodexSessionRecord(
                session_key=session_key,
                session_id=result.session_id,
                report_id=selected.report_id if selected is not None else None,
                database_path=_relative_or_absolute(self.repo_root, database_path),
                transcript_path=_relative_or_absolute(self.repo_root, result.transcript_path),
                updated_at=utc_now(),
            )
            self.state = self.state.upsert_codex_session(session)
            self._save()
            self.console.print(_codex_response_panel(result.message))

    def settings_menu(self) -> None:
        while True:
            self.console.print(
                _settings_panel(self.state.settings, self.codex_adapter, self.repo_root)
            )
            choice = self._ask(
                "1 Symbol, 2 Mode, 3 Output, 4 Cache, 5 Codex, 6 Back",
                default="6",
            ).strip()
            settings = self.state.settings
            if choice == "1":
                self._clear_screen()
                settings = settings.with_updates(
                    default_symbol=self._ask(
                        "Remembered symbol (blank asks each run)",
                        default=settings.default_symbol,
                    )
                )
            elif choice == "2":
                self._clear_screen()
                mode = (
                    self._ask(
                        "Default mode live/offline",
                        default=settings.default_mode,
                    )
                    .strip()
                    .lower()
                )
                if mode not in {"live", "offline"}:
                    self._clear_screen()
                    self.console.print("[yellow]Mode must be live or offline.[/yellow]")
                    continue
                settings = settings.with_updates(default_mode=mode)
            elif choice == "3":
                self._clear_screen()
                settings = settings.with_updates(
                    output_dir=Path(self._ask("Output directory", default=str(settings.output_dir)))
                )
            elif choice == "4":
                self._clear_screen()
                settings = settings.with_updates(
                    cache_dir=Path(self._ask("Cache directory", default=str(settings.cache_dir)))
                )
            elif choice == "5":
                self._clear_screen()
                settings = self._configure_codex(settings)
            elif choice == "6":
                self._clear_screen()
                self._save()
                return
            else:
                self._clear_screen()
                self.console.print("[yellow]Choose a listed option.[/yellow]")
                continue
            self.state = replace(self.state, settings=settings)
            self._save()
            self._clear_screen()

    def _configure_research_request(self, request: ResearchRequest) -> ResearchRequest:
        symbol = self._ask("Symbol", default=request.symbol).strip().upper()
        raw_date = self._ask("Report date", default=request.run_date.isoformat()).strip()
        mode = self._ask("Mode live/offline", default=request.mode).strip().lower()
        output = self._ask("Output directory", default=str(request.output_dir)).strip()
        cache = self._ask("Cache directory", default=str(request.cache_dir or "cache")).strip()
        if mode not in {"live", "offline"}:
            raise ValueError("Mode must be 'live' or 'offline'.")
        resolved_mode: AppMode = "offline" if mode == "offline" else "live"
        return ResearchRequest(
            symbol=symbol or request.symbol,
            run_date=date.fromisoformat(raw_date),
            mode=resolved_mode,
            output_dir=Path(output or request.output_dir),
            cache_dir=Path(cache) if cache else None,
            fixture_dir=request.fixture_dir,
        )

    def _collect_evaluation_values(
        self,
        command: str,
        selected: ReportIndexEntry | None,
    ) -> dict[str, object]:
        spec = next(item for item in EVALUATION_COMMAND_SPECS if item.name == command)
        values: dict[str, object] = {}
        for field in spec.fields:
            default = self._field_default(field.name, selected) or field.default
            prompt = field.label if field.required else f"{field.label} (optional)"
            raw_value = self._ask(prompt, default=default or "")
            if raw_value.strip() or field.required:
                values[field.name] = raw_value.strip()
        return values

    def _field_default(self, field_name: str, selected: ReportIndexEntry | None) -> str | None:
        if selected is None:
            return None
        if field_name == "run_id":
            return selected.run_id
        if field_name == "artifact_dir":
            return selected.audit_dir.as_posix()
        if field_name == "report_date":
            return selected.report_date.isoformat()
        return None

    def _configure_codex(self, settings: AppSettings) -> AppSettings:
        executable = self._ask("Codex executable", default=settings.codex_executable)
        model = self._ask("Codex model (blank for default)", default=settings.codex_model or "")
        profile = self._ask(
            "Codex profile (blank for default)",
            default=settings.codex_profile or "",
        )
        web_search = self._ask(
            "Enable Codex web search yes/no",
            default="yes" if settings.enable_web_search else "no",
        )
        return settings.with_updates(
            codex_executable=executable,
            codex_model=model or None,
            codex_profile=profile or None,
            enable_web_search=web_search.strip().lower() not in {"no", "n", "0", "false"},
        )

    def _default_database_path(self, selected: ReportIndexEntry | None) -> Path | None:
        if selected is None:
            return None
        database_path = selected.resolve_database_path(self.repo_root)
        return database_path if database_path is not None else None

    def _agent_chat_database_path(self, selected: ReportIndexEntry | None) -> Path:
        default_database_path = self.repo_root / "data/prediction-research.sqlite3"
        selected_database_path = self._default_database_path(selected)
        if selected_database_path is not None and selected_database_path.exists():
            return selected_database_path
        return default_database_path

    def _ensure_research_symbol(self, request: ResearchRequest) -> ResearchRequest:
        if request.symbol.strip():
            return request
        symbol = self._ask("Symbol", default="").strip().upper()
        if not symbol:
            raise ValueError("Symbol is required.")
        return replace(request, symbol=symbol)

    def _resolve_research_request(self, request: ResearchRequest) -> ResearchRequest:
        return ResearchRequest(
            symbol=request.symbol,
            run_date=request.run_date,
            mode=request.mode,
            output_dir=self._resolve_path(request.output_dir),
            cache_dir=(self._resolve_path(request.cache_dir) if request.cache_dir else None),
            fixture_dir=(self._resolve_path(request.fixture_dir) if request.fixture_dir else None),
        )

    def _codex_turn_request(
        self,
        *,
        message: str,
        selected: ReportIndexEntry | None,
        database_path: Path,
        session_key: str,
        session: CodexSessionRecord | None,
    ) -> CodexTurnRequest:
        return CodexTurnRequest(
            user_message=message,
            repo_root=self.repo_root,
            database_path=database_path,
            session_dir=self.repo_root / "data" / "codex-sessions" / _slug(session_key),
            settings=self.state.settings,
            report_json_path=(selected.resolve_json_path(self.repo_root) if selected else None),
            report_markdown_path=(
                selected.resolve_markdown_path(self.repo_root) if selected else None
            ),
            session_id=session.session_id if session else None,
            transcript_path=(
                self._resolve_path(session.transcript_path) if session is not None else None
            ),
        )

    def _send_codex_turn_with_activity(self, request: CodexTurnRequest) -> CodexTurnResult:
        activity = _CodexActivity()
        with Live(
            activity.render(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        ) as live:

            def handle_event(event: dict[str, object]) -> None:
                activity.record_event(event)
                live.update(activity.render())

            try:
                result = self.codex_adapter.send(
                    request,
                    on_event=handle_event,
                )
            except CodexAgentError:
                activity.record_status("Codex turn failed.")
                live.update(activity.render(done=True))
                raise
            live.update(activity.render(done=True))
            return result

    def _evaluation_extra_write_roots(
        self,
        selected: ReportIndexEntry | None,
        database_path: Path,
    ) -> tuple[Path, ...]:
        candidates = [database_path.parent]
        if selected is not None:
            candidates.extend(
                [
                    selected.resolve_audit_dir(self.repo_root),
                    selected.resolve_json_path(self.repo_root).parent,
                    selected.resolve_markdown_path(self.repo_root).parent,
                ]
            )
        roots: dict[str, Path] = {}
        for candidate in candidates:
            resolved = candidate.resolve()
            roots[resolved.as_posix()] = resolved
        return tuple(roots.values())

    def _report_by_choice(self, choice: str) -> ReportIndexEntry | None:
        try:
            index = int(choice) - 1
        except ValueError:
            return None
        if index < 0 or index >= len(self.state.reports):
            return None
        return self.state.reports[index]

    def _render_report_paths(self, entry: ReportIndexEntry) -> None:
        table = Table(title="Report Paths", expand=True)
        table.add_column("Artifact")
        table.add_column("Path")
        table.add_row("Markdown", str(entry.resolve_markdown_path(self.repo_root)))
        table.add_row("JSON", str(entry.resolve_json_path(self.repo_root)))
        table.add_row("Audit", str(entry.resolve_audit_dir(self.repo_root)))
        database = entry.resolve_database_path(self.repo_root)
        table.add_row("Database", str(database) if database is not None else "unknown")
        self.console.print(table)

    def _ask(self, prompt: str, *, default: str) -> str:
        suffix = f" [{default}]" if default else ""
        value = self.input_func(f"{prompt}{suffix}: ")
        return value if value.strip() else default

    def _clear_screen(self) -> None:
        if self.console.is_terminal:
            self.console.file.write(_CLEAR_SCROLLBACK_SEQUENCE)
            self.console.file.flush()
        self.console.clear()

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo_root / path

    def _save(self) -> None:
        save_app_state(self.repo_root, self.state)


def _home_panel(state: AppState) -> Panel:
    selected = state.selected_report()
    selected_text = (
        f"{selected.symbol} {selected.report_date.isoformat()}" if selected is not None else "none"
    )
    default_symbol = state.settings.default_symbol or "choose symbol"
    body = "\n".join(
        [
            "1 Research",
            "2 Reports",
            "3 Evaluation",
            "4 Agent Chat",
            "5 Settings",
            "6 Exit",
            "",
            f"Default: {default_symbol} / {state.settings.default_mode}",
            f"Selected report: {selected_text}",
        ]
    )
    return Panel(body, title="Prediction Research Assistant", border_style="cyan")


def _research_panel(request: ResearchRequest) -> Panel:
    body = "\n".join(
        [
            f"Symbol: {request.symbol or 'choose before run'}",
            f"Date: {request.run_date.isoformat()}",
            f"Mode: {request.mode}",
            f"Output: {request.output_dir}",
            f"Cache: {request.cache_dir or 'none'}",
        ]
    )
    return Panel(body, title="Research Defaults", border_style="blue")


def _settings_panel(
    settings: AppSettings,
    adapter: CodexAgentAdapter,
    repo_root: Path,
) -> Panel:
    health = adapter.health(settings, repo_root=repo_root)
    body = "\n".join(
        [
            f"Remembered symbol: {settings.default_symbol or 'ask each run'}",
            f"Default mode: {settings.default_mode}",
            f"Output directory: {settings.output_dir}",
            f"Cache directory: {settings.cache_dir}",
            f"Codex executable: {settings.codex_executable}",
            f"Codex model: {settings.codex_model or 'default'}",
            f"Codex profile: {settings.codex_profile or 'default'}",
            f"Web search: {'on' if settings.enable_web_search else 'off'}",
            f"Agent health: {health.message}",
        ]
    )
    return Panel(body, title="Settings", border_style="green")


def _codex_response_panel(message: str) -> Panel:
    return Panel(Markdown(message), title="Codex")


def _relative_or_absolute(repo_root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    try:
        return resolved.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return resolved


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return slug[:80] or "global"


def _looks_like_stale_codex_session(message: str) -> bool:
    normalized = message.lower()
    return "session" in normalized and any(
        marker in normalized
        for marker in ("not found", "no such", "could not", "cannot", "invalid", "resume")
    )


class _CodexActivity:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self.record_status("Starting Codex turn.")

    def record_event(self, event: dict[str, object]) -> None:
        line = _codex_activity_line(event)
        if line is not None:
            self.record_status(line)

    def record_status(self, message: str) -> None:
        line = message.strip()
        if not line or (self._lines and self._lines[-1] == line):
            return
        self._lines.append(line)

    def render(self, *, done: bool = False) -> Panel:
        status = (
            Text("Response ready", style="green")
            if done
            else Spinner("dots", text=Text("Thinking", style="cyan"))
        )
        body = Group(
            status,
            Text("\n".join(self._lines[-_MAX_CODEX_ACTIVITY_LINES:]), style="dim"),
        )
        return Panel(
            body,
            title="Codex Activity",
            border_style="green" if done else "cyan",
        )


def _codex_activity_line(event: dict[str, object]) -> str | None:
    event_type = _string_field(event, "type", "event", "kind")
    item = _dict_field(event, "item") or _dict_field(event, "payload")
    item_type = _string_field(item, "type") if item is not None else None

    if event_type == "thread.started":
        return "Session connected."
    if event_type == "turn.started":
        return "Turn started."
    if event_type == "turn.completed":
        return _turn_completed_line(event)
    if event_type is not None and "error" in event_type.lower():
        return "Codex reported an error."

    if item is not None:
        stage = _activity_state(event_type, _string_field(item, "status"))
        if item_type == "command_execution":
            return _command_activity_line(item, stage)
        if _is_tool_item(item_type):
            return _tool_activity_line(event, item, stage)
        if item_type == "reasoning":
            return _reasoning_activity_line(item)
        if item_type == "agent_message":
            return None
        if event_type is not None and event_type.startswith("item."):
            return None

    if event_type is not None:
        return f"Observed Codex event: {_humanize_event_type(event_type)}."
    return None


def _activity_state(event_type: str | None, status: str | None) -> str:
    normalized_status = (status or "").lower()
    normalized_event = (event_type or "").lower()
    if normalized_status in {"failed", "error"}:
        return "Failed"
    if normalized_status in {"completed", "succeeded", "success"}:
        return "Finished"
    if normalized_status in {"in_progress", "running", "started"}:
        return "Started"
    if normalized_event.endswith(".completed"):
        return "Finished"
    if normalized_event.endswith(".started"):
        return "Started"
    return "Observed"


def _command_activity_line(item: dict[str, object], stage: str) -> str:
    command = _display_name(_string_field(item, "command"))
    exit_code = item.get("exit_code")
    if stage == "Finished":
        suffix = f" (exit {exit_code})" if isinstance(exit_code, int) else ""
        return _activity_sentence(f"Finished shell command{suffix}", command)
    if stage == "Failed":
        suffix = f" (exit {exit_code})" if isinstance(exit_code, int) else ""
        return _activity_sentence(f"Failed shell command{suffix}", command)
    return _activity_sentence("Running shell command", command)


def _tool_activity_line(
    event: dict[str, object],
    item: dict[str, object],
    stage: str,
) -> str:
    name = _tool_name(event, item)
    server = _string_field(item, "server") or _string_field(event, "server")
    arguments = _argument_summary(_arguments_payload(item) or _arguments_payload(event))
    result = _result_summary(item) if stage in {"Finished", "Failed"} else None
    if name is None:
        name = f"MCP tool on {server}" if server is not None else "tool"
    prefix = f"{stage} MCP tool" if server is not None else f"{stage} tool"
    detail = f"{name}{arguments}"
    if result is not None:
        detail = f"{detail}; {result}"
    return _activity_sentence(prefix, detail)


def _tool_name(event: dict[str, object], item: dict[str, object]) -> str | None:
    for payload in (item, event):
        name = _string_field(
            payload,
            "tool_name",
            "toolName",
            "name",
            "function",
            "tool",
            "command",
        )
        if name is not None:
            return _display_name(name)
    return None


def _is_tool_item(item_type: str | None) -> bool:
    if item_type is None:
        return False
    normalized = item_type.lower()
    return any(marker in normalized for marker in ("tool", "function_call", "mcp"))


def _reasoning_activity_line(item: dict[str, object]) -> str:
    summary = _reasoning_summary(item)
    if summary is not None:
        return _activity_sentence("Reasoning", summary)
    return "Reasoning through evidence and next steps."


def _reasoning_summary(item: dict[str, object]) -> str | None:
    summary = item.get("summary")
    if not isinstance(summary, list):
        return None
    texts: list[str] = []
    for entry in summary:
        if not isinstance(entry, dict):
            continue
        text = _string_field(entry, "text")
        if text is not None:
            texts.append(text)
    if not texts:
        return None
    return _display_name(" ".join(texts))


def _turn_completed_line(event: dict[str, object]) -> str:
    usage = _dict_field(event, "usage")
    if usage is None:
        return "Turn completed."
    output_tokens = usage.get("output_tokens")
    reasoning_tokens = usage.get("reasoning_output_tokens")
    details: list[str] = []
    if isinstance(output_tokens, int):
        details.append(f"output {output_tokens}")
    if isinstance(reasoning_tokens, int):
        details.append(f"reasoning {reasoning_tokens}")
    if not details:
        return "Turn completed."
    return f"Turn completed ({', '.join(details)} tokens)."


def _arguments_payload(payload: dict[str, object]) -> object | None:
    for key in ("arguments", "args", "input", "parameters"):
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _argument_summary(payload: object | None) -> str:
    arguments = _coerce_mapping(payload)
    if not arguments:
        return ""
    parts: list[str] = []
    for key, value in arguments.items():
        if len(parts) >= 4:
            parts.append("...")
            break
        if _is_sensitive_activity_key(key):
            parts.append(f"{key}=<redacted>")
            continue
        parts.append(f"{key}={_format_activity_value(value)}")
    return f"({', '.join(parts)})" if parts else ""


def _result_summary(item: dict[str, object]) -> str | None:
    if (error := _string_field(item, "error")) is not None:
        return f"error={_display_name(error)}"
    for key in ("result", "output"):
        value = item.get(key)
        if value is not None:
            return _structured_result_summary(value)
    output = _string_field(item, "aggregated_output")
    if output is not None:
        text = _first_nonempty_line(output)
        return f"output={text}" if text is not None else None
    return None


def _structured_result_summary(value: object) -> str | None:
    mapping = _coerce_mapping(value)
    if mapping:
        preferred = (
            "status",
            "run_id",
            "symbol",
            "report_path",
            "markdown_path",
            "json_path",
            "evidence_count",
            "candidate_count",
            "artifact_count",
            "warning_count",
        )
        parts = [
            f"{key}={_format_activity_value(mapping[key])}"
            for key in preferred
            if key in mapping and not _is_sensitive_activity_key(key)
        ]
        if parts:
            return ", ".join(parts[:4])
    if isinstance(value, str):
        text = _first_nonempty_line(value)
        if text is not None:
            return f"result={text}"
    return None


def _coerce_mapping(value: object | None) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _format_activity_value(value: object) -> str:
    if isinstance(value, str):
        return _redacted_or_display_value(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    if isinstance(value, list):
        return f"{len(value)} items"
    if isinstance(value, dict):
        return f"{len(value)} fields"
    return _redacted_or_display_value(str(value))


def _redacted_or_display_value(value: str) -> str:
    if _looks_sensitive_activity_text(value):
        return "<redacted>"
    compacted = _compact_activity_path(value)
    return _display_name(compacted or value, max_length=_MAX_ACTIVITY_VALUE_LENGTH) or ""


def _first_nonempty_line(value: str) -> str | None:
    for line in value.splitlines():
        stripped = line.strip()
        if stripped:
            return _redacted_or_display_value(stripped)
    return None


def _activity_sentence(prefix: str, detail: str | None) -> str:
    if detail is None:
        return f"{prefix}."
    trimmed = _display_name(
        detail,
        max_length=max(20, _MAX_ACTIVITY_LINE_LENGTH - len(prefix) - 3),
    )
    if not trimmed:
        return f"{prefix}."
    if trimmed.endswith("..."):
        return f"{prefix}: {trimmed}"
    return f"{prefix}: {trimmed.rstrip('.')}."


def _string_field(payload: dict[str, object] | None, *keys: str) -> str | None:
    if payload is None:
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _dict_field(payload: dict[str, object], key: str) -> dict[str, object] | None:
    value = payload.get(key)
    return value if isinstance(value, dict) else None


def _is_sensitive_activity_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(marker in normalized for marker in _SENSITIVE_ACTIVITY_MARKERS)


def _looks_sensitive_activity_text(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in _SENSITIVE_ACTIVITY_MARKERS)


def _display_name(value: str | None, *, max_length: int = _MAX_ACTIVITY_LINE_LENGTH) -> str | None:
    if value is None:
        return None
    name = " ".join(value.strip().split())
    if not name:
        return None
    if len(name) > max_length:
        return f"{name[: max_length - 3]}..."
    return name


def _compact_activity_path(value: str) -> str | None:
    normalized = value.replace("\\", "/")
    if "://" in normalized:
        return None
    if "/" not in normalized:
        return None
    looks_like_path = ":/" in normalized or normalized.startswith(("/", "./", "../"))
    if not looks_like_path:
        return None
    parts = [part for part in normalized.split("/") if part]
    if len(parts) <= 3:
        return normalized
    return ".../" + "/".join(parts[-3:])


def _humanize_event_type(value: str) -> str:
    label = value.replace("_", " ").replace("-", " ").strip()
    if not label:
        return "event"
    return label[:80]


__all__ = ["TerminalApp", "run_app"]
