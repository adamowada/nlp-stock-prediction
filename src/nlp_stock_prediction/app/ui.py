"""Persistent Rich terminal menu for the research assistant."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Protocol

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from nlp_stock_prediction.app.codex_agent import (
    CodexAgentAdapter,
    CodexAgentError,
    CodexTurnRequest,
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
        database_path = self._default_database_path(selected)
        if selected is not None and database_path is None:
            self.console.print(
                Panel(
                    "Selected report does not have a recovered research database path. "
                    "Re-run it from the app, or use a report with a known database path.",
                    title="Codex Agent Health",
                    border_style="yellow",
                )
            )
            return
        if database_path is None:
            database_path = self.repo_root / "data/prediction-research.sqlite3"
        if selected is not None and not database_path.exists():
            self.console.print(
                Panel(
                    f"Selected report database not found: {database_path}",
                    title="Codex Agent Health",
                    border_style="yellow",
                )
            )
            return
        if selected is None and not database_path.exists():
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
                result = self.codex_adapter.send(
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
                        result = self.codex_adapter.send(
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
            self.console.print(Panel(result.message, title="Codex"))

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


__all__ = ["TerminalApp", "run_app"]
