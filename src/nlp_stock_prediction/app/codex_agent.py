"""Codex CLI adapter for report-aware agent chat."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from nlp_stock_prediction.app.settings import AppSettings


class CodexAgentError(RuntimeError):
    """Raised when the Codex subprocess cannot complete a chat turn."""


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


StdoutLineHandler = Callable[[str], None]
CodexEventHandler = Callable[[dict[str, object]], None]


class ProcessRunner(Protocol):
    def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        on_stdout_line: StdoutLineHandler | None = None,
    ) -> ProcessResult: ...


class SubprocessRunner:
    def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        on_stdout_line: StdoutLineHandler | None = None,
    ) -> ProcessResult:
        if on_stdout_line is not None:
            return self._run_streaming(command, cwd=cwd, on_stdout_line=on_stdout_line)
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return ProcessResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    def _run_streaming(
        self,
        command: list[str],
        *,
        cwd: Path,
        on_stdout_line: StdoutLineHandler,
    ) -> ProcessResult:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def read_stderr() -> None:
            if process.stderr is None:
                return
            for line in process.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        if process.stdout is not None:
            for line in process.stdout:
                stdout_lines.append(line)
                on_stdout_line(line)
        returncode = process.wait()
        stderr_thread.join()
        return ProcessResult(
            returncode=returncode,
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
        )


@dataclass(frozen=True)
class CodexHealth:
    codex_available: bool
    mcp_available: bool
    message: str


@dataclass(frozen=True)
class CodexTurnRequest:
    user_message: str
    repo_root: Path
    database_path: Path
    session_dir: Path
    settings: AppSettings
    report_json_path: Path | None = None
    report_markdown_path: Path | None = None
    session_id: str | None = None
    transcript_path: Path | None = None


@dataclass(frozen=True)
class CodexTurnResult:
    session_id: str
    message: str
    transcript_path: Path
    last_message_path: Path
    raw_events: tuple[dict[str, object], ...]


class CodexAgentAdapter:
    def __init__(
        self,
        *,
        runner: ProcessRunner | None = None,
        python_executable: Path | None = None,
    ) -> None:
        self._runner = runner or SubprocessRunner()
        self._python_executable = python_executable or Path(sys.executable)

    def health(self, settings: AppSettings, *, repo_root: Path | None = None) -> CodexHealth:
        codex_path = _resolve_codex_executable(settings.codex_executable)
        codex_available = codex_path is not None
        mcp_available = _python_has_mcp_server(self._python_executable, repo_root)
        if codex_available and mcp_available:
            return CodexHealth(True, True, "Codex agent chat is ready.")
        if not codex_available:
            return CodexHealth(
                False,
                mcp_available,
                (
                    "Codex CLI is not available. Add it to PATH, or set the Codex executable "
                    "in Settings to the full codex.exe path."
                ),
            )
        return CodexHealth(
            True,
            False,
            'Install the MCP extra first: python -m pip install -e ".[codex-smoke]"',
        )

    def send(
        self,
        request: CodexTurnRequest,
        *,
        on_event: CodexEventHandler | None = None,
    ) -> CodexTurnResult:
        request.session_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = request.transcript_path or request.session_dir / "transcript.jsonl"
        turn_index = _next_turn_index(transcript_path)
        last_message_path = request.session_dir / f"turn-{turn_index:04d}-last-message.md"
        prompt = _build_prompt(request)
        command = (
            self._resume_command(request, prompt, last_message_path)
            if request.session_id
            else self._start_command(request, prompt, last_message_path)
        )
        _append_transcript(transcript_path, role="user", content=request.user_message)
        result = self._runner.run(
            command,
            cwd=request.repo_root,
            on_stdout_line=_event_stream_handler(on_event) if on_event is not None else None,
        )
        if result.returncode != 0:
            _append_transcript(transcript_path, role="error", content=result.stderr.strip())
            raise CodexAgentError(result.stderr.strip() or "Codex chat turn failed.")
        events = parse_codex_jsonl(result.stdout)
        session_id = request.session_id or extract_session_id(events)
        if session_id is None:
            raise CodexAgentError("Codex did not report a resumable session id.")
        message = _read_last_message(last_message_path, events)
        _append_transcript(transcript_path, role="assistant", content=message)
        return CodexTurnResult(
            session_id=session_id,
            message=message,
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            raw_events=events,
        )

    def _base_command(self, request: CodexTurnRequest) -> list[str]:
        codex_executable = (
            _resolve_codex_executable(request.settings.codex_executable)
            or request.settings.codex_executable
        )
        command = [codex_executable, "--ask-for-approval", "never"]
        if request.settings.enable_web_search:
            command.append("--search")
        if request.settings.codex_model is not None:
            command.extend(["--model", request.settings.codex_model])
        if request.settings.codex_profile is not None:
            command.extend(["--profile", request.settings.codex_profile])
        command.extend(self._mcp_config_args(request.repo_root, request.database_path))
        return command

    def _start_command(
        self,
        request: CodexTurnRequest,
        prompt: str,
        last_message_path: Path,
    ) -> list[str]:
        return [
            *self._base_command(request),
            "exec",
            "--json",
            "-s",
            "danger-full-access",
            "-C",
            str(request.repo_root),
            "--output-last-message",
            str(last_message_path),
            prompt,
        ]

    def _resume_command(
        self,
        request: CodexTurnRequest,
        prompt: str,
        last_message_path: Path,
    ) -> list[str]:
        if request.session_id is None:
            raise ValueError("resume command requires session_id")
        return [
            *self._base_command(request),
            "exec",
            "resume",
            "--json",
            "--output-last-message",
            str(last_message_path),
            request.session_id,
            prompt,
        ]

    def _mcp_config_args(self, repo_root: Path, database_path: Path) -> list[str]:
        bootstrap = (
            "import sys; "
            f"sys.path.insert(0, {json.dumps(str(repo_root / 'src'))}); "
            "from nlp_stock_prediction.codex_mcp import main; main()"
        )
        args = json.dumps(
            [
                "-B",
                "-c",
                bootstrap,
                "--repo-root",
                str(repo_root),
                "--database",
                _path_arg(repo_root, database_path),
            ]
        )
        return [
            "-c",
            f"mcp_servers.nlp-stock-prediction.command={json.dumps(str(self._python_executable))}",
            "-c",
            f"mcp_servers.nlp-stock-prediction.args={args}",
        ]


def parse_codex_jsonl(payload: str | None) -> tuple[dict[str, object], ...]:
    events: list[dict[str, object]] = []
    for line in (payload or "").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return tuple(events)


def _event_stream_handler(on_event: CodexEventHandler) -> StdoutLineHandler:
    def handle_line(line: str) -> None:
        for event in parse_codex_jsonl(line):
            on_event(event)

    return handle_line


def extract_session_id(events: tuple[dict[str, object], ...]) -> str | None:
    for event in events:
        value = _session_id_from_object(event)
        if value is not None:
            return value
    return None


def _session_id_from_object(value: object) -> str | None:
    if isinstance(value, dict):
        for key in (
            "session_id",
            "sessionId",
            "thread_id",
            "threadId",
            "conversation_id",
            "conversationId",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        event_type = value.get("type")
        identifier = value.get("id")
        if isinstance(event_type, str) and ("session" in event_type or "thread" in event_type):
            if isinstance(identifier, str) and identifier.strip():
                return identifier.strip()
            payload = value.get("payload")
            if isinstance(payload, dict):
                payload_identifier = payload.get("id")
                if isinstance(payload_identifier, str) and payload_identifier.strip():
                    return payload_identifier.strip()
        for item in value.values():
            nested = _session_id_from_object(item)
            if nested is not None:
                return nested
    if isinstance(value, list):
        for item in value:
            nested = _session_id_from_object(item)
            if nested is not None:
                return nested
    return None


def _build_prompt(request: CodexTurnRequest) -> str:
    context_lines = [
        "You are the in-app Codex research assistant for nlp-stock-prediction.",
        "Use the local MCP research/evaluation tools when they help answer the user.",
        "You may be launched with full filesystem permissions and no approval prompts.",
        (
            "Treat this chat surface as read-only for source and project configuration: do not "
            "create, edit, delete, format, stage, commit, push, or otherwise mutate source files "
            "unless the user explicitly asks for source-code changes in this chat."
        ),
        (
            "Report/database/audit artifacts may be read, and MCP tools may write their normal "
            "declared artifacts when the user asks for research or evaluation work."
        ),
        (
            "If you use web search outside MCP-recorded tools, label those claims as unaudited "
            "chat context rather than report evidence."
        ),
        "Keep conclusions high-level and concise unless the user asks for audit detail.",
        (
            "Do not reveal private chain-of-thought. If the user asks how you are thinking, "
            "provide concise reasoning summaries, visible assumptions, and observable "
            "tool/activity details instead."
        ),
        "Do not provide trading instructions, position sizing, or buy/sell commands.",
        f"Research database: {_path_arg(request.repo_root, request.database_path)}",
    ]
    if request.report_json_path is not None:
        context_lines.append(
            f"Selected report JSON: {_path_arg(request.repo_root, request.report_json_path)}"
        )
    if request.report_markdown_path is not None:
        context_lines.append(
            "Selected report Markdown: "
            f"{_path_arg(request.repo_root, request.report_markdown_path)}"
        )
    context_lines.extend(["", "User message:", request.user_message])
    return "\n".join(context_lines)


def _read_last_message(
    last_message_path: Path,
    events: tuple[dict[str, object], ...],
) -> str:
    if last_message_path.exists():
        return last_message_path.read_text(encoding="utf-8").strip()
    for event in reversed(events):
        text = _text_from_event(event)
        if text is not None:
            return text
    return "Codex completed the turn but did not emit a final message."


def _text_from_event(value: object) -> str | None:
    if isinstance(value, dict):
        for key in ("message", "text", "content"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for item in value.values():
            nested = _text_from_event(item)
            if nested is not None:
                return nested
    if isinstance(value, list):
        for item in value:
            nested = _text_from_event(item)
            if nested is not None:
                return nested
    return None


def _append_transcript(path: Path, *, role: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "role": role,
        "content": content,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _next_turn_index(transcript_path: Path) -> int:
    if not transcript_path.exists():
        return 1
    return sum(1 for _line in transcript_path.read_text(encoding="utf-8").splitlines()) // 2 + 1


def _path_arg(repo_root: Path, path: Path) -> str:
    resolved = path if path.is_absolute() else repo_root / path
    try:
        return resolved.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_codex_executable(executable: str) -> str | None:
    configured = executable.strip() or "codex"
    configured_path = Path(configured)
    if configured_path.is_absolute() or configured_path.parent != Path("."):
        return str(configured_path) if configured_path.exists() else None

    path_match = shutil.which(configured)
    if path_match is not None:
        return path_match

    if configured.lower() not in {"codex", "codex.exe"}:
        return None
    for candidate in _codex_executable_candidates():
        if candidate.exists():
            return str(candidate)
    return None


def _codex_executable_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        local_root = Path(local_app_data)
        candidates.extend(
            [
                local_root / "OpenAI" / "Codex" / "bin" / "codex.exe",
                local_root / "OpenAI" / "Codex" / "bin" / "codex",
                local_root / "Microsoft" / "WindowsApps" / "codex.exe",
            ]
        )

    for env_name in ("ProgramFiles", "ProgramW6432"):
        program_files = os.environ.get(env_name)
        if not program_files:
            continue
        windows_apps = Path(program_files) / "WindowsApps"
        candidates.extend(windows_apps.glob("OpenAI.Codex_*/*/resources/codex.exe"))
        candidates.extend(windows_apps.glob("OpenAI.Codex_*/*/resources/codex"))

    return tuple(dict.fromkeys(candidates))


def _python_has_mcp_server(python_executable: Path, repo_root: Path | None) -> bool:
    src_path = None if repo_root is None else repo_root / "src"
    path_setup = "" if src_path is None else f"sys.path.insert(0, {json.dumps(str(src_path))}); "
    script = (
        f"import sys; {path_setup}import mcp.server.fastmcp; import nlp_stock_prediction.codex_mcp"
    )
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except OSError, subprocess.SubprocessError:
        return False
    return completed.returncode == 0


__all__ = [
    "CodexAgentAdapter",
    "CodexAgentError",
    "CodexEventHandler",
    "CodexHealth",
    "CodexTurnRequest",
    "CodexTurnResult",
    "ProcessResult",
    "ProcessRunner",
    "StdoutLineHandler",
    "extract_session_id",
    "parse_codex_jsonl",
]
