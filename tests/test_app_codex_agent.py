from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from nlp_stock_prediction.app.codex_agent import (
    CodexAgentAdapter,
    CodexTurnRequest,
    ProcessResult,
    StdoutLineHandler,
    SubprocessRunner,
    _resolve_codex_executable,
    extract_session_id,
    parse_codex_jsonl,
)
from nlp_stock_prediction.app.settings import AppSettings

pytestmark = pytest.mark.unit


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        on_stdout_line: StdoutLineHandler | None = None,
    ) -> ProcessResult:
        del cwd
        self.commands.append(command)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("High-level conclusion from Codex.\n", encoding="utf-8")
        stdout = json.dumps({"type": "thread.started", "thread_id": "thread-123"}) + "\n"
        if on_stdout_line is not None:
            on_stdout_line(stdout)
        return ProcessResult(
            returncode=0,
            stdout=stdout,
            stderr="",
        )


def test_codex_jsonl_session_id_parsing() -> None:
    events = parse_codex_jsonl(
        "\n".join(
            [
                '{"type":"noise"}',
                '{"type":"thread.started","thread_id":"thread-abc"}',
                '{"type":"session_meta","payload":{"id":"session-abc"}}',
                "not-json",
            ]
        )
    )

    assert extract_session_id(events) == "thread-abc"


def test_codex_jsonl_session_id_parses_session_meta_payload() -> None:
    events = parse_codex_jsonl('{"type":"session_meta","payload":{"id":"session-abc"}}\n')

    assert extract_session_id(events) == "session-abc"


def test_codex_jsonl_parser_treats_missing_output_as_empty() -> None:
    assert parse_codex_jsonl(None) == ()


def test_codex_jsonl_parser_accepts_json_array_payloads() -> None:
    events = parse_codex_jsonl(
        '[{"type":"thread.started","thread_id":"thread-array"},{"type":"turn.completed"}]\n'
    )

    assert events == (
        {"type": "thread.started", "thread_id": "thread-array"},
        {"type": "turn.completed"},
    )


def test_subprocess_runner_decodes_codex_output_as_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_kwargs: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del command
        seen_kwargs.update(kwargs)
        return SimpleNamespace(returncode=0, stdout='{"type":"done"}\n', stderr=None)

    monkeypatch.setattr("nlp_stock_prediction.app.codex_agent.subprocess.run", fake_run)

    result = SubprocessRunner().run(["codex", "--json"], cwd=tmp_path)

    assert result.stdout == '{"type":"done"}\n'
    assert result.stderr == ""
    assert seen_kwargs["encoding"] == "utf-8"
    assert seen_kwargs["errors"] == "replace"
    assert seen_kwargs["text"] is True


def test_subprocess_runner_streams_codex_stdout_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_kwargs: dict[str, object] = {}

    class FakeProcess:
        stdout = StringIO('{"type":"thread.started","thread_id":"thread-123"}\nnot-json\n')
        stderr = StringIO("warning\n")

        def wait(self) -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        del command
        seen_kwargs.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("nlp_stock_prediction.app.codex_agent.subprocess.Popen", fake_popen)

    streamed_lines: list[str] = []
    result = SubprocessRunner().run(
        ["codex", "--json"],
        cwd=tmp_path,
        on_stdout_line=streamed_lines.append,
    )

    assert streamed_lines == [
        '{"type":"thread.started","thread_id":"thread-123"}\n',
        "not-json\n",
    ]
    assert result.stdout == '{"type":"thread.started","thread_id":"thread-123"}\nnot-json\n'
    assert result.stderr == "warning\n"
    assert result.returncode == 0
    assert seen_kwargs["encoding"] == "utf-8"
    assert seen_kwargs["errors"] == "replace"
    assert seen_kwargs["text"] is True


def test_codex_adapter_starts_and_resumes_session(tmp_path: Path) -> None:
    runner = FakeRunner()
    adapter = CodexAgentAdapter(runner=runner, python_executable=Path("python"))
    settings = AppSettings(codex_executable="codex", enable_web_search=True)

    first = adapter.send(
        CodexTurnRequest(
            user_message="Summarize the selected report.",
            repo_root=tmp_path,
            database_path=tmp_path / "data" / "prediction-research.sqlite3",
            session_dir=tmp_path / "data" / "codex-sessions" / "global",
            settings=settings,
        )
    )
    second = adapter.send(
        CodexTurnRequest(
            user_message="What changed?",
            repo_root=tmp_path,
            database_path=tmp_path / "data" / "prediction-research.sqlite3",
            session_dir=tmp_path / "data" / "codex-sessions" / "global",
            settings=settings,
            session_id=first.session_id,
            transcript_path=first.transcript_path,
        )
    )

    assert first.session_id == "thread-123"
    assert second.session_id == "thread-123"
    assert "exec" in runner.commands[0]
    assert "resume" in runner.commands[1]
    assert "--search" in runner.commands[0]
    assert "mcp_servers.nlp-stock-prediction.args" in " ".join(runner.commands[0])
    assert "full filesystem permissions" in runner.commands[0][-1]
    assert "do not create, edit, delete, format, stage, commit, push" in runner.commands[0][-1]
    assert "Do not reveal private chain-of-thought" in runner.commands[0][-1]
    assert "0.0-1.0 research/evidence scales" in runner.commands[0][-1]
    assert "source reliability counts" in runner.commands[0][-1]
    assert "trading-strategy context" in runner.commands[0][-1]
    assert "Do not place trades, size positions" in runner.commands[0][-1]
    transcript = first.transcript_path.read_text(encoding="utf-8")
    assert "Summarize the selected report" in transcript
    assert "High-level conclusion from Codex." in transcript


def test_codex_adapter_forwards_streamed_events(tmp_path: Path) -> None:
    runner = FakeRunner()
    adapter = CodexAgentAdapter(runner=runner, python_executable=Path("python"))
    settings = AppSettings(codex_executable="codex", enable_web_search=False)
    events: list[dict[str, object]] = []

    adapter.send(
        CodexTurnRequest(
            user_message="Summarize the selected report.",
            repo_root=tmp_path,
            database_path=tmp_path / "data" / "prediction-research.sqlite3",
            session_dir=tmp_path / "data" / "codex-sessions" / "global",
            settings=settings,
        ),
        on_event=events.append,
    )

    assert events == [{"type": "thread.started", "thread_id": "thread-123"}]


def test_codex_executable_falls_back_to_windows_install_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = tmp_path / "OpenAI" / "Codex" / "bin" / "codex.exe"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("", encoding="utf-8")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr("nlp_stock_prediction.app.codex_agent.shutil.which", lambda _: None)

    assert _resolve_codex_executable("codex") == str(fallback)
