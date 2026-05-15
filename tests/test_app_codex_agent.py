from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_stock_prediction.app.codex_agent import (
    CodexAgentAdapter,
    CodexTurnRequest,
    ProcessResult,
    extract_session_id,
    parse_codex_jsonl,
)
from nlp_stock_prediction.app.settings import AppSettings

pytestmark = pytest.mark.unit


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, command: list[str], *, cwd: Path) -> ProcessResult:
        del cwd
        self.commands.append(command)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("High-level conclusion from Codex.\n", encoding="utf-8")
        return ProcessResult(
            returncode=0,
            stdout=json.dumps({"type": "thread.started", "thread_id": "thread-123"}) + "\n",
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
    assert "Do not provide trading instructions" in runner.commands[0][-1]
    transcript = first.transcript_path.read_text(encoding="utf-8")
    assert "Summarize the selected report" in transcript
    assert "High-level conclusion from Codex." in transcript
