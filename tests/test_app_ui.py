from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from nlp_stock_prediction.app.ui import TerminalApp, _codex_activity_line, _CodexActivity
from nlp_stock_prediction.cli import build_parser
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.orchestration.report_bundle import ReportBundle
from nlp_stock_prediction.pipeline import generate_daily_report
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle

pytestmark = pytest.mark.unit


def _input(answers: Iterable[str]) -> tuple[StringIO, Callable[[str], str]]:
    iterator = iter(answers)
    buffer = StringIO()

    def ask(prompt: str) -> str:
        buffer.write(prompt)
        return next(iterator)

    return buffer, ask


def _fixture_bundle(tmp_path: Path, config: RunConfig) -> ReportBundle:
    fixture = build_offline_fixture_bundle(config)
    report_dir = tmp_path / "reports" / config.run_date.isoformat() / config.symbol.lower()
    return ReportBundle(
        report_dir=report_dir,
        markdown_path=report_dir / "report.md",
        json_path=report_dir / "report.json",
        audit_dir=report_dir / "audit",
        audit_manifest_path=report_dir / "audit" / "audit-manifest.json",
        report=fixture.report,
        tool_records=(),
        database_path=tmp_path / "data" / "prediction-research.sqlite3",
    )


def test_cli_parser_exposes_app_command() -> None:
    args = build_parser().parse_args(["app"])

    assert args.command == "app"


def test_terminal_app_runs_research_with_today_live_defaults(tmp_path: Path) -> None:
    seen_configs: list[RunConfig] = []

    def fake_generator(config: RunConfig) -> ReportBundle:
        seen_configs.append(config)
        return _fixture_bundle(tmp_path, config)

    _prompts, ask = _input(["1", "1", "MSFT", "6"])
    console_output = StringIO()
    app = TerminalApp(
        repo_root=tmp_path,
        console=Console(file=console_output, force_terminal=False, color_system=None),
        input_func=ask,
        report_generator=fake_generator,
    )

    assert app.run() == 0
    assert seen_configs
    assert seen_configs[0].run_date == date.today()
    assert seen_configs[0].source_mode == "live"
    assert seen_configs[0].live_providers is True
    assert seen_configs[0].offline is False
    assert seen_configs[0].symbol == "MSFT"
    assert (tmp_path / "data" / "app-state.json").exists()
    assert "Research Complete" in console_output.getvalue()


def test_research_menu_back_does_not_run_live_defaults(tmp_path: Path) -> None:
    seen_configs: list[RunConfig] = []

    def fake_generator(config: RunConfig) -> ReportBundle:
        seen_configs.append(config)
        return _fixture_bundle(tmp_path, config)

    _prompts, ask = _input(["1", "back", "6"])
    app = TerminalApp(
        repo_root=tmp_path,
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
        input_func=ask,
        report_generator=fake_generator,
    )

    assert app.run() == 0
    assert seen_configs == []


def test_research_menu_rejects_invalid_mode_before_running(tmp_path: Path) -> None:
    seen_configs: list[RunConfig] = []

    def fake_generator(config: RunConfig) -> ReportBundle:
        seen_configs.append(config)
        return _fixture_bundle(tmp_path, config)

    _prompts, ask = _input(["1", "2", "TSLA", "2026-05-12", "offlne", "reports", "cache", "6"])
    output = StringIO()
    app = TerminalApp(
        repo_root=tmp_path,
        console=Console(file=output, force_terminal=False, color_system=None),
        input_func=ask,
        report_generator=fake_generator,
    )

    assert app.run() == 0
    assert seen_configs == []
    assert "Mode must be 'live' or 'offline'." in output.getvalue()


def test_terminal_app_clears_screen_between_menu_commands(tmp_path: Path) -> None:
    _prompts, ask = _input(["1", "invalid", "6"])
    output = StringIO()
    app = TerminalApp(
        repo_root=tmp_path,
        console=Console(file=output, force_terminal=True, color_system=None),
        input_func=ask,
    )

    assert app.run() == 0
    rendered = output.getvalue()
    clear_sequence = "\x1b[2J\x1b[H"
    assert rendered.startswith(clear_sequence)
    assert rendered.count(clear_sequence) >= 3
    assert rendered.index("Research Defaults") > rendered.index(clear_sequence)
    assert "Choose a listed option." in rendered


def test_codex_activity_renders_thinking_status_without_private_content() -> None:
    activity = _CodexActivity()
    activity.record_event(
        {
            "type": "item.completed",
            "item": {
                "type": "function_call",
                "name": "shell_command",
                "arguments": "private chain-of-thought should not render",
            },
        }
    )
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)

    console.print(activity.render())

    rendered = output.getvalue()
    assert "Thinking" in rendered
    assert "Finished tool: shell_command." in rendered
    assert "private chain-of-thought" not in rendered


def test_codex_activity_summarizes_reasoning_events() -> None:
    assert (
        _codex_activity_line(
            {
                "type": "item.completed",
                "item": {
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "Checking report evidence and provider health.",
                        }
                    ],
                },
            }
        )
        == "Reasoning: Checking report evidence and provider health."
    )


def test_codex_activity_names_mcp_tool_and_safe_arguments() -> None:
    assert _codex_activity_line(
        {
            "type": "item.started",
            "item": {
                "type": "mcp_tool_call",
                "server": "nlp-stock-prediction",
                "tool": "inspect_research_run",
                "arguments": (
                    '{"run_id":"phase4-2026-05-14-nflx","include_artifacts":true,'
                    '"api_key":"secret"}'
                ),
                "status": "in_progress",
            },
        }
    ) == (
        "Started MCP tool: inspect_research_run(run_id=phase4-2026-05-14-nflx, "
        "include_artifacts=true, api_key=<redacted>)."
    )


def test_codex_activity_summarizes_shell_command_completion() -> None:
    assert (
        _codex_activity_line(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "Get-Location",
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        )
        == "Finished shell command (exit 0): Get-Location."
    )


def test_codex_activity_suppresses_unhelpful_item_lifecycle_events() -> None:
    assert _codex_activity_line({"type": "item.started", "item": {"type": "unknown"}}) is None
    assert _codex_activity_line({"type": "item.completed", "item": {"type": "unknown"}}) is None


@pytest.mark.integration
def test_terminal_app_can_run_offline_research_flow(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    cache_root = tmp_path / "cache"
    _prompts, ask = _input(
        [
            "1",
            "2",
            "TSLA",
            "2026-05-12",
            "offline",
            str(report_root),
            str(cache_root),
            "6",
        ]
    )

    app = TerminalApp(
        repo_root=tmp_path,
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
        input_func=ask,
        report_generator=generate_daily_report,
    )

    assert app.run() == 0
    assert (report_root / "2026-05-12" / "tsla" / "report.md").exists()
    assert (report_root / "2026-05-12" / "tsla" / "report.json").exists()
