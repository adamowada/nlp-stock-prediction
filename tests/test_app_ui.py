from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from nlp_stock_prediction.app.ui import TerminalApp
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

    _prompts, ask = _input(["1", "1", "6"])
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
    assert (tmp_path / "data" / "app-state.json").exists()
    assert "Research Complete" in console_output.getvalue()


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
