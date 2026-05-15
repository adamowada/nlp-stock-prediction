from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from nlp_stock_prediction.cli import main
from nlp_stock_prediction.contracts import RunConfig
from nlp_stock_prediction.orchestration.report_bundle import ReportBundle
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle
from nlp_stock_prediction.terminal_ui import render_research_complete

pytestmark = pytest.mark.unit


def _console(buffer: StringIO) -> Console:
    return Console(file=buffer, force_terminal=False, width=120, color_system=None)


def test_research_terminal_ui_renders_report_dashboard(tmp_path: Path) -> None:
    config = RunConfig(
        run_date=date(2026, 5, 12),
        output_dir=tmp_path,
        symbol="TSLA",
        offline=True,
        source_mode="offline",
    )
    fixture_bundle = build_offline_fixture_bundle(config)
    bundle = ReportBundle(
        report_dir=tmp_path / "2026-05-12" / "tsla",
        markdown_path=tmp_path / "2026-05-12" / "tsla" / "report.md",
        json_path=tmp_path / "2026-05-12" / "tsla" / "report.json",
        audit_dir=tmp_path / "2026-05-12" / "tsla" / "audit",
        audit_manifest_path=tmp_path / "2026-05-12" / "tsla" / "audit" / "audit-manifest.json",
        report=fixture_bundle.report,
        tool_records=(),
    )
    buffer = StringIO()

    render_research_complete(bundle, console=_console(buffer))

    output = buffer.getvalue()
    assert "Prediction Research Terminal" in output
    assert "research-2026-05-12" in output
    assert "Evidence-backed research, not trading instructions." in output
    assert "report.md" in output
    assert "offline-fixture" in output
    assert "prediction-tsla-volatility-context" in output


def test_tui_command_runs_research_with_rich_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "reports"

    exit_code = main(
        [
            "tui",
            "--date",
            "2026-05-12",
            "--output",
            str(output_dir),
            "--offline",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Prediction Research Terminal" in captured.out
    assert "report.md" in captured.out
    assert captured.err == ""
    assert (output_dir / "2026-05-12" / "tsla" / "report.md").exists()
    assert (output_dir / "2026-05-12" / "tsla" / "report.json").exists()
