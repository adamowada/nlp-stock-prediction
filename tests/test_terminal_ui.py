from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from nlp_stock_prediction.cli import CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE, main
from nlp_stock_prediction.contracts import PredictionStatus, RunConfig
from nlp_stock_prediction.orchestration.report_bundle import ReportBundle
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle
from nlp_stock_prediction.terminal_ui import (
    _candidate_trace_reference_count,
    render_research_complete,
)

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


def test_research_terminal_ui_prints_literal_one_line_paths(tmp_path: Path) -> None:
    config = RunConfig(
        run_date=date(2026, 5, 12),
        output_dir=tmp_path,
        symbol="TSLA",
        offline=True,
        source_mode="offline",
    )
    fixture_bundle = build_offline_fixture_bundle(config)
    report_dir = tmp_path / "[draft]" / "2026-05-12" / "tsla"
    bundle = ReportBundle(
        report_dir=report_dir,
        markdown_path=report_dir / "report.md",
        json_path=report_dir / "report.json",
        audit_dir=report_dir / "audit",
        audit_manifest_path=report_dir / "audit" / "audit-manifest.json",
        report=fixture_bundle.report,
        tool_records=(),
    )
    buffer = StringIO()

    render_research_complete(bundle, console=Console(file=buffer, width=80, color_system=None))

    output = buffer.getvalue()
    assert "[draft]" in output
    assert "..." not in output
    assert all("…" not in line for line in output.splitlines() if "[draft]" in line)
    assert f"Wrote Markdown report: {bundle.markdown_path}" in output.splitlines()
    assert f"Wrote JSON report: {bundle.json_path}" in output.splitlines()
    assert f"Wrote audit artifacts: {bundle.audit_dir}" in output.splitlines()


def test_candidate_trace_reference_count_includes_dissent_and_uncertainty(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        run_date=date(2026, 5, 12),
        output_dir=tmp_path,
        symbol="TSLA",
        offline=True,
        source_mode="offline",
    )
    report = build_offline_fixture_bundle(config).report
    dissent = (
        report.prediction_candidates[0]
        .dissenting_evidence[0]
        .model_copy(update={"impact": "contradicts"})
    )
    candidate = report.prediction_candidates[0].model_copy(
        update={
            "status": PredictionStatus.CONTRADICTED,
            "evidence_for": (),
            "evidence_against": (),
            "dissenting_evidence": (dissent,),
        }
    )

    assert _candidate_trace_reference_count(candidate) == 2


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
    assert "\x1b[" not in captured.out
    assert captured.err == ""
    assert (output_dir / "2026-05-12" / "tsla" / "report.md").exists()
    assert (output_dir / "2026-05-12" / "tsla" / "report.json").exists()


def test_tui_command_rejects_missing_inputs_without_prompting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["tui"])
    captured = capsys.readouterr()

    assert exit_code == CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
    assert captured.out == ""
    assert "tui requires --date, --output, --offline or --live" in captured.err
    assert "Report date" not in captured.err
    assert "\x1b[" not in captured.err
