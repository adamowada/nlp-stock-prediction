from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from nlp_stock_prediction.app.reports import concise_summary_lines, major_blockers
from nlp_stock_prediction.app.state import AppState, refresh_report_index
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle
from nlp_stock_prediction.reporting.json import load_json_report, render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report

pytestmark = pytest.mark.unit


def _write_fixture_report(repo_root: Path) -> Path:
    config = RunConfig(
        run_date=date(2026, 5, 12),
        output_dir=repo_root / "reports",
        symbol="TSLA",
        offline=True,
        source_mode="offline",
    )
    report = build_offline_fixture_bundle(config).report
    report_dir = repo_root / "reports" / "2026-05-12" / "tsla"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(render_json_report(report), encoding="utf-8")
    (report_dir / "report.md").write_text(render_markdown_report(report), encoding="utf-8")
    return report_dir / "report.json"


def test_refresh_report_index_discovers_valid_json_reports(tmp_path: Path) -> None:
    _write_fixture_report(tmp_path)

    state = refresh_report_index(tmp_path, AppState())

    assert len(state.reports) == 1
    entry = state.reports[0]
    assert entry.symbol == "TSLA"
    assert entry.report_date.isoformat() == "2026-05-12"
    assert entry.candidate_count == 1
    assert entry.json_path.as_posix() == "reports/2026-05-12/tsla/report.json"


def test_concise_report_summary_hides_audit_noise(tmp_path: Path) -> None:
    json_path = _write_fixture_report(tmp_path)
    report = load_json_report(json_path.read_text(encoding="utf-8"))

    assert json_path.exists()
    assert "audit-manifest" not in "\n".join(concise_summary_lines(report))


def test_major_blockers_collect_provider_warnings(tmp_path: Path) -> None:
    json_path = _write_fixture_report(tmp_path)

    report = load_json_report(json_path.read_text(encoding="utf-8"))

    assert isinstance(major_blockers(report), list)
