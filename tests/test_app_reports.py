from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from nlp_stock_prediction.app.reports import concise_summary_lines, major_blockers
from nlp_stock_prediction.app.settings import AppSettings
from nlp_stock_prediction.app.state import AppState, refresh_report_index
from nlp_stock_prediction.contracts.enums import ProviderStatus, WarningCode, WarningSeverity
from nlp_stock_prediction.contracts.provenance import ProviderHealth, ProviderWarning
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle
from nlp_stock_prediction.reporting.json import load_json_report, render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report

pytestmark = pytest.mark.unit


def _write_fixture_report(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    database_path: Path | None = None,
    with_provider_warning: bool = False,
) -> Path:
    report_root = output_dir or repo_root / "reports"
    config = RunConfig(
        run_date=date(2026, 5, 12),
        output_dir=report_root,
        symbol="TSLA",
        offline=True,
        source_mode="offline",
    )
    report = build_offline_fixture_bundle(config).report
    command_args = dict(report.command_args)
    if database_path is not None:
        command_args["database_path"] = database_path.as_posix()
    provider_health = report.provider_health
    if with_provider_warning:
        now = datetime(2026, 5, 12, tzinfo=UTC)
        provider_health = (
            *provider_health,
            ProviderHealth(
                provider_name="fixture-provider",
                status=ProviderStatus.STALE,
                checked_at=now,
                warnings=(
                    ProviderWarning(
                        code=WarningCode.STALE_DATA,
                        severity=WarningSeverity.WARNING,
                        message="fixture data is stale",
                        occurred_at=now,
                    ),
                ),
            ),
        )
    report = report.model_copy(
        update={"command_args": command_args, "provider_health": provider_health}
    )
    report_dir = report_root / "2026-05-12" / "tsla"
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


def test_refresh_report_index_scans_configured_output_dir(tmp_path: Path) -> None:
    _write_fixture_report(tmp_path, output_dir=tmp_path / "custom-reports")

    state = refresh_report_index(
        tmp_path,
        AppState(settings=AppSettings(output_dir=Path("custom-reports"))),
    )

    assert len(state.reports) == 1
    assert state.reports[0].json_path.as_posix() == "custom-reports/2026-05-12/tsla/report.json"


def test_refresh_report_index_recovers_database_path_from_report(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "research-runtime.sqlite3"
    _write_fixture_report(tmp_path, database_path=database_path)

    state = refresh_report_index(tmp_path, AppState())

    assert state.reports[0].resolve_database_path(tmp_path) == database_path


def test_concise_report_summary_hides_audit_noise(tmp_path: Path) -> None:
    json_path = _write_fixture_report(tmp_path)
    report = load_json_report(json_path.read_text(encoding="utf-8"))
    summary = "\n".join(concise_summary_lines(report))

    assert json_path.exists()
    assert "TSLA | 2026-05-12" in summary
    assert "evidence_supported" in summary
    assert "confidence" in summary
    assert "audit-manifest" not in summary


def test_major_blockers_collect_provider_warnings(tmp_path: Path) -> None:
    json_path = _write_fixture_report(tmp_path, with_provider_warning=True)

    report = load_json_report(json_path.read_text(encoding="utf-8"))

    assert "fixture-provider: fixture data is stale" in major_blockers(report)
