from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import RunConfig
from nlp_stock_prediction.contracts.report import AuditManifest
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.reporting.view import ReportView

RUN_DATE = date(2026, 5, 12)


def _report_markdown(tmp_path: Path) -> str:
    config = RunConfig(run_date=RUN_DATE, output_dir=tmp_path, offline=True)
    return render_markdown_report(build_offline_fixture_bundle(config).report)


@pytest.mark.unit
def test_markdown_renders_instrument_identity_and_availability(tmp_path: Path) -> None:
    markdown = _report_markdown(tmp_path)

    assert "#### Identity & Availability" in markdown
    assert "- Instrument ID: `instrument:equity:us:tsla`" in markdown
    assert "- Asset class: stock" in markdown
    assert "- Venue: NASDAQ" in markdown
    assert "- Aliases: `TESLA`, `TSLA.US`" in markdown
    assert "offline-fixture/fixture-symbol: `TSLA`" in markdown
    assert "- Data availability:" in markdown
    assert "offline-fixture research-fixture: available" in markdown
    assert "- Tradability/access evidence:" in markdown
    assert "raw `TSLA:fixture-tradability`" in markdown
    assert "- Related instruments/proxies:" in markdown
    assert "`instrument:etf:us:spy` broad_market_proxy" in markdown
    assert "evidence `fixture-market-spy-001`" in markdown


@pytest.mark.unit
def test_markdown_renders_universe_resolution_states(tmp_path: Path) -> None:
    markdown = _report_markdown(tmp_path)

    assert "## Universe Resolution" in markdown
    assert "`TSLA`: resolved; selected `instrument:equity:us:tsla`" in markdown
    assert "`AI`: ambiguous" in markdown
    assert "`OTC:MISSING`: unsupported" in markdown
    assert "`DELISTED`: unavailable" in markdown
    assert "`instrument:equity:us:ai` AI (stock)" in markdown
    assert "`instrument:crypto:ai-token` AI (crypto)" in markdown
    assert "AI maps to multiple fixture instruments; no default selection made." in markdown


@pytest.mark.unit
def test_markdown_renders_evidence_ledger_provenance(tmp_path: Path) -> None:
    markdown = _report_markdown(tmp_path)

    assert "## Evidence Ledger" in markdown
    assert "`fixture-news-tsla-001` news_article via offline-fixture" in markdown
    assert "URL: https://example.com/fixtures/tsla-delivery-context" in markdown
    assert "Fetched: 2026-05-12T21:00:00+00:00" in markdown
    assert "Observed: 2026-05-12T21:00:00+00:00" in markdown
    assert "Freshness: fresh" in markdown
    assert "Instrument IDs: `instrument:equity:us:tsla`" in markdown
    assert "Tickers: `TSLA`" in markdown
    assert "Raw identifier: `fixture-news-tsla-001`" in markdown
    assert "Raw snapshot/artifact IDs: `raw-fixture-news-tsla-001`" in markdown


@pytest.mark.unit
def test_report_view_prepares_render_lookup_maps(tmp_path: Path) -> None:
    config = RunConfig(run_date=RUN_DATE, output_dir=tmp_path, offline=True)
    report = build_offline_fixture_bundle(config).report

    view = ReportView.from_report(report)
    section = report.instrument_sections[0]

    assert view.instrument_for_section(section) == report.instruments[0]
    assert (
        tuple(candidate.candidate_id for candidate in view.candidates_for_section(section))
        == section.prediction_candidate_ids
    )
    assert view.provider_warnings == tuple(
        warning for health in report.provider_health for warning in health.warnings
    )
    assert isinstance(report.audit_manifest, AuditManifest)
    assert view.audit_artifacts == report.audit_manifest.artifacts


@pytest.mark.unit
def test_candidate_rendering_preserves_context_without_advice_labels(tmp_path: Path) -> None:
    markdown = _report_markdown(tmp_path)

    assert "Evidence for: `fixture-news-tsla-001`" in markdown
    assert "Evidence against: `fixture-market-spy-001`" in markdown
    assert "Dissenting evidence: limits: Broad-index regime evidence" in markdown
    assert "Assumptions: Offline fixtures are a deterministic contract exercise." in markdown
    assert "Uncertainties: Synthetic fixture evidence cannot substitute" in markdown
    assert "Uncertainty drivers: `uncertainty-tsla-fixture-freshness` high" in markdown
    assert "What would change: `change-tsla-live-provider-refresh` provider_refresh" in markdown
    assert "## Prior-Outcome Review" in markdown
    assert "`prior-outcome-tsla-unavailable` not_available" in markdown
    assert "## Material Claim Traceability" in markdown
    assert "`claim-tsla-headline-sensitivity` analysis" in markdown
    assert "## Report Source References" in markdown
    assert "`source-ref-tsla-news` source_evidence" in markdown
    assert "Recommendation:" not in markdown
    assert "Trade instruction:" not in markdown


@pytest.mark.unit
def test_json_renderer_preserves_phase3_report_fields(tmp_path: Path) -> None:
    config = RunConfig(run_date=RUN_DATE, output_dir=tmp_path, offline=True)
    report = build_offline_fixture_bundle(config).report

    payload = json.loads(render_json_report(report))

    assert payload["instruments"][0]["asset_class"] == "stock"
    assert payload["instruments"][0]["venue"] == "NASDAQ"
    assert payload["instruments"][0]["provider_ids"][0]["identifier"] == "TSLA"
    assert payload["instruments"][0]["related_instruments"][0]["evidence_ids"] == [
        "fixture-market-spy-001"
    ]
    assert payload["instrument_resolutions"][1]["status"] == "ambiguous"
    assert payload["evidence_sources"][0]["provenance"]["raw_snapshot_id"] == (
        "raw-fixture-news-tsla-001"
    )
    assert payload["prediction_candidates"][0]["change_triggers"][0]["trigger_id"] == (
        "change-tsla-live-provider-refresh"
    )
    assert payload["prior_outcome_reviews"][0]["status"] == "not_available"
    assert payload["material_claim_traces"][0]["source_reference_ids"] == ["source-ref-tsla-news"]
