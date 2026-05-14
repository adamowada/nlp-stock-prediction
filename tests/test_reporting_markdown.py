from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import (
    DEFAULT_JSON_REPORT_CONTRACT,
    DEFAULT_MARKDOWN_REPORT_OUTLINE,
    Direction,
    InsufficientEvidenceReport,
    PredictionStatus,
    PredictionType,
    ReportSourceReference,
    RunConfig,
)
from nlp_stock_prediction.contracts.report import AuditManifest, DailyReport
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle
from nlp_stock_prediction.reporting.json import load_json_report, render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.reporting.view import ReportView

RUN_DATE = date(2026, 5, 12)


def _report_markdown(tmp_path: Path) -> str:
    config = RunConfig(run_date=RUN_DATE, output_dir=tmp_path, offline=True)
    return render_markdown_report(build_offline_fixture_bundle(config).report)


def _fixture_report(tmp_path: Path) -> DailyReport:
    config = RunConfig(run_date=RUN_DATE, output_dir=tmp_path, offline=True)
    return build_offline_fixture_bundle(config).report


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
def test_markdown_surfaces_json_report_metadata_provider_health_and_audit(
    tmp_path: Path,
) -> None:
    report = _fixture_report(tmp_path)
    markdown = render_markdown_report(report)
    payload = json.loads(render_json_report(report))
    provider_health = payload["provider_health"][0]
    audit_manifest = payload["audit_manifest"]

    assert "## Report Metadata" in markdown
    assert f"- Report schema: `{payload['schema_version']}`" in markdown
    assert f"- Timezone: `{payload['timezone']}`" in markdown
    assert "Command args:" in markdown
    assert "offline=True" in markdown
    assert "## Provider Health" in markdown
    assert f"`{provider_health['provider_name']}`: status {provider_health['status']}" in markdown
    assert f"- Manifest schema: `{audit_manifest['schema_version']}`" in markdown
    assert "Prediction trace IDs: `prediction-tsla-volatility-context`" in markdown
    assert "- No audit artifact files listed in manifest." in markdown
    assert "- Audit manifest unavailable." not in markdown


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "direction", "prediction_type"),
    (
        (
            PredictionStatus.EVIDENCE_SUPPORTED,
            Direction.MIXED,
            PredictionType.VOLATILITY,
        ),
        (PredictionStatus.CONTRADICTED, Direction.BEARISH, PredictionType.DIRECTIONAL),
        (PredictionStatus.EVIDENCE_SUPPORTED, Direction.NEUTRAL, PredictionType.NEUTRAL),
    ),
)
def test_markdown_covers_supported_contradicted_and_neutral_candidate_states(
    tmp_path: Path,
    status: PredictionStatus,
    direction: Direction,
    prediction_type: PredictionType,
) -> None:
    report = _fixture_report(tmp_path)
    candidate = report.prediction_candidates[0].model_copy(
        update={
            "status": status,
            "direction": direction,
            "prediction_type": prediction_type,
        }
    )

    rendered_report = DailyReport.model_validate(
        report.model_copy(update={"prediction_candidates": (candidate,)}).model_dump(mode="python")
    )
    markdown = render_markdown_report(rendered_report)

    assert f"Status: {status.value}; direction: {direction.value};" in markdown
    assert f"type: {prediction_type.value};" in markdown
    assert "Report-authored scenario:" in markdown
    assert "Evidence for: `fixture-news-tsla-001` (observed source claims)" in markdown
    assert "Evidence against: `fixture-market-spy-001` (observed source claims)" in markdown
    assert "Baseline:" in markdown
    assert "Dissenting evidence:" in markdown
    assert "Uncertainties:" in markdown
    assert "Prior outcome reviews:" in markdown
    assert "Recommendation:" not in markdown
    assert "Trade instruction:" not in markdown
    assert re.search(r"\b(buy|sell)\s+TSLA\b", markdown, flags=re.IGNORECASE) is None


@pytest.mark.unit
def test_markdown_covers_structured_insufficient_evidence_report(tmp_path: Path) -> None:
    report = _fixture_report(tmp_path)
    insufficient = InsufficientEvidenceReport(
        summary="No supported prediction scenario remains after artifact validation.",
        blocking_reasons=("Required technical package artifact was malformed.",),
        provider_names=("report-assembly",),
        evidence=report.instrument_sections[0].evidence,
        metadata={"excluded_candidate_ids": ["prediction-tsla-volatility-context"]},
    )
    sections = tuple(
        section.model_copy(update={"prediction_candidate_ids": ()})
        for section in report.instrument_sections
    )
    source_references = tuple(
        reference.model_copy(update={"candidate_ids": ()})
        if isinstance(reference, ReportSourceReference)
        else reference
        for reference in report.source_references
    )
    candidate_free_report = report.model_copy(
        update={
            "instrument_sections": sections,
            "prediction_candidates": (),
            "material_claim_traces": (),
            "prior_outcome_reviews": (),
            "source_references": source_references,
            "insufficient_evidence": insufficient,
            "insufficient_evidence_summary": insufficient.summary,
        }
    )
    candidate_free_report = DailyReport.model_validate(
        candidate_free_report.model_dump(mode="python")
    )

    markdown = render_markdown_report(candidate_free_report)

    assert "## Prediction Scenarios Or Insufficient-Evidence Summary" in markdown
    assert "No supported prediction scenario remains after artifact validation." in markdown
    assert "Required technical package artifact was malformed." in markdown
    assert "Providers: `report-assembly`" in markdown
    assert "Metadata: excluded_candidate_ids=['prediction-tsla-volatility-context']" in markdown
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


@pytest.mark.unit
def test_json_report_contract_round_trips_and_covers_material_markdown_sections(
    tmp_path: Path,
) -> None:
    report = _fixture_report(tmp_path)
    json_text = render_json_report(report)

    round_tripped = load_json_report(json_text)
    json_headings = tuple(
        section.heading for section in DEFAULT_JSON_REPORT_CONTRACT.material_sections
    )

    assert round_tripped == report
    assert DEFAULT_JSON_REPORT_CONTRACT.report_schema_version == report.schema_version
    assert DEFAULT_JSON_REPORT_CONTRACT.markdown_outline_schema_version == (
        DEFAULT_MARKDOWN_REPORT_OUTLINE.schema_version
    )
    assert set(DEFAULT_MARKDOWN_REPORT_OUTLINE.heading_order[1:]).issubset(json_headings)
