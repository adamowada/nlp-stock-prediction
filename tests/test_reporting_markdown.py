from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import RunConfig
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report

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
def test_candidate_rendering_preserves_context_without_advice_labels(tmp_path: Path) -> None:
    markdown = _report_markdown(tmp_path)

    assert "Evidence for: `fixture-news-tsla-001`" in markdown
    assert "Evidence against: `fixture-market-spy-001`" in markdown
    assert "Assumptions: Offline fixtures are a deterministic contract exercise." in markdown
    assert "Uncertainties: Synthetic fixture evidence cannot substitute" in markdown
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
