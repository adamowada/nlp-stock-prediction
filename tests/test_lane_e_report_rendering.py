from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.contracts import (
    DEFAULT_MARKDOWN_REPORT_OUTLINE,
    AuditManifest,
    JsonObject,
    RiskProfile,
    RunConfig,
)
from nlp_stock_prediction.reporting.fixtures import (
    build_offline_fixture_bundle,
    build_offline_fixture_report,
)
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report


def _ticker_markdown_block(rendered: str, tickers: tuple[str, ...], ticker: str) -> str:
    index = tickers.index(ticker)
    start = rendered.index(f"### {ticker}")
    if index + 1 < len(tickers):
        end = rendered.index(f"### {tickers[index + 1]}", start + 1)
    else:
        end = rendered.index("## Qualified Trading Strategies Or No-Trade Summary", start + 1)
    return rendered[start:end]


@pytest.mark.unit
def test_markdown_report_includes_required_lane_e_sections() -> None:
    report = build_offline_fixture_report(
        RunConfig(
            run_date=date(2026, 5, 11),
            output_dir=Path("reports"),
            capital=Decimal("1000"),
            risk_profile=RiskProfile.EXPLORATORY,
            offline=True,
        )
    )

    rendered = render_markdown_report(report)

    assert "# Daily Stock Opportunity Report" in rendered
    assert "Report date: 2026-05-11" in rendered
    assert "Data freshness:" in rendered
    assert "Provider Warnings" in rendered
    assert "Qualified Trading Strategies" in rendered
    assert "candidate-tsla-shares-swing" in rendered
    assert "#### App Analysis" in rendered
    assert "separate from the observed discussion above" in rendered
    assert "Confidence inputs:" in rendered
    risk_controls_line = (
        "Risk controls: profile exploratory; defined risk yes; margin required no; passed yes"
    )
    assert risk_controls_line in rendered
    assert "Score components:" in rendered
    assert "Penalties:" in rendered
    assert "Warning refs: fixture-market-data:stale_data" in rendered
    assert "Audit Artifacts" in rendered

    expected_ticker_subsections = (
        "Observed Discussion",
        "Social And News",
        "Strategy Clusters",
        "App Analysis",
        "Technical Analysis",
        "Fundamental Analysis",
        "Sector Context",
        "Macro Context",
        "Opportunity Notes",
        "Evidence References",
    )
    for ticker in report.ticker_discovery.tickers:
        assert f"### {ticker}" in rendered
        for subsection in DEFAULT_MARKDOWN_REPORT_OUTLINE.required_ticker_subsections:
            assert f"#### {subsection}" in rendered
        block = _ticker_markdown_block(rendered, report.ticker_discovery.tickers, ticker)
        positions = [
            block.index(f"#### {subsection}") for subsection in expected_ticker_subsections
        ]
        assert positions == sorted(positions)
        assert "Action: qualified" not in block
        assert "candidate-tsla-shares-swing" not in block


@pytest.mark.unit
def test_offline_fixture_report_preserves_recommendation_traceability() -> None:
    report = build_offline_fixture_report(
        RunConfig(
            run_date=date(2026, 5, 11),
            output_dir=Path("reports"),
            capital=Decimal("1000"),
            risk_profile=RiskProfile.EXPLORATORY,
            offline=True,
        )
    )

    assert len(report.ticker_sections) == 6
    assert report.trade_candidates[0].candidate_id == "candidate-tsla-shares-swing"
    assert report.trade_candidates[0].evidence[0].evidence_id == "evidence-tsla-reddit-1"
    assert report.trade_candidates[0].score_input_ids == ("scoring-input-tsla",)
    assert report.trade_candidates[0].metadata["confidence_inputs"] == {
        "reddit_strategy_confidence": 0.78,
        "technical_alignment": 0.72,
        "freshness_penalty": 0.05,
    }

    first_section = report.ticker_sections[0]
    assert first_section.ticker == "TSLA"
    assert first_section.strategy_clusters[0].evidence[0].evidence_id == "evidence-tsla-reddit-1"
    assert first_section.data_quality["raw_snapshot_ids"] == [
        "raw-reddit-card-2026-05-11",
        "raw-reddit-discussion-2026-05-11",
        "raw-market-data-2026-05-11",
    ]
    assert first_section.data_quality["evidence_ids"] == ["evidence-tsla-reddit-1"]
    assert first_section.data_quality["evidence_freshness"] == "fresh"
    assert first_section.data_quality["provider_metadata"] == {"fixture": True, "offline": True}
    assert report.provider_health[1].warnings[0].metadata["fallback"] == "offline_fixture"
    assert report.data_freshness.missing_provider_names == ("fixture-sec-edgar",)
    assert report.provider_health[2].status.value == "unconfigured"
    assert report.provider_health[2].warnings[0].code.value == "missing_credentials"
    assert report.evidence_sources[0].provenance.provider_metadata == {
        "fixture": True,
        "offline": True,
    }


@pytest.mark.unit
def test_json_report_preserves_stage_4_recommendation_and_quality_inputs() -> None:
    report = build_offline_fixture_report(
        RunConfig(
            run_date=date(2026, 5, 11),
            output_dir=Path("reports"),
            capital=Decimal("1000"),
            risk_profile=RiskProfile.EXPLORATORY,
            offline=True,
        )
    )

    payload = cast(JsonObject, json.loads(render_json_report(report)))
    evidence_sources = cast(list[JsonObject], payload["evidence_sources"])
    ticker_sections = cast(list[JsonObject], payload["ticker_sections"])
    trade_candidates = cast(list[JsonObject], payload["trade_candidates"])
    provider_health = cast(list[JsonObject], payload["provider_health"])

    assert len(ticker_sections) == 6
    assert [section["ticker"] for section in ticker_sections] == [
        "TSLA",
        "NVDA",
        "AMD",
        "AAPL",
        "MU",
        "SPY",
    ]
    first_evidence = evidence_sources[0]
    first_provenance = cast(JsonObject, first_evidence["provenance"])
    assert first_evidence["evidence_id"] == "evidence-tsla-reddit-1"
    assert first_provenance["provider_name"] == "fixture-reddit"
    assert first_provenance["freshness_status"] == "fresh"
    assert first_provenance["provider_metadata"] == {"fixture": True, "offline": True}
    assert first_provenance["raw_snapshot_id"] == "raw-reddit-discussion-2026-05-11"
    assert first_provenance["permalink"] == (
        "https://reddit.example/r/wallstreetbets/comments/2026-05-11/tsla"
    )
    for section in ticker_sections:
        data_quality = cast(JsonObject, section["data_quality"])
        expected_evidence_ids = [f"evidence-{str(section['ticker']).lower()}-reddit-1"]
        assert data_quality["evidence_ids"] == expected_evidence_ids
        assert data_quality["evidence_freshness"] == "fresh"
        assert data_quality["provider_metadata"] == {"fixture": True, "offline": True}
        assert "fixture-market-data:stale_data" in cast(list[str], data_quality["warning_ids"])

    candidate = trade_candidates[0]
    score = cast(JsonObject, candidate["score"])
    risk_plan = cast(JsonObject, candidate["risk_plan"])
    metadata = cast(JsonObject, candidate["metadata"])
    score_components = cast(list[JsonObject], score["components"])
    penalties = cast(list[JsonObject], score["penalties"])
    evidence = cast(list[JsonObject], candidate["evidence"])
    overall_score = cast(float, score["overall_score"])
    threshold = cast(float, score["threshold"])
    market_health = provider_health[1]
    market_warnings = cast(list[JsonObject], market_health["warnings"])
    missing_health = provider_health[2]
    missing_warnings = cast(list[JsonObject], missing_health["warnings"])

    assert candidate["action"] == "qualified"
    assert risk_plan["passed"] is True
    assert overall_score >= threshold
    assert candidate["risks"] == ["Retail discussion can be crowded, stale, sarcastic, or wrong."]
    assert evidence[0]["evidence_id"] == "evidence-tsla-reddit-1"
    assert candidate["score_input_ids"] == ["scoring-input-tsla"]
    assert metadata["confidence_inputs"] == {
        "reddit_strategy_confidence": 0.78,
        "technical_alignment": 0.72,
        "freshness_penalty": 0.05,
    }
    assert score_components[0]["data_reference_ids"] == [
        "normalized-evidence",
        "analysis-tsla-offline",
    ]
    assert penalties[0]["warning_ids"] == ["fixture-market-data:stale_data"]
    assert market_warnings[0]["code"] == "stale_data"
    assert missing_health["provider_name"] == "fixture-sec-edgar"
    assert missing_health["status"] == "unconfigured"
    assert missing_warnings[0]["code"] == "missing_credentials"


@pytest.mark.unit
def test_offline_fixture_report_with_zero_capital_has_no_qualified_candidate() -> None:
    bundle = build_offline_fixture_bundle(
        RunConfig(
            run_date=date(2026, 5, 11),
            output_dir=Path("reports"),
            capital=Decimal("0"),
            risk_profile=RiskProfile.EXPLORATORY,
            offline=True,
        )
    )
    report = bundle.report
    records = cast(list[dict[str, Any]], bundle.audit_payloads["scoring-inputs.json"]["records"])
    scoring_input = records[0]

    assert report.trade_candidates == ()
    assert report.no_trade_summary == "No qualified trades passed the offline fixture risk gates."
    assert report.ticker_sections[0].recommendation_ids == ()
    assert isinstance(report.audit_manifest, AuditManifest)
    assert report.audit_manifest.recommendation_trace_ids == ()
    assert scoring_input["risk_plan"]["passed"] is False
    assert scoring_input["risk_plan"]["failed_gates"] == ["account-capital-must-be-positive"]

    rendered = render_markdown_report(report)
    assert "### No-Trade Summary" in rendered
    assert "### Qualified Trading Strategies" not in rendered
    assert "No qualified trades passed the offline fixture risk gates." in rendered
