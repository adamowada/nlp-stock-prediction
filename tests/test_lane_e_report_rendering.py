from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import DEFAULT_MARKDOWN_REPORT_OUTLINE, RiskProfile, RunConfig
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report


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
    assert "Educational research only" in rendered
    assert "No automatic trading" in rendered
    assert "Qualified Trading Strategies" in rendered
    assert "candidate-tsla-shares-swing" in rendered
    assert "Audit Artifacts" in rendered

    for ticker in report.ticker_discovery.tickers:
        assert f"### {ticker}" in rendered
        for subsection in DEFAULT_MARKDOWN_REPORT_OUTLINE.required_ticker_subsections:
            assert f"#### {subsection}" in rendered


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
    assert first_section.data_quality["raw_snapshot_ids"] == ["raw-reddit-card-2026-05-11"]
    assert report.provider_health[1].warnings[0].metadata["fallback"] == "offline_fixture"
