from __future__ import annotations

import os
from datetime import UTC, date, datetime

import pytest

from nlp_stock_prediction.contracts import (
    ExtractionRequest,
    FreshnessStatus,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
)
from nlp_stock_prediction.extraction import (
    STRATEGY_EXTRACTION_SCHEMA_VERSION,
    validate_llm_strategy_payloads,
)


@pytest.mark.live_api
@pytest.mark.llm
def test_live_llm_smoke_scaffold_requires_explicit_opt_in() -> None:
    if os.environ.get("NLP_STOCK_PREDICTION_LIVE_LLM_SMOKE") != "1":
        pytest.skip(
            "Live LLM smoke is scaffolded only; leave NLP_STOCK_PREDICTION_LIVE_LLM_SMOKE "
            "unset until live LLM adapter wiring is enabled."
        )

    pytest.fail(
        "Live LLM adapter wiring is intentionally not enabled for the V1 CLI yet; "
        "this scaffold documents the opt-in marker and fixture validation shape."
    )


@pytest.mark.llm
@pytest.mark.unit
def test_live_llm_smoke_fixture_shape_stays_schema_valid_without_network() -> None:
    now = datetime(2026, 5, 11, 12, 0, tzinfo=UTC)
    evidence = SourceEvidence(
        evidence_id="evidence-live-smoke-shape",
        source_kind=SourceKind.REDDIT_COMMENT,
        ticker="NVDA",
        text="NVDA calls into earnings, but size tiny because IV can crush.",
        provenance=SourceProvenance(
            provider_name="fixture-reddit",
            source_kind=SourceKind.REDDIT_COMMENT,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=now,
            observed_at=now,
            source_url="https://example.invalid/live-smoke",
            permalink="https://reddit.example.invalid/live-smoke",
            raw_identifier="raw-live-smoke-comment",
            raw_snapshot_id="raw-live-smoke-snapshot",
            freshness_status=FreshnessStatus.FRESH,
        ),
    )
    request = ExtractionRequest(
        request_id="live-smoke-shape",
        run_date=date(2026, 5, 11),
        tickers=("NVDA",),
        evidence=(evidence,),
        prompt_version="lane-c-live-smoke-v1",
        schema_version=STRATEGY_EXTRACTION_SCHEMA_VERSION,
    )

    result = validate_llm_strategy_payloads(
        [
            {
                "strategy_id": "strategy-nvda-live-smoke-shape",
                "ticker": "NVDA",
                "label": "NVDA calls into earnings",
                "direction": "bullish",
                "instrument": "call_option",
                "position_type": "long",
                "time_horizon": "weekly",
                "catalyst": "earnings",
                "risk_or_hedge": "Size tiny because IV can crush.",
                "slang_terms": ["calls"],
                "evidence": [
                    {
                        "evidence_id": "evidence-live-smoke-shape",
                        "quote": "NVDA calls into earnings",
                        "relevance": 0.88,
                    }
                ],
                "confidence": 0.7,
                "sarcasm_joke_risk": 0.1,
            }
        ],
        evidence=request.evidence,
        provider_name="fixture-llm",
        occurred_at=now,
    )

    assert result.warnings == ()
    assert result.strategies[0].ticker == "NVDA"
