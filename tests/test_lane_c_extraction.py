from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from nlp_stock_prediction.contracts import (
    Direction,
    EvidenceReference,
    ExtractionRequest,
    FreshnessStatus,
    InstrumentType,
    PositionType,
    ProviderStatus,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    StrategyExtraction,
    TimeHorizon,
    WarningCode,
)
from nlp_stock_prediction.extraction import (
    STRATEGY_EXTRACTION_REQUIRED_FIELDS,
    STRATEGY_EXTRACTION_SCHEMA_VERSION,
    FixtureLLMExtractor,
    cluster_strategies,
    parse_llm_json_response,
    validate_llm_strategy_payloads,
)

RUN_DATE = date(2026, 5, 11)
NOW = datetime(2026, 5, 11, 12, 0, tzinfo=UTC)


def _provenance(provider_name: str = "fixture-reddit") -> SourceProvenance:
    return SourceProvenance(
        provider_name=provider_name,
        source_kind=SourceKind.REDDIT_COMMENT,
        retrieval_method=RetrievalMethod.FIXTURE,
        fetched_at=NOW,
        observed_at=NOW,
        source_url=f"https://example.invalid/{provider_name}/thread",
        permalink=f"https://reddit.example.invalid/{provider_name}/comments/abc",
        raw_identifier=f"raw-{provider_name}-comment-1",
        raw_snapshot_id=f"raw-{provider_name}-snapshot",
        query="TSLA daily thread",
        cache_key=f"{provider_name}:TSLA:2026-05-11",
        freshness_status=FreshnessStatus.FRESH,
        freshness_seconds=120,
        provider_metadata={"fixture": True},
    )


def _evidence(
    evidence_id: str = "evidence-tsla-earnings-calls",
    *,
    text: str = "TSLA weekly calls into earnings; premium can go to zero if IV crush hits.",
    ticker: str = "TSLA",
) -> SourceEvidence:
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=SourceKind.REDDIT_COMMENT,
        ticker=ticker,
        title="Daily discussion",
        text=text,
        author_hash="author-hash",
        created_at=NOW,
        score=77,
        permalink="https://reddit.example.invalid/r/wallstreetbets/comments/abc",
        matched_tickers=(ticker,),
        provenance=_provenance(),
        metadata={"rank": 1},
    )


def _valid_payload(
    *,
    evidence_id: str = "evidence-tsla-earnings-calls",
    quote: str = "weekly calls into earnings",
    strategy_id: str = "strategy-tsla-weekly-calls-earnings",
    label: str = "TSLA weekly calls into earnings",
    catalyst: str | None = "earnings",
    sarcasm_joke_risk: float = 0.12,
) -> dict[str, object]:
    return {
        "strategy_id": strategy_id,
        "ticker": "TSLA",
        "label": label,
        "direction": "bullish",
        "instrument": "call_option",
        "position_type": "long",
        "time_horizon": "weekly",
        "catalyst": catalyst,
        "risk_or_hedge": "Premium can go to zero if IV crush hits.",
        "slang_terms": ["calls", "weeklies"],
        "evidence": [
            {
                "evidence_id": evidence_id,
                "quote": quote,
                "relevance": 0.94,
            }
        ],
        "confidence": 0.78,
        "sarcasm_joke_risk": sarcasm_joke_risk,
    }


@pytest.mark.llm
@pytest.mark.unit
def test_schema_scaffold_lists_required_strategy_fields() -> None:
    assert STRATEGY_EXTRACTION_SCHEMA_VERSION == "strategy-extraction-v1"
    assert {
        "ticker",
        "label",
        "direction",
        "instrument",
        "position_type",
        "time_horizon",
        "catalyst",
        "risk_or_hedge",
        "slang_terms",
        "evidence",
        "confidence",
        "sarcasm_joke_risk",
    }.issubset(STRATEGY_EXTRACTION_REQUIRED_FIELDS)


@pytest.mark.llm
@pytest.mark.unit
def test_validate_payload_accepts_discussed_strategy_and_sets_quote_offsets() -> None:
    evidence = _evidence()

    result = validate_llm_strategy_payloads(
        [_valid_payload()],
        evidence=(evidence,),
        provider_name="fixture-llm",
        occurred_at=NOW,
    )

    assert result.warnings == ()
    assert result.rejected_count == 0
    assert len(result.strategies) == 1
    strategy = result.strategies[0]
    assert strategy.label == "TSLA weekly calls into earnings"
    assert strategy.evidence[0].quote == "weekly calls into earnings"
    assert strategy.evidence[0].start_char == evidence.text.index("weekly calls")
    assert strategy.evidence[0].end_char == strategy.evidence[0].start_char + len(
        "weekly calls into earnings"
    )


@pytest.mark.llm
@pytest.mark.unit
def test_validate_payload_rejects_missing_evidence_and_mismatched_quote() -> None:
    evidence = _evidence()
    missing_evidence = _valid_payload(evidence_id="evidence-does-not-exist")
    mismatched_quote = _valid_payload(
        strategy_id="strategy-tsla-hallucinated-fda",
        quote="FDA approval tomorrow",
        catalyst="FDA approval",
    )

    result = validate_llm_strategy_payloads(
        [missing_evidence, mismatched_quote],
        evidence=(evidence,),
        provider_name="fixture-llm",
        occurred_at=NOW,
    )

    assert result.strategies == ()
    assert result.rejected_count == 2
    assert [warning.code for warning in result.warnings] == [
        WarningCode.LLM_EVIDENCE_MISMATCH,
        WarningCode.LLM_EVIDENCE_MISMATCH,
    ]


@pytest.mark.llm
@pytest.mark.unit
def test_validate_payload_rejects_direct_recommendations_and_unsupported_claims() -> None:
    evidence = _evidence(text="TSLA calls are getting spammed in the daily thread.")
    direct_recommendation = {
        **_valid_payload(quote="TSLA calls", catalyst=None),
        "recommendation_action": "qualified",
    }
    unsupported_catalyst = _valid_payload(
        strategy_id="strategy-tsla-fda-approval",
        quote="TSLA calls",
        catalyst="FDA approval tomorrow",
    )

    result = validate_llm_strategy_payloads(
        [direct_recommendation, unsupported_catalyst],
        evidence=(evidence,),
        provider_name="fixture-llm",
        occurred_at=NOW,
    )

    assert result.strategies == ()
    assert result.rejected_count == 2
    assert [warning.code for warning in result.warnings] == [
        WarningCode.UNSUPPORTED_CLAIM,
        WarningCode.UNSUPPORTED_CLAIM,
    ]


@pytest.mark.llm
@pytest.mark.unit
def test_validate_payload_preserves_sarcasm_joke_risk_without_promoting_claim_to_fact() -> None:
    evidence = _evidence(
        evidence_id="evidence-tsla-joke-risk",
        text="TSLA calls into earnings, totally guaranteed yacht money lol.",
    )

    result = validate_llm_strategy_payloads(
        [
            _valid_payload(
                evidence_id="evidence-tsla-joke-risk",
                quote="TSLA calls into earnings",
                sarcasm_joke_risk=0.87,
            )
        ],
        evidence=(evidence,),
        provider_name="fixture-llm",
        occurred_at=NOW,
    )

    assert result.warnings == ()
    assert result.strategies[0].sarcasm_joke_risk == 0.87
    assert result.strategies[0].confidence == 0.78


def _strategy(
    strategy_id: str,
    *,
    label: str,
    catalyst: str,
    evidence_id: str,
    confidence: float,
    direction: Direction = Direction.BULLISH,
    instrument: InstrumentType = InstrumentType.CALL_OPTION,
    position_type: PositionType = PositionType.LONG,
    sarcasm_joke_risk: float = 0.1,
) -> StrategyExtraction:
    return StrategyExtraction(
        strategy_id=strategy_id,
        ticker="TSLA",
        label=label,
        direction=direction,
        instrument=instrument,
        position_type=position_type,
        time_horizon=TimeHorizon.WEEKLY,
        catalyst=catalyst,
        risk_or_hedge="Premium at risk.",
        slang_terms=("calls",),
        evidence=(
            EvidenceReference(
                evidence_id=evidence_id,
                quote=label,
                relevance=0.9,
            ),
        ),
        confidence=confidence,
        sarcasm_joke_risk=sarcasm_joke_risk,
    )


@pytest.mark.llm
@pytest.mark.unit
def test_cluster_strategies_groups_near_duplicate_call_variants_into_earnings() -> None:
    clusters = cluster_strategies(
        (
            _strategy(
                "strategy-tsla-buy-calls",
                label="buy TSLA calls",
                catalyst="earnings",
                evidence_id="evidence-1",
                confidence=0.7,
            ),
            _strategy(
                "strategy-tsla-weekly-calls",
                label="weekly TSLA calls",
                catalyst="ER",
                evidence_id="evidence-2",
                confidence=0.8,
            ),
            _strategy(
                "strategy-tsla-calls-into-earnings",
                label="TSLA calls into earnings",
                catalyst="earnings report",
                evidence_id="evidence-3",
                confidence=0.9,
            ),
        )
    )

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.ticker == "TSLA"
    assert cluster.direction == Direction.BULLISH
    assert cluster.instrument == InstrumentType.CALL_OPTION
    assert cluster.time_horizon == TimeHorizon.WEEKLY
    assert cluster.catalyst_summary == "earnings"
    assert cluster.member_strategy_ids == (
        "strategy-tsla-buy-calls",
        "strategy-tsla-weekly-calls",
        "strategy-tsla-calls-into-earnings",
    )
    assert [reference.evidence_id for reference in cluster.evidence] == [
        "evidence-1",
        "evidence-2",
        "evidence-3",
    ]
    assert cluster.confidence == pytest.approx(0.8)


@pytest.mark.llm
@pytest.mark.unit
def test_cluster_strategies_flags_high_sarcasm_joke_risk() -> None:
    clusters = cluster_strategies(
        (
            _strategy(
                "strategy-tsla-joke-calls",
                label="TSLA calls are guaranteed yacht money lol",
                catalyst="earnings",
                evidence_id="evidence-joke-risk",
                confidence=0.78,
                sarcasm_joke_risk=0.88,
            ),
        ),
        occurred_at=NOW,
    )

    assert len(clusters) == 1
    warning = clusters[0].warnings[0]
    assert warning.code == WarningCode.UNSUPPORTED_CLAIM
    assert warning.message.startswith("High sarcasm/joke risk")
    assert warning.metadata["risk_type"] == "high_sarcasm_joke_risk"
    assert warning.metadata["risk"] == 0.88
    assert warning.metadata["member_strategy_ids"] == ["strategy-tsla-joke-calls"]


@pytest.mark.llm
@pytest.mark.unit
def test_cluster_strategies_flags_conflicting_source_evidence() -> None:
    clusters = cluster_strategies(
        (
            _strategy(
                "strategy-tsla-long-shares",
                label="long TSLA shares into earnings",
                catalyst="earnings",
                evidence_id="evidence-bullish",
                confidence=0.72,
                direction=Direction.BULLISH,
                instrument=InstrumentType.SHARES,
                position_type=PositionType.LONG,
            ),
            _strategy(
                "strategy-tsla-short-shares",
                label="short TSLA shares into earnings",
                catalyst="ER",
                evidence_id="evidence-bearish",
                confidence=0.69,
                direction=Direction.BEARISH,
                instrument=InstrumentType.SHARES,
                position_type=PositionType.SHORT,
            ),
        ),
        occurred_at=NOW,
    )

    assert len(clusters) == 2
    warnings = [cluster.warnings[0] for cluster in clusters]
    assert {warning.code for warning in warnings} == {WarningCode.PARTIAL_DATA}
    assert {warning.metadata["risk_type"] for warning in warnings} == {
        "conflicting_source_evidence"
    }
    assert all(warning.metadata["directions"] == ["bearish", "bullish"] for warning in warnings)


@pytest.mark.llm
@pytest.mark.unit
def test_parse_llm_json_response_accepts_strategies_envelope() -> None:
    parsed = parse_llm_json_response(
        """
        {
          "strategies": [
            {
              "strategy_id": "strategy-tsla-weekly-calls-earnings",
              "ticker": "TSLA"
            }
          ]
        }
        """
    )

    assert parsed == (
        {
            "strategy_id": "strategy-tsla-weekly-calls-earnings",
            "ticker": "TSLA",
        },
    )


@pytest.mark.llm
@pytest.mark.integration
def test_fixture_llm_extractor_returns_partial_when_some_payloads_are_rejected() -> None:
    evidence = (_evidence(),)
    extractor = FixtureLLMExtractor(
        payloads=(
            _valid_payload(),
            _valid_payload(strategy_id="strategy-bad-quote", quote="FDA approval tomorrow"),
        ),
        provider_name="fixture-llm",
        fetched_at=NOW,
    )
    request = ExtractionRequest(
        request_id="extract-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        evidence=evidence,
        prompt_version="lane-c-fixture-v1",
        schema_version=STRATEGY_EXTRACTION_SCHEMA_VERSION,
    )

    result = extractor.extract_strategies(request)

    assert result.status == ProviderStatus.PARTIAL
    assert result.data is not None
    assert len(result.data) == 1
    assert result.data[0].strategy_id == "strategy-tsla-weekly-calls-earnings"
    assert [warning.code for warning in result.warnings] == [WarningCode.LLM_EVIDENCE_MISMATCH]
