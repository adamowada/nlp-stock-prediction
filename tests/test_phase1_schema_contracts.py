from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    LEGACY_TRADE_INTERFACE_NOTE,
    LEGACY_TRADE_INTERFACE_STATUS,
    AnalysisBundle,
    AnalysisSignal,
    CredentialState,
    DataReference,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    FundamentalAnalysis,
    InstrumentType,
    MacroContext,
    MetricValue,
    PositionType,
    ProviderHealth,
    ProviderStatus,
    ProviderWarning,
    RecommendationAction,
    RetrievalMethod,
    RiskAssessment,
    RiskProfile,
    ScoreBreakdown,
    ScoreComponent,
    SectorContext,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    StrategyCluster,
    StrategyExtraction,
    TechnicalAnalysis,
    TextSpan,
    TimeHorizon,
    TradeCandidate,
    WarningCode,
    WarningSeverity,
)

RUN_DATE = date(2026, 5, 11)


def test_legacy_trade_interface_is_explicitly_quarantined() -> None:
    assert LEGACY_TRADE_INTERFACE_STATUS == "legacy_compatibility_only"
    assert "legacy" in LEGACY_TRADE_INTERFACE_NOTE
    assert "must not frame output as trade recommendations" in LEGACY_TRADE_INTERFACE_NOTE


def _now() -> datetime:
    return datetime(2026, 5, 11, 12, 0, tzinfo=UTC)


def _provenance(
    *,
    provider_name: str = "fixture-reddit",
    source_kind: SourceKind = SourceKind.REDDIT_POST,
    retrieval_method: RetrievalMethod = RetrievalMethod.FIXTURE,
) -> SourceProvenance:
    return SourceProvenance(
        provider_name=provider_name,
        source_kind=source_kind,
        retrieval_method=retrieval_method,
        fetched_at=_now(),
        observed_at=_now(),
        source_url="https://example.test/wsb/post/abc",
        permalink="https://reddit.test/r/wallstreetbets/comments/abc",
        raw_identifier="provider-record-abc",
        raw_snapshot_id="raw-snapshot-abc",
        query="TSLA daily thread",
        cache_key="fixture-reddit:2026-05-11:TSLA",
        freshness_status=FreshnessStatus.FRESH,
        freshness_seconds=300,
        provider_metadata={"fixture": True, "rank": 1, "tags": ["daily", "ticker-card"]},
    )


def _warning(
    *,
    code: WarningCode = WarningCode.PARTIAL_DATA,
    severity: WarningSeverity = WarningSeverity.WARNING,
    provider_name: str = "fixture-reddit",
) -> ProviderWarning:
    return ProviderWarning(
        code=code,
        severity=severity,
        message="fixture provider returned partial discussion data",
        provider_name=provider_name,
        retryable=True,
        occurred_at=_now(),
        stale_after=datetime(2026, 5, 11, 13, 0, tzinfo=UTC),
        raw_snapshot_id="raw-snapshot-warning",
        source_url="https://example.test/provider/status",
        metadata={"attempt": 2, "lane": "schema"},
    )


def _evidence_ref(evidence_id: str = "evidence-tsla-1") -> EvidenceReference:
    return EvidenceReference(
        evidence_id=evidence_id,
        quote="TSLA calls into CPI if volume holds",
        start_char=0,
        end_char=34,
        relevance=0.92,
    )


def _source_evidence(evidence_id: str = "evidence-tsla-1") -> SourceEvidence:
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=SourceKind.REDDIT_POST,
        ticker=" tsla ",
        title="Daily discussion",
        text="TSLA calls into CPI if volume holds",
        author_hash="author-hash-1",
        created_at=_now(),
        score=147,
        permalink="https://reddit.test/r/wallstreetbets/comments/abc",
        matched_tickers=("tsla", "NVDA"),
        match_spans=(TextSpan(text="TSLA", start_char=0, end_char=4),),
        provenance=_provenance(),
        metadata={"sentiment_hint": "bullish", "upvotes": 147},
    )


def _score_component(name: str = "evidence-strength") -> ScoreComponent:
    return ScoreComponent(
        name=name,
        raw_value=Decimal("0.82"),
        normalized_score=0.82,
        weight=0.35,
        contribution=0.287,
        rationale="Multiple recent discussion items support the same setup.",
        evidence=(_evidence_ref(),),
        data_reference_ids=("analysis-tsla-1",),
        warning_ids=("warning-partial-discussion",),
    )


def _score_breakdown(*, failed_gates: tuple[str, ...] = ()) -> ScoreBreakdown:
    return ScoreBreakdown(
        score_version="phase1-test-score-v1",
        overall_score=0.74,
        confidence=0.68,
        threshold=0.6,
        components=(_score_component(),),
        penalties=(
            ScoreComponent(
                name="joke-risk",
                raw_value=0.25,
                normalized_score=0.25,
                weight=0.1,
                contribution=-0.025,
                rationale="Some source language may be sarcastic.",
                evidence=(_evidence_ref(),),
            ),
        ),
        failed_gates=failed_gates,
    )


def _risk_assessment() -> RiskAssessment:
    return RiskAssessment(
        risk_profile=RiskProfile.EXPLORATORY,
        defined_risk=True,
        margin_required=False,
        max_account_risk_pct=Decimal("0.01"),
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("10"),
        position_size_pct=Decimal("0.01"),
        passed=True,
        sizing_basis="maximum loss is capped at one percent of account capital",
    )


@pytest.mark.schema
def test_contract_models_are_frozen_and_forbid_unknown_fields() -> None:
    provenance = _provenance()
    provider_name_field = "provider_name"

    with pytest.raises(ValidationError):
        setattr(provenance, provider_name_field, "changed-provider")

    with pytest.raises(ValidationError):
        SourceProvenance.model_validate(
            {
                "provider_name": "fixture-reddit",
                "source_kind": "reddit_post",
                "retrieval_method": "fixture",
                "fetched_at": _now(),
                "unexpected_field": "not part of the public schema",
            }
        )


@pytest.mark.schema
def test_base_aliases_normalize_tickers_and_reject_out_of_range_values() -> None:
    evidence = _source_evidence()

    assert evidence.ticker == "TSLA"
    assert evidence.matched_tickers == ("TSLA", "NVDA")
    assert evidence.text == "TSLA calls into CPI if volume holds"

    with pytest.raises(ValidationError):
        SourceEvidence.model_validate(
            {
                **_source_evidence().model_dump(),
                "ticker": "$TSLA",
            }
        )

    with pytest.raises(ValidationError):
        SourceEvidence.model_validate(
            {
                **_source_evidence().model_dump(),
                "evidence_id": 123,
            }
        )

    with pytest.raises(ValidationError):
        SourceEvidence.model_validate(
            {
                **_source_evidence().model_dump(),
                "text": None,
            }
        )

    with pytest.raises(ValidationError):
        StrategyExtraction.model_validate(
            {
                **_strategy_extraction().model_dump(),
                "ticker": None,
            }
        )

    with pytest.raises(ValidationError):
        StrategyExtraction.model_validate(
            {
                **_strategy_extraction().model_dump(),
                "confidence": 1.01,
            }
        )

    with pytest.raises(ValidationError):
        ScoreComponent.model_validate(
            {
                **_score_component().model_dump(),
                "normalized_score": -0.01,
            }
        )


@pytest.mark.schema
def test_json_metadata_fields_reject_non_serializable_values() -> None:
    with pytest.raises(ValidationError):
        SourceProvenance.model_validate(
            {
                **_provenance().model_dump(),
                "provider_metadata": {"bad": object()},
            }
        )

    with pytest.raises(ValidationError):
        MetricValue.model_validate(
            {
                "name": "relative-volume",
                "value": 2.1,
                "metadata": {"bad": object()},
            }
        )

    with pytest.raises(ValidationError):
        TradeCandidate.model_validate(
            {
                **_trade_candidate().model_dump(),
                "metadata": {"bad": object()},
            }
        )

    with pytest.raises(ValidationError):
        SourceProvenance.model_validate(
            {
                **_provenance().model_dump(),
                "provider_metadata": {"bad": float("nan")},
            }
        )


@pytest.mark.schema
def test_json_metadata_is_deeply_immutable_after_validation() -> None:
    provenance = _provenance()

    with pytest.raises(TypeError):
        provenance.provider_metadata["new"] = "value"
    tags = provenance.provider_metadata["tags"]
    assert isinstance(tags, list)
    with pytest.raises(TypeError):
        tags.append("mutated")


@pytest.mark.schema
def test_source_provenance_serializes_traceability_fields() -> None:
    provenance = _provenance()

    dumped = provenance.model_dump(mode="json")

    assert dumped["provider_name"] == "fixture-reddit"
    assert dumped["source_kind"] == "reddit_post"
    assert dumped["retrieval_method"] == "fixture"
    assert dumped["freshness_status"] == "fresh"
    assert dumped["raw_snapshot_id"] == "raw-snapshot-abc"
    assert dumped["provider_metadata"]["tags"] == ["daily", "ticker-card"]
    assert isinstance(dumped["fetched_at"], str)


@pytest.mark.schema
def test_source_provenance_requires_external_traceability_and_aware_timestamps() -> None:
    with pytest.raises(ValidationError, match="source_url or permalink"):
        SourceProvenance(
            provider_name="fixture-reddit",
            source_kind=SourceKind.REDDIT_POST,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=_now(),
            raw_identifier="raw-post-1",
            raw_snapshot_id="raw-snapshot-1",
            freshness_status=FreshnessStatus.FRESH,
        )

    with pytest.raises(ValidationError, match="raw_identifier"):
        SourceProvenance(
            provider_name="fixture-reddit",
            source_kind=SourceKind.REDDIT_POST,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=_now(),
            source_url="https://example.test/post/1",
            raw_snapshot_id="raw-snapshot-1",
            freshness_status=FreshnessStatus.FRESH,
        )

    with pytest.raises(ValidationError, match="timestamps must be timezone-aware"):
        SourceProvenance(
            provider_name="fixture-reddit",
            source_kind=SourceKind.REDDIT_POST,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=datetime(2026, 5, 11, 12, 0),
            source_url="https://example.test/post/1",
            raw_identifier="raw-post-1",
            raw_snapshot_id="raw-snapshot-1",
            freshness_status=FreshnessStatus.FRESH,
        )


@pytest.mark.schema
def test_provider_warning_and_health_validate_bounds_and_status_helpers() -> None:
    health = ProviderHealth(
        provider_name="fixture-reddit",
        status=ProviderStatus.OK,
        checked_at=_now(),
        credential_state=CredentialState.NOT_REQUIRED,
        latency_ms=12,
        rate_limit_remaining=99,
        last_success_at=_now(),
        warnings=(_warning(severity=WarningSeverity.INFO),),
    )

    assert health.ok is True
    assert health.warnings[0].metadata["lane"] == "schema"

    failed_health = ProviderHealth(
        provider_name="fixture-reddit",
        status=ProviderStatus.FAILED,
        checked_at=_now(),
        credential_state=CredentialState.NOT_REQUIRED,
    )
    assert failed_health.ok is False

    with pytest.raises(ValidationError):
        ProviderHealth.model_validate(
            {
                "provider_name": "fixture-reddit",
                "status": "ok",
                "checked_at": _now(),
                "latency_ms": -1,
            }
        )


@pytest.mark.schema
def test_evidence_reference_validates_offsets_and_relevance_bounds() -> None:
    reference = _evidence_ref()

    assert reference.evidence_id == "evidence-tsla-1"
    assert reference.relevance == 0.92

    with pytest.raises(ValidationError):
        EvidenceReference(
            evidence_id="evidence-tsla-1",
            start_char=-1,
        )

    with pytest.raises(ValidationError):
        EvidenceReference(
            evidence_id="evidence-tsla-1",
            start_char=12,
            end_char=4,
        )

    with pytest.raises(ValidationError):
        EvidenceReference(
            evidence_id="evidence-tsla-1",
            relevance=1.01,
        )


@pytest.mark.schema
def test_data_reference_keeps_auditable_json_metadata_and_type_taxonomy() -> None:
    reference = DataReference(
        reference_id="analysis-tsla-1",
        reference_type="analysis",
        path="audit/analysis/tsla.json",
        sha256="a" * 64,
        metadata={"ticker": "TSLA", "records": 1},
    )

    dumped = reference.model_dump(mode="json")

    assert dumped["reference_type"] == "analysis"
    assert dumped["metadata"]["ticker"] == "TSLA"

    with pytest.raises(ValidationError):
        DataReference.model_validate(
            {
                **reference.model_dump(),
                "reference_type": "unknown_artifact",
            }
        )


@pytest.mark.schema
def test_text_span_requires_non_empty_text_and_monotonic_offsets() -> None:
    assert TextSpan(text="TSLA", start_char=0, end_char=4).text == "TSLA"

    with pytest.raises(ValidationError):
        TextSpan(text="   ", start_char=0, end_char=0)

    with pytest.raises(ValidationError):
        TextSpan(text="TSLA", start_char=12, end_char=4)


@pytest.mark.schema
def test_source_evidence_preserves_normalized_evidence_and_provenance() -> None:
    evidence = _source_evidence()
    dumped = evidence.model_dump(mode="json")

    assert evidence.ticker == "TSLA"
    assert evidence.matched_tickers == ("TSLA", "NVDA")
    assert evidence.match_spans[0].text == "TSLA"
    assert dumped["provenance"]["provider_name"] == "fixture-reddit"
    assert dumped["provenance"]["raw_identifier"] == "provider-record-abc"
    assert dumped["metadata"]["sentiment_hint"] == "bullish"


@pytest.mark.schema
def test_source_evidence_rejects_empty_text_and_bad_match_spans() -> None:
    with pytest.raises(ValidationError):
        SourceEvidence.model_validate(
            {
                **_source_evidence().model_dump(),
                "text": "   ",
            }
        )

    with pytest.raises(ValidationError):
        SourceEvidence.model_validate(
            {
                **_source_evidence().model_dump(),
                "match_spans": [{"text": "TSLA", "start_char": 8, "end_char": 4}],
            }
        )


def _strategy_extraction() -> StrategyExtraction:
    return StrategyExtraction(
        strategy_id="strategy-tsla-calls-cpi",
        ticker="tsla",
        label="TSLA calls into CPI",
        direction=Direction.BULLISH,
        instrument=InstrumentType.CALL_OPTION,
        position_type=PositionType.LONG,
        time_horizon=TimeHorizon.WEEKLY,
        catalyst="CPI print and sustained call volume",
        risk_or_hedge="Treat as speculative and size small.",
        slang_terms=("calls", "weekly"),
        evidence=(_evidence_ref(),),
        confidence=0.72,
        sarcasm_joke_risk=0.2,
        warnings=(_warning(code=WarningCode.UNSUPPORTED_CLAIM),),
    )


@pytest.mark.schema
def test_strategy_extraction_requires_evidence_and_valid_confidence_inputs() -> None:
    extraction = _strategy_extraction()

    assert extraction.ticker == "TSLA"
    assert extraction.evidence[0].evidence_id == "evidence-tsla-1"

    with pytest.raises(ValidationError):
        StrategyExtraction.model_validate(
            {
                **extraction.model_dump(),
                "evidence": (),
            }
        )

    with pytest.raises(ValidationError):
        StrategyExtraction.model_validate(
            {
                **extraction.model_dump(),
                "sarcasm_joke_risk": -0.01,
            }
        )


@pytest.mark.schema
def test_strategy_cluster_requires_members_and_cited_evidence() -> None:
    cluster = StrategyCluster(
        cluster_id="cluster-tsla-calls-cpi",
        ticker="TSLA",
        direction=Direction.BULLISH,
        instrument=InstrumentType.CALL_OPTION,
        time_horizon=TimeHorizon.WEEKLY,
        catalyst_summary="CPI print discussed as a near-term volatility catalyst.",
        member_strategy_ids=("strategy-tsla-calls-cpi",),
        evidence=(_evidence_ref(),),
        confidence=0.69,
        warnings=(_warning(),),
    )

    assert cluster.member_strategy_ids == ("strategy-tsla-calls-cpi",)
    assert cluster.evidence[0].quote is not None

    with pytest.raises(ValidationError):
        StrategyCluster.model_validate(
            {
                **cluster.model_dump(),
                "member_strategy_ids": (),
            }
        )

    with pytest.raises(ValidationError):
        StrategyCluster.model_validate(
            {
                **cluster.model_dump(),
                "evidence": (),
            }
        )


@pytest.mark.schema
def test_metric_value_serializes_values_with_optional_provenance() -> None:
    metric = MetricValue(
        name="relative-volume",
        value=Decimal("1.75"),
        unit="x",
        as_of=RUN_DATE,
        provenance=_provenance(
            provider_name="fixture-market-data",
            source_kind=SourceKind.MARKET_DATA,
        ),
        metadata={"window": "20d"},
    )

    dumped = metric.model_dump(mode="json")

    assert dumped["name"] == "relative-volume"
    assert dumped["value"] == "1.75"
    assert dumped["provenance"]["source_kind"] == "market_data"


@pytest.mark.schema
def test_analysis_components_validate_summary_confidence_and_ticker_fields() -> None:
    technical = TechnicalAnalysis(
        ticker=" tsla ",
        summary="Price is testing prior resistance on above-average volume.",
        signal=AnalysisSignal.SUPPORTS,
        confidence=0.64,
        metrics=(
            MetricValue(
                name="relative-volume",
                value=Decimal("1.75"),
                unit="x",
                as_of=RUN_DATE,
                provenance=_provenance(source_kind=SourceKind.MARKET_DATA),
            ),
        ),
        evidence=(_evidence_ref(),),
        warnings=(_warning(),),
        assumptions=("Market data fixture is treated as current for schema tests.",),
        trend="uptrend",
        support_levels=(Decimal("170.00"),),
        resistance_levels=(Decimal("185.50"),),
    )
    sector = SectorContext(
        ticker="tsla",
        summary="Auto peers are mixed while mega-cap risk appetite is supportive.",
        signal=AnalysisSignal.MIXED,
        confidence=0.5,
        sector="Consumer Discretionary",
        peers=("gm", "F"),
        benchmark_symbol="xly",
    )

    assert technical.ticker == "TSLA"
    assert technical.resistance_levels == (Decimal("185.50"),)
    assert sector.peers == ("GM", "F")
    assert sector.benchmark_symbol == "XLY"

    with pytest.raises(ValidationError):
        TechnicalAnalysis.model_validate(
            {
                **technical.model_dump(),
                "summary": "   ",
            }
        )

    with pytest.raises(ValidationError):
        SectorContext.model_validate(
            {
                **sector.model_dump(),
                "confidence": 1.5,
            }
        )


@pytest.mark.schema
def test_analysis_bundle_preserves_multi_lane_context_and_json_inputs() -> None:
    bundle = AnalysisBundle(
        analysis_id="analysis-tsla-1",
        ticker="tsla",
        as_of=RUN_DATE,
        strategy_cluster_ids=("cluster-tsla-calls-cpi",),
        technical=TechnicalAnalysis(
            ticker="TSLA",
            summary="Technical setup supports the observed bullish discussion.",
            signal=AnalysisSignal.SUPPORTS,
            confidence=0.64,
            evidence=(_evidence_ref(),),
        ),
        fundamental=FundamentalAnalysis(
            ticker="TSLA",
            summary="Valuation remains a risk despite revenue growth.",
            signal=AnalysisSignal.CONFLICTS,
            confidence=0.52,
            valuation_summary="High multiple versus broader market.",
        ),
        sector=SectorContext(
            ticker="TSLA",
            summary="Sector context is mixed.",
            signal=AnalysisSignal.MIXED,
            confidence=0.5,
        ),
        macro=MacroContext(
            as_of=RUN_DATE,
            summary="Macro event risk is elevated around CPI.",
            signal=AnalysisSignal.MIXED,
            confidence=0.61,
            horizon=TimeHorizon.WEEKLY,
            supportive_factors=("Risk appetite has improved.",),
            conflicting_factors=("Inflation surprise could pressure growth stocks.",),
        ),
        signals=(AnalysisSignal.SUPPORTS, AnalysisSignal.CONFLICTS, AnalysisSignal.MIXED),
        contradictions=("Fundamental valuation conflicts with near-term momentum.",),
        assumptions=("Fixture market data is current as of the run date.",),
        confidence_inputs={"technical": 0.64, "fundamental": 0.52, "macro": 0.61},
        evidence=(_evidence_ref(),),
        warnings=(_warning(),),
    )

    dumped = bundle.model_dump(mode="json")

    assert bundle.ticker == "TSLA"
    assert dumped["technical"]["signal"] == "supports"
    assert dumped["macro"]["horizon"] == "weekly"
    assert dumped["confidence_inputs"]["technical"] == 0.64

    with pytest.raises(ValidationError):
        AnalysisBundle.model_validate(
            {
                **bundle.model_dump(),
                "confidence_inputs": {"bad": object()},
            }
        )


@pytest.mark.schema
def test_score_component_and_breakdown_validate_scoring_bounds() -> None:
    breakdown = _score_breakdown(failed_gates=("requires-live-market-confirmation",))

    assert breakdown.components[0].normalized_score == 0.82
    assert breakdown.penalties[0].contribution < 0
    assert breakdown.failed_gates == ("requires-live-market-confirmation",)

    with pytest.raises(ValidationError):
        ScoreComponent.model_validate(
            {
                **_score_component().model_dump(),
                "weight": -0.01,
            }
        )

    with pytest.raises(ValidationError):
        ScoreBreakdown.model_validate(
            {
                **breakdown.model_dump(),
                "threshold": 1.1,
            }
        )

    with pytest.raises(ValidationError):
        ScoreBreakdown.model_validate(
            {
                **breakdown.model_dump(),
                "components": (),
            }
        )


@pytest.mark.schema
def test_risk_assessment_enforces_account_risk_and_sizing_bounds() -> None:
    risk = _risk_assessment()

    assert risk.risk_profile == RiskProfile.EXPLORATORY
    assert risk.passed is True
    assert risk.position_size_pct == Decimal("0.01")

    with pytest.raises(ValidationError):
        RiskAssessment.model_validate(
            {
                **risk.model_dump(),
                "max_account_risk_pct": Decimal("1.01"),
            }
        )

    with pytest.raises(ValidationError):
        RiskAssessment.model_validate(
            {
                **risk.model_dump(),
                "position_size_pct": Decimal("-0.01"),
            }
        )

    with pytest.raises(ValidationError):
        RiskAssessment.model_validate(
            {
                **risk.model_dump(),
                "account_capital": Decimal("-1"),
            }
        )


def _trade_candidate(
    *,
    action: RecommendationAction = RecommendationAction.QUALIFIED,
    evidence: tuple[EvidenceReference, ...] = (_evidence_ref(),),
) -> TradeCandidate:
    return TradeCandidate(
        candidate_id="candidate-tsla-calls-cpi",
        ticker="tsla",
        action=action,
        strategy_cluster_id="cluster-tsla-calls-cpi",
        instrument=InstrumentType.CALL_OPTION,
        direction=Direction.BULLISH,
        position_type=PositionType.LONG,
        time_horizon=TimeHorizon.WEEKLY,
        thesis="Observed discussion and technical context support a small defined-risk idea.",
        entry_logic="Only consider after price confirms above prior resistance.",
        invalidation_criteria="Invalidate if price loses support or CPI reaction reverses.",
        risk_plan=_risk_assessment(),
        catalysts=("CPI print", "sustained options volume"),
        score=_score_breakdown(),
        assumptions=("Fixture evidence is representative for this schema contract.",),
        risks=("High volatility can invalidate the setup quickly.",),
        contradictions=("Fundamental valuation remains stretched.",),
        evidence=evidence,
        score_input_ids=("analysis-tsla-1", "cluster-tsla-calls-cpi"),
        warnings=(_warning(),),
        metadata={"run_date": "2026-05-11", "trace_id": "trace-candidate-tsla"},
    )


@pytest.mark.schema
def test_trade_candidate_requires_evidence_for_actionable_recommendations() -> None:
    with pytest.raises(ValidationError):
        _trade_candidate(evidence=())

    no_trade = _trade_candidate(
        action=RecommendationAction.NO_TRADE,
        evidence=(),
    )

    assert no_trade.action == RecommendationAction.NO_TRADE
    assert no_trade.evidence == ()


@pytest.mark.schema
def test_qualified_trade_candidate_requires_passing_risk_and_score_gates() -> None:
    candidate_payload = _trade_candidate().model_dump()

    with pytest.raises(ValidationError, match="pass risk gates"):
        TradeCandidate.model_validate(
            {
                **candidate_payload,
                "risk_plan": {
                    **_risk_assessment().model_dump(),
                    "passed": False,
                },
            }
        )

    with pytest.raises(ValidationError, match="failed risk gates"):
        TradeCandidate.model_validate(
            {
                **candidate_payload,
                "risk_plan": {
                    **_risk_assessment().model_dump(),
                    "failed_gates": ("position-size-too-large",),
                },
            }
        )

    with pytest.raises(ValidationError, match="failed score gates"):
        TradeCandidate.model_validate(
            {
                **candidate_payload,
                "score": _score_breakdown(
                    failed_gates=("requires-live-market-confirmation",)
                ).model_dump(),
            }
        )

    with pytest.raises(ValidationError, match="meet score threshold"):
        TradeCandidate.model_validate(
            {
                **candidate_payload,
                "score": {
                    **_score_breakdown().model_dump(),
                    "overall_score": 0.2,
                    "threshold": 0.7,
                },
            }
        )


@pytest.mark.schema
def test_trade_candidate_serializes_recommendation_risk_and_evidence_spine() -> None:
    candidate = _trade_candidate()
    dumped = candidate.model_dump(mode="json")

    assert candidate.ticker == "TSLA"
    assert dumped["action"] == "qualified"
    assert dumped["risk_plan"]["risk_profile"] == "exploratory"
    assert dumped["score"]["score_version"] == "phase1-test-score-v1"
    assert dumped["score"]["components"][0]["evidence"][0]["evidence_id"] == "evidence-tsla-1"
    assert dumped["evidence"][0]["quote"] == "TSLA calls into CPI if volume holds"

    with pytest.raises(ValidationError):
        TradeCandidate.model_validate(
            {
                **candidate.model_dump(),
                "entry_logic": "   ",
            }
        )
