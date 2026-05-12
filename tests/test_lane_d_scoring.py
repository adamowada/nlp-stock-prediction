from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal

import pytest

from nlp_stock_prediction.contracts import (
    AnalysisBundle,
    AnalysisSignal,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    FundamentalAnalysis,
    InstrumentType,
    JsonObject,
    MacroContext,
    PositionType,
    ProviderWarning,
    RecommendationAction,
    RiskProfile,
    SectorContext,
    StrategyCluster,
    TechnicalAnalysis,
    TechnicalMlSignal,
    TimeHorizon,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.scoring import (
    RecommendationSignals,
    build_no_trade_summary,
    score_strategy_cluster,
    select_qualified_candidates,
)
from nlp_stock_prediction.scoring.risk import assess_risk

RUN_DATE = date(2026, 5, 11)
DISCLAIMER_ID = "educational-disclaimer-v1"


def _evidence_ref(evidence_id: str = "evidence-nvda-1") -> EvidenceReference:
    return EvidenceReference(
        evidence_id=evidence_id,
        quote="NVDA call spread into earnings if it holds the 20 day",
        start_char=0,
        end_char=54,
        relevance=0.91,
    )


def _cluster(
    *,
    ticker: str = "NVDA",
    instrument: InstrumentType = InstrumentType.OPTION_SPREAD,
    direction: Direction = Direction.BULLISH,
    confidence: float = 0.78,
    warnings: tuple[ProviderWarning, ...] = (),
) -> StrategyCluster:
    return StrategyCluster(
        cluster_id=f"cluster-{ticker.lower()}-spread",
        ticker=ticker,
        direction=direction,
        instrument=instrument,
        time_horizon=TimeHorizon.WEEKLY,
        catalyst_summary="Earnings and sustained call-spread discussion.",
        member_strategy_ids=(f"strategy-{ticker.lower()}-spread-1",),
        evidence=(_evidence_ref(),),
        confidence=confidence,
        warnings=warnings,
    )


def _ml_signal(
    *,
    signal: AnalysisSignal = AnalysisSignal.SUPPORTS,
    status: Literal["usable", "weak", "stale", "conflicting", "unavailable"] = "usable",
    calibrated_confidence: float = 0.31,
) -> TechnicalMlSignal:
    probability = 0.61
    if signal == AnalysisSignal.CONFLICTS:
        probability = 0.39
    elif signal == AnalysisSignal.MIXED:
        probability = 0.50
    return TechnicalMlSignal(
        model_hash="fixture-model-hash",
        dataset_hash="fixture-dataset-hash",
        as_of=RUN_DATE,
        feature_end=RUN_DATE,
        prediction_horizon_sessions=1,
        probability_positive=probability,
        calibrated_confidence=calibrated_confidence,
        signal=signal,
        status=status,
        freshness_status=FreshnessStatus.FRESH,
        validation_accuracy=0.58,
        validation_brier_score=0.21,
        warning_ids=(f"ml-technical-signal:{status}",) if status != "usable" else (),
    )


def _analysis_bundle(
    *,
    ticker: str = "NVDA",
    technical_signal: AnalysisSignal = AnalysisSignal.SUPPORTS,
    technical_ml_signal: TechnicalMlSignal | None = None,
    fundamental_signal: AnalysisSignal = AnalysisSignal.SUPPORTS,
    sector_signal: AnalysisSignal = AnalysisSignal.SUPPORTS,
    macro_signal: AnalysisSignal = AnalysisSignal.SUPPORTS,
    contradictions: tuple[str, ...] = (),
) -> AnalysisBundle:
    return AnalysisBundle(
        analysis_id=f"analysis-{ticker.lower()}-1",
        ticker=ticker,
        as_of=RUN_DATE,
        strategy_cluster_ids=(f"cluster-{ticker.lower()}-spread",),
        technical=TechnicalAnalysis(
            ticker=ticker,
            summary="Price trend and volume support the observed setup.",
            signal=technical_signal,
            confidence=0.82,
            trend="uptrend" if technical_signal != AnalysisSignal.CONFLICTS else "downtrend",
            support_levels=(Decimal("940"),),
            resistance_levels=(Decimal("1010"),),
            ml_signal=technical_ml_signal,
        ),
        fundamental=FundamentalAnalysis(
            ticker=ticker,
            summary="Fundamentals support the setup despite valuation risk.",
            signal=fundamental_signal,
            confidence=0.74,
            valuation_summary="Premium valuation is offset by growth.",
            profitability_summary="Highly profitable.",
            growth_summary="Revenue growth is strong.",
        ),
        sector=SectorContext(
            ticker=ticker,
            summary="Sector peers support the relative setup.",
            signal=sector_signal,
            confidence=0.68,
            sector="Semiconductors",
            peers=("AMD", "INTC"),
        ),
        macro=MacroContext(
            as_of=RUN_DATE,
            summary="Macro backdrop supports the strategy horizon.",
            signal=macro_signal,
            confidence=0.66,
            horizon=TimeHorizon.WEEKLY,
            supportive_factors=("Cooling inflation supports risk appetite.",),
        ),
        signals=(technical_signal, fundamental_signal, sector_signal, macro_signal),
        contradictions=contradictions,
        assumptions=("Fixture analysis is deterministic.",),
        confidence_inputs={
            "technical": 0.82,
            "fundamental": 0.74,
            "sector": 0.68,
            "macro": 0.66,
        },
        evidence=(_evidence_ref(),),
    )


def _cluster_warning(risk_type: str, *, risk: float | None = None) -> ProviderWarning:
    metadata: JsonObject = {"risk_type": risk_type}
    if risk is not None:
        metadata["risk"] = risk
    return ProviderWarning(
        code=(
            WarningCode.UNSUPPORTED_CLAIM
            if risk_type == "high_sarcasm_joke_risk"
            else WarningCode.PARTIAL_DATA
        ),
        severity=WarningSeverity.WARNING,
        message=f"Fixture cluster warning: {risk_type}",
        provider_name="strategy-clustering",
        occurred_at=datetime(2026, 5, 11, 12, 0, tzinfo=UTC),
        metadata=metadata,
    )


@pytest.mark.unit
def test_default_risk_controls_reject_margin_naked_options_and_oversized_loss() -> None:
    allowed = assess_risk(
        instrument=InstrumentType.SHARES,
        position_type=PositionType.LONG,
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("10"),
        risk_profile=RiskProfile.EXPLORATORY,
    )
    oversized = assess_risk(
        instrument=InstrumentType.SHARES,
        position_type=PositionType.LONG,
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("10.01"),
        risk_profile=RiskProfile.EXPLORATORY,
    )
    naked_margin_option = assess_risk(
        instrument=InstrumentType.CALL_OPTION,
        position_type=PositionType.SHORT,
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("50"),
        uses_margin=True,
        risk_profile=RiskProfile.EXPLORATORY,
    )

    assert allowed.passed is True
    assert allowed.max_account_risk_pct == Decimal("0.01")
    assert allowed.position_size_pct == Decimal("0.01")
    assert oversized.passed is False
    assert "max-account-risk-exceeded" in oversized.failed_gates
    assert naked_margin_option.passed is False
    assert "margin-not-allowed" in naked_margin_option.failed_gates
    assert "naked-options-not-allowed" in naked_margin_option.failed_gates


@pytest.mark.unit
def test_risk_controls_use_percentage_only_sizing_without_capital() -> None:
    risk = assess_risk(
        instrument=InstrumentType.OPTION_SPREAD,
        position_type=PositionType.DEFINED_RISK,
        max_loss_estimate=Decimal("75"),
        risk_profile=RiskProfile.BALANCED,
    )

    assert risk.passed is True
    assert risk.account_capital is None
    assert risk.position_size_pct == Decimal("0.01")
    assert risk.sizing_basis is not None
    assert "percentage-only" in risk.sizing_basis


@pytest.mark.unit
def test_scoring_emits_qualified_defined_risk_candidate_with_auditable_components() -> None:
    cluster = _cluster()
    analysis = _analysis_bundle()
    signals = RecommendationSignals(
        reddit_mentions=9,
        reddit_unique_sources=5,
        reddit_relevance=0.86,
        social_mentions=3,
        news_mentions=2,
        catalyst_relevance=0.9,
        liquidity_score=0.82,
    )

    candidate = score_strategy_cluster(
        cluster=cluster,
        analysis=analysis,
        signals=signals,
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("8"),
        risk_profile=RiskProfile.EXPLORATORY,
        disclaimer_id=DISCLAIMER_ID,
    )
    component_names = {component.name for component in candidate.score.components}

    assert candidate.action == RecommendationAction.QUALIFIED
    assert candidate.risk_plan.passed is True
    assert candidate.risk_plan.max_loss_estimate == Decimal("8")
    assert candidate.score.overall_score >= candidate.score.threshold
    assert {
        "reddit-strength",
        "social-news-catalyst",
        "technical-alignment",
        "fundamentals",
        "sector-context",
        "macro-context",
        "liquidity-risk-suitability",
    } <= component_names
    assert candidate.score.failed_gates == ()
    assert candidate.score_input_ids == ("analysis-nvda-1", "cluster-nvda-spread")
    confidence_inputs = candidate.metadata["confidence_inputs"]
    assert isinstance(confidence_inputs, dict)
    assert confidence_inputs["technical"] == 0.82
    assert candidate.evidence == cluster.evidence


@pytest.mark.unit
def test_ml_signal_conflict_penalizes_and_prevents_qualification() -> None:
    candidate = score_strategy_cluster(
        cluster=_cluster(direction=Direction.BULLISH),
        analysis=_analysis_bundle(technical_ml_signal=_ml_signal(signal=AnalysisSignal.CONFLICTS)),
        signals=RecommendationSignals(
            reddit_mentions=9,
            reddit_unique_sources=5,
            reddit_relevance=0.90,
            social_mentions=4,
            news_mentions=2,
            catalyst_relevance=0.95,
            liquidity_score=0.90,
        ),
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("8"),
        risk_profile=RiskProfile.EXPLORATORY,
        disclaimer_id=DISCLAIMER_ID,
    )

    assert candidate.action == RecommendationAction.WATCH
    assert "ml-technical-conflict" in candidate.score.failed_gates
    assert any(
        penalty.name == "ml-technical-conflict-penalty" for penalty in candidate.score.penalties
    )


@pytest.mark.unit
def test_weak_ml_signal_penalizes_and_prevents_qualification() -> None:
    candidate = score_strategy_cluster(
        cluster=_cluster(direction=Direction.BULLISH),
        analysis=_analysis_bundle(
            technical_ml_signal=_ml_signal(
                signal=AnalysisSignal.SUPPORTS,
                status="weak",
                calibrated_confidence=0.08,
            )
        ),
        signals=RecommendationSignals(
            reddit_mentions=9,
            reddit_unique_sources=5,
            reddit_relevance=0.90,
            social_mentions=4,
            news_mentions=2,
            catalyst_relevance=0.95,
            liquidity_score=0.90,
        ),
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("8"),
        risk_profile=RiskProfile.EXPLORATORY,
        disclaimer_id=DISCLAIMER_ID,
    )

    assert candidate.action == RecommendationAction.WATCH
    assert "ml-signal-not-actionable" in candidate.score.failed_gates
    penalty_names = {penalty.name for penalty in candidate.score.penalties}
    assert "ml-signal-quality-penalty" in penalty_names
    assert "ml-signal-weak-penalty" in penalty_names


@pytest.mark.unit
def test_contradiction_penalties_lower_score_and_prevent_qualification() -> None:
    cluster = _cluster(confidence=0.72)
    analysis = _analysis_bundle(
        technical_signal=AnalysisSignal.SUPPORTS,
        fundamental_signal=AnalysisSignal.CONFLICTS,
        sector_signal=AnalysisSignal.MIXED,
        macro_signal=AnalysisSignal.CONFLICTS,
        contradictions=(
            "Premium valuation conflicts with near-term call buying.",
            "Macro volatility conflicts with weekly options timing.",
            "Sector breadth is weaker than the ticker setup.",
        ),
    )
    signals = RecommendationSignals(
        reddit_mentions=4,
        reddit_unique_sources=2,
        reddit_relevance=0.62,
        social_mentions=0,
        news_mentions=1,
        catalyst_relevance=0.35,
        liquidity_score=0.55,
    )

    candidate = score_strategy_cluster(
        cluster=cluster,
        analysis=analysis,
        signals=signals,
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("8"),
        risk_profile=RiskProfile.EXPLORATORY,
        disclaimer_id=DISCLAIMER_ID,
    )

    assert candidate.action == RecommendationAction.WATCH
    assert "conflicting-analysis" in candidate.score.failed_gates
    assert "score-below-threshold" in candidate.score.failed_gates
    assert any(penalty.name == "contradiction-penalty" for penalty in candidate.score.penalties)
    assert candidate.contradictions == analysis.contradictions


@pytest.mark.unit
def test_unsupported_recommendation_instrument_is_avoided_and_not_qualified() -> None:
    candidate = score_strategy_cluster(
        cluster=_cluster(instrument=InstrumentType.UNKNOWN),
        analysis=_analysis_bundle(),
        signals=RecommendationSignals(
            reddit_mentions=9,
            reddit_unique_sources=5,
            reddit_relevance=0.90,
            social_mentions=4,
            news_mentions=2,
            catalyst_relevance=0.95,
            liquidity_score=0.90,
        ),
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("8"),
        risk_profile=RiskProfile.EXPLORATORY,
        disclaimer_id=DISCLAIMER_ID,
    )

    qualified = select_qualified_candidates((candidate,))

    assert candidate.action == RecommendationAction.AVOID
    assert candidate.position_type == PositionType.WATCH_ONLY
    assert candidate.risk_plan.passed is False
    assert "unsupported-instrument" in candidate.risk_plan.failed_gates
    assert "defined-risk-required" in candidate.risk_plan.failed_gates
    assert qualified == ()
    assert candidate.risks[0].startswith("Risk gates failed:")


@pytest.mark.unit
def test_high_sarcasm_joke_warning_penalizes_and_prevents_qualification() -> None:
    warning = _cluster_warning("high_sarcasm_joke_risk", risk=0.88)
    candidate = score_strategy_cluster(
        cluster=_cluster(warnings=(warning,)),
        analysis=_analysis_bundle(),
        signals=RecommendationSignals(
            reddit_mentions=9,
            reddit_unique_sources=5,
            reddit_relevance=0.90,
            social_mentions=4,
            news_mentions=2,
            catalyst_relevance=0.95,
            liquidity_score=0.90,
        ),
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("8"),
        risk_profile=RiskProfile.EXPLORATORY,
        disclaimer_id=DISCLAIMER_ID,
    )

    assert candidate.action == RecommendationAction.WATCH
    assert candidate.risk_plan.passed is True
    assert "high-sarcasm-joke-risk" in candidate.score.failed_gates
    assert any(penalty.name == "sarcasm-joke-risk-penalty" for penalty in candidate.score.penalties)
    assert candidate.warnings == (warning,)
    assert select_qualified_candidates((candidate,)) == ()


@pytest.mark.unit
def test_conflicting_source_evidence_warning_penalizes_and_prevents_qualification() -> None:
    warning = _cluster_warning("conflicting_source_evidence")
    candidate = score_strategy_cluster(
        cluster=_cluster(warnings=(warning,)),
        analysis=_analysis_bundle(),
        signals=RecommendationSignals(
            reddit_mentions=9,
            reddit_unique_sources=5,
            reddit_relevance=0.90,
            social_mentions=4,
            news_mentions=2,
            catalyst_relevance=0.95,
            liquidity_score=0.90,
        ),
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("8"),
        risk_profile=RiskProfile.EXPLORATORY,
        disclaimer_id=DISCLAIMER_ID,
    )

    assert candidate.action == RecommendationAction.WATCH
    assert "conflicting-source-evidence" in candidate.score.failed_gates
    assert any(penalty.name == "source-conflict-penalty" for penalty in candidate.score.penalties)
    assert candidate.warnings == (warning,)
    assert select_qualified_candidates((candidate,)) == ()


@pytest.mark.unit
def test_no_trade_days_are_valid_when_no_candidates_qualify() -> None:
    candidate = score_strategy_cluster(
        cluster=_cluster(confidence=0.40),
        analysis=_analysis_bundle(
            technical_signal=AnalysisSignal.CONFLICTS,
            fundamental_signal=AnalysisSignal.CONFLICTS,
            sector_signal=AnalysisSignal.CONFLICTS,
            macro_signal=AnalysisSignal.CONFLICTS,
            contradictions=("Every analysis layer conflicts with the observed strategy.",),
        ),
        signals=RecommendationSignals(
            reddit_mentions=1,
            reddit_unique_sources=1,
            reddit_relevance=0.25,
            social_mentions=0,
            news_mentions=0,
            catalyst_relevance=0.0,
            liquidity_score=0.35,
        ),
        account_capital=Decimal("1000"),
        max_loss_estimate=Decimal("12"),
        risk_profile=RiskProfile.EXPLORATORY,
        disclaimer_id=DISCLAIMER_ID,
    )

    qualified = select_qualified_candidates((candidate,))
    summary = build_no_trade_summary((candidate,))

    assert qualified == ()
    assert candidate.action == RecommendationAction.AVOID
    assert "No qualified trade candidates" in summary
    assert "risk gates" in summary
