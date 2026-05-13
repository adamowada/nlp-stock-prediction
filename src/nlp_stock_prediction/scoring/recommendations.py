"""Recommendation scoring with auditable components and penalties."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from nlp_stock_prediction.analysis._metrics import clamp
from nlp_stock_prediction.contracts import (
    AnalysisBundle,
    AnalysisComponent,
    AnalysisSignal,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    InstrumentType,
    PositionType,
    ProviderWarning,
    RecommendationAction,
    RiskProfile,
    ScoreBreakdown,
    ScoreComponent,
    StrategyCluster,
    TechnicalMlSignal,
    TradeCandidate,
)
from nlp_stock_prediction.scoring.risk import assess_risk

SCORE_VERSION = "lane-d-score-v2"

_TIMESFM_MODEL_KIND = "timesfm_2_5_lora_evaluation"
_TIMESFM_MAX_INTERVAL_WIDTH = 0.20
_TIMESFM_MIN_VALIDATION_ACCURACY = 0.52
_TIMESFM_MIN_CONFIDENCE = 0.15
_TIMESFM_MAX_TECHNICAL_BOOST = 0.08
_TECHNICAL_ALIGNMENT_WEIGHT = 0.18


@dataclass(frozen=True, slots=True)
class _TechnicalMlScoringEffect:
    base_score: float
    final_score: float
    adjustment: float
    rationale: str
    data_reference_ids: tuple[str, ...] = ()
    warning_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecommendationSignals:
    """Non-contract scoring signals supplied by provider and extraction lanes."""

    reddit_mentions: int = 0
    reddit_unique_sources: int = 0
    reddit_relevance: float = 0.0
    social_mentions: int = 0
    news_mentions: int = 0
    catalyst_relevance: float = 0.0
    liquidity_score: float = 0.5


def score_strategy_cluster(
    *,
    cluster: StrategyCluster,
    analysis: AnalysisBundle,
    signals: RecommendationSignals,
    risk_profile: RiskProfile = RiskProfile.EXPLORATORY,
    account_capital: Decimal | None = None,
    max_loss_estimate: Decimal | None = None,
    uses_margin: bool = False,
) -> TradeCandidate:
    """Score one evidence-backed strategy cluster into a trade candidate."""

    position_type = _infer_position_type(cluster)
    risk_plan = assess_risk(
        instrument=cluster.instrument,
        position_type=position_type,
        risk_profile=risk_profile,
        account_capital=account_capital,
        max_loss_estimate=max_loss_estimate,
        uses_margin=uses_margin,
    )
    components = _score_components(
        cluster=cluster,
        analysis=analysis,
        signals=signals,
        risk_passed=risk_plan.passed,
    )
    penalties = _penalties(analysis, cluster)
    threshold = _threshold(risk_profile)
    overall_score = _overall_score(components, penalties)
    failed_gates = _failed_score_gates(
        overall_score=overall_score,
        threshold=threshold,
        contradiction_count=len(analysis.contradictions),
        cluster_warnings=cluster.warnings,
        analysis=analysis,
        cluster=cluster,
    )
    if not risk_plan.passed:
        action = RecommendationAction.AVOID
    elif failed_gates:
        action = RecommendationAction.WATCH
    else:
        action = RecommendationAction.QUALIFIED
    score = ScoreBreakdown(
        score_version=SCORE_VERSION,
        overall_score=overall_score,
        confidence=_confidence(cluster, analysis, components, penalties),
        threshold=threshold,
        components=components,
        penalties=penalties,
        failed_gates=failed_gates,
    )

    return TradeCandidate(
        candidate_id=_candidate_id(cluster),
        ticker=cluster.ticker,
        action=action,
        strategy_cluster_id=cluster.cluster_id,
        instrument=cluster.instrument,
        direction=cluster.direction,
        position_type=position_type,
        time_horizon=cluster.time_horizon,
        thesis=_thesis(cluster, analysis, action),
        entry_logic=_entry_logic(analysis),
        invalidation_criteria=_invalidation_criteria(analysis),
        risk_plan=risk_plan,
        catalysts=_catalysts(cluster),
        score=score,
        assumptions=analysis.assumptions,
        risks=_risks(risk_plan.failed_gates, analysis.contradictions),
        contradictions=analysis.contradictions,
        evidence=cluster.evidence,
        score_input_ids=(analysis.analysis_id, cluster.cluster_id),
        warnings=(*cluster.warnings, *analysis.warnings),
        metadata={
            "score_version": SCORE_VERSION,
            "confidence_inputs": analysis.confidence_inputs,
            "component_weights": {component.name: component.weight for component in components},
        },
    )


def select_qualified_candidates(
    candidates: tuple[TradeCandidate, ...],
) -> tuple[TradeCandidate, ...]:
    return tuple(
        candidate for candidate in candidates if candidate.action == RecommendationAction.QUALIFIED
    )


def build_no_trade_summary(candidates: tuple[TradeCandidate, ...]) -> str:
    if any(candidate.action == RecommendationAction.QUALIFIED for candidate in candidates):
        return "Qualified trade candidates were produced."
    risk_failures = sum(1 for candidate in candidates if candidate.risk_plan.failed_gates)
    score_failures = sum(1 for candidate in candidates if candidate.score.failed_gates)
    return (
        "No qualified trade candidates passed Lane D scoring. "
        f"{risk_failures} candidate(s) failed risk gates and "
        f"{score_failures} candidate(s) failed score gates."
    )


def _score_components(
    *,
    cluster: StrategyCluster,
    analysis: AnalysisBundle,
    signals: RecommendationSignals,
    risk_passed: bool,
) -> tuple[ScoreComponent, ...]:
    technical_effect = _technical_alignment_effect(analysis.technical, cluster.direction)
    return (
        _component(
            name="reddit-strength",
            normalized_score=_reddit_strength(cluster, signals),
            weight=0.20,
            rationale=(
                "Reddit discussion strength combines cluster confidence, volume, and relevance."
            ),
            evidence=cluster.evidence,
            raw_value=f"{signals.reddit_mentions} mentions",
        ),
        _component(
            name="social-news-catalyst",
            normalized_score=_social_news_strength(signals),
            weight=0.15,
            rationale=(
                "Social/news catalyst strength uses external mentions and catalyst relevance."
            ),
            evidence=cluster.evidence,
            raw_value=f"{signals.social_mentions} social / {signals.news_mentions} news",
        ),
        _component(
            name="technical-alignment",
            normalized_score=technical_effect.final_score,
            weight=_TECHNICAL_ALIGNMENT_WEIGHT,
            rationale=technical_effect.rationale,
            evidence=analysis.evidence,
            raw_value=_technical_alignment_raw_value(technical_effect),
            data_reference_ids=technical_effect.data_reference_ids,
            warning_ids=technical_effect.warning_ids,
        ),
        _component(
            name="fundamentals",
            normalized_score=_analysis_alignment(analysis.fundamental, cluster.direction),
            weight=0.14,
            rationale="Company fundamentals support or conflict with the setup.",
            evidence=analysis.evidence,
        ),
        _component(
            name="sector-context",
            normalized_score=_analysis_alignment(analysis.sector, cluster.direction),
            weight=0.10,
            rationale="Sector and peer context support or conflict with the setup.",
            evidence=analysis.evidence,
        ),
        _component(
            name="macro-context",
            normalized_score=_analysis_alignment(analysis.macro, cluster.direction),
            weight=0.10,
            rationale="Macro conditions are mapped to the strategy horizon.",
            evidence=analysis.evidence,
        ),
        _component(
            name="liquidity-risk-suitability",
            normalized_score=_liquidity_risk(signals.liquidity_score, risk_passed),
            weight=0.13,
            rationale="Liquidity and retail risk gates determine practical suitability.",
            evidence=analysis.evidence,
            raw_value=round(signals.liquidity_score, 4),
        ),
    )


def _component(
    *,
    name: str,
    normalized_score: float,
    weight: float,
    rationale: str,
    evidence: tuple[EvidenceReference, ...] = (),
    data_reference_ids: tuple[str, ...] = (),
    warning_ids: tuple[str, ...] = (),
    raw_value: float | int | str | None = None,
) -> ScoreComponent:
    bounded_score = round(clamp(normalized_score), 4)
    return ScoreComponent(
        name=name,
        raw_value=raw_value,
        normalized_score=bounded_score,
        weight=weight,
        contribution=round(bounded_score * weight, 6),
        rationale=rationale,
        evidence=evidence,
        data_reference_ids=data_reference_ids,
        warning_ids=warning_ids,
    )


def _penalties(
    analysis: AnalysisBundle,
    cluster: StrategyCluster,
) -> tuple[ScoreComponent, ...]:
    penalties: list[ScoreComponent] = []
    if analysis.contradictions:
        penalty_score = clamp(len(analysis.contradictions) / 3.0)
        contribution = round(-(penalty_score * 0.18), 6)
        penalties.append(
            ScoreComponent(
                name="contradiction-penalty",
                raw_value=len(analysis.contradictions),
                normalized_score=round(penalty_score, 4),
                weight=0.18,
                contribution=contribution,
                rationale="Contradictory analysis inputs reduce recommendation confidence.",
                evidence=analysis.evidence,
            )
        )

    sarcasm_warnings = _warnings_by_risk_type(cluster.warnings, "high_sarcasm_joke_risk")
    if sarcasm_warnings:
        risk = max(_warning_numeric_metadata(warning, "risk") for warning in sarcasm_warnings)
        normalized_risk = round(clamp(risk), 4)
        penalties.append(
            ScoreComponent(
                name="sarcasm-joke-risk-penalty",
                raw_value=normalized_risk,
                normalized_score=normalized_risk,
                weight=0.12,
                contribution=round(-(normalized_risk * 0.12), 6),
                rationale=("High sarcasm or joke risk in the cited discussion reduces confidence."),
                evidence=cluster.evidence,
                warning_ids=_warning_ids(sarcasm_warnings),
            )
        )

    source_conflict_warnings = _warnings_by_risk_type(
        cluster.warnings,
        "conflicting_source_evidence",
    )
    if source_conflict_warnings:
        penalties.append(
            ScoreComponent(
                name="source-conflict-penalty",
                raw_value="opposing source evidence",
                normalized_score=1.0,
                weight=0.18,
                contribution=-0.18,
                rationale=(
                    "Opposing source evidence for the same setup keeps the strategy watch-only."
                ),
                evidence=cluster.evidence,
                warning_ids=_warning_ids(source_conflict_warnings),
            )
        )

    ml_signal = analysis.technical.ml_signal if analysis.technical is not None else None
    if ml_signal is not None:
        penalties.extend(_ml_signal_penalties(ml_signal, cluster.direction, analysis.technical))

    return tuple(penalties)


def _ml_signal_penalties(
    ml_signal: TechnicalMlSignal,
    direction: Direction,
    technical: AnalysisComponent | None = None,
) -> tuple[ScoreComponent, ...]:
    if _is_timesfm_signal(ml_signal):
        return _timesfm_signal_penalties(ml_signal, direction, technical)

    penalties: list[ScoreComponent] = []
    if ml_signal.status != "usable":
        penalties.append(
            ScoreComponent(
                name="ml-signal-quality-penalty",
                raw_value=ml_signal.status,
                normalized_score=0.75,
                weight=0.10,
                contribution=-0.075,
                rationale=(
                    "The local ML technical sidecar is present but not actionable enough "
                    "to support a trade."
                ),
                warning_ids=ml_signal.warning_ids,
            )
        )
    if _ml_signal_conflicts_direction(ml_signal, direction):
        penalties.append(
            ScoreComponent(
                name="ml-technical-conflict-penalty",
                raw_value=ml_signal.probability_positive,
                normalized_score=1.0,
                weight=0.12,
                contribution=-0.12,
                rationale=(
                    "The ML technical sidecar conflicts with the observed strategy direction."
                ),
                warning_ids=ml_signal.warning_ids,
            )
        )
    elif ml_signal.signal == AnalysisSignal.MIXED or ml_signal.calibrated_confidence < 0.15:
        penalties.append(
            ScoreComponent(
                name="ml-signal-weak-penalty",
                raw_value=ml_signal.calibrated_confidence,
                normalized_score=0.40,
                weight=0.06,
                contribution=-0.024,
                rationale="The ML technical sidecar is too weak to increase confidence.",
                warning_ids=ml_signal.warning_ids,
            )
        )
    return tuple(penalties)


def _timesfm_signal_penalties(
    ml_signal: TechnicalMlSignal,
    direction: Direction,
    technical: AnalysisComponent | None,
) -> tuple[ScoreComponent, ...]:
    penalties: list[ScoreComponent] = []
    warning_ids = ml_signal.warning_ids
    if ml_signal.status == "unavailable":
        penalties.append(
            ScoreComponent(
                name="timesfm-signal-unavailable-penalty",
                raw_value=ml_signal.status,
                normalized_score=1.0,
                weight=0.10,
                contribution=-0.10,
                rationale="TimesFM output is unavailable and cannot support the technical score.",
                warning_ids=warning_ids,
            )
        )
    elif ml_signal.status != "usable":
        penalties.append(
            ScoreComponent(
                name="timesfm-signal-quality-penalty",
                raw_value=ml_signal.status,
                normalized_score=0.75,
                weight=0.10,
                contribution=-0.075,
                rationale=(
                    "TimesFM output is present but failed freshness or evaluation guardrails."
                ),
                warning_ids=warning_ids,
            )
        )
    if _timesfm_signal_is_stale(ml_signal):
        penalties.append(
            ScoreComponent(
                name="timesfm-stale-signal-penalty",
                raw_value=ml_signal.freshness_status.value,
                normalized_score=0.70,
                weight=0.08,
                contribution=-0.056,
                rationale="TimesFM evaluation data is stale for recommendation scoring.",
                warning_ids=warning_ids,
            )
        )
    if _timesfm_signal_has_poor_evaluation(ml_signal):
        penalties.append(
            ScoreComponent(
                name="timesfm-evaluation-quality-penalty",
                raw_value=ml_signal.validation_accuracy,
                normalized_score=0.60,
                weight=0.08,
                contribution=-0.048,
                rationale="TimesFM evaluation quality is below the scoring support threshold.",
                warning_ids=warning_ids,
            )
        )
    if _timesfm_signal_has_wide_interval(ml_signal):
        penalties.append(
            ScoreComponent(
                name="timesfm-wide-interval-penalty",
                raw_value=ml_signal.forecast_interval_width,
                normalized_score=0.50,
                weight=0.07,
                contribution=-0.035,
                rationale="TimesFM forecast interval is too wide to add confidence.",
                warning_ids=warning_ids,
            )
        )
    if _timesfm_signal_conflicts(ml_signal, direction, technical):
        penalties.append(
            ScoreComponent(
                name="timesfm-technical-conflict-penalty",
                raw_value=ml_signal.probability_positive,
                normalized_score=1.0,
                weight=0.12,
                contribution=-0.12,
                rationale=(
                    "TimesFM output conflicts with the observed strategy direction or "
                    "deterministic technical analysis."
                ),
                warning_ids=warning_ids,
            )
        )
    elif ml_signal.signal == AnalysisSignal.MIXED or (
        ml_signal.status == "usable" and ml_signal.calibrated_confidence < _TIMESFM_MIN_CONFIDENCE
    ):
        penalties.append(
            ScoreComponent(
                name="timesfm-signal-weak-penalty",
                raw_value=ml_signal.calibrated_confidence,
                normalized_score=0.40,
                weight=0.06,
                contribution=-0.024,
                rationale="TimesFM output is too weak to increase technical confidence.",
                warning_ids=warning_ids,
            )
        )
    return tuple(penalties)


def _technical_alignment_effect(
    component: AnalysisComponent | None,
    direction: Direction,
) -> _TechnicalMlScoringEffect:
    base_score = round(_analysis_alignment(component, direction), 4)
    if component is None:
        return _TechnicalMlScoringEffect(
            base_score=base_score,
            final_score=base_score,
            adjustment=0.0,
            rationale="Technical analysis alignment with the discussed direction.",
        )
    ml_signal = getattr(component, "ml_signal", None)
    if not isinstance(ml_signal, TechnicalMlSignal) or not _is_timesfm_signal(ml_signal):
        return _TechnicalMlScoringEffect(
            base_score=base_score,
            final_score=base_score,
            adjustment=0.0,
            rationale="Technical analysis alignment with the discussed direction.",
        )
    data_reference_ids = tuple(
        reference_id
        for reference_id in (
            ml_signal.source_artifact_id,
            ml_signal.source_artifact_sha256,
        )
        if reference_id
    )
    if not _timesfm_signal_can_boost(ml_signal, direction, component):
        return _TechnicalMlScoringEffect(
            base_score=base_score,
            final_score=base_score,
            adjustment=0.0,
            rationale=(
                "Technical analysis alignment with the discussed direction. TimesFM sidecar "
                "made no positive adjustment after freshness, evaluation, uncertainty, and "
                "conflict guardrails."
            ),
            data_reference_ids=data_reference_ids,
            warning_ids=ml_signal.warning_ids,
        )
    requested_adjustment = min(
        _TIMESFM_MAX_TECHNICAL_BOOST,
        ml_signal.calibrated_confidence * 0.10,
    )
    final_score = round(clamp(base_score + requested_adjustment), 4)
    adjustment = round(final_score - base_score, 4)
    return _TechnicalMlScoringEffect(
        base_score=base_score,
        final_score=final_score,
        adjustment=adjustment,
        rationale=(
            "Technical analysis alignment with the discussed direction. TimesFM sidecar "
            f"adjusted only this technical component by {adjustment:+.4f} after guardrails."
        ),
        data_reference_ids=data_reference_ids,
        warning_ids=ml_signal.warning_ids,
    )


def _technical_alignment_raw_value(effect: _TechnicalMlScoringEffect) -> str | None:
    if effect.adjustment == 0.0 and not effect.data_reference_ids and not effect.warning_ids:
        return None
    return (
        f"deterministic_score={effect.base_score:.4f}; "
        f"timesfm_adjustment={effect.adjustment:+.4f}; "
        f"final_score={effect.final_score:.4f}"
    )


def _is_timesfm_signal(ml_signal: TechnicalMlSignal) -> bool:
    return ml_signal.metadata.get("model_kind") == _TIMESFM_MODEL_KIND


def _timesfm_signal_can_boost(
    ml_signal: TechnicalMlSignal,
    direction: Direction,
    technical: AnalysisComponent | None,
) -> bool:
    return (
        ml_signal.status == "usable"
        and not _timesfm_signal_is_stale(ml_signal)
        and not _timesfm_signal_has_poor_evaluation(ml_signal)
        and not _timesfm_signal_has_wide_interval(ml_signal)
        and not _timesfm_signal_conflicts(ml_signal, direction, technical)
        and ml_signal.signal != AnalysisSignal.MIXED
        and ml_signal.calibrated_confidence >= _TIMESFM_MIN_CONFIDENCE
    )


def _timesfm_signal_is_stale(ml_signal: TechnicalMlSignal) -> bool:
    return ml_signal.status == "stale" or ml_signal.freshness_status == FreshnessStatus.STALE


def _timesfm_signal_has_poor_evaluation(ml_signal: TechnicalMlSignal) -> bool:
    if ml_signal.status == "weak":
        return True
    if ml_signal.validation_accuracy is None:
        return ml_signal.status == "usable"
    return ml_signal.validation_accuracy < _TIMESFM_MIN_VALIDATION_ACCURACY


def _timesfm_signal_has_wide_interval(ml_signal: TechnicalMlSignal) -> bool:
    width = ml_signal.forecast_interval_width
    return width is not None and width > _TIMESFM_MAX_INTERVAL_WIDTH


def _timesfm_signal_conflicts(
    ml_signal: TechnicalMlSignal,
    direction: Direction,
    technical: AnalysisComponent | None,
) -> bool:
    if _ml_signal_conflicts_direction(ml_signal, direction):
        return True
    if technical is None or technical.signal in {AnalysisSignal.MIXED, AnalysisSignal.NEUTRAL}:
        return False
    return technical.signal != AnalysisSignal.UNKNOWN and ml_signal.signal not in {
        AnalysisSignal.UNKNOWN,
        AnalysisSignal.MIXED,
        technical.signal,
    }


def _reddit_strength(cluster: StrategyCluster, signals: RecommendationSignals) -> float:
    mention_score = min(signals.reddit_mentions, 8) / 8.0
    source_score = min(signals.reddit_unique_sources, 5) / 5.0
    relevance = clamp(signals.reddit_relevance)
    return cluster.confidence * 0.25 + mention_score * 0.30 + source_score * 0.20 + relevance * 0.25


def _social_news_strength(signals: RecommendationSignals) -> float:
    mention_score = min(signals.social_mentions + signals.news_mentions, 5) / 5.0
    catalyst_relevance = clamp(signals.catalyst_relevance)
    return mention_score * 0.45 + catalyst_relevance * 0.55


def _analysis_alignment(
    component: AnalysisComponent | None,
    direction: Direction,
) -> float:
    if component is None:
        return 0.35
    base_score = _signal_score(component.signal, direction)
    confidence_adjustment = 0.5 + component.confidence / 2.0
    return base_score * confidence_adjustment


def _signal_score(signal: AnalysisSignal, direction: Direction) -> float:
    if signal == AnalysisSignal.SUPPORTS:
        return 0.90 if direction != Direction.BEARISH else 0.35
    if signal == AnalysisSignal.CONFLICTS:
        return 0.15 if direction != Direction.BEARISH else 0.75
    if signal == AnalysisSignal.MIXED:
        return 0.55
    if signal == AnalysisSignal.NEUTRAL:
        return 0.50
    return 0.35


def _liquidity_risk(liquidity_score: float, risk_passed: bool) -> float:
    risk_score = 1.0 if risk_passed else 0.0
    return clamp(liquidity_score) * 0.55 + risk_score * 0.45


def _overall_score(
    components: tuple[ScoreComponent, ...],
    penalties: tuple[ScoreComponent, ...],
) -> float:
    contribution = sum(component.contribution for component in components) + sum(
        penalty.contribution for penalty in penalties
    )
    return round(clamp(contribution), 4)


def _failed_score_gates(
    *,
    overall_score: float,
    threshold: float,
    contradiction_count: int,
    cluster_warnings: tuple[ProviderWarning, ...],
    analysis: AnalysisBundle,
    cluster: StrategyCluster,
) -> tuple[str, ...]:
    failed_gates: list[str] = []
    if contradiction_count >= 2:
        failed_gates.append("conflicting-analysis")
    warning_risk_types = _warning_risk_types(cluster_warnings)
    if "conflicting_source_evidence" in warning_risk_types:
        failed_gates.append("conflicting-source-evidence")
    if "high_sarcasm_joke_risk" in warning_risk_types:
        failed_gates.append("high-sarcasm-joke-risk")
    ml_signal = analysis.technical.ml_signal if analysis.technical is not None else None
    if ml_signal is not None:
        if _is_timesfm_signal(ml_signal):
            failed_gates.extend(_timesfm_failed_gates(ml_signal, cluster.direction, analysis))
        else:
            if ml_signal.status != "usable":
                failed_gates.append("ml-signal-not-actionable")
            if _ml_signal_conflicts_direction(ml_signal, cluster.direction):
                failed_gates.append("ml-technical-conflict")
    timesfm_effect = _technical_alignment_effect(analysis.technical, cluster.direction)
    if timesfm_effect.adjustment > 0:
        score_without_timesfm = round(
            clamp(overall_score - (timesfm_effect.adjustment * _TECHNICAL_ALIGNMENT_WEIGHT)),
            4,
        )
        if score_without_timesfm < threshold <= overall_score:
            failed_gates.append("timesfm-cannot-qualify-standalone")
    if overall_score < threshold:
        failed_gates.append("score-below-threshold")
    return tuple(failed_gates)


def _timesfm_failed_gates(
    ml_signal: TechnicalMlSignal,
    direction: Direction,
    analysis: AnalysisBundle,
) -> tuple[str, ...]:
    technical = analysis.technical
    failed_gates: list[str] = []
    if ml_signal.status != "usable":
        failed_gates.append("timesfm-signal-not-actionable")
    if ml_signal.status == "unavailable":
        failed_gates.append("timesfm-signal-unavailable")
    if _timesfm_signal_is_stale(ml_signal):
        failed_gates.append("timesfm-stale-signal")
    if _timesfm_signal_has_poor_evaluation(ml_signal):
        failed_gates.append("timesfm-evaluation-underqualified")
    if _timesfm_signal_has_wide_interval(ml_signal):
        failed_gates.append("timesfm-wide-interval")
    if _timesfm_signal_conflicts(ml_signal, direction, technical):
        failed_gates.append("timesfm-technical-conflict")
    elif (
        ml_signal.signal == AnalysisSignal.MIXED
        or ml_signal.calibrated_confidence < _TIMESFM_MIN_CONFIDENCE
    ):
        failed_gates.append("timesfm-signal-weak")
    return tuple(dict.fromkeys(failed_gates))


def _ml_signal_conflicts_direction(
    ml_signal: TechnicalMlSignal,
    direction: Direction,
) -> bool:
    if direction == Direction.BEARISH:
        return ml_signal.signal == AnalysisSignal.SUPPORTS
    if direction == Direction.BULLISH:
        return ml_signal.signal == AnalysisSignal.CONFLICTS
    return False


def _warning_risk_types(warnings: tuple[ProviderWarning, ...]) -> set[str]:
    return {
        risk_type
        for warning in warnings
        if isinstance((risk_type := warning.metadata.get("risk_type")), str)
    }


def _warnings_by_risk_type(
    warnings: tuple[ProviderWarning, ...],
    risk_type: str,
) -> tuple[ProviderWarning, ...]:
    return tuple(warning for warning in warnings if warning.metadata.get("risk_type") == risk_type)


def _warning_numeric_metadata(warning: ProviderWarning, key: str) -> float:
    value = warning.metadata.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 1.0


def _warning_ids(warnings: tuple[ProviderWarning, ...]) -> tuple[str, ...]:
    warning_ids = (
        f"{warning.provider_name or 'unknown-provider'}:{warning.code.value}"
        for warning in warnings
    )
    return tuple(dict.fromkeys(warning_ids))


def _threshold(risk_profile: RiskProfile) -> float:
    if risk_profile == RiskProfile.CONSERVATIVE:
        return 0.72
    if risk_profile == RiskProfile.BALANCED:
        return 0.66
    return 0.60


def _confidence(
    cluster: StrategyCluster,
    analysis: AnalysisBundle,
    components: tuple[ScoreComponent, ...],
    penalties: tuple[ScoreComponent, ...],
) -> float:
    component_score = sum(component.normalized_score for component in components) / len(components)
    analysis_confidence = _analysis_confidence(analysis)
    penalty_drag = sum(penalty.normalized_score for penalty in penalties) * 0.10
    confidence = (
        cluster.confidence * 0.35
        + component_score * 0.40
        + analysis_confidence * 0.25
        - penalty_drag
    )
    return round(clamp(confidence), 4)


def _analysis_confidence(analysis: AnalysisBundle) -> float:
    confidences = [
        component.confidence
        for component in (analysis.technical, analysis.fundamental, analysis.sector, analysis.macro)
        if component is not None
    ]
    if not confidences:
        return 0.35
    return sum(confidences) / len(confidences)


def _infer_position_type(cluster: StrategyCluster) -> PositionType:
    if cluster.instrument == InstrumentType.SHARES:
        if cluster.direction == Direction.BEARISH:
            return PositionType.WATCH_ONLY
        return PositionType.LONG
    if cluster.instrument in {InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION}:
        return PositionType.LONG
    if cluster.instrument == InstrumentType.OPTION_SPREAD:
        return PositionType.DEFINED_RISK
    if cluster.instrument in {InstrumentType.CASH_SECURED_PUT, InstrumentType.COVERED_CALL}:
        return PositionType.INCOME
    return PositionType.WATCH_ONLY


def _candidate_id(cluster: StrategyCluster) -> str:
    return f"candidate-{cluster.cluster_id.removeprefix('cluster-')}"


def _thesis(
    cluster: StrategyCluster,
    analysis: AnalysisBundle,
    action: RecommendationAction,
) -> str:
    if action == RecommendationAction.QUALIFIED:
        prefix = "Observed discussion and Lane D analysis support a qualified defined-risk idea."
    elif action == RecommendationAction.WATCH:
        prefix = "Observed discussion exists, but the setup remains watch-only after scoring."
    else:
        prefix = "Observed discussion exists, but risk or score gates argue against trading it."
    return f"{prefix} Cluster {cluster.cluster_id} maps to analysis {analysis.analysis_id}."


def _entry_logic(analysis: AnalysisBundle) -> str:
    if analysis.technical and analysis.technical.resistance_levels:
        level = analysis.technical.resistance_levels[0]
        return f"Wait for price confirmation near or above resistance around {level}."
    return "Wait for fresh confirmation from price, volume, and catalyst evidence."


def _invalidation_criteria(analysis: AnalysisBundle) -> str:
    if analysis.technical and analysis.technical.support_levels:
        level = analysis.technical.support_levels[0]
        return f"Invalidate if price loses support around {level} or catalyst evidence reverses."
    return "Invalidate if the evidence, price trend, or macro context reverses."


def _catalysts(cluster: StrategyCluster) -> tuple[str, ...]:
    return (cluster.catalyst_summary,) if cluster.catalyst_summary else ()


def _risks(
    failed_risk_gates: tuple[str, ...],
    contradictions: tuple[str, ...],
) -> tuple[str, ...]:
    risks: list[str] = []
    if failed_risk_gates:
        risks.append("Risk gates failed: " + ", ".join(failed_risk_gates) + ".")
    risks.extend(contradictions)
    if not risks:
        risks.append("Speculative retail ideas can still lose the full defined risk amount.")
    return tuple(risks)
