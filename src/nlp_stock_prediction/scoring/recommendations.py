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
    InstrumentType,
    PositionType,
    RecommendationAction,
    RiskProfile,
    ScoreBreakdown,
    ScoreComponent,
    StrategyCluster,
    TradeCandidate,
)
from nlp_stock_prediction.scoring.risk import assess_risk

SCORE_VERSION = "lane-d-score-v1"


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
    disclaimer_id: str,
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
    penalties = _penalties(analysis)
    threshold = _threshold(risk_profile)
    overall_score = _overall_score(components, penalties)
    failed_gates = _failed_score_gates(
        overall_score=overall_score,
        threshold=threshold,
        contradiction_count=len(analysis.contradictions),
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
        warnings=analysis.warnings,
        disclaimer_id=disclaimer_id,
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
            normalized_score=_analysis_alignment(analysis.technical, cluster.direction),
            weight=0.18,
            rationale="Technical analysis alignment with the discussed direction.",
            evidence=analysis.evidence,
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
    )


def _penalties(analysis: AnalysisBundle) -> tuple[ScoreComponent, ...]:
    if not analysis.contradictions:
        return ()
    penalty_score = clamp(len(analysis.contradictions) / 3.0)
    contribution = round(-(penalty_score * 0.18), 6)
    return (
        ScoreComponent(
            name="contradiction-penalty",
            raw_value=len(analysis.contradictions),
            normalized_score=round(penalty_score, 4),
            weight=0.18,
            contribution=contribution,
            rationale="Contradictory analysis inputs reduce recommendation confidence.",
            evidence=analysis.evidence,
        ),
    )


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
) -> tuple[str, ...]:
    failed_gates: list[str] = []
    if contradiction_count >= 2:
        failed_gates.append("conflicting-analysis")
    if overall_score < threshold:
        failed_gates.append("score-below-threshold")
    return tuple(failed_gates)


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
