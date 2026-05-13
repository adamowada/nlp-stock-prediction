"""Recommendation scoring and risk contracts."""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import (
    Confidence,
    ContractModel,
    JsonObject,
    NonEmptyStr,
    PositiveDecimal,
    Score,
    TickerSymbol,
)
from nlp_stock_prediction.contracts.enums import (
    Direction,
    InstrumentType,
    PositionType,
    RecommendationAction,
    RiskProfile,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.provenance import EvidenceReference, ProviderWarning


class ScoreComponent(ContractModel):
    """One weighted score input used to explain a candidate."""

    name: NonEmptyStr
    raw_value: float | Decimal | str | None = None
    normalized_score: Score
    weight: float = Field(ge=0.0)
    contribution: float
    rationale: NonEmptyStr
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    data_reference_ids: tuple[str, ...] = Field(default_factory=tuple)
    warning_ids: tuple[str, ...] = Field(default_factory=tuple)


class ScoreBreakdown(ContractModel):
    """Auditable confidence and scoring inputs for a trade candidate."""

    score_version: NonEmptyStr
    overall_score: Score
    confidence: Confidence
    threshold: Score
    components: tuple[ScoreComponent, ...]
    penalties: tuple[ScoreComponent, ...] = Field(default_factory=tuple)
    failed_gates: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def require_components(self) -> ScoreBreakdown:
        if not self.components:
            raise ValueError("score breakdowns must include at least one component")
        return self


class RiskAssessment(ContractModel):
    """Risk and sizing constraints independent of recommendation score."""

    risk_profile: RiskProfile
    defined_risk: bool
    margin_required: bool = False
    max_account_risk_pct: Decimal = Field(default=Decimal("0.01"), ge=Decimal("0"), le=Decimal("1"))
    account_capital: PositiveDecimal | None = None
    max_loss_estimate: PositiveDecimal | None = None
    position_size_pct: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("1"))
    passed: bool
    failed_gates: tuple[str, ...] = Field(default_factory=tuple)
    sizing_basis: str | None = None


class TradeCandidate(ContractModel):
    """App-generated opportunity candidate grounded in evidence and risk policy."""

    candidate_id: NonEmptyStr
    ticker: TickerSymbol
    action: RecommendationAction
    strategy_cluster_id: str | None = None
    instrument: InstrumentType
    direction: Direction
    position_type: PositionType
    time_horizon: TimeHorizon
    thesis: NonEmptyStr
    entry_logic: NonEmptyStr
    invalidation_criteria: NonEmptyStr
    risk_plan: RiskAssessment
    catalysts: tuple[str, ...] = Field(default_factory=tuple)
    score: ScoreBreakdown
    assumptions: tuple[str, ...] = Field(default_factory=tuple)
    risks: tuple[str, ...] = Field(default_factory=tuple)
    contradictions: tuple[str, ...] = Field(default_factory=tuple)
    evidence: tuple[EvidenceReference, ...]
    score_input_ids: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_evidence_for_actionable_candidates(self) -> TradeCandidate:
        if self.action != RecommendationAction.NO_TRADE and not self.evidence:
            raise ValueError("actionable trade candidates must cite evidence")
        if self.action == RecommendationAction.QUALIFIED:
            if not self.risk_plan.passed:
                raise ValueError("qualified trade candidates must pass risk gates")
            if self.risk_plan.failed_gates:
                raise ValueError("qualified trade candidates must not include failed risk gates")
            if self.score.failed_gates:
                raise ValueError("qualified trade candidates must not include failed score gates")
            if self.score.overall_score < self.score.threshold:
                raise ValueError("qualified trade candidates must meet score threshold")
        return self


__all__ = ["RiskAssessment", "ScoreBreakdown", "ScoreComponent", "TradeCandidate"]
