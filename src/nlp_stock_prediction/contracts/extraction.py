"""LLM strategy extraction and clustering contracts."""

from __future__ import annotations

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import Confidence, ContractModel, NonEmptyStr, TickerSymbol
from nlp_stock_prediction.contracts.enums import (
    Direction,
    InstrumentType,
    PositionType,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.provenance import EvidenceReference, ProviderWarning


class StrategyExtraction(ContractModel):
    """One evidence-backed strategy observed in discussion."""

    strategy_id: NonEmptyStr
    ticker: TickerSymbol
    label: NonEmptyStr
    direction: Direction
    instrument: InstrumentType
    position_type: PositionType
    time_horizon: TimeHorizon
    catalyst: str | None = None
    risk_or_hedge: str | None = None
    slang_terms: tuple[str, ...] = Field(default_factory=tuple)
    evidence: tuple[EvidenceReference, ...]
    confidence: Confidence
    sarcasm_joke_risk: Confidence = 0.0
    warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def require_evidence(self) -> StrategyExtraction:
        if not self.evidence:
            raise ValueError("strategy extractions must cite at least one evidence record")
        return self


class StrategyCluster(ContractModel):
    """Near-duplicate strategy group."""

    cluster_id: NonEmptyStr
    ticker: TickerSymbol
    direction: Direction
    instrument: InstrumentType
    time_horizon: TimeHorizon
    catalyst_summary: str | None = None
    member_strategy_ids: tuple[NonEmptyStr, ...]
    evidence: tuple[EvidenceReference, ...]
    confidence: Confidence
    warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def require_members_and_evidence(self) -> StrategyCluster:
        if not self.member_strategy_ids:
            raise ValueError("strategy clusters must include member_strategy_ids")
        if not self.evidence:
            raise ValueError("strategy clusters must cite evidence")
        return self


__all__ = ["StrategyCluster", "StrategyExtraction"]
