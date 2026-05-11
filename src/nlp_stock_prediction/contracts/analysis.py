"""Analysis context contracts."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import Field

from nlp_stock_prediction.contracts.base import (
    Confidence,
    ContractModel,
    JsonObject,
    NonEmptyStr,
    TickerSymbol,
)
from nlp_stock_prediction.contracts.enums import AnalysisSignal, TimeHorizon
from nlp_stock_prediction.contracts.provenance import (
    EvidenceReference,
    ProviderWarning,
    SourceProvenance,
)


class MetricValue(ContractModel):
    """A numeric or textual metric with provenance."""

    name: NonEmptyStr
    value: Decimal | float | int | str | None
    unit: str | None = None
    as_of: date | datetime | None = None
    provenance: SourceProvenance | None = None
    metadata: JsonObject = Field(default_factory=dict)


class AnalysisComponent(ContractModel):
    """Shared component used by technical, fundamental, sector, and macro analysis."""

    summary: NonEmptyStr
    signal: AnalysisSignal = AnalysisSignal.UNKNOWN
    confidence: Confidence = 0.0
    metrics: tuple[MetricValue, ...] = Field(default_factory=tuple)
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)
    assumptions: tuple[str, ...] = Field(default_factory=tuple)


class TechnicalAnalysis(AnalysisComponent):
    ticker: TickerSymbol
    trend: str | None = None
    support_levels: tuple[Decimal, ...] = Field(default_factory=tuple)
    resistance_levels: tuple[Decimal, ...] = Field(default_factory=tuple)
    volume_summary: str | None = None
    volatility_summary: str | None = None
    gap_summary: str | None = None
    candlestick_summary: str | None = None


class FundamentalAnalysis(AnalysisComponent):
    ticker: TickerSymbol
    valuation_summary: str | None = None
    profitability_summary: str | None = None
    growth_summary: str | None = None
    balance_sheet_risk: str | None = None
    earnings_timing: str | None = None
    notable_filings: tuple[str, ...] = Field(default_factory=tuple)


class SectorContext(AnalysisComponent):
    ticker: TickerSymbol
    sector: str | None = None
    peers: tuple[TickerSymbol, ...] = Field(default_factory=tuple)
    benchmark_symbol: TickerSymbol | None = None


class MacroContext(AnalysisComponent):
    as_of: date | datetime
    horizon: TimeHorizon = TimeHorizon.UNKNOWN
    supportive_factors: tuple[str, ...] = Field(default_factory=tuple)
    conflicting_factors: tuple[str, ...] = Field(default_factory=tuple)


class AnalysisBundle(ContractModel):
    """Combined analysis for one ticker or strategy cluster."""

    analysis_id: NonEmptyStr
    ticker: TickerSymbol
    as_of: date | datetime
    strategy_cluster_ids: tuple[str, ...] = Field(default_factory=tuple)
    technical: TechnicalAnalysis | None = None
    fundamental: FundamentalAnalysis | None = None
    sector: SectorContext | None = None
    macro: MacroContext | None = None
    signals: tuple[AnalysisSignal, ...] = Field(default_factory=tuple)
    contradictions: tuple[str, ...] = Field(default_factory=tuple)
    assumptions: tuple[str, ...] = Field(default_factory=tuple)
    confidence_inputs: JsonObject = Field(default_factory=dict)
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)


__all__ = [
    "AnalysisBundle",
    "AnalysisComponent",
    "FundamentalAnalysis",
    "MacroContext",
    "MetricValue",
    "SectorContext",
    "TechnicalAnalysis",
]
