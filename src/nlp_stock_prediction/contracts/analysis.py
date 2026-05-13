"""Analysis context contracts."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from nlp_stock_prediction.contracts.base import (
    Confidence,
    ContractModel,
    JsonObject,
    NonEmptyStr,
    TickerSymbol,
)
from nlp_stock_prediction.contracts.enums import AnalysisSignal, FreshnessStatus, TimeHorizon
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.provenance import (
    EvidenceReference,
    ProviderWarning,
    SourceProvenance,
)
from nlp_stock_prediction.contracts.providers import ProviderRequest


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


class TechnicalMlSignal(ContractModel):
    """Conservative sidecar for local ML-assisted technical-analysis output."""

    model_config = ConfigDict(extra="ignore")

    model_hash: NonEmptyStr
    dataset_hash: NonEmptyStr
    as_of: date | datetime
    feature_end: date | datetime
    prediction_horizon_sessions: int = Field(ge=1)
    probability_positive: Confidence
    calibrated_confidence: Confidence
    signal: AnalysisSignal
    status: Literal["usable", "weak", "stale", "conflicting", "unavailable"] = "usable"
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    expected_return: float | None = None
    forecast_interval_width: float | None = Field(default=None, ge=0.0)
    source_artifact_id: str | None = None
    source_artifact_sha256: str | None = None
    validation_accuracy: Confidence | None = None
    validation_brier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    warning_ids: tuple[str, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)


class FundamentalAgentSignal(ContractModel):
    """Report-facing summary of a validated fundamental NLP agent result."""

    request_id: NonEmptyStr
    provider_name: NonEmptyStr
    runner_name: NonEmptyStr
    raw_response_id: str | None = None
    signal: AnalysisSignal
    confidence: Confidence
    summary: NonEmptyStr
    source_evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    claim_count: int = Field(ge=0)
    risk_count: int = Field(ge=0)
    contradictions: tuple[str, ...] = Field(default_factory=tuple)
    warning_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence_inputs: JsonObject = Field(default_factory=dict)


class TechnicalAnalysis(AnalysisComponent):
    ticker: TickerSymbol
    trend: str | None = None
    support_levels: tuple[Decimal, ...] = Field(default_factory=tuple)
    resistance_levels: tuple[Decimal, ...] = Field(default_factory=tuple)
    volume_summary: str | None = None
    volatility_summary: str | None = None
    gap_summary: str | None = None
    candlestick_summary: str | None = None
    ml_signal: TechnicalMlSignal | None = None


class FundamentalAnalysis(AnalysisComponent):
    ticker: TickerSymbol
    valuation_summary: str | None = None
    profitability_summary: str | None = None
    growth_summary: str | None = None
    balance_sheet_risk: str | None = None
    earnings_timing: str | None = None
    notable_filings: tuple[str, ...] = Field(default_factory=tuple)
    agent_signal: FundamentalAgentSignal | None = None


class FundamentalNlpAnalysisRequest(ProviderRequest):
    """Evidence packet sent to the optional fundamental-analysis agent lane."""

    request_id: NonEmptyStr
    ticker: TickerSymbol
    run_date: date
    as_of: date | datetime
    evidence: tuple[SourceEvidence, ...] = Field(default_factory=tuple)
    metrics: tuple[MetricValue, ...] = Field(default_factory=tuple)
    prompt_version: NonEmptyStr
    schema_version: NonEmptyStr
    instructions: str | None = None
    focus_areas: tuple[str, ...] = Field(default_factory=tuple)
    options: JsonObject = Field(default_factory=dict)

    @property
    def source_evidence_ids(self) -> tuple[str, ...]:
        """Return the normalized evidence IDs available for agent citations."""

        return tuple(record.evidence_id for record in self.evidence)


class FundamentalNlpCitation(ContractModel):
    """A quote-level citation from an agent claim back to normalized source evidence."""

    evidence_id: NonEmptyStr
    quote: NonEmptyStr
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_span_order(self) -> FundamentalNlpCitation:
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char must be greater than or equal to start_char")
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("citation offsets must include both start_char and end_char")
        return self

    def as_evidence_reference(self) -> EvidenceReference:
        """Return the shared evidence-reference shape used by report contracts."""

        return EvidenceReference(
            evidence_id=self.evidence_id,
            quote=self.quote,
            start_char=self.start_char,
            end_char=self.end_char,
            relevance=self.relevance,
        )


class FundamentalNlpClaim(ContractModel):
    """A sourced fundamental claim with an explicit observed-vs-interpreted label."""

    claim_id: NonEmptyStr
    claim_type: Literal["observed", "interpretation", "risk", "assumption"]
    text: NonEmptyStr
    citations: tuple[FundamentalNlpCitation, ...] = Field(default_factory=tuple)
    confidence: Confidence = 0.0
    metadata: JsonObject = Field(default_factory=dict)


class FundamentalNlpRisk(ContractModel):
    """A cited risk identified by the agent for later report and audit lanes."""

    risk_id: NonEmptyStr
    text: NonEmptyStr
    severity: Literal["low", "medium", "high", "unknown"] = "unknown"
    citations: tuple[FundamentalNlpCitation, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)


class FundamentalAgentAuditMetadata(ContractModel):
    """Agent-run metadata that can be copied into future audit artifacts."""

    provider_name: NonEmptyStr
    runner_name: NonEmptyStr
    prompt_version: NonEmptyStr
    schema_version: NonEmptyStr
    prompt_sha256: str | None = None
    response_sha256: str | None = None
    raw_response_id: str | None = None
    audit_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)


class FundamentalNlpAnalysisResponse(AnalysisComponent):
    """Structured output from the optional Codex-agent fundamental lane."""

    request_id: NonEmptyStr
    ticker: TickerSymbol
    as_of: date | datetime
    source_evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    citations: tuple[FundamentalNlpCitation, ...] = Field(default_factory=tuple)
    claims: tuple[FundamentalNlpClaim, ...] = Field(default_factory=tuple)
    risks: tuple[FundamentalNlpRisk, ...] = Field(default_factory=tuple)
    contradictions: tuple[str, ...] = Field(default_factory=tuple)
    confidence_inputs: JsonObject = Field(default_factory=dict)
    validation_warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)
    audit: FundamentalAgentAuditMetadata

    @model_validator(mode="after")
    def validate_citations_reference_declared_sources(self) -> FundamentalNlpAnalysisResponse:
        source_ids = set(self.source_evidence_ids)
        if not source_ids:
            if self.citations or self.claims or self.risks:
                raise ValueError("agent responses with claims must declare source_evidence_ids")
            return self

        citation_ids = [citation.evidence_id for citation in self.citations]
        for claim in self.claims:
            citation_ids.extend(citation.evidence_id for citation in claim.citations)
        for risk in self.risks:
            citation_ids.extend(citation.evidence_id for citation in risk.citations)

        unknown_ids = sorted(
            {evidence_id for evidence_id in citation_ids if evidence_id not in source_ids}
        )
        if unknown_ids:
            raise ValueError("agent citations must reference declared source_evidence_ids")
        return self


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
    "FundamentalAgentAuditMetadata",
    "FundamentalAgentSignal",
    "FundamentalAnalysis",
    "FundamentalNlpAnalysisRequest",
    "FundamentalNlpAnalysisResponse",
    "FundamentalNlpCitation",
    "FundamentalNlpClaim",
    "FundamentalNlpRisk",
    "MacroContext",
    "MetricValue",
    "SectorContext",
    "TechnicalAnalysis",
    "TechnicalMlSignal",
]
