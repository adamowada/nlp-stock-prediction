"""Report and audit artifact contracts."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.analysis import (
    FundamentalAnalysis,
    MacroContext,
    SectorContext,
    TechnicalAnalysis,
)
from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, NonEmptyStr, TickerSymbol
from nlp_stock_prediction.contracts.discovery import TickerDiscoveryResult
from nlp_stock_prediction.contracts.enums import RiskProfile
from nlp_stock_prediction.contracts.extraction import StrategyCluster
from nlp_stock_prediction.contracts.provenance import (
    DataReference,
    EvidenceReference,
    ProviderHealth,
)
from nlp_stock_prediction.contracts.recommendation import TradeCandidate


class Disclaimer(ContractModel):
    """Report disclaimer contract."""

    disclaimer_id: NonEmptyStr
    version: NonEmptyStr
    text: NonEmptyStr
    educational_only: bool = True
    not_financial_advice: bool = True
    no_auto_trading: bool = True
    applies_to: tuple[str, ...] = ("report", "trade_candidates")


class DataFreshnessSummary(ContractModel):
    """Freshness summary shown in report headers and JSON."""

    as_of: datetime
    summary: NonEmptyStr
    stale_provider_names: tuple[str, ...] = Field(default_factory=tuple)
    missing_provider_names: tuple[str, ...] = Field(default_factory=tuple)


class TickerReportSection(ContractModel):
    """One ticker section in the Markdown and JSON report."""

    ticker: TickerSymbol
    company_name: str | None = None
    discovery_refs: tuple[str, ...] = Field(default_factory=tuple)
    observed_discussion_summary: str | None = None
    social_news_summary: str | None = None
    strategy_clusters: tuple[StrategyCluster, ...] = Field(default_factory=tuple)
    technical_analysis: TechnicalAnalysis | None = None
    fundamental_analysis: FundamentalAnalysis | None = None
    sector_context: SectorContext | None = None
    macro_context: MacroContext | None = None
    opportunity_notes: tuple[str, ...] = Field(default_factory=tuple)
    recommendation_ids: tuple[str, ...] = Field(default_factory=tuple)
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    warning_ids: tuple[str, ...] = Field(default_factory=tuple)
    data_quality: JsonObject = Field(default_factory=dict)


class AuditArtifact(ContractModel):
    """One file written to the report audit directory."""

    artifact_id: NonEmptyStr
    artifact_type: Literal[
        "raw_snapshot",
        "normalized_evidence",
        "extraction_output",
        "analysis_context",
        "scoring_input",
        "markdown_report",
        "json_report",
        "provider_result",
    ]
    path: NonEmptyStr
    created_at: datetime
    produced_by: NonEmptyStr
    sha256: str | None = None
    record_count: int | None = Field(default=None, ge=0)
    metadata: JsonObject = Field(default_factory=dict)


class AuditManifest(ContractModel):
    """Audit spine for a generated report."""

    run_id: NonEmptyStr
    schema_version: NonEmptyStr
    created_at: datetime
    artifacts: tuple[AuditArtifact, ...] = Field(default_factory=tuple)
    provider_run_ids: tuple[str, ...] = Field(default_factory=tuple)
    model_versions: JsonObject = Field(default_factory=dict)
    prompt_versions: JsonObject = Field(default_factory=dict)
    config_hash: str | None = None
    command_args: JsonObject = Field(default_factory=dict)
    recommendation_trace_ids: tuple[str, ...] = Field(default_factory=tuple)


class DailyReport(ContractModel):
    """Top-level Markdown/JSON report contract."""

    schema_version: NonEmptyStr
    run_id: NonEmptyStr
    report_date: date
    generated_at: datetime
    timezone: NonEmptyStr
    app_version: str | None = None
    git_sha: str | None = None
    config_hash: str | None = None
    command_args: JsonObject = Field(default_factory=dict)
    risk_profile: RiskProfile
    account_capital: str | None = None
    disclaimer: Disclaimer
    ticker_discovery: TickerDiscoveryResult
    data_freshness: DataFreshnessSummary
    provider_health: tuple[ProviderHealth, ...] = Field(default_factory=tuple)
    ticker_sections: tuple[TickerReportSection, ...]
    trade_candidates: tuple[TradeCandidate, ...] = Field(default_factory=tuple)
    no_trade_summary: str | None = None
    audit_manifest: AuditManifest | DataReference | None = None

    @model_validator(mode="after")
    def validate_report_ticker_shape(self) -> DailyReport:
        section_tickers = tuple(section.ticker for section in self.ticker_sections)
        if len(section_tickers) != 6:
            raise ValueError("daily reports must include exactly six ticker sections")
        if tuple(self.ticker_discovery.tickers) != section_tickers:
            raise ValueError("ticker sections must match discovered tickers in order")
        if not self.trade_candidates and not self.no_trade_summary:
            raise ValueError("reports without trade candidates must include no_trade_summary")
        return self


__all__ = [
    "AuditArtifact",
    "AuditManifest",
    "DailyReport",
    "DataFreshnessSummary",
    "Disclaimer",
    "TickerReportSection",
]
