"""Report and audit artifact contracts."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.analysis import (
    FundamentalAnalysis,
    MacroContext,
    SectorContext,
    TechnicalAnalysis,
)
from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
    TickerSymbol,
)
from nlp_stock_prediction.contracts.discovery import TickerDiscoveryResult
from nlp_stock_prediction.contracts.enums import RiskProfile
from nlp_stock_prediction.contracts.evidence import SourceEvidence
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

    @model_validator(mode="after")
    def validate_v1_guardrails(self) -> Disclaimer:
        if not self.educational_only:
            raise ValueError("v1 disclaimers must be educational_only")
        if not self.not_financial_advice:
            raise ValueError("v1 disclaimers must be not_financial_advice")
        if not self.no_auto_trading:
            raise ValueError("v1 disclaimers must prohibit auto-trading")
        return self


class DataFreshnessSummary(ContractModel):
    """Freshness summary shown in report headers and JSON."""

    as_of: AwareDatetime
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
    created_at: AwareDatetime
    produced_by: NonEmptyStr
    sha256: str | None = None
    record_count: int | None = Field(default=None, ge=0)
    metadata: JsonObject = Field(default_factory=dict)


class AuditManifest(ContractModel):
    """Audit spine for a generated report."""

    run_id: NonEmptyStr
    schema_version: NonEmptyStr
    created_at: AwareDatetime
    artifacts: tuple[AuditArtifact, ...] = Field(default_factory=tuple)
    provider_run_ids: tuple[str, ...] = Field(default_factory=tuple)
    model_versions: JsonObject = Field(default_factory=dict)
    prompt_versions: JsonObject = Field(default_factory=dict)
    config_hash: str | None = None
    command_args: JsonObject = Field(default_factory=dict)
    recommendation_trace_ids: tuple[str, ...] = Field(default_factory=tuple)


class MarkdownReportOutline(ContractModel):
    """Minimal Markdown renderer contract frozen before report rendering is implemented."""

    schema_version: NonEmptyStr
    heading_order: tuple[NonEmptyStr, ...]
    ticker_section_heading_template: NonEmptyStr
    required_ticker_subsections: tuple[NonEmptyStr, ...]
    final_section_headings: tuple[NonEmptyStr, ...]
    required_footer_headings: tuple[NonEmptyStr, ...]
    require_disclaimer: bool = True
    require_evidence_references: bool = True
    require_audit_artifacts: bool = True

    @model_validator(mode="after")
    def validate_outline(self) -> MarkdownReportOutline:
        if len(set(self.heading_order)) != len(self.heading_order):
            raise ValueError("Markdown heading_order entries must be unique")
        if "{ticker}" not in self.ticker_section_heading_template:
            raise ValueError("ticker section heading template must include {ticker}")
        if not self.required_ticker_subsections:
            raise ValueError("Markdown outline requires ticker subsections")
        if not self.final_section_headings:
            raise ValueError("Markdown outline requires final section headings")
        return self


DEFAULT_MARKDOWN_REPORT_OUTLINE = MarkdownReportOutline(
    schema_version="markdown-report.v1",
    heading_order=(
        "Daily Stock Opportunity Report",
        "Data Freshness",
        "Provider Warnings",
        "Ticker Sections",
        "Qualified Trading Strategies Or No-Trade Summary",
        "Disclaimer",
        "Audit Artifacts",
    ),
    ticker_section_heading_template="{ticker}",
    required_ticker_subsections=(
        "Observed Discussion",
        "Social And News",
        "Strategy Clusters",
        "Technical Analysis",
        "Fundamental Analysis",
        "Sector Context",
        "Macro Context",
        "Opportunity Notes",
        "Evidence References",
    ),
    final_section_headings=(
        "Qualified Trading Strategies",
        "No-Trade Summary",
    ),
    required_footer_headings=("Disclaimer", "Audit Artifacts"),
)


class DailyReport(ContractModel):
    """Top-level Markdown/JSON report contract."""

    schema_version: NonEmptyStr
    run_id: NonEmptyStr
    report_date: date
    generated_at: AwareDatetime
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
    evidence_sources: tuple[SourceEvidence, ...] = Field(default_factory=tuple)
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
        candidate_ids = tuple(candidate.candidate_id for candidate in self.trade_candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("trade candidate ids must be unique")
        source_evidence_ids = tuple(evidence.evidence_id for evidence in self.evidence_sources)
        if len(set(source_evidence_ids)) != len(source_evidence_ids):
            raise ValueError("report evidence_sources ids must be unique")
        candidate_by_id = {candidate.candidate_id: candidate for candidate in self.trade_candidates}
        section_references: dict[str, TickerSymbol] = {}
        cited_evidence_ids: set[str] = set()
        for section in self.ticker_sections:
            cited_evidence_ids.update(reference.evidence_id for reference in section.evidence)
            for cluster in section.strategy_clusters:
                cited_evidence_ids.update(reference.evidence_id for reference in cluster.evidence)
            for component in (
                section.technical_analysis,
                section.fundamental_analysis,
                section.sector_context,
                section.macro_context,
            ):
                if component is not None:
                    cited_evidence_ids.update(
                        reference.evidence_id for reference in component.evidence
                    )
            for recommendation_id in section.recommendation_ids:
                if recommendation_id not in candidate_by_id:
                    raise ValueError("ticker section recommendation_ids must reference candidates")
                if recommendation_id in section_references:
                    raise ValueError(
                        "trade candidates must be referenced by exactly one ticker section"
                    )
                section_references[recommendation_id] = section.ticker
        for candidate in self.trade_candidates:
            if candidate.ticker not in self.ticker_discovery.tickers:
                raise ValueError("trade candidates must use discovered tickers")
            if candidate.disclaimer_id != self.disclaimer.disclaimer_id:
                raise ValueError("trade candidate disclaimer_id must match report disclaimer")
            if candidate.candidate_id not in section_references:
                raise ValueError("trade candidates must be referenced by a ticker section")
            if section_references[candidate.candidate_id] != candidate.ticker:
                raise ValueError("ticker section recommendation_ids must match candidate ticker")
            cited_evidence_ids.update(reference.evidence_id for reference in candidate.evidence)
            for score_component in (*candidate.score.components, *candidate.score.penalties):
                cited_evidence_ids.update(
                    reference.evidence_id for reference in score_component.evidence
                )
        if self.evidence_sources:
            missing_evidence_ids = cited_evidence_ids.difference(source_evidence_ids)
            if missing_evidence_ids:
                raise ValueError("report evidence_sources must include every cited evidence_id")
        return self


__all__ = [
    "DEFAULT_MARKDOWN_REPORT_OUTLINE",
    "AuditArtifact",
    "AuditManifest",
    "DailyReport",
    "DataFreshnessSummary",
    "Disclaimer",
    "MarkdownReportOutline",
    "TickerReportSection",
]
