"""Provider request, result, and protocol contracts."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from pydantic import Field

from nlp_stock_prediction.contracts.analysis import (
    FundamentalAnalysis,
    MacroContext,
    MetricValue,
    SectorContext,
    TechnicalAnalysis,
)
from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, NonEmptyStr, TickerSymbol
from nlp_stock_prediction.contracts.discovery import TickerDiscoveryResult
from nlp_stock_prediction.contracts.enums import ProviderStatus, RiskProfile, TimeHorizon
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.extraction import StrategyExtraction
from nlp_stock_prediction.contracts.provenance import ProviderHealth, ProviderWarning


class DateWindow(ContractModel):
    start: date | datetime
    end: date | datetime


class ProviderRequest(ContractModel):
    """Common request envelope captured in provider fixtures and audit artifacts."""

    request_id: NonEmptyStr
    run_date: date
    tickers: tuple[TickerSymbol, ...] = Field(default_factory=tuple)
    window: DateWindow | None = None
    limit: int | None = Field(default=None, ge=1)
    query: str | None = None
    options: JsonObject = Field(default_factory=dict)


class ProviderResult[T](ContractModel):
    """Shared provider result envelope.

    Expected upstream/data problems should be represented here instead of raised, so the
    reporting lane can degrade gracefully.
    """

    provider_name: NonEmptyStr
    status: ProviderStatus
    request: ProviderRequest
    fetched_at: datetime
    data: T | None = None
    warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)
    health: ProviderHealth
    raw_snapshot_id: str | None = None
    cache_key: str | None = None


class RunConfig(ContractModel):
    """CLI run configuration contract."""

    run_date: date
    output_dir: Path
    capital: Decimal | None = Field(default=None, ge=Decimal("0"))
    risk_profile: RiskProfile = RiskProfile.EXPLORATORY
    fixture_dir: Path | None = None
    cache_dir: Path | None = None
    offline: bool = False


class TickerDiscoveryRequest(ProviderRequest):
    source_url: str | None = None


class EvidenceRequest(ProviderRequest):
    include_comments: bool = True
    include_posts: bool = True


class MarketDataRequest(ProviderRequest):
    interval: NonEmptyStr = "1d"
    adjusted: bool = True


class FundamentalsRequest(ProviderRequest):
    fiscal_period: str | None = None


class MacroRequest(ProviderRequest):
    series_ids: tuple[str, ...] = Field(default_factory=tuple)
    horizon: TimeHorizon = TimeHorizon.UNKNOWN


class ExtractionRequest(ProviderRequest):
    evidence: tuple[SourceEvidence, ...]
    prompt_version: NonEmptyStr
    schema_version: NonEmptyStr


class PriceBar(ContractModel):
    ticker: TickerSymbol
    timestamp: date | datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = Field(ge=0)
    adjusted_close: Decimal | None = None


class MarketSnapshot(ContractModel):
    ticker: TickerSymbol
    bars: tuple[PriceBar, ...] = Field(default_factory=tuple)
    liquidity_metrics: tuple[MetricValue, ...] = Field(default_factory=tuple)


class FundamentalsSnapshot(ContractModel):
    ticker: TickerSymbol
    company_name: str | None = None
    metrics: tuple[MetricValue, ...] = Field(default_factory=tuple)
    analysis_seed: FundamentalAnalysis | None = None


class MacroSeries(ContractModel):
    series_id: NonEmptyStr
    name: NonEmptyStr
    values: tuple[MetricValue, ...] = Field(default_factory=tuple)


class MacroSnapshot(ContractModel):
    series: tuple[MacroSeries, ...] = Field(default_factory=tuple)
    context_seed: MacroContext | None = None


class RedditProvider(Protocol):
    provider_name: str

    def discover_tickers(
        self, request: TickerDiscoveryRequest
    ) -> ProviderResult[TickerDiscoveryResult]: ...

    def fetch_discussion(
        self, request: EvidenceRequest
    ) -> ProviderResult[tuple[SourceEvidence, ...]]: ...

    def health(self) -> ProviderHealth: ...


class XProvider(Protocol):
    provider_name: str

    def fetch_social_posts(
        self, request: EvidenceRequest
    ) -> ProviderResult[tuple[SourceEvidence, ...]]: ...

    def health(self) -> ProviderHealth: ...


class NewsProvider(Protocol):
    provider_name: str

    def fetch_articles(
        self, request: EvidenceRequest
    ) -> ProviderResult[tuple[SourceEvidence, ...]]: ...

    def health(self) -> ProviderHealth: ...


class MarketDataProvider(Protocol):
    provider_name: str

    def fetch_daily_candles(self, request: MarketDataRequest) -> ProviderResult[MarketSnapshot]: ...

    def build_technical_seed(
        self, request: MarketDataRequest
    ) -> ProviderResult[TechnicalAnalysis | None]: ...

    def health(self) -> ProviderHealth: ...


class FundamentalsProvider(Protocol):
    provider_name: str

    def fetch_fundamentals(
        self, request: FundamentalsRequest
    ) -> ProviderResult[FundamentalsSnapshot]: ...

    def build_sector_context(
        self, request: FundamentalsRequest
    ) -> ProviderResult[SectorContext | None]: ...

    def health(self) -> ProviderHealth: ...


class MacroProvider(Protocol):
    provider_name: str

    def fetch_macro(self, request: MacroRequest) -> ProviderResult[MacroSnapshot]: ...

    def health(self) -> ProviderHealth: ...


class LLMExtractor(Protocol):
    provider_name: str

    def extract_strategies(
        self, request: ExtractionRequest
    ) -> ProviderResult[tuple[StrategyExtraction, ...]]: ...

    def health(self) -> ProviderHealth: ...


__all__ = [
    "DateWindow",
    "EvidenceRequest",
    "ExtractionRequest",
    "FundamentalsProvider",
    "FundamentalsRequest",
    "FundamentalsSnapshot",
    "LLMExtractor",
    "MacroProvider",
    "MacroRequest",
    "MacroSeries",
    "MacroSnapshot",
    "MarketDataProvider",
    "MarketDataRequest",
    "MarketSnapshot",
    "NewsProvider",
    "PriceBar",
    "ProviderRequest",
    "ProviderResult",
    "RedditProvider",
    "RunConfig",
    "TickerDiscoveryRequest",
    "XProvider",
]
