"""Provider request, result, and protocol contracts."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
    TickerSymbol,
)
from nlp_stock_prediction.contracts.discovery import TickerDiscoveryResult
from nlp_stock_prediction.contracts.enums import (
    ProviderStatus,
    RiskProfile,
    TimeHorizon,
    WarningCode,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.extraction import StrategyExtraction
from nlp_stock_prediction.contracts.provenance import ProviderHealth, ProviderWarning


class DateWindow(ContractModel):
    start: date | datetime
    end: date | datetime

    @model_validator(mode="after")
    def validate_order(self) -> DateWindow:
        if isinstance(self.start, datetime) and isinstance(self.end, datetime):
            if self.end < self.start:
                raise ValueError("date window end must be greater than or equal to start")
            return self
        start_date = self.start.date() if isinstance(self.start, datetime) else self.start
        end_date = self.end.date() if isinstance(self.end, datetime) else self.end
        if end_date < start_date:
            raise ValueError("date window end must be greater than or equal to start")
        return self


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
    fetched_at: AwareDatetime
    data: T | None = None
    warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)
    health: ProviderHealth
    raw_snapshot_id: str | None = None
    cache_key: str | None = None

    @model_validator(mode="after")
    def validate_provider_result_consistency(self) -> ProviderResult[T]:
        if self.provider_name != self.health.provider_name:
            raise ValueError("provider result and health provider_name must match")
        if self.status != self.health.status:
            raise ValueError("provider result status must match health status")
        if self.status == ProviderStatus.OK and self.data is None:
            raise ValueError("ok provider results must include data")
        if self.status != ProviderStatus.OK and not self.warnings:
            raise ValueError("non-ok provider results must include at least one warning")
        if self.status == ProviderStatus.EMPTY and self.data is not None:
            raise ValueError("empty provider results must not include data")
        if self.status == ProviderStatus.EMPTY and all(
            warning.code != WarningCode.NO_DATA for warning in self.warnings
        ):
            raise ValueError("empty provider results must include a no_data warning")
        for warning in self.warnings:
            if warning.provider_name is not None and warning.provider_name != self.provider_name:
                raise ValueError("provider result warnings must match provider_name")
        if (
            self.status
            in {
                ProviderStatus.FAILED,
                ProviderStatus.UNCONFIGURED,
                ProviderStatus.UNAUTHORIZED,
                ProviderStatus.RATE_LIMITED,
                ProviderStatus.MALFORMED,
            }
            and self.data is not None
        ):
            raise ValueError("hard provider failures must not include data")
        return self


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
    timestamp: date | AwareDatetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = Field(ge=0)
    adjusted_close: Decimal | None = None


class ProviderMetric(ContractModel):
    """Provider-supplied fact before analysis lanes interpret it."""

    name: NonEmptyStr
    value: Decimal | float | int | str | None
    unit: str | None = None
    as_of: date | datetime | None = None
    metadata: JsonObject = Field(default_factory=dict)


class FundamentalsSnapshot(ContractModel):
    ticker: TickerSymbol
    company_name: str | None = None
    metrics: tuple[ProviderMetric, ...] = Field(default_factory=tuple)


class MarketSnapshot(ContractModel):
    ticker: TickerSymbol
    bars: tuple[PriceBar, ...] = Field(default_factory=tuple)
    liquidity_metrics: tuple[ProviderMetric, ...] = Field(default_factory=tuple)


class MacroSeries(ContractModel):
    series_id: NonEmptyStr
    name: NonEmptyStr
    values: tuple[ProviderMetric, ...] = Field(default_factory=tuple)


class MacroSnapshot(ContractModel):
    series: tuple[MacroSeries, ...] = Field(default_factory=tuple)


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

    def health(self) -> ProviderHealth: ...


class FundamentalsProvider(Protocol):
    provider_name: str

    def fetch_fundamentals(
        self, request: FundamentalsRequest
    ) -> ProviderResult[FundamentalsSnapshot]: ...

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
    "ProviderMetric",
    "ProviderRequest",
    "ProviderResult",
    "RedditProvider",
    "RunConfig",
    "TickerDiscoveryRequest",
    "XProvider",
]
