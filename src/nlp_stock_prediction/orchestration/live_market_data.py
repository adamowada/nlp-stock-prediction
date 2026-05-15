"""Shared live market-data provider selection."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

from nlp_stock_prediction.contracts import (
    CredentialState,
    JsonObject,
    MarketDataProvider,
    MarketDataRequest,
    MarketSnapshot,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    RetrievalMethod,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.providers._base import ProviderCache, provider_health, provider_warning
from nlp_stock_prediction.providers.candlecharts import (
    CANDLECHARTS_ENDPOINT,
    CandlechartsMarketDataProvider,
)
from nlp_stock_prediction.providers.market import (
    ALPHA_VANTAGE_ENDPOINT,
    AlphaVantageMarketDataProvider,
    YahooFinanceChartMarketDataProvider,
    yahoo_finance_chart_source_url,
)
from nlp_stock_prediction.providers.scraping import HtmlCache

ALPHA_VANTAGE_API_KEY_ENV = "NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY"
FALLBACK_ALPHA_VANTAGE_API_KEY_ENVS = (
    "ALPHA_VANTAGE_API_KEY",
    "MARKET_DATA_ALPHA_VANTAGE_API_KEY",
)
SUPPORTED_LIVE_MARKET_OUTCOME_ASSET_CLASSES = frozenset({"stock", "etf"})


@dataclass(frozen=True)
class LiveMarketDataSelection:
    """One real market-data source selected for a live workflow."""

    provider: MarketDataProvider
    source_url: str | Path | None = None
    role: str = "primary"
    retrieval_method: RetrievalMethod | None = None
    options: JsonObject = field(default_factory=dict)


@dataclass
class FallbackMarketDataProvider:
    """Try live market-data providers in order until one returns usable bars."""

    selections: tuple[LiveMarketDataSelection, ...]
    provider_name: str = "live-market-data-fallback"

    def __post_init__(self) -> None:
        if not self.selections:
            raise ValueError("fallback market-data provider requires at least one selection")

    def fetch_daily_candles(
        self,
        request: MarketDataRequest,
    ) -> ProviderResult[MarketSnapshot]:
        prior_warnings: list[ProviderWarning] = []
        last_result: ProviderResult[MarketSnapshot] | None = None
        last_prior_warnings: tuple[ProviderWarning, ...] = ()
        for selection in self.selections:
            result = selection.provider.fetch_daily_candles(request)
            if _has_usable_market_bars(result):
                return _with_prior_provider_warnings(result, tuple(prior_warnings))
            last_result = result
            last_prior_warnings = tuple(prior_warnings)
            prior_warnings.extend(result.warnings)
        if last_result is None:
            raise RuntimeError("fallback market-data provider has no provider selections")
        return _with_prior_provider_warnings(last_result, last_prior_warnings)

    def health(self) -> ProviderHealth:
        return provider_health(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            checked_at=datetime.now(UTC),
            credential_state=CredentialState.NOT_REQUIRED,
        )

    def source_url_for_provider(self, provider_name: str) -> str | Path | None:
        for selection in self.selections:
            if selection.provider.provider_name == provider_name:
                return selection.source_url
        return None


@dataclass(frozen=True)
class LiveMarketDataSelector:
    """Build live stock/ETF market-data selections from credentials and public sources."""

    cache_root: Path | None = None
    env: Mapping[str, str] | None = None
    now: Callable[[], datetime] | None = None

    def primary_selection(self, symbol: str) -> LiveMarketDataSelection:
        normalized_symbol = _normalize_symbol(symbol)
        alpha_key = self._first_env(
            ALPHA_VANTAGE_API_KEY_ENV,
            *FALLBACK_ALPHA_VANTAGE_API_KEY_ENVS,
        )
        if alpha_key:
            primary = LiveMarketDataSelection(
                provider=AlphaVantageMarketDataProvider(
                    api_key=alpha_key,
                    cache=self._json_cache(),
                    now=self._provider_now,
                ),
                source_url=alpha_vantage_source_url(normalized_symbol),
                role="primary",
                retrieval_method=RetrievalMethod.OFFICIAL_API,
            )
            fallbacks = (
                self._yahoo_selection(normalized_symbol, role="fallback"),
                self._candlecharts_selection(normalized_symbol, role="fallback"),
            )
            return LiveMarketDataSelection(
                provider=FallbackMarketDataProvider((primary, *fallbacks)),
                source_url=primary.source_url,
                role="primary",
                retrieval_method=RetrievalMethod.OFFICIAL_API,
                options={
                    "fallback_providers": [
                        selection.provider.provider_name for selection in fallbacks
                    ]
                },
            )
        return self._yahoo_selection(normalized_symbol, role="primary")

    def outcome_selections(
        self, *, symbol: str, asset_class: str
    ) -> tuple[LiveMarketDataSelection, ...]:
        if asset_class.strip().lower() not in SUPPORTED_LIVE_MARKET_OUTCOME_ASSET_CLASSES:
            return ()
        normalized_symbol = _normalize_symbol(symbol)
        alpha_key = self._first_env(
            ALPHA_VANTAGE_API_KEY_ENV,
            *FALLBACK_ALPHA_VANTAGE_API_KEY_ENVS,
        )
        return (
            LiveMarketDataSelection(
                provider=AlphaVantageMarketDataProvider(
                    api_key=alpha_key,
                    cache=self._json_cache(),
                    now=self._provider_now,
                ),
                source_url=alpha_vantage_source_url(normalized_symbol),
                role="primary",
                retrieval_method=RetrievalMethod.OFFICIAL_API,
            ),
            self._yahoo_selection(normalized_symbol, role="fallback"),
            self._candlecharts_selection(normalized_symbol, role="fallback"),
        )

    def _yahoo_selection(self, symbol: str, *, role: str) -> LiveMarketDataSelection:
        return LiveMarketDataSelection(
            provider=YahooFinanceChartMarketDataProvider(
                cache=self._json_cache(),
                now=self._provider_now,
            ),
            source_url=yahoo_finance_chart_source_url(symbol),
            role=role,
            retrieval_method=RetrievalMethod.PUBLIC_SCRAPE,
        )

    def _candlecharts_selection(self, symbol: str, *, role: str) -> LiveMarketDataSelection:
        return LiveMarketDataSelection(
            provider=CandlechartsMarketDataProvider(
                allow_live=True,
                cache=self._html_cache(),
                now=self._provider_now,
            ),
            source_url=candlecharts_source_url(symbol),
            role=role,
            retrieval_method=RetrievalMethod.PUBLIC_SCRAPE,
        )

    def _json_cache(self) -> ProviderCache | None:
        if self.cache_root is None:
            return None
        return ProviderCache(self.cache_root / "json")

    def _html_cache(self) -> HtmlCache | None:
        if self.cache_root is None:
            return None
        return HtmlCache(self.cache_root / "html")

    def _first_env(self, *names: str) -> str | None:
        env = os.environ if self.env is None else self.env
        for name in names:
            value = env.get(name)
            if value is not None and value.strip():
                return value.strip()
        return None

    def _provider_now(self) -> datetime:
        if self.now is None:
            return datetime.now(UTC)
        return self.now()


def alpha_vantage_source_url(symbol: str) -> str:
    return f"{ALPHA_VANTAGE_ENDPOINT}?" + urlencode(
        {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": _normalize_symbol(symbol),
            "outputsize": "compact",
        }
    )


def candlecharts_source_url(symbol: str) -> str:
    return f"{CANDLECHARTS_ENDPOINT}?{urlencode({'symbol': _normalize_symbol(symbol)})}"


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _has_usable_market_bars(result: ProviderResult[MarketSnapshot]) -> bool:
    return (
        result.status in {ProviderStatus.OK, ProviderStatus.PARTIAL, ProviderStatus.STALE}
        and result.data is not None
        and bool(result.data.bars)
    )


def _with_prior_provider_warnings(
    result: ProviderResult[MarketSnapshot],
    prior_warnings: tuple[ProviderWarning, ...],
) -> ProviderResult[MarketSnapshot]:
    if not prior_warnings:
        return result
    fallback_warning = _fallback_attempt_warning(result, prior_warnings)
    warnings = (fallback_warning, *result.warnings)
    return ProviderResult[MarketSnapshot](
        provider_name=result.provider_name,
        status=result.status,
        request=result.request,
        fetched_at=result.fetched_at,
        data=result.data,
        warnings=warnings,
        health=ProviderHealth(
            provider_name=result.health.provider_name,
            status=result.health.status,
            checked_at=result.health.checked_at,
            credential_state=result.health.credential_state,
            latency_ms=result.health.latency_ms,
            rate_limit_remaining=result.health.rate_limit_remaining,
            rate_limit_reset_at=result.health.rate_limit_reset_at,
            last_success_at=result.health.last_success_at,
            warnings=warnings,
        ),
        raw_snapshot_id=result.raw_snapshot_id,
        cache_key=result.cache_key,
    )


def _fallback_attempt_warning(
    result: ProviderResult[MarketSnapshot],
    prior_warnings: tuple[ProviderWarning, ...],
) -> ProviderWarning:
    providers = tuple(
        provider_name
        for provider_name in dict.fromkeys(warning.provider_name for warning in prior_warnings)
        if provider_name is not None
    )
    first_message = prior_warnings[0].message if prior_warnings else "unknown upstream failure"
    return provider_warning(
        provider_name=result.provider_name,
        code=WarningCode.PARTIAL_DATA,
        severity=WarningSeverity.WARNING,
        message=(
            "Used fallback market data after prior provider failure: "
            f"{', '.join(providers)}; first warning: {first_message}"
        ),
        occurred_at=result.fetched_at,
        raw_snapshot_id=result.raw_snapshot_id,
        metadata={
            "fallback_used": True,
            "prior_provider_names": list(providers),
            "prior_warning_count": len(prior_warnings),
            "prior_warning_codes": [warning.code.value for warning in prior_warnings],
        },
    )


__all__ = [
    "ALPHA_VANTAGE_API_KEY_ENV",
    "FALLBACK_ALPHA_VANTAGE_API_KEY_ENVS",
    "SUPPORTED_LIVE_MARKET_OUTCOME_ASSET_CLASSES",
    "FallbackMarketDataProvider",
    "LiveMarketDataSelection",
    "LiveMarketDataSelector",
    "alpha_vantage_source_url",
    "candlecharts_source_url",
    "yahoo_finance_chart_source_url",
]
