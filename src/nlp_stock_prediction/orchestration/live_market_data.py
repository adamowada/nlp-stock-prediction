"""Shared live market-data provider selection."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

from nlp_stock_prediction.contracts import JsonObject, MarketDataProvider, RetrievalMethod
from nlp_stock_prediction.providers._base import ProviderCache
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
            return LiveMarketDataSelection(
                provider=AlphaVantageMarketDataProvider(
                    api_key=alpha_key,
                    cache=self._json_cache(),
                    now=self._provider_now,
                ),
                source_url=alpha_vantage_source_url(normalized_symbol),
                role="primary",
                retrieval_method=RetrievalMethod.OFFICIAL_API,
            )
        return LiveMarketDataSelection(
            provider=YahooFinanceChartMarketDataProvider(
                cache=self._json_cache(),
                now=self._provider_now,
            ),
            source_url=yahoo_finance_chart_source_url(normalized_symbol),
            role="primary",
            retrieval_method=RetrievalMethod.PUBLIC_SCRAPE,
        )

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
            LiveMarketDataSelection(
                provider=YahooFinanceChartMarketDataProvider(
                    cache=self._json_cache(),
                    now=self._provider_now,
                ),
                source_url=yahoo_finance_chart_source_url(normalized_symbol),
                role="fallback",
                retrieval_method=RetrievalMethod.PUBLIC_SCRAPE,
            ),
            LiveMarketDataSelection(
                provider=CandlechartsMarketDataProvider(
                    allow_live=True,
                    cache=self._html_cache(),
                    now=self._provider_now,
                ),
                source_url=candlecharts_source_url(normalized_symbol),
                role="fallback",
                retrieval_method=RetrievalMethod.PUBLIC_SCRAPE,
            ),
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


__all__ = [
    "ALPHA_VANTAGE_API_KEY_ENV",
    "FALLBACK_ALPHA_VANTAGE_API_KEY_ENVS",
    "SUPPORTED_LIVE_MARKET_OUTCOME_ASSET_CLASSES",
    "LiveMarketDataSelection",
    "LiveMarketDataSelector",
    "alpha_vantage_source_url",
    "candlecharts_source_url",
    "yahoo_finance_chart_source_url",
]
