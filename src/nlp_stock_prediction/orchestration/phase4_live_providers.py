"""Live provider factory for the Phase 4/5 report path."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nlp_stock_prediction.contracts.providers import (
    FundamentalsProvider,
    MacroProvider,
    MarketDataProvider,
    NewsProvider,
    RedditProvider,
    XProvider,
)
from nlp_stock_prediction.orchestration.live_market_data import (
    ALPHA_VANTAGE_API_KEY_ENV,
    FALLBACK_ALPHA_VANTAGE_API_KEY_ENVS,
    LiveMarketDataSelector,
)
from nlp_stock_prediction.orchestration.phase4_universe_discovery import (
    Phase4LiveSymbolUniverseProvider,
    UniverseDiscoveryProvider,
)
from nlp_stock_prediction.providers._base import ProviderCache
from nlp_stock_prediction.providers.apnews import APNewsProvider, APNewsProviderConfig
from nlp_stock_prediction.providers.fred import FredMacroProvider
from nlp_stock_prediction.providers.market import (
    AlphaVantageFundamentalsProvider,
)
from nlp_stock_prediction.providers.reddit_scrape import RedditPublicPageProvider
from nlp_stock_prediction.providers.scraping import HtmlCache, configured_scrape_user_agent
from nlp_stock_prediction.providers.sec_edgar import SecEdgarFundamentalsProvider
from nlp_stock_prediction.providers.social import XRecentSearchProvider

FRED_API_KEY_ENV = "NLP_STOCK_PREDICTION_FRED_API_KEY"
X_BEARER_TOKEN_ENV = "NLP_STOCK_PREDICTION_X_BEARER_TOKEN"
SEC_CIK_MAP_ENV = "NLP_STOCK_PREDICTION_SEC_CIK_MAP"
LIVE_USER_AGENT_ENV = "NLP_STOCK_PREDICTION_LIVE_USER_AGENT"

_FALLBACK_FRED_API_KEY_ENVS = ("FRED_API_KEY",)
_FALLBACK_X_BEARER_TOKEN_ENVS = ("X_BEARER_TOKEN",)
_DEFAULT_SEC_TICKER_CIK_MAP: Mapping[str, str] = {
    "AAPL": "320193",
    "AMZN": "1018724",
    "GOOGL": "1652044",
    "META": "1326801",
    "MSFT": "789019",
    "NVDA": "1045810",
    "TSLA": "1318605",
}


class Phase4LiveProviderFactoryProtocol(Protocol):
    """Provider factory surface consumed by the Phase 4 service."""

    def universe_provider(self) -> UniverseDiscoveryProvider: ...

    def market_data_provider(self, symbol: str) -> MarketDataProvider: ...

    def market_data_source_query_url(self, symbol: str) -> str | None: ...

    def reddit_provider(self) -> RedditProvider | None: ...

    def x_provider(self, symbol: str) -> XProvider | None: ...

    def news_providers(self, symbol: str) -> tuple[NewsProvider, ...]: ...

    def fundamentals_providers(self, symbol: str) -> tuple[FundamentalsProvider, ...]: ...

    def macro_providers(self, symbol: str) -> tuple[MacroProvider, ...]: ...


@dataclass(frozen=True)
class Phase4LiveProviderFactory:
    """Build live providers from environment credentials and optional local caches."""

    cache_root: Path | None = None
    env: Mapping[str, str] | None = None

    def universe_provider(self) -> UniverseDiscoveryProvider:
        return Phase4LiveSymbolUniverseProvider()

    def market_data_provider(self, symbol: str) -> MarketDataProvider:
        return self._market_data_selector().primary_selection(symbol).provider

    def market_data_source_query_url(self, symbol: str) -> str | None:
        selection = self._market_data_selector().primary_selection(symbol)
        return None if selection.source_url is None else str(selection.source_url)

    def reddit_provider(self) -> RedditProvider | None:
        return RedditPublicPageProvider(
            allow_live_scraping=True,
            user_agent=self._scrape_user_agent(),
        )

    def x_provider(self, symbol: str) -> XProvider | None:
        del symbol
        return XRecentSearchProvider(
            bearer_token=self._first_env(X_BEARER_TOKEN_ENV, *_FALLBACK_X_BEARER_TOKEN_ENVS),
            cache=self._json_cache(),
        )

    def news_providers(self, symbol: str) -> tuple[NewsProvider, ...]:
        del symbol
        return (
            APNewsProvider(
                config=APNewsProviderConfig(user_agent=self._scrape_user_agent()),
                cache=self._html_cache(),
            ),
        )

    def fundamentals_providers(self, symbol: str) -> tuple[FundamentalsProvider, ...]:
        normalized_symbol = symbol.strip().upper()
        return (
            AlphaVantageFundamentalsProvider(
                api_key=self._first_env(
                    ALPHA_VANTAGE_API_KEY_ENV,
                    *FALLBACK_ALPHA_VANTAGE_API_KEY_ENVS,
                ),
                cache=self._json_cache(),
            ),
            SecEdgarFundamentalsProvider(
                ticker_cik_map=self._sec_ticker_cik_map(normalized_symbol),
                cache=self._json_cache(),
            ),
        )

    def macro_providers(self, symbol: str) -> tuple[MacroProvider, ...]:
        del symbol
        return (
            FredMacroProvider(
                api_key=self._first_env(FRED_API_KEY_ENV, *_FALLBACK_FRED_API_KEY_ENVS),
                cache=self._json_cache(),
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

    def _market_data_selector(self) -> LiveMarketDataSelector:
        return LiveMarketDataSelector(
            cache_root=self.cache_root,
            env=self._env,
        )

    def _scrape_user_agent(self) -> str:
        return self._first_env(LIVE_USER_AGENT_ENV) or configured_scrape_user_agent(self._env)

    def _sec_ticker_cik_map(self, requested_symbol: str) -> Mapping[str, str]:
        del requested_symbol
        configured = _parse_sec_cik_map(self._env.get(SEC_CIK_MAP_ENV))
        return {**_DEFAULT_SEC_TICKER_CIK_MAP, **configured}

    def _first_env(self, *names: str) -> str | None:
        for name in names:
            value = self._env.get(name)
            if value is not None and value.strip():
                return value.strip()
        return None

    @property
    def _env(self) -> Mapping[str, str]:
        return os.environ if self.env is None else self.env


def _parse_sec_cik_map(value: str | None) -> dict[str, str]:
    if value is None or not value.strip():
        return {}
    parsed: dict[str, str] = {}
    for raw_entry in value.split(","):
        if "=" not in raw_entry:
            continue
        raw_symbol, raw_cik = raw_entry.split("=", 1)
        symbol = raw_symbol.strip().upper()
        cik = "".join(character for character in raw_cik if character.isdigit())
        if symbol and cik:
            parsed[symbol] = cik
    return parsed


__all__ = [
    "ALPHA_VANTAGE_API_KEY_ENV",
    "FRED_API_KEY_ENV",
    "LIVE_USER_AGENT_ENV",
    "SEC_CIK_MAP_ENV",
    "X_BEARER_TOKEN_ENV",
    "Phase4LiveProviderFactory",
    "Phase4LiveProviderFactoryProtocol",
]
