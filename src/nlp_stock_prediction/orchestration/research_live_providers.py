"""Live provider factory for the Research Stage/5 report path."""

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
)
from nlp_stock_prediction.orchestration.live_market_data import (
    ALPHA_VANTAGE_API_KEY_ENV,
    FALLBACK_ALPHA_VANTAGE_API_KEY_ENVS,
    LiveMarketDataSelector,
)
from nlp_stock_prediction.orchestration.research_universe_discovery import (
    ResearchLiveSymbolUniverseProvider,
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

FRED_API_KEY_ENV = "NLP_STOCK_PREDICTION_FRED_API_KEY"
LIVE_USER_AGENT_ENV = "NLP_STOCK_PREDICTION_LIVE_USER_AGENT"

_FALLBACK_FRED_API_KEY_ENVS = ("FRED_API_KEY",)


class ResearchLiveProviderFactoryProtocol(Protocol):
    """Provider factory surface consumed by the Research Stage service."""

    def universe_provider(self) -> UniverseDiscoveryProvider: ...

    def market_data_provider(self, symbol: str) -> MarketDataProvider: ...

    def market_data_source_query_url(self, symbol: str) -> str | None: ...

    def reddit_provider(self) -> RedditProvider | None: ...

    def news_providers(self, symbol: str) -> tuple[NewsProvider, ...]: ...

    def fundamentals_providers(self, symbol: str) -> tuple[FundamentalsProvider, ...]: ...

    def macro_providers(self, symbol: str) -> tuple[MacroProvider, ...]: ...


@dataclass(frozen=True)
class ResearchLiveProviderFactory:
    """Build live providers from environment credentials and optional local caches."""

    cache_root: Path | None = None
    env: Mapping[str, str] | None = None

    def universe_provider(self) -> UniverseDiscoveryProvider:
        return ResearchLiveSymbolUniverseProvider()

    def market_data_provider(self, symbol: str) -> MarketDataProvider:
        return self._market_data_selector().primary_selection(symbol).provider

    def market_data_source_query_url(self, symbol: str) -> str | None:
        selection = self._market_data_selector().primary_selection(symbol)
        return None if selection.source_url is None else str(selection.source_url)

    def reddit_provider(self) -> RedditProvider | None:
        return RedditPublicPageProvider(
            allow_live_scraping=True,
            user_agent=self._scrape_user_agent(),
            cache=self._html_cache(),
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
        del symbol
        return (
            AlphaVantageFundamentalsProvider(
                api_key=self._first_env(
                    ALPHA_VANTAGE_API_KEY_ENV,
                    *FALLBACK_ALPHA_VANTAGE_API_KEY_ENVS,
                ),
                cache=self._json_cache(),
            ),
            SecEdgarFundamentalsProvider(
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

    def _first_env(self, *names: str) -> str | None:
        for name in names:
            value = self._env.get(name)
            if value is not None and value.strip():
                return value.strip()
        return None

    @property
    def _env(self) -> Mapping[str, str]:
        return os.environ if self.env is None else self.env


__all__ = [
    "ALPHA_VANTAGE_API_KEY_ENV",
    "FRED_API_KEY_ENV",
    "LIVE_USER_AGENT_ENV",
    "ResearchLiveProviderFactory",
    "ResearchLiveProviderFactoryProtocol",
]
