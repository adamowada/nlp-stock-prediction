"""Run-mode adapters for Phase 4 report execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.providers import (
    FundamentalsProvider,
    MacroProvider,
    MarketDataProvider,
    NewsProvider,
    RedditProvider,
    XProvider,
)
from nlp_stock_prediction.orchestration.phase2_common import symbol_slug
from nlp_stock_prediction.orchestration.phase4_fixture_providers import (
    Phase4FixtureProviderFactory,
)
from nlp_stock_prediction.orchestration.phase4_live_providers import (
    Phase4LiveProviderFactoryProtocol,
)
from nlp_stock_prediction.orchestration.phase4_universe_discovery import (
    PHASE4_FIXTURE_PROVIDER,
    PHASE4_LIVE_SYMBOL_PROVIDER,
    UniverseDiscoveryProvider,
)
from nlp_stock_prediction.orchestration.report_data_modes import (
    LIVE_REPORT_DATA_MODE,
    ReportDataMode,
    report_data_mode_metadata,
)
from nlp_stock_prediction.providers.candlecharts import CandlechartsMarketDataProvider
from nlp_stock_prediction.storage.sqlite import SQLiteStore


@dataclass(frozen=True)
class Phase4MarketDataSelection:
    provider: MarketDataProvider
    source_url: str | Path | None
    options: JsonObject


@dataclass(frozen=True)
class Phase4SocialProviderSelection:
    reddit_provider: RedditProvider | None
    x_provider: XProvider | None


@dataclass(frozen=True)
class Phase4RunModeAdapter:
    """Concrete live/offline behavior behind one Phase 4 run-mode Interface."""

    report_data_mode: ReportDataMode
    store: SQLiteStore
    fixtures: Phase4FixtureProviderFactory
    live_providers: Phase4LiveProviderFactoryProtocol

    @property
    def metadata(self) -> JsonObject:
        return report_data_mode_metadata(self.report_data_mode)

    @property
    def is_live(self) -> bool:
        return self.report_data_mode == LIVE_REPORT_DATA_MODE

    def universe_provider(self) -> UniverseDiscoveryProvider | None:
        return self.live_providers.universe_provider() if self.is_live else None

    def market_data_selection(self, symbol: str) -> Phase4MarketDataSelection:
        normalized_symbol = symbol.strip().upper()
        if self.is_live:
            source_url = self.live_providers.market_data_source_query_url(normalized_symbol)
            options = dict(self.metadata)
            if source_url is not None:
                options["source_query_url"] = source_url
            return Phase4MarketDataSelection(
                provider=self.live_providers.market_data_provider(normalized_symbol),
                source_url=source_url,
                options=options,
            )
        fixture_path = self.fixtures.market_html_path(normalized_symbol)
        return Phase4MarketDataSelection(
            provider=CandlechartsMarketDataProvider(
                html_path=fixture_path,
                allow_live=False,
            ),
            source_url=fixture_path,
            options=dict(self.metadata),
        )

    def social_providers(self, symbol: str) -> Phase4SocialProviderSelection:
        normalized_symbol = symbol.strip().upper()
        if self.is_live:
            return Phase4SocialProviderSelection(
                reddit_provider=self.live_providers.reddit_provider(),
                x_provider=self.live_providers.x_provider(normalized_symbol),
            )
        return Phase4SocialProviderSelection(
            reddit_provider=self.fixtures.reddit_provider(),
            x_provider=self.fixtures.x_provider(normalized_symbol),
        )

    def news_providers(self, symbol: str) -> tuple[NewsProvider, ...]:
        normalized_symbol = symbol.strip().upper()
        if self.is_live:
            return self.live_providers.news_providers(normalized_symbol)
        return self.fixtures.news_providers(normalized_symbol)

    def fundamentals_providers(self, symbol: str) -> tuple[FundamentalsProvider, ...]:
        normalized_symbol = symbol.strip().upper()
        if self.is_live:
            return self.live_providers.fundamentals_providers(normalized_symbol)
        return self.fixtures.fundamentals_providers(normalized_symbol)

    def macro_providers(self, symbol: str) -> tuple[MacroProvider, ...]:
        normalized_symbol = symbol.strip().upper()
        return self.live_providers.macro_providers(normalized_symbol) if self.is_live else ()

    def instrument_id(self, symbol: str) -> str:
        normalized_symbol = symbol.strip().upper()
        if self.is_live:
            instrument = self.store.find_instrument_by_provider_id(
                PHASE4_LIVE_SYMBOL_PROVIDER,
                "symbol",
                normalized_symbol,
            )
            if instrument is not None:
                return instrument.instrument_id
            for discovered in self.store.find_instruments_by_symbol_or_alias(normalized_symbol):
                if discovered.metadata.get("live_provider_symbol") is True:
                    return discovered.instrument_id
            return f"instrument:live:unknown:{symbol_slug(normalized_symbol)}"

        instrument = self.store.find_instrument_by_provider_id(
            PHASE4_FIXTURE_PROVIDER,
            "fixture-symbol",
            normalized_symbol,
        )
        if instrument is not None:
            return instrument.instrument_id
        instruments = self.store.find_instruments_by_symbol_or_alias(normalized_symbol)
        if instruments:
            return instruments[0].instrument_id
        return f"instrument:codex:{normalized_symbol}"

    def no_evidence_candidate_message(self, evidence_count: int) -> str | None:
        if self.is_live and evidence_count == 0:
            return (
                "No attributable live provider evidence was available; report rendering "
                "will emit structured insufficient evidence."
            )
        return None


__all__ = [
    "Phase4MarketDataSelection",
    "Phase4RunModeAdapter",
    "Phase4SocialProviderSelection",
]
