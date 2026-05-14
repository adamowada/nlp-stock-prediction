"""Fixture provider factory for offline Phase 4 tool runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.providers import (
    FundamentalsProvider,
    NewsProvider,
    RedditProvider,
)
from nlp_stock_prediction.orchestration.phase2_common import utc_now
from nlp_stock_prediction.providers._base import JsonResponse, ProviderTransportError
from nlp_stock_prediction.providers.news import PublicNewsProvider, PublicNewsProviderConfig
from nlp_stock_prediction.providers.sec_edgar import SecEdgarFundamentalsProvider
from nlp_stock_prediction.providers.social import XRecentSearchProvider
from nlp_stock_prediction.reddit.provider import FixtureRedditProvider


@dataclass(frozen=True)
class Phase4FixtureProviderFactory:
    """Build offline providers from the repository fixture tree."""

    repo_root: Path

    def market_html_path(self, symbol: str) -> Path | None:
        return self.fixture_path(
            "raw",
            "candlecharts",
            f"public_ohlcv_{symbol.strip().lower()}.html",
        )

    def reddit_provider(self) -> RedditProvider | None:
        card_path = self.fixture_path("reddit", "devvit_card_normal.html")
        records = self.fixture_json("reddit", "discussion_records.json")
        if card_path is None or not isinstance(records, list):
            return None
        return cast(
            RedditProvider,
            FixtureRedditProvider(
                ticker_card_html=card_path.read_text(encoding="utf-8"),
                discussion_records=cast(list[dict[str, object]], records),
                fetched_at=utc_now(),
                raw_ticker_snapshot_id="raw-reddit-ticker-card",
                raw_discussion_snapshot_id="raw-reddit-discussion",
            ),
        )

    def x_provider(self, symbol: str) -> XRecentSearchProvider | None:
        if symbol.upper() != "TSLA":
            return None
        payload = self.fixture_json("raw", "x", "recent_tsla.json")
        if payload is None:
            return None
        provider = XRecentSearchProvider(
            bearer_token="fixture-token",
            transport=_StaticJsonTransport({"tweets/search/recent": cast(JsonObject, payload)}),
            now=utc_now,
        )
        provider.provider_name = "fixture-x-recent-search"
        return provider

    def news_providers(self, symbol: str) -> tuple[NewsProvider, ...]:
        if symbol.upper() != "TSLA":
            return ()
        payload = self.fixture_json("raw", "news", "tsla.json")
        if payload is None:
            return ()
        return (
            PublicNewsProvider(
                config=PublicNewsProviderConfig(
                    provider_name="fixture-news",
                    endpoint="https://news.example.invalid/v1/search",
                    api_key_param="token",
                    query_param="search",
                ),
                api_key="fixture-key",
                transport=_StaticJsonTransport(
                    {"news.example.invalid/v1/search": cast(JsonObject, payload)}
                ),
                now=utc_now,
            ),
        )

    def fundamentals_providers(self, symbol: str) -> tuple[FundamentalsProvider, ...]:
        if symbol.upper() != "TSLA":
            return ()
        companyfacts = self.fixture_json("raw", "sec_edgar", "companyfacts_tsla.json")
        submissions = self.fixture_json("raw", "sec_edgar", "submissions_tsla.json")
        if companyfacts is None or submissions is None:
            return ()
        provider = SecEdgarFundamentalsProvider(
            ticker_cik_map={symbol.upper(): "1318605"},
            user_agent="nlp-stock-prediction fixture-runtime contact@example.test",
            transport=_StaticJsonTransport(
                {
                    "companyfacts": cast(JsonObject, companyfacts),
                    "submissions": cast(JsonObject, submissions),
                }
            ),
            now=utc_now,
        )
        provider.provider_name = "fixture-sec-edgar"
        return (provider,)

    def fixture_path(self, *parts: str) -> Path | None:
        path = self.repo_root.joinpath("tests", "fixtures", *parts)
        return path if path.exists() else None

    def fixture_json(self, *parts: str) -> object | None:
        path = self.fixture_path(*parts)
        if path is None:
            return None
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
        return loaded


@dataclass(frozen=True)
class _StaticJsonTransport:
    responses: Mapping[str, JsonObject]

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> JsonResponse:
        del headers, timeout
        for url_fragment, payload in self.responses.items():
            if url_fragment in url:
                return JsonResponse(payload=payload)
        raise ProviderTransportError(
            f"No fixture JSON response is registered for URL: {url}",
            error_type="fixture_not_found",
        )


__all__ = ["Phase4FixtureProviderFactory"]
