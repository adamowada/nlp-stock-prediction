"""Fixture provider factory for offline Research Stage tool runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.providers import (
    FundamentalsProvider,
    NewsProvider,
    RedditProvider,
)
from nlp_stock_prediction.providers._base import JsonResponse, ProviderTransportError
from nlp_stock_prediction.providers.news import PublicNewsProvider, PublicNewsProviderConfig
from nlp_stock_prediction.providers.reddit_scrape import (
    RedditPublicPageProvider,
    StaticHtmlTransport,
)
from nlp_stock_prediction.providers.sec_edgar import SecEdgarFundamentalsProvider

OFFLINE_FIXTURE_FETCHED_AT = datetime(2026, 5, 14, tzinfo=UTC)


@dataclass(frozen=True)
class ResearchFixtureProviderFactory:
    """Build offline providers from the repository fixture tree."""

    repo_root: Path
    fetched_at: datetime = OFFLINE_FIXTURE_FETCHED_AT

    def market_html_path(self, symbol: str) -> Path | None:
        return self.fixture_path(
            "raw",
            "candlecharts",
            f"public_ohlcv_{symbol.strip().lower()}.html",
        )

    def reddit_provider(self) -> RedditProvider | None:
        search_path = self.fixture_path("reddit", "public_search_tsla.html")
        discussion_path = self.fixture_path("reddit", "public_post_discussion.html")
        if search_path is None or discussion_path is None:
            return None
        return cast(
            RedditProvider,
            RedditPublicPageProvider(
                transport=StaticHtmlTransport(
                    {
                        "/search/": search_path.read_text(encoding="utf-8"),
                        "public001/daily_watch": discussion_path.read_text(encoding="utf-8"),
                    }
                ),
                discussion_page_limit=1,
                stale_after_seconds=7 * 86_400,
                now=lambda: self.fetched_at,
            ),
        )

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
                now=lambda: self.fetched_at,
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
            user_agent="nlp-stock-prediction fixture-runtime contact@example.test",
            transport=_StaticJsonTransport(
                {
                    "company_tickers_exchange": _company_tickers_exchange_payload(
                        cik="1318605",
                        ticker=symbol.upper(),
                        name="Tesla, Inc.",
                        exchange="Nasdaq",
                    ),
                    "companyfacts": cast(JsonObject, companyfacts),
                    "submissions": cast(JsonObject, submissions),
                }
            ),
            now=lambda: self.fetched_at,
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


def _company_tickers_exchange_payload(
    *,
    cik: str,
    ticker: str,
    name: str,
    exchange: str,
) -> JsonObject:
    return {
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [[int(cik), name, ticker, exchange]],
    }


__all__ = ["ResearchFixtureProviderFactory"]
