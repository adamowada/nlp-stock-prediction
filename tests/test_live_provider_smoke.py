from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from types import TracebackType
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from nlp_stock_prediction.contracts import (
    EvidenceRequest,
    MarketDataRequest,
    ProviderStatus,
    TickerDiscoveryRequest,
)
from nlp_stock_prediction.providers.apnews import APNewsProvider
from nlp_stock_prediction.providers.candlecharts import CandlechartsMarketDataProvider
from nlp_stock_prediction.providers.market import YahooFinanceChartMarketDataProvider
from nlp_stock_prediction.providers.reddit_scrape import RedditPublicPageProvider
from nlp_stock_prediction.providers.social import XRecentSearchProvider

ALLOW_LIVE_ENV = "NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS"
LIVE_USER_AGENT_ENV = "NLP_STOCK_PREDICTION_LIVE_USER_AGENT"
LIVE_SCRAPE_URL_ENV = "NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL"
LIVE_SCRAPE_EXPECT_TEXT_ENV = "NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT"
X_BEARER_TOKEN_ENV = "NLP_STOCK_PREDICTION_X_BEARER_TOKEN"


@dataclass(frozen=True)
class LiveHttpResponse:
    status: int
    body: bytes


class _LiveHttpContext(Protocol):
    status: int

    def read(self, limit: int) -> bytes: ...

    def __enter__(self) -> _LiveHttpContext: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _UrlOpen(Protocol):
    def __call__(self, request: Request, *, timeout: float) -> _LiveHttpContext: ...


def _env_value(name: str, env: Mapping[str, str] | None = None) -> str | None:
    value = (env or os.environ).get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _live_tests_enabled(env: Mapping[str, str] | None = None) -> bool:
    return _env_value(ALLOW_LIVE_ENV, env) == "1"


def _require_live_tests_enabled(kind: str, env: Mapping[str, str] | None = None) -> None:
    if not _live_tests_enabled(env):
        pytest.skip(f"Set {ALLOW_LIVE_ENV}=1 to run {kind} smoke checks.")


def _require_env(name: str, purpose: str, env: Mapping[str, str] | None = None) -> str:
    value = _env_value(name, env)
    if value is None:
        pytest.fail(f"Live smoke configuration missing: set {name} to {purpose}.")
    return value


def _require_http_url(url: str, *, env_name: str) -> None:
    if not url.startswith(("https://", "http://")):
        pytest.fail(
            f"{env_name} must be an http:// or https:// URL for the live scraping smoke check."
        )


def _fetch_live_url(
    request: Request,
    *,
    smoke_name: str,
    byte_limit: int,
    timeout_seconds: float = 10.0,
    opener: _UrlOpen | None = None,
) -> LiveHttpResponse:
    active_opener = opener or cast(_UrlOpen, urlopen)
    try:
        with active_opener(request, timeout=timeout_seconds) as response:
            return LiveHttpResponse(status=response.status, body=response.read(byte_limit))
    except HTTPError as exc:
        pytest.fail(
            f"{smoke_name} live smoke received HTTP {exc.code} from {request.full_url}; "
            "verify the configured URL, credentials, quota, and upstream availability."
        )
    except (URLError, TimeoutError, OSError) as exc:
        pytest.fail(
            f"{smoke_name} live smoke could not reach {request.full_url}: {exc}; "
            "verify network access, configured credentials, quota, and upstream availability."
        )


@pytest.mark.live_api
def test_live_sec_company_tickers_official_api_smoke() -> None:
    _require_live_tests_enabled("live API")
    user_agent = _require_env(
        LIVE_USER_AGENT_ENV,
        "a contact User-Agent for SEC requests, for example 'your-name your-email@example.com'",
    )
    request = Request(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )

    response = _fetch_live_url(request, smoke_name="SEC company tickers", byte_limit=2_000_000)
    assert response.status == 200, "SEC company_tickers live smoke expected HTTP 200."
    try:
        payload = json.loads(response.body)
    except json.JSONDecodeError as exc:
        pytest.fail(f"SEC company_tickers response was not valid JSON: {exc}.")
    assert isinstance(payload, dict), "SEC company_tickers response must be a JSON object."
    records = list(payload.values())
    assert records, "SEC company_tickers response did not include any company records."
    first_record = records[0]
    assert isinstance(first_record, dict), (
        "SEC company_tickers company record must be a JSON object."
    )
    assert {"ticker", "title", "cik_str"}.issubset(first_record), (
        "SEC company_tickers record did not include the expected ticker/title/cik_str fields."
    )
    assert isinstance(first_record["ticker"], str) and first_record["ticker"].strip(), (
        "SEC company_tickers record included an empty ticker."
    )


@pytest.mark.live_scraping
def test_live_public_scraping_configured_url_smoke() -> None:
    _require_live_tests_enabled("live scraping")
    scrape_url = _require_env(
        LIVE_SCRAPE_URL_ENV,
        "the narrow public page to smoke-test, for example 'https://example.com/'",
    )
    expected_text = _require_env(
        LIVE_SCRAPE_EXPECT_TEXT_ENV,
        "literal page text expected in the configured scraping smoke response",
    )
    _require_http_url(scrape_url, env_name=LIVE_SCRAPE_URL_ENV)
    request = Request(scrape_url, headers={"User-Agent": "nlp-stock-prediction-live-smoke/0.1"})

    response = _fetch_live_url(request, smoke_name="configured public scraping", byte_limit=2048)
    body_text = response.body.decode("utf-8", errors="replace")

    assert response.status < 500, "Configured scraping smoke endpoint returned a server error."
    assert expected_text in body_text, (
        f"Configured scraping smoke endpoint did not include expected text from "
        f"{LIVE_SCRAPE_EXPECT_TEXT_ENV}."
    )


@pytest.mark.live_api
def test_live_x_recent_search_smoke() -> None:
    _require_live_tests_enabled("X recent-search live API")
    bearer_token = _require_env(
        X_BEARER_TOKEN_ENV,
        "an X app-only Bearer Token for recent-search smoke coverage",
    )
    provider = XRecentSearchProvider(bearer_token=bearer_token)

    result = provider.fetch_social_posts(
        EvidenceRequest(
            request_id="live-x-aapl-smoke",
            run_date=date(2026, 5, 11),
            tickers=("AAPL",),
        )
    )

    assert result.status in {ProviderStatus.OK, ProviderStatus.EMPTY, ProviderStatus.STALE}
    assert result.health.credential_state.value == "configured"


@pytest.mark.live_api
def test_live_yahoo_finance_chart_market_data_smoke() -> None:
    _require_live_tests_enabled("Yahoo Finance chart live API")
    provider = YahooFinanceChartMarketDataProvider()

    result = provider.fetch_daily_candles(
        MarketDataRequest(
            request_id="live-yahoo-aapl-smoke",
            run_date=date(2026, 5, 13),
            tickers=("AAPL",),
        )
    )

    if result.status not in {ProviderStatus.OK, ProviderStatus.PARTIAL, ProviderStatus.STALE}:
        pytest.fail(
            "Yahoo Finance chart live market-data smoke did not return usable daily candles: "
            + "; ".join(warning.message for warning in result.warnings)
        )
    assert result.data is not None
    assert result.data.bars
    assert result.health.credential_state.value == "not_required"


@pytest.mark.live_scraping
def test_live_reddit_public_page_shape_smoke() -> None:
    _require_live_tests_enabled("Reddit public-page live scraping")
    provider = RedditPublicPageProvider(allow_live_scraping=True)

    result = provider.discover_tickers(
        TickerDiscoveryRequest(
            request_id="live-reddit-wsb-smoke",
            run_date=date(2026, 5, 11),
            source_url="https://www.reddit.com/r/wallstreetbets/",
        )
    )

    if result.status not in {ProviderStatus.OK, ProviderStatus.PARTIAL}:
        pytest.fail(
            "Reddit public-page live scraping smoke did not return usable ticker output: "
            + "; ".join(warning.message for warning in result.warnings)
        )
    assert result.data is not None
    assert result.data.candidates


@pytest.mark.live_scraping
def test_live_apnews_public_hub_shape_smoke() -> None:
    _require_live_tests_enabled("AP News public-page live scraping")
    provider = APNewsProvider()

    result = provider.fetch_articles(
        EvidenceRequest(
            request_id="live-apnews-financial-markets-smoke",
            run_date=date(2026, 5, 11),
            tickers=("AAPL",),
            limit=1,
        )
    )

    if result.status not in {ProviderStatus.OK, ProviderStatus.PARTIAL}:
        pytest.fail(
            "AP News public hub live scraping smoke did not return usable evidence: "
            + "; ".join(warning.message for warning in result.warnings)
        )
    assert result.data is not None
    assert result.data


@pytest.mark.live_scraping
def test_live_candlecharts_feasibility_shape_smoke() -> None:
    _require_live_tests_enabled("Candlecharts public-page live scraping")
    provider = CandlechartsMarketDataProvider(allow_live=True)

    result = provider.fetch_daily_candles(
        MarketDataRequest(
            request_id="live-candlecharts-aapl-smoke",
            run_date=date(2026, 5, 11),
            tickers=("AAPL",),
        )
    )

    if result.status not in {ProviderStatus.OK, ProviderStatus.PARTIAL}:
        pytest.fail(
            "Candlecharts public-page live scraping smoke did not return usable candles: "
            + "; ".join(warning.message for warning in result.warnings)
        )
    assert result.data is not None
    assert result.data.bars


@pytest.mark.unit
def test_live_smoke_requires_explicit_global_opt_in() -> None:
    with pytest.raises(pytest.skip.Exception, match=ALLOW_LIVE_ENV):
        _require_live_tests_enabled("fixture", env={})


@pytest.mark.unit
def test_live_smoke_fails_on_missing_named_env_after_opt_in() -> None:
    env = {ALLOW_LIVE_ENV: "1"}

    with pytest.raises(pytest.fail.Exception, match=LIVE_USER_AGENT_ENV):
        _require_env(LIVE_USER_AGENT_ENV, "fixture value", env=env)


@pytest.mark.unit
def test_fetch_live_url_reads_bounded_response() -> None:
    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            return None

        def read(self, limit: int) -> bytes:
            assert limit == 6
            return b"ticker"

    def fake_opener(request: Request, *, timeout: float) -> FakeResponse:
        assert request.full_url == "https://example.com/"
        assert timeout == 1.5
        return FakeResponse()

    response = _fetch_live_url(
        Request("https://example.com/"),
        smoke_name="fixture",
        byte_limit=6,
        timeout_seconds=1.5,
        opener=fake_opener,
    )

    assert response == LiveHttpResponse(status=200, body=b"ticker")


@pytest.mark.unit
def test_fetch_live_url_failure_message_is_actionable() -> None:
    def fake_opener(request: Request, *, timeout: float) -> _LiveHttpContext:
        raise URLError("no route to host")

    with pytest.raises(pytest.fail.Exception, match="verify network access"):
        _fetch_live_url(
            Request("https://example.com/"),
            smoke_name="fixture",
            byte_limit=6,
            opener=fake_opener,
        )
