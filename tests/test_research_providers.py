from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError

import pytest

from nlp_stock_prediction.contracts import (
    CredentialState,
    EvidenceRequest,
    FreshnessStatus,
    FundamentalsRequest,
    MacroRequest,
    MarketDataRequest,
    ProviderResult,
    ProviderStatus,
    RetrievalMethod,
    SourceKind,
    TimeHorizon,
    WarningCode,
)
from nlp_stock_prediction.providers._base import (
    JsonResponse,
    MalformedProviderResponse,
    ProviderCache,
    ProviderTransportError,
    UrllibJsonTransport,
)
from nlp_stock_prediction.providers.execution import ProviderExecutionContext
from nlp_stock_prediction.providers.fred import FredMacroProvider
from nlp_stock_prediction.providers.market import (
    AlphaVantageFundamentalsProvider,
    AlphaVantageMarketDataProvider,
    YahooFinanceChartMarketDataProvider,
)
from nlp_stock_prediction.providers.news import PublicNewsProvider, PublicNewsProviderConfig
from nlp_stock_prediction.providers.sec_edgar import SecEdgarFundamentalsProvider
from nlp_stock_prediction.providers.social import XRecentSearchProvider, build_x_recent_search_query

RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "raw"


def _fixture(*parts: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((FIXTURE_ROOT.joinpath(*parts)).read_text(encoding="utf-8")),
    )


def _yahoo_chart_payload(
    *,
    timestamps: Sequence[datetime],
    opens: Sequence[str | None],
    highs: Sequence[str | None],
    lows: Sequence[str | None],
    closes: Sequence[str | None],
    adjusted_closes: Sequence[str | None],
    volumes: Sequence[int | None],
) -> dict[str, Any]:
    return {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "TSLA", "instrumentType": "EQUITY"},
                    "timestamp": [int(timestamp.timestamp()) for timestamp in timestamps],
                    "indicators": {
                        "quote": [
                            {
                                "open": list(opens),
                                "high": list(highs),
                                "low": list(lows),
                                "close": list(closes),
                                "volume": list(volumes),
                            }
                        ],
                        "adjclose": [{"adjclose": list(adjusted_closes)}],
                    },
                }
            ],
            "error": None,
        }
    }


@pytest.mark.unit
def test_provider_execution_context_preserves_fetch_identity_on_rate_limit() -> None:
    request = MarketDataRequest(
        request_id="provider-execution-rate-limit",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result: ProviderResult[object] = ProviderExecutionContext(
        provider_name="fixture-provider",
        request=request,
        fetched_at=FETCHED_AT,
        credential_state=CredentialState.CONFIGURED,
        raw_snapshot_id="raw-fixture-provider",
        cache_key="cache-fixture-provider",
    ).rate_limited("fixture provider quota exhausted")

    assert result.status == ProviderStatus.RATE_LIMITED
    assert result.raw_snapshot_id == "raw-fixture-provider"
    assert result.cache_key == "cache-fixture-provider"
    assert result.warnings[0].raw_snapshot_id == "raw-fixture-provider"


@dataclass
class _FakeJsonTransport:
    responses: dict[str, JsonResponse]
    calls: list[str] = field(default_factory=list)
    headers: list[Mapping[str, str] | None] = field(default_factory=list)

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> JsonResponse:
        del timeout
        self.calls.append(url)
        self.headers.append(headers)
        for url_fragment, response in self.responses.items():
            if url_fragment in url:
                return response
        raise AssertionError(f"Unexpected URL: {url}")


@dataclass
class _FailingJsonTransport:
    error: ProviderTransportError
    calls: list[str] = field(default_factory=list)
    headers: list[Mapping[str, str] | None] = field(default_factory=list)

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> JsonResponse:
        del timeout
        self.calls.append(url)
        self.headers.append(headers)
        raise self.error


@dataclass
class _MalformedJsonTransport:
    error: MalformedProviderResponse
    calls: list[str] = field(default_factory=list)
    headers: list[Mapping[str, str] | None] = field(default_factory=list)

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> JsonResponse:
        del timeout
        self.calls.append(url)
        self.headers.append(headers)
        raise self.error


@dataclass
class _SequencedJsonTransport:
    responses: tuple[JsonResponse, ...]
    calls: list[str] = field(default_factory=list)

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> JsonResponse:
        del headers, timeout
        self.calls.append(url)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


@pytest.mark.unit
def test_x_recent_search_builds_cashtag_query() -> None:
    assert build_x_recent_search_query("tsla") == "$TSLA lang:en -is:retweet"
    assert build_x_recent_search_query("NVDA", lang="en", exclude_retweets=False) == "$NVDA lang:en"


@pytest.mark.contract
def test_x_provider_maps_recent_search_posts_to_evidence() -> None:
    transport = _FakeJsonTransport(
        {"tweets/search/recent": JsonResponse(payload=_fixture("x", "recent_tsla.json"))}
    )
    provider = XRecentSearchProvider(
        bearer_token="fixture-token",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="x-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        limit=10,
    )

    result = provider.fetch_social_posts(request)

    assert result.status == ProviderStatus.OK
    assert result.health.credential_state == CredentialState.CONFIGURED
    assert result.data is not None
    assert len(result.data) == 1
    evidence = result.data[0]
    assert evidence.source_kind == SourceKind.X_POST
    assert evidence.ticker == "TSLA"
    assert evidence.text == "$TSLA call spreads into robotaxi catalyst. Risk stays defined."
    assert evidence.author_hash is not None
    assert evidence.author_hash != "raw-author-1"
    assert evidence.permalink == "https://x.com/i/web/status/1789000000000000001"
    assert evidence.matched_tickers == ("TSLA",)
    assert evidence.match_spans[0].text == "$TSLA"
    assert evidence.provenance.provider_name == "x-recent-search"
    assert evidence.provenance.retrieval_method == RetrievalMethod.OFFICIAL_API
    assert evidence.provenance.query == "$TSLA lang:en -is:retweet"
    assert evidence.provenance.provider_metadata["sort_order"] == "relevancy"
    assert evidence.provenance.raw_snapshot_id == result.raw_snapshot_id
    assert evidence.provenance.freshness_status == FreshnessStatus.FRESH
    assert "query=%24TSLA+lang%3Aen+-is%3Aretweet" in transport.calls[0]
    assert "sort_order=relevancy" in transport.calls[0]
    assert "max_results=10" in transport.calls[0]


@pytest.mark.contract
def test_x_provider_defaults_to_relevancy_and_fifty_posts() -> None:
    transport = _FakeJsonTransport(
        {"tweets/search/recent": JsonResponse(payload=_fixture("x", "recent_tsla.json"))}
    )
    provider = XRecentSearchProvider(
        bearer_token="fixture-token",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="x-tsla-defaults-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_social_posts(request)

    assert result.status == ProviderStatus.OK
    assert "sort_order=relevancy" in transport.calls[0]
    assert "max_results=50" in transport.calls[0]


@pytest.mark.contract
def test_x_provider_clamps_api_limit_and_slices_results_locally() -> None:
    payload = _fixture("x", "recent_tsla.json")
    first = cast(list[dict[str, object]], payload["data"])[0]
    payload["data"] = [
        {**first, "id": f"178900000000000000{index}", "text": f"$TSLA post {index}"}
        for index in range(12)
    ]
    transport = _FakeJsonTransport({"tweets/search/recent": JsonResponse(payload=payload)})
    provider = XRecentSearchProvider(
        bearer_token="fixture-token",
        transport=transport,
        now=lambda: FETCHED_AT,
    )

    result = provider.fetch_social_posts(
        EvidenceRequest(
            request_id="x-tsla-small-limit-2026-05-11",
            run_date=RUN_DATE,
            tickers=("TSLA",),
            limit=3,
        )
    )

    assert "max_results=10" in transport.calls[0]
    assert result.data is not None
    assert len(result.data) == 3


@pytest.mark.unit
def test_urllib_json_transport_classifies_wrapped_socket_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout_urlopen(*_args: object, **_kwargs: object) -> object:
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr("nlp_stock_prediction.providers._base.urlopen", timeout_urlopen)

    with pytest.raises(ProviderTransportError) as exc:
        UrllibJsonTransport().get_json("https://example.com/data.json")

    assert exc.value.error_type == "timeout"


@pytest.mark.contract
def test_x_provider_returns_unconfigured_warning_without_credentials() -> None:
    provider = XRecentSearchProvider(transport=_FakeJsonTransport({}), now=lambda: FETCHED_AT)
    request = EvidenceRequest(
        request_id="x-missing-token-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_social_posts(request)

    assert result.status == ProviderStatus.UNCONFIGURED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.MISSING_CREDENTIALS
    assert result.health.credential_state == CredentialState.MISSING


@pytest.mark.contract
def test_x_provider_returns_empty_without_query_or_ticker() -> None:
    transport = _FakeJsonTransport({})
    provider = XRecentSearchProvider(
        bearer_token="fixture-token",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="x-empty-query-2026-05-11",
        run_date=RUN_DATE,
        tickers=(),
    )

    result = provider.fetch_social_posts(request)

    assert result.status == ProviderStatus.EMPTY
    assert result.warnings[0].code == WarningCode.NO_DATA
    assert transport.calls == []


@pytest.mark.contract
def test_x_provider_treats_no_result_meta_as_empty() -> None:
    transport = _FakeJsonTransport(
        {"tweets/search/recent": JsonResponse(payload={"meta": {"result_count": 0}})}
    )
    provider = XRecentSearchProvider(
        bearer_token="fixture-token",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="x-no-results-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_social_posts(request)

    assert result.status == ProviderStatus.EMPTY
    assert result.warnings[0].code == WarningCode.NO_DATA


@pytest.mark.contract
def test_public_news_provider_uses_configured_mapping_and_normalizes_articles() -> None:
    config = PublicNewsProviderConfig(
        provider_name="fixture-news",
        endpoint="https://news.example.invalid/v1/search",
        api_key_param="token",
        query_param="search",
    )
    transport = _FakeJsonTransport(
        {"news.example.invalid/v1/search": JsonResponse(payload=_fixture("news", "tsla.json"))}
    )
    provider = PublicNewsProvider(
        config=config,
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="news-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        query="TSLA",
        limit=3,
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.OK
    assert result.data is not None
    article = result.data[0]
    assert article.source_kind == SourceKind.NEWS_ARTICLE
    assert article.title == "TSLA shares rise after robotaxi update"
    assert article.text.startswith("TSLA shares rose Monday")
    assert article.permalink == "https://news.example.invalid/tesla-robotaxi"
    assert article.provenance.provider_name == "fixture-news"
    assert article.provenance.query == "TSLA"
    assert article.provenance.provider_metadata["source_name"] == "Example Markets"
    assert "search=TSLA" in transport.calls[0]
    assert "token=fixture-key" in transport.calls[0]


@pytest.mark.contract
def test_public_news_provider_redacts_api_key_from_source_query_metadata() -> None:
    config = PublicNewsProviderConfig(
        provider_name="fixture-news",
        endpoint="https://news.example.invalid/v1/search",
        api_key_param="apiKey",
        query_param="search",
    )
    transport = _FakeJsonTransport(
        {"news.example.invalid/v1/search": JsonResponse(payload=_fixture("news", "tsla.json"))}
    )
    provider = PublicNewsProvider(
        config=config,
        api_key="super-secret-news-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="news-redaction-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        query="TSLA",
        limit=3,
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.OK
    assert result.data is not None
    source_query_url = result.data[0].provenance.provider_metadata["source_query_url"]
    assert isinstance(source_query_url, str)
    assert "super-secret-news-key" not in source_query_url
    assert "apiKey=REDACTED" in source_query_url
    assert "super-secret-news-key" in transport.calls[0]


@pytest.mark.contract
def test_public_news_provider_returns_unconfigured_warning_for_required_key() -> None:
    provider = PublicNewsProvider(
        config=PublicNewsProviderConfig(provider_name="fixture-news"),
        transport=_FakeJsonTransport({}),
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="news-unconfigured-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.UNCONFIGURED
    assert result.warnings[0].code == WarningCode.MISSING_CREDENTIALS
    assert result.health.credential_state == CredentialState.MISSING


@pytest.mark.contract
def test_public_news_provider_returns_no_data_warning_for_empty_articles() -> None:
    config = PublicNewsProviderConfig(
        provider_name="fixture-news",
        endpoint="https://news.example.invalid/v1/search",
        api_key_param="token",
        query_param="search",
    )
    transport = _FakeJsonTransport(
        {"news.example.invalid/v1/search": JsonResponse(payload={"articles": []})}
    )
    provider = PublicNewsProvider(
        config=config,
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="news-empty-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        query="TSLA",
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.EMPTY
    assert result.data is None
    assert result.warnings[0].code == WarningCode.NO_DATA
    assert "returned no articles" in result.warnings[0].message
    assert result.health.status == ProviderStatus.EMPTY


@pytest.mark.contract
def test_public_news_provider_does_not_attribute_unmatched_articles_to_requested_ticker() -> None:
    config = PublicNewsProviderConfig(
        provider_name="fixture-news",
        endpoint="https://news.example.invalid/v1/search",
        api_key_param="token",
        query_param="search",
    )
    payload = {
        "articles": [
            {
                "title": "Copper miners rally on supply concerns",
                "description": "The article never mentions the requested symbol.",
                "url": "https://news.example.invalid/copper",
                "publishedAt": FETCHED_AT.isoformat(),
            }
        ]
    }
    transport = _FakeJsonTransport(
        {"news.example.invalid/v1/search": JsonResponse(payload=payload)}
    )
    provider = PublicNewsProvider(
        config=config,
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="news-unmatched-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        query="TSLA",
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.EMPTY
    assert result.data is None
    assert result.warnings[0].code == WarningCode.NO_DATA


@pytest.mark.contract
def test_x_provider_does_not_attribute_unmatched_posts_to_requested_ticker() -> None:
    payload = {
        "data": [
            {
                "id": "1789000000000000999",
                "text": "Copper miners rally on supply concerns.",
                "created_at": FETCHED_AT.isoformat(),
                "public_metrics": {"like_count": 4},
            }
        ]
    }
    transport = _FakeJsonTransport({"tweets/search/recent": JsonResponse(payload=payload)})
    provider = XRecentSearchProvider(
        bearer_token="fixture-token",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="x-unmatched-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_social_posts(request)

    assert result.status == ProviderStatus.EMPTY
    assert result.data is None
    assert result.warnings[0].code == WarningCode.NO_DATA


@pytest.mark.contract
def test_public_news_provider_maps_rate_limit_transport_failure() -> None:
    transport = _FailingJsonTransport(
        ProviderTransportError(
            "fixture news quota exhausted",
            status_code=429,
            retryable=True,
            error_type="http_error",
        )
    )
    provider = PublicNewsProvider(
        config=PublicNewsProviderConfig(provider_name="fixture-news"),
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="news-rate-limited-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.RATE_LIMITED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.RATE_LIMITED
    assert result.warnings[0].retryable is True
    assert result.warnings[0].provider_status_code == 429
    assert result.warnings[0].provider_error_type == "http_error"
    assert result.health.status == ProviderStatus.RATE_LIMITED
    assert result.health.rate_limit_remaining == 0
    assert len(transport.calls) == 3


@pytest.mark.contract
def test_public_news_provider_maps_upstream_unavailable_transport_failure() -> None:
    transport = _FailingJsonTransport(
        ProviderTransportError(
            "fixture news upstream unavailable",
            status_code=503,
            retryable=True,
            error_type="http_error",
        )
    )
    provider = PublicNewsProvider(
        config=PublicNewsProviderConfig(provider_name="fixture-news"),
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="news-upstream-unavailable-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.FAILED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.UPSTREAM_UNAVAILABLE
    assert result.warnings[0].retryable is True
    assert result.warnings[0].provider_status_code == 503
    assert result.warnings[0].provider_error_type == "http_error"
    assert result.health.status == ProviderStatus.FAILED
    assert len(transport.calls) == 3


@pytest.mark.contract
def test_public_news_provider_maps_timeout_transport_failure() -> None:
    transport = _FailingJsonTransport(
        ProviderTransportError(
            "fixture news timed out",
            retryable=True,
            error_type="timeout",
        )
    )
    provider = PublicNewsProvider(
        config=PublicNewsProviderConfig(provider_name="fixture-news"),
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="news-timeout-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.FAILED
    assert result.warnings[0].code == WarningCode.TIMEOUT
    assert result.warnings[0].provider_error_type == "timeout"
    assert result.health.status == ProviderStatus.FAILED


@pytest.mark.contract
def test_public_news_provider_maps_malformed_transport_without_snapshot() -> None:
    transport = _MalformedJsonTransport(MalformedProviderResponse("provider returned invalid JSON"))
    provider = PublicNewsProvider(
        config=PublicNewsProviderConfig(provider_name="fixture-news"),
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = EvidenceRequest(
        request_id="news-malformed-json-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.MALFORMED
    assert result.data is None
    assert result.raw_snapshot_id is None
    assert result.cache_key is not None
    assert result.warnings[0].code == WarningCode.MALFORMED_RESPONSE
    assert len(transport.calls) == 1


@pytest.mark.contract
def test_alpha_vantage_daily_candles_map_and_reuse_cache(tmp_path: Path) -> None:
    transport = _FakeJsonTransport(
        {
            "TIME_SERIES_DAILY_ADJUSTED": JsonResponse(
                payload=_fixture("alpha_vantage", "daily_tsla.json")
            )
        }
    )
    provider = AlphaVantageMarketDataProvider(
        api_key="fixture-key",
        transport=transport,
        cache=ProviderCache(tmp_path),
        now=lambda: FETCHED_AT,
    )
    request = MarketDataRequest(
        request_id="market-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    first = provider.fetch_daily_candles(request)
    second = provider.fetch_daily_candles(request)

    assert first.status == ProviderStatus.OK
    assert first.data is not None
    assert first.data.ticker == "TSLA"
    assert first.data.bars[0].timestamp == RUN_DATE
    assert first.data.bars[0].close == Decimal("184.25")
    assert first.data.bars[0].volume == 123_456_789
    assert first.data.liquidity_metrics[0].name == "average_volume"
    assert first.cache_key == second.cache_key
    assert first.raw_snapshot_id == second.raw_snapshot_id
    assert len(transport.calls) == 1
    assert tmp_path.joinpath("2026-05-11", "TSLA", "alpha-vantage-daily").exists()


@pytest.mark.contract
def test_alpha_vantage_daily_candles_marks_stale_market_data() -> None:
    payload: dict[str, Any] = {
        "Time Series (Daily)": {
            "2026-04-28": {
                "1. open": "181.00",
                "2. high": "186.00",
                "3. low": "180.50",
                "4. close": "184.25",
                "5. adjusted close": "184.25",
                "6. volume": "123456789",
            }
        }
    }
    transport = _FakeJsonTransport({"TIME_SERIES_DAILY_ADJUSTED": JsonResponse(payload=payload)})
    provider = AlphaVantageMarketDataProvider(
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
        stale_after_days=5,
    )
    request = MarketDataRequest(
        request_id="market-stale-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_daily_candles(request)

    assert result.status == ProviderStatus.STALE
    assert result.data is not None
    assert result.data.bars[0].timestamp == date(2026, 4, 28)
    assert result.warnings[0].code == WarningCode.STALE_DATA
    assert result.warnings[0].metadata["latest_date"] == "2026-04-28"
    assert result.health.status == ProviderStatus.STALE


@pytest.mark.contract
def test_alpha_vantage_daily_candles_reports_malformed_volume() -> None:
    payload: dict[str, Any] = {
        "Time Series (Daily)": {
            "2026-05-11": {
                "1. open": "181.00",
                "2. high": "186.00",
                "3. low": "180.50",
                "4. close": "184.25",
                "5. adjusted close": "184.25",
                "6. volume": "not-a-number",
            }
        }
    }
    transport = _FakeJsonTransport({"TIME_SERIES_DAILY_ADJUSTED": JsonResponse(payload=payload)})
    provider = AlphaVantageMarketDataProvider(
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = MarketDataRequest(
        request_id="market-tsla-malformed-volume-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_daily_candles(request)

    assert result.status == ProviderStatus.MALFORMED
    assert result.data is None
    assert result.raw_snapshot_id is not None
    assert result.cache_key is not None
    assert result.warnings[0].code == WarningCode.MALFORMED_RESPONSE
    assert "invalid OHLCV" in result.warnings[0].message


@pytest.mark.contract
def test_alpha_vantage_daily_candles_does_not_cache_provider_note_payloads(
    tmp_path: Path,
) -> None:
    valid_payload = _fixture("alpha_vantage", "daily_tsla.json")
    transport = _SequencedJsonTransport(
        (
            JsonResponse(payload={"Note": "API call frequency exceeded."}),
            JsonResponse(payload=valid_payload),
        )
    )
    provider = AlphaVantageMarketDataProvider(
        api_key="fixture-key",
        transport=transport,
        cache=ProviderCache(tmp_path),
        now=lambda: FETCHED_AT,
    )
    request = MarketDataRequest(
        request_id="market-tsla-note-cache-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    first = provider.fetch_daily_candles(request)
    second = provider.fetch_daily_candles(request)

    assert first.status == ProviderStatus.RATE_LIMITED
    assert first.data is None
    assert second.status == ProviderStatus.OK
    assert second.data is not None
    assert len(transport.calls) == 2


@pytest.mark.contract
def test_alpha_vantage_market_provider_returns_missing_credentials_warning() -> None:
    provider = AlphaVantageMarketDataProvider(
        transport=_FakeJsonTransport({}),
        now=lambda: FETCHED_AT,
    )
    request = MarketDataRequest(
        request_id="market-missing-key-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_daily_candles(request)

    assert result.status == ProviderStatus.UNCONFIGURED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.MISSING_CREDENTIALS
    assert result.warnings[0].metadata["credential_name"] == "Alpha Vantage API key"
    assert result.health.credential_state == CredentialState.MISSING


@pytest.mark.contract
def test_yahoo_finance_chart_daily_candles_map_and_reuse_cache(tmp_path: Path) -> None:
    payload = _yahoo_chart_payload(
        timestamps=(
            datetime(2026, 5, 11, 13, 30, tzinfo=UTC),
            datetime(2026, 5, 10, 13, 30, tzinfo=UTC),
        ),
        opens=("181.00", "179.00"),
        highs=("186.00", "182.00"),
        lows=("180.50", "178.50"),
        closes=("184.25", "181.75"),
        adjusted_closes=("184.10", "181.60"),
        volumes=(123_456_789, 98_765_432),
    )
    transport = _FakeJsonTransport({"finance/chart/TSLA": JsonResponse(payload=payload)})
    provider = YahooFinanceChartMarketDataProvider(
        transport=transport,
        cache=ProviderCache(tmp_path),
        now=lambda: FETCHED_AT,
    )
    request = MarketDataRequest(
        request_id="market-yahoo-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    first = provider.fetch_daily_candles(request)
    second = provider.fetch_daily_candles(request)

    assert first.status == ProviderStatus.OK
    assert first.health.credential_state == CredentialState.NOT_REQUIRED
    assert first.data is not None
    assert first.data.ticker == "TSLA"
    assert first.data.bars[0].timestamp == RUN_DATE
    assert first.data.bars[0].close == Decimal("184.25")
    assert first.data.bars[0].adjusted_close == Decimal("184.10")
    assert first.data.bars[0].volume == 123_456_789
    assert first.data.liquidity_metrics[0].name == "average_volume"
    assert first.cache_key == second.cache_key
    assert first.raw_snapshot_id == second.raw_snapshot_id
    assert len(transport.calls) == 1
    assert "period1=" in transport.calls[0]
    assert "period2=" in transport.calls[0]
    assert transport.headers[0] is not None
    assert transport.headers[0]["Accept"] == "application/json"
    assert tmp_path.joinpath("2026-05-11", "TSLA", "yahoo-finance-chart-daily").exists()


@pytest.mark.contract
def test_yahoo_finance_chart_daily_candles_returns_partial_for_incomplete_rows() -> None:
    payload = _yahoo_chart_payload(
        timestamps=(
            datetime(2026, 5, 11, 13, 30, tzinfo=UTC),
            datetime(2026, 5, 10, 13, 30, tzinfo=UTC),
        ),
        opens=("181.00", None),
        highs=("186.00", "182.00"),
        lows=("180.50", "178.50"),
        closes=("184.25", "181.75"),
        adjusted_closes=("184.10", None),
        volumes=(123_456_789, 98_765_432),
    )
    transport = _FakeJsonTransport({"finance/chart/TSLA": JsonResponse(payload=payload)})
    provider = YahooFinanceChartMarketDataProvider(
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = MarketDataRequest(
        request_id="market-yahoo-partial-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_daily_candles(request)

    assert result.status == ProviderStatus.PARTIAL
    assert result.data is not None
    assert len(result.data.bars) == 1
    assert result.warnings[0].code == WarningCode.PARTIAL_DATA
    assert result.warnings[0].metadata["skipped_row_count"] == 1


@pytest.mark.contract
def test_alpha_vantage_company_overview_maps_fundamental_metrics() -> None:
    transport = _FakeJsonTransport(
        {"OVERVIEW": JsonResponse(payload=_fixture("alpha_vantage", "overview_tsla.json"))}
    )
    provider = AlphaVantageFundamentalsProvider(
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = FundamentalsRequest(
        request_id="fundamentals-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_fundamentals(request)

    assert result.status == ProviderStatus.OK
    assert result.data is not None
    assert result.data.company_name == "Tesla, Inc."
    metrics = {metric.name: metric for metric in result.data.metrics}
    assert metrics["market_cap"].value == Decimal("575000000000")
    assert metrics["pe_ratio"].value == Decimal("48.5")
    assert metrics["sector"].value == "Consumer Cyclical"
    assert metrics["market_cap"].as_of == date(2026, 3, 31)


@pytest.mark.contract
def test_alpha_vantage_fundamentals_provider_returns_missing_credentials_warning() -> None:
    provider = AlphaVantageFundamentalsProvider(
        transport=_FakeJsonTransport({}),
        now=lambda: FETCHED_AT,
    )
    request = FundamentalsRequest(
        request_id="fundamentals-missing-key-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_fundamentals(request)

    assert result.status == ProviderStatus.UNCONFIGURED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.MISSING_CREDENTIALS
    assert result.warnings[0].metadata["credential_name"] == "Alpha Vantage API key"
    assert result.health.credential_state == CredentialState.MISSING


@pytest.mark.contract
def test_sec_edgar_maps_company_facts_and_recent_filings() -> None:
    transport = _FakeJsonTransport(
        {
            "companyfacts": JsonResponse(payload=_fixture("sec_edgar", "companyfacts_tsla.json")),
            "submissions": JsonResponse(payload=_fixture("sec_edgar", "submissions_tsla.json")),
        }
    )
    provider = SecEdgarFundamentalsProvider(
        ticker_cik_map={"TSLA": "1318605"},
        user_agent="nlp-stock-prediction-test contact@example.test",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = FundamentalsRequest(
        request_id="sec-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_fundamentals(request)

    assert result.status == ProviderStatus.OK
    assert result.data is not None
    assert result.data.company_name == "Tesla, Inc."
    metrics = {metric.name: metric for metric in result.data.metrics}
    assert metrics["sec_revenues"].value == Decimal("21301000000")
    assert metrics["sec_revenues"].metadata["form"] == "10-Q"
    assert metrics["sec_recent_filing_10_q"].value == "10-Q"
    assert metrics["sec_recent_filing_10_q"].metadata["accession_number"] == (
        "0001628280-26-012345"
    )
    assert transport.headers[0] is not None
    assert "User-Agent" in transport.headers[0]


@pytest.mark.contract
def test_sec_edgar_warns_when_ticker_cik_mapping_is_unconfigured() -> None:
    provider = SecEdgarFundamentalsProvider(
        ticker_cik_map={},
        user_agent="nlp-stock-prediction-test contact@example.test",
        transport=_FakeJsonTransport({}),
        now=lambda: FETCHED_AT,
    )
    request = FundamentalsRequest(
        request_id="sec-no-cik-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_fundamentals(request)

    assert result.status == ProviderStatus.UNCONFIGURED
    assert result.warnings[0].code == WarningCode.NO_DATA


@pytest.mark.contract
def test_fred_macro_provider_maps_series_and_emits_stale_warning() -> None:
    transport = _FakeJsonTransport(
        {"series/observations": JsonResponse(payload=_fixture("fred", "fedfunds_stale.json"))}
    )
    provider = FredMacroProvider(
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
        stale_after_days=30,
    )
    request = MacroRequest(
        request_id="fred-2026-05-11",
        run_date=RUN_DATE,
        series_ids=("FEDFUNDS",),
        horizon=TimeHorizon.MONTHLY,
    )

    result = provider.fetch_macro(request)

    assert result.status == ProviderStatus.STALE
    assert result.warnings[0].code == WarningCode.STALE_DATA
    assert result.data is not None
    series = result.data.series[0]
    assert series.series_id == "FEDFUNDS"
    assert series.name == "Effective Federal Funds Rate"
    assert series.values[0].name == "observation"
    assert series.values[0].value == Decimal("4.25")
    assert series.values[0].as_of == date(2026, 3, 1)


@pytest.mark.contract
def test_fred_macro_provider_converts_mapping_failures_to_partial_warning() -> None:
    transport = _FakeJsonTransport(
        {
            "series_id=FEDFUNDS": JsonResponse(payload=_fixture("fred", "fedfunds_stale.json")),
            "series_id=UNRATE": JsonResponse(payload={"observations": {"not": "a list"}}),
        }
    )
    provider = FredMacroProvider(
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
        stale_after_days=365,
    )
    request = MacroRequest(
        request_id="fred-partial-malformed-2026-05-11",
        run_date=RUN_DATE,
        series_ids=("FEDFUNDS", "UNRATE"),
    )

    result = provider.fetch_macro(request)

    assert result.status == ProviderStatus.PARTIAL
    assert result.data is not None
    assert [series.series_id for series in result.data.series] == ["FEDFUNDS"]
    assert len(result.warnings) == 1
    assert result.warnings[0].code == WarningCode.MALFORMED_RESPONSE
    assert result.warnings[0].metadata["series_id"] == "UNRATE"


@pytest.mark.contract
def test_fred_macro_provider_returns_warning_result_when_all_mapping_fails() -> None:
    transport = _FakeJsonTransport({"series_id=UNRATE": JsonResponse(payload={"observations": []})})
    provider = FredMacroProvider(
        api_key="fixture-key",
        transport=transport,
        now=lambda: FETCHED_AT,
    )
    request = MacroRequest(
        request_id="fred-malformed-2026-05-11",
        run_date=RUN_DATE,
        series_ids=("UNRATE",),
    )

    result = provider.fetch_macro(request)

    assert result.status == ProviderStatus.MALFORMED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.MALFORMED_RESPONSE
    assert result.warnings[0].metadata["series_id"] == "UNRATE"


@pytest.mark.contract
def test_fred_macro_provider_returns_missing_credentials_warning() -> None:
    provider = FredMacroProvider(transport=_FakeJsonTransport({}), now=lambda: FETCHED_AT)
    request = MacroRequest(
        request_id="fred-missing-key-2026-05-11",
        run_date=RUN_DATE,
        series_ids=("FEDFUNDS",),
    )

    result = provider.fetch_macro(request)

    assert result.status == ProviderStatus.UNCONFIGURED
    assert result.warnings[0].code == WarningCode.MISSING_CREDENTIALS
    assert result.health.credential_state == CredentialState.MISSING
