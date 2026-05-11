from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.contracts import (
    CredentialState,
    EvidenceRequest,
    FreshnessStatus,
    FundamentalsRequest,
    MacroRequest,
    MarketDataRequest,
    ProviderStatus,
    RetrievalMethod,
    SourceKind,
    TimeHorizon,
    WarningCode,
)
from nlp_stock_prediction.providers._base import JsonResponse, ProviderCache
from nlp_stock_prediction.providers.fred import FredMacroProvider
from nlp_stock_prediction.providers.market import (
    AlphaVantageFundamentalsProvider,
    AlphaVantageMarketDataProvider,
)
from nlp_stock_prediction.providers.news import PublicNewsProvider, PublicNewsProviderConfig
from nlp_stock_prediction.providers.sec_edgar import SecEdgarFundamentalsProvider
from nlp_stock_prediction.providers.social import XRecentSearchProvider, build_x_recent_search_query

RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "raw"


def _fixture(*parts: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((FIXTURE_ROOT.joinpath(*parts)).read_text()))


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


@pytest.mark.unit
def test_x_recent_search_builds_cashtag_query() -> None:
    assert build_x_recent_search_query("tsla") == "$TSLA lang:en"
    assert build_x_recent_search_query("NVDA", lang="en", exclude_retweets=True) == (
        "$NVDA lang:en -is:retweet"
    )


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
    assert evidence.provenance.query == "$TSLA lang:en"
    assert evidence.provenance.raw_snapshot_id == result.raw_snapshot_id
    assert evidence.provenance.freshness_status == FreshnessStatus.FRESH
    assert "query=%24TSLA+lang%3Aen" in transport.calls[0]


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
    assert article.title == "Tesla shares rise after robotaxi update"
    assert article.text.startswith("Tesla shares rose Monday")
    assert article.permalink == "https://news.example.invalid/tesla-robotaxi"
    assert article.provenance.provider_name == "fixture-news"
    assert article.provenance.query == "TSLA"
    assert article.provenance.provider_metadata["source_name"] == "Example Markets"
    assert "search=TSLA" in transport.calls[0]
    assert "token=fixture-key" in transport.calls[0]


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
    assert result.warnings[0].code == WarningCode.MALFORMED_RESPONSE
    assert "invalid OHLCV" in result.warnings[0].message


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

    assert result.status == ProviderStatus.FAILED
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
