from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import NoReturn

import pytest

from nlp_stock_prediction.contracts import (
    CredentialState,
    MarketDataRequest,
    MarketSnapshot,
    PriceBar,
    ProviderResult,
    ProviderStatus,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.orchestration.live_market_data import (
    FallbackMarketDataProvider,
    LiveMarketDataSelection,
    LiveMarketDataSelector,
)
from nlp_stock_prediction.orchestration.phase4_common import retrieval_method_for_provider
from nlp_stock_prediction.orchestration.phase4_live_providers import Phase4LiveProviderFactory
from nlp_stock_prediction.providers._base import provider_result, provider_warning
from nlp_stock_prediction.providers.market import YahooFinanceChartMarketDataProvider

NOW = datetime(2026, 5, 15, 18, 0, tzinfo=UTC)


@pytest.mark.unit
def test_phase4_live_provider_factory_uses_public_yahoo_market_data_without_alpha_key() -> None:
    factory = Phase4LiveProviderFactory(env={})

    provider = factory.market_data_provider("aapl")
    source_url = factory.market_data_source_query_url("aapl")

    assert isinstance(provider, YahooFinanceChartMarketDataProvider)
    assert provider.provider_name == "yahoo-finance-chart"
    assert retrieval_method_for_provider(provider.provider_name).value == "public_scrape"
    assert source_url is not None
    assert "query1.finance.yahoo.com/v8/finance/chart/AAPL" in source_url
    assert "apikey" not in source_url.lower()
    assert "fixture" not in source_url.lower()


@pytest.mark.unit
def test_phase4_live_provider_factory_uses_shared_market_data_selector() -> None:
    factory = Phase4LiveProviderFactory(env={})
    selector = LiveMarketDataSelector(env={})
    selection = selector.primary_selection("aapl")

    assert factory.market_data_provider("aapl").provider_name == selection.provider.provider_name
    assert factory.market_data_source_query_url("aapl") == selection.source_url


@pytest.mark.unit
def test_live_market_data_selector_uses_public_fallbacks_when_alpha_key_is_configured() -> None:
    selection = LiveMarketDataSelector(env={"NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY": "key"})
    primary = selection.primary_selection("aapl")

    assert isinstance(primary.provider, FallbackMarketDataProvider)
    assert primary.source_url is not None
    assert "alphavantage.co/query" in str(primary.source_url)
    assert primary.provider.source_url_for_provider("yahoo-finance-chart") is not None
    assert "query1.finance.yahoo.com/v8/finance/chart/AAPL" in str(
        primary.provider.source_url_for_provider("yahoo-finance-chart")
    )


@pytest.mark.unit
def test_fallback_market_data_provider_uses_yahoo_after_alpha_rate_limit() -> None:
    request = MarketDataRequest(
        request_id="market-aapl",
        run_date=date(2026, 5, 15),
        tickers=("AAPL",),
        query="AAPL",
    )
    alpha = _StaticProvider(
        "alpha-vantage-market-data",
        _provider_result(
            provider_name="alpha-vantage-market-data",
            request=request,
            status=ProviderStatus.RATE_LIMITED,
            data=None,
            warning_code=WarningCode.RATE_LIMITED,
            message="Alpha Vantage response indicates rate limit or notice",
            credential_state=CredentialState.CONFIGURED,
        ),
    )
    yahoo = _StaticProvider(
        "yahoo-finance-chart",
        _provider_result(
            provider_name="yahoo-finance-chart",
            request=request,
            status=ProviderStatus.OK,
            data=MarketSnapshot(ticker="AAPL", bars=(_bar("AAPL"),)),
            credential_state=CredentialState.NOT_REQUIRED,
        ),
    )
    provider = FallbackMarketDataProvider(
        (
            LiveMarketDataSelection(
                provider=alpha,
                source_url="https://alpha.example/AAPL",
                role="primary",
            ),
            LiveMarketDataSelection(
                provider=yahoo,
                source_url="https://yahoo.example/AAPL",
                role="fallback",
            ),
        )
    )

    result = provider.fetch_daily_candles(request)

    assert result.provider_name == "yahoo-finance-chart"
    assert result.status == ProviderStatus.OK
    assert result.data is not None
    assert result.data.bars
    assert alpha.call_count == 1
    assert yahoo.call_count == 1
    assert [warning.provider_name for warning in result.warnings] == ["yahoo-finance-chart"]
    assert "alpha-vantage-market-data" in result.warnings[0].message
    assert result.warnings[0].metadata["fallback_used"] is True
    assert provider.source_url_for_provider("yahoo-finance-chart") == "https://yahoo.example/AAPL"


@pytest.mark.unit
def test_live_market_data_selector_builds_real_outcome_fallback_chain() -> None:
    selections = LiveMarketDataSelector(env={}).outcome_selections(
        symbol="aapl",
        asset_class="stock",
    )

    assert [selection.role for selection in selections] == ["primary", "fallback", "fallback"]
    assert [
        selection.retrieval_method.value for selection in selections if selection.retrieval_method
    ] == [
        "official_api",
        "public_scrape",
        "public_scrape",
    ]
    assert all(selection.source_url is not None for selection in selections)
    assert not any("fixture" in str(selection.source_url).lower() for selection in selections)
    assert (
        LiveMarketDataSelector(env={}).outcome_selections(
            symbol="SPX",
            asset_class="index",
        )
        == ()
    )


class _StaticProvider:
    def __init__(
        self,
        provider_name: str,
        result: ProviderResult[MarketSnapshot],
    ) -> None:
        self.provider_name = provider_name
        self.result = result
        self.call_count = 0

    def fetch_daily_candles(
        self,
        request: MarketDataRequest,
    ) -> ProviderResult[MarketSnapshot]:
        del request
        self.call_count += 1
        return self.result

    def health(self) -> NoReturn:
        raise NotImplementedError


def _provider_result(
    *,
    provider_name: str,
    request: MarketDataRequest,
    status: ProviderStatus,
    data: MarketSnapshot | None,
    credential_state: CredentialState,
    warning_code: WarningCode | None = None,
    message: str | None = None,
) -> ProviderResult[MarketSnapshot]:
    warnings = (
        (
            provider_warning(
                provider_name=provider_name,
                code=warning_code,
                severity=WarningSeverity.WARNING,
                message=message or "provider warning",
                occurred_at=NOW,
            ),
        )
        if warning_code is not None
        else ()
    )
    return provider_result(
        provider_name=provider_name,
        status=status,
        request=request,
        fetched_at=NOW,
        credential_state=credential_state,
        data=data,
        warnings=warnings,
    )


def _bar(ticker: str) -> PriceBar:
    return PriceBar(
        ticker=ticker,
        timestamp=date(2026, 5, 15),
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10.50"),
        volume=100,
    )
