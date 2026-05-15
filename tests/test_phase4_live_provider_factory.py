from __future__ import annotations

import pytest

from nlp_stock_prediction.orchestration.live_market_data import LiveMarketDataSelector
from nlp_stock_prediction.orchestration.phase4_common import retrieval_method_for_provider
from nlp_stock_prediction.orchestration.phase4_live_providers import Phase4LiveProviderFactory
from nlp_stock_prediction.providers.market import YahooFinanceChartMarketDataProvider


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
