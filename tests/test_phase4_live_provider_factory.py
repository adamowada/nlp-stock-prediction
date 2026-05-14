from __future__ import annotations

import pytest

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
