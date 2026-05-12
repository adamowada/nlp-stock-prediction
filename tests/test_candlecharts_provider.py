from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import (
    MarketDataRequest,
    ProviderStatus,
    WarningCode,
)
from nlp_stock_prediction.providers.candlecharts import CandlechartsMarketDataProvider

RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "raw" / "candlecharts"


def _html(name: str) -> str:
    return FIXTURE_ROOT.joinpath(name).read_text(encoding="utf-8")


@pytest.mark.contract
def test_candlecharts_public_html_json_maps_ohlcv() -> None:
    provider = CandlechartsMarketDataProvider(
        html=_html("public_ohlcv_tsla.html"),
        now=lambda: FETCHED_AT,
    )
    request = MarketDataRequest(
        request_id="candlecharts-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_daily_candles(request)

    assert result.status == ProviderStatus.OK
    assert result.warnings == ()
    assert result.data is not None
    assert result.data.ticker == "TSLA"
    assert result.data.bars[0].timestamp == RUN_DATE
    assert result.data.bars[0].open == Decimal("181.00")
    assert result.data.bars[0].close == Decimal("184.25")
    assert result.data.bars[0].volume == 123_456_789
    assert result.data.liquidity_metrics[0].name == "average_volume"
    assert result.raw_snapshot_id is not None
    assert result.raw_snapshot_id.startswith("raw-candlecharts-public-html-")
    assert result.cache_key is not None


@pytest.mark.contract
def test_candlecharts_widget_only_html_returns_unavailable_warning() -> None:
    provider = CandlechartsMarketDataProvider(
        html=_html("widget_only_tsla.html"),
        now=lambda: FETCHED_AT,
    )
    request = MarketDataRequest(
        request_id="candlecharts-widget-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_daily_candles(request)

    assert result.status == ProviderStatus.EMPTY
    assert result.data is None
    assert result.raw_snapshot_id is not None
    warning = result.warnings[0]
    assert warning.code == WarningCode.NO_DATA
    assert "embedded TradingView/widget" in warning.message
    assert warning.metadata["reason"] == "widget_only"
    assert warning.metadata["scraped_tradingview_internals"] is False


@pytest.mark.contract
def test_candlecharts_returns_no_data_when_request_has_no_symbol() -> None:
    provider = CandlechartsMarketDataProvider(
        html=_html("public_ohlcv_tsla.html"),
        now=lambda: FETCHED_AT,
    )
    request = MarketDataRequest(
        request_id="candlecharts-missing-symbol-2026-05-11",
        run_date=RUN_DATE,
        tickers=(),
    )

    result = provider.fetch_daily_candles(request)

    assert result.status == ProviderStatus.EMPTY
    assert result.data is None
    assert result.warnings[0].code == WarningCode.NO_DATA
    assert "did not include a ticker" in result.warnings[0].message


@pytest.mark.contract
def test_candlecharts_malformed_ohlcv_returns_warning_result() -> None:
    provider = CandlechartsMarketDataProvider(
        html=_html("malformed_ohlcv_tsla.html"),
        now=lambda: FETCHED_AT,
    )
    request = MarketDataRequest(
        request_id="candlecharts-malformed-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
    )

    result = provider.fetch_daily_candles(request)

    assert result.status == ProviderStatus.MALFORMED
    assert result.data is None
    assert result.raw_snapshot_id is not None
    assert result.warnings[0].code == WarningCode.MALFORMED_RESPONSE
    assert "invalid OHLCV" in result.warnings[0].message


@pytest.mark.contract
def test_candlecharts_marks_stale_ohlcv_data() -> None:
    provider = CandlechartsMarketDataProvider(
        html=_html("stale_ohlcv_tsla.html"),
        now=lambda: FETCHED_AT,
        stale_after_days=5,
    )
    request = MarketDataRequest(
        request_id="candlecharts-stale-tsla-2026-05-11",
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
