"""Market data and company-overview provider adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, cast
from urllib.parse import quote, urlencode

from nlp_stock_prediction.contracts.enums import (
    CredentialState,
    ProviderStatus,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.contracts.provenance import ProviderHealth, ProviderWarning
from nlp_stock_prediction.contracts.providers import (
    FundamentalsRequest,
    FundamentalsSnapshot,
    MarketDataRequest,
    MarketSnapshot,
    PriceBar,
    ProviderMetric,
    ProviderResult,
)
from nlp_stock_prediction.providers._base import (
    JsonFetch,
    JsonTransport,
    MalformedProviderResponse,
    ProviderCache,
    ProviderTransportError,
    UrllibJsonTransport,
    append_query_params,
    build_cache_key,
    fetch_json,
    first_ticker,
    malformed_result,
    missing_credentials_result,
    no_data_result,
    parse_decimal,
    parse_provider_date,
    provider_health,
    provider_result,
    provider_warning,
    transport_error_result,
    utc_now,
)
from nlp_stock_prediction.providers.execution import rate_limited_result

ALPHA_VANTAGE_ENDPOINT = "https://www.alphavantage.co/query"
YAHOO_FINANCE_CHART_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_FINANCE_CHART_SOURCE = "yahoo-finance-chart-daily"


class AlphaVantageRateLimitNotice(Exception):
    """Raised when Alpha Vantage returns a quota/rate-limit notice payload."""


class AlphaVantageMarketDataProvider:
    """Daily candle provider using the Alpha Vantage daily-adjusted response shape."""

    provider_name = "alpha-vantage-market-data"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = ALPHA_VANTAGE_ENDPOINT,
        transport: JsonTransport | None = None,
        cache: ProviderCache | None = None,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 10.0,
        stale_after_days: int = 5,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._transport = transport or UrllibJsonTransport()
        self._cache = cache
        self._now = now
        self._timeout = timeout
        self._stale_after_days = stale_after_days

    def fetch_daily_candles(self, request: MarketDataRequest) -> ProviderResult[MarketSnapshot]:
        fetched_at = self._now()
        if not self._api_key:
            return missing_credentials_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                credential_name="Alpha Vantage API key",
            )
        ticker = first_ticker(request)
        if ticker is None:
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message="Market data request did not include a ticker",
                credential_state=CredentialState.CONFIGURED,
            )
        url = append_query_params(
            self._endpoint,
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "outputsize": "compact",
                "apikey": self._api_key,
            },
        )
        cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="alpha-vantage-daily",
            run_date=request.run_date,
            tickers=(ticker,),
            query=ticker,
            url=url,
            options={"interval": request.interval, "adjusted": request.adjusted},
        )
        fetched: JsonFetch | None = None
        try:
            fetched = fetch_json(
                transport=self._transport,
                url=url,
                run_date=request.run_date,
                ticker=ticker,
                source="alpha-vantage-daily",
                cache_key=cache_key,
                fetched_at=fetched_at,
                cache=self._cache,
                timeout=self._timeout,
                cacheable_payload=_is_alpha_vantage_cacheable,
            )
            snapshot = self._map_daily_payload(ticker, fetched.payload)
        except ProviderTransportError as exc:
            return transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=CredentialState.CONFIGURED,
            )
        except AlphaVantageRateLimitNotice as exc:
            assert fetched is not None
            return rate_limited_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                raw_snapshot_id=fetched.raw_snapshot_id,
                cache_key=fetched.cache_key,
            )
        except MalformedProviderResponse as exc:
            return malformed_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                credential_state=CredentialState.CONFIGURED,
                raw_snapshot_id=fetched.raw_snapshot_id if fetched is not None else None,
                cache_key=fetched.cache_key if fetched is not None else cache_key,
            )
        assert fetched is not None
        if not snapshot.bars:
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=f"Alpha Vantage returned no daily candles for {ticker}",
                credential_state=CredentialState.CONFIGURED,
                raw_snapshot_id=fetched.raw_snapshot_id,
                cache_key=fetched.cache_key,
            )
        warnings: tuple[ProviderWarning, ...] = ()
        status = ProviderStatus.OK
        latest_date = _bar_date(snapshot.bars[0].timestamp)
        if (request.run_date - latest_date).days > self._stale_after_days:
            status = ProviderStatus.STALE
            warnings = (
                provider_warning(
                    provider_name=self.provider_name,
                    code=WarningCode.STALE_DATA,
                    severity=WarningSeverity.WARNING,
                    message=f"Latest daily candle for {ticker} is stale: {latest_date.isoformat()}",
                    occurred_at=fetched_at,
                    raw_snapshot_id=fetched.raw_snapshot_id,
                    metadata={"latest_date": latest_date.isoformat()},
                ),
            )
        return provider_result(
            provider_name=self.provider_name,
            status=status,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.CONFIGURED,
            data=snapshot,
            warnings=warnings,
            raw_snapshot_id=fetched.raw_snapshot_id,
            cache_key=fetched.cache_key,
        )

    def health(self) -> ProviderHealth:
        checked_at = self._now()
        if not self._api_key:
            warning = provider_warning(
                provider_name=self.provider_name,
                code=WarningCode.MISSING_CREDENTIALS,
                severity=WarningSeverity.ERROR,
                message="Alpha Vantage API key is not configured",
                occurred_at=checked_at,
            )
            return provider_health(
                provider_name=self.provider_name,
                status=ProviderStatus.UNCONFIGURED,
                checked_at=checked_at,
                credential_state=CredentialState.MISSING,
                warnings=(warning,),
            )
        return provider_health(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            checked_at=checked_at,
            credential_state=CredentialState.CONFIGURED,
        )

    def _map_daily_payload(self, ticker: str, payload: dict[str, object]) -> MarketSnapshot:
        _raise_for_alpha_vantage_message(payload)
        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict):
            raise MalformedProviderResponse("Alpha Vantage daily response missing time series")
        bars: list[PriceBar] = []
        for date_text, raw_bar in sorted(series.items(), reverse=True):
            if not isinstance(raw_bar, dict):
                raise MalformedProviderResponse("Alpha Vantage daily bar must be an object")
            bar_date = parse_provider_date(date_text)
            if bar_date is None:
                raise MalformedProviderResponse("Alpha Vantage daily bar has invalid date")
            open_price = parse_decimal(raw_bar.get("1. open"))
            high = parse_decimal(raw_bar.get("2. high"))
            low = parse_decimal(raw_bar.get("3. low"))
            close = parse_decimal(raw_bar.get("4. close"))
            adjusted_close = parse_decimal(raw_bar.get("5. adjusted close"))
            volume_text = raw_bar.get("6. volume")
            volume = _parse_volume(volume_text)
            if None in {open_price, high, low, close} or volume is None:
                raise MalformedProviderResponse(
                    "Alpha Vantage daily bar has missing or invalid OHLCV fields"
                )
            try:
                bars.append(
                    PriceBar(
                        ticker=ticker,
                        timestamp=bar_date,
                        open=open_price or Decimal("0"),
                        high=high or Decimal("0"),
                        low=low or Decimal("0"),
                        close=close or Decimal("0"),
                        adjusted_close=adjusted_close,
                        volume=volume,
                    )
                )
            except ValueError as exc:
                raise MalformedProviderResponse(
                    "Alpha Vantage daily bar has invalid OHLCV price relationships"
                ) from exc
        return MarketSnapshot(
            ticker=ticker,
            bars=tuple(bars),
            liquidity_metrics=_liquidity_metrics(bars),
        )


class AlphaVantageFundamentalsProvider:
    """Company overview adapter using the Alpha Vantage OVERVIEW response shape."""

    provider_name = "alpha-vantage-fundamentals"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = ALPHA_VANTAGE_ENDPOINT,
        transport: JsonTransport | None = None,
        cache: ProviderCache | None = None,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._transport = transport or UrllibJsonTransport()
        self._cache = cache
        self._now = now
        self._timeout = timeout

    def fetch_fundamentals(
        self, request: FundamentalsRequest
    ) -> ProviderResult[FundamentalsSnapshot]:
        fetched_at = self._now()
        if not self._api_key:
            return missing_credentials_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                credential_name="Alpha Vantage API key",
            )
        ticker = first_ticker(request)
        if ticker is None:
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message="Fundamentals request did not include a ticker",
                credential_state=CredentialState.CONFIGURED,
            )
        url = append_query_params(
            self._endpoint,
            {"function": "OVERVIEW", "symbol": ticker, "apikey": self._api_key},
        )
        cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="alpha-vantage-overview",
            run_date=request.run_date,
            tickers=(ticker,),
            query=ticker,
            url=url,
        )
        fetched: JsonFetch | None = None
        try:
            fetched = fetch_json(
                transport=self._transport,
                url=url,
                run_date=request.run_date,
                ticker=ticker,
                source="alpha-vantage-overview",
                cache_key=cache_key,
                fetched_at=fetched_at,
                cache=self._cache,
                timeout=self._timeout,
                cacheable_payload=_is_alpha_vantage_cacheable,
            )
            snapshot = self._map_overview_payload(ticker, fetched)
        except ProviderTransportError as exc:
            return transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=CredentialState.CONFIGURED,
            )
        except AlphaVantageRateLimitNotice as exc:
            assert fetched is not None
            return rate_limited_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                raw_snapshot_id=fetched.raw_snapshot_id,
                cache_key=fetched.cache_key,
            )
        except MalformedProviderResponse as exc:
            return malformed_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                credential_state=CredentialState.CONFIGURED,
                raw_snapshot_id=fetched.raw_snapshot_id if fetched is not None else None,
                cache_key=fetched.cache_key if fetched is not None else cache_key,
            )
        assert fetched is not None
        if not snapshot.metrics:
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=f"Alpha Vantage returned no company overview metrics for {ticker}",
                credential_state=CredentialState.CONFIGURED,
                raw_snapshot_id=fetched.raw_snapshot_id,
                cache_key=fetched.cache_key,
            )
        return provider_result(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.CONFIGURED,
            data=snapshot,
            raw_snapshot_id=fetched.raw_snapshot_id,
            cache_key=fetched.cache_key,
        )

    def health(self) -> ProviderHealth:
        checked_at = self._now()
        if not self._api_key:
            warning = provider_warning(
                provider_name=self.provider_name,
                code=WarningCode.MISSING_CREDENTIALS,
                severity=WarningSeverity.ERROR,
                message="Alpha Vantage API key is not configured",
                occurred_at=checked_at,
            )
            return provider_health(
                provider_name=self.provider_name,
                status=ProviderStatus.UNCONFIGURED,
                checked_at=checked_at,
                credential_state=CredentialState.MISSING,
                warnings=(warning,),
            )
        return provider_health(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            checked_at=checked_at,
            credential_state=CredentialState.CONFIGURED,
        )

    def _map_overview_payload(
        self,
        ticker: str,
        fetched: JsonFetch,
    ) -> FundamentalsSnapshot:
        payload = fetched.payload
        _raise_for_alpha_vantage_message(payload)
        company_name = _optional_text(payload.get("Name"))
        latest_quarter = parse_provider_date(payload.get("LatestQuarter"))
        metrics: list[ProviderMetric] = []
        for field_name, metric_name, unit in _OVERVIEW_METRICS:
            raw_value = payload.get(field_name)
            if raw_value is None:
                continue
            value = _metric_value(raw_value)
            if value is None:
                continue
            metrics.append(
                ProviderMetric(
                    name=metric_name,
                    value=value,
                    unit=unit,
                    as_of=latest_quarter,
                    metadata={
                        "provider_field": field_name,
                        "raw_snapshot_id": fetched.raw_snapshot_id,
                    },
                )
            )
        return FundamentalsSnapshot(
            ticker=ticker,
            company_name=company_name,
            metrics=tuple(metrics),
        )


class YahooFinanceChartError(Exception):
    """Raised when Yahoo Finance returns a semantic chart error payload."""


class YahooFinanceChartMarketDataProvider:
    """Daily candle provider using Yahoo Finance's public chart JSON endpoint."""

    provider_name = "yahoo-finance-chart"

    def __init__(
        self,
        *,
        endpoint: str = YAHOO_FINANCE_CHART_ENDPOINT,
        transport: JsonTransport | None = None,
        cache: ProviderCache | None = None,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 10.0,
        stale_after_days: int = 5,
        user_agent: str = "Mozilla/5.0 nlp-stock-prediction-live-market-data/0.1",
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._transport = transport or UrllibJsonTransport()
        self._cache = cache
        self._now = now
        self._timeout = timeout
        self._stale_after_days = stale_after_days
        self._user_agent = user_agent

    def fetch_daily_candles(self, request: MarketDataRequest) -> ProviderResult[MarketSnapshot]:
        fetched_at = self._now()
        ticker = first_ticker(request)
        if ticker is None:
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message="Yahoo Finance chart request did not include a ticker",
                credential_state=CredentialState.NOT_REQUIRED,
            )
        start_date, end_date = _request_date_range(request)
        url = yahoo_finance_chart_source_url(
            ticker,
            endpoint=self._endpoint,
            start_date=start_date,
            end_date=end_date,
        )
        cache_key = build_cache_key(
            provider_name=self.provider_name,
            source=YAHOO_FINANCE_CHART_SOURCE,
            run_date=request.run_date,
            tickers=(ticker,),
            query=ticker,
            url=url,
            options={
                "interval": request.interval,
                "adjusted": request.adjusted,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )
        fetched: JsonFetch | None = None
        try:
            fetched = fetch_json(
                transport=self._transport,
                url=url,
                run_date=request.run_date,
                ticker=ticker,
                source=YAHOO_FINANCE_CHART_SOURCE,
                cache_key=cache_key,
                fetched_at=fetched_at,
                cache=self._cache,
                headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                timeout=self._timeout,
                cacheable_payload=_is_yahoo_chart_cacheable,
            )
            snapshot, skipped_rows = self._map_chart_payload(ticker, fetched.payload)
        except ProviderTransportError as exc:
            return transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=CredentialState.NOT_REQUIRED,
            )
        except YahooFinanceChartError as exc:
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                credential_state=CredentialState.NOT_REQUIRED,
                raw_snapshot_id=fetched.raw_snapshot_id if fetched is not None else None,
                cache_key=fetched.cache_key if fetched is not None else cache_key,
            )
        except MalformedProviderResponse as exc:
            return malformed_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                credential_state=CredentialState.NOT_REQUIRED,
                raw_snapshot_id=fetched.raw_snapshot_id if fetched is not None else None,
                cache_key=fetched.cache_key if fetched is not None else cache_key,
            )
        assert fetched is not None
        if not snapshot.bars:
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=f"Yahoo Finance Chart returned no daily candles for {ticker}",
                credential_state=CredentialState.NOT_REQUIRED,
                raw_snapshot_id=fetched.raw_snapshot_id,
                cache_key=fetched.cache_key,
            )

        warnings: list[ProviderWarning] = []
        status = ProviderStatus.OK
        if skipped_rows:
            status = ProviderStatus.PARTIAL
            warnings.append(
                provider_warning(
                    provider_name=self.provider_name,
                    code=WarningCode.PARTIAL_DATA,
                    severity=WarningSeverity.WARNING,
                    message=(
                        f"Skipped {skipped_rows} incomplete Yahoo Finance daily candle row(s) "
                        f"for {ticker}."
                    ),
                    occurred_at=fetched_at,
                    raw_snapshot_id=fetched.raw_snapshot_id,
                    metadata={"skipped_row_count": skipped_rows},
                )
            )
        latest_date = _bar_date(snapshot.bars[0].timestamp)
        if (request.run_date - latest_date).days > self._stale_after_days:
            status = ProviderStatus.STALE
            warnings.append(
                provider_warning(
                    provider_name=self.provider_name,
                    code=WarningCode.STALE_DATA,
                    severity=WarningSeverity.WARNING,
                    message=(
                        f"Latest Yahoo Finance daily candle for {ticker} is stale: "
                        f"{latest_date.isoformat()}"
                    ),
                    occurred_at=fetched_at,
                    raw_snapshot_id=fetched.raw_snapshot_id,
                    metadata={"latest_date": latest_date.isoformat()},
                )
            )
        return provider_result(
            provider_name=self.provider_name,
            status=status,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            data=snapshot,
            warnings=tuple(warnings),
            raw_snapshot_id=fetched.raw_snapshot_id,
            cache_key=fetched.cache_key,
        )

    def health(self) -> ProviderHealth:
        return provider_health(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            checked_at=self._now(),
            credential_state=CredentialState.NOT_REQUIRED,
        )

    def _map_chart_payload(
        self,
        ticker: str,
        payload: dict[str, object],
    ) -> tuple[MarketSnapshot, int]:
        chart = payload.get("chart")
        if not isinstance(chart, dict):
            raise MalformedProviderResponse("Yahoo Finance chart response missing chart object")
        _raise_for_yahoo_chart_error(chart)
        results = chart.get("result")
        if not isinstance(results, list) or not results:
            raise YahooFinanceChartError(
                f"Yahoo Finance Chart returned no chart result for {ticker}"
            )
        result = results[0]
        if not isinstance(result, dict):
            raise MalformedProviderResponse("Yahoo Finance chart result must be an object")
        timestamps = result.get("timestamp")
        if not isinstance(timestamps, list) or not timestamps:
            raise YahooFinanceChartError(f"Yahoo Finance Chart returned no timestamps for {ticker}")
        indicators = result.get("indicators")
        if not isinstance(indicators, dict):
            raise MalformedProviderResponse("Yahoo Finance chart response missing indicators")
        quotes = indicators.get("quote")
        if not isinstance(quotes, list) or not quotes or not isinstance(quotes[0], dict):
            raise MalformedProviderResponse("Yahoo Finance chart response missing quote series")
        quote_series = cast(dict[str, object], quotes[0])
        adjclose_series = _first_indicator_series(indicators.get("adjclose"))
        bars: list[PriceBar] = []
        skipped_rows = 0
        for index, raw_timestamp in enumerate(timestamps):
            bar = _yahoo_price_bar(
                ticker=ticker,
                timestamp=raw_timestamp,
                index=index,
                quote_series=quote_series,
                adjclose_series=adjclose_series,
            )
            if bar is None:
                skipped_rows += 1
                continue
            bars.append(bar)
        bars.sort(key=lambda bar: _bar_date(bar.timestamp), reverse=True)
        return (
            MarketSnapshot(
                ticker=ticker,
                bars=tuple(bars),
                liquidity_metrics=_liquidity_metrics(bars),
            ),
            skipped_rows,
        )


def _raise_for_alpha_vantage_message(payload: dict[str, object]) -> None:
    if "Note" in payload or "Information" in payload:
        raise AlphaVantageRateLimitNotice("Alpha Vantage response indicates rate limit or notice")
    if "Error Message" in payload:
        raise MalformedProviderResponse("Alpha Vantage response contains an error message")


def _is_alpha_vantage_cacheable(payload: dict[str, object]) -> bool:
    return not any(key in payload for key in ("Note", "Information", "Error Message"))


def _is_yahoo_chart_cacheable(payload: dict[str, object]) -> bool:
    chart = payload.get("chart")
    return not (isinstance(chart, dict) and chart.get("error"))


def _raise_for_yahoo_chart_error(chart: dict[Any, Any]) -> None:
    error = chart.get("error")
    if not isinstance(error, dict) or not error:
        return
    code = _optional_text(error.get("code")) or "unknown"
    description = _optional_text(error.get("description")) or "unknown error"
    raise YahooFinanceChartError(f"Yahoo Finance Chart returned {code}: {description}")


def yahoo_finance_chart_source_url(
    symbol: str,
    *,
    endpoint: str = YAHOO_FINANCE_CHART_ENDPOINT,
    start_date: date | None = None,
    end_date: date | None = None,
) -> str:
    normalized_symbol = symbol.strip().upper()
    params: dict[str, object] = {
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    if start_date is None or end_date is None:
        params["range"] = "1y"
    else:
        params["period1"] = _unix_day_start(start_date)
        params["period2"] = _unix_day_start(end_date + timedelta(days=1))
    return f"{endpoint.rstrip('/')}/{quote(normalized_symbol, safe='')}?{urlencode(params)}"


def _request_date_range(request: MarketDataRequest) -> tuple[date, date]:
    if request.window is None:
        return request.run_date - timedelta(days=370), request.run_date
    start = _date_from_window_value(request.window.start)
    end = _date_from_window_value(request.window.end)
    return start, end


def _date_from_window_value(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _unix_day_start(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=UTC).timestamp())


def _first_indicator_series(value: object) -> dict[str, object]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return cast(dict[str, object], value[0])
    return {}


def _sequence_value(series: dict[str, object], key: str, index: int) -> object:
    values = series.get(key)
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        return None
    if index >= len(values):
        return None
    return values[index]


def _yahoo_price_bar(
    *,
    ticker: str,
    timestamp: object,
    index: int,
    quote_series: dict[str, object],
    adjclose_series: dict[str, object],
) -> PriceBar | None:
    if not isinstance(timestamp, int | float) or isinstance(timestamp, bool):
        return None
    open_price = parse_decimal(_sequence_value(quote_series, "open", index))
    high = parse_decimal(_sequence_value(quote_series, "high", index))
    low = parse_decimal(_sequence_value(quote_series, "low", index))
    close = parse_decimal(_sequence_value(quote_series, "close", index))
    adjusted_close = parse_decimal(_sequence_value(adjclose_series, "adjclose", index))
    volume = _parse_volume(_sequence_value(quote_series, "volume", index))
    if None in {open_price, high, low, close} or volume is None:
        return None
    try:
        return PriceBar(
            ticker=ticker,
            timestamp=datetime.fromtimestamp(float(timestamp), tz=UTC).date(),
            open=open_price or Decimal("0"),
            high=high or Decimal("0"),
            low=low or Decimal("0"),
            close=close or Decimal("0"),
            adjusted_close=adjusted_close,
            volume=volume,
        )
    except ValueError:
        return None


def _liquidity_metrics(bars: list[PriceBar]) -> tuple[ProviderMetric, ...]:
    if not bars:
        return ()
    latest_date = _bar_date(bars[0].timestamp)
    average_volume = sum(bar.volume for bar in bars) // len(bars)
    total_dollar_volume = sum(bar.close * Decimal(bar.volume) for bar in bars)
    average_dollar_volume = total_dollar_volume / Decimal(len(bars))
    return (
        ProviderMetric(
            name="average_volume",
            value=average_volume,
            unit="shares",
            as_of=latest_date,
            metadata={"window": f"{len(bars)}d"},
        ),
        ProviderMetric(
            name="average_dollar_volume",
            value=average_dollar_volume,
            unit="USD",
            as_of=latest_date,
            metadata={"window": f"{len(bars)}d"},
        ),
    )


def _bar_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_volume(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        volume = int(str(value).replace(",", ""))
    except ValueError:
        return None
    return volume if volume >= 0 else None


def _metric_value(value: object) -> Decimal | str | None:
    decimal_value = parse_decimal(value)
    if decimal_value is not None:
        return decimal_value
    text = _optional_text(value)
    if text in {None, "None", "null", "-"}:
        return None
    return text


_OVERVIEW_METRICS = (
    ("MarketCapitalization", "market_cap", "USD"),
    ("PERatio", "pe_ratio", None),
    ("ForwardPE", "forward_pe", None),
    ("EPS", "eps", "USD/share"),
    ("RevenueTTM", "revenue_ttm", "USD"),
    ("ProfitMargin", "profit_margin", "ratio"),
    ("QuarterlyRevenueGrowthYOY", "quarterly_revenue_growth_yoy", "ratio"),
    ("QuarterlyEarningsGrowthYOY", "quarterly_earnings_growth_yoy", "ratio"),
    ("Sector", "sector", None),
    ("Industry", "industry", None),
)


__all__ = [
    "ALPHA_VANTAGE_ENDPOINT",
    "YAHOO_FINANCE_CHART_ENDPOINT",
    "AlphaVantageFundamentalsProvider",
    "AlphaVantageMarketDataProvider",
    "YahooFinanceChartMarketDataProvider",
    "yahoo_finance_chart_source_url",
]
