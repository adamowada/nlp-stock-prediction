"""Market data and company-overview provider adapters."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from typing import TypeVar

from nlp_stock_prediction.contracts import (
    CredentialState,
    FundamentalsRequest,
    FundamentalsSnapshot,
    MarketDataRequest,
    MarketSnapshot,
    PriceBar,
    ProviderHealth,
    ProviderMetric,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    WarningCode,
    WarningSeverity,
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

ALPHA_VANTAGE_ENDPOINT = "https://www.alphavantage.co/query"
TProviderPayload = TypeVar("TProviderPayload")


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
            return _alpha_vantage_rate_limited_result(
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
                cache_key=cache_key,
            )
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
            return _alpha_vantage_rate_limited_result(
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
                cache_key=cache_key,
            )
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


def _raise_for_alpha_vantage_message(payload: dict[str, object]) -> None:
    if "Note" in payload or "Information" in payload:
        raise AlphaVantageRateLimitNotice("Alpha Vantage response indicates rate limit or notice")
    if "Error Message" in payload:
        raise MalformedProviderResponse("Alpha Vantage response contains an error message")


def _alpha_vantage_rate_limited_result(
    *,
    provider_name: str,
    request: MarketDataRequest | FundamentalsRequest,
    fetched_at: datetime,
    message: str,
    raw_snapshot_id: str,
    cache_key: str,
) -> ProviderResult[TProviderPayload]:
    warning = provider_warning(
        provider_name=provider_name,
        code=WarningCode.RATE_LIMITED,
        severity=WarningSeverity.ERROR,
        message=message,
        occurred_at=fetched_at,
        raw_snapshot_id=raw_snapshot_id,
        provider_error_type="rate_limit",
    )
    return provider_result(
        provider_name=provider_name,
        status=ProviderStatus.RATE_LIMITED,
        request=request,
        fetched_at=fetched_at,
        credential_state=CredentialState.CONFIGURED,
        warnings=(warning,),
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


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


__all__ = ["AlphaVantageFundamentalsProvider", "AlphaVantageMarketDataProvider"]
