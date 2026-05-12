"""Candlecharts public HTML feasibility probe and market-data adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nlp_stock_prediction.contracts import (
    CredentialState,
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
    JsonPayload,
    MalformedProviderResponse,
    ProviderCache,
    ProviderTransportError,
    append_query_params,
    build_cache_key,
    first_ticker,
    malformed_result,
    no_data_result,
    parse_decimal,
    parse_provider_date,
    provider_health,
    provider_result,
    provider_warning,
    raw_snapshot_id_for_payload,
    transport_error_result,
    utc_now,
)

CANDLECHARTS_ENDPOINT = "https://candlecharts.com/live-charts"
CANDLECHARTS_SOURCE = "candlecharts-public-html"


@dataclass(frozen=True)
class HtmlResponse:
    """Raw HTML response plus HTTP metadata for injectable fixture/live transports."""

    text: str
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)


class HtmlTransport(Protocol):
    """Small HTML transport kept local until shared scraping interfaces land."""

    def get_text(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> HtmlResponse: ...


class UrllibHtmlTransport:
    """Stdlib HTML transport used only when callers explicitly opt into live fetching."""

    def get_text(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> HtmlResponse:
        request = Request(url, headers=dict(headers or {}))
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
                return HtmlResponse(
                    text=body.decode(charset, errors="replace"),
                    status_code=int(getattr(response, "status", 200)),
                    headers=dict(response.headers.items()),
                )
        except HTTPError as exc:
            raise ProviderTransportError(
                str(exc),
                status_code=exc.code,
                retryable=exc.code in {408, 425, 429, 500, 502, 503, 504},
                error_type="http_error",
            ) from exc
        except URLError as exc:
            raise ProviderTransportError(str(exc), retryable=True, error_type="url_error") from exc


@dataclass(frozen=True)
class HtmlFetch:
    html: str
    raw_snapshot_id: str
    cache_key: str
    source_url: str
    cache_hit: bool = False


@dataclass(frozen=True)
class _ScriptBlock:
    script_type: str
    text: str


@dataclass(frozen=True)
class _ParseResult:
    snapshot: MarketSnapshot | None
    widget_sources: tuple[str, ...]


class CandlechartsMarketDataProvider:
    """Probe Candlecharts public HTML for first-party OHLCV without scraping widgets."""

    provider_name = "candlecharts-market-data"

    def __init__(
        self,
        *,
        html: str | None = None,
        html_path: Path | None = None,
        endpoint: str = CANDLECHARTS_ENDPOINT,
        transport: HtmlTransport | None = None,
        cache: ProviderCache | None = None,
        allow_live: bool = False,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 10.0,
        stale_after_days: int = 5,
    ) -> None:
        if html is not None and html_path is not None:
            raise ValueError("Candlecharts provider accepts html or html_path, not both")
        self._html = html
        self._html_path = html_path
        self._endpoint = endpoint
        self._transport = transport if transport is not None else UrllibHtmlTransport()
        self._cache = cache
        self._allow_live = allow_live or transport is not None
        self._now = now
        self._timeout = timeout
        self._stale_after_days = stale_after_days

    def probe(self, request: MarketDataRequest) -> ProviderResult[MarketSnapshot]:
        """Alias for callers that want the feasibility-probe wording explicitly."""

        return self.fetch_daily_candles(request)

    def fetch_daily_candles(self, request: MarketDataRequest) -> ProviderResult[MarketSnapshot]:
        fetched_at = self._now()
        ticker = first_ticker(request)
        if ticker is None:
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message="Candlecharts market data request did not include a ticker",
                credential_state=CredentialState.NOT_REQUIRED,
            )

        source_url = append_query_params(self._endpoint, {"symbol": ticker})
        cache_key = build_cache_key(
            provider_name=self.provider_name,
            source=CANDLECHARTS_SOURCE,
            run_date=request.run_date,
            tickers=(ticker,),
            query=ticker,
            url=source_url,
            options={"interval": request.interval, "adjusted": request.adjusted},
        )
        if not self._has_html_source():
            return self._no_html_source_result(request, fetched_at)

        raw_snapshot_id: str | None = None
        try:
            fetched = self._fetch_html(
                request=request,
                ticker=ticker,
                source_url=source_url,
                cache_key=cache_key,
                fetched_at=fetched_at,
            )
            raw_snapshot_id = fetched.raw_snapshot_id
            parsed = self._map_html(ticker=ticker, fetched=fetched)
        except ProviderTransportError as exc:
            return transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=CredentialState.NOT_REQUIRED,
            )
        except MalformedProviderResponse as exc:
            return malformed_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                credential_state=CredentialState.NOT_REQUIRED,
                raw_snapshot_id=raw_snapshot_id,
                cache_key=cache_key,
            )

        if parsed.snapshot is None:
            if parsed.widget_sources:
                return self._widget_only_result(
                    request=request,
                    fetched=fetched,
                    fetched_at=fetched_at,
                    widget_sources=parsed.widget_sources,
                )
            return self._no_public_ohlcv_result(
                request=request,
                fetched=fetched,
                fetched_at=fetched_at,
            )

        warnings: tuple[ProviderWarning, ...] = ()
        status = ProviderStatus.OK
        latest_date = _bar_date(parsed.snapshot.bars[0].timestamp)
        if (request.run_date - latest_date).days > self._stale_after_days:
            status = ProviderStatus.STALE
            warnings = (
                provider_warning(
                    provider_name=self.provider_name,
                    code=WarningCode.STALE_DATA,
                    severity=WarningSeverity.WARNING,
                    message=(
                        f"Latest Candlecharts public OHLCV candle for {ticker} is stale: "
                        f"{latest_date.isoformat()}"
                    ),
                    occurred_at=fetched_at,
                    raw_snapshot_id=fetched.raw_snapshot_id,
                    source_url=fetched.source_url,
                    metadata={"latest_date": latest_date.isoformat()},
                ),
            )
        return provider_result(
            provider_name=self.provider_name,
            status=status,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            data=parsed.snapshot,
            warnings=warnings,
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

    def _has_html_source(self) -> bool:
        return self._html is not None or self._html_path is not None or self._allow_live

    def _fetch_html(
        self,
        *,
        request: MarketDataRequest,
        ticker: str,
        source_url: str,
        cache_key: str,
        fetched_at: datetime,
    ) -> HtmlFetch:
        if self._html is not None:
            return _html_fetch(
                html=self._html,
                source_url=source_url,
                cache_key=cache_key,
                cache_hit=False,
            )
        if self._html_path is not None:
            return _html_fetch(
                html=self._html_path.read_text(encoding="utf-8"),
                source_url=str(self._html_path),
                cache_key=cache_key,
                cache_hit=False,
            )
        cached = self._load_cached_html(request, ticker, cache_key)
        if cached is not None:
            return cached
        response = self._transport.get_text(source_url, timeout=self._timeout)
        fetched = _html_fetch(
            html=response.text,
            source_url=source_url,
            cache_key=cache_key,
            cache_hit=False,
        )
        if self._cache is not None:
            payload: JsonPayload = {"source_url": source_url, "html": response.text}
            record = self._cache.save_json(
                run_date=request.run_date,
                ticker=ticker,
                source=CANDLECHARTS_SOURCE,
                cache_key=cache_key,
                payload=payload,
                fetched_at=fetched_at,
            )
            return HtmlFetch(
                html=response.text,
                raw_snapshot_id=record.raw_snapshot_id,
                cache_key=record.cache_key,
                source_url=source_url,
                cache_hit=False,
            )
        return fetched

    def _load_cached_html(
        self,
        request: MarketDataRequest,
        ticker: str,
        cache_key: str,
    ) -> HtmlFetch | None:
        if self._cache is None:
            return None
        cached = self._cache.load_json(
            run_date=request.run_date,
            ticker=ticker,
            source=CANDLECHARTS_SOURCE,
            cache_key=cache_key,
        )
        if cached is None:
            return None
        html = cached.payload.get("html")
        source_url = cached.payload.get("source_url")
        if not isinstance(html, str) or not isinstance(source_url, str):
            return None
        return HtmlFetch(
            html=html,
            raw_snapshot_id=cached.raw_snapshot_id,
            cache_key=cached.cache_key,
            source_url=source_url,
            cache_hit=True,
        )

    def _map_html(self, *, ticker: str, fetched: HtmlFetch) -> _ParseResult:
        parser = _CandlechartsHtmlParser()
        parser.feed(fetched.html)
        widget_sources = parser.widget_sources()
        for payload in _json_payloads_from_parser(parser):
            bars = _bars_from_payload(payload, ticker)
            if bars:
                return _ParseResult(
                    snapshot=MarketSnapshot(
                        ticker=ticker,
                        bars=bars,
                        liquidity_metrics=_liquidity_metrics(list(bars)),
                    ),
                    widget_sources=widget_sources,
                )
        table_bars = _bars_from_table_rows(parser.table_rows, ticker)
        if table_bars:
            return _ParseResult(
                snapshot=MarketSnapshot(
                    ticker=ticker,
                    bars=table_bars,
                    liquidity_metrics=_liquidity_metrics(list(table_bars)),
                ),
                widget_sources=widget_sources,
            )
        return _ParseResult(snapshot=None, widget_sources=widget_sources)

    def _no_html_source_result(
        self,
        request: MarketDataRequest,
        fetched_at: datetime,
    ) -> ProviderResult[MarketSnapshot]:
        warning = provider_warning(
            provider_name=self.provider_name,
            code=WarningCode.NO_DATA,
            severity=WarningSeverity.INFO,
            message=(
                "Candlecharts live HTML fetching is disabled by default; provide fixture HTML, "
                "an HTML fixture path, or explicitly opt into live fetching"
            ),
            occurred_at=fetched_at,
            metadata={"reason": "no_html_source", "live_requests_enabled_by_default": False},
        )
        return provider_result(
            provider_name=self.provider_name,
            status=ProviderStatus.EMPTY,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            warnings=(warning,),
        )

    def _widget_only_result(
        self,
        *,
        request: MarketDataRequest,
        fetched: HtmlFetch,
        fetched_at: datetime,
        widget_sources: tuple[str, ...],
    ) -> ProviderResult[MarketSnapshot]:
        warning = provider_warning(
            provider_name=self.provider_name,
            code=WarningCode.NO_DATA,
            severity=WarningSeverity.WARNING,
            message=(
                "Candlecharts public HTML appears unavailable for OHLCV extraction: the page "
                "only exposes an embedded TradingView/widget path, and TradingView internals "
                "were not scraped"
            ),
            occurred_at=fetched_at,
            raw_snapshot_id=fetched.raw_snapshot_id,
            source_url=fetched.source_url,
            metadata={
                "reason": "widget_only",
                "widget_sources": list(widget_sources),
                "scraped_tradingview_internals": False,
                "cache_hit": fetched.cache_hit,
            },
        )
        return provider_result(
            provider_name=self.provider_name,
            status=ProviderStatus.EMPTY,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            warnings=(warning,),
            raw_snapshot_id=fetched.raw_snapshot_id,
            cache_key=fetched.cache_key,
        )

    def _no_public_ohlcv_result(
        self,
        *,
        request: MarketDataRequest,
        fetched: HtmlFetch,
        fetched_at: datetime,
    ) -> ProviderResult[MarketSnapshot]:
        warning = provider_warning(
            provider_name=self.provider_name,
            code=WarningCode.NO_DATA,
            severity=WarningSeverity.INFO,
            message="Candlecharts public HTML did not contain usable first-party OHLCV data",
            occurred_at=fetched_at,
            raw_snapshot_id=fetched.raw_snapshot_id,
            source_url=fetched.source_url,
            metadata={"reason": "no_public_ohlcv", "cache_hit": fetched.cache_hit},
        )
        return provider_result(
            provider_name=self.provider_name,
            status=ProviderStatus.EMPTY,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            warnings=(warning,),
            raw_snapshot_id=fetched.raw_snapshot_id,
            cache_key=fetched.cache_key,
        )


class _CandlechartsHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[_ScriptBlock] = []
        self.json_attributes: list[str] = []
        self.table_rows: list[list[str]] = []
        self._widget_sources: list[str] = []
        self._in_script = False
        self._script_type = ""
        self._script_parts: list[str] = []
        self._in_ohlcv_table = False
        self._table_depth = 0
        self._in_row = False
        self._current_row: list[str] = []
        self._in_cell = False
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {key.lower(): value or "" for key, value in attrs}
        self._record_widget_attrs(attrs_map)
        for key in ("data-ohlcv", "data-candles", "data-bars"):
            value = attrs_map.get(key)
            if value:
                self.json_attributes.append(unescape(value))
        if tag == "script":
            src = attrs_map.get("src", "")
            if src:
                self._record_widget_source(src)
            self._in_script = True
            self._script_type = attrs_map.get("type", "")
            self._script_parts = []
            return
        if tag == "iframe":
            self._record_widget_source(attrs_map.get("src", ""))
        if tag == "table":
            if self._in_ohlcv_table:
                self._table_depth += 1
            elif _attrs_look_like_ohlcv_table(attrs_map):
                self._in_ohlcv_table = True
                self._table_depth = 1
        elif self._in_ohlcv_table and tag == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script:
            text = "".join(self._script_parts).strip()
            if text:
                self.scripts.append(_ScriptBlock(script_type=self._script_type, text=text))
                if _text_mentions_widget(text):
                    self._record_widget_source("inline:tradingview-widget")
            self._in_script = False
            self._script_type = ""
            self._script_parts = []
            return
        if self._in_cell and tag in {"td", "th"}:
            self._current_row.append(" ".join("".join(self._cell_parts).split()))
            self._in_cell = False
            self._cell_parts = []
        elif self._in_row and tag == "tr":
            if any(cell for cell in self._current_row):
                self.table_rows.append(self._current_row)
            self._in_row = False
            self._current_row = []
        elif self._in_ohlcv_table and tag == "table":
            self._table_depth -= 1
            if self._table_depth <= 0:
                self._in_ohlcv_table = False
                self._table_depth = 0

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_parts.append(data)
        elif self._in_cell:
            self._cell_parts.append(data)

    def widget_sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(source for source in self._widget_sources if source))

    def _record_widget_attrs(self, attrs_map: Mapping[str, str]) -> None:
        for value in attrs_map.values():
            self._record_widget_source(value)

    def _record_widget_source(self, value: str) -> None:
        if _text_mentions_widget(value):
            self._widget_sources.append(value)


def _html_fetch(
    *,
    html: str,
    source_url: str,
    cache_key: str,
    cache_hit: bool,
) -> HtmlFetch:
    payload: JsonPayload = {"source_url": source_url, "html": html}
    return HtmlFetch(
        html=html,
        raw_snapshot_id=raw_snapshot_id_for_payload(CANDLECHARTS_SOURCE, payload),
        cache_key=cache_key,
        source_url=source_url,
        cache_hit=cache_hit,
    )


def _json_payloads_from_parser(parser: _CandlechartsHtmlParser) -> tuple[object, ...]:
    payloads: list[object] = []
    for raw_value in parser.json_attributes:
        payloads.append(_loads_json(raw_value, context="Candlecharts data attribute"))
    for script in parser.scripts:
        if "json" not in script.script_type.lower():
            continue
        payloads.append(_loads_json(script.text, context="Candlecharts JSON script"))
    return tuple(payloads)


def _loads_json(value: str, *, context: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise MalformedProviderResponse(f"{context} is malformed JSON") from exc


def _bars_from_payload(payload: object, ticker: str) -> tuple[PriceBar, ...]:
    for raw_bars, symbol in _walk_ohlcv_collections(payload, inherited_symbol=None):
        if symbol is not None and _normalize_symbol(symbol) != ticker.upper():
            continue
        bars = tuple(_map_raw_bar(raw_bar, ticker) for raw_bar in raw_bars)
        if bars:
            return _sort_bars(bars)
    return ()


def _walk_ohlcv_collections(
    value: object,
    *,
    inherited_symbol: str | None,
) -> tuple[tuple[Sequence[Mapping[str, object]], str | None], ...]:
    collections: list[tuple[Sequence[Mapping[str, object]], str | None]] = []
    if isinstance(value, Mapping):
        symbol = _symbol_from_mapping(value) or inherited_symbol
        for key, child in value.items():
            if _is_ohlcv_collection_key(str(key)) and _looks_like_bar_list(child):
                collections.append((cast(Sequence[Mapping[str, object]], child), symbol))
            else:
                collections.extend(_walk_ohlcv_collections(child, inherited_symbol=symbol))
    elif isinstance(value, list):
        if _looks_like_bar_list(value):
            collections.append((cast(Sequence[Mapping[str, object]], value), inherited_symbol))
        else:
            for child in value:
                collections.extend(
                    _walk_ohlcv_collections(child, inherited_symbol=inherited_symbol)
                )
    return tuple(collections)


def _looks_like_bar_list(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return isinstance(value[0], Mapping) and _is_bar_mapping(value[0])


def _is_bar_mapping(value: Mapping[object, object]) -> bool:
    lower_keys = {str(key).lower() for key in value}
    return bool(lower_keys & _TIME_KEYS) and all(
        bool(lower_keys & aliases) for aliases in _REQUIRED_PRICE_KEYS
    )


def _bars_from_table_rows(rows: Sequence[Sequence[str]], ticker: str) -> tuple[PriceBar, ...]:
    if len(rows) < 2:
        return ()
    headers = [cell.strip().lower().replace(" ", "_") for cell in rows[0]]
    if not _headers_contain_ohlcv(headers):
        return ()
    bars: list[PriceBar] = []
    for row in rows[1:]:
        raw_bar = {
            header: row[index]
            for index, header in enumerate(headers)
            if index < len(row) and header
        }
        bars.append(_map_raw_bar(raw_bar, ticker))
    return _sort_bars(tuple(bars))


def _map_raw_bar(raw_bar: Mapping[str, object], ticker: str) -> PriceBar:
    timestamp = _parse_bar_date(_first_value(raw_bar, _TIME_KEYS))
    open_price = _positive_decimal(_first_value(raw_bar, {"open", "o"}))
    high = _positive_decimal(_first_value(raw_bar, {"high", "h"}))
    low = _positive_decimal(_first_value(raw_bar, {"low", "l"}))
    close = _positive_decimal(_first_value(raw_bar, {"close", "c"}))
    volume = _parse_volume(_first_value(raw_bar, {"volume", "vol", "v"}))
    adjusted_close = parse_decimal(
        _first_value(raw_bar, {"adjusted_close", "adjustedclose", "adj_close", "adjclose"})
    )
    if None in {timestamp, open_price, high, low, close} or volume is None:
        raise MalformedProviderResponse("Candlecharts bar has missing or invalid OHLCV fields")
    open_checked = cast(Decimal, open_price)
    high_checked = cast(Decimal, high)
    low_checked = cast(Decimal, low)
    close_checked = cast(Decimal, close)
    if high_checked < low_checked or high_checked < max(open_checked, close_checked):
        raise MalformedProviderResponse("Candlecharts bar has invalid OHLCV price relationships")
    if low_checked > min(open_checked, close_checked):
        raise MalformedProviderResponse("Candlecharts bar has invalid OHLCV price relationships")
    return PriceBar(
        ticker=ticker,
        timestamp=cast(date, timestamp),
        open=open_checked,
        high=high_checked,
        low=low_checked,
        close=close_checked,
        adjusted_close=adjusted_close,
        volume=volume,
    )


def _first_value(raw_bar: Mapping[str, object], aliases: set[str]) -> object:
    for key, value in raw_bar.items():
        if str(key).lower().replace(" ", "_") in aliases:
            return value
    return None


def _parse_bar_date(value: object) -> date | None:
    parsed = parse_provider_date(value)
    if parsed is not None:
        return parsed
    if isinstance(value, int | float) and not isinstance(value, bool):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1000
        return datetime.fromtimestamp(seconds, tz=UTC).date()
    return None


def _positive_decimal(value: object) -> Decimal | None:
    decimal_value = parse_decimal(value)
    if decimal_value is None or decimal_value <= 0:
        return None
    return decimal_value


def _parse_volume(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        volume = int(str(value).replace(",", ""))
    except ValueError:
        return None
    return volume if volume >= 0 else None


def _sort_bars(bars: tuple[PriceBar, ...]) -> tuple[PriceBar, ...]:
    return tuple(sorted(bars, key=lambda bar: _bar_date(bar.timestamp), reverse=True))


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


def _symbol_from_mapping(value: Mapping[object, object]) -> str | None:
    for key in ("symbol", "ticker"):
        raw_symbol = value.get(key)
        if raw_symbol is not None:
            return str(raw_symbol)
    return None


def _normalize_symbol(value: str) -> str:
    return value.strip().upper().removeprefix("$").split(":")[-1]


def _is_ohlcv_collection_key(value: str) -> bool:
    return value.lower().replace("-", "_") in {"ohlcv", "candles", "bars", "price_bars"}


def _headers_contain_ohlcv(headers: Sequence[str]) -> bool:
    header_set = set(headers)
    return bool(header_set & _TIME_KEYS) and all(
        bool(header_set & aliases) for aliases in _REQUIRED_PRICE_KEYS
    )


def _attrs_look_like_ohlcv_table(attrs_map: Mapping[str, str]) -> bool:
    text = " ".join(attrs_map.get(key, "") for key in ("id", "class", "data-testid"))
    normalized = text.lower()
    return "ohlcv" in normalized or "candles" in normalized or "price-bars" in normalized


def _text_mentions_widget(value: str) -> bool:
    normalized = value.lower()
    return "tradingview" in normalized or "widgetembed" in normalized


_TIME_KEYS = {"date", "time", "timestamp", "datetime"}
_REQUIRED_PRICE_KEYS = (
    {"open", "o"},
    {"high", "h"},
    {"low", "l"},
    {"close", "c"},
    {"volume", "vol", "v"},
)


__all__ = [
    "CandlechartsMarketDataProvider",
    "HtmlResponse",
    "HtmlTransport",
    "UrllibHtmlTransport",
]
