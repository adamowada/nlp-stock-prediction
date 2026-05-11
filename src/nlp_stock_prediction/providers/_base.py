"""Shared helpers for concrete provider adapters."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from nlp_stock_prediction.contracts import (
    CredentialState,
    EvidenceRequest,
    FreshnessStatus,
    ProviderHealth,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    SourceProvenance,
    TextSpan,
    WarningCode,
    WarningSeverity,
)

T = TypeVar("T")
JsonPayload = dict[str, Any]
NowFunc = Callable[[], datetime]


@dataclass(frozen=True)
class JsonResponse:
    """Parsed JSON response plus HTTP metadata from a provider transport."""

    payload: JsonPayload
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)


class JsonTransport(Protocol):
    """Small transport interface so tests can inject deterministic provider fixtures."""

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> JsonResponse: ...


class ProviderTransportError(Exception):
    """Transport or upstream failure that should become a ProviderResult warning."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        error_type: str = "transport_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.error_type = error_type


class MalformedProviderResponse(Exception):
    """A provider response could not be mapped into normalized contracts."""


class UrllibJsonTransport:
    """Stdlib urllib-backed JSON transport used by live opt-in callers."""

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> JsonResponse:
        request = Request(url, headers=dict(headers or {}))
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise MalformedProviderResponse("provider JSON root must be an object")
                status_code = int(getattr(response, "status", 200))
                response_headers = dict(response.headers.items())
                return JsonResponse(
                    payload=cast(JsonPayload, payload),
                    status_code=status_code,
                    headers=response_headers,
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
        except json.JSONDecodeError as exc:
            raise MalformedProviderResponse("provider returned invalid JSON") from exc


@dataclass(frozen=True)
class CacheRecord:
    payload: JsonPayload
    raw_snapshot_id: str
    cache_key: str


class ProviderCache:
    """File cache partitioned by run date, ticker, and provider source."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, run_date: date, ticker: str | None, source: str, cache_key: str) -> Path:
        ticker_component = safe_path_component(ticker or "all")
        source_component = safe_path_component(source)
        cache_filename = f"{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:16]}.json"
        return (
            self.root / run_date.isoformat() / ticker_component / source_component / cache_filename
        )

    def load_json(
        self,
        *,
        run_date: date,
        ticker: str | None,
        source: str,
        cache_key: str,
    ) -> CacheRecord | None:
        path = self.path_for(run_date, ticker, source, cache_key)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            return None
        raw_snapshot_id = raw.get("raw_snapshot_id")
        if not isinstance(raw_snapshot_id, str) or not raw_snapshot_id:
            raw_snapshot_id = raw_snapshot_id_for_payload(source, cast(JsonPayload, payload))
        return CacheRecord(
            payload=cast(JsonPayload, payload),
            raw_snapshot_id=raw_snapshot_id,
            cache_key=cache_key,
        )

    def save_json(
        self,
        *,
        run_date: date,
        ticker: str | None,
        source: str,
        cache_key: str,
        payload: JsonPayload,
        fetched_at: datetime,
    ) -> CacheRecord:
        path = self.path_for(run_date, ticker, source, cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_snapshot_id = raw_snapshot_id_for_payload(source, payload)
        path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "cached_at": fetched_at.isoformat().replace("+00:00", "Z"),
                    "raw_snapshot_id": raw_snapshot_id,
                    "payload": payload,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return CacheRecord(payload=payload, raw_snapshot_id=raw_snapshot_id, cache_key=cache_key)


@dataclass(frozen=True)
class JsonFetch:
    payload: JsonPayload
    raw_snapshot_id: str
    cache_key: str
    cache_hit: bool
    status_code: int


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_provider_datetime(value: object, *, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return ensure_aware_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        normalized = value.strip().replace("Z", "+00:00")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
            parsed_date = date.fromisoformat(normalized)
            return datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=UTC)
        try:
            return ensure_aware_utc(datetime.fromisoformat(normalized))
        except ValueError:
            return ensure_aware_utc(fallback)
    return ensure_aware_utc(fallback)


def parse_provider_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def parse_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"None", "null", "NaN", ".", "-"}:
        return None
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def safe_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return safe.strip("-") or "unknown"


def snake_case(value: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    second_pass = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass)
    return re.sub(r"[^a-z0-9]+", "_", second_pass.lower()).strip("_")


def stable_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def raw_snapshot_id_for_payload(source: str, payload: JsonPayload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"raw-{safe_path_component(source)}-{stable_hash(encoded, length=20)}"


def build_cache_key(
    *,
    provider_name: str,
    source: str,
    run_date: date,
    tickers: Sequence[str] = (),
    query: str | None = None,
    url: str | None = None,
    options: Mapping[str, object] | None = None,
) -> str:
    tickers_upper = [ticker.upper() for ticker in tickers]
    payload = {
        "provider_name": provider_name,
        "source": source,
        "run_date": run_date.isoformat(),
        "tickers": tickers_upper,
        "query": query,
        "url": url,
        "options": dict(sorted((options or {}).items())),
    }
    digest = stable_hash(json.dumps(payload, sort_keys=True, default=str), length=16)
    ticker_part = "-".join(tickers_upper) or "all"
    return f"{provider_name}:{source}:{run_date.isoformat()}:{ticker_part}:{digest}"


def append_query_params(url: str, params: Mapping[str, object | None]) -> str:
    split = urlsplit(url)
    existing = parse_qsl(split.query, keep_blank_values=True)
    additions = [(key, str(value)) for key, value in params.items() if value is not None]
    query = urlencode(existing + additions)
    return urlunsplit((split.scheme, split.netloc, split.path, query, split.fragment))


def fetch_json(
    *,
    transport: JsonTransport,
    url: str,
    run_date: date,
    ticker: str | None,
    source: str,
    cache_key: str,
    fetched_at: datetime,
    cache: ProviderCache | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 10.0,
) -> JsonFetch:
    if cache is not None:
        cached = cache.load_json(
            run_date=run_date,
            ticker=ticker,
            source=source,
            cache_key=cache_key,
        )
        if cached is not None:
            return JsonFetch(
                payload=cached.payload,
                raw_snapshot_id=cached.raw_snapshot_id,
                cache_key=cached.cache_key,
                cache_hit=True,
                status_code=200,
            )
    response = transport.get_json(url, headers=headers, timeout=timeout)
    if cache is None:
        return JsonFetch(
            payload=response.payload,
            raw_snapshot_id=raw_snapshot_id_for_payload(source, response.payload),
            cache_key=cache_key,
            cache_hit=False,
            status_code=response.status_code,
        )
    record = cache.save_json(
        run_date=run_date,
        ticker=ticker,
        source=source,
        cache_key=cache_key,
        payload=response.payload,
        fetched_at=fetched_at,
    )
    return JsonFetch(
        payload=record.payload,
        raw_snapshot_id=record.raw_snapshot_id,
        cache_key=record.cache_key,
        cache_hit=False,
        status_code=response.status_code,
    )


def provider_warning(
    *,
    provider_name: str,
    code: WarningCode,
    severity: WarningSeverity,
    message: str,
    occurred_at: datetime,
    retryable: bool = False,
    provider_status_code: int | None = None,
    provider_error_type: str | None = None,
    raw_snapshot_id: str | None = None,
    source_url: str | None = None,
    stale_after: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ProviderWarning:
    return ProviderWarning(
        code=code,
        severity=severity,
        message=message,
        provider_name=provider_name,
        retryable=retryable,
        provider_status_code=provider_status_code,
        provider_error_type=provider_error_type,
        occurred_at=ensure_aware_utc(occurred_at),
        stale_after=ensure_aware_utc(stale_after) if stale_after else None,
        raw_snapshot_id=raw_snapshot_id,
        source_url=source_url,
        metadata=dict(metadata or {}),
    )


def provider_health(
    *,
    provider_name: str,
    status: ProviderStatus,
    checked_at: datetime,
    credential_state: CredentialState,
    warnings: tuple[ProviderWarning, ...] = (),
    latency_ms: int | None = None,
    rate_limit_remaining: int | None = None,
    last_success_at: datetime | None = None,
) -> ProviderHealth:
    successful_statuses = {ProviderStatus.OK, ProviderStatus.PARTIAL, ProviderStatus.STALE}
    return ProviderHealth(
        provider_name=provider_name,
        status=status,
        checked_at=ensure_aware_utc(checked_at),
        credential_state=credential_state,
        latency_ms=latency_ms,
        rate_limit_remaining=rate_limit_remaining,
        last_success_at=(
            ensure_aware_utc(last_success_at or checked_at)
            if status in successful_statuses
            else None
        ),
        warnings=warnings,
    )


def provider_result[T](
    *,
    provider_name: str,
    status: ProviderStatus,
    request: ProviderRequest,
    fetched_at: datetime,
    credential_state: CredentialState,
    data: T | None = None,
    warnings: tuple[ProviderWarning, ...] = (),
    raw_snapshot_id: str | None = None,
    cache_key: str | None = None,
    latency_ms: int | None = None,
    rate_limit_remaining: int | None = None,
) -> ProviderResult[T]:
    fetched_at_utc = ensure_aware_utc(fetched_at)
    return ProviderResult[T](
        provider_name=provider_name,
        status=status,
        request=request,
        fetched_at=fetched_at_utc,
        data=data,
        warnings=warnings,
        health=provider_health(
            provider_name=provider_name,
            status=status,
            checked_at=fetched_at_utc,
            credential_state=credential_state,
            warnings=warnings,
            latency_ms=latency_ms,
            rate_limit_remaining=rate_limit_remaining,
        ),
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


def missing_credentials_result(
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    credential_name: str,
) -> ProviderResult[T]:
    warning = provider_warning(
        provider_name=provider_name,
        code=WarningCode.MISSING_CREDENTIALS,
        severity=WarningSeverity.ERROR,
        message=f"{provider_name} is missing required credential: {credential_name}",
        occurred_at=fetched_at,
        provider_error_type="missing_credentials",
        metadata={"credential_name": credential_name},
    )
    return provider_result(
        provider_name=provider_name,
        status=ProviderStatus.UNCONFIGURED,
        request=request,
        fetched_at=fetched_at,
        credential_state=CredentialState.MISSING,
        warnings=(warning,),
    )


def no_data_result(
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    message: str,
    credential_state: CredentialState = CredentialState.NOT_REQUIRED,
    raw_snapshot_id: str | None = None,
    cache_key: str | None = None,
) -> ProviderResult[T]:
    warning = provider_warning(
        provider_name=provider_name,
        code=WarningCode.NO_DATA,
        severity=WarningSeverity.INFO,
        message=message,
        occurred_at=fetched_at,
        raw_snapshot_id=raw_snapshot_id,
    )
    return provider_result(
        provider_name=provider_name,
        status=ProviderStatus.EMPTY,
        request=request,
        fetched_at=fetched_at,
        credential_state=credential_state,
        warnings=(warning,),
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


def malformed_result(
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    message: str,
    credential_state: CredentialState,
    raw_snapshot_id: str | None = None,
    cache_key: str | None = None,
) -> ProviderResult[T]:
    warning = provider_warning(
        provider_name=provider_name,
        code=WarningCode.MALFORMED_RESPONSE,
        severity=WarningSeverity.ERROR,
        message=message,
        occurred_at=fetched_at,
        provider_error_type="malformed_response",
        raw_snapshot_id=raw_snapshot_id,
    )
    return provider_result(
        provider_name=provider_name,
        status=ProviderStatus.MALFORMED,
        request=request,
        fetched_at=fetched_at,
        credential_state=credential_state,
        warnings=(warning,),
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


def transport_error_result(
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    error: ProviderTransportError,
    credential_state: CredentialState,
) -> ProviderResult[T]:
    if error.status_code in {401, 403}:
        status = ProviderStatus.UNAUTHORIZED
        code = WarningCode.AUTH_FAILED
    elif error.status_code == 429:
        status = ProviderStatus.RATE_LIMITED
        code = WarningCode.RATE_LIMITED
    else:
        status = ProviderStatus.FAILED
        code = WarningCode.UPSTREAM_UNAVAILABLE
    warning = provider_warning(
        provider_name=provider_name,
        code=code,
        severity=WarningSeverity.ERROR,
        message=str(error),
        occurred_at=fetched_at,
        retryable=error.retryable,
        provider_status_code=error.status_code,
        provider_error_type=error.error_type,
    )
    health_credential_state = (
        CredentialState.INVALID if status == ProviderStatus.UNAUTHORIZED else credential_state
    )
    return provider_result(
        provider_name=provider_name,
        status=status,
        request=request,
        fetched_at=fetched_at,
        credential_state=health_credential_state,
        warnings=(warning,),
        rate_limit_remaining=0 if status == ProviderStatus.RATE_LIMITED else None,
    )


def freshness_status(
    *,
    observed_at: datetime,
    fetched_at: datetime,
    stale_after_seconds: int,
) -> tuple[FreshnessStatus, int]:
    observed = ensure_aware_utc(observed_at)
    fetched = ensure_aware_utc(fetched_at)
    seconds = max(0, int((fetched - observed).total_seconds()))
    status = FreshnessStatus.STALE if seconds > stale_after_seconds else FreshnessStatus.FRESH
    return status, seconds


def source_provenance(
    *,
    provider_name: str,
    source_kind: Any,
    retrieval_method: Any,
    fetched_at: datetime,
    source_url: str,
    raw_identifier: str,
    raw_snapshot_id: str,
    observed_at: datetime | None = None,
    permalink: str | None = None,
    query: str | None = None,
    cache_key: str | None = None,
    freshness: FreshnessStatus = FreshnessStatus.FRESH,
    freshness_seconds: int | None = None,
    provider_metadata: Mapping[str, object] | None = None,
) -> SourceProvenance:
    return SourceProvenance(
        provider_name=provider_name,
        source_kind=source_kind,
        retrieval_method=retrieval_method,
        fetched_at=ensure_aware_utc(fetched_at),
        observed_at=ensure_aware_utc(observed_at) if observed_at else None,
        source_url=source_url,
        permalink=permalink,
        raw_identifier=raw_identifier,
        raw_snapshot_id=raw_snapshot_id,
        query=query,
        cache_key=cache_key,
        freshness_status=freshness,
        freshness_seconds=freshness_seconds,
        provider_metadata=dict(provider_metadata or {}),
    )


def first_ticker(request: ProviderRequest) -> str | None:
    return request.tickers[0] if request.tickers else None


def query_from_tickers(request: EvidenceRequest) -> str:
    if request.query:
        return request.query
    if not request.tickers:
        return ""
    return " OR ".join(request.tickers)


def find_ticker_matches(
    text: str,
    tickers: Sequence[str],
) -> tuple[tuple[str, ...], tuple[TextSpan, ...]]:
    matched: list[str] = []
    spans: list[TextSpan] = []
    for ticker in tickers:
        normalized = ticker.upper()
        patterns = (
            re.compile(rf"\${re.escape(normalized)}\b", re.IGNORECASE),
            re.compile(rf"(?<![A-Z0-9$]){re.escape(normalized)}(?![A-Z0-9])", re.IGNORECASE),
        )
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                if normalized not in matched:
                    matched.append(normalized)
                spans.append(
                    TextSpan(
                        text=match.group(0),
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )
                break
    return tuple(matched), tuple(spans)
