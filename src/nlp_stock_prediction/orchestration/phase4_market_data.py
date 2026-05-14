"""Phase 4 market-data tool backed by provider-result contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, cast

from pydantic import Field

from nlp_stock_prediction.contracts import (
    AuditArtifact,
    ContractModel,
    CredentialState,
    FreshnessStatus,
    JsonObject,
    JsonValue,
    MarketDataProvider,
    MarketDataRequest,
    MarketSnapshot,
    NonEmptyStr,
    PriceBar,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    RetrievalMethod,
    TickerSymbol,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.ml.ohlcv import calendar_date, timestamp_key_for
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase2_common import stable_digest, symbol_slug, utc_now
from nlp_stock_prediction.orchestration.phase4_common import (
    retrieval_method_for_provider,
    sanitize_source_query_url,
)
from nlp_stock_prediction.orchestration.phase4_execution import safe_phase4_tool_execution
from nlp_stock_prediction.providers._base import provider_health, provider_warning
from nlp_stock_prediction.storage.records import SourceQueryRecord, ToolRunRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE4_MARKET_DATA_SCHEMA_VERSION = "phase4.market-data.v1"
PHASE4_MARKET_DATA_TOOL_NAME = "phase4_market_data"
PHASE4_MARKET_DATA_TOOL_VERSION = PHASE4_MARKET_DATA_SCHEMA_VERSION


class MarketDataProvenance(ContractModel):
    """Provider/source-query traceability for a market-data artifact."""

    provider_name: NonEmptyStr
    source_query_id: NonEmptyStr
    query: NonEmptyStr
    url: str | None = None
    retrieved_at: datetime
    retrieval_method: RetrievalMethod = RetrievalMethod.FIXTURE
    raw_identifier: str | None = None
    raw_snapshot_id: str | None = None
    cache_key: str | None = None
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: JsonObject = Field(default_factory=dict)


class Phase4MarketDataArtifact(ContractModel):
    """Stable JSON payload written by the Phase 4 market-data tool."""

    schema_version: NonEmptyStr = PHASE4_MARKET_DATA_SCHEMA_VERSION
    run_id: NonEmptyStr
    tool_run_id: NonEmptyStr
    artifact_id: NonEmptyStr
    symbol: TickerSymbol
    generated_at: datetime
    status: ProviderStatus
    freshness_status: FreshnessStatus
    latest_usable_bar: PriceBar | None = None
    bar_count: int = Field(ge=0)
    provider_result: ProviderResult[MarketSnapshot]
    bars: tuple[PriceBar, ...] = ()
    liquidity_metrics: tuple[JsonObject, ...] = ()
    provenance: MarketDataProvenance
    warnings: tuple[ProviderWarning, ...] = ()
    metadata: JsonObject = Field(default_factory=dict)


@dataclass(frozen=True)
class MarketDataToolResult:
    """Market-data provider output plus the indexed artifact metadata."""

    provider_result: ProviderResult[MarketSnapshot]
    artifact: AuditArtifact
    artifact_payload: Phase4MarketDataArtifact
    tool_run_id: str
    source_query_id: str

    @property
    def latest_usable_bar(self) -> PriceBar | None:
        return self.artifact_payload.latest_usable_bar


class Phase4MarketDataProviderTool(Protocol):
    """Callable surface for fixture and live market-data tool adapters."""

    def run(
        self,
        *,
        run_id: str,
        run_date: date,
        symbol: str,
        instrument_id: str | None = None,
        request_id: str | None = None,
        interval: str = "1d",
        adjusted: bool = True,
        source_url: str | Path | None = None,
        options: JsonObject | None = None,
    ) -> MarketDataToolResult: ...


@dataclass(frozen=True)
class Phase4MarketDataTool:
    """Fetch daily OHLCV through a provider and index a first-class artifact."""

    store: SQLiteStore
    repo_root: Path
    artifact_dir: Path
    provider: MarketDataProvider
    now: Callable[[], datetime] = utc_now
    retrieval_method: RetrievalMethod | None = None

    def run(
        self,
        *,
        run_id: str,
        run_date: date,
        symbol: str,
        instrument_id: str | None = None,
        request_id: str | None = None,
        interval: str = "1d",
        adjusted: bool = True,
        source_url: str | Path | None = None,
        options: JsonObject | None = None,
    ) -> MarketDataToolResult:
        normalized_symbol = _normalize_symbol(symbol)
        started_at = self.now()
        request = MarketDataRequest(
            request_id=request_id or _market_data_request_id(run_id, normalized_symbol),
            run_date=run_date,
            tickers=(normalized_symbol,),
            query=normalized_symbol,
            interval=interval,
            adjusted=adjusted,
            options={} if options is None else options,
        )
        tool_run_id = _market_data_tool_run_id(run_id, normalized_symbol)
        artifact_id = _market_data_artifact_id(run_id, normalized_symbol)
        inputs: JsonObject = {
            "symbol": normalized_symbol,
            "instrument_id": instrument_id,
            "request_id": request.request_id,
            "interval": interval,
            "adjusted": adjusted,
            "source_url": str(source_url) if source_url is not None else None,
        }
        with safe_phase4_tool_execution(
            store=self.store,
            artifact_roots=(self.artifact_dir,),
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=PHASE4_MARKET_DATA_TOOL_NAME,
            tool_version=PHASE4_MARKET_DATA_TOOL_VERSION,
            started_at=started_at,
            inputs=inputs,
        ):
            provider_result = sanitize_market_data_provider_result(
                self.provider.fetch_daily_candles(request),
                requested_symbol=normalized_symbol,
            )
            completed_at = self.now()
            retrieval_method = self.retrieval_method or retrieval_method_for_provider(
                provider_result.provider_name
            )
            source_query_id = _market_data_source_query_id(
                run_id=run_id,
                provider_name=provider_result.provider_name,
                symbol=normalized_symbol,
            )
            resolved_source_url = _source_url(
                source_url=source_url,
                provider_name=provider_result.provider_name,
                symbol=normalized_symbol,
                retrieval_method=retrieval_method,
            )
            freshness_status = freshness_status_for_provider_result(provider_result)
            latest_bar = latest_usable_bar(provider_result.data)
            warnings = provider_result.warnings

            self.store.record_tool_run(
                ToolRunRecord(
                    tool_run_id=tool_run_id,
                    run_id=run_id,
                    tool_name=PHASE4_MARKET_DATA_TOOL_NAME,
                    tool_version=PHASE4_MARKET_DATA_TOOL_VERSION,
                    status=_tool_run_status(provider_result.status, warnings),
                    started_at=started_at,
                    completed_at=completed_at,
                    inputs={
                        **inputs,
                        "provider": provider_result.provider_name,
                    },
                    warnings=tuple(warning.message for warning in warnings),
                )
            )
            self.store.record_source_query(
                SourceQueryRecord(
                    source_query_id=source_query_id,
                    tool_run_id=tool_run_id,
                    provider=provider_result.provider_name,
                    query=normalized_symbol,
                    url=resolved_source_url,
                    retrieved_at=provider_result.fetched_at,
                    metadata={
                        "run_id": run_id,
                        "request_id": request.request_id,
                        "status": provider_result.status.value,
                        "freshness_status": freshness_status.value,
                        "raw_snapshot_id": provider_result.raw_snapshot_id,
                        "cache_key": provider_result.cache_key,
                        "retrieval_method": retrieval_method.value,
                        **_data_quality_metadata(provider_result),
                    },
                )
            )

            artifact_payload = build_market_data_artifact(
                run_id=run_id,
                tool_run_id=tool_run_id,
                artifact_id=artifact_id,
                generated_at=completed_at,
                provider_result=provider_result,
                source_query_id=source_query_id,
                source_url=resolved_source_url,
                retrieval_method=retrieval_method,
                instrument_id=instrument_id,
            )
            artifact = ArtifactIndex.for_directory(
                store=self.store,
                repo_root=self.repo_root,
                base_dir=self.artifact_dir,
                created_at=completed_at,
                produced_by=PHASE4_MARKET_DATA_TOOL_NAME,
                tool_run_id=tool_run_id,
                schema_version=PHASE4_MARKET_DATA_SCHEMA_VERSION,
            ).write_json(
                artifact_id=artifact_id,
                artifact_type="market_data",
                filename=f"market-data/{symbol_slug(normalized_symbol)}.json",
                payload=market_data_artifact_payload(artifact_payload),
                record_count=artifact_payload.bar_count,
                metadata={
                    "symbol": normalized_symbol,
                    "instrument_id": instrument_id,
                    "provider": provider_result.provider_name,
                    "status": provider_result.status.value,
                    "freshness_status": freshness_status.value,
                    "latest_usable_bar": (
                        calendar_date(latest_bar.timestamp).isoformat() if latest_bar else None
                    ),
                    "source_query_id": source_query_id,
                    "warning_count": len(warnings),
                    **_data_quality_metadata(provider_result),
                },
            )
            return MarketDataToolResult(
                provider_result=provider_result,
                artifact=artifact,
                artifact_payload=artifact_payload,
                tool_run_id=tool_run_id,
                source_query_id=source_query_id,
            )


def build_market_data_artifact(
    *,
    run_id: str,
    tool_run_id: str,
    artifact_id: str,
    generated_at: datetime,
    provider_result: ProviderResult[MarketSnapshot],
    source_query_id: str,
    source_url: str | None,
    retrieval_method: RetrievalMethod | None = None,
    instrument_id: str | None = None,
) -> Phase4MarketDataArtifact:
    """Build the typed payload without writing files or touching SQLite."""

    provider_result = sanitize_market_data_provider_result(
        provider_result,
        requested_symbol=_request_symbol(provider_result.request),
    )
    resolved_retrieval_method = retrieval_method or retrieval_method_for_provider(
        provider_result.provider_name
    )
    snapshot = provider_result.data
    bars = snapshot.bars if snapshot is not None else ()
    latest_bar = latest_usable_bar(snapshot)
    freshness_status = freshness_status_for_provider_result(provider_result)
    liquidity_metrics = (
        tuple(
            cast(JsonObject, metric.model_dump(mode="json"))
            for metric in snapshot.liquidity_metrics
        )
        if snapshot is not None
        else ()
    )
    symbol = snapshot.ticker if snapshot is not None else _request_symbol(provider_result.request)
    return Phase4MarketDataArtifact(
        run_id=run_id,
        tool_run_id=tool_run_id,
        artifact_id=artifact_id,
        symbol=symbol,
        generated_at=generated_at,
        status=provider_result.status,
        freshness_status=freshness_status,
        latest_usable_bar=latest_bar,
        bar_count=len(bars),
        provider_result=provider_result,
        bars=bars,
        liquidity_metrics=liquidity_metrics,
        provenance=MarketDataProvenance(
            provider_name=provider_result.provider_name,
            source_query_id=source_query_id,
            query=symbol,
            url=source_url,
            retrieved_at=provider_result.fetched_at,
            retrieval_method=resolved_retrieval_method,
            raw_identifier=f"{symbol}:{provider_result.provider_name}:daily-ohlcv",
            raw_snapshot_id=provider_result.raw_snapshot_id,
            cache_key=provider_result.cache_key,
            freshness_status=freshness_status,
            extraction_confidence=_extraction_confidence(provider_result.status),
            metadata={
                "instrument_id": instrument_id,
                "request_id": provider_result.request.request_id,
                "interval": getattr(provider_result.request, "interval", None),
                "adjusted": getattr(provider_result.request, "adjusted", None),
                **_data_quality_metadata(provider_result),
            },
        ),
        warnings=provider_result.warnings,
        metadata={
            "instrument_id": instrument_id,
            "request_id": provider_result.request.request_id,
            "health": provider_result.health.model_dump(mode="json"),
            **_data_quality_metadata(provider_result),
        },
    )


def market_data_artifact_payload(artifact: Phase4MarketDataArtifact) -> JsonObject:
    """Serialize a Phase 4 market-data artifact to stable JSON-compatible data."""

    return cast(JsonObject, artifact.model_dump(mode="json"))


def latest_usable_bar(snapshot: MarketSnapshot | None) -> PriceBar | None:
    """Return the latest provider bar after normalizing date and aware datetime keys."""

    if snapshot is None or not snapshot.bars:
        return None
    return max(snapshot.bars, key=lambda bar: timestamp_key_for(bar.timestamp))


def freshness_status_for_provider_result(
    provider_result: ProviderResult[MarketSnapshot],
) -> FreshnessStatus:
    """Map provider status into the freshness vocabulary used by reports."""

    if provider_result.status == ProviderStatus.STALE:
        return FreshnessStatus.STALE
    if provider_result.status in {ProviderStatus.OK, ProviderStatus.PARTIAL}:
        return (
            FreshnessStatus.FRESH
            if latest_usable_bar(provider_result.data)
            else FreshnessStatus.UNKNOWN
        )
    if provider_result.status in {ProviderStatus.EMPTY, ProviderStatus.UNCONFIGURED}:
        return FreshnessStatus.MISSING
    return FreshnessStatus.UNKNOWN


def sanitize_market_data_provider_result(
    provider_result: ProviderResult[MarketSnapshot],
    *,
    requested_symbol: str,
) -> ProviderResult[MarketSnapshot]:
    """Reject mismatched identities and remove bars after the request run date."""

    requested = _normalize_symbol(requested_symbol)
    snapshot = provider_result.data
    if snapshot is None:
        return provider_result

    warnings = list(provider_result.warnings)
    if _normalize_symbol(snapshot.ticker) != requested:
        warnings.extend(
            (
                _quality_warning(
                    provider_result=provider_result,
                    code=WarningCode.SCHEMA_MISMATCH,
                    severity=WarningSeverity.ERROR,
                    message=(
                        f"Provider returned market snapshot for {snapshot.ticker}, "
                        f"but {requested} was requested."
                    ),
                    metadata={
                        "requested_symbol": requested,
                        "returned_snapshot_ticker": snapshot.ticker,
                        "rejected_snapshot": True,
                    },
                ),
                _quality_warning(
                    provider_result=provider_result,
                    code=WarningCode.NO_DATA,
                    severity=WarningSeverity.WARNING,
                    message=(
                        f"No usable OHLCV bars remained for {requested} after identity validation."
                    ),
                    metadata={
                        "requested_symbol": requested,
                        "returned_snapshot_ticker": snapshot.ticker,
                        "rejected_snapshot": True,
                    },
                ),
            )
        )
        return _replace_provider_result(
            provider_result,
            status=ProviderStatus.EMPTY,
            data=None,
            warnings=_dedupe_warnings(tuple(warnings)),
        )

    usable_bars: list[PriceBar] = []
    rejected_identity_bars: list[PriceBar] = []
    excluded_future_bars: list[PriceBar] = []
    for bar in snapshot.bars:
        if _normalize_symbol(bar.ticker) != requested:
            rejected_identity_bars.append(bar)
            continue
        if calendar_date(bar.timestamp) > provider_result.request.run_date:
            excluded_future_bars.append(bar)
            continue
        usable_bars.append(bar)

    if rejected_identity_bars:
        warnings.append(
            _quality_warning(
                provider_result=provider_result,
                code=WarningCode.SCHEMA_MISMATCH,
                severity=WarningSeverity.ERROR,
                message=(
                    f"Provider returned {len(rejected_identity_bars)} OHLCV bar(s) whose "
                    f"ticker did not match requested symbol {requested}."
                ),
                metadata={
                    "requested_symbol": requested,
                    "rejected_bar_count": len(rejected_identity_bars),
                    "rejected_identity_bars": _bar_audit_records(rejected_identity_bars),
                },
            )
        )
    if excluded_future_bars:
        warnings.append(
            _quality_warning(
                provider_result=provider_result,
                code=WarningCode.PARTIAL_DATA,
                severity=WarningSeverity.WARNING,
                message=(
                    f"Excluded {len(excluded_future_bars)} OHLCV bar(s) for {requested} after "
                    f"run_date {provider_result.request.run_date.isoformat()}."
                ),
                metadata={
                    "requested_symbol": requested,
                    "run_date": provider_result.request.run_date.isoformat(),
                    "excluded_future_bar_count": len(excluded_future_bars),
                    "excluded_future_bars": _bar_audit_records(excluded_future_bars),
                },
            )
        )

    if not rejected_identity_bars and not excluded_future_bars:
        return provider_result

    if not usable_bars:
        warnings.append(
            _quality_warning(
                provider_result=provider_result,
                code=WarningCode.NO_DATA,
                severity=WarningSeverity.WARNING,
                message=(
                    f"No usable OHLCV bars remained for {requested} after identity/as-of "
                    "validation."
                ),
                metadata={
                    "requested_symbol": requested,
                    "run_date": provider_result.request.run_date.isoformat(),
                    "rejected_bar_count": len(rejected_identity_bars),
                    "excluded_future_bar_count": len(excluded_future_bars),
                },
            )
        )
        return _replace_provider_result(
            provider_result,
            status=ProviderStatus.EMPTY,
            data=None,
            warnings=_dedupe_warnings(tuple(warnings)),
        )

    status = (
        ProviderStatus.PARTIAL
        if provider_result.status == ProviderStatus.OK
        else provider_result.status
    )
    filtered_snapshot = snapshot.model_copy(
        update={
            "bars": tuple(usable_bars),
            "liquidity_metrics": (),
        }
    )
    return _replace_provider_result(
        provider_result,
        status=status,
        data=filtered_snapshot,
        warnings=_dedupe_warnings(tuple(warnings)),
    )


def load_phase4_market_data_artifact(path: Path) -> Phase4MarketDataArtifact:
    """Load a previously written market-data artifact."""

    return Phase4MarketDataArtifact.model_validate_json(path.read_text(encoding="utf-8"))


def _request_symbol(request: ProviderRequest) -> str:
    if request.tickers:
        return request.tickers[0]
    if request.query:
        return _normalize_symbol(request.query)
    return "UNKNOWN"


def _market_data_request_id(run_id: str, symbol: str) -> str:
    return f"phase4-market-data-{stable_digest(f'{run_id}:{symbol}')}"


def _market_data_tool_run_id(run_id: str, symbol: str) -> str:
    return f"tool-phase4-market-data-{stable_digest(f'{run_id}:{symbol}')}"


def _market_data_artifact_id(run_id: str, symbol: str) -> str:
    return f"artifact-market-data-{stable_digest(f'{run_id}:{symbol}')}"


def _market_data_source_query_id(*, run_id: str, provider_name: str, symbol: str) -> str:
    return f"query-market-data-{stable_digest(f'{run_id}:{provider_name}:{symbol}')}"


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _source_url(
    *,
    source_url: str | Path | None,
    provider_name: str,
    symbol: str,
    retrieval_method: RetrievalMethod,
) -> str | None:
    if source_url is not None:
        if isinstance(source_url, Path):
            return str(source_url)
        return sanitize_source_query_url(str(source_url))
    if retrieval_method != RetrievalMethod.FIXTURE:
        return None
    return f"fixture://{provider_name}/{symbol_slug(symbol)}/daily-ohlcv"


def _quality_warning(
    *,
    provider_result: ProviderResult[MarketSnapshot],
    code: WarningCode,
    severity: WarningSeverity,
    message: str,
    metadata: JsonObject,
) -> ProviderWarning:
    return provider_warning(
        provider_name=provider_result.provider_name,
        code=code,
        severity=severity,
        message=message,
        occurred_at=provider_result.fetched_at,
        raw_snapshot_id=provider_result.raw_snapshot_id,
        metadata=metadata,
    )


def _replace_provider_result(
    provider_result: ProviderResult[MarketSnapshot],
    *,
    status: ProviderStatus,
    data: MarketSnapshot | None,
    warnings: tuple[ProviderWarning, ...],
) -> ProviderResult[MarketSnapshot]:
    return ProviderResult[MarketSnapshot](
        provider_name=provider_result.provider_name,
        status=status,
        request=provider_result.request,
        fetched_at=provider_result.fetched_at,
        data=data,
        warnings=warnings,
        health=provider_health(
            provider_name=provider_result.provider_name,
            status=status,
            checked_at=provider_result.health.checked_at,
            credential_state=(
                provider_result.health.credential_state
                if provider_result.health is not None
                else CredentialState.NOT_REQUIRED
            ),
            warnings=warnings,
            latency_ms=provider_result.health.latency_ms,
            rate_limit_remaining=provider_result.health.rate_limit_remaining,
            last_success_at=provider_result.health.last_success_at,
        ),
        raw_snapshot_id=provider_result.raw_snapshot_id,
        cache_key=provider_result.cache_key,
    )


def _bar_audit_records(bars: list[PriceBar]) -> list[JsonValue]:
    return [
        cast(
            JsonValue,
            {
                "ticker": bar.ticker,
                "timestamp": calendar_date(bar.timestamp).isoformat(),
                "close": str(bar.close),
                "volume": bar.volume,
            },
        )
        for bar in bars
    ]


def _data_quality_metadata(provider_result: ProviderResult[MarketSnapshot]) -> JsonObject:
    metadata: JsonObject = {}
    for warning in provider_result.warnings:
        for key in (
            "rejected_snapshot",
            "returned_snapshot_ticker",
            "rejected_bar_count",
            "rejected_identity_bars",
            "excluded_future_bar_count",
            "excluded_future_bars",
        ):
            if key in warning.metadata:
                metadata[key] = warning.metadata[key]
    return metadata


def _dedupe_warnings(warnings: tuple[ProviderWarning, ...]) -> tuple[ProviderWarning, ...]:
    records: dict[tuple[str, str, str | None], ProviderWarning] = {}
    for warning in warnings:
        key = (warning.code.value, warning.message, warning.provider_name)
        records.setdefault(key, warning)
    return tuple(records.values())


def _tool_run_status(status: ProviderStatus, warnings: tuple[ProviderWarning, ...]) -> str:
    if status == ProviderStatus.OK and not warnings:
        return "successful"
    if status in {ProviderStatus.OK, ProviderStatus.PARTIAL, ProviderStatus.STALE}:
        return "partial" if warnings else "successful"
    if status == ProviderStatus.EMPTY:
        return "empty"
    return "failed" if warnings else "empty"


def _extraction_confidence(status: ProviderStatus) -> float | None:
    if status == ProviderStatus.OK:
        return 1.0
    if status in {ProviderStatus.PARTIAL, ProviderStatus.STALE}:
        return 0.75
    if status == ProviderStatus.EMPTY:
        return 0.0
    return None


__all__ = [
    "PHASE4_MARKET_DATA_SCHEMA_VERSION",
    "PHASE4_MARKET_DATA_TOOL_NAME",
    "PHASE4_MARKET_DATA_TOOL_VERSION",
    "MarketDataProvenance",
    "MarketDataToolResult",
    "Phase4MarketDataArtifact",
    "Phase4MarketDataProviderTool",
    "Phase4MarketDataTool",
    "build_market_data_artifact",
    "freshness_status_for_provider_result",
    "latest_usable_bar",
    "load_phase4_market_data_artifact",
    "market_data_artifact_payload",
    "sanitize_market_data_provider_result",
]
