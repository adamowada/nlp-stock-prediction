"""Phase 4 technical-package tool built from market-data artifacts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from nlp_stock_prediction.analysis.ml_signal import (
    apply_technical_ml_signal,
    build_timesfm_forecast_signal,
)
from nlp_stock_prediction.analysis.technical import analyze_technical_snapshot
from nlp_stock_prediction.contracts import (
    AuditArtifact,
    ContractModel,
    FreshnessStatus,
    JsonObject,
    MarketSnapshot,
    MetricValue,
    NonEmptyStr,
    PriceBar,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    TechnicalAnalysis,
    TechnicalMlSignal,
    TickerSymbol,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.ml.ohlcv import calendar_date, timestamp_key_for
from nlp_stock_prediction.ml.timesfm.contracts import TimesFmForecastArtifact
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase2_common import (
    file_sha256,
    stable_digest,
    symbol_slug,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase4_execution import safe_phase4_tool_execution
from nlp_stock_prediction.orchestration.phase4_market_data import (
    MarketDataToolResult,
    Phase4MarketDataArtifact,
    freshness_status_for_provider_result,
    latest_usable_bar,
    load_phase4_market_data_artifact,
    sanitize_market_data_provider_result,
)
from nlp_stock_prediction.providers._base import provider_warning
from nlp_stock_prediction.storage.records import ToolRunRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE4_TECHNICAL_PACKAGE_SCHEMA_VERSION = "phase4.technical-package.v1"
PHASE4_TECHNICAL_PACKAGE_TOOL_NAME = "phase4_technical_package"
PHASE4_TECHNICAL_PACKAGE_TOOL_VERSION = PHASE4_TECHNICAL_PACKAGE_SCHEMA_VERSION
TECHNICAL_PACKAGE_PREDICTION_POLICY = (
    "Technical indicators and raw TimesFM sidecars are audit context only; they cannot create a "
    "reportable prediction without independent evidence."
)

TechnicalPackageStatus = Literal["ok", "warning", "unavailable"]
MarketDataInput = (
    MarketDataToolResult | Phase4MarketDataArtifact | ProviderResult[MarketSnapshot] | Path | str
)
TimesFmSidecarInput = TimesFmForecastArtifact | Path | str | None


class TechnicalBaselineContext(ContractModel):
    """Deterministic baseline context derived from OHLCV bars."""

    symbol: TickerSymbol
    latest_bar_timestamp: date | datetime | None = None
    latest_close: Decimal | None = None
    bar_count: int = Field(ge=0)
    close_return_1d: float | None = None
    close_return_5d: float | None = None
    close_return_20d: float | None = None
    average_volume_20d: float | None = None
    average_dollar_volume_20d: float | None = None
    recent_low_20d: Decimal | None = None
    recent_high_20d: Decimal | None = None
    high_low_range_20d_pct: float | None = None
    metadata: JsonObject = Field(default_factory=dict)


class Phase4TechnicalPackageArtifact(ContractModel):
    """Stable JSON payload written by the Phase 4 technical-package tool."""

    schema_version: NonEmptyStr = PHASE4_TECHNICAL_PACKAGE_SCHEMA_VERSION
    run_id: NonEmptyStr
    tool_run_id: NonEmptyStr
    artifact_id: NonEmptyStr
    symbol: TickerSymbol
    generated_at: datetime
    status: TechnicalPackageStatus
    freshness_status: FreshnessStatus
    market_data_status: ProviderStatus | None = None
    market_data_artifact_id: str | None = None
    market_data_artifact_path: str | None = None
    market_data_artifact_sha256: str | None = None
    market_data_source_query_id: str | None = None
    latest_usable_bar: PriceBar | None = None
    baseline_context: TechnicalBaselineContext
    technical_analysis: TechnicalAnalysis
    deterministic_indicators: tuple[MetricValue, ...] = ()
    timesfm_sidecar: TechnicalMlSignal | None = None
    timesfm_applied_to_analysis: bool = False
    prediction_policy: NonEmptyStr = TECHNICAL_PACKAGE_PREDICTION_POLICY
    warnings: tuple[ProviderWarning, ...] = ()
    assumptions: tuple[str, ...] = ()
    metadata: JsonObject = Field(default_factory=dict)


@dataclass(frozen=True)
class TechnicalPackageToolResult:
    """Technical-package output plus the indexed artifact metadata."""

    technical_analysis: TechnicalAnalysis
    artifact: AuditArtifact
    artifact_payload: Phase4TechnicalPackageArtifact
    tool_run_id: str


@dataclass(frozen=True)
class _MarketDataBundle:
    provider_result: ProviderResult[MarketSnapshot] | None
    market_artifact: Phase4MarketDataArtifact | None
    artifact_id: str | None
    artifact_path: str | None
    artifact_sha256: str | None
    warnings: tuple[ProviderWarning, ...] = ()


@dataclass(frozen=True)
class _TimesFmBundle:
    signal: TechnicalMlSignal | None
    warnings: tuple[ProviderWarning, ...] = ()


@dataclass(frozen=True)
class Phase4TechnicalPackageTool:
    """Compute deterministic indicators and write a technical-package artifact."""

    store: SQLiteStore
    repo_root: Path
    artifact_dir: Path
    now: Callable[[], datetime] = utc_now

    def run(
        self,
        *,
        run_id: str,
        symbol: str,
        market_data: MarketDataInput,
        timesfm_sidecar: TimesFmSidecarInput = None,
        instrument_id: str | None = None,
    ) -> TechnicalPackageToolResult:
        normalized_symbol = _normalize_symbol(symbol)
        started_at = self.now()
        generated_at = self.now()
        tool_run_id = _technical_package_tool_run_id(run_id, normalized_symbol)
        artifact_id = _technical_package_artifact_id(run_id, normalized_symbol)
        market_bundle = _sanitize_market_bundle(
            _coerce_market_data_input(market_data, generated_at, repo_root=self.repo_root),
            symbol=normalized_symbol,
        )
        snapshot = _snapshot_for_technical_input(
            symbol=normalized_symbol,
            provider_result=market_bundle.provider_result,
        )
        latest_bar = latest_usable_bar(snapshot)
        market_warnings = (
            market_bundle.provider_result.warnings
            if market_bundle.provider_result is not None
            else ()
        )
        timesfm_bundle = _coerce_timesfm_sidecar(
            timesfm_sidecar,
            generated_at,
            symbol=normalized_symbol,
        )
        internal_warnings = _technical_warnings(
            symbol=normalized_symbol,
            snapshot=snapshot,
            freshness_status=_freshness_status(market_bundle),
            market_data_status=(
                market_bundle.provider_result.status
                if market_bundle.provider_result is not None
                else None
            ),
            timesfm_signal=timesfm_bundle.signal,
            generated_at=generated_at,
        )
        warnings = _dedupe_warnings(
            (
                *market_bundle.warnings,
                *market_warnings,
                *timesfm_bundle.warnings,
                *internal_warnings,
            )
        )

        analysis = analyze_technical_snapshot(
            snapshot,
            as_of=calendar_date(latest_bar.timestamp) if latest_bar is not None else generated_at,
        )
        has_deterministic_bars = bool(snapshot.bars)
        timesfm_applied = timesfm_bundle.signal is not None and has_deterministic_bars
        if timesfm_applied:
            analysis = apply_technical_ml_signal(analysis, timesfm_bundle.signal)
        analysis = analysis.model_copy(
            update={
                "warnings": warnings,
                "assumptions": _dedupe_strings(
                    (
                        *analysis.assumptions,
                        "Market data is technical context, not a trade instruction.",
                        TECHNICAL_PACKAGE_PREDICTION_POLICY,
                        *_timesfm_assumptions(
                            timesfm_bundle.signal,
                            has_deterministic_bars=has_deterministic_bars,
                        ),
                    )
                ),
            }
        )
        status = _package_status(
            has_bars=has_deterministic_bars,
            freshness_status=_freshness_status(market_bundle),
            warnings=warnings,
        )
        artifact_payload = Phase4TechnicalPackageArtifact(
            run_id=run_id,
            tool_run_id=tool_run_id,
            artifact_id=artifact_id,
            symbol=normalized_symbol,
            generated_at=generated_at,
            status=status,
            freshness_status=_freshness_status(market_bundle),
            market_data_status=(
                market_bundle.provider_result.status
                if market_bundle.provider_result is not None
                else None
            ),
            market_data_artifact_id=market_bundle.artifact_id,
            market_data_artifact_path=market_bundle.artifact_path,
            market_data_artifact_sha256=market_bundle.artifact_sha256,
            market_data_source_query_id=(
                market_bundle.market_artifact.provenance.source_query_id
                if market_bundle.market_artifact is not None
                else None
            ),
            latest_usable_bar=latest_bar,
            baseline_context=build_baseline_context(snapshot),
            technical_analysis=analysis,
            deterministic_indicators=analysis.metrics,
            timesfm_sidecar=timesfm_bundle.signal,
            timesfm_applied_to_analysis=timesfm_applied,
            warnings=warnings,
            assumptions=analysis.assumptions,
            metadata={
                "instrument_id": instrument_id,
                "bar_count": len(snapshot.bars),
                "market_data_artifact_id": market_bundle.artifact_id,
                "timesfm_sidecar_present": timesfm_bundle.signal is not None,
                "timesfm_sidecar_source_only": True,
            },
        )
        inputs: JsonObject = {
            "symbol": normalized_symbol,
            "instrument_id": instrument_id,
            "market_data_artifact_id": market_bundle.artifact_id,
            "market_data_status": (
                market_bundle.provider_result.status.value
                if market_bundle.provider_result is not None
                else None
            ),
            "timesfm_sidecar_present": timesfm_bundle.signal is not None,
        }
        with safe_phase4_tool_execution(
            store=self.store,
            artifact_roots=(self.artifact_dir,),
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=PHASE4_TECHNICAL_PACKAGE_TOOL_NAME,
            tool_version=PHASE4_TECHNICAL_PACKAGE_TOOL_VERSION,
            started_at=started_at,
            inputs=inputs,
        ):
            self.store.record_tool_run(
                ToolRunRecord(
                    tool_run_id=tool_run_id,
                    run_id=run_id,
                    tool_name=PHASE4_TECHNICAL_PACKAGE_TOOL_NAME,
                    tool_version=PHASE4_TECHNICAL_PACKAGE_TOOL_VERSION,
                    status=_tool_run_status(status, warnings),
                    started_at=started_at,
                    completed_at=generated_at,
                    inputs=inputs,
                    warnings=tuple(warning.message for warning in warnings),
                )
            )
            artifact = ArtifactIndex.for_directory(
                store=self.store,
                repo_root=self.repo_root,
                base_dir=self.artifact_dir,
                created_at=generated_at,
                produced_by=PHASE4_TECHNICAL_PACKAGE_TOOL_NAME,
                tool_run_id=tool_run_id,
                schema_version=PHASE4_TECHNICAL_PACKAGE_SCHEMA_VERSION,
            ).write_json(
                artifact_id=artifact_id,
                artifact_type="technical_package",
                filename=f"technical-package/{symbol_slug(normalized_symbol)}.json",
                payload=technical_package_artifact_payload(artifact_payload),
                record_count=len(analysis.metrics),
                metadata={
                    "symbol": normalized_symbol,
                    "instrument_id": instrument_id,
                    "status": status,
                    "freshness_status": artifact_payload.freshness_status.value,
                    "market_data_artifact_id": market_bundle.artifact_id,
                    "latest_usable_bar": (
                        calendar_date(latest_bar.timestamp).isoformat() if latest_bar else None
                    ),
                    "timesfm_sidecar_present": timesfm_bundle.signal is not None,
                    "timesfm_applied_to_analysis": timesfm_applied,
                    "warning_count": len(warnings),
                },
            )
            return TechnicalPackageToolResult(
                technical_analysis=analysis,
                artifact=artifact,
                artifact_payload=artifact_payload,
                tool_run_id=tool_run_id,
            )


def build_baseline_context(snapshot: MarketSnapshot) -> TechnicalBaselineContext:
    """Build deterministic comparison context from sorted OHLCV bars."""

    bars = _sorted_bars(snapshot.bars)
    latest = bars[-1] if bars else None
    recent = bars[-min(20, len(bars)) :] if bars else ()
    average_volume = round(sum(bar.volume for bar in recent) / len(recent), 4) if recent else None
    average_dollar_volume = (
        round(
            sum(float(bar.close) * bar.volume for bar in recent) / len(recent),
            4,
        )
        if recent
        else None
    )
    recent_low = min((bar.low for bar in recent), default=None)
    recent_high = max((bar.high for bar in recent), default=None)
    high_low_range_pct = None
    if latest is not None and recent_low is not None and recent_high is not None:
        high_low_range_pct = round(float((recent_high - recent_low) / latest.close), 6)
    return TechnicalBaselineContext(
        symbol=snapshot.ticker,
        latest_bar_timestamp=latest.timestamp if latest is not None else None,
        latest_close=latest.close if latest is not None else None,
        bar_count=len(bars),
        close_return_1d=_close_return(bars, 1),
        close_return_5d=_close_return(bars, 5),
        close_return_20d=_close_return(bars, 20),
        average_volume_20d=average_volume,
        average_dollar_volume_20d=average_dollar_volume,
        recent_low_20d=recent_low,
        recent_high_20d=recent_high,
        high_low_range_20d_pct=high_low_range_pct,
        metadata={"windows": [1, 5, 20], "baseline": "recent_ohlcv_history"},
    )


def technical_package_artifact_payload(artifact: Phase4TechnicalPackageArtifact) -> JsonObject:
    """Serialize a Phase 4 technical-package artifact to stable JSON-compatible data."""

    return cast(JsonObject, artifact.model_dump(mode="json"))


def load_phase4_technical_package_artifact(path: Path) -> Phase4TechnicalPackageArtifact:
    """Load a previously written technical-package artifact."""

    return Phase4TechnicalPackageArtifact.model_validate_json(path.read_text(encoding="utf-8"))


def _coerce_market_data_input(
    market_data: MarketDataInput,
    generated_at: datetime,
    *,
    repo_root: Path,
) -> _MarketDataBundle:
    if isinstance(market_data, MarketDataToolResult):
        return _MarketDataBundle(
            provider_result=market_data.provider_result,
            market_artifact=market_data.artifact_payload,
            artifact_id=market_data.artifact.artifact_id,
            artifact_path=market_data.artifact.path,
            artifact_sha256=market_data.artifact.sha256,
        )
    if isinstance(market_data, Phase4MarketDataArtifact):
        return _MarketDataBundle(
            provider_result=market_data.provider_result,
            market_artifact=market_data,
            artifact_id=market_data.artifact_id,
            artifact_path=None,
            artifact_sha256=None,
        )
    if isinstance(market_data, ProviderResult):
        return _MarketDataBundle(
            provider_result=market_data,
            market_artifact=None,
            artifact_id=None,
            artifact_path=None,
            artifact_sha256=None,
        )
    raw_path = Path(market_data)
    path = raw_path if raw_path.is_absolute() else repo_root / raw_path
    try:
        artifact = load_phase4_market_data_artifact(path)
        return _MarketDataBundle(
            provider_result=artifact.provider_result,
            market_artifact=artifact,
            artifact_id=artifact.artifact_id,
            artifact_path=str(path),
            artifact_sha256=file_sha256(path),
        )
    except (OSError, ValueError) as exc:
        return _MarketDataBundle(
            provider_result=None,
            market_artifact=None,
            artifact_id=None,
            artifact_path=str(path),
            artifact_sha256=None,
            warnings=(
                _warning(
                    code=WarningCode.MALFORMED_RESPONSE,
                    severity=WarningSeverity.WARNING,
                    message=f"Market-data artifact could not be loaded: {exc}",
                    occurred_at=generated_at,
                    metadata={"path": str(path)},
                ),
            ),
        )


def _coerce_timesfm_sidecar(
    timesfm_sidecar: TimesFmSidecarInput,
    generated_at: datetime,
    *,
    symbol: str,
) -> _TimesFmBundle:
    if timesfm_sidecar is None:
        return _TimesFmBundle(signal=None)
    if isinstance(timesfm_sidecar, TimesFmForecastArtifact):
        return _timesfm_bundle_from_artifact(
            timesfm_sidecar,
            generated_at=generated_at,
            symbol=symbol,
        )
    path = Path(timesfm_sidecar)
    try:
        artifact = TimesFmForecastArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        return _timesfm_bundle_from_artifact(
            artifact,
            generated_at=generated_at,
            symbol=symbol,
            artifact_path=path,
            artifact_sha256=file_sha256(path),
        )
    except (OSError, ValueError) as exc:
        return _TimesFmBundle(
            signal=None,
            warnings=(
                _warning(
                    code=WarningCode.MALFORMED_RESPONSE,
                    severity=WarningSeverity.WARNING,
                    message=f"TimesFM sidecar artifact could not be loaded: {exc}",
                    occurred_at=generated_at,
                    metadata={"path": str(path), "sidecar_only": True},
                ),
            ),
        )


def _sanitize_market_bundle(
    bundle: _MarketDataBundle,
    *,
    symbol: str,
) -> _MarketDataBundle:
    if bundle.provider_result is None:
        return bundle
    sanitized = sanitize_market_data_provider_result(
        bundle.provider_result,
        requested_symbol=symbol,
    )
    market_artifact = bundle.market_artifact
    if market_artifact is not None and sanitized != bundle.provider_result:
        market_artifact = market_artifact.model_copy(
            update={
                "provider_result": sanitized,
                "bars": sanitized.data.bars if sanitized.data is not None else (),
                "bar_count": len(sanitized.data.bars) if sanitized.data is not None else 0,
                "latest_usable_bar": latest_usable_bar(sanitized.data),
                "warnings": sanitized.warnings,
                "status": sanitized.status,
                "freshness_status": freshness_status_for_provider_result(sanitized),
            }
        )
    return _MarketDataBundle(
        provider_result=sanitized,
        market_artifact=market_artifact,
        artifact_id=bundle.artifact_id,
        artifact_path=bundle.artifact_path,
        artifact_sha256=bundle.artifact_sha256,
        warnings=bundle.warnings,
    )


def _timesfm_bundle_from_artifact(
    artifact: TimesFmForecastArtifact,
    *,
    generated_at: datetime,
    symbol: str,
    artifact_path: Path | None = None,
    artifact_sha256: str | None = None,
) -> _TimesFmBundle:
    if _normalize_symbol(artifact.ticker) != symbol:
        return _TimesFmBundle(
            signal=None,
            warnings=(
                _warning(
                    code=WarningCode.SCHEMA_MISMATCH,
                    severity=WarningSeverity.WARNING,
                    message=(
                        f"TimesFM sidecar ticker {artifact.ticker} did not match requested "
                        f"symbol {symbol}; sidecar was not applied."
                    ),
                    occurred_at=generated_at,
                    metadata={
                        "symbol": symbol,
                        "sidecar_ticker": artifact.ticker,
                        "sidecar_only": True,
                    },
                ),
            ),
        )
    if artifact.status == "unavailable":
        return _TimesFmBundle(
            signal=None,
            warnings=(
                _warning(
                    code=WarningCode.NO_DATA,
                    severity=WarningSeverity.WARNING,
                    message="TimesFM sidecar was unavailable and was not applied.",
                    occurred_at=generated_at,
                    metadata={
                        "symbol": symbol,
                        "sidecar_ticker": artifact.ticker,
                        "timesfm_status": artifact.status,
                        "warning_ids": list(artifact.warning_ids),
                        "sidecar_only": True,
                    },
                ),
            ),
        )
    return _TimesFmBundle(
        signal=build_timesfm_forecast_signal(
            artifact,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
        )
    )


def _snapshot_for_technical_input(
    *,
    symbol: str,
    provider_result: ProviderResult[MarketSnapshot] | None,
) -> MarketSnapshot:
    if provider_result is not None and provider_result.data is not None:
        return provider_result.data
    return MarketSnapshot(ticker=symbol, bars=())


def _technical_warnings(
    *,
    symbol: str,
    snapshot: MarketSnapshot,
    freshness_status: FreshnessStatus,
    market_data_status: ProviderStatus | None,
    timesfm_signal: TechnicalMlSignal | None,
    generated_at: datetime,
) -> tuple[ProviderWarning, ...]:
    warnings: list[ProviderWarning] = []
    if not snapshot.bars:
        warnings.append(
            _warning(
                code=WarningCode.NO_DATA,
                severity=WarningSeverity.WARNING,
                message=(
                    f"No usable OHLCV bars were available for {symbol}; "
                    "technical package is unavailable."
                ),
                occurred_at=generated_at,
                metadata={
                    "symbol": symbol,
                    "market_data_status": _status_value(market_data_status),
                },
            )
        )
    if freshness_status == FreshnessStatus.STALE:
        warnings.append(
            _warning(
                code=WarningCode.STALE_DATA,
                severity=WarningSeverity.WARNING,
                message=(
                    f"Market data for {symbol} is stale; deterministic indicators are context only."
                ),
                occurred_at=generated_at,
                metadata={"symbol": symbol},
            )
        )
    if timesfm_signal is not None and not snapshot.bars:
        warnings.append(
            _warning(
                code=WarningCode.UNSUPPORTED_CLAIM,
                severity=WarningSeverity.WARNING,
                message=(
                    "TimesFM sidecar was not applied because raw TimesFM cannot be the sole "
                    "source of a reportable prediction."
                ),
                occurred_at=generated_at,
                metadata={
                    "symbol": symbol,
                    "sidecar_only": True,
                    "timesfm_status": timesfm_signal.status,
                },
            )
        )
    return tuple(warnings)


def _warning(
    *,
    code: WarningCode,
    severity: WarningSeverity,
    message: str,
    occurred_at: datetime,
    metadata: JsonObject | None = None,
) -> ProviderWarning:
    return provider_warning(
        provider_name=PHASE4_TECHNICAL_PACKAGE_TOOL_NAME,
        code=code,
        severity=severity,
        message=message,
        occurred_at=occurred_at,
        metadata={} if metadata is None else metadata,
    )


def _freshness_status(bundle: _MarketDataBundle) -> FreshnessStatus:
    if bundle.market_artifact is not None:
        return bundle.market_artifact.freshness_status
    if bundle.provider_result is not None:
        return freshness_status_for_provider_result(bundle.provider_result)
    return FreshnessStatus.MISSING


def _package_status(
    *,
    has_bars: bool,
    freshness_status: FreshnessStatus,
    warnings: tuple[ProviderWarning, ...],
) -> TechnicalPackageStatus:
    if not has_bars:
        return "unavailable"
    if freshness_status == FreshnessStatus.STALE or warnings:
        return "warning"
    return "ok"


def _tool_run_status(
    status: TechnicalPackageStatus,
    warnings: tuple[ProviderWarning, ...],
) -> str:
    if status == "ok" and not warnings:
        return "successful"
    if status == "unavailable":
        return "empty"
    return "partial" if warnings else "successful"


def _sorted_bars(bars: Sequence[PriceBar]) -> tuple[PriceBar, ...]:
    return tuple(sorted(bars, key=lambda bar: timestamp_key_for(bar.timestamp)))


def _close_return(bars: Sequence[PriceBar], sessions: int) -> float | None:
    if len(bars) <= sessions:
        return None
    latest = bars[-1].close
    previous = bars[-sessions - 1].close
    if previous == Decimal("0"):
        return None
    return round(float((latest - previous) / previous), 6)


def _technical_package_tool_run_id(run_id: str, symbol: str) -> str:
    return f"tool-phase4-technical-package-{stable_digest(f'{run_id}:{symbol}')}"


def _technical_package_artifact_id(run_id: str, symbol: str) -> str:
    return f"artifact-technical-package-{stable_digest(f'{run_id}:{symbol}')}"


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _timesfm_assumptions(
    signal: TechnicalMlSignal | None,
    *,
    has_deterministic_bars: bool,
) -> tuple[str, ...]:
    if signal is None or has_deterministic_bars:
        return ()
    return (
        "TimesFM sidecar was not applied because deterministic OHLCV indicators were unavailable.",
    )


def _dedupe_warnings(warnings: tuple[ProviderWarning, ...]) -> tuple[ProviderWarning, ...]:
    records: dict[tuple[str, str, str | None], ProviderWarning] = {}
    for warning in warnings:
        key = (warning.code.value, warning.message, warning.provider_name)
        records.setdefault(key, warning)
    return tuple(records.values())


def _status_value(status: ProviderStatus | None) -> str | None:
    return status.value if status is not None else None


__all__ = [
    "PHASE4_TECHNICAL_PACKAGE_SCHEMA_VERSION",
    "PHASE4_TECHNICAL_PACKAGE_TOOL_NAME",
    "PHASE4_TECHNICAL_PACKAGE_TOOL_VERSION",
    "TECHNICAL_PACKAGE_PREDICTION_POLICY",
    "MarketDataInput",
    "Phase4TechnicalPackageArtifact",
    "Phase4TechnicalPackageTool",
    "TechnicalBaselineContext",
    "TechnicalPackageStatus",
    "TechnicalPackageToolResult",
    "TimesFmSidecarInput",
    "build_baseline_context",
    "load_phase4_technical_package_artifact",
    "technical_package_artifact_payload",
]
