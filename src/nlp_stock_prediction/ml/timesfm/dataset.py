"""TimesFM-compatible OHLCV window datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date, datetime
from math import isfinite
from typing import Literal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts import PriceBar
from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, NonEmptyStr, TickerSymbol
from nlp_stock_prediction.ml.ohlcv import (
    DatasetValidationError,
    validate_ohlcv_bars,
)
from nlp_stock_prediction.ml.ohlcv import (
    require_aware_datetime as _require_aware_datetime,
)
from nlp_stock_prediction.ml.ohlcv import (
    timestamp_key_for as _timestamp_key,
)

TimesFmTargetField = Literal["auto", "close", "adjusted_close"]
ResolvedTargetField = Literal["close", "adjusted_close"]
TimesFmSplitName = Literal["train", "validation", "test"]


class TimesFmDatasetConfig(ContractModel):
    """Windowing settings for TimesFM 2.5 local technical-analysis datasets."""

    context_length: int = Field(default=128, ge=2)
    horizon_length: int = Field(default=16, ge=1)
    stride: int = Field(default=1, ge=1)
    target_field: TimesFmTargetField = "auto"
    train_fraction: float = Field(default=0.70, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    purge_between_splits: bool = True
    split_like_move_threshold_pct: float = Field(default=0.40, gt=0.0, lt=1.0)
    adjustment_ratio_drift_threshold_pct: float = Field(default=0.05, gt=0.0, lt=1.0)
    as_of: date | datetime | None = None
    max_latest_bar_age_days: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_window_config(self) -> TimesFmDatasetConfig:
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction plus validation_fraction must leave a test split")
        if self.max_latest_bar_age_days is not None and self.as_of is None:
            raise ValueError("max_latest_bar_age_days requires as_of")
        if isinstance(self.as_of, datetime):
            _require_aware_datetime(self.as_of, field_name="as_of")
        return self


class TimesFmWindow(ContractModel):
    """One univariate TimesFM context/future window."""

    ticker: TickerSymbol
    split: TimesFmSplitName
    target_field: ResolvedTargetField
    context_length: int = Field(ge=2)
    horizon_length: int = Field(ge=1)
    context_start: date | datetime
    context_end: date | datetime
    horizon_start: date | datetime
    horizon_end: date | datetime
    context_start_index: int = Field(ge=0)
    context_end_index: int = Field(ge=0)
    horizon_start_index: int = Field(ge=0)
    horizon_end_index: int = Field(ge=0)
    context_values: tuple[float, ...]
    future_values: tuple[float, ...]
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_window(self) -> TimesFmWindow:
        if len(self.context_values) != self.context_length:
            raise ValueError("context value count must match context_length")
        if len(self.future_values) != self.horizon_length:
            raise ValueError("future value count must match horizon_length")
        if any(not isfinite(value) for value in (*self.context_values, *self.future_values)):
            raise ValueError("TimesFM window values must be finite")
        if self.context_start_index + self.context_length - 1 != self.context_end_index:
            raise ValueError("context indices must match context_length")
        if self.context_end_index + 1 != self.horizon_start_index:
            raise ValueError("future horizon must start immediately after context")
        if self.horizon_start_index + self.horizon_length - 1 != self.horizon_end_index:
            raise ValueError("horizon indices must match horizon_length")
        if _timestamp_key(self.context_start) > _timestamp_key(self.context_end):
            raise ValueError("context start must be before or equal to context end")
        if _timestamp_key(self.context_end) >= _timestamp_key(self.horizon_start):
            raise ValueError("future labels must start after context")
        if _timestamp_key(self.horizon_start) > _timestamp_key(self.horizon_end):
            raise ValueError("horizon start must be before or equal to horizon end")
        return self


class TimesFmContextWindow(ContractModel):
    """Latest context-only TimesFM window used for live inference."""

    ticker: TickerSymbol
    target_field: ResolvedTargetField
    context_length: int = Field(ge=2)
    horizon_length: int = Field(ge=1)
    context_start: date | datetime
    context_end: date | datetime
    context_start_index: int = Field(ge=0)
    context_end_index: int = Field(ge=0)
    context_values: tuple[float, ...]
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_context_window(self) -> TimesFmContextWindow:
        if len(self.context_values) != self.context_length:
            raise ValueError("context value count must match context_length")
        if any(not isfinite(value) for value in self.context_values):
            raise ValueError("TimesFM context values must be finite")
        if self.context_start_index + self.context_length - 1 != self.context_end_index:
            raise ValueError("context indices must match context_length")
        if _timestamp_key(self.context_start) > _timestamp_key(self.context_end):
            raise ValueError("context start must be before or equal to context end")
        return self


class TimesFmDataset(ContractModel):
    """A deterministic univariate TimesFM dataset with temporal train/validation/test splits."""

    ticker: TickerSymbol
    target_field: ResolvedTargetField
    context_length: int = Field(ge=2)
    horizon_length: int = Field(ge=1)
    stride: int = Field(ge=1)
    windows: tuple[TimesFmWindow, ...]
    latest_context_window: TimesFmContextWindow
    dataset_hash: NonEmptyStr
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dataset(self) -> TimesFmDataset:
        if not self.windows:
            raise ValueError("TimesFM dataset must include at least one window")
        for window in self.windows:
            if window.ticker != self.ticker:
                raise ValueError("TimesFM windows must match dataset ticker")
            if window.target_field != self.target_field:
                raise ValueError("TimesFM windows must match dataset target field")
            if window.context_length != self.context_length:
                raise ValueError("TimesFM windows must match dataset context length")
            if window.horizon_length != self.horizon_length:
                raise ValueError("TimesFM windows must match dataset horizon length")
        if self.latest_context_window.ticker != self.ticker:
            raise ValueError("TimesFM latest context must match dataset ticker")
        if self.latest_context_window.target_field != self.target_field:
            raise ValueError("TimesFM latest context must match dataset target field")
        if self.latest_context_window.context_length != self.context_length:
            raise ValueError("TimesFM latest context must match dataset context length")
        if self.latest_context_window.horizon_length != self.horizon_length:
            raise ValueError("TimesFM latest context must match dataset horizon length")
        if not self.train_windows or not self.validation_windows or not self.test_windows:
            raise ValueError("TimesFM dataset requires train, validation, and test windows")
        return self

    @property
    def train_windows(self) -> tuple[TimesFmWindow, ...]:
        return tuple(window for window in self.windows if window.split == "train")

    @property
    def validation_windows(self) -> tuple[TimesFmWindow, ...]:
        return tuple(window for window in self.windows if window.split == "validation")

    @property
    def test_windows(self) -> tuple[TimesFmWindow, ...]:
        return tuple(window for window in self.windows if window.split == "test")


def build_timesfm_dataset(
    ticker: str,
    bars: Sequence[PriceBar],
    *,
    config: TimesFmDatasetConfig | None = None,
) -> TimesFmDataset:
    """Build leakage-checked univariate TimesFM context/future windows."""

    settings = config or TimesFmDatasetConfig()
    sorted_bars = tuple(sorted(bars, key=lambda bar: _timestamp_key(bar.timestamp)))
    _validate_bars(ticker, sorted_bars, settings)
    target_field = _resolve_target_field(sorted_bars, settings.target_field)
    raw_windows = _build_raw_windows(ticker, sorted_bars, target_field, settings)
    latest_context_window = _build_latest_context_window(
        ticker, sorted_bars, target_field, settings
    )
    split_assignments, purged_window_count = _split_windows(raw_windows, settings)
    windows = tuple(
        window.model_copy(update={"split": split})
        for window, split in zip(raw_windows, split_assignments, strict=True)
        if split is not None
    )
    _validate_split_purges(windows)
    dataset_hash = _hash_dataset(
        ticker=ticker,
        target_field=target_field,
        config=settings,
        windows=windows,
        latest_context_window=latest_context_window,
    )
    metadata = _dataset_metadata(
        sorted_bars=sorted_bars,
        target_field=target_field,
        settings=settings,
        windows=windows,
        raw_window_count=len(raw_windows),
        purged_window_count=purged_window_count,
    )
    return TimesFmDataset(
        ticker=ticker,
        target_field=target_field,
        context_length=settings.context_length,
        horizon_length=settings.horizon_length,
        stride=settings.stride,
        windows=windows,
        latest_context_window=latest_context_window,
        dataset_hash=dataset_hash,
        metadata=metadata,
    )


def _build_raw_windows(
    ticker: str,
    bars: Sequence[PriceBar],
    target_field: ResolvedTargetField,
    config: TimesFmDatasetConfig,
) -> tuple[TimesFmWindow, ...]:
    windows: list[TimesFmWindow] = []
    last_start = len(bars) - config.context_length - config.horizon_length
    for start_index in range(0, last_start + 1, config.stride):
        context_start_index = start_index
        context_end_index = start_index + config.context_length - 1
        horizon_start_index = context_end_index + 1
        horizon_end_index = horizon_start_index + config.horizon_length - 1
        context_bars = bars[context_start_index : context_end_index + 1]
        future_bars = bars[horizon_start_index : horizon_end_index + 1]
        windows.append(
            TimesFmWindow(
                ticker=ticker,
                split="train",
                target_field=target_field,
                context_length=config.context_length,
                horizon_length=config.horizon_length,
                context_start=context_bars[0].timestamp,
                context_end=context_bars[-1].timestamp,
                horizon_start=future_bars[0].timestamp,
                horizon_end=future_bars[-1].timestamp,
                context_start_index=context_start_index,
                context_end_index=context_end_index,
                horizon_start_index=horizon_start_index,
                horizon_end_index=horizon_end_index,
                context_values=tuple(_target_value(bar, target_field) for bar in context_bars),
                future_values=tuple(_target_value(bar, target_field) for bar in future_bars),
                metadata={
                    "target_policy": "univariate_price_forecast",
                    "normalization": "none_timesfm_internal_instance_normalization",
                    "stride": config.stride,
                    "context_timestamps": [
                        _timestamp_to_string(bar.timestamp) for bar in context_bars
                    ],
                    "future_timestamps": [
                        _timestamp_to_string(bar.timestamp) for bar in future_bars
                    ],
                },
            )
        )
    if not windows:
        raise DatasetValidationError(
            "insufficient history: no TimesFM windows remain after context and horizon settings"
        )
    return tuple(windows)


def _build_latest_context_window(
    ticker: str,
    bars: Sequence[PriceBar],
    target_field: ResolvedTargetField,
    config: TimesFmDatasetConfig,
) -> TimesFmContextWindow:
    context_start_index = len(bars) - config.context_length
    context_end_index = len(bars) - 1
    context_bars = bars[context_start_index : context_end_index + 1]
    return TimesFmContextWindow(
        ticker=ticker,
        target_field=target_field,
        context_length=config.context_length,
        horizon_length=config.horizon_length,
        context_start=context_bars[0].timestamp,
        context_end=context_bars[-1].timestamp,
        context_start_index=context_start_index,
        context_end_index=context_end_index,
        context_values=tuple(_target_value(bar, target_field) for bar in context_bars),
        metadata={
            "target_policy": "univariate_price_forecast",
            "window_role": "latest_live_inference_context",
            "normalization": "none_timesfm_internal_instance_normalization",
            "context_timestamps": [_timestamp_to_string(bar.timestamp) for bar in context_bars],
        },
    )


def _split_windows(
    windows: Sequence[TimesFmWindow],
    config: TimesFmDatasetConfig,
) -> tuple[tuple[TimesFmSplitName | None, ...], int]:
    total_windows = len(windows)
    purge_gap = _purge_gap_window_count(config)
    minimum_windows = 3 + (2 * purge_gap)
    if total_windows < minimum_windows:
        raise DatasetValidationError(
            "insufficient history: TimesFM windows cannot fill train, validation, and test splits"
        )

    train_count = int(total_windows * config.train_fraction)
    train_count = max(1, min(train_count, total_windows - (2 * purge_gap) - 2))
    validation_start = train_count + purge_gap
    remaining_after_validation_start = total_windows - validation_start
    validation_count = int(total_windows * config.validation_fraction)
    validation_count = max(
        1, min(validation_count, remaining_after_validation_start - purge_gap - 1)
    )
    test_start = validation_start + validation_count + purge_gap
    test_count = total_windows - test_start
    if test_count < 1:
        raise DatasetValidationError(
            "insufficient history: TimesFM split settings leave no test windows"
        )

    assignments: list[TimesFmSplitName | None] = [None] * total_windows
    for index in range(train_count):
        assignments[index] = "train"
    for index in range(validation_start, validation_start + validation_count):
        assignments[index] = "validation"
    for index in range(test_start, total_windows):
        assignments[index] = "test"
    purged_window_count = assignments.count(None)
    return tuple(assignments), purged_window_count


def _purge_gap_window_count(config: TimesFmDatasetConfig) -> int:
    if not config.purge_between_splits:
        return 0
    lookback_and_label_span = config.context_length + config.horizon_length - 1
    return lookback_and_label_span // config.stride


def _validate_split_purges(windows: Sequence[TimesFmWindow]) -> None:
    train_windows = tuple(window for window in windows if window.split == "train")
    validation_windows = tuple(window for window in windows if window.split == "validation")
    test_windows = tuple(window for window in windows if window.split == "test")
    _validate_adjacent_split_purge(train_windows, validation_windows)
    _validate_adjacent_split_purge(validation_windows, test_windows)


def _validate_adjacent_split_purge(
    previous_split: Sequence[TimesFmWindow],
    next_split: Sequence[TimesFmWindow],
) -> None:
    if not previous_split or not next_split:
        return
    if previous_split[-1].horizon_end_index >= next_split[0].context_start_index:
        raise DatasetValidationError(
            "lookahead leakage: TimesFM split purge does not separate labels from features"
        )


def _validate_bars(
    ticker: str,
    bars: Sequence[PriceBar],
    config: TimesFmDatasetConfig,
) -> None:
    required_count = config.context_length + config.horizon_length
    validate_ohlcv_bars(
        ticker,
        bars,
        required_count=required_count,
        as_of=config.as_of,
        max_latest_bar_age_days=config.max_latest_bar_age_days,
        split_like_move_threshold_pct=config.split_like_move_threshold_pct,
        adjustment_ratio_drift_threshold_pct=config.adjustment_ratio_drift_threshold_pct,
        require_adjusted_close=config.target_field == "adjusted_close",
    )


def _resolve_target_field(
    bars: Sequence[PriceBar],
    requested: TimesFmTargetField,
) -> ResolvedTargetField:
    if requested == "close":
        return "close"
    if requested == "adjusted_close":
        return "adjusted_close"
    if all(bar.adjusted_close is not None for bar in bars):
        return "adjusted_close"
    return "close"


def _target_value(bar: PriceBar, target_field: ResolvedTargetField) -> float:
    value = bar.adjusted_close if target_field == "adjusted_close" else bar.close
    if value is None:
        raise DatasetValidationError("adjusted_close target requested but a bar is missing it")
    return float(value)


def _dataset_metadata(
    *,
    sorted_bars: Sequence[PriceBar],
    target_field: ResolvedTargetField,
    settings: TimesFmDatasetConfig,
    windows: Sequence[TimesFmWindow],
    raw_window_count: int,
    purged_window_count: int,
) -> JsonObject:
    split_counts: JsonObject = {
        "train": len([window for window in windows if window.split == "train"]),
        "validation": len([window for window in windows if window.split == "validation"]),
        "test": len([window for window in windows if window.split == "test"]),
    }
    return {
        "schema_version": "ml.timesfm.dataset.v1",
        "source": "ohlcv_bars",
        "target_field": target_field,
        "target_policy": "auto_adjusted_close_else_close",
        "normalization": "none_timesfm_internal_instance_normalization",
        "bar_count": len(sorted_bars),
        "raw_window_count": raw_window_count,
        "window_count": len(windows),
        "purged_window_count": purged_window_count,
        "split_counts": split_counts,
        "latest_bar_timestamp": _timestamp_to_string(sorted_bars[-1].timestamp),
        "as_of": _timestamp_to_string(settings.as_of) if settings.as_of is not None else None,
        "max_latest_bar_age_days": settings.max_latest_bar_age_days,
        "not_advice": True,
    }


def _hash_dataset(
    *,
    ticker: str,
    target_field: ResolvedTargetField,
    config: TimesFmDatasetConfig,
    windows: Sequence[TimesFmWindow],
    latest_context_window: TimesFmContextWindow,
) -> str:
    payload = {
        "ticker": ticker.upper(),
        "target_field": target_field,
        "config": _hash_config_payload(config),
        "latest_context_window": {
            "context_start": _timestamp_to_string(latest_context_window.context_start),
            "context_end": _timestamp_to_string(latest_context_window.context_end),
            "context_values": [round(value, 12) for value in latest_context_window.context_values],
        },
        "windows": [
            {
                "split": window.split,
                "context_start": _timestamp_to_string(window.context_start),
                "context_end": _timestamp_to_string(window.context_end),
                "horizon_start": _timestamp_to_string(window.horizon_start),
                "horizon_end": _timestamp_to_string(window.horizon_end),
                "context_values": [round(value, 12) for value in window.context_values],
                "future_values": [round(value, 12) for value in window.future_values],
            }
            for window in windows
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_config_payload(config: TimesFmDatasetConfig) -> dict[str, object]:
    payload = config.model_dump(mode="json")
    payload.pop("as_of", None)
    payload.pop("max_latest_bar_age_days", None)
    return payload


def _timestamp_to_string(value: date | datetime) -> str:
    return value.isoformat()


__all__ = [
    "ResolvedTargetField",
    "TimesFmContextWindow",
    "TimesFmDataset",
    "TimesFmDatasetConfig",
    "TimesFmSplitName",
    "TimesFmTargetField",
    "TimesFmWindow",
    "build_timesfm_dataset",
]
