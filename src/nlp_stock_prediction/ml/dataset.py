"""Dataset construction for local ML-assisted technical analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from math import isfinite, sqrt

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts import PriceBar
from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, NonEmptyStr, TickerSymbol

FEATURE_NAMES: tuple[str, ...] = (
    "candle_body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "intraday_return_pct",
    "overnight_gap_pct",
    "true_range_pct",
    "close_to_close_return_pct",
    "rolling_return_mean_pct",
    "rolling_volatility_pct",
    "relative_volume",
    "volume_change_pct",
    "close_position_pct",
)


class DatasetValidationError(ValueError):
    """Raised when OHLCV bars are unsafe for ML dataset generation."""


class TechnicalDatasetConfig(ContractModel):
    """Feature and label settings for deterministic technical-analysis datasets."""

    feature_window: int = Field(default=5, ge=2)
    label_horizon_sessions: int = Field(default=1, ge=1)
    positive_return_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    minimum_rows: int = Field(default=3, ge=1)
    split_like_move_threshold_pct: float = Field(default=0.40, gt=0.0, lt=1.0)
    adjustment_ratio_drift_threshold_pct: float = Field(default=0.05, gt=0.0, lt=1.0)
    as_of: date | datetime | None = None
    max_latest_bar_age_days: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_freshness_gate(self) -> TechnicalDatasetConfig:
        if self.max_latest_bar_age_days is not None and self.as_of is None:
            raise ValueError("max_latest_bar_age_days requires as_of")
        if isinstance(self.as_of, datetime):
            _require_aware_datetime(self.as_of, field_name="as_of")
        return self


class TechnicalFeatureRow(ContractModel):
    """One supervised training example built only from bars known at ``feature_end``."""

    ticker: TickerSymbol
    feature_start: date | datetime
    feature_end: date | datetime
    label_end: date | datetime
    feature_end_index: int = Field(ge=0)
    label_horizon_sessions: int = Field(ge=1)
    feature_values: tuple[float, ...]
    forward_return: float
    target: int = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_row(self) -> TechnicalFeatureRow:
        if len(self.feature_values) != len(FEATURE_NAMES):
            raise ValueError("feature value count must match feature names")
        if any(not isfinite(value) for value in self.feature_values):
            raise ValueError("feature values must be finite")
        if not isfinite(self.forward_return):
            raise ValueError("forward return must be finite")
        if _timestamp_key(self.feature_start) > _timestamp_key(self.feature_end):
            raise ValueError("feature start must be before or equal to feature end")
        if _timestamp_key(self.label_end) <= _timestamp_key(self.feature_end):
            raise ValueError("label end must be after feature end")
        return self


class TechnicalDataset(ContractModel):
    """Contract-shaped ML dataset with stable feature names and provenance metadata."""

    ticker: TickerSymbol
    feature_names: tuple[NonEmptyStr, ...]
    label_name: NonEmptyStr = "forward_return_positive"
    label_horizon_sessions: int = Field(ge=1)
    positive_return_threshold: float
    rows: tuple[TechnicalFeatureRow, ...]
    dataset_hash: NonEmptyStr
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dataset(self) -> TechnicalDataset:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("dataset feature names must match the canonical technical feature set")
        if len(self.rows) < 1:
            raise ValueError("dataset must contain at least one row")
        for row in self.rows:
            if row.ticker != self.ticker:
                raise ValueError("dataset rows must match dataset ticker")
            if row.label_horizon_sessions != self.label_horizon_sessions:
                raise ValueError("dataset rows must match label horizon")
        return self


class DatasetSplit(ContractModel):
    """Temporal train/validation split with a purged gap to prevent label overlap."""

    train_rows: tuple[TechnicalFeatureRow, ...]
    validation_rows: tuple[TechnicalFeatureRow, ...]
    purged_row_count: int = Field(ge=0)
    train_fraction: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_split(self) -> DatasetSplit:
        if not self.train_rows:
            raise ValueError("temporal split requires at least one train row")
        if not self.validation_rows:
            raise ValueError("temporal split requires at least one validation row")
        if _timestamp_key(self.train_rows[-1].label_end) >= _timestamp_key(
            self.validation_rows[0].feature_end
        ):
            raise ValueError("train labels must end before validation features begin")
        return self


def build_technical_dataset(
    ticker: str,
    bars: Sequence[PriceBar],
    *,
    config: TechnicalDatasetConfig | None = None,
) -> TechnicalDataset:
    """Build feature rows and forward-return labels from validated OHLCV bars."""

    settings = config or TechnicalDatasetConfig()
    sorted_bars = tuple(sorted(bars, key=lambda bar: _timestamp_key(bar.timestamp)))
    _validate_bars(ticker, sorted_bars, settings)

    rows: list[TechnicalFeatureRow] = []
    first_feature_index = settings.feature_window
    last_feature_index = len(sorted_bars) - settings.label_horizon_sessions - 1
    for feature_end_index in range(first_feature_index, last_feature_index + 1):
        row = _build_feature_row(ticker, sorted_bars, feature_end_index, settings)
        rows.append(row)

    if len(rows) < settings.minimum_rows:
        raise DatasetValidationError(
            "insufficient history: not enough rows remain after feature windows and labels"
        )

    row_tuple = tuple(rows)
    dataset_hash = _hash_dataset(
        ticker=ticker,
        config=settings,
        feature_names=FEATURE_NAMES,
        rows=row_tuple,
    )
    return TechnicalDataset(
        ticker=ticker,
        feature_names=FEATURE_NAMES,
        label_horizon_sessions=settings.label_horizon_sessions,
        positive_return_threshold=settings.positive_return_threshold,
        rows=row_tuple,
        dataset_hash=dataset_hash,
        metadata={
            "source": "ohlcv_bars",
            "feature_window": settings.feature_window,
            "feature_policy": "features use current and historical bars only",
            "label_policy": "binary target is based on forward close return",
            "latest_bar_timestamp": _timestamp_to_string(sorted_bars[-1].timestamp),
            "as_of": _timestamp_to_string(settings.as_of) if settings.as_of is not None else None,
            "max_latest_bar_age_days": settings.max_latest_bar_age_days,
            "row_count": len(row_tuple),
            "not_advice": True,
        },
    )


def temporal_train_validation_split(
    dataset: TechnicalDataset,
    *,
    train_fraction: float = 0.70,
) -> DatasetSplit:
    """Create a chronological split with an embargo equal to the label horizon."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    row_count = len(dataset.rows)
    train_count = int(row_count * train_fraction)
    train_count = max(1, min(train_count, row_count - 1))
    validation_start = train_count + dataset.label_horizon_sessions
    if validation_start >= row_count:
        raise DatasetValidationError(
            "insufficient history: temporal split leaves no validation rows after purge"
        )
    return DatasetSplit(
        train_rows=dataset.rows[:train_count],
        validation_rows=dataset.rows[validation_start:],
        purged_row_count=validation_start - train_count,
        train_fraction=train_fraction,
    )


def _validate_bars(
    ticker: str,
    bars: Sequence[PriceBar],
    config: TechnicalDatasetConfig,
) -> None:
    required_count = config.feature_window + config.label_horizon_sessions + config.minimum_rows
    if len(bars) < required_count:
        raise DatasetValidationError(
            f"insufficient history: need at least {required_count} bars, got {len(bars)}"
        )
    _validate_freshness(bars, config)

    timestamps: set[int] = set()
    adjusted_close_presence: list[bool] = []
    previous_close: Decimal | None = None
    previous_adjustment_ratio: Decimal | None = None
    for bar in bars:
        if bar.ticker != ticker.upper():
            raise DatasetValidationError("ticker mismatch between requested ticker and bars")
        timestamp_key = _timestamp_key(bar.timestamp)
        if timestamp_key in timestamps:
            raise DatasetValidationError("duplicate bar timestamp detected")
        timestamps.add(timestamp_key)
        _validate_ohlcv_fields(bar)

        adjusted_close_presence.append(bar.adjusted_close is not None)
        if previous_close is not None:
            close_return = abs((_as_decimal(bar.close) - previous_close) / previous_close)
            if close_return >= Decimal(str(config.split_like_move_threshold_pct)):
                raise DatasetValidationError(
                    "split leakage: split-like close-to-close discontinuity detected"
                )
        previous_close = _as_decimal(bar.close)

        if bar.adjusted_close is not None:
            adjustment_ratio = _as_decimal(bar.adjusted_close) / _as_decimal(bar.close)
            if previous_adjustment_ratio is not None:
                ratio_drift = abs(adjustment_ratio - previous_adjustment_ratio)
                drift_pct = ratio_drift / max(abs(previous_adjustment_ratio), Decimal("0.0001"))
                if drift_pct >= Decimal(str(config.adjustment_ratio_drift_threshold_pct)):
                    raise DatasetValidationError(
                        "split leakage: changing adjusted-close ratio detected"
                    )
            previous_adjustment_ratio = adjustment_ratio
        elif previous_adjustment_ratio is not None:
            raise DatasetValidationError(
                "split leakage: partial adjusted_close history can leak split adjustments"
            )

    if any(adjusted_close_presence) and not all(adjusted_close_presence):
        raise DatasetValidationError(
            "split leakage: partial adjusted_close history can leak split adjustments"
        )


def _validate_freshness(
    bars: Sequence[PriceBar],
    config: TechnicalDatasetConfig,
) -> None:
    if config.as_of is None:
        return
    latest_timestamp = bars[-1].timestamp
    if _timestamp_after(latest_timestamp, config.as_of):
        raise DatasetValidationError("lookahead leakage: latest bar is after dataset as_of")
    if config.max_latest_bar_age_days is None:
        return
    age_days = (_calendar_date(config.as_of) - _calendar_date(latest_timestamp)).days
    if age_days > config.max_latest_bar_age_days:
        raise DatasetValidationError(
            "stale data: latest bar is older than the configured freshness gate"
        )


def _validate_ohlcv_fields(bar: PriceBar) -> None:
    raw_values = {
        "open": getattr(bar, "open", None),
        "high": getattr(bar, "high", None),
        "low": getattr(bar, "low", None),
        "close": getattr(bar, "close", None),
        "volume": getattr(bar, "volume", None),
    }
    if any(value is None for value in raw_values.values()):
        raise DatasetValidationError("missing OHLCV field detected")
    open_price = _as_decimal(raw_values["open"])
    high = _as_decimal(raw_values["high"])
    low = _as_decimal(raw_values["low"])
    close = _as_decimal(raw_values["close"])
    volume = _as_int(raw_values["volume"])
    if min(open_price, high, low, close) <= Decimal("0"):
        raise DatasetValidationError("impossible OHLCV price: prices must be positive")
    if high < low or high < max(open_price, close) or low > min(open_price, close):
        raise DatasetValidationError("impossible OHLCV price ordering")
    if volume < 0:
        raise DatasetValidationError("impossible OHLCV volume: volume must be non-negative")
    if bar.adjusted_close is not None and _as_decimal(bar.adjusted_close) <= Decimal("0"):
        raise DatasetValidationError("impossible adjusted close: must be positive")


def _build_feature_row(
    ticker: str,
    bars: Sequence[PriceBar],
    feature_end_index: int,
    config: TechnicalDatasetConfig,
) -> TechnicalFeatureRow:
    current = bars[feature_end_index]
    previous = bars[feature_end_index - 1]
    label = bars[feature_end_index + config.label_horizon_sessions]
    feature_start_index = feature_end_index - config.feature_window + 1
    returns = [
        _close_to_close_return(bars[index - 1], bars[index])
        for index in range(feature_end_index - config.feature_window + 1, feature_end_index + 1)
    ]
    prior_volumes = list(
        bars[index].volume
        for index in range(feature_end_index - config.feature_window, feature_end_index)
    )
    feature_values = (
        _candle_body_pct(current),
        _upper_shadow_pct(current),
        _lower_shadow_pct(current),
        _intraday_return_pct(current),
        _overnight_gap_pct(previous, current),
        _true_range_pct(previous, current),
        _close_to_close_return(previous, current),
        _mean(returns),
        _population_stddev(returns),
        _relative_volume(current.volume, prior_volumes),
        _volume_change_pct(previous.volume, current.volume),
        _close_position_pct(current),
    )
    forward_return = _forward_return(current, label)
    return TechnicalFeatureRow(
        ticker=ticker,
        feature_start=bars[feature_start_index].timestamp,
        feature_end=current.timestamp,
        label_end=label.timestamp,
        feature_end_index=feature_end_index,
        label_horizon_sessions=config.label_horizon_sessions,
        feature_values=feature_values,
        forward_return=forward_return,
        target=int(forward_return > config.positive_return_threshold),
    )


def _hash_dataset(
    *,
    ticker: str,
    config: TechnicalDatasetConfig,
    feature_names: Sequence[str],
    rows: Sequence[TechnicalFeatureRow],
) -> str:
    payload = {
        "ticker": ticker.upper(),
        "config": _feature_config_payload(config),
        "feature_names": tuple(feature_names),
        "rows": [
            {
                "feature_end": _timestamp_to_string(row.feature_end),
                "label_end": _timestamp_to_string(row.label_end),
                "feature_values": [round(value, 12) for value in row.feature_values],
                "forward_return": round(row.forward_return, 12),
                "target": row.target,
            }
            for row in rows
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_config_payload(config: TechnicalDatasetConfig) -> dict[str, object]:
    payload = config.model_dump(mode="json")
    payload.pop("as_of", None)
    payload.pop("max_latest_bar_age_days", None)
    return payload


def _timestamp_key(value: date | datetime) -> int:
    if isinstance(value, datetime):
        return int(value.timestamp())
    return value.toordinal()


def _timestamp_to_string(value: date | datetime) -> str:
    return value.isoformat()


def _calendar_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return _utc_datetime(value).date()
    return value


def _timestamp_after(left: date | datetime, right: date | datetime) -> bool:
    if isinstance(left, datetime) and isinstance(right, datetime):
        return _utc_datetime(left) > _utc_datetime(right)
    return _calendar_date(left) > _calendar_date(right)


def _utc_datetime(value: datetime) -> datetime:
    _require_aware_datetime(value, field_name="timestamp")
    return value.astimezone(UTC)


def _require_aware_datetime(value: datetime, *, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} datetime must include a timezone")


def _as_decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int | str):
        return Decimal(str(value))
    raise DatasetValidationError("missing OHLCV field detected")


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise DatasetValidationError("missing OHLCV field detected")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value)
    raise DatasetValidationError("missing OHLCV field detected")


def _as_float(value: Decimal) -> float:
    return float(value)


def _safe_pct(numerator: Decimal, denominator: Decimal) -> float:
    if denominator == Decimal("0"):
        return 0.0
    return _as_float(numerator / denominator)


def _candle_body_pct(bar: PriceBar) -> float:
    return _safe_pct(abs(bar.close - bar.open), bar.close)


def _upper_shadow_pct(bar: PriceBar) -> float:
    return _safe_pct(bar.high - max(bar.open, bar.close), bar.close)


def _lower_shadow_pct(bar: PriceBar) -> float:
    return _safe_pct(min(bar.open, bar.close) - bar.low, bar.close)


def _intraday_return_pct(bar: PriceBar) -> float:
    return _safe_pct(bar.close - bar.open, bar.open)


def _overnight_gap_pct(previous: PriceBar, current: PriceBar) -> float:
    return _safe_pct(current.open - previous.close, previous.close)


def _true_range_pct(previous: PriceBar, current: PriceBar) -> float:
    true_range = max(
        current.high - current.low,
        abs(current.high - previous.close),
        abs(current.low - previous.close),
    )
    return _safe_pct(true_range, previous.close)


def _close_to_close_return(previous: PriceBar, current: PriceBar) -> float:
    return _safe_pct(current.close - previous.close, previous.close)


def _forward_return(current: PriceBar, label: PriceBar) -> float:
    return _safe_pct(label.close - current.close, current.close)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _population_stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return sqrt(variance)


def _relative_volume(current_volume: int, prior_volumes: Sequence[int]) -> float:
    positive_volumes = [volume for volume in prior_volumes if volume > 0]
    if not positive_volumes:
        return 1.0
    return current_volume / _mean([float(volume) for volume in positive_volumes])


def _volume_change_pct(previous_volume: int, current_volume: int) -> float:
    if previous_volume <= 0:
        return 0.0
    return (current_volume - previous_volume) / previous_volume


def _close_position_pct(bar: PriceBar) -> float:
    price_range = bar.high - bar.low
    if price_range == Decimal("0"):
        return 0.5
    return _as_float((bar.close - bar.low) / price_range)


__all__ = [
    "FEATURE_NAMES",
    "DatasetSplit",
    "DatasetValidationError",
    "TechnicalDataset",
    "TechnicalDatasetConfig",
    "TechnicalFeatureRow",
    "build_technical_dataset",
    "temporal_train_validation_split",
]
