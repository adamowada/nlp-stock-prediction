"""Shared OHLCV validation helpers for local technical datasets."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal

from nlp_stock_prediction.contracts import PriceBar


class DatasetValidationError(ValueError):
    """Raised when OHLCV bars are unsafe for ML dataset generation."""


def validate_ohlcv_bars(
    ticker: str,
    bars: Sequence[PriceBar],
    *,
    required_count: int,
    as_of: date | datetime | None,
    max_latest_bar_age_days: int | None,
    split_like_move_threshold_pct: float,
    adjustment_ratio_drift_threshold_pct: float,
    require_adjusted_close: bool = False,
) -> None:
    if len(bars) < required_count:
        raise DatasetValidationError(
            f"insufficient history: need at least {required_count} bars, got {len(bars)}"
        )
    validate_freshness(
        bars,
        as_of=as_of,
        max_latest_bar_age_days=max_latest_bar_age_days,
    )

    timestamps: set[int] = set()
    adjusted_close_presence: list[bool] = []
    previous_close: Decimal | None = None
    previous_adjustment_ratio: Decimal | None = None
    for bar in bars:
        if bar.ticker != ticker.upper():
            raise DatasetValidationError("ticker mismatch between requested ticker and bars")
        timestamp_key = timestamp_key_for(bar.timestamp)
        if timestamp_key in timestamps:
            raise DatasetValidationError("duplicate bar timestamp detected")
        timestamps.add(timestamp_key)
        validate_ohlcv_fields(bar)

        adjusted_close_presence.append(bar.adjusted_close is not None)
        if previous_close is not None:
            close_return = abs((as_decimal(bar.close) - previous_close) / previous_close)
            if close_return >= Decimal(str(split_like_move_threshold_pct)):
                raise DatasetValidationError(
                    "split leakage: split-like close-to-close discontinuity detected"
                )
        previous_close = as_decimal(bar.close)

        if bar.adjusted_close is not None:
            adjustment_ratio = as_decimal(bar.adjusted_close) / as_decimal(bar.close)
            if previous_adjustment_ratio is not None:
                ratio_drift = abs(adjustment_ratio - previous_adjustment_ratio)
                drift_pct = ratio_drift / max(abs(previous_adjustment_ratio), Decimal("0.0001"))
                if drift_pct >= Decimal(str(adjustment_ratio_drift_threshold_pct)):
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
    if require_adjusted_close and not all(adjusted_close_presence):
        raise DatasetValidationError("adjusted_close target requested but not all bars include it")


def validate_freshness(
    bars: Sequence[PriceBar],
    *,
    as_of: date | datetime | None,
    max_latest_bar_age_days: int | None,
) -> None:
    if as_of is None:
        return
    latest_timestamp = bars[-1].timestamp
    if timestamp_after(latest_timestamp, as_of):
        raise DatasetValidationError("lookahead leakage: latest bar is after dataset as_of")
    if max_latest_bar_age_days is None:
        return
    age_days = (calendar_date(as_of) - calendar_date(latest_timestamp)).days
    if age_days > max_latest_bar_age_days:
        raise DatasetValidationError(
            "stale data: latest bar is older than the configured freshness gate"
        )


def validate_ohlcv_fields(bar: PriceBar) -> None:
    raw_values = {
        "open": getattr(bar, "open", None),
        "high": getattr(bar, "high", None),
        "low": getattr(bar, "low", None),
        "close": getattr(bar, "close", None),
        "volume": getattr(bar, "volume", None),
    }
    if any(value is None for value in raw_values.values()):
        raise DatasetValidationError("missing OHLCV field detected")
    open_price = as_decimal(raw_values["open"])
    high = as_decimal(raw_values["high"])
    low = as_decimal(raw_values["low"])
    close = as_decimal(raw_values["close"])
    volume = as_int(raw_values["volume"])
    if min(open_price, high, low, close) <= Decimal("0"):
        raise DatasetValidationError("impossible OHLCV price: prices must be positive")
    if high < low or high < max(open_price, close) or low > min(open_price, close):
        raise DatasetValidationError("impossible OHLCV price ordering")
    if volume < 0:
        raise DatasetValidationError("impossible OHLCV volume: volume must be non-negative")
    if bar.adjusted_close is not None and as_decimal(bar.adjusted_close) <= Decimal("0"):
        raise DatasetValidationError("impossible adjusted close: must be positive")


def timestamp_key_for(value: date | datetime) -> int:
    if isinstance(value, datetime):
        return int(value.timestamp())
    return value.toordinal()


def calendar_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return utc_datetime(value).date()
    return value


def timestamp_after(left: date | datetime, right: date | datetime) -> bool:
    if isinstance(left, datetime) and isinstance(right, datetime):
        return utc_datetime(left) > utc_datetime(right)
    return calendar_date(left) > calendar_date(right)


def utc_datetime(value: datetime) -> datetime:
    require_aware_datetime(value, field_name="timestamp")
    return value.astimezone(UTC)


def require_aware_datetime(value: datetime, *, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} datetime must include a timezone")


def as_decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise DatasetValidationError("missing OHLCV field detected")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int | str):
        return Decimal(str(value))
    raise DatasetValidationError("missing OHLCV field detected")


def as_int(value: object) -> int:
    if isinstance(value, bool):
        raise DatasetValidationError("missing OHLCV field detected")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value)
    raise DatasetValidationError("missing OHLCV field detected")


__all__ = [
    "DatasetValidationError",
    "as_decimal",
    "as_int",
    "calendar_date",
    "require_aware_datetime",
    "timestamp_after",
    "timestamp_key_for",
    "utc_datetime",
    "validate_freshness",
    "validate_ohlcv_bars",
    "validate_ohlcv_fields",
]
