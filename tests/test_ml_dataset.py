from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from nlp_stock_prediction.contracts import PriceBar
from nlp_stock_prediction.ml.dataset import (
    DatasetValidationError,
    TechnicalDatasetConfig,
    build_technical_dataset,
    temporal_train_validation_split,
)

RUN_DATE = date(2026, 5, 11)


def _bar(
    index: int,
    *,
    close: Decimal,
    previous_close: Decimal | None = None,
    volume: int = 1_000_000,
    adjusted_close: Decimal | None = None,
) -> PriceBar:
    prior = previous_close if previous_close is not None else close - Decimal("0.25")
    open_price = prior * Decimal("1.002")
    high = max(open_price, close) * Decimal("1.01")
    low = min(open_price, close) * Decimal("0.99")
    return PriceBar(
        ticker="tsla",
        timestamp=RUN_DATE - timedelta(days=16 - index),
        open=open_price.quantize(Decimal("0.0001")),
        high=high.quantize(Decimal("0.0001")),
        low=low.quantize(Decimal("0.0001")),
        close=close.quantize(Decimal("0.0001")),
        volume=volume,
        adjusted_close=adjusted_close,
    )


def _bars(count: int = 17) -> tuple[PriceBar, ...]:
    bars: list[PriceBar] = []
    previous_close: Decimal | None = None
    for index in range(count):
        cycle = Decimal((index % 6) - 2) * Decimal("0.45")
        drift = Decimal(index) * Decimal("0.18")
        close = Decimal("100") + drift + cycle
        bars.append(
            _bar(
                index,
                close=close,
                previous_close=previous_close,
                volume=900_000 + (index % 5) * 40_000,
            )
        )
        previous_close = close
    return tuple(bars)


@pytest.mark.unit
def test_build_technical_dataset_derives_features_and_forward_labels() -> None:
    config = TechnicalDatasetConfig(feature_window=4, label_horizon_sessions=2)

    dataset = build_technical_dataset("TSLA", _bars(), config=config)

    first = dataset.rows[0]
    current_close = Decimal("100") + Decimal(4) * Decimal("0.18") + Decimal(2) * Decimal("0.45")
    future_close = Decimal("100") + Decimal(6) * Decimal("0.18") - Decimal(2) * Decimal("0.45")
    expected_forward_return = float((future_close - current_close) / current_close)

    assert dataset.ticker == "TSLA"
    assert dataset.label_name == "forward_return_positive"
    assert len(dataset.dataset_hash) == 64
    assert {
        "candle_body_pct",
        "rolling_volatility_pct",
        "relative_volume",
        "overnight_gap_pct",
    } <= set(dataset.feature_names)
    assert len(first.feature_values) == len(dataset.feature_names)
    assert first.feature_end_index == 4
    assert first.feature_end < first.label_end
    assert first.forward_return == pytest.approx(expected_forward_return)
    assert first.target == int(expected_forward_return > 0.0)
    assert all(isinstance(value, float) for value in first.feature_values)


@pytest.mark.unit
def test_dataset_rejects_duplicate_bar_timestamps() -> None:
    bars = list(_bars())
    bars[4] = PriceBar(
        ticker="TSLA",
        timestamp=bars[3].timestamp,
        open=bars[4].open,
        high=bars[4].high,
        low=bars[4].low,
        close=bars[4].close,
        volume=bars[4].volume,
    )

    with pytest.raises(DatasetValidationError, match="duplicate"):
        build_technical_dataset("TSLA", bars)


@pytest.mark.unit
def test_dataset_rejects_missing_or_impossible_ohlcv() -> None:
    missing_close = _bar(20, close=Decimal("100"))
    object.__setattr__(missing_close, "close", None)

    with pytest.raises(DatasetValidationError, match="missing OHLCV"):
        build_technical_dataset("TSLA", (*_bars(8), missing_close))

    impossible = PriceBar(
        ticker="TSLA",
        timestamp=RUN_DATE + timedelta(days=1),
        open=Decimal("100"),
        high=Decimal("98"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=1_000,
    )

    with pytest.raises(DatasetValidationError, match="impossible"):
        build_technical_dataset("TSLA", (*_bars(8), impossible))


@pytest.mark.unit
def test_dataset_rejects_split_like_leakage() -> None:
    bars = list(_bars(10))
    split_day = bars[7]
    bars[7] = PriceBar(
        ticker="TSLA",
        timestamp=split_day.timestamp,
        open=Decimal("25.20"),
        high=Decimal("25.80"),
        low=Decimal("24.90"),
        close=Decimal("25.10"),
        volume=4_000_000,
    )

    with pytest.raises(DatasetValidationError, match="split leakage"):
        build_technical_dataset("TSLA", bars)


@pytest.mark.unit
def test_temporal_split_purges_label_overlap_between_train_and_validation() -> None:
    dataset = build_technical_dataset(
        "TSLA",
        _bars(28),
        config=TechnicalDatasetConfig(feature_window=4, label_horizon_sessions=3),
    )

    split = temporal_train_validation_split(dataset, train_fraction=0.6)

    assert split.train_rows
    assert split.validation_rows
    assert split.purged_row_count >= dataset.label_horizon_sessions
    assert split.train_rows[-1].label_end < split.validation_rows[0].feature_end
