from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
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


def _replace_bar(
    bar: PriceBar,
    *,
    timestamp: date | datetime | None = None,
    close: Decimal | None = None,
    volume: int | None = None,
    adjusted_close: Decimal | None = None,
) -> PriceBar:
    new_close = close if close is not None else bar.close
    high = max(bar.open, new_close, bar.high)
    low = min(bar.open, new_close, bar.low)
    return PriceBar(
        ticker=bar.ticker,
        timestamp=timestamp if timestamp is not None else bar.timestamp,
        open=bar.open,
        high=high,
        low=low,
        close=new_close,
        volume=volume if volume is not None else bar.volume,
        adjusted_close=adjusted_close,
    )


@pytest.mark.unit
def test_build_technical_dataset_derives_features_and_forward_labels() -> None:
    config = TechnicalDatasetConfig(feature_window=4, label_horizon_sessions=2)
    bars = _bars()

    dataset = build_technical_dataset("TSLA", bars, config=config)

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
    assert first.feature_start == bars[0].timestamp
    assert first.feature_end_index == 4
    assert first.feature_end < first.label_end
    assert first.forward_return == pytest.approx(expected_forward_return)
    assert first.target == int(expected_forward_return > 0.0)
    assert all(isinstance(value, float) for value in first.feature_values)
    assert dataset.metadata["feature_lookback_includes_prior_bar"] is True


@pytest.mark.unit
def test_feature_rows_do_not_change_when_unseen_future_bars_change() -> None:
    config = TechnicalDatasetConfig(feature_window=4, label_horizon_sessions=2)
    base_bars = list(_bars(20))
    altered_bars = list(base_bars)
    sentinel_timestamp = altered_bars[14].timestamp
    altered_bars[14] = _replace_bar(
        altered_bars[14],
        close=(altered_bars[14].close * Decimal("1.10")).quantize(Decimal("0.0001")),
        volume=altered_bars[14].volume * 9,
    )

    base = build_technical_dataset("TSLA", base_bars, config=config)
    altered = build_technical_dataset("TSLA", altered_bars, config=config)

    unchanged_pairs = [
        (base_row, altered_row)
        for base_row, altered_row in zip(base.rows, altered.rows, strict=True)
        if base_row.label_end < sentinel_timestamp
    ]
    changed_pairs = [
        (base_row, altered_row)
        for base_row, altered_row in zip(base.rows, altered.rows, strict=True)
        if base_row.label_end == sentinel_timestamp
    ]

    assert unchanged_pairs
    assert changed_pairs
    for base_row, altered_row in unchanged_pairs:
        assert altered_row.feature_values == base_row.feature_values
        assert altered_row.forward_return == base_row.forward_return
    assert any(
        altered_row.forward_return != base_row.forward_return
        for base_row, altered_row in changed_pairs
    )


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
def test_dataset_rejects_duplicate_mixed_date_and_datetime_timestamps() -> None:
    bars = list(_bars())
    duplicate_day = bars[3].timestamp
    assert isinstance(duplicate_day, date)
    bars[4] = _replace_bar(
        bars[4],
        timestamp=datetime(
            duplicate_day.year,
            duplicate_day.month,
            duplicate_day.day,
            tzinfo=UTC,
        ),
    )

    with pytest.raises(DatasetValidationError, match="duplicate"):
        build_technical_dataset("TSLA", bars)


@pytest.mark.unit
def test_dataset_rejects_missing_or_impossible_ohlcv() -> None:
    missing_close = _bar(20, close=Decimal("100"))
    object.__setattr__(missing_close, "close", None)

    with pytest.raises(DatasetValidationError, match="missing OHLCV"):
        build_technical_dataset("TSLA", (*_bars(8), missing_close))

    impossible = _replace_bar(
        _bar(21, close=Decimal("100")),
        timestamp=RUN_DATE + timedelta(days=1),
    )
    object.__setattr__(impossible, "high", Decimal("98"))
    object.__setattr__(impossible, "low", Decimal("99"))

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
def test_dataset_rejects_partial_adjusted_close_history() -> None:
    bars = [_replace_bar(bar, adjusted_close=bar.close * Decimal("0.99")) for bar in _bars(10)]
    bars[6] = _replace_bar(bars[6], adjusted_close=None)

    with pytest.raises(DatasetValidationError, match="partial adjusted_close"):
        build_technical_dataset("TSLA", bars)


@pytest.mark.unit
def test_dataset_rejects_stale_or_future_bars_when_as_of_gate_is_configured() -> None:
    stale_config = TechnicalDatasetConfig(
        feature_window=4,
        label_horizon_sessions=2,
        as_of=RUN_DATE + timedelta(days=10),
        max_latest_bar_age_days=5,
    )

    with pytest.raises(DatasetValidationError, match="stale data"):
        build_technical_dataset("TSLA", _bars(17), config=stale_config)

    future_config = TechnicalDatasetConfig(
        feature_window=4,
        label_horizon_sessions=2,
        as_of=RUN_DATE - timedelta(days=1),
        max_latest_bar_age_days=5,
    )

    with pytest.raises(DatasetValidationError, match="lookahead leakage"):
        build_technical_dataset("TSLA", _bars(17), config=future_config)


@pytest.mark.unit
def test_dataset_rejects_intraday_future_bars_when_as_of_is_datetime() -> None:
    start = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    bars = tuple(
        _replace_bar(bar, timestamp=start + timedelta(hours=index))
        for index, bar in enumerate(_bars(17))
    )
    config = TechnicalDatasetConfig(
        feature_window=4,
        label_horizon_sessions=2,
        as_of=start + timedelta(hours=15),
        max_latest_bar_age_days=1,
    )

    with pytest.raises(DatasetValidationError, match="lookahead leakage"):
        build_technical_dataset("TSLA", bars, config=config)


@pytest.mark.unit
def test_freshness_gate_requires_as_of_and_aware_datetime() -> None:
    with pytest.raises(ValueError, match="requires as_of"):
        TechnicalDatasetConfig(max_latest_bar_age_days=5)

    with pytest.raises(ValueError, match="timezone"):
        TechnicalDatasetConfig(as_of=datetime(2026, 5, 11, 12, 0))


@pytest.mark.unit
def test_freshness_gate_metadata_does_not_change_dataset_hash() -> None:
    base_config = TechnicalDatasetConfig(feature_window=4, label_horizon_sessions=2)
    gated_config = TechnicalDatasetConfig(
        feature_window=4,
        label_horizon_sessions=2,
        as_of=RUN_DATE,
        max_latest_bar_age_days=5,
    )

    base = build_technical_dataset("TSLA", _bars(17), config=base_config)
    gated = build_technical_dataset("TSLA", _bars(17), config=gated_config)

    assert gated.dataset_hash == base.dataset_hash


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
