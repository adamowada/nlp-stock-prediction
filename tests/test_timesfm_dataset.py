from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import PriceBar
from nlp_stock_prediction.ml.dataset import DatasetValidationError
from nlp_stock_prediction.ml.timesfm.dataset import (
    TimesFmDatasetConfig,
    build_timesfm_dataset,
)
from nlp_stock_prediction.ml.train import load_price_bars_csv

RUN_DATE = date(2026, 5, 11)


def _bar(
    index: int,
    *,
    close: Decimal,
    previous_close: Decimal | None = None,
    volume: int = 1_000_000,
    adjusted_close: Decimal | None = None,
    timestamp: date | datetime | None = None,
) -> PriceBar:
    prior = previous_close if previous_close is not None else close - Decimal("0.25")
    open_price = prior * Decimal("1.002")
    high = max(open_price, close) * Decimal("1.01")
    low = min(open_price, close) * Decimal("0.99")
    return PriceBar(
        ticker="tsla",
        timestamp=timestamp if timestamp is not None else RUN_DATE - timedelta(days=40 - index),
        open=open_price.quantize(Decimal("0.0001")),
        high=high.quantize(Decimal("0.0001")),
        low=low.quantize(Decimal("0.0001")),
        close=close.quantize(Decimal("0.0001")),
        volume=volume,
        adjusted_close=adjusted_close.quantize(Decimal("0.0001"))
        if adjusted_close is not None
        else None,
    )


def _bars(count: int = 40, *, adjusted: bool = False) -> tuple[PriceBar, ...]:
    bars: list[PriceBar] = []
    previous_close: Decimal | None = None
    for index in range(count):
        cycle = Decimal((index % 7) - 3) * Decimal("0.27")
        drift = Decimal(index) * Decimal("0.14")
        close = Decimal("100") + drift + cycle
        adjusted_close = close * Decimal("0.95") if adjusted else None
        bars.append(
            _bar(
                index,
                close=close,
                previous_close=previous_close,
                volume=900_000 + (index % 5) * 40_000,
                adjusted_close=adjusted_close,
            )
        )
        previous_close = close
    return tuple(bars)


def _config() -> TimesFmDatasetConfig:
    return TimesFmDatasetConfig(
        context_length=4,
        horizon_length=2,
        train_fraction=0.5,
        validation_fraction=0.25,
    )


@pytest.mark.unit
def test_timesfm_dataset_uses_adjusted_close_when_complete_and_hashes_are_stable() -> None:
    bars = _bars(adjusted=True)
    dataset = build_timesfm_dataset("TSLA", bars, config=_config())
    repeated = build_timesfm_dataset("TSLA", bars, config=_config())
    first_window = dataset.windows[0]

    assert dataset.target_field == "adjusted_close"
    assert dataset.dataset_hash == repeated.dataset_hash
    assert len(dataset.dataset_hash) == 64
    assert first_window.context_values == _adjusted_values(bars[:4])
    assert first_window.future_values == _adjusted_values(bars[4:6])
    assert first_window.context_start == bars[0].timestamp
    assert first_window.context_end == bars[3].timestamp
    assert first_window.horizon_start == bars[4].timestamp
    assert first_window.horizon_end == bars[5].timestamp
    assert first_window.context_end_index < first_window.horizon_start_index
    assert dataset.metadata["normalization"] == "none_timesfm_internal_instance_normalization"
    assert dataset.metadata["target_policy"] == "auto_adjusted_close_else_close"
    assert dataset.metadata["split_counts"] == {
        "train": len(dataset.train_windows),
        "validation": len(dataset.validation_windows),
        "test": len(dataset.test_windows),
    }


@pytest.mark.unit
def test_timesfm_dataset_falls_back_to_close_and_preserves_temporal_split_purge() -> None:
    bars = _bars()
    dataset = build_timesfm_dataset("TSLA", bars, config=_config())

    assert dataset.target_field == "close"
    assert dataset.windows[0].context_values == tuple(float(bar.close) for bar in bars[:4])
    assert dataset.windows[0].future_values == tuple(float(bar.close) for bar in bars[4:6])
    assert (
        dataset.validation_windows[0].context_start_index
        - dataset.train_windows[-1].context_start_index
        == dataset.horizon_length + 1
    )
    assert (
        dataset.test_windows[0].context_start_index
        - dataset.validation_windows[-1].context_start_index
        == dataset.horizon_length + 1
    )
    assert dataset.metadata["purged_window_count"] == 4


@pytest.mark.unit
def test_timesfm_dataset_context_windows_do_not_include_future_labels() -> None:
    dataset = build_timesfm_dataset("TSLA", _bars(), config=_config())

    for window in dataset.windows:
        context_indices = set(range(window.context_start_index, window.context_end_index + 1))
        future_indices = set(range(window.horizon_start_index, window.horizon_end_index + 1))
        assert context_indices.isdisjoint(future_indices)
        assert max(context_indices) + 1 == min(future_indices)
        assert len(window.context_values) == 4
        assert len(window.future_values) == 2


@pytest.mark.unit
def test_timesfm_dataset_can_be_built_from_fixture_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "tsla.csv"
    bars = _bars(adjusted=True)
    _write_bars_csv(csv_path, bars)

    loaded_bars = load_price_bars_csv(csv_path, ticker="TSLA")
    dataset = build_timesfm_dataset("TSLA", loaded_bars, config=_config())
    repeated = build_timesfm_dataset("TSLA", loaded_bars, config=_config())

    assert len(dataset.train_windows) > 0
    assert len(dataset.validation_windows) > 0
    assert len(dataset.test_windows) > 0
    assert dataset.dataset_hash == repeated.dataset_hash
    assert dataset.metadata["raw_window_count"] == 35


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scenario", "config", "expected_error"),
    (
        (
            "insufficient",
            TimesFmDatasetConfig(context_length=4, horizon_length=2),
            "insufficient history",
        ),
        (
            "stale",
            TimesFmDatasetConfig(
                context_length=4,
                horizon_length=2,
                as_of=RUN_DATE + timedelta(days=10),
                max_latest_bar_age_days=5,
            ),
            "stale data",
        ),
        (
            "future",
            TimesFmDatasetConfig(
                context_length=4,
                horizon_length=2,
                as_of=datetime(2026, 5, 11, 20, 0, tzinfo=UTC),
                max_latest_bar_age_days=5,
            ),
            "lookahead leakage",
        ),
        (
            "duplicate",
            TimesFmDatasetConfig(context_length=4, horizon_length=2),
            "duplicate bar timestamp",
        ),
        (
            "split",
            TimesFmDatasetConfig(context_length=4, horizon_length=2),
            "split leakage",
        ),
        (
            "partial_adjusted",
            TimesFmDatasetConfig(context_length=4, horizon_length=2),
            "partial adjusted_close",
        ),
    ),
)
def test_timesfm_dataset_rejects_unsafe_ohlcv_inputs(
    scenario: str,
    config: TimesFmDatasetConfig,
    expected_error: str,
) -> None:
    bars = _unsafe_bars(scenario)

    with pytest.raises(DatasetValidationError, match=expected_error):
        build_timesfm_dataset("TSLA", bars, config=config)


def _unsafe_bars(scenario: str) -> tuple[PriceBar, ...]:
    if scenario == "insufficient":
        return _bars(count=5)
    if scenario == "stale":
        return _bars()
    if scenario == "future":
        bars = _bars()
        return (*bars[:-1], _replace_timestamp(bars[-1], RUN_DATE + timedelta(days=1)))
    if scenario == "duplicate":
        bars = _bars()
        return (*bars[:-1], _replace_timestamp(bars[-1], bars[-2].timestamp))
    if scenario == "split":
        return _split_leakage_bars()
    if scenario == "partial_adjusted":
        return _partial_adjusted_bars()
    raise AssertionError(f"Unhandled scenario: {scenario}")


def _replace_timestamp(bar: PriceBar, timestamp: date | datetime) -> PriceBar:
    return PriceBar(
        ticker=bar.ticker,
        timestamp=timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        adjusted_close=bar.adjusted_close,
    )


def _split_leakage_bars() -> tuple[PriceBar, ...]:
    bars = list(_bars())
    bad = bars[10]
    bars[10] = PriceBar(
        ticker=bad.ticker,
        timestamp=bad.timestamp,
        open=bad.open,
        high=Decimal("220"),
        low=bad.low,
        close=Decimal("200"),
        volume=bad.volume,
    )
    return tuple(bars)


def _partial_adjusted_bars() -> tuple[PriceBar, ...]:
    bars = list(_bars(adjusted=True))
    bar = bars[-1]
    bars[-1] = PriceBar(
        ticker=bar.ticker,
        timestamp=bar.timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        adjusted_close=None,
    )
    return tuple(bars)


def _write_bars_csv(path: Path, bars: tuple[PriceBar, ...]) -> None:
    rows = ["timestamp,open,high,low,close,volume,adjusted_close"]
    rows.extend(
        ",".join(
            (
                str(bar.timestamp),
                str(bar.open),
                str(bar.high),
                str(bar.low),
                str(bar.close),
                str(bar.volume),
                str(bar.adjusted_close or ""),
            )
        )
        for bar in bars
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _adjusted_values(bars: tuple[PriceBar, ...]) -> tuple[float, ...]:
    values: list[float] = []
    for bar in bars:
        assert bar.adjusted_close is not None
        values.append(float(bar.adjusted_close))
    return tuple(values)
