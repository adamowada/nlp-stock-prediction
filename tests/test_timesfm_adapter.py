from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import pytest

from nlp_stock_prediction.contracts import PriceBar
from nlp_stock_prediction.ml import timesfm
from nlp_stock_prediction.ml.timesfm.adapter import (
    TimesFmForecastConfig,
    _InferenceStack,
    forecast_timesfm_dataset,
    write_forecast_artifact,
)
from nlp_stock_prediction.ml.timesfm.contracts import TimesFmForecastArtifact
from nlp_stock_prediction.ml.timesfm.dataset import (
    TimesFmDataset,
    TimesFmDatasetConfig,
    build_timesfm_dataset,
)

RUN_DATE = date(2026, 5, 11)


def test_timesfm_package_marks_training_surfaces_as_legacy() -> None:
    assert "train" in timesfm.LEGACY_TIMESFM_TRAINING_MODULES
    assert "focused_hpo" in timesfm.LEGACY_TIMESFM_TRAINING_MODULES
    assert "legacy" in timesfm.LEGACY_TIMESFM_TRAINING_NOTE.lower()


def _bar(
    index: int,
    *,
    close: Decimal,
    previous_close: Decimal | None = None,
    volume: int = 1_000_000,
) -> PriceBar:
    prior = previous_close if previous_close is not None else close - Decimal("0.25")
    open_price = prior * Decimal("1.002")
    high = max(open_price, close) * Decimal("1.01")
    low = min(open_price, close) * Decimal("0.99")
    return PriceBar(
        ticker="tsla",
        timestamp=RUN_DATE - timedelta(days=40 - index),
        open=open_price.quantize(Decimal("0.0001")),
        high=high.quantize(Decimal("0.0001")),
        low=low.quantize(Decimal("0.0001")),
        close=close.quantize(Decimal("0.0001")),
        volume=volume,
        adjusted_close=None,
    )


def _bars(count: int = 40) -> tuple[PriceBar, ...]:
    bars: list[PriceBar] = []
    previous_close: Decimal | None = None
    for index in range(count):
        cycle = Decimal((index % 7) - 3) * Decimal("0.27")
        drift = Decimal(index) * Decimal("0.14")
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


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeTensor(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    float32 = "float32"
    cuda = _FakeCuda()

    @staticmethod
    def tensor(values: list[float], *, dtype: object, device: str) -> _FakeTensor:
        assert dtype == _FakeTorch.float32
        assert device == "cpu"
        return _FakeTensor(values)

    @staticmethod
    def no_grad() -> _FakeNoGrad:
        return _FakeNoGrad()


class _FakeOutput:
    mean_predictions: ClassVar[list[list[float]]] = [[111.0, 112.0]]
    full_predictions: ClassVar[list[list[list[float]]]] = [
        [
            [109.0, 111.0, 113.0],
            [110.0, 112.0, 114.0],
        ]
    ]


class _ShortFullPredictionOutput(_FakeOutput):
    mean_predictions: ClassVar[list[list[float]]] = [[111.0, 112.0]]
    full_predictions: ClassVar[list[list[list[float]]]] = [[[109.0, 111.0, 113.0]]]


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = type("_Config", (), {"_commit_hash": "fake-revision"})()

    def to(self, _target: object) -> _FakeModel:
        return self

    def eval(self) -> None:
        return None

    def __call__(self, **kwargs: Any) -> _FakeOutput:
        self.calls.append(kwargs)
        return _FakeOutput()


class _FailingModel(_FakeModel):
    def __call__(self, **kwargs: Any) -> _FakeOutput:
        raise RuntimeError("model failed")


class _ShortFullPredictionModel(_FakeModel):
    def __call__(self, **kwargs: Any) -> _FakeOutput:
        self.calls.append(kwargs)
        return _ShortFullPredictionOutput()


def _dataset() -> TimesFmDataset:
    return build_timesfm_dataset(
        "TSLA",
        _bars(),
        config=TimesFmDatasetConfig(
            context_length=4,
            horizon_length=2,
            train_fraction=0.5,
            validation_fraction=0.25,
        ),
    )


@pytest.mark.unit
def test_timesfm_adapter_builds_usable_artifact_with_fake_model_without_cuda() -> None:
    dataset = _dataset()
    model = _FakeModel()
    artifact = forecast_timesfm_dataset(
        dataset,
        config=TimesFmForecastConfig(
            requested_device="cpu",
            forecast_timestamp=datetime(2026, 5, 12, tzinfo=UTC),
        ),
        stack=_InferenceStack(torch=_FakeTorch(), model_cls=object),
        model=model,
    )

    assert artifact.status == "usable"
    assert artifact.model_id == "google/timesfm-2.5-200m-transformers"
    assert artifact.model_revision == "fake-revision"
    assert artifact.dataset_hash == dataset.dataset_hash
    assert len(artifact.input_hash) == 64
    assert artifact.forecast_horizon_sessions == 2
    assert artifact.forecast_timestamp == datetime(2026, 5, 12, tzinfo=UTC)
    assert artifact.point_forecast == (111.0, 112.0)
    assert [forecast.quantile_index for forecast in artifact.quantile_forecasts] == [0, 1, 2]
    assert artifact.expected_return is not None
    assert artifact.interval_width is not None
    assert artifact.directional_probability_proxy == 1.0
    assert artifact.uncertainty_score is not None
    assert "future_values" not in model.calls[0]
    assert model.calls[0]["past_values"][0] == list(dataset.windows[-1].context_values)


@pytest.mark.unit
def test_timesfm_adapter_writes_forecast_artifact_json(tmp_path: Path) -> None:
    dataset = _dataset()
    artifact = forecast_timesfm_dataset(
        dataset,
        config=TimesFmForecastConfig(
            requested_device="cpu",
            forecast_timestamp=datetime(2026, 5, 12, tzinfo=UTC),
        ),
        stack=_InferenceStack(torch=_FakeTorch(), model_cls=object),
        model=_FakeModel(),
    )
    output_path = tmp_path / "forecast.json"

    write_forecast_artifact(artifact, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "ml.timesfm.forecast.v1"
    assert payload["status"] == "usable"
    assert payload["dataset_hash"] == dataset.dataset_hash
    assert payload["model_revision"] == "fake-revision"


@pytest.mark.unit
def test_timesfm_adapter_degrades_model_failures_to_unavailable_artifact() -> None:
    dataset = _dataset()

    artifact = forecast_timesfm_dataset(
        dataset,
        config=TimesFmForecastConfig(
            requested_device="cpu",
            forecast_timestamp=datetime(2026, 5, 12, tzinfo=UTC),
        ),
        stack=_InferenceStack(torch=_FakeTorch(), model_cls=object),
        model=_FailingModel(),
    )

    assert isinstance(artifact, TimesFmForecastArtifact)
    assert artifact.status == "unavailable"
    assert artifact.point_forecast == ()
    assert artifact.quantile_forecasts == ()
    assert artifact.warning_ids == ("timesfm_forecast_unavailable:RuntimeError",)
    assert artifact.metadata["error"] == "model failed"


@pytest.mark.unit
def test_timesfm_adapter_degrades_short_full_predictions_to_unavailable_artifact() -> None:
    dataset = _dataset()

    artifact = forecast_timesfm_dataset(
        dataset,
        config=TimesFmForecastConfig(
            requested_device="cpu",
            forecast_timestamp=datetime(2026, 5, 12, tzinfo=UTC),
        ),
        stack=_InferenceStack(torch=_FakeTorch(), model_cls=object),
        model=_ShortFullPredictionModel(),
    )

    assert artifact.status == "unavailable"
    assert artifact.warning_ids == ("timesfm_forecast_unavailable:TimesFmForecastError",)
    error = artifact.metadata["error"]
    assert isinstance(error, str)
    assert "full prediction horizon" in error
