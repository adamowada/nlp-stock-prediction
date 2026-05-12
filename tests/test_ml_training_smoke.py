from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from math import isfinite
from pathlib import Path

import pytest

from nlp_stock_prediction.analysis import apply_technical_ml_signal, build_technical_ml_signal
from nlp_stock_prediction.contracts import AnalysisSignal, PriceBar, TechnicalAnalysis
from nlp_stock_prediction.ml.dataset import TechnicalDatasetConfig, build_technical_dataset
from nlp_stock_prediction.ml.training import (
    TrainingConfig,
    detect_training_device,
    evaluate_model,
    load_model_artifact,
    train_technical_model,
    write_training_artifacts,
)

RUN_DATE = date(2026, 5, 11)


def _training_bars(count: int = 48) -> tuple[PriceBar, ...]:
    bars: list[PriceBar] = []
    previous_close = Decimal("100")
    for index in range(count):
        trend_step = Decimal("0.18") if (index // 6) % 2 == 0 else Decimal("-0.11")
        wave = Decimal((index % 6) - 2) * Decimal("0.55")
        close = Decimal("100") + Decimal(index) * trend_step + wave
        open_price = previous_close * Decimal("1.001")
        high = max(open_price, close) * Decimal("1.008")
        low = min(open_price, close) * Decimal("0.992")
        bars.append(
            PriceBar(
                ticker="AMD",
                timestamp=RUN_DATE - timedelta(days=count - index),
                open=open_price.quantize(Decimal("0.0001")),
                high=high.quantize(Decimal("0.0001")),
                low=low.quantize(Decimal("0.0001")),
                close=close.quantize(Decimal("0.0001")),
                volume=800_000 + (index % 7) * 50_000,
            )
        )
        previous_close = close
    return tuple(bars)


@pytest.mark.unit
def test_cpu_training_smoke_records_reproducible_metadata() -> None:
    dataset = build_technical_dataset(
        "AMD",
        _training_bars(),
        config=TechnicalDatasetConfig(feature_window=5, label_horizon_sessions=2),
    )

    result = train_technical_model(
        dataset,
        config=TrainingConfig(
            epochs=35,
            learning_rate=0.08,
            seed=7,
            train_fraction=0.7,
            requested_device="cpu",
        ),
    )

    assert result.device.selected_device == "cpu"
    assert result.model.dataset_hash == dataset.dataset_hash
    assert result.model.feature_names == dataset.feature_names
    assert len(result.model.model_hash) == 64
    assert result.train_metrics.samples > 0
    assert result.validation_metrics.samples > 0
    assert isfinite(result.validation_metrics.log_loss)
    assert 0.0 <= result.validation_metrics.accuracy <= 1.0
    assert "not investment advice" in result.usage_limitations.lower()


@pytest.mark.unit
def test_model_evaluation_and_artifact_round_trip(tmp_path: Path) -> None:
    dataset = build_technical_dataset(
        "AMD",
        _training_bars(),
        config=TechnicalDatasetConfig(feature_window=5, label_horizon_sessions=2),
    )
    result = train_technical_model(
        dataset,
        config=TrainingConfig(epochs=20, seed=11, requested_device="cpu"),
    )

    paths = write_training_artifacts(result, tmp_path)
    reloaded = load_model_artifact(paths.model_path)
    evaluation = evaluate_model(reloaded, dataset.rows[-8:])

    assert paths.model_path.exists()
    assert paths.metrics_path.exists()
    assert paths.metadata_path.exists()
    assert reloaded.model_hash == result.model.model_hash
    assert evaluation.metrics.samples == 8
    assert len(evaluation.predictions) == 8
    assert all(0.0 <= prediction.probability <= 1.0 for prediction in evaluation.predictions)


@pytest.mark.unit
def test_technical_ml_signal_sidecar_from_evaluation_attaches_metrics() -> None:
    dataset = build_technical_dataset(
        "AMD",
        _training_bars(),
        config=TechnicalDatasetConfig(feature_window=5, label_horizon_sessions=2),
    )
    result = train_technical_model(
        dataset,
        config=TrainingConfig(epochs=20, seed=13, requested_device="cpu"),
    )
    evaluation = evaluate_model(result.model, dataset.rows[-8:])

    signal = build_technical_ml_signal(
        model=result.model,
        evaluation=evaluation,
        as_of=RUN_DATE,
        min_validation_accuracy=0.0,
        min_confidence=0.0,
    )
    analysis = TechnicalAnalysis(
        ticker="AMD",
        summary="Baseline deterministic technical analysis is mixed.",
        signal=AnalysisSignal.MIXED,
        confidence=0.55,
    )

    integrated = apply_technical_ml_signal(analysis, signal)

    assert integrated.ml_signal == signal
    assert signal.metadata["validation_samples"] == 8
    assert any(metric.name == "ml-positive-return-probability" for metric in integrated.metrics)
    assert any(metric.name == "ml-validation-brier-score" for metric in integrated.metrics)
    assert "ML sidecar:" in integrated.summary


@pytest.mark.unit
def test_auto_device_detection_does_not_require_cuda() -> None:
    device = detect_training_device("auto")

    assert device.selected_device in {"cpu", "cuda"}
    assert isinstance(device.cuda_available, bool)
