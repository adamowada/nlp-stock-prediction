from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from math import isfinite
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.analysis import apply_technical_ml_signal, build_technical_ml_signal
from nlp_stock_prediction.contracts import AnalysisSignal, JsonObject, PriceBar, TechnicalAnalysis
from nlp_stock_prediction.ml.dataset import TechnicalDatasetConfig, build_technical_dataset
from nlp_stock_prediction.ml.training import (
    TrainingConfig,
    TrainingDeviceMetadata,
    detect_training_device,
    evaluate_model,
    load_model_artifact,
    train_technical_model,
    write_training_artifacts,
)

RUN_DATE = date(2026, 5, 11)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    metadata = result.model.training_metadata
    assert metadata["seed"] == 7
    assert metadata["dataset_hash"] == dataset.dataset_hash
    assert metadata["model_hash"] == result.model.model_hash
    split_metadata = cast(JsonObject, metadata["split"])
    validation_metrics = cast(JsonObject, metadata["validation_metrics"])
    runtime_metadata = cast(JsonObject, metadata["runtime"])
    assert split_metadata["purged_row_count"] == result.split.purged_row_count
    assert validation_metrics["samples"] == result.validation_metrics.samples
    assert runtime_metadata["selected_device"] == "cpu"
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
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))

    assert paths.model_path.exists()
    assert paths.metrics_path.exists()
    assert paths.metadata_path.exists()
    assert len(paths.model_sha256) == 64
    assert len(paths.metrics_sha256) == 64
    assert len(paths.metadata_sha256) == 64
    assert paths.model_sha256 == _file_sha256(paths.model_path)
    assert paths.metrics_sha256 == _file_sha256(paths.metrics_path)
    assert paths.metadata_sha256 == _file_sha256(paths.metadata_path)
    assert reloaded.model_hash == result.model.model_hash
    assert metadata["schema_version"] == "ml.training_metadata.v1"
    assert metadata["model_artifact_sha256"] == paths.model_sha256
    assert metadata["metrics_artifact_sha256"] == paths.metrics_sha256
    assert metadata["device"]["selected_device"] == result.device.selected_device
    assert metadata["split"]["purged_row_count"] == result.split.purged_row_count
    assert metadata["metrics"]["validation"]["samples"] == result.validation_metrics.samples
    assert metrics["schema_version"] == "ml.training_metrics.v1"
    assert metrics["model_hash"] == result.model.model_hash
    assert evaluation.metrics.samples == 8
    assert len(evaluation.predictions) == 8
    assert all(0.0 <= prediction.probability <= 1.0 for prediction in evaluation.predictions)
    with pytest.raises(TypeError):
        result.model.training_metadata["mutable"] = True


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


@pytest.mark.unit
def test_cuda_detection_records_gpu_metadata_without_claiming_gpu_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch)

    auto_device = detect_training_device("auto")
    requested_cuda = detect_training_device("cuda")

    for device in (auto_device, requested_cuda):
        assert isinstance(device, TrainingDeviceMetadata)
        assert device.selected_device == "cpu"
        assert device.cuda_available is True
        assert device.gpu_name == "NVIDIA GeForce RTX 3090"
        assert device.cuda_version == "12.1"
        assert device.backend == "pure-python-logistic-regression"
        assert any("executes on CPU" in note for note in device.notes)


@pytest.mark.unit
def test_training_command_writes_reproducible_metadata_payload(tmp_path: Path) -> None:
    csv_path = tmp_path / "amd.csv"
    output_dir = tmp_path / "artifacts"
    _write_training_csv(csv_path, _training_bars())

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nlp_stock_prediction.ml.train",
            "--csv",
            str(csv_path),
            "--ticker",
            "AMD",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--epochs",
            "12",
            "--seed",
            "23",
            "--as-of",
            RUN_DATE.isoformat(),
            "--max-latest-bar-age-days",
            "5",
        ],
        cwd=PROJECT_ROOT,
        env=_module_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    metadata = json.loads(Path(payload["metadata_path"]).read_text(encoding="utf-8"))

    assert payload["selected_device"] == "cpu"
    assert payload["backend"] == "pure-python-logistic-regression"
    assert payload["cuda_available"] is False
    assert payload["gpu_name"] is None
    assert payload["dataset_hash"] == metadata["dataset_hash"]
    assert payload["model_hash"] == metadata["model_hash"]
    assert payload["artifact_sha256"]["model"] == metadata["model_artifact_sha256"]
    assert payload["artifact_sha256"]["metrics"] == metadata["metrics_artifact_sha256"]
    assert len(payload["artifact_sha256"]["metadata"]) == 64
    assert payload["artifact_sha256"]["model"] == _file_sha256(Path(payload["model_path"]))
    assert payload["artifact_sha256"]["metrics"] == _file_sha256(Path(payload["metrics_path"]))
    assert payload["artifact_sha256"]["metadata"] == _file_sha256(Path(payload["metadata_path"]))
    assert metadata["config"]["seed"] == 23
    assert metadata["config"]["requested_device"] == "cpu"
    assert metadata["runtime"]["selected_device"] == "cpu"
    assert metadata["split"]["validation_rows"] > 0
    assert metadata["metrics"]["validation"]["samples"] == metadata["split"]["validation_rows"]
    assert metadata["usage_limitations"] == payload["usage_limitations"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scenario", "expected_error"),
    (
        ("stale", "stale data"),
        ("future", "lookahead leakage"),
    ),
)
def test_training_command_surfaces_freshness_gate_failures(
    tmp_path: Path,
    scenario: str,
    expected_error: str,
) -> None:
    csv_path = tmp_path / "amd.csv"
    output_dir = tmp_path / "artifacts"
    if scenario == "future":
        bars = _training_bars_with_latest_timestamp(datetime_text="2026-05-11 21:00:00+00:00")
        as_of = "2026-05-11 13:00:00+00:00"
    else:
        bars = _training_bars()
        as_of = (RUN_DATE + timedelta(days=10)).isoformat()
    _write_training_csv(csv_path, bars)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nlp_stock_prediction.ml.train",
            "--csv",
            str(csv_path),
            "--ticker",
            "AMD",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--as-of",
            as_of,
            "--max-latest-bar-age-days",
            "5",
        ],
        cwd=PROJECT_ROOT,
        env=_module_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr


def _write_training_csv(path: Path, bars: tuple[PriceBar, ...]) -> None:
    rows = ["timestamp,open,high,low,close,volume"]
    rows.extend(
        f"{bar.timestamp},{bar.open},{bar.high},{bar.low},{bar.close},{bar.volume}" for bar in bars
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _training_bars_with_latest_timestamp(*, datetime_text: str) -> tuple[PriceBar, ...]:
    bars = list(_training_bars())
    latest = bars[-1]
    bars[-1] = PriceBar(
        ticker=latest.ticker,
        timestamp=datetime.fromisoformat(datetime_text),
        open=latest.open,
        high=latest.high,
        low=latest.low,
        close=latest.close,
        volume=latest.volume,
    )
    return tuple(bars)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "NVIDIA GeForce RTX 3090"

    class _FakeVersion:
        cuda = "12.1"

    class _FakeTorch:
        cuda = _FakeCuda()
        version = _FakeVersion()

    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "torch":
            return _FakeTorch()
        return original_import(name, package)

    monkeypatch.setattr("nlp_stock_prediction.ml.training.importlib.import_module", fake_import)


def _module_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in (
        "COMSPEC",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "WINDIR",
    ):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path
    env["NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS"] = "0"
    return env
