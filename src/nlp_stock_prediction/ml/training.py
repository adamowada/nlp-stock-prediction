"""CPU-friendly ML training and evaluation for technical-analysis datasets."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import random
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from math import exp, isfinite, log, sqrt
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
)
from nlp_stock_prediction.ml.dataset import (
    DatasetSplit,
    TechnicalDataset,
    TechnicalFeatureRow,
    temporal_train_validation_split,
)

DeviceRequest = Literal["auto", "cpu", "cuda"]
SelectedDevice = Literal["cpu", "cuda"]

USAGE_LIMITATIONS = (
    "Experimental technical-analysis model output for research only; not investment advice, "
    "not a recommendation, and not a substitute for evidence, risk gates, or human review."
)


class TrainingConfig(ContractModel):
    """Hyperparameters for the transparent logistic-regression baseline."""

    epochs: int = Field(default=60, ge=1)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    l2_penalty: float = Field(default=0.001, ge=0.0)
    seed: int = 42
    train_fraction: float = Field(default=0.70, gt=0.0, lt=1.0)
    requested_device: DeviceRequest = "auto"


class TrainingDeviceMetadata(ContractModel):
    """Captured hardware/runtime metadata for a training run."""

    requested_device: DeviceRequest
    selected_device: SelectedDevice
    cuda_available: bool
    gpu_name: str | None = None
    cuda_version: str | None = None
    backend: NonEmptyStr
    notes: tuple[str, ...] = Field(default_factory=tuple)


class ModelMetrics(ContractModel):
    """Deterministic classification metrics for a row set."""

    samples: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    log_loss: float = Field(ge=0.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    positive_rate: float = Field(ge=0.0, le=1.0)
    mean_probability: float = Field(ge=0.0, le=1.0)
    baseline_accuracy: float = Field(ge=0.0, le=1.0)


class TechnicalLogisticModel(ContractModel):
    """Small, inspectable baseline model trained from technical feature rows."""

    model_kind: NonEmptyStr = "logistic_regression_baseline"
    feature_names: tuple[NonEmptyStr, ...]
    feature_means: tuple[float, ...]
    feature_stds: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    feature_window: int = Field(ge=2)
    label_horizon_sessions: int = Field(ge=1)
    positive_return_threshold: float
    dataset_hash: NonEmptyStr
    model_hash: NonEmptyStr
    trained_at: AwareDatetime
    training_metadata: JsonObject = Field(default_factory=dict)
    usage_limitations: NonEmptyStr = USAGE_LIMITATIONS

    @model_validator(mode="after")
    def validate_model_shape(self) -> TechnicalLogisticModel:
        feature_count = len(self.feature_names)
        if not (
            len(self.feature_means) == len(self.feature_stds) == len(self.weights) == feature_count
        ):
            raise ValueError("model feature names, scaler values, and weights must align")
        numeric_values = (*self.feature_means, *self.feature_stds, *self.weights, self.bias)
        if any(not isfinite(value) for value in numeric_values):
            raise ValueError("model numeric values must be finite")
        if any(value <= 0.0 for value in self.feature_stds):
            raise ValueError("feature standard deviations must be positive")
        return self


class Prediction(ContractModel):
    """One probability prediction tied back to the labeled source row."""

    ticker: NonEmptyStr
    feature_end: date | datetime
    label_end: date | datetime
    probability: float = Field(ge=0.0, le=1.0)
    predicted_target: int = Field(ge=0, le=1)
    actual_target: int = Field(ge=0, le=1)
    forward_return: float


class EvaluationResult(ContractModel):
    """Evaluation output with metrics and row-level probabilities."""

    model_hash: NonEmptyStr
    metrics: ModelMetrics
    predictions: tuple[Prediction, ...]
    usage_limitations: NonEmptyStr = USAGE_LIMITATIONS


class TrainingResult(ContractModel):
    """Full training output for audit and artifact writing."""

    model: TechnicalLogisticModel
    config: TrainingConfig
    device: TrainingDeviceMetadata
    split: DatasetSplit
    train_metrics: ModelMetrics
    validation_metrics: ModelMetrics
    dataset_hash: NonEmptyStr
    trained_at: AwareDatetime
    runtime_metadata: JsonObject = Field(default_factory=dict)
    usage_limitations: NonEmptyStr = USAGE_LIMITATIONS


class TrainingArtifactPaths(ContractModel):
    """Paths written by ``write_training_artifacts``."""

    model_path: Path
    metrics_path: Path
    metadata_path: Path
    model_sha256: NonEmptyStr
    metrics_sha256: NonEmptyStr
    metadata_sha256: NonEmptyStr


def detect_training_device(requested_device: DeviceRequest = "auto") -> TrainingDeviceMetadata:
    """Detect optional CUDA support without making torch a required dependency."""

    notes: list[str] = []
    cuda_available = False
    gpu_name: str | None = None
    cuda_version: str | None = None
    try:
        torch_module = importlib.import_module("torch")
    except ImportError:
        notes.append("torch is not installed; using the pure-Python CPU baseline")
    else:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            cuda_available = True
            gpu_name = str(cuda.get_device_name(0))
            version = getattr(torch_module, "version", None)
            version_cuda = getattr(version, "cuda", None)
            cuda_version = str(version_cuda) if version_cuda is not None else None
        else:
            notes.append("torch is installed but CUDA is not available")

    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    selected_device: SelectedDevice = "cpu"
    backend = "pure-python-logistic-regression"
    if cuda_available and requested_device == "cuda":
        notes.append("CUDA was requested and detected; current baseline still executes on CPU")
    elif cuda_available:
        notes.append("CUDA detected and recorded; current baseline still executes on CPU")
    return TrainingDeviceMetadata(
        requested_device=requested_device,
        selected_device=selected_device,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        backend=backend,
        notes=tuple(notes),
    )


def train_technical_model(
    dataset: TechnicalDataset,
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train a compact logistic baseline and return metadata-rich outputs."""

    settings = config or TrainingConfig()
    device = detect_training_device(settings.requested_device)
    split = temporal_train_validation_split(dataset, train_fraction=settings.train_fraction)
    feature_means, feature_stds = _fit_scaler(split.train_rows)
    weights, bias = _fit_logistic_regression(
        split.train_rows,
        feature_means=feature_means,
        feature_stds=feature_stds,
        config=settings,
    )
    trained_at = datetime.now(UTC)
    model_hash = _hash_model(
        dataset=dataset,
        config=settings,
        weights=weights,
        bias=bias,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    feature_window_value = dataset.metadata.get("feature_window")
    feature_window = (
        feature_window_value
        if isinstance(feature_window_value, int)
        else _infer_feature_window(split.train_rows)
    )
    model = TechnicalLogisticModel(
        feature_names=dataset.feature_names,
        feature_means=feature_means,
        feature_stds=feature_stds,
        weights=weights,
        bias=bias,
        feature_window=feature_window,
        label_horizon_sessions=dataset.label_horizon_sessions,
        positive_return_threshold=dataset.positive_return_threshold,
        dataset_hash=dataset.dataset_hash,
        model_hash=model_hash,
        trained_at=trained_at,
        training_metadata={
            "seed": settings.seed,
            "epochs": settings.epochs,
            "learning_rate": settings.learning_rate,
            "l2_penalty": settings.l2_penalty,
            "train_fraction": settings.train_fraction,
            "requested_device": settings.requested_device,
            "selected_device": device.selected_device,
            "cuda_available": device.cuda_available,
            "gpu_name": device.gpu_name,
            "cuda_version": device.cuda_version,
            "dataset_hash": dataset.dataset_hash,
            "not_advice": True,
        },
    )
    train_evaluation = evaluate_model(model, split.train_rows)
    validation_evaluation = evaluate_model(model, split.validation_rows)
    runtime_metadata = _runtime_metadata(device)
    model = TechnicalLogisticModel.model_validate(
        {
            **model.model_dump(mode="python"),
            "training_metadata": {
                **dict(model.training_metadata),
                "model_hash": model.model_hash,
                "feature_count": len(model.feature_names),
                "split": _split_metadata(split),
                "train_metrics": train_evaluation.metrics.model_dump(mode="json"),
                "validation_metrics": validation_evaluation.metrics.model_dump(mode="json"),
                "runtime": runtime_metadata,
            },
        }
    )
    return TrainingResult(
        model=model,
        config=settings,
        device=device,
        split=split,
        train_metrics=train_evaluation.metrics,
        validation_metrics=validation_evaluation.metrics,
        dataset_hash=dataset.dataset_hash,
        trained_at=trained_at,
        runtime_metadata=runtime_metadata,
    )


def evaluate_model(
    model: TechnicalLogisticModel,
    rows: Sequence[TechnicalFeatureRow],
) -> EvaluationResult:
    """Evaluate a trained baseline against labeled rows."""

    predictions = tuple(_predict_row(model, row) for row in rows)
    return EvaluationResult(
        model_hash=model.model_hash,
        metrics=_classification_metrics(predictions),
        predictions=predictions,
    )


def write_training_artifacts(result: TrainingResult, output_dir: Path) -> TrainingArtifactPaths:
    """Write model, metrics, and metadata JSON artifacts outside the repository by default."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.json"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "metadata.json"
    model_path.write_text(result.model.model_dump_json(indent=2), encoding="utf-8")
    metrics_payload = {
        "schema_version": "ml.training_metrics.v1",
        "dataset_hash": result.dataset_hash,
        "model_hash": result.model.model_hash,
        "train": result.train_metrics.model_dump(mode="json"),
        "validation": result.validation_metrics.model_dump(mode="json"),
        "usage_limitations": result.usage_limitations,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    model_sha256 = _file_sha256(model_path)
    metrics_sha256 = _file_sha256(metrics_path)
    metadata_payload = {
        "schema_version": "ml.training_metadata.v1",
        "config": result.config.model_dump(mode="json"),
        "device": result.device.model_dump(mode="json"),
        "dataset_hash": result.dataset_hash,
        "feature_names": list(result.model.feature_names),
        "model_artifact_sha256": model_sha256,
        "model_hash": result.model.model_hash,
        "metrics": {
            "train": result.train_metrics.model_dump(mode="json"),
            "validation": result.validation_metrics.model_dump(mode="json"),
        },
        "metrics_artifact_sha256": metrics_sha256,
        "runtime": result.runtime_metadata,
        "split": _split_metadata(result.split),
        "trained_at": result.trained_at.isoformat(),
        "usage_limitations": result.usage_limitations,
    }
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata_sha256 = _file_sha256(metadata_path)
    return TrainingArtifactPaths(
        model_path=model_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        model_sha256=model_sha256,
        metrics_sha256=metrics_sha256,
        metadata_sha256=metadata_sha256,
    )


def load_model_artifact(path: Path) -> TechnicalLogisticModel:
    """Load a model JSON artifact written by ``write_training_artifacts``."""

    return TechnicalLogisticModel.model_validate_json(path.read_text(encoding="utf-8"))


def _fit_logistic_regression(
    rows: Sequence[TechnicalFeatureRow],
    *,
    feature_means: Sequence[float],
    feature_stds: Sequence[float],
    config: TrainingConfig,
) -> tuple[tuple[float, ...], float]:
    feature_count = len(rows[0].feature_values)
    rng = random.Random(config.seed)
    weights = [rng.uniform(-0.01, 0.01) for _ in range(feature_count)]
    bias = 0.0
    scaled_rows = [_scale_features(row.feature_values, feature_means, feature_stds) for row in rows]
    targets = [float(row.target) for row in rows]
    sample_count = float(len(rows))

    for _epoch in range(config.epochs):
        gradients = [0.0 for _ in weights]
        bias_gradient = 0.0
        for features, target in zip(scaled_rows, targets, strict=True):
            probability = _sigmoid(_dot(weights, features) + bias)
            error = probability - target
            bias_gradient += error
            for index, value in enumerate(features):
                gradients[index] += error * value
        for index, gradient in enumerate(gradients):
            l2_gradient = config.l2_penalty * weights[index]
            weights[index] -= config.learning_rate * ((gradient / sample_count) + l2_gradient)
        bias -= config.learning_rate * (bias_gradient / sample_count)

    return tuple(round(weight, 12) for weight in weights), round(bias, 12)


def _fit_scaler(rows: Sequence[TechnicalFeatureRow]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    feature_count = len(rows[0].feature_values)
    columns = [[row.feature_values[index] for row in rows] for index in range(feature_count)]
    means = tuple(_mean(column) for column in columns)
    stds = tuple(max(_stddev(column), 1e-9) for column in columns)
    return means, stds


def _predict_row(model: TechnicalLogisticModel, row: TechnicalFeatureRow) -> Prediction:
    scaled = _scale_features(row.feature_values, model.feature_means, model.feature_stds)
    probability = _sigmoid(_dot(model.weights, scaled) + model.bias)
    return Prediction(
        ticker=row.ticker,
        feature_end=row.feature_end,
        label_end=row.label_end,
        probability=probability,
        predicted_target=int(probability >= model.threshold),
        actual_target=row.target,
        forward_return=row.forward_return,
    )


def _classification_metrics(predictions: Sequence[Prediction]) -> ModelMetrics:
    sample_count = len(predictions)
    if sample_count == 0:
        return ModelMetrics(
            samples=0,
            accuracy=0.0,
            log_loss=0.0,
            brier_score=0.0,
            positive_rate=0.0,
            mean_probability=0.0,
            baseline_accuracy=0.0,
        )
    positives = sum(prediction.actual_target for prediction in predictions)
    positive_rate = positives / sample_count
    correct = sum(
        int(prediction.predicted_target == prediction.actual_target) for prediction in predictions
    )
    probabilities = [prediction.probability for prediction in predictions]
    targets = [prediction.actual_target for prediction in predictions]
    epsilon = 1e-12
    log_loss_value = (
        -sum(
            target * log(max(probability, epsilon))
            + (1 - target) * log(max(1.0 - probability, epsilon))
            for probability, target in zip(probabilities, targets, strict=True)
        )
        / sample_count
    )
    brier_score = (
        sum(
            (probability - target) ** 2
            for probability, target in zip(probabilities, targets, strict=True)
        )
        / sample_count
    )
    mean_probability = _mean(probabilities)
    return ModelMetrics(
        samples=sample_count,
        accuracy=round(correct / sample_count, 6),
        log_loss=round(log_loss_value, 6),
        brier_score=round(brier_score, 6),
        positive_rate=round(positive_rate, 6),
        mean_probability=round(mean_probability, 6),
        baseline_accuracy=round(max(positive_rate, 1.0 - positive_rate), 6),
    )


def _scale_features(
    feature_values: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        (value - mean) / std for value, mean, std in zip(feature_values, means, stds, strict=True)
    )


def _hash_model(
    *,
    dataset: TechnicalDataset,
    config: TrainingConfig,
    weights: Sequence[float],
    bias: float,
    feature_means: Sequence[float],
    feature_stds: Sequence[float],
) -> str:
    payload = {
        "dataset_hash": dataset.dataset_hash,
        "feature_names": dataset.feature_names,
        "config": config.model_dump(mode="json"),
        "weights": [round(value, 12) for value in weights],
        "bias": round(bias, 12),
        "feature_means": [round(value, 12) for value in feature_means],
        "feature_stds": [round(value, 12) for value in feature_stds],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_metadata(device: TrainingDeviceMetadata) -> JsonObject:
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "selected_device": device.selected_device,
        "cuda_available": device.cuda_available,
        "gpu_name": device.gpu_name,
        "cuda_version": device.cuda_version,
        "backend": device.backend,
    }


def _split_metadata(split: DatasetSplit) -> JsonObject:
    return {
        "train_rows": len(split.train_rows),
        "validation_rows": len(split.validation_rows),
        "purged_row_count": split.purged_row_count,
        "train_fraction": split.train_fraction,
        "train_feature_start": _timestamp_to_string(split.train_rows[0].feature_start),
        "train_feature_end": _timestamp_to_string(split.train_rows[-1].feature_end),
        "train_label_end": _timestamp_to_string(split.train_rows[-1].label_end),
        "validation_feature_start": _timestamp_to_string(split.validation_rows[0].feature_start),
        "validation_feature_end": _timestamp_to_string(split.validation_rows[-1].feature_end),
        "validation_label_end": _timestamp_to_string(split.validation_rows[-1].label_end),
    }


def _timestamp_to_string(value: date | datetime) -> str:
    return value.isoformat()


def _infer_feature_window(rows: Sequence[TechnicalFeatureRow]) -> int:
    first = rows[0]
    return max(2, first.feature_end_index + 1)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1.0
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return sqrt(variance)


__all__ = [
    "EvaluationResult",
    "ModelMetrics",
    "Prediction",
    "TechnicalLogisticModel",
    "TrainingArtifactPaths",
    "TrainingConfig",
    "TrainingDeviceMetadata",
    "TrainingResult",
    "detect_training_device",
    "evaluate_model",
    "load_model_artifact",
    "train_technical_model",
    "write_training_artifacts",
]
