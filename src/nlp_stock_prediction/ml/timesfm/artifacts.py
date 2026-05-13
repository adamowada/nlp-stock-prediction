"""Legacy artifact contracts and writers for local TimesFM LoRA training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
    TickerSymbol,
)

DeviceRequest = Literal["auto", "cpu", "cuda"]
SelectedDevice = Literal["cpu", "cuda"]
TimesFmTrainingSourceKind = Literal["csv", "synthetic"]
TimesFmLoraBias = Literal["none", "all", "lora_only"]


class TimesFmLoraConfig(ContractModel):
    """Conservative PEFT LoRA adapter configuration for TimesFM 2.5."""

    r: int = Field(default=4, ge=1)
    lora_alpha: int = Field(default=8, ge=1)
    target_modules: NonEmptyStr = "all-linear"
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    bias: TimesFmLoraBias = "none"


class TimesFmTrainingConfig(ContractModel):
    """Runtime configuration for one local TimesFM LoRA training run."""

    model_id: NonEmptyStr = "google/timesfm-2.5-200m-transformers"
    model_revision: str | None = None
    requested_device: DeviceRequest = "auto"
    epochs: int = Field(default=1, ge=1)
    max_steps: int = Field(default=20, ge=1)
    batch_size: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    seed: int = 42
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    validation_batches: int = Field(default=2, ge=1)
    lora: TimesFmLoraConfig = Field(default_factory=TimesFmLoraConfig)


class TimesFmTrainingDeviceMetadata(ContractModel):
    """Captured device and CUDA metadata for a TimesFM training run."""

    requested_device: DeviceRequest
    selected_device: SelectedDevice
    cuda_available: bool
    gpu_name: str | None = None
    cuda_version: str | None = None
    backend: NonEmptyStr
    notes: tuple[str, ...] = Field(default_factory=tuple)


class TimesFmLossMetrics(ContractModel):
    """Loss metrics for the TimesFM training or validation split."""

    steps: int = Field(ge=0)
    samples: int = Field(ge=0)
    losses: tuple[float, ...] = Field(default_factory=tuple)
    final_loss: float | None = Field(default=None, ge=0.0)
    mean_loss: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_losses(self) -> TimesFmLossMetrics:
        if any(not isfinite(value) or value < 0.0 for value in self.losses):
            raise ValueError("TimesFM losses must be finite and non-negative")
        if self.steps != len(self.losses):
            raise ValueError("TimesFM metric steps must match recorded losses")
        if self.steps > 0 and not self.losses:
            raise ValueError("TimesFM metrics with steps must include losses")
        if self.steps == 0 and self.losses:
            raise ValueError("TimesFM metrics cannot include losses when steps is zero")
        if self.steps > 0 and (self.final_loss is None or self.mean_loss is None):
            raise ValueError("TimesFM metrics with losses must include final and mean loss")
        if self.steps == 0 and (self.final_loss is not None or self.mean_loss is not None):
            raise ValueError("TimesFM metrics without losses cannot include loss summaries")
        return self


class TimesFmTrainingResult(ContractModel):
    """Serializable metadata for one local TimesFM LoRA training run."""

    model_config = ConfigDict(extra="ignore")

    schema_version: NonEmptyStr = "ml.timesfm.training_result.v1"
    ticker: TickerSymbol
    model_id: NonEmptyStr
    model_revision: str | None = None
    source_kind: TimesFmTrainingSourceKind
    source_path: str | None = None
    source_sha256: NonEmptyStr
    csv_sha256: str | None = None
    dataset_hash: NonEmptyStr
    trained_at: AwareDatetime
    config: TimesFmTrainingConfig
    device: TimesFmTrainingDeviceMetadata
    dataset_metadata: JsonObject
    split_metadata: JsonObject
    train_metrics: TimesFmLossMetrics
    validation_metrics: TimesFmLossMetrics
    trainable_parameters: int = Field(ge=0)
    total_parameters: int = Field(ge=1)
    trainable_param_percent: float = Field(ge=0.0, le=100.0)
    runtime_metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_training_result(self) -> TimesFmTrainingResult:
        if self.source_kind == "csv" and not self.csv_sha256:
            raise ValueError("CSV TimesFM training results must include csv_sha256")
        if self.trainable_parameters > self.total_parameters:
            raise ValueError("trainable parameters cannot exceed total parameters")
        return self


class TimesFmTrainingArtifactPaths(ContractModel):
    """Paths and hashes written by ``write_timesfm_training_artifacts``."""

    adapter_dir: Path
    metrics_path: Path
    metadata_path: Path
    adapter_sha256: NonEmptyStr
    metrics_sha256: NonEmptyStr
    metadata_sha256: NonEmptyStr


def write_timesfm_training_artifacts(
    result: TimesFmTrainingResult,
    output_dir: Path,
    *,
    adapter_saver: Callable[[Path], None],
) -> TimesFmTrainingArtifactPaths:
    """Write adapter, metrics, and metadata artifacts for a TimesFM LoRA run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_saver(adapter_dir)
    adapter_sha256 = directory_sha256(adapter_dir)

    metrics_path = output_dir / "training-metrics.json"
    metrics_payload = _metrics_payload(result)
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_sha256 = file_sha256(metrics_path)

    metadata_path = output_dir / "training-metadata.json"
    metadata_payload = _metadata_payload(
        result,
        adapter_sha256=adapter_sha256,
        metrics_sha256=metrics_sha256,
    )
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_sha256 = file_sha256(metadata_path)

    return TimesFmTrainingArtifactPaths(
        adapter_dir=adapter_dir,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        adapter_sha256=adapter_sha256,
        metrics_sha256=metrics_sha256,
        metadata_sha256=metadata_sha256,
    )


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for one artifact file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(path: Path) -> str:
    """Return a stable SHA-256 digest for all files under a directory."""

    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"adapter directory {path} did not contain any files")
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _metrics_payload(result: TimesFmTrainingResult) -> JsonObject:
    return {
        "schema_version": "ml.timesfm.training_metrics.v1",
        "ticker": result.ticker,
        "model_id": result.model_id,
        "model_revision": result.model_revision,
        "dataset_hash": result.dataset_hash,
        "source_kind": result.source_kind,
        "source_sha256": result.source_sha256,
        "csv_sha256": result.csv_sha256,
        "train": result.train_metrics.model_dump(mode="json"),
        "validation": result.validation_metrics.model_dump(mode="json"),
    }


def _metadata_payload(
    result: TimesFmTrainingResult,
    *,
    adapter_sha256: str,
    metrics_sha256: str,
) -> JsonObject:
    return {
        "schema_version": "ml.timesfm.training_metadata.v1",
        "ticker": result.ticker,
        "model_id": result.model_id,
        "model_revision": result.model_revision,
        "source_kind": result.source_kind,
        "source_path": result.source_path,
        "source_sha256": result.source_sha256,
        "csv_sha256": result.csv_sha256,
        "dataset_hash": result.dataset_hash,
        "trained_at": result.trained_at.isoformat(),
        "config": result.config.model_dump(mode="json"),
        "device": result.device.model_dump(mode="json"),
        "dataset": result.dataset_metadata,
        "split": result.split_metadata,
        "metrics": {
            "train": result.train_metrics.model_dump(mode="json"),
            "validation": result.validation_metrics.model_dump(mode="json"),
        },
        "adapter": {
            "adapter_dir_name": "adapter",
            "adapter_artifact_sha256": adapter_sha256,
            "trainable_parameters": result.trainable_parameters,
            "total_parameters": result.total_parameters,
            "trainable_param_percent": result.trainable_param_percent,
        },
        "metrics_artifact_sha256": metrics_sha256,
        "runtime": result.runtime_metadata,
    }


__all__ = [
    "DeviceRequest",
    "SelectedDevice",
    "TimesFmLoraBias",
    "TimesFmLoraConfig",
    "TimesFmLossMetrics",
    "TimesFmTrainingArtifactPaths",
    "TimesFmTrainingConfig",
    "TimesFmTrainingDeviceMetadata",
    "TimesFmTrainingResult",
    "TimesFmTrainingSourceKind",
    "directory_sha256",
    "file_sha256",
    "write_timesfm_training_artifacts",
]
