"""Command line entry point for local TimesFM 2.5 LoRA training."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from importlib import metadata
from pathlib import Path
from typing import Any, cast

from nlp_stock_prediction.contracts import PriceBar
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.ml.timesfm.artifacts import (
    DeviceRequest,
    SelectedDevice,
    TimesFmLoraConfig,
    TimesFmLossMetrics,
    TimesFmTrainingArtifactPaths,
    TimesFmTrainingConfig,
    TimesFmTrainingDeviceMetadata,
    TimesFmTrainingResult,
    TimesFmTrainingSourceKind,
    write_timesfm_training_artifacts,
)
from nlp_stock_prediction.ml.timesfm.dataset import (
    TimesFmDataset,
    TimesFmDatasetConfig,
    TimesFmWindow,
    build_timesfm_dataset,
)
from nlp_stock_prediction.ml.timesfm.smoke import DEFAULT_MODEL_ID
from nlp_stock_prediction.ml.train import load_price_bars_csv

DEFAULT_OUTPUT_DIR = Path("artifacts/ml/timesfm-train-smoke")
_INSTALL_HINT = (
    "Install the optional TimesFM stack first. On Windows with an NVIDIA GPU, install a CUDA "
    "PyTorch wheel from https://pytorch.org/get-started/locally/ and then run "
    '`python -m pip install -e ".[timesfm]"`.'
)


class TimesFmTrainError(RuntimeError):
    """Raised for expected TimesFM training setup or runtime errors."""


@dataclass(frozen=True)
class TimesFmTrainingSource:
    """Source data identity for one TimesFM training run."""

    kind: TimesFmTrainingSourceKind
    sha256: str
    path: str | None = None


@dataclass(frozen=True)
class TimesFmTrainingRun:
    """In-memory training run plus the PEFT adapter model to save."""

    result: TimesFmTrainingResult
    adapter_model: Any


@dataclass(frozen=True)
class _TimesFmTrainingStack:
    torch: Any
    model_cls: Any
    lora_config_cls: Any
    get_peft_model: Any


TrainingRunner = Callable[
    [TimesFmDataset, TimesFmTrainingConfig, TimesFmTrainingSource],
    TimesFmTrainingRun,
]
ArtifactWriter = Callable[[TimesFmTrainingRun, Path], TimesFmTrainingArtifactPaths]


def train_timesfm_lora(
    dataset: TimesFmDataset,
    config: TimesFmTrainingConfig,
    source: TimesFmTrainingSource,
    *,
    stack: _TimesFmTrainingStack | None = None,
    model: Any | None = None,
) -> TimesFmTrainingRun:
    """Fine-tune a local TimesFM 2.5 model with PEFT LoRA on TimesFM windows."""

    training_stack = stack or _load_timesfm_training_stack()
    torch = training_stack.torch
    selected_device = _select_device(torch, config.requested_device)
    _seed_everything(torch, config.seed, selected_device=selected_device)

    loaded_model = model or _load_model(training_stack, config)
    model_revision = _model_revision(loaded_model, config)
    loaded_model = loaded_model.to(selected_device).to(torch.float32)
    lora_config = training_stack.lora_config_cls(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
    )
    adapter_model = training_stack.get_peft_model(loaded_model, lora_config)
    adapter_model.train()

    trainable_parameters = _count_parameters(adapter_model, require_trainable=True)
    total_parameters = _count_parameters(adapter_model, require_trainable=False)
    if trainable_parameters == 0:
        raise TimesFmTrainError("LoRA setup produced no trainable TimesFM parameters.")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapter_model.parameters() if bool(parameter.requires_grad)],
        lr=config.learning_rate,
    )
    rng = random.Random(config.seed)
    train_metrics = _run_training_steps(
        torch,
        adapter_model,
        optimizer,
        dataset.train_windows,
        config,
        selected_device=selected_device,
        rng=rng,
    )
    validation_metrics = _run_validation_steps(
        torch,
        adapter_model,
        dataset.validation_windows,
        config,
        selected_device=selected_device,
    )
    device = _device_metadata(torch, config.requested_device, selected_device)
    result = TimesFmTrainingResult(
        ticker=dataset.ticker,
        model_id=config.model_id,
        model_revision=model_revision,
        source_kind=source.kind,
        source_path=source.path,
        source_sha256=source.sha256,
        csv_sha256=source.sha256 if source.kind == "csv" else None,
        dataset_hash=dataset.dataset_hash,
        trained_at=datetime.now(UTC),
        config=config,
        device=device,
        dataset_metadata=_dataset_metadata(dataset),
        split_metadata=_split_metadata(dataset),
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        trainable_param_percent=round(trainable_parameters / total_parameters * 100, 6),
        runtime_metadata=_runtime_metadata(torch, device),
    )
    return TimesFmTrainingRun(result=result, adapter_model=adapter_model)


def write_timesfm_training_run(
    run: TimesFmTrainingRun,
    output_dir: Path,
) -> TimesFmTrainingArtifactPaths:
    """Write a trained PEFT adapter and its metadata artifacts."""

    def save_adapter(adapter_dir: Path) -> None:
        save_pretrained = getattr(run.adapter_model, "save_pretrained", None)
        if save_pretrained is None:
            raise TimesFmTrainError("Trained TimesFM adapter does not expose save_pretrained.")
        save_pretrained(str(adapter_dir))

    return write_timesfm_training_artifacts(
        run.result,
        output_dir,
        adapter_saver=save_adapter,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: TrainingRunner = train_timesfm_lora,
    artifact_writer: ArtifactWriter = write_timesfm_training_run,
) -> int:
    """CLI entry point for ``python -m nlp_stock_prediction.ml.timesfm.train``."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        bars, source = _load_training_bars(args)
        dataset = build_timesfm_dataset(
            args.ticker,
            bars,
            config=TimesFmDatasetConfig(
                context_length=args.context_length,
                horizon_length=args.horizon_length,
                target_field=args.target_field,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                as_of=_parse_timestamp(args.as_of) if args.as_of else None,
                max_latest_bar_age_days=args.max_latest_bar_age_days,
            ),
        )
        config = TimesFmTrainingConfig(
            model_id=args.model_id,
            model_revision=args.model_revision,
            requested_device=args.device,
            epochs=args.epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            gradient_clip_norm=args.gradient_clip_norm,
            validation_batches=args.validation_batches,
            lora=TimesFmLoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
            ),
        )
        run = runner(dataset, config, source)
        paths = artifact_writer(run, Path(args.output_dir))
    except (OSError, TimesFmTrainError, ValueError) as exc:
        print(f"TimesFM training failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "adapter_dir": str(paths.adapter_dir),
                "metrics_path": str(paths.metrics_path),
                "metadata_path": str(paths.metadata_path),
                "artifact_sha256": {
                    "adapter": paths.adapter_sha256,
                    "metrics": paths.metrics_sha256,
                    "metadata": paths.metadata_sha256,
                },
                "dataset_hash": run.result.dataset_hash,
                "model_id": run.result.model_id,
                "model_revision": run.result.model_revision,
                "source_kind": run.result.source_kind,
                "source_sha256": run.result.source_sha256,
                "csv_sha256": run.result.csv_sha256,
                "selected_device": run.result.device.selected_device,
                "cuda_available": run.result.device.cuda_available,
                "gpu_name": run.result.device.gpu_name,
                "train_final_loss": run.result.train_metrics.final_loss,
                "validation_final_loss": run.result.validation_metrics.final_loss,
                "usage_limitations": run.result.usage_limitations,
            },
            sort_keys=True,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Local OHLCV CSV path.")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic OHLCV bars for a local training smoke.",
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--horizon-length", type=int, default=16)
    parser.add_argument(
        "--target-field",
        choices=("auto", "close", "adjusted_close"),
        default="auto",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--synthetic-bars", type=int, default=180)
    parser.add_argument("--as-of")
    parser.add_argument("--max-latest-bar-age-days", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--lora-bias", choices=("none", "all", "lora_only"), default="none")
    return parser


def _load_training_bars(
    args: argparse.Namespace,
) -> tuple[tuple[PriceBar, ...], TimesFmTrainingSource]:
    if args.synthetic:
        bars = _synthetic_price_bars(args.ticker, count=args.synthetic_bars)
        return (
            bars,
            TimesFmTrainingSource(
                kind="synthetic",
                sha256=_hash_price_bars(bars),
                path=None,
            ),
        )
    csv_path = Path(args.csv)
    bars = load_price_bars_csv(csv_path, ticker=args.ticker)
    return (
        bars,
        TimesFmTrainingSource(
            kind="csv",
            sha256=_file_sha256(csv_path),
            path=str(csv_path),
        ),
    )


def _load_timesfm_training_stack() -> _TimesFmTrainingStack:
    torch = _import_required_module("torch", "torch")
    transformers = _import_required_module("transformers", "transformers>=5.8")
    peft = _import_required_module("peft", "peft>=0.19")
    model_cls = getattr(transformers, "TimesFm2_5ModelForPrediction", None)
    if model_cls is None:
        raise TimesFmTrainError(
            "Installed Transformers does not expose TimesFm2_5ModelForPrediction. "
            'Upgrade the optional TimesFM stack with `python -m pip install -e ".[timesfm]"`.'
        )
    lora_config_cls = getattr(peft, "LoraConfig", None)
    get_peft_model = getattr(peft, "get_peft_model", None)
    if lora_config_cls is None or get_peft_model is None:
        raise TimesFmTrainError(
            "Installed PEFT package does not expose LoRA helpers. "
            'Upgrade the optional TimesFM stack with `python -m pip install -e ".[timesfm]"`.'
        )
    return _TimesFmTrainingStack(
        torch=torch,
        model_cls=model_cls,
        lora_config_cls=lora_config_cls,
        get_peft_model=get_peft_model,
    )


def _import_required_module(module_name: str, package_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or module_name
        raise TimesFmTrainError(
            "Could not import optional TimesFM dependency "
            f"`{package_hint}` because `{missing_name}` is missing. {_INSTALL_HINT}"
        ) from exc


def _select_device(torch: Any, requested_device: DeviceRequest) -> str:
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda":
        if not cuda_available:
            raise TimesFmTrainError(
                "CUDA was requested but PyTorch cannot access a CUDA GPU. Verify `nvidia-smi`, "
                "install a CUDA-enabled PyTorch wheel, and rerun the training command."
            )
        return "cuda"
    if requested_device == "cpu":
        return "cpu"
    return "cuda" if cuda_available else "cpu"


def _load_model(stack: _TimesFmTrainingStack, config: TimesFmTrainingConfig) -> Any:
    try:
        if config.model_revision is None:
            return stack.model_cls.from_pretrained(config.model_id)
        return stack.model_cls.from_pretrained(config.model_id, revision=config.model_revision)
    except Exception as exc:
        raise TimesFmTrainError(
            "Failed to load TimesFM 2.5 from Hugging Face. Check internet access, local cache "
            f"permissions, HF_TOKEN if rate-limited, and model ID {config.model_id}. "
            f"Original error: {exc}"
        ) from exc


def _run_training_steps(
    torch: Any,
    model: Any,
    optimizer: Any,
    windows: Sequence[TimesFmWindow],
    config: TimesFmTrainingConfig,
    *,
    selected_device: str,
    rng: random.Random,
) -> TimesFmLossMetrics:
    losses: list[float] = []
    samples = 0
    step_count = 0
    for _epoch in range(config.epochs):
        if step_count >= config.max_steps:
            break
        for _ in range(config.max_steps - step_count):
            past_values, future_values = _sample_batch(
                torch,
                windows,
                config.batch_size,
                selected_device=selected_device,
                rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                past_values=past_values,
                future_values=future_values,
                forecast_context_len=max(windows[0].context_length, 256),
            )
            loss = _output_loss(outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            losses.append(_loss_to_float(loss))
            samples += len(past_values)
            step_count += 1
            if step_count >= config.max_steps:
                break
    return _loss_metrics(losses, samples=samples)


def _run_validation_steps(
    torch: Any,
    model: Any,
    windows: Sequence[TimesFmWindow],
    config: TimesFmTrainingConfig,
    *,
    selected_device: str,
) -> TimesFmLossMetrics:
    losses: list[float] = []
    samples = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch in _validation_batches(
                windows,
                batch_size=config.batch_size,
                max_batches=config.validation_batches,
            ):
                past_values, future_values = _batch_to_tensors(
                    torch,
                    batch,
                    selected_device=selected_device,
                )
                outputs = model(
                    past_values=past_values,
                    future_values=future_values,
                    forecast_context_len=max(batch[0].context_length, 256),
                )
                losses.append(_loss_to_float(_output_loss(outputs)))
                samples += len(batch)
    finally:
        model.train()
    return _loss_metrics(losses, samples=samples)


def _sample_batch(
    torch: Any,
    windows: Sequence[TimesFmWindow],
    batch_size: int,
    *,
    selected_device: str,
    rng: random.Random,
) -> tuple[list[Any], Any]:
    if not windows:
        raise TimesFmTrainError("TimesFM training requires at least one train window.")
    batch = [rng.choice(windows) for _ in range(batch_size)]
    return _batch_to_tensors(torch, batch, selected_device=selected_device)


def _batch_to_tensors(
    torch: Any,
    windows: Sequence[TimesFmWindow],
    *,
    selected_device: str,
) -> tuple[list[Any], Any]:
    past_values = [
        torch.tensor(list(window.context_values), dtype=torch.float32, device=selected_device)
        for window in windows
    ]
    future_values = torch.stack(
        [
            torch.tensor(list(window.future_values), dtype=torch.float32, device=selected_device)
            for window in windows
        ]
    )
    return past_values, future_values


def _validation_batches(
    windows: Sequence[TimesFmWindow],
    *,
    batch_size: int,
    max_batches: int,
) -> tuple[tuple[TimesFmWindow, ...], ...]:
    batches: list[tuple[TimesFmWindow, ...]] = []
    offset = 0
    while offset < len(windows) and len(batches) < max_batches:
        batch = tuple(windows[offset : offset + batch_size])
        if batch:
            batches.append(batch)
        offset += batch_size
    return tuple(batches)


def _output_loss(outputs: Any) -> Any:
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise TimesFmTrainError("TimesFM forward pass did not return a training loss.")
    return loss


def _loss_to_float(loss: Any) -> float:
    value = loss
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return float(value)


def _loss_metrics(losses: Sequence[float], *, samples: int) -> TimesFmLossMetrics:
    if not losses:
        return TimesFmLossMetrics(steps=0, samples=samples)
    return TimesFmLossMetrics(
        steps=len(losses),
        samples=samples,
        losses=tuple(round(value, 8) for value in losses),
        final_loss=round(losses[-1], 8),
        mean_loss=round(sum(losses) / len(losses), 8),
    )


def _count_parameters(model: Any, *, require_trainable: bool) -> int:
    return sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if not require_trainable or bool(parameter.requires_grad)
    )


def _seed_everything(torch: Any, seed: int, *, selected_device: str) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if selected_device == "cuda" and hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(seed)


def _device_metadata(
    torch: Any,
    requested_device: DeviceRequest,
    selected_device: str,
) -> TimesFmTrainingDeviceMetadata:
    cuda_available = bool(torch.cuda.is_available())
    notes: list[str] = []
    if cuda_available and selected_device == "cpu":
        notes.append("CUDA detected but CPU was selected for this run.")
    if not cuda_available and selected_device == "cpu":
        notes.append("CUDA is not available; using CPU execution.")
    version = getattr(torch, "version", None)
    cuda_version = getattr(version, "cuda", None)
    return TimesFmTrainingDeviceMetadata(
        requested_device=requested_device,
        selected_device=cast(SelectedDevice, selected_device),
        cuda_available=cuda_available,
        gpu_name=str(torch.cuda.get_device_name(0)) if cuda_available else None,
        cuda_version=str(cuda_version) if cuda_version is not None else None,
        backend="timesfm-2.5-peft-lora",
        notes=tuple(notes),
    )


def _runtime_metadata(torch: Any, device: TimesFmTrainingDeviceMetadata) -> JsonObject:
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": _package_version("torch", fallback=str(getattr(torch, "__version__", ""))),
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
        "selected_device": device.selected_device,
        "cuda_available": device.cuda_available,
        "gpu_name": device.gpu_name,
        "cuda_version": device.cuda_version,
        "backend": device.backend,
    }


def _dataset_metadata(dataset: TimesFmDataset) -> JsonObject:
    payload = dataset.model_dump(mode="json")
    return cast(JsonObject, payload["metadata"])


def _split_metadata(dataset: TimesFmDataset) -> JsonObject:
    return {
        "train_windows": len(dataset.train_windows),
        "validation_windows": len(dataset.validation_windows),
        "test_windows": len(dataset.test_windows),
        "context_length": dataset.context_length,
        "horizon_length": dataset.horizon_length,
        "target_field": dataset.target_field,
        "train_context_start": _timestamp_to_string(dataset.train_windows[0].context_start),
        "train_context_end": _timestamp_to_string(dataset.train_windows[-1].context_end),
        "validation_context_start": _timestamp_to_string(
            dataset.validation_windows[0].context_start
        ),
        "validation_context_end": _timestamp_to_string(dataset.validation_windows[-1].context_end),
        "test_context_start": _timestamp_to_string(dataset.test_windows[0].context_start),
        "test_context_end": _timestamp_to_string(dataset.test_windows[-1].context_end),
    }


def _model_revision(model: Any, config: TimesFmTrainingConfig) -> str | None:
    if config.model_revision is not None:
        return config.model_revision
    model_config = getattr(model, "config", None)
    revision = getattr(model_config, "_commit_hash", None)
    if isinstance(revision, str) and revision:
        return revision
    return None


def _parse_timestamp(value: str) -> date | datetime:
    stripped = value.strip()
    try:
        return date.fromisoformat(stripped)
    except ValueError:
        parsed = datetime.fromisoformat(stripped)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("datetime values must include a timezone") from None
        return parsed


def _synthetic_price_bars(ticker: str, *, count: int) -> tuple[PriceBar, ...]:
    if count < 40:
        raise ValueError("synthetic TimesFM training smoke requires at least 40 bars")
    random.seed(42)
    run_date = date(2026, 5, 11)
    bars: list[PriceBar] = []
    previous_close = Decimal("100")
    for index in range(count):
        trend = Decimal(index) * Decimal("0.08")
        cycle = Decimal((index % 11) - 5) * Decimal("0.17")
        noise = Decimal(str(round(random.uniform(-0.04, 0.04), 4)))
        close = Decimal("100") + trend + cycle + noise
        open_price = previous_close * Decimal("1.001")
        high = max(open_price, close) * Decimal("1.006")
        low = min(open_price, close) * Decimal("0.994")
        bars.append(
            PriceBar(
                ticker=ticker,
                timestamp=run_date - timedelta(days=count - index),
                open=open_price.quantize(Decimal("0.0001")),
                high=high.quantize(Decimal("0.0001")),
                low=low.quantize(Decimal("0.0001")),
                close=close.quantize(Decimal("0.0001")),
                volume=1_000_000 + (index % 9) * 25_000,
                adjusted_close=(close * Decimal("0.97")).quantize(Decimal("0.0001")),
            )
        )
        previous_close = close
    return tuple(bars)


def _hash_price_bars(bars: Sequence[PriceBar]) -> str:
    payload = [bar.model_dump(mode="json") for bar in bars]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp_to_string(value: date | datetime) -> str:
    return value.isoformat()


def _package_version(distribution: str, *, fallback: str = "unknown") -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "TimesFmTrainError",
    "TimesFmTrainingRun",
    "TimesFmTrainingSource",
    "train_timesfm_lora",
    "write_timesfm_training_run",
]
