"""Windows-first TimesFM 2.5 local smoke gate."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

DeviceRequest = Literal["auto", "cpu", "cuda"]

DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-transformers"
DEFAULT_OUTPUT_PATH = Path("artifacts/ml/timesfm-smoke/smoke-result.json")
_INSTALL_HINT = (
    "Install the optional TimesFM stack first. On Windows with an NVIDIA GPU, install a CUDA "
    "PyTorch wheel from https://pytorch.org/get-started/locally/ and then run "
    '`python -m pip install -e ".[timesfm]"`.'
)


class TimesFmSmokeError(RuntimeError):
    """Raised for expected, actionable TimesFM smoke failures."""


@dataclass(frozen=True)
class TimesFmSmokeConfig:
    """Configuration for a tiny TimesFM load, forecast, and LoRA optimization smoke."""

    model_id: str = DEFAULT_MODEL_ID
    requested_device: DeviceRequest = "auto"
    output_path: Path = DEFAULT_OUTPUT_PATH
    steps: int = 2
    context_len: int = 128
    horizon_len: int = 16
    batch_size: int = 2
    seed: int = 42
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class TimesFmSmokeResult:
    """Serializable result for the local TimesFM environment gate."""

    status: str
    model_id: str
    platform: str
    python: str
    torch: str
    transformers: str
    peft: str
    cuda_available: bool
    cuda_runtime: str | None
    requested_device: str
    selected_device: str
    device_name: str | None
    device_capability: list[int] | None
    total_vram_gb: float | None
    context_len: int
    horizon_len: int
    batch_size: int
    steps: int
    forecast_mean_shape: list[int]
    forecast_full_shape: list[int]
    losses: list[float]
    elapsed_seconds: float
    trainable_params: int
    total_params: int
    trainable_param_percent: float
    memory_allocated_gb: float | None
    max_memory_allocated_gb: float | None
    limitation: str

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable payload."""

        return asdict(self)


@dataclass(frozen=True)
class _TimesFmStack:
    torch: Any
    model_cls: Any
    lora_config_cls: Any
    get_peft_model: Any


def run_timesfm_smoke(config: TimesFmSmokeConfig) -> TimesFmSmokeResult:
    """Run a tiny TimesFM 2.5 inference and LoRA training smoke."""

    _validate_config(config)
    stack = _load_timesfm_stack()
    torch = stack.torch
    selected_device = _select_device(torch, config.requested_device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if selected_device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    try:
        model = stack.model_cls.from_pretrained(config.model_id)
    except Exception as exc:
        raise TimesFmSmokeError(
            "Failed to load TimesFM 2.5 from Hugging Face. Check internet access, "
            "Hugging Face availability, local cache permissions, and HF_TOKEN if rate-limited. "
            f"Model ID: {config.model_id}. Original error: {exc}"
        ) from exc

    model = model.to(selected_device).to(torch.float32)
    model.eval()
    with torch.no_grad():
        forecast_outputs = model(
            past_values=[torch.linspace(0, 1, config.context_len, device=selected_device)],
            forecast_context_len=max(config.context_len, 256),
        )
    forecast_mean_shape = _tensor_shape(forecast_outputs.mean_predictions)
    forecast_full_shape = _tensor_shape(forecast_outputs.full_predictions)

    lora_config = stack.lora_config_cls(
        r=4,
        lora_alpha=8,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
    )
    model = stack.get_peft_model(model, lora_config)
    model.train()
    trainable_params = _count_parameters(model, require_trainable=True)
    total_params = _count_parameters(model, require_trainable=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    series = _synthetic_series(torch, config)
    losses: list[float] = []
    for _ in range(config.steps):
        past_values, future_values = _sample_batch(torch, series, config, selected_device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            past_values=past_values,
            future_values=future_values,
            forecast_context_len=max(config.context_len, 256),
        )
        loss = outputs.loss
        if loss is None:
            raise TimesFmSmokeError("TimesFM forward pass did not return a training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    elapsed_seconds = time.perf_counter() - start
    cuda_available = bool(torch.cuda.is_available())
    return TimesFmSmokeResult(
        status="passed",
        model_id=config.model_id,
        platform=platform.platform(),
        python=platform.python_version(),
        torch=_package_version("torch", fallback=str(getattr(torch, "__version__", "unknown"))),
        transformers=_package_version("transformers"),
        peft=_package_version("peft"),
        cuda_available=cuda_available,
        cuda_runtime=getattr(getattr(torch, "version", None), "cuda", None),
        requested_device=config.requested_device,
        selected_device=selected_device,
        device_name=torch.cuda.get_device_name(0) if cuda_available else None,
        device_capability=list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        total_vram_gb=_total_vram_gb(torch) if cuda_available else None,
        context_len=config.context_len,
        horizon_len=config.horizon_len,
        batch_size=config.batch_size,
        steps=config.steps,
        forecast_mean_shape=forecast_mean_shape,
        forecast_full_shape=forecast_full_shape,
        losses=losses,
        elapsed_seconds=round(elapsed_seconds, 3),
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_param_percent=round(trainable_params / total_params * 100, 4),
        memory_allocated_gb=_cuda_memory_gb(torch, "memory_allocated")
        if selected_device == "cuda"
        else None,
        max_memory_allocated_gb=_cuda_memory_gb(torch, "max_memory_allocated")
        if selected_device == "cuda"
        else None,
        limitation=(
            "Environment smoke only; synthetic data and tiny LoRA steps do not produce a usable "
            "financial model or recommendation."
        ),
    )


def write_smoke_result(result: TimesFmSmokeResult, path: Path) -> None:
    """Write the smoke result JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[TimesFmSmokeConfig], TimesFmSmokeResult] = run_timesfm_smoke,
) -> int:
    """CLI entry point for ``python -m nlp_stock_prediction.ml.timesfm.smoke``."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    config = TimesFmSmokeConfig(
        model_id=args.model_id,
        requested_device=args.device,
        output_path=Path(args.output),
        steps=args.steps,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )
    try:
        result = runner(config)
        write_smoke_result(result, config.output_path)
    except TimesFmSmokeError as exc:
        print(f"TimesFM smoke failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_json(), sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--horizon-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser


def _validate_config(config: TimesFmSmokeConfig) -> None:
    if config.steps < 1:
        raise TimesFmSmokeError("steps must be at least 1 to verify LoRA optimization.")
    if config.context_len < 16:
        raise TimesFmSmokeError("context length must be at least 16.")
    if config.horizon_len < 1:
        raise TimesFmSmokeError("horizon length must be at least 1.")
    if config.batch_size < 1:
        raise TimesFmSmokeError("batch size must be at least 1.")
    if config.learning_rate <= 0:
        raise TimesFmSmokeError("learning rate must be positive.")


def _load_timesfm_stack() -> _TimesFmStack:
    torch = _import_required_module("torch", "torch")
    transformers = _import_required_module("transformers", "transformers>=5.8")
    peft = _import_required_module("peft", "peft>=0.19")
    model_cls = getattr(transformers, "TimesFm2_5ModelForPrediction", None)
    if model_cls is None:
        raise TimesFmSmokeError(
            "Installed Transformers does not expose TimesFm2_5ModelForPrediction. "
            'Upgrade the optional TimesFM stack with `python -m pip install -e ".[timesfm]"`.'
        )
    lora_config_cls = getattr(peft, "LoraConfig", None)
    get_peft_model = getattr(peft, "get_peft_model", None)
    if lora_config_cls is None or get_peft_model is None:
        raise TimesFmSmokeError(
            "Installed PEFT package does not expose LoRA helpers. "
            'Upgrade the optional TimesFM stack with `python -m pip install -e ".[timesfm]"`.'
        )
    return _TimesFmStack(
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
        raise TimesFmSmokeError(
            "Could not import optional TimesFM dependency "
            f"`{package_hint}` because `{missing_name}` is missing. {_INSTALL_HINT}"
        ) from exc


def _select_device(torch: Any, requested_device: DeviceRequest) -> str:
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda":
        if not cuda_available:
            raise TimesFmSmokeError(
                "CUDA was requested but PyTorch cannot access a CUDA GPU. Verify `nvidia-smi`, "
                "install a CUDA-enabled PyTorch wheel, and rerun the smoke command."
            )
        return "cuda"
    if requested_device == "cpu":
        return "cpu"
    return "cuda" if cuda_available else "cpu"


def _synthetic_series(torch: Any, config: TimesFmSmokeConfig) -> list[Any]:
    total_len = config.context_len + config.horizon_len + 32
    series: list[Any] = []
    for index in range(max(8, config.batch_size * 4)):
        timeline = torch.arange(total_len, dtype=torch.float32)
        trend = 0.002 * (index + 1) * timeline
        seasonal = 0.08 * torch.sin(timeline / (5.0 + index % 5))
        momentum = 0.03 * torch.cos(timeline / (11.0 + index % 7))
        noise = 0.01 * torch.randn(total_len)
        series.append(100.0 + index + trend + seasonal + momentum + noise)
    return series


def _sample_batch(
    torch: Any,
    series: Sequence[Any],
    config: TimesFmSmokeConfig,
    selected_device: str,
) -> tuple[list[Any], Any]:
    past_values: list[Any] = []
    future_values: list[Any] = []
    for _ in range(config.batch_size):
        item = random.choice(series)
        max_start = int(item.numel()) - config.context_len - config.horizon_len
        offset = random.randint(0, max_start)
        past_values.append(item[offset : offset + config.context_len].to(selected_device))
        future_values.append(
            item[offset + config.context_len : offset + config.context_len + config.horizon_len].to(
                selected_device
            )
        )
    return past_values, torch.stack(future_values)


def _tensor_shape(tensor: Any) -> list[int]:
    return [int(dimension) for dimension in tensor.shape]


def _count_parameters(model: Any, *, require_trainable: bool) -> int:
    return sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if not require_trainable or bool(parameter.requires_grad)
    )


def _cuda_memory_gb(torch: Any, method_name: str) -> float:
    method = getattr(torch.cuda, method_name)
    return round(float(method()) / (1024**3), 3)


def _total_vram_gb(torch: Any) -> float:
    return round(float(torch.cuda.get_device_properties(0).total_memory) / (1024**3), 2)


def _package_version(distribution: str, *, fallback: str = "unknown") -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


if __name__ == "__main__":
    raise SystemExit(main())
