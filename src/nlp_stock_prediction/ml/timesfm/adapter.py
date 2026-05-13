"""Local TimesFM 2.5 inference adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from nlp_stock_prediction.contracts import PriceBar
from nlp_stock_prediction.ml.timesfm.contracts import (
    TimesFmForecastArtifact,
    TimesFmQuantileForecast,
)
from nlp_stock_prediction.ml.timesfm.dataset import (
    TimesFmDataset,
    TimesFmDatasetConfig,
    TimesFmWindow,
    build_timesfm_dataset,
)
from nlp_stock_prediction.ml.train import load_price_bars_csv

DeviceRequest = Literal["auto", "cpu", "cuda"]

DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-transformers"
DEFAULT_FORECAST_OUTPUT_PATH = Path("artifacts/ml/timesfm-forecast-smoke/forecast.json")
_INSTALL_HINT = (
    "Install the optional TimesFM stack first. On Windows with an NVIDIA GPU, install a CUDA "
    "PyTorch wheel from https://pytorch.org/get-started/locally/ and then run "
    '`python -m pip install -e ".[timesfm-raw]"`.'
)


class TimesFmForecastError(RuntimeError):
    """Raised for expected TimesFM adapter setup errors."""


@dataclass(frozen=True)
class TimesFmForecastConfig:
    """Configuration for one local TimesFM forecast run."""

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = None
    requested_device: DeviceRequest = "auto"
    forecast_timestamp: datetime | None = None


@dataclass(frozen=True)
class _InferenceStack:
    torch: Any
    model_cls: Any


def forecast_timesfm_dataset(
    dataset: TimesFmDataset,
    *,
    config: TimesFmForecastConfig | None = None,
    stack: _InferenceStack | None = None,
    model: Any | None = None,
) -> TimesFmForecastArtifact:
    """Run TimesFM inference on the latest context window in ``dataset``.

    Expected model/runtime failures return an ``unavailable`` artifact rather than raising so later
    report integration can surface the failure without breaking the full report pipeline.
    """

    settings = config or TimesFmForecastConfig()
    window = _latest_window(dataset)
    forecast_timestamp = settings.forecast_timestamp or datetime.now(UTC)
    input_hash = _hash_input(dataset, window)
    try:
        return _forecast_with_model(
            dataset,
            window,
            input_hash=input_hash,
            forecast_timestamp=forecast_timestamp,
            config=settings,
            stack=stack,
            model=model,
        )
    except Exception as exc:
        return _unavailable_artifact(
            dataset,
            window,
            input_hash=input_hash,
            forecast_timestamp=forecast_timestamp,
            config=settings,
            warning=f"timesfm_forecast_unavailable:{type(exc).__name__}",
            error_message=str(exc),
        )


def write_forecast_artifact(artifact: TimesFmForecastArtifact, path: Path) -> None:
    """Write a TimesFM forecast artifact as stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[TimesFmDataset, TimesFmForecastConfig], TimesFmForecastArtifact]
    | None = None,
) -> int:
    """CLI entry point for ``python -m nlp_stock_prediction.ml.timesfm.adapter``."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        bars = _load_cli_bars(args)
        dataset = build_timesfm_dataset(
            args.ticker,
            bars,
            config=TimesFmDatasetConfig(
                context_length=args.context_length,
                horizon_length=args.horizon_length,
                target_field=args.target_field,
                as_of=_parse_timestamp(args.as_of) if args.as_of else None,
                max_latest_bar_age_days=args.max_latest_bar_age_days,
            ),
        )
        config = TimesFmForecastConfig(
            model_id=args.model_id,
            model_revision=args.model_revision,
            requested_device=args.device,
        )
        artifact = (
            runner(dataset, config)
            if runner is not None
            else forecast_timesfm_dataset(
                dataset,
                config=config,
            )
        )
        output_path = Path(args.output)
        write_forecast_artifact(artifact, output_path)
    except (TimesFmForecastError, ValueError) as exc:
        print(f"TimesFM forecast failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(artifact.model_dump(mode="json"), sort_keys=True))
    return 0 if artifact.status != "unavailable" else 2


def _forecast_with_model(
    dataset: TimesFmDataset,
    window: TimesFmWindow,
    *,
    input_hash: str,
    forecast_timestamp: datetime,
    config: TimesFmForecastConfig,
    stack: _InferenceStack | None,
    model: Any | None,
) -> TimesFmForecastArtifact:
    inference_stack = stack or _load_inference_stack()
    torch = inference_stack.torch
    selected_device = _select_device(torch, config.requested_device)
    loaded_model = model or _load_model(inference_stack, config)
    model_revision = _model_revision(loaded_model, config)
    loaded_model = loaded_model.to(selected_device).to(torch.float32)
    loaded_model.eval()
    context_tensor = torch.tensor(
        list(window.context_values),
        dtype=torch.float32,
        device=selected_device,
    )
    with torch.no_grad():
        outputs = loaded_model(
            past_values=[context_tensor],
            forecast_context_len=max(window.context_length, 256),
        )
    point_forecast = tuple(_first_batch_sequence(outputs.mean_predictions)[: window.horizon_length])
    if len(point_forecast) != window.horizon_length:
        raise TimesFmForecastError("TimesFM returned fewer point forecasts than requested horizon")
    full_predictions = _validated_full_predictions(
        _first_batch_matrix(outputs.full_predictions),
        horizon_length=window.horizon_length,
    )
    quantile_forecasts = _quantile_forecasts(full_predictions)
    summary = _forecast_summary(
        last_context_value=window.context_values[-1],
        point_forecast=point_forecast,
        full_predictions=full_predictions,
    )
    status = "usable" if quantile_forecasts else "weak"
    warning_ids = () if quantile_forecasts else ("timesfm_forecast_missing_full_predictions",)
    return TimesFmForecastArtifact(
        status=status,
        ticker=dataset.ticker,
        model_id=config.model_id,
        model_revision=model_revision,
        dataset_hash=dataset.dataset_hash,
        input_hash=input_hash,
        forecast_timestamp=forecast_timestamp,
        target_field=dataset.target_field,
        context_start=_as_datetime(window.context_start),
        context_end=_as_datetime(window.context_end),
        forecast_horizon_sessions=window.horizon_length,
        point_forecast=point_forecast,
        quantile_forecasts=quantile_forecasts,
        expected_return=summary["expected_return"],
        interval_width=summary["interval_width"],
        directional_probability_proxy=summary["directional_probability_proxy"],
        uncertainty_score=summary["uncertainty_score"],
        warning_ids=warning_ids,
        metadata={
            "selected_device": selected_device,
            "context_length": window.context_length,
            "horizon_length": window.horizon_length,
            "context_end_index": window.context_end_index,
            "forecast_source": "timesfm_transformers_local",
            "torch_version": _package_version("torch"),
            "transformers_version": _package_version("transformers"),
            "quantile_count": len(quantile_forecasts),
        },
    )


def _unavailable_artifact(
    dataset: TimesFmDataset,
    window: TimesFmWindow,
    *,
    input_hash: str,
    forecast_timestamp: datetime,
    config: TimesFmForecastConfig,
    warning: str,
    error_message: str,
) -> TimesFmForecastArtifact:
    return TimesFmForecastArtifact(
        status="unavailable",
        ticker=dataset.ticker,
        model_id=config.model_id,
        model_revision=config.model_revision,
        dataset_hash=dataset.dataset_hash,
        input_hash=input_hash,
        forecast_timestamp=forecast_timestamp,
        target_field=dataset.target_field,
        context_start=_as_datetime(window.context_start),
        context_end=_as_datetime(window.context_end),
        forecast_horizon_sessions=window.horizon_length,
        warning_ids=(warning,),
        metadata={
            "error": error_message,
            "context_length": window.context_length,
            "horizon_length": window.horizon_length,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Local OHLCV CSV path.")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic OHLCV bars for a local adapter smoke.",
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--output", default=str(DEFAULT_FORECAST_OUTPUT_PATH))
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
    parser.add_argument("--synthetic-bars", type=int, default=180)
    parser.add_argument("--as-of")
    parser.add_argument("--max-latest-bar-age-days", type=int)
    return parser


def _load_cli_bars(args: argparse.Namespace) -> tuple[PriceBar, ...]:
    if args.synthetic:
        return _synthetic_price_bars(args.ticker, count=args.synthetic_bars)
    return load_price_bars_csv(Path(args.csv), ticker=args.ticker)


def _load_inference_stack() -> _InferenceStack:
    torch = _import_required_module("torch", "torch")
    transformers = _import_required_module("transformers", "transformers>=5.8")
    model_cls = getattr(transformers, "TimesFm2_5ModelForPrediction", None)
    if model_cls is None:
        raise TimesFmForecastError(
            "Installed Transformers does not expose TimesFm2_5ModelForPrediction. "
            'Upgrade the optional TimesFM stack with `python -m pip install -e ".[timesfm-raw]"`.'
        )
    return _InferenceStack(torch=torch, model_cls=model_cls)


def _import_required_module(module_name: str, package_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or module_name
        raise TimesFmForecastError(
            "Could not import optional TimesFM dependency "
            f"`{package_hint}` because `{missing_name}` is missing. {_INSTALL_HINT}"
        ) from exc


def _select_device(torch: Any, requested_device: DeviceRequest) -> str:
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda":
        if not cuda_available:
            raise TimesFmForecastError(
                "CUDA was requested but PyTorch cannot access a CUDA GPU. Verify `nvidia-smi`, "
                "install a CUDA-enabled PyTorch wheel, and rerun the forecast command."
            )
        return "cuda"
    if requested_device == "cpu":
        return "cpu"
    return "cuda" if cuda_available else "cpu"


def _load_model(stack: _InferenceStack, config: TimesFmForecastConfig) -> Any:
    try:
        if config.model_revision is None:
            return stack.model_cls.from_pretrained(config.model_id)
        return stack.model_cls.from_pretrained(config.model_id, revision=config.model_revision)
    except Exception as exc:
        raise TimesFmForecastError(
            "Failed to load TimesFM 2.5 from Hugging Face. Check internet access, local cache "
            f"permissions, HF_TOKEN if rate-limited, and model ID {config.model_id}. "
            f"Original error: {exc}"
        ) from exc


def _latest_window(dataset: TimesFmDataset) -> TimesFmWindow:
    return max(dataset.windows, key=lambda window: window.context_end_index)


def _hash_input(dataset: TimesFmDataset, window: TimesFmWindow) -> str:
    payload = {
        "dataset_hash": dataset.dataset_hash,
        "ticker": dataset.ticker,
        "target_field": dataset.target_field,
        "context_start": str(window.context_start),
        "context_end": str(window.context_end),
        "context_values": [round(value, 12) for value in window.context_values],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_batch_sequence(value: Any) -> tuple[float, ...]:
    payload = _to_python(value)
    if not isinstance(payload, list) or not payload:
        raise TimesFmForecastError("TimesFM point forecast output has an unexpected shape")
    first = payload[0] if isinstance(payload[0], list) else payload
    return tuple(_finite_float(item, field_name="point forecast") for item in first)


def _first_batch_matrix(value: Any) -> tuple[tuple[float, ...], ...]:
    payload = _to_python(value)
    if not isinstance(payload, list) or not payload:
        return ()
    first = payload[0]
    if not isinstance(first, list) or not first:
        return ()
    rows: list[tuple[float, ...]] = []
    for row in first:
        if not isinstance(row, list):
            return ()
        rows.append(tuple(_finite_float(item, field_name="full prediction") for item in row))
    return tuple(rows)


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _quantile_forecasts(
    full_predictions: Sequence[Sequence[float]],
) -> tuple[TimesFmQuantileForecast, ...]:
    if not full_predictions:
        return ()
    quantile_count = min(len(row) for row in full_predictions)
    if quantile_count <= 0:
        return ()
    return tuple(
        TimesFmQuantileForecast(
            quantile_index=index,
            values=tuple(row[index] for row in full_predictions),
        )
        for index in range(quantile_count)
    )


def _validated_full_predictions(
    full_predictions: tuple[tuple[float, ...], ...],
    *,
    horizon_length: int,
) -> tuple[tuple[float, ...], ...]:
    if not full_predictions:
        return ()
    if len(full_predictions) < horizon_length:
        raise TimesFmForecastError(
            "TimesFM full prediction horizon did not match the requested forecast horizon"
        )
    sliced_predictions = full_predictions[:horizon_length]
    quantile_count = len(sliced_predictions[0])
    if quantile_count == 0:
        return ()
    if any(len(row) != quantile_count for row in sliced_predictions):
        raise TimesFmForecastError("TimesFM full prediction rows have inconsistent widths")
    return sliced_predictions


def _forecast_summary(
    *,
    last_context_value: float,
    point_forecast: Sequence[float],
    full_predictions: Sequence[Sequence[float]],
) -> dict[str, float | None]:
    if last_context_value == 0 or not point_forecast:
        return {
            "expected_return": None,
            "interval_width": None,
            "directional_probability_proxy": None,
            "uncertainty_score": None,
        }
    expected_return = (point_forecast[-1] / last_context_value) - 1.0
    final_full_predictions = tuple(row[-1] for row in _transpose(full_predictions))
    if final_full_predictions:
        lower_bound = min(final_full_predictions)
        upper_bound = max(final_full_predictions)
        interval_width = abs(upper_bound - lower_bound) / abs(last_context_value)
        directional_probability_proxy = _clamp(
            len([value for value in final_full_predictions if value > last_context_value])
            / len(final_full_predictions)
        )
        uncertainty_score = _clamp(interval_width)
    else:
        interval_width = None
        directional_probability_proxy = 1.0 if point_forecast[-1] > last_context_value else 0.0
        uncertainty_score = None
    return {
        "expected_return": round(expected_return, 8),
        "interval_width": round(interval_width, 8) if interval_width is not None else None,
        "directional_probability_proxy": round(directional_probability_proxy, 6),
        "uncertainty_score": round(uncertainty_score, 6) if uncertainty_score is not None else None,
    }


def _transpose(rows: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if not rows:
        return ()
    width = min(len(row) for row in rows)
    return tuple(tuple(row[index] for row in rows) for index in range(width))


def _finite_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, int | float):
        raise TimesFmForecastError(f"{field_name} output must be numeric")
    float_value = float(value)
    if not math.isfinite(float_value):
        raise TimesFmForecastError(f"{field_name} output must be finite")
    return float_value


def _model_revision(model: Any, config: TimesFmForecastConfig) -> str | None:
    if config.model_revision is not None:
        return config.model_revision
    model_config = getattr(model, "config", None)
    revision = getattr(model_config, "_commit_hash", None)
    if isinstance(revision, str) and revision:
        return revision
    return None


def _as_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise TimesFmForecastError("TimesFM forecast timestamps must be timezone-aware")
        return value.astimezone(UTC)
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


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
        raise ValueError("synthetic TimesFM forecast smoke requires at least 40 bars")
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


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FORECAST_OUTPUT_PATH",
    "DEFAULT_MODEL_ID",
    "TimesFmForecastConfig",
    "TimesFmForecastError",
    "forecast_timesfm_dataset",
    "write_forecast_artifact",
]
