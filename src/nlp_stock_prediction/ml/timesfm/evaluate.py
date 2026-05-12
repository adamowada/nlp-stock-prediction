"""Command line entry point for local TimesFM 2.5 rolling evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts import PriceBar
from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
    TickerSymbol,
)
from nlp_stock_prediction.ml.timesfm.artifacts import (
    DeviceRequest,
    SelectedDevice,
    directory_sha256,
    file_sha256,
)
from nlp_stock_prediction.ml.timesfm.dataset import (
    TimesFmDataset,
    TimesFmDatasetConfig,
    TimesFmWindow,
    build_timesfm_dataset,
)
from nlp_stock_prediction.ml.train import load_price_bars_csv

TimesFmEvaluationStatus = Literal["suitable", "weak", "unavailable"]
TimesFmBaselineName = Literal["last_close_persistence", "recent_mean_return"]
TimesFmEvaluationSourceKind = Literal["csv", "synthetic"]

DEFAULT_EVALUATION_OUTPUT_PATH = Path("artifacts/ml/timesfm-eval-smoke/evaluation.json")
_INSTALL_HINT = (
    "Install the optional TimesFM stack first. On Windows with an NVIDIA GPU, install a CUDA "
    "PyTorch wheel from https://pytorch.org/get-started/locally/ and then run "
    '`python -m pip install -e ".[timesfm]"`.'
)
_EVALUATION_LIMITATION = (
    "Experimental local TimesFM technical-analysis evaluation for research only; not investment "
    "advice, not a standalone recommendation, and not live trading instructions."
)
_BLOCKING_MODEL_SOURCE_WARNING_IDS = frozenset({"timesfm_adapter_hash_mismatch"})


class TimesFmEvaluateError(RuntimeError):
    """Raised for expected TimesFM evaluation setup or runtime errors."""


class TimesFmEvaluationConfig(ContractModel):
    """Configuration for one rolling TimesFM evaluation run."""

    requested_device: DeviceRequest = "auto"
    max_windows: int | None = Field(default=None, ge=1)
    min_evaluation_windows: int = Field(default=3, ge=1)
    min_directional_accuracy: float = Field(default=0.50, ge=0.0, le=1.0)
    min_directional_accuracy_delta_vs_best_baseline: float = Field(default=0.0, ge=-1.0, le=1.0)
    max_rmse_ratio_vs_best_baseline: float = Field(default=1.0, gt=0.0)
    suitability_max_latest_bar_age_days: int | None = Field(default=None, ge=0)
    as_of: date | datetime | None = None

    @model_validator(mode="after")
    def validate_evaluation_config(self) -> TimesFmEvaluationConfig:
        if self.suitability_max_latest_bar_age_days is not None and self.as_of is None:
            raise ValueError("suitability_max_latest_bar_age_days requires as_of")
        if isinstance(self.as_of, datetime) and (
            self.as_of.tzinfo is None or self.as_of.utcoffset() is None
        ):
            raise ValueError("as_of datetime values must include a timezone")
        return self


class TimesFmWindowPrediction(ContractModel):
    """Prediction values for one TimesFM evaluation window."""

    point_forecast: tuple[float, ...]
    full_predictions: tuple[tuple[float, ...], ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_prediction(self) -> TimesFmWindowPrediction:
        if not self.point_forecast:
            raise ValueError("point forecast cannot be empty")
        numeric_values = list(self.point_forecast)
        numeric_values.extend(value for row in self.full_predictions for value in row)
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("TimesFM prediction values must be finite")
        if self.full_predictions:
            width = len(self.full_predictions[0])
            if width == 0 or any(len(row) != width for row in self.full_predictions):
                raise ValueError("full prediction rows must have a consistent non-empty width")
        return self


class TimesFmEvaluationMetrics(ContractModel):
    """Aggregate point-forecast and interval metrics."""

    sample_count: int = Field(ge=0)
    mae: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    directional_accuracy: float = Field(ge=0.0, le=1.0)
    mean_return_error: float
    interval_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_interval_width: float | None = Field(default=None, ge=0.0)
    calibration_proxy: float | None = Field(default=None, ge=0.0, le=1.0)


class TimesFmBaselineEvaluation(ContractModel):
    """One baseline comparison against TimesFM point forecasts."""

    name: TimesFmBaselineName
    metrics: TimesFmEvaluationMetrics
    mae_delta_vs_timesfm: float
    rmse_delta_vs_timesfm: float
    directional_accuracy_delta_vs_timesfm: float


class TimesFmEvaluationRecord(ContractModel):
    """One rolling evaluation window record."""

    context_end: date | datetime
    horizon_end: date | datetime
    actual_final_value: float
    timesfm_final_value: float
    persistence_final_value: float
    recent_mean_return_final_value: float
    actual_return: float
    timesfm_return: float
    persistence_return: float
    recent_mean_return: float
    interval_lower: float | None = None
    interval_upper: float | None = None
    interval_covered: bool | None = None


class TimesFmForwardForecast(ContractModel):
    """Forward TimesFM forecast from the latest available local context window."""

    context_start: date | datetime
    context_end: date | datetime
    forecast_horizon_sessions: int = Field(ge=1)
    point_forecast: tuple[float, ...]
    expected_return: float
    interval_lower: float | None = None
    interval_upper: float | None = None
    interval_width: float | None = Field(default=None, ge=0.0)
    directional_probability_proxy: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_forward_forecast(self) -> TimesFmForwardForecast:
        if len(self.point_forecast) != self.forecast_horizon_sessions:
            raise ValueError("forward forecast length must match forecast horizon")
        if any(not math.isfinite(value) for value in self.point_forecast):
            raise ValueError("forward forecast values must be finite")
        if not math.isfinite(self.expected_return):
            raise ValueError("forward forecast expected_return must be finite")
        if (self.interval_lower is None) != (self.interval_upper is None):
            raise ValueError("forward forecast interval bounds must be present together")
        if (
            self.interval_lower is not None
            and self.interval_upper is not None
            and self.interval_lower > self.interval_upper
        ):
            raise ValueError("forward forecast interval_lower cannot exceed interval_upper")
        return self


class TimesFmEvaluationArtifact(ContractModel):
    """Serializable TimesFM rolling evaluation artifact."""

    schema_version: NonEmptyStr = "ml.timesfm.evaluation.v1"
    status: TimesFmEvaluationStatus
    suitable_for_scoring: bool
    suitability_reasons: tuple[str, ...]
    warning_ids: tuple[str, ...] = Field(default_factory=tuple)
    ticker: TickerSymbol
    model_id: NonEmptyStr
    model_revision: str | None = None
    model_hash: NonEmptyStr
    adapter_sha256: NonEmptyStr
    training_metadata_sha256: NonEmptyStr
    dataset_hash: NonEmptyStr
    evaluation_source_kind: TimesFmEvaluationSourceKind
    evaluation_source_sha256: NonEmptyStr
    evaluated_at: AwareDatetime
    latest_bar_timestamp: date | datetime
    as_of: date | datetime | None = None
    config: TimesFmEvaluationConfig
    metrics: TimesFmEvaluationMetrics
    baselines: tuple[TimesFmBaselineEvaluation, ...]
    records: tuple[TimesFmEvaluationRecord, ...]
    forward_forecast: TimesFmForwardForecast | None = None
    training_metadata: JsonObject
    runtime_metadata: JsonObject = Field(default_factory=dict)
    usage_limitations: NonEmptyStr = _EVALUATION_LIMITATION

    @model_validator(mode="after")
    def validate_artifact(self) -> TimesFmEvaluationArtifact:
        if self.suitable_for_scoring and self.status != "suitable":
            raise ValueError("suitable_for_scoring requires suitable status")
        if self.status != "suitable" and not self.suitability_reasons:
            raise ValueError("weak or unavailable evaluations must include suitability reasons")
        if self.metrics.sample_count != len(self.records):
            raise ValueError("evaluation metric sample count must match records")
        if not self.baselines:
            raise ValueError("TimesFM evaluation requires baseline comparisons")
        return self


@dataclass(frozen=True)
class TimesFmEvaluationModelSource:
    """Resolved trained adapter metadata for evaluation."""

    model_dir: Path
    adapter_dir: Path
    training_ticker: str
    model_id: str
    model_revision: str | None
    adapter_sha256: str
    recorded_adapter_sha256: str | None
    training_metadata_sha256: str
    training_metadata: JsonObject
    warning_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimesFmEvaluationDataSource:
    """Identity for the evaluation OHLCV data source."""

    kind: TimesFmEvaluationSourceKind
    sha256: str
    path: str | None = None


@dataclass(frozen=True)
class _TimesFmEvaluationStack:
    torch: Any
    model_cls: Any
    peft_model_cls: Any


WindowPredictor = Callable[[TimesFmWindow], TimesFmWindowPrediction]
EvaluationRunner = Callable[
    [
        TimesFmDataset,
        TimesFmEvaluationModelSource,
        TimesFmEvaluationDataSource,
        TimesFmEvaluationConfig,
    ],
    TimesFmEvaluationArtifact,
]


def evaluate_timesfm_dataset(
    dataset: TimesFmDataset,
    model_source: TimesFmEvaluationModelSource,
    data_source: TimesFmEvaluationDataSource,
    config: TimesFmEvaluationConfig,
    *,
    predictor: WindowPredictor | None = None,
) -> TimesFmEvaluationArtifact:
    """Evaluate a trained TimesFM adapter on rolling held-out windows."""

    if model_source.training_ticker != dataset.ticker:
        raise TimesFmEvaluateError(
            "TimesFM training metadata ticker "
            f"{model_source.training_ticker} does not match evaluation ticker {dataset.ticker}."
        )

    selected_predictor: WindowPredictor
    runtime_metadata: JsonObject
    if predictor is None:
        selected_predictor, runtime_metadata = _load_model_predictor(model_source, config)
    else:
        selected_predictor = predictor
        runtime_metadata = {"backend": "injected_predictor"}

    windows = _evaluation_windows(dataset, config)
    records = tuple(_evaluate_window(window, selected_predictor(window)) for window in windows)
    forward_window = _forward_forecast_window(dataset)
    forward_forecast = _forward_forecast(forward_window, selected_predictor(forward_window))
    metrics = _metrics(
        tuple(record.timesfm_final_value for record in records),
        tuple(record.actual_final_value for record in records),
        tuple(record.actual_return for record in records),
        tuple(record.timesfm_return for record in records),
        interval_covered=tuple(record.interval_covered for record in records),
        interval_widths=tuple(_interval_width(record) for record in records),
    )
    baselines = _baseline_evaluations(records, timesfm_metrics=metrics)
    suitability_reasons = _suitability_reasons(
        metrics,
        baselines,
        dataset=dataset,
        config=config,
    )
    suitability_reasons.extend(
        warning_id
        for warning_id in model_source.warning_ids
        if warning_id in _BLOCKING_MODEL_SOURCE_WARNING_IDS
    )
    status: TimesFmEvaluationStatus = "suitable" if not suitability_reasons else "weak"
    model_hash = _model_hash(
        model_id=model_source.model_id,
        model_revision=model_source.model_revision,
        adapter_sha256=model_source.adapter_sha256,
        training_metadata_sha256=model_source.training_metadata_sha256,
    )
    return TimesFmEvaluationArtifact(
        status=status,
        suitable_for_scoring=status == "suitable",
        suitability_reasons=tuple(suitability_reasons),
        warning_ids=model_source.warning_ids,
        ticker=dataset.ticker,
        model_id=model_source.model_id,
        model_revision=model_source.model_revision,
        model_hash=model_hash,
        adapter_sha256=model_source.adapter_sha256,
        training_metadata_sha256=model_source.training_metadata_sha256,
        dataset_hash=dataset.dataset_hash,
        evaluation_source_kind=data_source.kind,
        evaluation_source_sha256=data_source.sha256,
        evaluated_at=datetime.now(UTC),
        latest_bar_timestamp=_latest_bar_timestamp(dataset),
        as_of=config.as_of,
        config=config,
        metrics=metrics,
        baselines=baselines,
        records=records,
        forward_forecast=forward_forecast,
        training_metadata=model_source.training_metadata,
        runtime_metadata={
            **runtime_metadata,
            "baseline_note": "local_logistic_baseline_not_configured",
        },
    )


def write_timesfm_evaluation_artifact(
    artifact: TimesFmEvaluationArtifact,
    output_path: Path,
) -> None:
    """Write a TimesFM evaluation artifact as stable JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: EvaluationRunner = evaluate_timesfm_dataset,
) -> int:
    """CLI entry point for ``python -m nlp_stock_prediction.ml.timesfm.evaluate``."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        bars, data_source = _load_evaluation_bars(args)
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
        model_source = load_timesfm_evaluation_model_source(Path(args.model_dir))
        config = TimesFmEvaluationConfig(
            requested_device=args.device,
            max_windows=args.max_windows,
            min_evaluation_windows=args.min_evaluation_windows,
            min_directional_accuracy=args.min_directional_accuracy,
            min_directional_accuracy_delta_vs_best_baseline=(
                args.min_directional_accuracy_delta_vs_best_baseline
            ),
            max_rmse_ratio_vs_best_baseline=args.max_rmse_ratio_vs_best_baseline,
            suitability_max_latest_bar_age_days=args.suitability_max_latest_bar_age_days,
            as_of=_parse_timestamp(args.as_of) if args.as_of else None,
        )
        artifact = runner(dataset, model_source, data_source, config)
        write_timesfm_evaluation_artifact(artifact, Path(args.output))
    except (OSError, TimesFmEvaluateError, ValueError) as exc:
        print(f"TimesFM evaluation failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output": str(Path(args.output)),
                "status": artifact.status,
                "suitable_for_scoring": artifact.suitable_for_scoring,
                "suitability_reasons": list(artifact.suitability_reasons),
                "model_hash": artifact.model_hash,
                "adapter_sha256": artifact.adapter_sha256,
                "dataset_hash": artifact.dataset_hash,
                "sample_count": artifact.metrics.sample_count,
                "timesfm_rmse": artifact.metrics.rmse,
                "timesfm_directional_accuracy": artifact.metrics.directional_accuracy,
                "baseline_rmse": {
                    baseline.name: baseline.metrics.rmse for baseline in artifact.baselines
                },
                "usage_limitations": artifact.usage_limitations,
            },
            sort_keys=True,
        )
    )
    return 0


def load_timesfm_evaluation_model_source(model_dir: Path) -> TimesFmEvaluationModelSource:
    """Load adapter metadata and hashes from a Stage 4 TimesFM training directory."""

    metadata_path = model_dir / "training-metadata.json"
    if not metadata_path.exists():
        raise TimesFmEvaluateError(f"missing TimesFM training metadata: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TimesFmEvaluateError("TimesFM training metadata must be a JSON object")
    adapter_payload = payload.get("adapter")
    if not isinstance(adapter_payload, dict):
        raise TimesFmEvaluateError("TimesFM training metadata missing adapter section")
    adapter_dir_name = adapter_payload.get("adapter_dir_name", "adapter")
    if not isinstance(adapter_dir_name, str) or not adapter_dir_name.strip():
        raise TimesFmEvaluateError("TimesFM adapter_dir_name must be a non-empty string")
    adapter_dir = model_dir / adapter_dir_name
    adapter_sha256 = directory_sha256(adapter_dir)
    recorded_adapter_sha256 = adapter_payload.get("adapter_artifact_sha256")
    warnings: list[str] = []
    if isinstance(recorded_adapter_sha256, str) and recorded_adapter_sha256 != adapter_sha256:
        warnings.append("timesfm_adapter_hash_mismatch")
    training_ticker = payload.get("ticker")
    if not isinstance(training_ticker, str) or not training_ticker.strip():
        raise TimesFmEvaluateError("TimesFM training metadata missing ticker")
    model_id = payload.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise TimesFmEvaluateError("TimesFM training metadata missing model_id")
    model_revision = payload.get("model_revision")
    if model_revision is not None and not isinstance(model_revision, str):
        raise TimesFmEvaluateError("TimesFM model_revision must be a string or null")
    return TimesFmEvaluationModelSource(
        model_dir=model_dir,
        adapter_dir=adapter_dir,
        training_ticker=training_ticker.upper(),
        model_id=model_id,
        model_revision=model_revision,
        adapter_sha256=adapter_sha256,
        recorded_adapter_sha256=recorded_adapter_sha256
        if isinstance(recorded_adapter_sha256, str)
        else None,
        training_metadata_sha256=file_sha256(metadata_path),
        training_metadata=cast(JsonObject, payload),
        warning_ids=tuple(warnings),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Local OHLCV CSV path.")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic OHLCV bars for a local evaluation smoke.",
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default=str(DEFAULT_EVALUATION_OUTPUT_PATH))
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
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--min-evaluation-windows", type=int, default=3)
    parser.add_argument("--min-directional-accuracy", type=float, default=0.50)
    parser.add_argument(
        "--min-directional-accuracy-delta-vs-best-baseline",
        type=float,
        default=0.0,
    )
    parser.add_argument("--max-rmse-ratio-vs-best-baseline", type=float, default=1.0)
    parser.add_argument("--suitability-max-latest-bar-age-days", type=int)
    return parser


def _load_evaluation_bars(
    args: argparse.Namespace,
) -> tuple[tuple[PriceBar, ...], TimesFmEvaluationDataSource]:
    if args.synthetic:
        bars = _synthetic_price_bars(args.ticker, count=args.synthetic_bars)
        return (
            bars,
            TimesFmEvaluationDataSource(
                kind="synthetic",
                sha256=_hash_price_bars(bars),
                path=None,
            ),
        )
    csv_path = Path(args.csv)
    bars = load_price_bars_csv(csv_path, ticker=args.ticker)
    return (
        bars,
        TimesFmEvaluationDataSource(
            kind="csv",
            sha256=_file_sha256(csv_path),
            path=str(csv_path),
        ),
    )


def _load_model_predictor(
    model_source: TimesFmEvaluationModelSource,
    config: TimesFmEvaluationConfig,
) -> tuple[WindowPredictor, JsonObject]:
    stack = _load_evaluation_stack()
    torch = stack.torch
    selected_device = _select_device(torch, config.requested_device)
    try:
        if model_source.model_revision is None:
            base_model = stack.model_cls.from_pretrained(model_source.model_id)
        else:
            base_model = stack.model_cls.from_pretrained(
                model_source.model_id,
                revision=model_source.model_revision,
            )
    except Exception as exc:
        raise TimesFmEvaluateError(
            "Failed to load TimesFM 2.5 from Hugging Face. Check internet access, local cache "
            f"permissions, HF_TOKEN if rate-limited, and model ID {model_source.model_id}. "
            f"Original error: {exc}"
        ) from exc
    base_model = base_model.to(selected_device).to(torch.float32)
    try:
        model = stack.peft_model_cls.from_pretrained(base_model, str(model_source.adapter_dir))
    except Exception as exc:
        raise TimesFmEvaluateError(
            f"Failed to load TimesFM PEFT adapter from {model_source.adapter_dir}. "
            f"Original error: {exc}"
        ) from exc
    model = model.to(selected_device).to(torch.float32)
    model.eval()

    def predict(window: TimesFmWindow) -> TimesFmWindowPrediction:
        context_tensor = torch.tensor(
            list(window.context_values),
            dtype=torch.float32,
            device=selected_device,
        )
        with torch.no_grad():
            outputs = model(
                past_values=[context_tensor],
                forecast_context_len=max(window.context_length, 256),
            )
        point_forecast = _first_batch_sequence(outputs.mean_predictions)[: window.horizon_length]
        if len(point_forecast) != window.horizon_length:
            raise TimesFmEvaluateError(
                "TimesFM returned fewer point forecasts than requested evaluation horizon"
            )
        full_predictions = _validated_full_predictions(
            _first_batch_matrix(outputs.full_predictions),
            horizon_length=window.horizon_length,
        )
        return TimesFmWindowPrediction(
            point_forecast=point_forecast,
            full_predictions=full_predictions,
        )

    return (
        predict,
        {
            "backend": "timesfm-2.5-peft-lora-evaluation",
            "selected_device": selected_device,
            "torch_version": _package_version(
                "torch",
                fallback=str(getattr(torch, "__version__", "")),
            ),
            "transformers_version": _package_version("transformers"),
            "peft_version": _package_version("peft"),
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
    )


def _load_evaluation_stack() -> _TimesFmEvaluationStack:
    torch = _import_required_module("torch", "torch")
    transformers = _import_required_module("transformers", "transformers>=5.8")
    peft = _import_required_module("peft", "peft>=0.19")
    model_cls = getattr(transformers, "TimesFm2_5ModelForPrediction", None)
    peft_model_cls = getattr(peft, "PeftModel", None)
    if model_cls is None:
        raise TimesFmEvaluateError(
            "Installed Transformers does not expose TimesFm2_5ModelForPrediction. "
            'Upgrade the optional TimesFM stack with `python -m pip install -e ".[timesfm]"`.'
        )
    if peft_model_cls is None:
        raise TimesFmEvaluateError(
            "Installed PEFT package does not expose PeftModel. "
            'Upgrade the optional TimesFM stack with `python -m pip install -e ".[timesfm]"`.'
        )
    return _TimesFmEvaluationStack(
        torch=torch,
        model_cls=model_cls,
        peft_model_cls=peft_model_cls,
    )


def _import_required_module(module_name: str, package_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or module_name
        raise TimesFmEvaluateError(
            "Could not import optional TimesFM dependency "
            f"`{package_hint}` because `{missing_name}` is missing. {_INSTALL_HINT}"
        ) from exc


def _select_device(torch: Any, requested_device: DeviceRequest) -> SelectedDevice:
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda":
        if not cuda_available:
            raise TimesFmEvaluateError(
                "CUDA was requested but PyTorch cannot access a CUDA GPU. Verify `nvidia-smi`, "
                "install a CUDA-enabled PyTorch wheel, and rerun the evaluation command."
            )
        return "cuda"
    if requested_device == "cpu":
        return "cpu"
    return "cuda" if cuda_available else "cpu"


def _evaluation_windows(
    dataset: TimesFmDataset,
    config: TimesFmEvaluationConfig,
) -> tuple[TimesFmWindow, ...]:
    windows = tuple(sorted(dataset.test_windows, key=lambda window: window.context_end_index))
    if config.max_windows is not None:
        windows = windows[-config.max_windows :]
    if not windows:
        raise TimesFmEvaluateError("TimesFM evaluation requires at least one test window")
    return windows


def _evaluate_window(
    window: TimesFmWindow,
    prediction: TimesFmWindowPrediction,
) -> TimesFmEvaluationRecord:
    actual_final = window.future_values[-1]
    context_final = window.context_values[-1]
    timesfm_final = prediction.point_forecast[-1]
    recent_mean_final = _recent_mean_return_prediction(window)
    lower: float | None = None
    upper: float | None = None
    interval_covered: bool | None = None
    if prediction.full_predictions:
        final_values = tuple(row[-1] for row in _transpose(prediction.full_predictions))
        lower = min(final_values)
        upper = max(final_values)
        interval_covered = lower <= actual_final <= upper
    return TimesFmEvaluationRecord(
        context_end=window.context_end,
        horizon_end=window.horizon_end,
        actual_final_value=actual_final,
        timesfm_final_value=timesfm_final,
        persistence_final_value=context_final,
        recent_mean_return_final_value=recent_mean_final,
        actual_return=_safe_return(actual_final, context_final),
        timesfm_return=_safe_return(timesfm_final, context_final),
        persistence_return=0.0,
        recent_mean_return=_safe_return(recent_mean_final, context_final),
        interval_lower=lower,
        interval_upper=upper,
        interval_covered=interval_covered,
    )


def _forward_forecast_window(dataset: TimesFmDataset) -> TimesFmWindow:
    latest_window = max(dataset.windows, key=lambda window: window.horizon_end_index)
    known_values = (*latest_window.context_values, *latest_window.future_values)
    if len(known_values) < dataset.context_length:
        raise TimesFmEvaluateError("TimesFM dataset cannot provide a latest forecast context")
    context_values = tuple(known_values[-dataset.context_length :])
    context_timestamps = _latest_context_timestamps(latest_window, dataset.context_length)
    if context_timestamps:
        context_start = context_timestamps[0]
        context_end = context_timestamps[-1]
    else:
        context_start = latest_window.context_start
        context_end = latest_window.horizon_end
    context_end_index = latest_window.horizon_end_index
    context_start_index = context_end_index - dataset.context_length + 1
    horizon_start = _advance_timestamp(context_end, 1)
    horizon_end = _advance_timestamp(context_end, dataset.horizon_length)
    return TimesFmWindow(
        ticker=dataset.ticker,
        split="test",
        target_field=dataset.target_field,
        context_length=dataset.context_length,
        horizon_length=dataset.horizon_length,
        context_start=context_start,
        context_end=context_end,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        context_start_index=context_start_index,
        context_end_index=context_end_index,
        horizon_start_index=context_end_index + 1,
        horizon_end_index=context_end_index + dataset.horizon_length,
        context_values=context_values,
        future_values=tuple(context_values[-1] for _ in range(dataset.horizon_length)),
        metadata={
            "target_policy": "univariate_price_forecast",
            "forecast_context": True,
            "future_values": "placeholder_latest_close_not_used_for_forward_prediction",
        },
    )


def _forward_forecast(
    window: TimesFmWindow,
    prediction: TimesFmWindowPrediction,
) -> TimesFmForwardForecast:
    point_forecast = prediction.point_forecast[: window.horizon_length]
    if len(point_forecast) != window.horizon_length:
        raise TimesFmEvaluateError(
            "TimesFM returned fewer point forecasts than requested forward horizon"
        )
    full_predictions = _validated_full_predictions(
        prediction.full_predictions,
        horizon_length=window.horizon_length,
    )
    context_final = window.context_values[-1]
    expected_return = _safe_return(point_forecast[-1], context_final)
    final_values: tuple[float, ...] = ()
    if full_predictions:
        final_values = tuple(row[-1] for row in _transpose(full_predictions))
    interval_lower = min(final_values) if final_values else None
    interval_upper = max(final_values) if final_values else None
    interval_width = (
        abs(interval_upper - interval_lower) / abs(context_final)
        if interval_lower is not None and interval_upper is not None and context_final != 0
        else None
    )
    directional_probability_proxy = (
        sum(1 for value in final_values if value > context_final) / len(final_values)
        if final_values
        else None
    )
    return TimesFmForwardForecast(
        context_start=window.context_start,
        context_end=window.context_end,
        forecast_horizon_sessions=window.horizon_length,
        point_forecast=point_forecast,
        expected_return=round(expected_return, 8),
        interval_lower=round(interval_lower, 8) if interval_lower is not None else None,
        interval_upper=round(interval_upper, 8) if interval_upper is not None else None,
        interval_width=round(interval_width, 8) if interval_width is not None else None,
        directional_probability_proxy=round(directional_probability_proxy, 6)
        if directional_probability_proxy is not None
        else None,
    )


def _latest_context_timestamps(
    window: TimesFmWindow,
    context_length: int,
) -> tuple[date | datetime, ...]:
    context_values = window.metadata.get("context_timestamps")
    future_values = window.metadata.get("future_timestamps")
    if not isinstance(context_values, list) or not isinstance(future_values, list):
        return ()
    raw_timestamps = (*context_values, *future_values)
    if len(raw_timestamps) < context_length:
        return ()
    try:
        return tuple(_parse_metadata_timestamp(str(value)) for value in raw_timestamps)[
            -context_length:
        ]
    except ValueError:
        return ()


def _parse_metadata_timestamp(value: str) -> date | datetime:
    try:
        return date.fromisoformat(value)
    except ValueError:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("TimesFM metadata datetimes must include a timezone") from None
        return parsed


def _advance_timestamp(value: date | datetime, sessions: int) -> date | datetime:
    if isinstance(value, datetime):
        return value + timedelta(days=sessions)
    return value + timedelta(days=sessions)


def _baseline_evaluations(
    records: Sequence[TimesFmEvaluationRecord],
    *,
    timesfm_metrics: TimesFmEvaluationMetrics,
) -> tuple[TimesFmBaselineEvaluation, ...]:
    actual_values = tuple(record.actual_final_value for record in records)
    actual_returns = tuple(record.actual_return for record in records)
    baselines = (
        (
            "last_close_persistence",
            tuple(record.persistence_final_value for record in records),
            tuple(record.persistence_return for record in records),
        ),
        (
            "recent_mean_return",
            tuple(record.recent_mean_return_final_value for record in records),
            tuple(record.recent_mean_return for record in records),
        ),
    )
    evaluations: list[TimesFmBaselineEvaluation] = []
    for name, predictions, predicted_returns in baselines:
        baseline_metrics = _metrics(
            predictions,
            actual_values,
            actual_returns,
            predicted_returns,
        )
        evaluations.append(
            TimesFmBaselineEvaluation(
                name=cast(TimesFmBaselineName, name),
                metrics=baseline_metrics,
                mae_delta_vs_timesfm=round(baseline_metrics.mae - timesfm_metrics.mae, 8),
                rmse_delta_vs_timesfm=round(baseline_metrics.rmse - timesfm_metrics.rmse, 8),
                directional_accuracy_delta_vs_timesfm=round(
                    baseline_metrics.directional_accuracy - timesfm_metrics.directional_accuracy,
                    8,
                ),
            )
        )
    return tuple(evaluations)


def _metrics(
    predictions: Sequence[float],
    actual_values: Sequence[float],
    actual_returns: Sequence[float],
    predicted_returns: Sequence[float],
    *,
    interval_covered: Sequence[bool | None] = (),
    interval_widths: Sequence[float | None] = (),
) -> TimesFmEvaluationMetrics:
    if not predictions:
        return TimesFmEvaluationMetrics(
            sample_count=0,
            mae=0.0,
            rmse=0.0,
            directional_accuracy=0.0,
            mean_return_error=0.0,
        )
    errors = [
        prediction - actual for prediction, actual in zip(predictions, actual_values, strict=True)
    ]
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    direction_hits = [
        int(_direction(predicted) == _direction(actual))
        for predicted, actual in zip(predicted_returns, actual_returns, strict=True)
    ]
    return_errors = [
        abs(predicted - actual)
        for predicted, actual in zip(predicted_returns, actual_returns, strict=True)
    ]
    covered = [value for value in interval_covered if value is not None]
    widths = [value for value in interval_widths if value is not None]
    interval_coverage = sum(int(value) for value in covered) / len(covered) if covered else None
    mean_interval_width = sum(widths) / len(widths) if widths else None
    calibration_proxy = (
        1.0 - abs(interval_coverage - 0.80) if interval_coverage is not None else None
    )
    return TimesFmEvaluationMetrics(
        sample_count=len(predictions),
        mae=round(sum(abs_errors) / len(abs_errors), 8),
        rmse=round(math.sqrt(sum(squared_errors) / len(squared_errors)), 8),
        directional_accuracy=round(sum(direction_hits) / len(direction_hits), 8),
        mean_return_error=round(sum(return_errors) / len(return_errors), 8),
        interval_coverage=round(interval_coverage, 8) if interval_coverage is not None else None,
        mean_interval_width=round(mean_interval_width, 8)
        if mean_interval_width is not None
        else None,
        calibration_proxy=round(max(0.0, min(1.0, calibration_proxy)), 8)
        if calibration_proxy is not None
        else None,
    )


def _suitability_reasons(
    metrics: TimesFmEvaluationMetrics,
    baselines: Sequence[TimesFmBaselineEvaluation],
    *,
    dataset: TimesFmDataset,
    config: TimesFmEvaluationConfig,
) -> list[str]:
    reasons: list[str] = []
    if metrics.sample_count < config.min_evaluation_windows:
        reasons.append("insufficient_evaluation_windows")
    if metrics.directional_accuracy < config.min_directional_accuracy:
        reasons.append("low_directional_accuracy")
    best_baseline_rmse = min(baseline.metrics.rmse for baseline in baselines)
    if best_baseline_rmse == 0.0:
        if metrics.rmse > 0.0:
            reasons.append("underperforms_best_rmse_baseline")
    elif metrics.rmse > best_baseline_rmse * config.max_rmse_ratio_vs_best_baseline:
        reasons.append("underperforms_best_rmse_baseline")
    best_baseline_directional_accuracy = max(
        baseline.metrics.directional_accuracy for baseline in baselines
    )
    required_accuracy = (
        best_baseline_directional_accuracy + config.min_directional_accuracy_delta_vs_best_baseline
    )
    if metrics.directional_accuracy < required_accuracy:
        reasons.append("underperforms_best_directional_baseline")
    if _is_stale(dataset, config):
        reasons.append("stale_evaluation_data")
    return reasons


def _is_stale(dataset: TimesFmDataset, config: TimesFmEvaluationConfig) -> bool:
    if config.suitability_max_latest_bar_age_days is None or config.as_of is None:
        return False
    latest = _latest_bar_timestamp(dataset)
    as_of = config.as_of
    latest_date = latest.date() if isinstance(latest, datetime) else latest
    as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
    return (as_of_date - latest_date).days > config.suitability_max_latest_bar_age_days


def _recent_mean_return_prediction(window: TimesFmWindow) -> float:
    returns: list[float] = []
    for previous, current in zip(window.context_values, window.context_values[1:], strict=False):
        returns.append(_safe_return(current, previous))
    mean_return = sum(returns) / len(returns) if returns else 0.0
    return window.context_values[-1] * ((1.0 + mean_return) ** window.horizon_length)


def _safe_return(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value / baseline) - 1.0


def _direction(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _interval_width(record: TimesFmEvaluationRecord) -> float | None:
    if record.interval_lower is None or record.interval_upper is None:
        return None
    denominator = abs(record.persistence_final_value)
    if denominator == 0:
        return None
    return abs(record.interval_upper - record.interval_lower) / denominator


def _latest_bar_timestamp(dataset: TimesFmDataset) -> date | datetime:
    return max(dataset.windows, key=lambda window: window.horizon_end_index).horizon_end


def _first_batch_sequence(value: Any) -> tuple[float, ...]:
    payload = _to_python(value)
    if not isinstance(payload, list) or not payload:
        raise TimesFmEvaluateError("TimesFM point forecast output has an unexpected shape")
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


def _validated_full_predictions(
    full_predictions: tuple[tuple[float, ...], ...],
    *,
    horizon_length: int,
) -> tuple[tuple[float, ...], ...]:
    if not full_predictions:
        return ()
    if len(full_predictions) < horizon_length:
        raise TimesFmEvaluateError(
            "TimesFM full prediction horizon did not match the requested evaluation horizon"
        )
    sliced_predictions = full_predictions[:horizon_length]
    width = len(sliced_predictions[0])
    if width == 0:
        return ()
    if any(len(row) != width for row in sliced_predictions):
        raise TimesFmEvaluateError("TimesFM full prediction rows have inconsistent widths")
    return sliced_predictions


def _transpose(rows: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if not rows:
        return ()
    width = min(len(row) for row in rows)
    return tuple(tuple(row[index] for row in rows) for index in range(width))


def _finite_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, int | float):
        raise TimesFmEvaluateError(f"{field_name} output must be numeric")
    float_value = float(value)
    if not math.isfinite(float_value):
        raise TimesFmEvaluateError(f"{field_name} output must be finite")
    return float_value


def _model_hash(
    *,
    model_id: str,
    model_revision: str | None,
    adapter_sha256: str,
    training_metadata_sha256: str,
) -> str:
    payload = {
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter_sha256": adapter_sha256,
        "training_metadata_sha256": training_metadata_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        raise ValueError("synthetic TimesFM evaluation smoke requires at least 40 bars")
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


def _package_version(distribution: str, *, fallback: str = "unknown") -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EVALUATION_OUTPUT_PATH",
    "TimesFmBaselineEvaluation",
    "TimesFmBaselineName",
    "TimesFmEvaluateError",
    "TimesFmEvaluationArtifact",
    "TimesFmEvaluationConfig",
    "TimesFmEvaluationDataSource",
    "TimesFmEvaluationMetrics",
    "TimesFmEvaluationModelSource",
    "TimesFmEvaluationRecord",
    "TimesFmEvaluationSourceKind",
    "TimesFmEvaluationStatus",
    "TimesFmForwardForecast",
    "TimesFmWindowPrediction",
    "evaluate_timesfm_dataset",
    "load_timesfm_evaluation_model_source",
    "write_timesfm_evaluation_artifact",
]
