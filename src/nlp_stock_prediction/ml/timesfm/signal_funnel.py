"""Staged TimesFM signal funnel orchestration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from nlp_stock_prediction.analysis.technical import analyze_technical_snapshot
from nlp_stock_prediction.contracts import MarketSnapshot
from nlp_stock_prediction.contracts.base import ContractModel, NonEmptyStr, TickerSymbol
from nlp_stock_prediction.ml.timesfm.dataset import (
    TimesFmDataset,
    TimesFmDatasetConfig,
    TimesFmWindow,
    build_timesfm_dataset,
)
from nlp_stock_prediction.ml.timesfm.focused_hpo import (
    DEFAULT_AS_OF,
    DEFAULT_SYMBOLS,
    TickerPolicy,
    default_ticker_policies,
    dry_run_data_record,
    ensure_symbol_data,
    write_manifest,
)
from nlp_stock_prediction.ml.timesfm.smoke import DEFAULT_MODEL_ID
from nlp_stock_prediction.ml.train import load_price_bars_csv

DEFAULT_OUTPUT_ROOT = Path("artifacts/ml/timesfm-funnel")
DEFAULT_DATA_DIR = Path("data/ml/wsb_10y")
IMPLEMENTED_STAGES = ("data_check", "baseline_screen", "raw_timesfm_screen", "adapter_smoke")
FUNNEL_STAGES = (
    "data_check",
    "baseline_screen",
    "raw_timesfm_screen",
    "adapter_smoke",
    "survivor_hpo",
    "final_eval",
    "report_ready",
)

FunnelStage = Literal[
    "data_check",
    "baseline_screen",
    "raw_timesfm_screen",
    "adapter_smoke",
    "survivor_hpo",
    "final_eval",
    "report_ready",
]
FunnelStatus = Literal[
    "passed",
    "killed",
    "research_only",
    "borderline",
    "weak",
    "suitable",
    "failed",
    "skipped",
]
FunnelDecision = Literal[
    "continue",
    "stop",
    "run_adapter_smoke",
    "run_hpo",
    "promote_for_scoring",
    "audit_only",
]
FunnelProfile = Literal["quick", "walkaway", "full"]
DeviceRequest = Literal["auto", "cpu", "cuda"]


class SignalFunnelLeaderboardRow(ContractModel):
    """One ticker/stage/method result in the signal funnel leaderboard."""

    run_id: NonEmptyStr
    created_at: str
    as_of: str
    symbol: TickerSymbol
    stage: FunnelStage
    method: NonEmptyStr
    status: FunnelStatus
    kill_reason: str | None = None
    decision: FunnelDecision
    asset_type: NonEmptyStr
    history_start: str | None = None
    latest_bar: str | None = None
    bar_count: int | None = Field(default=None, ge=0)
    train_windows: int | None = Field(default=None, ge=0)
    validation_windows: int | None = Field(default=None, ge=0)
    test_windows: int | None = Field(default=None, ge=0)
    context_length: int = Field(ge=2)
    horizon_length: int = Field(ge=1)
    max_windows: int | None = Field(default=None, ge=1)
    runtime_seconds: float = Field(ge=0.0)
    device: DeviceRequest
    model_id: str | None = None
    model_revision: str | None = None
    adapter_sha256: str | None = None
    dataset_hash: str | None = None
    evaluation_artifact: str | None = None
    training_metadata: str | None = None
    rmse: float | None = None
    best_baseline_rmse: float | None = None
    rmse_ratio_vs_best_baseline: float | None = None
    directional_accuracy: float | None = None
    best_baseline_directional_accuracy: float | None = None
    directional_delta_vs_best_baseline: float | None = None
    raw_timesfm_rmse: float | None = None
    adapter_rmse_ratio_vs_raw: float | None = None
    adapter_directional_delta_vs_raw: float | None = None
    validation_mean_loss: float | None = None
    interval_coverage: float | None = None
    mean_interval_width: float | None = None
    calibration_proxy: float | None = None
    selected_for_next_stage: bool = False
    promoted_for_scoring: bool = False
    notes: str | None = None


class _WindowCounts(ContractModel):
    train: int = Field(ge=0)
    validation: int = Field(ge=0)
    test: int = Field(ge=0)
    raw: int = Field(ge=0)
    purged: int = Field(ge=0)


class _BaselineMetrics(ContractModel):
    name: NonEmptyStr
    rmse: float = Field(ge=0.0)
    directional_accuracy: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)


class _RawTimesFmPredictionBatch(ContractModel):
    model_id: str | None = None
    model_revision: str | None = None
    point_forecasts: tuple[tuple[float, ...], ...]
    full_predictions: tuple[tuple[tuple[float, ...], ...], ...] = Field(default_factory=tuple)
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)


class _RawTimesFmRecord(ContractModel):
    context_end: str
    horizon_end: str
    actual_final_value: float
    timesfm_final_value: float
    actual_return: float
    timesfm_return: float
    interval_lower: float | None = None
    interval_upper: float | None = None
    interval_covered: bool | None = None
    interval_width: float | None = None


class _RawTimesFmMetrics(ContractModel):
    sample_count: int = Field(ge=0)
    rmse: float = Field(ge=0.0)
    directional_accuracy: float = Field(ge=0.0, le=1.0)
    interval_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_interval_width: float | None = Field(default=None, ge=0.0)
    calibration_proxy: float | None = Field(default=None, ge=0.0, le=1.0)


RawTimesFmPredictor = Callable[
    [TimesFmDataset, Sequence[TimesFmWindow], argparse.Namespace],
    _RawTimesFmPredictionBatch,
]


class _AdapterSmokePredictionBatch(ContractModel):
    model_id: str | None = None
    model_revision: str | None = None
    adapter_sha256: str | None = None
    training_metadata_path: str | None = None
    validation_mean_loss: float | None = Field(default=None, ge=0.0)
    point_forecasts: tuple[tuple[float, ...], ...]
    full_predictions: tuple[tuple[tuple[float, ...], ...], ...] = Field(default_factory=tuple)
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)


AdapterSmokeRunner = Callable[
    [TimesFmDataset, Sequence[TimesFmWindow], Path, Path, argparse.Namespace],
    _AdapterSmokePredictionBatch,
]


def run_signal_funnel(
    args: argparse.Namespace,
    *,
    raw_predictor: RawTimesFmPredictor | None = None,
    adapter_smoke_runner: AdapterSmokeRunner | None = None,
) -> int:
    """Run the implemented signal-funnel stages and write manifest/leaderboard artifacts."""

    if args.stop_after is None:
        args.stop_after = _default_stop_after_for_profile(args.profile)
    symbols = _parse_symbols(args.symbols)
    as_of = date.fromisoformat(args.as_of)
    start = as_of - timedelta(days=365 * args.years)
    output_root = Path(args.output_root)
    data_dir = Path(args.data_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        data_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or _default_run_id(as_of)
    created_at = datetime.now(UTC).isoformat()
    policies = default_ticker_policies()
    rows: list[SignalFunnelLeaderboardRow] = []
    manifest: dict[str, Any] = {
        "schema_version": "ml.timesfm.signal_funnel_manifest.v1",
        "run_id": run_id,
        "created_at": created_at,
        "symbols": list(symbols),
        "as_of": as_of.isoformat(),
        "years": args.years,
        "profile": args.profile,
        "device": args.device,
        "dry_run": bool(args.dry_run),
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "implemented_stages": list(IMPLEMENTED_STAGES),
        "requested_stop_after": args.stop_after,
        "records": [],
        "skipped_stages": [
            {
                "stage": stage,
                "status": "skipped",
                "reason": "stage_not_implemented",
            }
            for stage in FUNNEL_STAGES
            if stage not in IMPLEMENTED_STAGES
        ],
    }

    for symbol in symbols:
        policy = policies.get(symbol, TickerPolicy(symbol=symbol))
        print(f"{symbol}: data_check")
        row = _run_data_check(
            symbol,
            policy,
            args=args,
            data_dir=data_dir,
            start=start,
            as_of=as_of,
            run_id=run_id,
            created_at=created_at,
        )
        rows.append(row)
        manifest["records"].append(row.model_dump(mode="json"))
        _write_signal_funnel_outputs(output_root, manifest, rows)
        if _should_stop_after("data_check", args.stop_after):
            continue
        print(f"{symbol}: baseline_screen")
        baseline_rows = _run_baseline_screen(
            symbol,
            policy,
            data_check_row=row,
            args=args,
            data_dir=data_dir,
            as_of=as_of,
            run_id=run_id,
            created_at=created_at,
        )
        rows.extend(baseline_rows)
        manifest["records"].extend(row.model_dump(mode="json") for row in baseline_rows)
        _write_signal_funnel_outputs(output_root, manifest, rows)
        if _should_stop_after("baseline_screen", args.stop_after):
            continue
        print(f"{symbol}: raw_timesfm_screen")
        raw_row = _run_raw_timesfm_screen(
            symbol,
            policy,
            data_check_row=row,
            baseline_rows=baseline_rows,
            args=args,
            data_dir=data_dir,
            output_root=output_root,
            as_of=as_of,
            run_id=run_id,
            created_at=created_at,
            raw_predictor=raw_predictor,
        )
        rows.append(raw_row)
        manifest["records"].append(raw_row.model_dump(mode="json"))
        _write_signal_funnel_outputs(output_root, manifest, rows)
        if _should_stop_after("raw_timesfm_screen", args.stop_after):
            continue
        print(f"{symbol}: adapter_smoke")
        adapter_row = _run_adapter_smoke(
            symbol,
            policy,
            data_check_row=row,
            raw_row=raw_row,
            args=args,
            data_dir=data_dir,
            output_root=output_root,
            as_of=as_of,
            run_id=run_id,
            created_at=created_at,
            adapter_smoke_runner=adapter_smoke_runner,
        )
        rows.append(adapter_row)
        manifest["records"].append(adapter_row.model_dump(mode="json"))
        _write_signal_funnel_outputs(output_root, manifest, rows)

    return 0 if all(row.status != "failed" for row in rows) else 1


def _run_data_check(
    symbol: str,
    policy: TickerPolicy,
    *,
    args: argparse.Namespace,
    data_dir: Path,
    start: date,
    as_of: date,
    run_id: str,
    created_at: str,
) -> SignalFunnelLeaderboardRow:
    started_at = time.perf_counter()
    history_start = max(start, policy.history_start_override or start).isoformat()
    latest_bar: str | None = None
    bar_count: int | None = None
    try:
        data_record = (
            dry_run_data_record(symbol, policy, data_dir=data_dir, start=start, as_of=as_of)
            if args.dry_run
            else ensure_symbol_data(
                symbol,
                policy,
                data_dir=data_dir,
                start=start,
                as_of=as_of,
                refresh=args.refresh_data,
                sleep_seconds=args.sleep_seconds,
            )
        )
        latest_bar = str(data_record.get("last")) if data_record.get("last") is not None else None
        bar_count = int(data_record["rows"])
        if args.dry_run:
            dataset_hash = None
            counts = _estimate_window_counts(
                row_count=bar_count,
                context_length=args.context_length,
                horizon_length=args.horizon_length,
            )
        else:
            bars = load_price_bars_csv(Path(str(data_record["csv_path"])), ticker=symbol)
            dataset = build_timesfm_dataset(
                symbol,
                bars,
                config=TimesFmDatasetConfig(
                    context_length=args.context_length,
                    horizon_length=args.horizon_length,
                    target_field=args.target_field,
                    as_of=as_of,
                    max_latest_bar_age_days=args.max_latest_bar_age_days,
                ),
            )
            dataset_hash = dataset.dataset_hash
            split_counts = cast(dict[str, Any], dataset.metadata["split_counts"])
            counts = _WindowCounts(
                train=int(split_counts["train"]),
                validation=int(split_counts["validation"]),
                test=int(split_counts["test"]),
                raw=int(cast(Any, dataset.metadata["raw_window_count"])),
                purged=int(cast(Any, dataset.metadata["purged_window_count"])),
            )
        status, decision, kill_reason = _stage0_decision(
            counts,
            min_evaluation_windows=args.min_evaluation_windows,
        )
        return _leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            history_start=history_start,
            latest_bar=latest_bar,
            bar_count=bar_count,
            counts=counts,
            status=status,
            decision=decision,
            kill_reason=kill_reason,
            dataset_hash=dataset_hash,
            runtime_seconds=_elapsed_seconds(started_at),
            notes=_stage0_notes(data_record, counts),
        )
    except Exception as exc:
        return _leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            history_start=history_start,
            latest_bar=latest_bar,
            bar_count=bar_count,
            counts=None,
            status="failed",
            decision="stop",
            kill_reason=str(exc),
            dataset_hash=None,
            runtime_seconds=_elapsed_seconds(started_at),
            notes="data_check failed before later funnel stages could run",
        )


def _run_baseline_screen(
    symbol: str,
    policy: TickerPolicy,
    *,
    data_check_row: SignalFunnelLeaderboardRow,
    args: argparse.Namespace,
    data_dir: Path,
    as_of: date,
    run_id: str,
    created_at: str,
) -> tuple[SignalFunnelLeaderboardRow, ...]:
    started_at = time.perf_counter()
    if data_check_row.status == "failed":
        return (
            _baseline_leaderboard_row(
                run_id=run_id,
                created_at=created_at,
                as_of=as_of,
                symbol=symbol,
                policy=policy,
                args=args,
                data_check_row=data_check_row,
                method="baseline_screen",
                status="skipped",
                decision="stop",
                kill_reason="data_check_failed",
                runtime_seconds=_elapsed_seconds(started_at),
                notes="baseline_screen skipped because data_check failed",
            ),
        )
    if args.dry_run:
        return (
            _baseline_leaderboard_row(
                run_id=run_id,
                created_at=created_at,
                as_of=as_of,
                symbol=symbol,
                policy=policy,
                args=args,
                data_check_row=data_check_row,
                method="baseline_screen",
                status="skipped",
                decision="audit_only",
                kill_reason="dry_run_no_csv_loaded",
                runtime_seconds=_elapsed_seconds(started_at),
                notes="baseline_screen requires real OHLCV rows and is skipped during dry-run",
            ),
        )

    try:
        bars = load_price_bars_csv(data_dir / f"{symbol}.csv", ticker=symbol)
        dataset = build_timesfm_dataset(
            symbol,
            bars,
            config=TimesFmDatasetConfig(
                context_length=args.context_length,
                horizon_length=args.horizon_length,
                target_field=args.target_field,
                as_of=as_of,
                max_latest_bar_age_days=args.max_latest_bar_age_days,
            ),
        )
        windows = _screen_windows(dataset, max_windows=args.screen_max_windows)
        baseline_metrics = _baseline_metrics(windows)
        best_rmse = min(metric.rmse for metric in baseline_metrics)
        best_directional_accuracy = max(metric.directional_accuracy for metric in baseline_metrics)
        status: FunnelStatus = (
            "research_only" if data_check_row.status == "research_only" else "passed"
        )
        decision: FunnelDecision = (
            "audit_only" if data_check_row.status == "research_only" else "continue"
        )
        selected_for_next_stage = data_check_row.status == "passed"
        runtime_seconds = _elapsed_seconds(started_at)
        rows: list[SignalFunnelLeaderboardRow] = [
            _baseline_leaderboard_row(
                run_id=run_id,
                created_at=created_at,
                as_of=as_of,
                symbol=symbol,
                policy=policy,
                args=args,
                data_check_row=data_check_row,
                method=metric.name,
                status=status,
                decision=decision,
                kill_reason=data_check_row.kill_reason if status == "research_only" else None,
                runtime_seconds=runtime_seconds,
                rmse=metric.rmse,
                best_baseline_rmse=best_rmse,
                rmse_ratio_vs_best_baseline=_safe_ratio(metric.rmse, best_rmse),
                directional_accuracy=metric.directional_accuracy,
                best_baseline_directional_accuracy=best_directional_accuracy,
                directional_delta_vs_best_baseline=round(
                    metric.directional_accuracy - best_directional_accuracy,
                    8,
                ),
                selected_for_next_stage=selected_for_next_stage,
                notes=f"sample_count={metric.sample_count}",
            )
            for metric in baseline_metrics
        ]
        technical = analyze_technical_snapshot(
            MarketSnapshot(ticker=symbol, bars=tuple(bars)),
            as_of=as_of,
        )
        rows.append(
            _baseline_leaderboard_row(
                run_id=run_id,
                created_at=created_at,
                as_of=as_of,
                symbol=symbol,
                policy=policy,
                args=args,
                data_check_row=data_check_row,
                method="deterministic_technical_analysis",
                status=status,
                decision=decision,
                kill_reason=data_check_row.kill_reason if status == "research_only" else None,
                runtime_seconds=runtime_seconds,
                selected_for_next_stage=selected_for_next_stage,
                notes=(
                    f"signal={technical.signal.value}; trend={technical.trend}; "
                    f"confidence={technical.confidence:.4f}; summary={technical.summary}"
                ),
            )
        )
        return tuple(rows)
    except Exception as exc:
        return (
            _baseline_leaderboard_row(
                run_id=run_id,
                created_at=created_at,
                as_of=as_of,
                symbol=symbol,
                policy=policy,
                args=args,
                data_check_row=data_check_row,
                method="baseline_screen",
                status="failed",
                decision="stop",
                kill_reason=str(exc),
                runtime_seconds=_elapsed_seconds(started_at),
                notes="baseline_screen failed before raw TimesFM screening could run",
            ),
        )


def _run_raw_timesfm_screen(
    symbol: str,
    policy: TickerPolicy,
    *,
    data_check_row: SignalFunnelLeaderboardRow,
    baseline_rows: Sequence[SignalFunnelLeaderboardRow],
    args: argparse.Namespace,
    data_dir: Path,
    output_root: Path,
    as_of: date,
    run_id: str,
    created_at: str,
    raw_predictor: RawTimesFmPredictor | None,
) -> SignalFunnelLeaderboardRow:
    started_at = time.perf_counter()
    if data_check_row.status == "failed":
        return _raw_timesfm_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status="skipped",
            decision="stop",
            kill_reason="data_check_failed",
            runtime_seconds=_elapsed_seconds(started_at),
            notes="raw_timesfm_screen skipped because data_check failed",
        )
    if args.dry_run:
        return _raw_timesfm_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status="skipped",
            decision="audit_only",
            kill_reason="dry_run_no_model_loaded",
            runtime_seconds=_elapsed_seconds(started_at),
            notes="raw_timesfm_screen requires real OHLCV rows and a TimesFM model",
        )
    if not any(
        row.stage == "baseline_screen" and row.status == "passed" and row.selected_for_next_stage
        for row in baseline_rows
    ):
        return _raw_timesfm_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status="skipped",
            decision="audit_only",
            kill_reason="baseline_screen_not_actionable",
            runtime_seconds=_elapsed_seconds(started_at),
            notes="raw_timesfm_screen skipped because baseline_screen did not select the ticker",
        )

    try:
        bars = load_price_bars_csv(data_dir / f"{symbol}.csv", ticker=symbol)
        dataset = build_timesfm_dataset(
            symbol,
            bars,
            config=TimesFmDatasetConfig(
                context_length=args.context_length,
                horizon_length=args.horizon_length,
                target_field=args.target_field,
                as_of=as_of,
                max_latest_bar_age_days=args.max_latest_bar_age_days,
            ),
        )
        windows = _screen_windows(dataset, max_windows=args.screen_max_windows)
        predictor = raw_predictor or _predict_raw_timesfm
        prediction_batch = predictor(dataset, windows, args)
        records = _raw_timesfm_records(windows, prediction_batch)
        metrics = _raw_timesfm_metrics(records)
        baseline_metrics = _baseline_metrics(windows)
        best_rmse = min(metric.rmse for metric in baseline_metrics)
        best_directional_accuracy = max(metric.directional_accuracy for metric in baseline_metrics)
        rmse_ratio = _safe_ratio(metrics.rmse, best_rmse)
        directional_delta = round(metrics.directional_accuracy - best_directional_accuracy, 8)
        status, decision, kill_reason, selected_for_next_stage = _raw_stage_decision(
            sample_count=metrics.sample_count,
            rmse_ratio=rmse_ratio,
            directional_delta=directional_delta,
            args=args,
        )
        artifact_path = _write_raw_timesfm_artifact(
            output_root,
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            dataset=dataset,
            status=status,
            decision=decision,
            kill_reason=kill_reason,
            prediction_batch=prediction_batch,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            rmse_ratio=rmse_ratio,
            directional_delta=directional_delta,
            records=records,
            args=args,
        )
        return _raw_timesfm_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status=status,
            decision=decision,
            kill_reason=kill_reason,
            runtime_seconds=_elapsed_seconds(started_at),
            model_id=prediction_batch.model_id or args.model_id,
            model_revision=prediction_batch.model_revision or args.model_revision,
            evaluation_artifact=str(artifact_path),
            rmse=metrics.rmse,
            best_baseline_rmse=best_rmse,
            rmse_ratio_vs_best_baseline=rmse_ratio,
            directional_accuracy=metrics.directional_accuracy,
            best_baseline_directional_accuracy=best_directional_accuracy,
            directional_delta_vs_best_baseline=directional_delta,
            raw_timesfm_rmse=metrics.rmse,
            interval_coverage=metrics.interval_coverage,
            mean_interval_width=metrics.mean_interval_width,
            calibration_proxy=metrics.calibration_proxy,
            selected_for_next_stage=selected_for_next_stage,
            notes=(
                f"sample_count={metrics.sample_count}; "
                f"backend={prediction_batch.runtime_metadata.get('backend', 'unknown')}"
            ),
        )
    except Exception as exc:
        return _raw_timesfm_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status="failed",
            decision="stop",
            kill_reason=str(exc),
            runtime_seconds=_elapsed_seconds(started_at),
            notes="raw_timesfm_screen failed before adapter smoke could run",
        )


def _predict_raw_timesfm(
    dataset: TimesFmDataset,
    windows: Sequence[TimesFmWindow],
    args: argparse.Namespace,
) -> _RawTimesFmPredictionBatch:
    from nlp_stock_prediction.ml.timesfm.adapter import (
        TimesFmForecastConfig,
        _first_batch_matrix,
        _first_batch_sequence,
        _load_inference_stack,
        _load_model,
        _model_revision,
        _select_device,
        _validated_full_predictions,
    )

    config = TimesFmForecastConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        requested_device=args.device,
    )
    stack = _load_inference_stack()
    torch = stack.torch
    selected_device = _select_device(torch, config.requested_device)
    loaded_model = _load_model(stack, config)
    model_revision = _model_revision(loaded_model, config)
    loaded_model = loaded_model.to(selected_device).to(torch.float32)
    loaded_model.eval()

    point_forecasts: list[tuple[float, ...]] = []
    full_predictions: list[tuple[tuple[float, ...], ...]] = []
    with torch.no_grad():
        for window in windows:
            context_tensor = torch.tensor(
                list(window.context_values),
                dtype=torch.float32,
                device=selected_device,
            )
            outputs = loaded_model(
                past_values=[context_tensor],
                forecast_context_len=max(window.context_length, 256),
            )
            point_forecast = tuple(
                _first_batch_sequence(outputs.mean_predictions)[: window.horizon_length]
            )
            if len(point_forecast) != window.horizon_length:
                raise ValueError("TimesFM returned fewer point forecasts than requested horizon")
            full_prediction = _validated_full_predictions(
                _first_batch_matrix(outputs.full_predictions),
                horizon_length=window.horizon_length,
            )
            point_forecasts.append(point_forecast)
            full_predictions.append(full_prediction)

    return _RawTimesFmPredictionBatch(
        model_id=config.model_id,
        model_revision=model_revision,
        point_forecasts=tuple(point_forecasts),
        full_predictions=tuple(full_predictions),
        runtime_metadata={
            "backend": "timesfm_transformers_local",
            "selected_device": selected_device,
        },
    )


def _raw_timesfm_records(
    windows: Sequence[TimesFmWindow],
    prediction_batch: _RawTimesFmPredictionBatch,
) -> tuple[_RawTimesFmRecord, ...]:
    return _forecast_records(
        windows,
        point_forecasts=prediction_batch.point_forecasts,
        full_predictions=prediction_batch.full_predictions,
        label="raw TimesFM",
    )


def _forecast_records(
    windows: Sequence[TimesFmWindow],
    *,
    point_forecasts: Sequence[Sequence[float]],
    full_predictions: Sequence[Sequence[Sequence[float]]] = (),
    label: str,
) -> tuple[_RawTimesFmRecord, ...]:
    if len(point_forecasts) != len(windows):
        raise ValueError(
            f"{label} prediction count does not match validation-window count: "
            f"{len(point_forecasts)}!={len(windows)}"
        )
    aligned_full_predictions = _aligned_full_predictions(
        full_predictions,
        window_count=len(windows),
        label=label,
    )
    records: list[_RawTimesFmRecord] = []
    for window, point_forecast, full_prediction in zip(
        windows,
        point_forecasts,
        aligned_full_predictions,
        strict=True,
    ):
        if len(point_forecast) != window.horizon_length:
            raise ValueError(
                f"{label} point forecast length does not match horizon length: "
                f"{len(point_forecast)}!={window.horizon_length}"
            )
        actual_final = window.future_values[-1]
        context_final = window.context_values[-1]
        timesfm_final = point_forecast[-1]
        lower: float | None = None
        upper: float | None = None
        interval_covered: bool | None = None
        interval_width: float | None = None
        if full_prediction:
            final_values = tuple(row[-1] for row in _transpose(full_prediction))
            lower = min(final_values)
            upper = max(final_values)
            interval_covered = lower <= actual_final <= upper
            interval_width = upper - lower
        records.append(
            _RawTimesFmRecord(
                context_end=str(window.context_end),
                horizon_end=str(window.horizon_end),
                actual_final_value=actual_final,
                timesfm_final_value=timesfm_final,
                actual_return=_safe_return(actual_final, context_final),
                timesfm_return=_safe_return(timesfm_final, context_final),
                interval_lower=lower,
                interval_upper=upper,
                interval_covered=interval_covered,
                interval_width=interval_width,
            )
        )
    return tuple(records)


def _aligned_full_predictions(
    full_predictions: Sequence[Sequence[Sequence[float]]],
    *,
    window_count: int,
    label: str,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    if not full_predictions:
        return tuple(() for _ in range(window_count))
    if len(full_predictions) != window_count:
        raise ValueError(
            f"{label} full-prediction count does not match validation-window count: "
            f"{len(full_predictions)}!={window_count}"
        )
    return tuple(
        tuple(tuple(row) for row in window_predictions) for window_predictions in full_predictions
    )


def _raw_timesfm_metrics(records: Sequence[_RawTimesFmRecord]) -> _RawTimesFmMetrics:
    if not records:
        return _RawTimesFmMetrics(
            sample_count=0,
            rmse=0.0,
            directional_accuracy=0.0,
        )
    squared_errors = [
        (record.timesfm_final_value - record.actual_final_value) ** 2 for record in records
    ]
    direction_hits = [
        int(_direction(record.timesfm_return) == _direction(record.actual_return))
        for record in records
    ]
    covered = [record.interval_covered for record in records if record.interval_covered is not None]
    widths = [record.interval_width for record in records if record.interval_width is not None]
    interval_coverage = sum(int(value) for value in covered) / len(covered) if covered else None
    mean_interval_width = sum(widths) / len(widths) if widths else None
    calibration_proxy = (
        1.0 - abs(interval_coverage - 0.80) if interval_coverage is not None else None
    )
    return _RawTimesFmMetrics(
        sample_count=len(records),
        rmse=round(math.sqrt(sum(squared_errors) / len(squared_errors)), 8),
        directional_accuracy=round(sum(direction_hits) / len(direction_hits), 8),
        interval_coverage=round(interval_coverage, 8) if interval_coverage is not None else None,
        mean_interval_width=round(mean_interval_width, 8)
        if mean_interval_width is not None
        else None,
        calibration_proxy=round(max(0.0, min(1.0, calibration_proxy)), 8)
        if calibration_proxy is not None
        else None,
    )


def _raw_stage_decision(
    *,
    sample_count: int,
    rmse_ratio: float | None,
    directional_delta: float,
    args: argparse.Namespace,
) -> tuple[FunnelStatus, FunnelDecision, str | None, bool]:
    if sample_count < args.min_evaluation_windows:
        return (
            "research_only",
            "audit_only",
            (
                "limited_validation_windows:"
                f"{sample_count}<min_evaluation_windows:{args.min_evaluation_windows}"
            ),
            False,
        )
    if rmse_ratio is None:
        return (
            "borderline",
            "audit_only",
            "raw_timesfm_rmse_ratio_unavailable",
            False,
        )
    if (
        rmse_ratio > args.raw_rmse_kill_threshold
        and directional_delta < args.raw_directional_kill_threshold
    ):
        return (
            "killed",
            "stop",
            (
                "raw_timesfm_underperformed_baselines:"
                f"rmse_ratio={rmse_ratio:.4f}>{args.raw_rmse_kill_threshold:.4f};"
                f"directional_delta={directional_delta:.4f}<"
                f"{args.raw_directional_kill_threshold:.4f}"
            ),
            False,
        )
    if (
        rmse_ratio <= args.raw_rmse_promote_threshold
        or directional_delta >= args.raw_directional_promote_threshold
    ):
        return "passed", "run_adapter_smoke", None, True
    if (
        rmse_ratio <= args.raw_rmse_research_threshold
        or directional_delta >= args.raw_directional_research_threshold
    ):
        return (
            "research_only",
            "audit_only",
            (
                "raw_timesfm_close_but_not_promoted:"
                f"rmse_ratio={rmse_ratio:.4f};directional_delta={directional_delta:.4f}"
            ),
            False,
        )
    return (
        "borderline",
        "audit_only",
        (
            "raw_timesfm_not_promising:"
            f"rmse_ratio={rmse_ratio:.4f};directional_delta={directional_delta:.4f}"
        ),
        False,
    )


def _write_raw_timesfm_artifact(
    output_root: Path,
    *,
    run_id: str,
    created_at: str,
    as_of: date,
    symbol: str,
    dataset: TimesFmDataset,
    status: FunnelStatus,
    decision: FunnelDecision,
    kill_reason: str | None,
    prediction_batch: _RawTimesFmPredictionBatch,
    metrics: _RawTimesFmMetrics,
    baseline_metrics: Sequence[_BaselineMetrics],
    rmse_ratio: float | None,
    directional_delta: float,
    records: Sequence[_RawTimesFmRecord],
    args: argparse.Namespace,
) -> Path:
    artifact_path = output_root / "raw_timesfm_screen" / f"{symbol}.evaluation.json"
    payload = {
        "schema_version": "ml.timesfm.raw_evaluation.v1",
        "run_id": run_id,
        "created_at": created_at,
        "as_of": as_of.isoformat(),
        "symbol": symbol,
        "stage": "raw_timesfm_screen",
        "method": "raw_timesfm_base",
        "status": status,
        "decision": decision,
        "kill_reason": kill_reason,
        "model_id": prediction_batch.model_id or args.model_id,
        "model_revision": prediction_batch.model_revision or args.model_revision,
        "dataset_hash": dataset.dataset_hash,
        "target_field": dataset.target_field,
        "context_length": dataset.context_length,
        "horizon_length": dataset.horizon_length,
        "max_windows": args.screen_max_windows,
        "metrics": metrics.model_dump(mode="json"),
        "baselines": [metric.model_dump(mode="json") for metric in baseline_metrics],
        "rmse_ratio_vs_best_baseline": rmse_ratio,
        "directional_delta_vs_best_baseline": directional_delta,
        "records": [record.model_dump(mode="json") for record in records],
        "runtime_metadata": prediction_batch.runtime_metadata,
    }
    write_manifest(artifact_path, payload)
    return artifact_path


def _run_adapter_smoke(
    symbol: str,
    policy: TickerPolicy,
    *,
    data_check_row: SignalFunnelLeaderboardRow,
    raw_row: SignalFunnelLeaderboardRow,
    args: argparse.Namespace,
    data_dir: Path,
    output_root: Path,
    as_of: date,
    run_id: str,
    created_at: str,
    adapter_smoke_runner: AdapterSmokeRunner | None,
) -> SignalFunnelLeaderboardRow:
    started_at = time.perf_counter()
    if data_check_row.status == "failed":
        return _adapter_smoke_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status="skipped",
            decision="stop",
            kill_reason="data_check_failed",
            runtime_seconds=_elapsed_seconds(started_at),
            notes="adapter_smoke skipped because data_check failed",
        )
    if args.dry_run:
        return _adapter_smoke_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status="skipped",
            decision="audit_only",
            kill_reason="dry_run_no_adapter_training",
            runtime_seconds=_elapsed_seconds(started_at),
            notes="adapter_smoke requires real OHLCV rows and a TimesFM training stack",
        )
    if (
        raw_row.stage != "raw_timesfm_screen"
        or raw_row.status != "passed"
        or raw_row.decision != "run_adapter_smoke"
        or not raw_row.selected_for_next_stage
    ):
        return _adapter_smoke_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status="skipped",
            decision="audit_only" if raw_row.status != "killed" else "stop",
            kill_reason="raw_timesfm_not_selected",
            runtime_seconds=_elapsed_seconds(started_at),
            raw_timesfm_rmse=raw_row.raw_timesfm_rmse,
            notes=f"adapter_smoke skipped because raw_timesfm_screen status={raw_row.status}",
        )

    reusable_row = _load_reusable_adapter_smoke_row(
        output_root,
        run_id=run_id,
        created_at=created_at,
        as_of=as_of,
        symbol=symbol,
        policy=policy,
        args=args,
        data_check_row=data_check_row,
        raw_row=raw_row,
        runtime_seconds=_elapsed_seconds(started_at),
    )
    if reusable_row is not None:
        return reusable_row

    try:
        csv_path = data_dir / f"{symbol}.csv"
        bars = load_price_bars_csv(csv_path, ticker=symbol)
        dataset = build_timesfm_dataset(
            symbol,
            bars,
            config=TimesFmDatasetConfig(
                context_length=args.context_length,
                horizon_length=args.horizon_length,
                target_field=args.target_field,
                as_of=as_of,
                max_latest_bar_age_days=args.max_latest_bar_age_days,
            ),
        )
        windows = _screen_windows(dataset, max_windows=args.screen_max_windows)
        runner = adapter_smoke_runner or _run_adapter_smoke_model
        smoke_dir = output_root / "adapter_smoke" / symbol / "smoke_lora"
        prediction_batch = runner(dataset, windows, csv_path, smoke_dir, args)
        records = _forecast_records(
            windows,
            point_forecasts=prediction_batch.point_forecasts,
            full_predictions=prediction_batch.full_predictions,
            label="adapter smoke",
        )
        metrics = _raw_timesfm_metrics(records)
        baseline_metrics = _baseline_metrics(windows)
        best_rmse = min(metric.rmse for metric in baseline_metrics)
        best_directional_accuracy = max(metric.directional_accuracy for metric in baseline_metrics)
        rmse_ratio = _safe_ratio(metrics.rmse, best_rmse)
        directional_delta = round(metrics.directional_accuracy - best_directional_accuracy, 8)
        adapter_rmse_ratio_vs_raw = (
            _safe_ratio(metrics.rmse, raw_row.raw_timesfm_rmse)
            if raw_row.raw_timesfm_rmse is not None
            else None
        )
        adapter_directional_delta_vs_raw = (
            round(metrics.directional_accuracy - raw_row.directional_accuracy, 8)
            if raw_row.directional_accuracy is not None
            else None
        )
        status, decision, kill_reason, selected_for_next_stage = _adapter_stage_decision(
            sample_count=metrics.sample_count,
            adapter_rmse=metrics.rmse,
            raw_rmse=raw_row.raw_timesfm_rmse,
            adapter_rmse_ratio_vs_raw=adapter_rmse_ratio_vs_raw,
            adapter_directional_delta_vs_raw=adapter_directional_delta_vs_raw,
            rmse_ratio_vs_best_baseline=rmse_ratio,
            directional_delta_vs_best_baseline=directional_delta,
            args=args,
        )
        artifact_path = _write_adapter_smoke_artifact(
            output_root,
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            dataset=dataset,
            status=status,
            decision=decision,
            kill_reason=kill_reason,
            prediction_batch=prediction_batch,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            raw_row=raw_row,
            rmse_ratio=rmse_ratio,
            directional_delta=directional_delta,
            adapter_rmse_ratio_vs_raw=adapter_rmse_ratio_vs_raw,
            adapter_directional_delta_vs_raw=adapter_directional_delta_vs_raw,
            records=records,
            args=args,
        )
        return _adapter_smoke_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status=status,
            decision=decision,
            kill_reason=kill_reason,
            runtime_seconds=_elapsed_seconds(started_at),
            model_id=prediction_batch.model_id or args.model_id,
            model_revision=prediction_batch.model_revision or args.model_revision,
            adapter_sha256=prediction_batch.adapter_sha256,
            evaluation_artifact=str(artifact_path),
            training_metadata=prediction_batch.training_metadata_path,
            rmse=metrics.rmse,
            best_baseline_rmse=best_rmse,
            rmse_ratio_vs_best_baseline=rmse_ratio,
            directional_accuracy=metrics.directional_accuracy,
            best_baseline_directional_accuracy=best_directional_accuracy,
            directional_delta_vs_best_baseline=directional_delta,
            raw_timesfm_rmse=raw_row.raw_timesfm_rmse,
            adapter_rmse_ratio_vs_raw=adapter_rmse_ratio_vs_raw,
            adapter_directional_delta_vs_raw=adapter_directional_delta_vs_raw,
            validation_mean_loss=prediction_batch.validation_mean_loss,
            interval_coverage=metrics.interval_coverage,
            mean_interval_width=metrics.mean_interval_width,
            calibration_proxy=metrics.calibration_proxy,
            selected_for_next_stage=selected_for_next_stage,
            notes=(
                f"sample_count={metrics.sample_count}; "
                f"backend={prediction_batch.runtime_metadata.get('backend', 'unknown')}"
            ),
        )
    except Exception as exc:
        return _adapter_smoke_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status="failed",
            decision="stop",
            kill_reason=str(exc),
            runtime_seconds=_elapsed_seconds(started_at),
            raw_timesfm_rmse=raw_row.raw_timesfm_rmse,
            notes="adapter_smoke failed before survivor HPO could run",
        )


def _run_adapter_smoke_model(
    dataset: TimesFmDataset,
    windows: Sequence[TimesFmWindow],
    csv_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> _AdapterSmokePredictionBatch:
    from nlp_stock_prediction.contracts.base import JsonObject
    from nlp_stock_prediction.ml.timesfm.artifacts import (
        TimesFmLoraConfig,
        TimesFmTrainingConfig,
        file_sha256,
    )
    from nlp_stock_prediction.ml.timesfm.evaluate import (
        TimesFmEvaluationConfig,
        TimesFmEvaluationModelSource,
        _load_model_predictor,
    )
    from nlp_stock_prediction.ml.timesfm.train import (
        TimesFmTrainingSource,
        train_timesfm_lora,
        write_timesfm_training_run,
    )

    source = TimesFmTrainingSource(kind="csv", sha256=file_sha256(csv_path), path=str(csv_path))
    training_config = TimesFmTrainingConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        requested_device=args.device,
        epochs=1,
        max_steps=args.smoke_max_steps,
        batch_size=args.smoke_batch_size,
        learning_rate=args.smoke_learning_rate,
        seed=args.seed,
        gradient_clip_norm=args.gradient_clip_norm,
        validation_batches=args.smoke_validation_batches,
        lora=TimesFmLoraConfig(
            r=args.smoke_lora_r,
            lora_alpha=args.smoke_lora_alpha,
            target_modules=args.smoke_lora_target_modules,
            lora_dropout=args.smoke_lora_dropout,
            bias=args.smoke_lora_bias,
        ),
    )
    training_run = train_timesfm_lora(dataset, training_config, source)
    paths = write_timesfm_training_run(training_run, output_dir)
    training_metadata = cast(
        JsonObject,
        json.loads(paths.metadata_path.read_text(encoding="utf-8")),
    )
    model_source = TimesFmEvaluationModelSource(
        model_dir=output_dir,
        adapter_dir=paths.adapter_dir,
        training_ticker=dataset.ticker,
        model_id=training_run.result.model_id,
        model_revision=training_run.result.model_revision,
        adapter_sha256=paths.adapter_sha256,
        recorded_adapter_sha256=paths.adapter_sha256,
        training_metadata_sha256=paths.metadata_sha256,
        training_metadata=training_metadata,
    )
    evaluation_config = TimesFmEvaluationConfig(
        requested_device=args.device,
        max_windows=args.screen_max_windows,
        min_evaluation_windows=args.min_evaluation_windows,
        as_of=date.fromisoformat(args.as_of),
    )
    predictor, runtime_metadata = _load_model_predictor(model_source, evaluation_config)
    point_forecasts: list[tuple[float, ...]] = []
    full_predictions: list[tuple[tuple[float, ...], ...]] = []
    for window in windows:
        prediction = predictor(window)
        point_forecasts.append(prediction.point_forecast)
        full_predictions.append(prediction.full_predictions)
    return _AdapterSmokePredictionBatch(
        model_id=training_run.result.model_id,
        model_revision=training_run.result.model_revision,
        adapter_sha256=paths.adapter_sha256,
        training_metadata_path=str(paths.metadata_path),
        validation_mean_loss=training_run.result.validation_metrics.mean_loss,
        point_forecasts=tuple(point_forecasts),
        full_predictions=tuple(full_predictions),
        runtime_metadata={
            **runtime_metadata,
            "training_backend": training_run.result.runtime_metadata.get("backend", "unknown"),
            "adapter_dir": str(paths.adapter_dir),
        },
    )


def _adapter_stage_decision(
    *,
    sample_count: int,
    adapter_rmse: float,
    raw_rmse: float | None,
    adapter_rmse_ratio_vs_raw: float | None,
    adapter_directional_delta_vs_raw: float | None,
    rmse_ratio_vs_best_baseline: float | None,
    directional_delta_vs_best_baseline: float,
    args: argparse.Namespace,
) -> tuple[FunnelStatus, FunnelDecision, str | None, bool]:
    if sample_count < args.min_evaluation_windows:
        return (
            "research_only",
            "audit_only",
            (
                "limited_validation_windows:"
                f"{sample_count}<min_evaluation_windows:{args.min_evaluation_windows}"
            ),
            False,
        )
    if raw_rmse is None:
        return (
            "borderline",
            "audit_only",
            "adapter_smoke_missing_raw_rmse",
            False,
        )
    rmse_non_improving = _adapter_rmse_non_improving(
        adapter_rmse=adapter_rmse,
        raw_rmse=raw_rmse,
        adapter_rmse_ratio_vs_raw=adapter_rmse_ratio_vs_raw,
    )
    directional_non_improving = (
        adapter_directional_delta_vs_raw is None or adapter_directional_delta_vs_raw <= 0.0
    )
    if rmse_non_improving and directional_non_improving:
        ratio_note = (
            "unavailable"
            if adapter_rmse_ratio_vs_raw is None
            else f"{adapter_rmse_ratio_vs_raw:.4f}"
        )
        delta_note = (
            "unavailable"
            if adapter_directional_delta_vs_raw is None
            else f"{adapter_directional_delta_vs_raw:.4f}"
        )
        return (
            "killed",
            "stop",
            (
                "adapter_smoke_no_lift_vs_raw:"
                f"rmse_ratio_vs_raw={ratio_note};"
                f"directional_delta_vs_raw={delta_note}"
            ),
            False,
        )
    has_positive_lift = not rmse_non_improving or (
        adapter_directional_delta_vs_raw is not None and adapter_directional_delta_vs_raw > 0.0
    )
    near_best_baseline = (
        rmse_ratio_vs_best_baseline is not None
        and rmse_ratio_vs_best_baseline <= args.adapter_rmse_promote_threshold
    ) or directional_delta_vs_best_baseline >= args.adapter_directional_promote_threshold
    if has_positive_lift and near_best_baseline:
        return "passed", "run_hpo", None, True
    if has_positive_lift:
        return (
            "research_only",
            "audit_only",
            (
                "adapter_smoke_lift_not_baseline_ready:"
                f"rmse_ratio_vs_best_baseline={rmse_ratio_vs_best_baseline};"
                f"directional_delta_vs_best_baseline={directional_delta_vs_best_baseline:.4f}"
            ),
            False,
        )
    return (
        "borderline",
        "audit_only",
        "adapter_smoke_inconclusive_lift",
        False,
    )


def _adapter_rmse_non_improving(
    *,
    adapter_rmse: float,
    raw_rmse: float,
    adapter_rmse_ratio_vs_raw: float | None,
) -> bool:
    if adapter_rmse_ratio_vs_raw is None:
        return adapter_rmse >= raw_rmse
    return adapter_rmse_ratio_vs_raw >= 1.0


def _write_adapter_smoke_artifact(
    output_root: Path,
    *,
    run_id: str,
    created_at: str,
    as_of: date,
    symbol: str,
    dataset: TimesFmDataset,
    status: FunnelStatus,
    decision: FunnelDecision,
    kill_reason: str | None,
    prediction_batch: _AdapterSmokePredictionBatch,
    metrics: _RawTimesFmMetrics,
    baseline_metrics: Sequence[_BaselineMetrics],
    raw_row: SignalFunnelLeaderboardRow,
    rmse_ratio: float | None,
    directional_delta: float,
    adapter_rmse_ratio_vs_raw: float | None,
    adapter_directional_delta_vs_raw: float | None,
    records: Sequence[_RawTimesFmRecord],
    args: argparse.Namespace,
) -> Path:
    artifact_path = _adapter_smoke_artifact_path(output_root, symbol)
    payload = {
        "schema_version": "ml.timesfm.adapter_smoke_evaluation.v1",
        "run_id": run_id,
        "created_at": created_at,
        "as_of": as_of.isoformat(),
        "symbol": symbol,
        "stage": "adapter_smoke",
        "method": "lora_adapter_smoke",
        "status": status,
        "decision": decision,
        "kill_reason": kill_reason,
        "model_id": prediction_batch.model_id or args.model_id,
        "model_revision": prediction_batch.model_revision or args.model_revision,
        "adapter_sha256": prediction_batch.adapter_sha256,
        "training_metadata": prediction_batch.training_metadata_path,
        "validation_mean_loss": prediction_batch.validation_mean_loss,
        "dataset_hash": dataset.dataset_hash,
        "target_field": dataset.target_field,
        "context_length": dataset.context_length,
        "horizon_length": dataset.horizon_length,
        "max_windows": args.screen_max_windows,
        "config": _adapter_smoke_config_payload(args),
        "metrics": metrics.model_dump(mode="json"),
        "baselines": [metric.model_dump(mode="json") for metric in baseline_metrics],
        "rmse_ratio_vs_best_baseline": rmse_ratio,
        "directional_delta_vs_best_baseline": directional_delta,
        "raw_timesfm": {
            "rmse": raw_row.raw_timesfm_rmse,
            "directional_accuracy": raw_row.directional_accuracy,
            "evaluation_artifact": raw_row.evaluation_artifact,
        },
        "adapter_rmse_ratio_vs_raw": adapter_rmse_ratio_vs_raw,
        "adapter_directional_delta_vs_raw": adapter_directional_delta_vs_raw,
        "records": [record.model_dump(mode="json") for record in records],
        "runtime_metadata": prediction_batch.runtime_metadata,
    }
    write_manifest(artifact_path, payload)
    return artifact_path


def _load_reusable_adapter_smoke_row(
    output_root: Path,
    *,
    run_id: str,
    created_at: str,
    as_of: date,
    symbol: str,
    policy: TickerPolicy,
    args: argparse.Namespace,
    data_check_row: SignalFunnelLeaderboardRow,
    raw_row: SignalFunnelLeaderboardRow,
    runtime_seconds: float,
) -> SignalFunnelLeaderboardRow | None:
    if args.refresh_runs:
        return None
    artifact_path = _adapter_smoke_artifact_path(output_root, symbol)
    if not artifact_path.exists():
        return None
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != "ml.timesfm.adapter_smoke_evaluation.v1":
            return None
        if payload.get("symbol") != symbol:
            return None
        if payload.get("dataset_hash") != data_check_row.dataset_hash:
            return None
        if payload.get("config") != _adapter_smoke_config_payload(args):
            return None
        raw_timesfm = payload.get("raw_timesfm")
        if not isinstance(raw_timesfm, dict):
            return None
        if raw_timesfm.get("rmse") != raw_row.raw_timesfm_rmse:
            return None
        metrics = cast(dict[str, Any], payload["metrics"])
        baselines = cast(list[dict[str, Any]], payload["baselines"])
        best_baseline_rmse = min(float(baseline["rmse"]) for baseline in baselines)
        best_baseline_directional_accuracy = max(
            float(baseline["directional_accuracy"]) for baseline in baselines
        )
        return _adapter_smoke_leaderboard_row(
            run_id=run_id,
            created_at=created_at,
            as_of=as_of,
            symbol=symbol,
            policy=policy,
            args=args,
            data_check_row=data_check_row,
            status=cast(FunnelStatus, payload["status"]),
            decision=cast(FunnelDecision, payload["decision"]),
            kill_reason=cast(str | None, payload.get("kill_reason")),
            runtime_seconds=runtime_seconds,
            model_id=cast(str | None, payload.get("model_id")),
            model_revision=cast(str | None, payload.get("model_revision")),
            adapter_sha256=cast(str | None, payload.get("adapter_sha256")),
            evaluation_artifact=str(artifact_path),
            training_metadata=cast(str | None, payload.get("training_metadata")),
            rmse=float(metrics["rmse"]),
            best_baseline_rmse=best_baseline_rmse,
            rmse_ratio_vs_best_baseline=cast(
                float | None,
                payload.get("rmse_ratio_vs_best_baseline"),
            ),
            directional_accuracy=float(metrics["directional_accuracy"]),
            best_baseline_directional_accuracy=best_baseline_directional_accuracy,
            directional_delta_vs_best_baseline=float(payload["directional_delta_vs_best_baseline"]),
            raw_timesfm_rmse=raw_row.raw_timesfm_rmse,
            adapter_rmse_ratio_vs_raw=cast(
                float | None,
                payload.get("adapter_rmse_ratio_vs_raw"),
            ),
            adapter_directional_delta_vs_raw=cast(
                float | None,
                payload.get("adapter_directional_delta_vs_raw"),
            ),
            validation_mean_loss=cast(float | None, payload.get("validation_mean_loss")),
            interval_coverage=cast(float | None, metrics.get("interval_coverage")),
            mean_interval_width=cast(float | None, metrics.get("mean_interval_width")),
            calibration_proxy=cast(float | None, metrics.get("calibration_proxy")),
            selected_for_next_stage=bool(payload["decision"] == "run_hpo"),
            notes=f"sample_count={metrics['sample_count']}; reused=true",
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _adapter_smoke_artifact_path(output_root: Path, symbol: str) -> Path:
    return output_root / "adapter_smoke" / f"{symbol}.evaluation.json"


def _adapter_smoke_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "context_length": args.context_length,
        "horizon_length": args.horizon_length,
        "target_field": args.target_field,
        "screen_max_windows": args.screen_max_windows,
        "smoke_max_steps": args.smoke_max_steps,
        "smoke_batch_size": args.smoke_batch_size,
        "smoke_learning_rate": args.smoke_learning_rate,
        "smoke_validation_batches": args.smoke_validation_batches,
        "smoke_lora_r": args.smoke_lora_r,
        "smoke_lora_alpha": args.smoke_lora_alpha,
        "smoke_lora_dropout": args.smoke_lora_dropout,
        "smoke_lora_target_modules": args.smoke_lora_target_modules,
        "smoke_lora_bias": args.smoke_lora_bias,
        "seed": args.seed,
        "gradient_clip_norm": args.gradient_clip_norm,
    }


def _transpose(rows: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if not rows:
        return ()
    width = min(len(row) for row in rows)
    return tuple(tuple(row[index] for row in rows) for index in range(width))


def _screen_windows(dataset: TimesFmDataset, *, max_windows: int) -> tuple[TimesFmWindow, ...]:
    windows = tuple(sorted(dataset.validation_windows, key=lambda window: window.context_end_index))
    return windows[-max_windows:] if len(windows) > max_windows else windows


def _baseline_metrics(windows: Sequence[TimesFmWindow]) -> tuple[_BaselineMetrics, ...]:
    persistence_predictions: list[float] = []
    persistence_returns: list[float] = []
    recent_mean_predictions: list[float] = []
    recent_mean_returns: list[float] = []
    actual_values: list[float] = []
    actual_returns: list[float] = []
    for window in windows:
        context_final = window.context_values[-1]
        actual_final = window.future_values[-1]
        recent_mean_final = _recent_mean_return_prediction(window)
        actual_values.append(actual_final)
        actual_returns.append(_safe_return(actual_final, context_final))
        persistence_predictions.append(context_final)
        persistence_returns.append(0.0)
        recent_mean_predictions.append(recent_mean_final)
        recent_mean_returns.append(_safe_return(recent_mean_final, context_final))
    return (
        _metric_summary(
            "last_close_persistence",
            predictions=persistence_predictions,
            actual_values=actual_values,
            predicted_returns=persistence_returns,
            actual_returns=actual_returns,
        ),
        _metric_summary(
            "recent_mean_return",
            predictions=recent_mean_predictions,
            actual_values=actual_values,
            predicted_returns=recent_mean_returns,
            actual_returns=actual_returns,
        ),
    )


def _metric_summary(
    name: str,
    *,
    predictions: Sequence[float],
    actual_values: Sequence[float],
    predicted_returns: Sequence[float],
    actual_returns: Sequence[float],
) -> _BaselineMetrics:
    if not predictions:
        return _BaselineMetrics(name=name, rmse=0.0, directional_accuracy=0.0, sample_count=0)
    squared_errors = [
        (prediction - actual) ** 2
        for prediction, actual in zip(predictions, actual_values, strict=True)
    ]
    direction_hits = [
        int(_direction(predicted) == _direction(actual))
        for predicted, actual in zip(predicted_returns, actual_returns, strict=True)
    ]
    return _BaselineMetrics(
        name=name,
        rmse=round((sum(squared_errors) / len(squared_errors)) ** 0.5, 8),
        directional_accuracy=round(sum(direction_hits) / len(direction_hits), 8),
        sample_count=len(predictions),
    )


def _recent_mean_return_prediction(window: TimesFmWindow) -> float:
    returns = [
        _safe_return(current, previous)
        for previous, current in zip(window.context_values, window.context_values[1:], strict=False)
    ]
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


def _safe_ratio(value: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return 0.0 if value == 0.0 else None
    return round(value / baseline, 8)


def _stage0_decision(
    counts: _WindowCounts,
    *,
    min_evaluation_windows: int,
) -> tuple[FunnelStatus, FunnelDecision, str | None]:
    if counts.test < min_evaluation_windows:
        return (
            "research_only",
            "audit_only",
            f"limited_test_windows:{counts.test}<min_evaluation_windows:{min_evaluation_windows}",
        )
    return "passed", "continue", None


def _estimate_window_counts(
    *,
    row_count: int,
    context_length: int,
    horizon_length: int,
) -> _WindowCounts:
    raw_window_count = max(0, row_count - context_length - horizon_length + 1)
    purge_gap = horizon_length
    minimum_windows = 3 + (2 * purge_gap)
    if raw_window_count < minimum_windows:
        raise ValueError(
            "insufficient history: TimesFM windows cannot fill train, validation, and test splits"
        )
    train_count = int(raw_window_count * 0.70)
    train_count = max(1, min(train_count, raw_window_count - (2 * purge_gap) - 2))
    validation_start = train_count + purge_gap
    remaining_after_validation_start = raw_window_count - validation_start
    validation_count = int(raw_window_count * 0.15)
    validation_count = max(
        1,
        min(validation_count, remaining_after_validation_start - purge_gap - 1),
    )
    test_start = validation_start + validation_count + purge_gap
    test_count = raw_window_count - test_start
    if test_count < 1:
        raise ValueError("insufficient history: TimesFM split settings leave no test windows")
    return _WindowCounts(
        train=train_count,
        validation=validation_count,
        test=test_count,
        raw=raw_window_count,
        purged=2 * purge_gap,
    )


def _leaderboard_row(
    *,
    run_id: str,
    created_at: str,
    as_of: date,
    symbol: str,
    policy: TickerPolicy,
    args: argparse.Namespace,
    history_start: str | None,
    latest_bar: str | None,
    bar_count: int | None,
    counts: _WindowCounts | None,
    status: FunnelStatus,
    decision: FunnelDecision,
    kill_reason: str | None,
    dataset_hash: str | None,
    runtime_seconds: float,
    notes: str | None,
) -> SignalFunnelLeaderboardRow:
    return SignalFunnelLeaderboardRow(
        run_id=run_id,
        created_at=created_at,
        as_of=as_of.isoformat(),
        symbol=symbol,
        stage="data_check",
        method="adjusted_ohlcv_dataset",
        status=status,
        kill_reason=kill_reason,
        decision=decision,
        asset_type=policy.asset_type,
        history_start=history_start,
        latest_bar=latest_bar,
        bar_count=bar_count,
        train_windows=counts.train if counts is not None else None,
        validation_windows=counts.validation if counts is not None else None,
        test_windows=counts.test if counts is not None else None,
        context_length=args.context_length,
        horizon_length=args.horizon_length,
        max_windows=None,
        runtime_seconds=runtime_seconds,
        device=args.device,
        dataset_hash=dataset_hash,
        selected_for_next_stage=status == "passed",
        promoted_for_scoring=False,
        notes=notes,
    )


def _baseline_leaderboard_row(
    *,
    run_id: str,
    created_at: str,
    as_of: date,
    symbol: str,
    policy: TickerPolicy,
    args: argparse.Namespace,
    data_check_row: SignalFunnelLeaderboardRow,
    method: str,
    status: FunnelStatus,
    decision: FunnelDecision,
    kill_reason: str | None,
    runtime_seconds: float,
    notes: str | None,
    rmse: float | None = None,
    best_baseline_rmse: float | None = None,
    rmse_ratio_vs_best_baseline: float | None = None,
    directional_accuracy: float | None = None,
    best_baseline_directional_accuracy: float | None = None,
    directional_delta_vs_best_baseline: float | None = None,
    selected_for_next_stage: bool = False,
) -> SignalFunnelLeaderboardRow:
    return SignalFunnelLeaderboardRow(
        run_id=run_id,
        created_at=created_at,
        as_of=as_of.isoformat(),
        symbol=symbol,
        stage="baseline_screen",
        method=method,
        status=status,
        kill_reason=kill_reason,
        decision=decision,
        asset_type=policy.asset_type,
        history_start=data_check_row.history_start,
        latest_bar=data_check_row.latest_bar,
        bar_count=data_check_row.bar_count,
        train_windows=data_check_row.train_windows,
        validation_windows=data_check_row.validation_windows,
        test_windows=data_check_row.test_windows,
        context_length=args.context_length,
        horizon_length=args.horizon_length,
        max_windows=args.screen_max_windows,
        runtime_seconds=runtime_seconds,
        device=args.device,
        dataset_hash=data_check_row.dataset_hash,
        rmse=rmse,
        best_baseline_rmse=best_baseline_rmse,
        rmse_ratio_vs_best_baseline=rmse_ratio_vs_best_baseline,
        directional_accuracy=directional_accuracy,
        best_baseline_directional_accuracy=best_baseline_directional_accuracy,
        directional_delta_vs_best_baseline=directional_delta_vs_best_baseline,
        selected_for_next_stage=selected_for_next_stage,
        promoted_for_scoring=False,
        notes=notes,
    )


def _raw_timesfm_leaderboard_row(
    *,
    run_id: str,
    created_at: str,
    as_of: date,
    symbol: str,
    policy: TickerPolicy,
    args: argparse.Namespace,
    data_check_row: SignalFunnelLeaderboardRow,
    status: FunnelStatus,
    decision: FunnelDecision,
    kill_reason: str | None,
    runtime_seconds: float,
    notes: str | None,
    model_id: str | None = None,
    model_revision: str | None = None,
    evaluation_artifact: str | None = None,
    rmse: float | None = None,
    best_baseline_rmse: float | None = None,
    rmse_ratio_vs_best_baseline: float | None = None,
    directional_accuracy: float | None = None,
    best_baseline_directional_accuracy: float | None = None,
    directional_delta_vs_best_baseline: float | None = None,
    raw_timesfm_rmse: float | None = None,
    interval_coverage: float | None = None,
    mean_interval_width: float | None = None,
    calibration_proxy: float | None = None,
    selected_for_next_stage: bool = False,
) -> SignalFunnelLeaderboardRow:
    return SignalFunnelLeaderboardRow(
        run_id=run_id,
        created_at=created_at,
        as_of=as_of.isoformat(),
        symbol=symbol,
        stage="raw_timesfm_screen",
        method="raw_timesfm_base",
        status=status,
        kill_reason=kill_reason,
        decision=decision,
        asset_type=policy.asset_type,
        history_start=data_check_row.history_start,
        latest_bar=data_check_row.latest_bar,
        bar_count=data_check_row.bar_count,
        train_windows=data_check_row.train_windows,
        validation_windows=data_check_row.validation_windows,
        test_windows=data_check_row.test_windows,
        context_length=args.context_length,
        horizon_length=args.horizon_length,
        max_windows=args.screen_max_windows,
        runtime_seconds=runtime_seconds,
        device=args.device,
        model_id=model_id,
        model_revision=model_revision,
        dataset_hash=data_check_row.dataset_hash,
        evaluation_artifact=evaluation_artifact,
        rmse=rmse,
        best_baseline_rmse=best_baseline_rmse,
        rmse_ratio_vs_best_baseline=rmse_ratio_vs_best_baseline,
        directional_accuracy=directional_accuracy,
        best_baseline_directional_accuracy=best_baseline_directional_accuracy,
        directional_delta_vs_best_baseline=directional_delta_vs_best_baseline,
        raw_timesfm_rmse=raw_timesfm_rmse,
        interval_coverage=interval_coverage,
        mean_interval_width=mean_interval_width,
        calibration_proxy=calibration_proxy,
        selected_for_next_stage=selected_for_next_stage,
        promoted_for_scoring=False,
        notes=notes,
    )


def _adapter_smoke_leaderboard_row(
    *,
    run_id: str,
    created_at: str,
    as_of: date,
    symbol: str,
    policy: TickerPolicy,
    args: argparse.Namespace,
    data_check_row: SignalFunnelLeaderboardRow,
    status: FunnelStatus,
    decision: FunnelDecision,
    kill_reason: str | None,
    runtime_seconds: float,
    notes: str | None,
    model_id: str | None = None,
    model_revision: str | None = None,
    adapter_sha256: str | None = None,
    evaluation_artifact: str | None = None,
    training_metadata: str | None = None,
    rmse: float | None = None,
    best_baseline_rmse: float | None = None,
    rmse_ratio_vs_best_baseline: float | None = None,
    directional_accuracy: float | None = None,
    best_baseline_directional_accuracy: float | None = None,
    directional_delta_vs_best_baseline: float | None = None,
    raw_timesfm_rmse: float | None = None,
    adapter_rmse_ratio_vs_raw: float | None = None,
    adapter_directional_delta_vs_raw: float | None = None,
    validation_mean_loss: float | None = None,
    interval_coverage: float | None = None,
    mean_interval_width: float | None = None,
    calibration_proxy: float | None = None,
    selected_for_next_stage: bool = False,
) -> SignalFunnelLeaderboardRow:
    return SignalFunnelLeaderboardRow(
        run_id=run_id,
        created_at=created_at,
        as_of=as_of.isoformat(),
        symbol=symbol,
        stage="adapter_smoke",
        method="lora_adapter_smoke",
        status=status,
        kill_reason=kill_reason,
        decision=decision,
        asset_type=policy.asset_type,
        history_start=data_check_row.history_start,
        latest_bar=data_check_row.latest_bar,
        bar_count=data_check_row.bar_count,
        train_windows=data_check_row.train_windows,
        validation_windows=data_check_row.validation_windows,
        test_windows=data_check_row.test_windows,
        context_length=args.context_length,
        horizon_length=args.horizon_length,
        max_windows=args.screen_max_windows,
        runtime_seconds=runtime_seconds,
        device=args.device,
        model_id=model_id,
        model_revision=model_revision,
        adapter_sha256=adapter_sha256,
        dataset_hash=data_check_row.dataset_hash,
        evaluation_artifact=evaluation_artifact,
        training_metadata=training_metadata,
        rmse=rmse,
        best_baseline_rmse=best_baseline_rmse,
        rmse_ratio_vs_best_baseline=rmse_ratio_vs_best_baseline,
        directional_accuracy=directional_accuracy,
        best_baseline_directional_accuracy=best_baseline_directional_accuracy,
        directional_delta_vs_best_baseline=directional_delta_vs_best_baseline,
        raw_timesfm_rmse=raw_timesfm_rmse,
        adapter_rmse_ratio_vs_raw=adapter_rmse_ratio_vs_raw,
        adapter_directional_delta_vs_raw=adapter_directional_delta_vs_raw,
        validation_mean_loss=validation_mean_loss,
        interval_coverage=interval_coverage,
        mean_interval_width=mean_interval_width,
        calibration_proxy=calibration_proxy,
        selected_for_next_stage=selected_for_next_stage,
        promoted_for_scoring=False,
        notes=notes,
    )


def _stage0_notes(data_record: dict[str, Any], counts: _WindowCounts) -> str:
    metadata = data_record.get("metadata")
    source = metadata.get("source") if isinstance(metadata, dict) else None
    return f"source={source or 'unknown'}; raw_windows={counts.raw}; purged_windows={counts.purged}"


def _write_signal_funnel_outputs(
    output_root: Path,
    manifest: dict[str, Any],
    rows: Sequence[SignalFunnelLeaderboardRow],
) -> None:
    write_manifest(output_root / "manifest.json", manifest)
    row_payloads = [row.model_dump(mode="json") for row in rows]
    write_manifest(output_root / "leaderboard.json", {"rows": row_payloads})
    _write_leaderboard_csv(output_root / "leaderboard.csv", row_payloads)


def _write_leaderboard_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SignalFunnelLeaderboardRow.model_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(symbol.strip().upper() for symbol in value.split(",") if symbol.strip())
    if not symbols:
        raise ValueError("at least one symbol is required")
    return symbols


def _default_run_id(as_of: date) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"timesfm-funnel-{as_of.isoformat()}-{timestamp}"


def _elapsed_seconds(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 6)


def _should_stop_after(completed_stage: FunnelStage, requested_stop_after: str) -> bool:
    return FUNNEL_STAGES.index(completed_stage) >= FUNNEL_STAGES.index(requested_stop_after)


def _default_stop_after_for_profile(profile: FunnelProfile) -> FunnelStage:
    if profile == "quick":
        return "raw_timesfm_screen"
    return "adapter_smoke"


def build_parser() -> argparse.ArgumentParser:
    """Build the signal-funnel CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--as-of", default=DEFAULT_AS_OF.isoformat())
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--profile", choices=("quick", "walkaway", "full"), default="walkaway")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument(
        "--stop-after",
        choices=FUNNEL_STAGES,
        default=None,
        help=(
            "Stop after the named funnel stage. Defaults to raw_timesfm_screen for quick "
            "profile and adapter_smoke otherwise."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--horizon-length", type=int, default=16)
    parser.add_argument(
        "--target-field",
        choices=("auto", "close", "adjusted_close"),
        default="adjusted_close",
    )
    parser.add_argument("--max-latest-bar-age-days", type=int, default=5)
    parser.add_argument("--min-evaluation-windows", type=int, default=3)
    parser.add_argument("--screen-max-windows", type=int, default=64)
    parser.add_argument("--smoke-max-steps", type=int, default=200)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--smoke-learning-rate", type=float, default=1e-4)
    parser.add_argument("--smoke-validation-batches", type=int, default=2)
    parser.add_argument("--smoke-lora-r", type=int, default=4)
    parser.add_argument("--smoke-lora-alpha", type=int, default=8)
    parser.add_argument("--smoke-lora-dropout", type=float, default=0.05)
    parser.add_argument("--smoke-lora-target-modules", default="all-linear")
    parser.add_argument("--smoke-lora-bias", choices=("none", "all", "lora_only"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-hpo-trials-per-ticker", type=int, default=24)
    parser.add_argument("--raw-rmse-kill-threshold", type=float, default=1.15)
    parser.add_argument("--raw-directional-kill-threshold", type=float, default=-0.05)
    parser.add_argument("--raw-rmse-promote-threshold", type=float, default=1.05)
    parser.add_argument("--raw-directional-promote-threshold", type=float, default=0.0)
    parser.add_argument("--raw-rmse-research-threshold", type=float, default=1.10)
    parser.add_argument("--raw-directional-research-threshold", type=float, default=-0.02)
    parser.add_argument("--adapter-rmse-promote-threshold", type=float, default=1.05)
    parser.add_argument("--adapter-directional-promote-threshold", type=float, default=0.02)
    parser.add_argument("--min-final-directional-accuracy", type=float, default=0.50)
    parser.add_argument("--max-final-rmse-ratio-vs-best-baseline", type=float, default=1.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    raw_predictor: RawTimesFmPredictor | None = None,
    adapter_smoke_runner: AdapterSmokeRunner | None = None,
) -> int:
    """CLI entry point for ``python -m nlp_stock_prediction.ml.timesfm.signal_funnel``."""

    return run_signal_funnel(
        build_parser().parse_args(argv),
        raw_predictor=raw_predictor,
        adapter_smoke_runner=adapter_smoke_runner,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_OUTPUT_ROOT",
    "SignalFunnelLeaderboardRow",
    "build_parser",
    "main",
    "run_signal_funnel",
]
