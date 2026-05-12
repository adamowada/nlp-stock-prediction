"""Staged TimesFM signal funnel orchestration."""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from nlp_stock_prediction.contracts.base import ContractModel, NonEmptyStr, TickerSymbol
from nlp_stock_prediction.ml.timesfm.dataset import TimesFmDatasetConfig, build_timesfm_dataset
from nlp_stock_prediction.ml.timesfm.focused_hpo import (
    DEFAULT_AS_OF,
    DEFAULT_SYMBOLS,
    TickerPolicy,
    default_ticker_policies,
    dry_run_data_record,
    ensure_symbol_data,
    write_manifest,
)
from nlp_stock_prediction.ml.train import load_price_bars_csv

DEFAULT_OUTPUT_ROOT = Path("artifacts/ml/timesfm-funnel")
DEFAULT_DATA_DIR = Path("data/ml/wsb_10y")
IMPLEMENTED_STAGES = ("data_check",)
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


def run_signal_funnel(args: argparse.Namespace) -> int:
    """Run the implemented signal-funnel stages and write manifest/leaderboard artifacts."""

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
        default="data_check",
        help="Only data_check is implemented in this stage-0 slice.",
    )
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
    parser.add_argument("--max-hpo-trials-per-ticker", type=int, default=24)
    parser.add_argument("--raw-rmse-kill-threshold", type=float, default=1.15)
    parser.add_argument("--raw-directional-kill-threshold", type=float, default=-0.05)
    parser.add_argument("--adapter-rmse-promote-threshold", type=float, default=1.05)
    parser.add_argument("--adapter-directional-promote-threshold", type=float, default=0.02)
    parser.add_argument("--min-final-directional-accuracy", type=float, default=0.50)
    parser.add_argument("--max-final-rmse-ratio-vs-best-baseline", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m nlp_stock_prediction.ml.timesfm.signal_funnel``."""

    return run_signal_funnel(build_parser().parse_args(argv))


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
