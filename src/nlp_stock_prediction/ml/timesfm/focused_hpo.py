"""Focused six-ticker TimesFM data collection and HPO orchestration."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from nlp_stock_prediction.ml.timesfm.smoke import DEFAULT_MODEL_ID

DEFAULT_SYMBOLS = ("MU", "SPY", "ASTS", "SNDK", "GOOG", "NVDA")
DEFAULT_OUTPUT_ROOT = Path("artifacts/ml/wsb-six-10y")
DEFAULT_DATA_DIR = Path("data/ml/wsb_10y")
DEFAULT_AS_OF = date(2026, 5, 11)
YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0 (compatible; nlp-stock-prediction-focused-timesfm/0.1)"
ADJUSTED_PRICE_POLICY = "split_dividend_adjusted_ohlcv_from_yahoo_adjclose_ratio"


@dataclass(frozen=True, slots=True)
class TickerPolicy:
    """Data and interpretation policy for a focused ticker."""

    symbol: str
    asset_type: str = "operating_company"
    history_start_override: date | None = None
    minimum_rows: int = 220
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HpoTrial:
    """One TimesFM LoRA HPO candidate."""

    context_length: int
    horizon_length: int
    max_steps: int
    batch_size: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    @property
    def run_id(self) -> str:
        lr = _float_token(self.learning_rate)
        dropout = _float_token(self.lora_dropout)
        return (
            f"ctx{self.context_length}_h{self.horizon_length}_steps{self.max_steps}"
            f"_bs{self.batch_size}_lr{lr}_r{self.lora_r}_a{self.lora_alpha}_do{dropout}"
        )


@dataclass(frozen=True, slots=True)
class EvaluationChoice:
    """Comparable summary for a completed evaluation artifact."""

    symbol: str
    trial: HpoTrial
    status: str
    suitable_for_scoring: bool
    directional_accuracy: float
    rmse: float
    best_baseline_rmse: float
    evaluation_path: Path
    run_dir: Path

    @property
    def rmse_ratio(self) -> float:
        if self.best_baseline_rmse <= 0:
            return math.inf if self.rmse > 0 else 0.0
        return self.rmse / self.best_baseline_rmse


def default_ticker_policies() -> dict[str, TickerPolicy]:
    """Return the current focused six policy map."""

    return {
        "MU": TickerPolicy(
            symbol="MU",
            notes=("Use full available 10-year adjusted OHLCV when the provider supplies it.",),
        ),
        "SPY": TickerPolicy(
            symbol="SPY",
            asset_type="etf",
            notes=(
                "ETF: TimesFM may use OHLCV, but fundamentals should be ETF-aware "
                "rather than company-style.",
            ),
        ),
        "ASTS": TickerPolicy(
            symbol="ASTS",
            notes=(
                "Use available public-company history; do not backfill with unrelated "
                "pre-listing history.",
            ),
        ),
        "SNDK": TickerPolicy(
            symbol="SNDK",
            history_start_override=date(2025, 2, 24),
            minimum_rows=180,
            notes=(
                "Current Sandisk standalone history starts on 2025-02-24 after separation "
                "from Western Digital; do not stitch old pre-2016 SNDK history.",
            ),
        ),
        "GOOG": TickerPolicy(
            symbol="GOOG",
            notes=("Use adjusted OHLCV so stock-split history is continuous.",),
        ),
        "NVDA": TickerPolicy(
            symbol="NVDA",
            notes=("Use adjusted OHLCV so 2021 and 2024 stock splits are continuous.",),
        ),
    }


def run_focused_hpo(args: argparse.Namespace) -> int:
    """Fetch focused data, run HPO, and write a manifest."""

    symbols = _parse_symbols(args.symbols)
    policies = default_ticker_policies()
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    as_of = date.fromisoformat(args.as_of)
    start = as_of - timedelta(days=365 * args.years)
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        data_dir.mkdir(parents=True, exist_ok=True)
    trials = hpo_trials_from_args(args)
    manifest: dict[str, Any] = {
        "schema_version": "ml.timesfm.focused_hpo_manifest.v1",
        "symbols": list(symbols),
        "as_of": as_of.isoformat(),
        "years": args.years,
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "price_policy": ADJUSTED_PRICE_POLICY,
        "max_trials_per_ticker": args.max_trials_per_ticker,
        "dry_run": bool(args.dry_run),
        "records": [],
    }

    for offset, symbol in enumerate(symbols, start=1):
        policy = policies.get(symbol, TickerPolicy(symbol=symbol))
        action = "planning" if args.dry_run else "collecting data"
        print(f"[{offset:02d}/{len(symbols):02d}] {symbol}: {action}")
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
            record = run_symbol_hpo(
                symbol,
                policy,
                data_record=data_record,
                output_root=output_root,
                trials=trials,
                args=args,
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            HTTPError,
            URLError,
            subprocess.SubprocessError,
        ) as exc:
            record = {
                "symbol": symbol,
                "status": "failed",
                "error": str(exc),
                "policy": _policy_payload(policy),
            }
            print(f"[{offset:02d}/{len(symbols):02d}] {symbol}: failed {exc}")
            if args.stop_on_first_error:
                manifest["records"].append(record)
                write_manifest(output_root / "focused-hpo-manifest.json", manifest)
                return 1
        manifest["records"].append(record)
        write_manifest(output_root / "focused-hpo-manifest.json", manifest)
    return 0 if all(record.get("status") != "failed" for record in manifest["records"]) else 1


def dry_run_data_record(
    symbol: str,
    policy: TickerPolicy,
    *,
    data_dir: Path,
    start: date,
    as_of: date,
) -> dict[str, Any]:
    """Return estimated data metadata without network or filesystem writes."""

    effective_start = max(start, policy.history_start_override or start)
    estimated_rows = max(0, int(((as_of - effective_start).days + 1) * 5 / 7))
    metadata = {
        "schema_version": "ml.timesfm.focused_ohlcv.v1",
        "symbol": symbol,
        "asset_type": policy.asset_type,
        "requested_start": start.isoformat(),
        "effective_start": effective_start.isoformat(),
        "as_of": as_of.isoformat(),
        "source": "dry-run-estimate",
        "source_url_template": YAHOO_CHART_URL,
        "price_policy": ADJUSTED_PRICE_POLICY,
        "lineage_notes": list(policy.notes),
        "estimated_rows": estimated_rows,
    }
    return {
        "csv_path": str(data_dir / f"{symbol}.csv"),
        "metadata_path": str(data_dir / f"{symbol}.metadata.json"),
        "rows": estimated_rows,
        "first": effective_start.isoformat(),
        "last": as_of.isoformat(),
        "metadata": metadata,
    }


def ensure_symbol_data(
    symbol: str,
    policy: TickerPolicy,
    *,
    data_dir: Path,
    start: date,
    as_of: date,
    refresh: bool,
    sleep_seconds: float,
) -> dict[str, Any]:
    """Fetch or reuse a focused ticker CSV and sidecar metadata."""

    effective_start = max(start, policy.history_start_override or start)
    csv_path = data_dir / f"{symbol}.csv"
    metadata_path = data_dir / f"{symbol}.metadata.json"
    if csv_path.exists() and metadata_path.exists() and not refresh:
        rows = _read_rows(csv_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        rows = fetch_adjusted_ohlcv(symbol, effective_start, as_of)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        metadata = {
            "schema_version": "ml.timesfm.focused_ohlcv.v1",
            "symbol": symbol,
            "asset_type": policy.asset_type,
            "requested_start": start.isoformat(),
            "effective_start": effective_start.isoformat(),
            "as_of": as_of.isoformat(),
            "source": "yahoo-chart",
            "source_url_template": YAHOO_CHART_URL,
            "price_policy": ADJUSTED_PRICE_POLICY,
            "lineage_notes": list(policy.notes),
            "rows": len(rows),
            "first": rows[0]["timestamp"] if rows else None,
            "last": rows[-1]["timestamp"] if rows else None,
        }
        write_ohlcv_csv(csv_path, rows)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if len(rows) < policy.minimum_rows:
        raise RuntimeError(
            f"{symbol} has only {len(rows)} usable rows; policy requires {policy.minimum_rows}"
        )
    return {
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
        "rows": len(rows),
        "first": rows[0]["timestamp"],
        "last": rows[-1]["timestamp"],
        "metadata": metadata,
    }


def fetch_adjusted_ohlcv(symbol: str, start: date, end: date) -> list[dict[str, str]]:
    """Fetch daily OHLCV and convert raw OHLC to adjusted OHLC."""

    yahoo_symbol = symbol.replace(".", "-")
    period1 = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp())
    exclusive_end = end + timedelta(days=1)
    period2 = int(
        datetime(exclusive_end.year, exclusive_end.month, exclusive_end.day, tzinfo=UTC).timestamp()
    )
    url = (
        YAHOO_CHART_URL.format(symbol=quote(yahoo_symbol))
        + f"?period1={period1}&period2={period2}&interval=1d&events=history"
        + "&includeAdjustedClose=true"
    )
    payload = json.loads(_get_text(url))
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(str(error))
    result = chart.get("result") or []
    if not result:
        raise RuntimeError(f"Yahoo chart payload contained no result for {symbol}")
    first = result[0]
    timestamps = first.get("timestamp") or []
    quote_payload = (first.get("indicators", {}).get("quote") or [{}])[0]
    adjusted_payload = (first.get("indicators", {}).get("adjclose") or [{}])[0]
    adjusted = adjusted_payload.get("adjclose") or []
    rows: list[dict[str, str]] = []
    for index, timestamp in enumerate(timestamps):
        raw_open = _at(quote_payload.get("open"), index)
        raw_high = _at(quote_payload.get("high"), index)
        raw_low = _at(quote_payload.get("low"), index)
        raw_close = _at(quote_payload.get("close"), index)
        volume = _at(quote_payload.get("volume"), index)
        adjusted_close = _at(adjusted, index)
        if any(value is None for value in (raw_open, raw_high, raw_low, raw_close, volume)):
            continue
        close_decimal = Decimal(str(raw_close))
        adjusted_close_decimal = (
            Decimal(str(adjusted_close)) if adjusted_close is not None else close_decimal
        )
        if close_decimal <= 0:
            continue
        factor = adjusted_close_decimal / close_decimal
        rows.append(
            {
                "timestamp": datetime.fromtimestamp(timestamp, UTC).date().isoformat(),
                "open": _format_decimal(Decimal(str(raw_open)) * factor),
                "high": _format_decimal(Decimal(str(raw_high)) * factor),
                "low": _format_decimal(Decimal(str(raw_low)) * factor),
                "close": _format_decimal(adjusted_close_decimal),
                "volume": str(int(volume)),
                "adjusted_close": _format_decimal(adjusted_close_decimal),
            }
        )
    if not rows:
        raise RuntimeError(f"Yahoo chart payload contained no usable OHLCV rows for {symbol}")
    return rows


def write_ohlcv_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write TimesFM-compatible OHLCV CSV rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjusted_close",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_symbol_hpo(
    symbol: str,
    policy: TickerPolicy,
    *,
    data_record: dict[str, Any],
    output_root: Path,
    trials: Sequence[HpoTrial],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run all selected HPO trials for one symbol and promote the best result."""

    symbol_root = output_root / symbol
    runs_root = symbol_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    csv_path = Path(str(data_record["csv_path"]))
    row_count = int(data_record["rows"])
    trial_records: list[dict[str, Any]] = []
    choices: list[EvaluationChoice] = []
    selected_trials = select_trainable_trials(
        trials,
        row_count=row_count,
        max_trials=0 if args.max_trials_per_ticker == 0 else args.max_trials_per_ticker,
    )
    if args.dry_run:
        return {
            "symbol": symbol,
            "status": "dry_run",
            "policy": _policy_payload(policy),
            "data": data_record,
            "trials": [_trial_payload(trial) for trial in selected_trials],
        }
    if not selected_trials:
        return {
            "symbol": symbol,
            "status": "failed",
            "error": f"no HPO candidates fit {row_count} data rows",
            "policy": _policy_payload(policy),
            "data": data_record,
        }

    for trial_index, trial in enumerate(selected_trials, start=1):
        run_dir = runs_root / trial.run_id
        evaluation_path = run_dir / "evaluation.json"
        print(f"  {symbol} trial {trial_index:02d}/{len(selected_trials):02d}: {trial.run_id}")
        if evaluation_path.exists() and not args.refresh_runs:
            choice = choice_from_evaluation(symbol, trial, evaluation_path, run_dir)
            choices.append(choice)
            trial_records.append(_trial_record(choice, reused=True))
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        train_command = _train_command(args, csv_path, symbol, run_dir, trial)
        evaluate_command = _evaluate_command(
            args,
            csv_path,
            symbol,
            run_dir,
            evaluation_path,
            trial,
        )
        try:
            log_path = run_dir / "train.log"
            _run_logged(train_command, log_path)
            log_path = run_dir / "evaluate.log"
            _run_logged(evaluate_command, log_path)
            choice = choice_from_evaluation(symbol, trial, evaluation_path, run_dir)
            choices.append(choice)
            trial_records.append(_trial_record(choice, reused=False))
        except subprocess.CalledProcessError as exc:
            failed_stage = "train" if log_path.name == "train.log" else "evaluate"
            trial_records.append(
                {
                    "run_id": trial.run_id,
                    "status": "failed",
                    "returncode": exc.returncode,
                    "failed_stage": failed_stage,
                    "trial": _trial_payload(trial),
                    "log_path": str(log_path),
                }
            )
            print(f"  {symbol} trial {trial.run_id}: failed {exc.returncode}")
            if args.stop_on_first_error:
                raise
    if not choices:
        return {
            "symbol": symbol,
            "status": "failed",
            "error": "all HPO trials failed",
            "policy": _policy_payload(policy),
            "data": data_record,
            "trials": trial_records,
        }
    best = choose_best_evaluation(choices)
    best_dir = symbol_root / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best.run_dir, best_dir)
    print(
        f"  {symbol} best: {best.status} acc={best.directional_accuracy:.6f} "
        f"rmse_ratio={best.rmse_ratio:.4f}"
    )
    return {
        "symbol": symbol,
        "status": best.status,
        "suitable_for_scoring": best.suitable_for_scoring,
        "policy": _policy_payload(policy),
        "data": data_record,
        "best": {
            "run_id": best.trial.run_id,
            "evaluation": str(best.evaluation_path),
            "promoted_dir": str(best_dir),
            "directional_accuracy": best.directional_accuracy,
            "rmse": best.rmse,
            "best_baseline_rmse": best.best_baseline_rmse,
            "rmse_ratio": round(best.rmse_ratio, 8),
            "trial": _trial_payload(best.trial),
        },
        "trials": trial_records,
    }


def hpo_trials_from_args(args: argparse.Namespace) -> tuple[HpoTrial, ...]:
    """Build deterministic HPO trials from CLI lists."""

    return build_hpo_trials(
        context_lengths=_parse_int_list(args.context_lengths),
        horizon_lengths=_parse_int_list(args.horizon_lengths),
        max_steps_values=_parse_int_list(args.max_steps_grid),
        batch_sizes=_parse_int_list(args.batch_sizes),
        learning_rates=_parse_float_list(args.learning_rates),
        lora_ranks=_parse_int_list(args.lora_ranks),
        lora_dropouts=_parse_float_list(args.lora_dropouts),
        lora_alpha_multiplier=args.lora_alpha_multiplier,
    )


def build_hpo_trials(
    *,
    context_lengths: Sequence[int],
    horizon_lengths: Sequence[int],
    max_steps_values: Sequence[int],
    batch_sizes: Sequence[int],
    learning_rates: Sequence[float],
    lora_ranks: Sequence[int],
    lora_dropouts: Sequence[float],
    lora_alpha_multiplier: int,
) -> tuple[HpoTrial, ...]:
    """Build and rank HPO candidates with the strong first-pass recipe first."""

    trials = {
        HpoTrial(
            context_length=context_length,
            horizon_length=horizon_length,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_r * lora_alpha_multiplier,
            lora_dropout=lora_dropout,
        )
        for (
            context_length,
            horizon_length,
            max_steps,
            batch_size,
            learning_rate,
            lora_r,
            lora_dropout,
        ) in itertools.product(
            context_lengths,
            horizon_lengths,
            max_steps_values,
            batch_sizes,
            learning_rates,
            lora_ranks,
            lora_dropouts,
        )
    }
    return tuple(sorted(trials, key=_trial_priority))


def select_trainable_trials(
    trials: Sequence[HpoTrial],
    *,
    row_count: int,
    max_trials: int,
) -> tuple[HpoTrial, ...]:
    """Filter candidates that cannot fit train/validation/test windows."""

    trainable = tuple(trial for trial in trials if row_count >= minimum_rows_for_trial(trial))
    return trainable if max_trials == 0 else trainable[:max_trials]


def minimum_rows_for_trial(trial: HpoTrial) -> int:
    """Return a conservative minimum row count for purged train/validation/test splits."""

    return trial.context_length + (3 * trial.horizon_length) + 2


def choose_best_evaluation(choices: Sequence[EvaluationChoice]) -> EvaluationChoice:
    """Choose the best completed trial without tuning against report scoring."""

    if not choices:
        raise ValueError("at least one evaluation choice is required")
    return min(
        choices,
        key=lambda choice: (
            0 if choice.suitable_for_scoring else 1,
            0 if choice.status == "suitable" else 1,
            choice.rmse_ratio,
            -choice.directional_accuracy,
            choice.rmse,
        ),
    )


def choice_from_evaluation(
    symbol: str,
    trial: HpoTrial,
    evaluation_path: Path,
    run_dir: Path,
) -> EvaluationChoice:
    """Load a TimesFM evaluation artifact into an HPO comparison record."""

    artifact = json.loads(evaluation_path.read_text(encoding="utf-8"))
    baselines = artifact.get("baselines")
    if not isinstance(baselines, list) or not baselines:
        raise ValueError(f"evaluation artifact missing baselines: {evaluation_path}")
    baseline_rmses = [
        float(baseline["metrics"]["rmse"]) for baseline in baselines if isinstance(baseline, dict)
    ]
    if not baseline_rmses:
        raise ValueError(f"evaluation artifact contains no baseline RMSE: {evaluation_path}")
    metrics = artifact["metrics"]
    return EvaluationChoice(
        symbol=symbol,
        trial=trial,
        status=str(artifact["status"]),
        suitable_for_scoring=bool(artifact["suitable_for_scoring"]),
        directional_accuracy=float(metrics["directional_accuracy"]),
        rmse=float(metrics["rmse"]),
        best_baseline_rmse=min(baseline_rmses),
        evaluation_path=evaluation_path,
        run_dir=run_dir,
    )


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the focused HPO manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the focused HPO CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--as-of", default=DEFAULT_AS_OF.isoformat())
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-first-error", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-batches", type=int, default=9999)
    parser.add_argument("--max-latest-bar-age-days", type=int, default=5)
    parser.add_argument("--suitability-max-latest-bar-age-days", type=int, default=5)
    parser.add_argument("--min-evaluation-windows", type=int, default=3)
    parser.add_argument("--max-trials-per-ticker", type=int, default=24)
    parser.add_argument("--learning-rates", default="1e-5,3e-5,1e-4")
    parser.add_argument("--lora-ranks", default="4,8,16")
    parser.add_argument("--lora-alpha-multiplier", type=int, default=2)
    parser.add_argument("--lora-dropouts", default="0.05,0.10,0.15")
    parser.add_argument("--max-steps-grid", default="500,1000,2000")
    parser.add_argument("--batch-sizes", default="4,8,16")
    parser.add_argument("--context-lengths", default="64,128,256")
    parser.add_argument("--horizon-lengths", default="5,10,16,20")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the focused TimesFM HPO workflow."""

    return run_focused_hpo(build_parser().parse_args(argv))


def _train_command(
    args: argparse.Namespace,
    csv_path: Path,
    symbol: str,
    run_dir: Path,
    trial: HpoTrial,
) -> list[str]:
    return [
        args.python_executable,
        "-m",
        "nlp_stock_prediction.ml.timesfm.train",
        "--csv",
        str(csv_path),
        "--ticker",
        symbol,
        "--device",
        args.device,
        "--model-id",
        args.model_id,
        "--output-dir",
        str(run_dir),
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(trial.max_steps),
        "--batch-size",
        str(trial.batch_size),
        "--learning-rate",
        str(trial.learning_rate),
        "--validation-batches",
        str(args.validation_batches),
        "--lora-r",
        str(trial.lora_r),
        "--lora-alpha",
        str(trial.lora_alpha),
        "--lora-dropout",
        str(trial.lora_dropout),
        "--context-length",
        str(trial.context_length),
        "--horizon-length",
        str(trial.horizon_length),
        "--target-field",
        "adjusted_close",
        "--as-of",
        args.as_of,
        "--max-latest-bar-age-days",
        str(args.max_latest_bar_age_days),
    ]


def _evaluate_command(
    args: argparse.Namespace,
    csv_path: Path,
    symbol: str,
    run_dir: Path,
    evaluation_path: Path,
    trial: HpoTrial,
) -> list[str]:
    return [
        args.python_executable,
        "-m",
        "nlp_stock_prediction.ml.timesfm.evaluate",
        "--csv",
        str(csv_path),
        "--ticker",
        symbol,
        "--model-dir",
        str(run_dir),
        "--device",
        args.device,
        "--output",
        str(evaluation_path),
        "--context-length",
        str(trial.context_length),
        "--horizon-length",
        str(trial.horizon_length),
        "--target-field",
        "adjusted_close",
        "--as-of",
        args.as_of,
        "--max-latest-bar-age-days",
        str(args.max_latest_bar_age_days),
        "--suitability-max-latest-bar-age-days",
        str(args.suitability_max_latest_bar_age_days),
        "--min-evaluation-windows",
        str(args.min_evaluation_windows),
    ]


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, list(command))


def _trial_record(choice: EvaluationChoice, *, reused: bool) -> dict[str, Any]:
    return {
        "run_id": choice.trial.run_id,
        "status": choice.status,
        "suitable_for_scoring": choice.suitable_for_scoring,
        "directional_accuracy": choice.directional_accuracy,
        "rmse": choice.rmse,
        "best_baseline_rmse": choice.best_baseline_rmse,
        "rmse_ratio": round(choice.rmse_ratio, 8),
        "evaluation": str(choice.evaluation_path),
        "reused": reused,
        "trial": _trial_payload(choice.trial),
    }


def _trial_payload(trial: HpoTrial) -> dict[str, int | float | str]:
    return {
        "run_id": trial.run_id,
        "context_length": trial.context_length,
        "horizon_length": trial.horizon_length,
        "max_steps": trial.max_steps,
        "batch_size": trial.batch_size,
        "learning_rate": trial.learning_rate,
        "lora_r": trial.lora_r,
        "lora_alpha": trial.lora_alpha,
        "lora_dropout": trial.lora_dropout,
    }


def _policy_payload(policy: TickerPolicy) -> dict[str, Any]:
    return {
        "symbol": policy.symbol,
        "asset_type": policy.asset_type,
        "history_start_override": policy.history_start_override.isoformat()
        if policy.history_start_override
        else None,
        "minimum_rows": policy.minimum_rows,
        "notes": list(policy.notes),
    }


def _trial_priority(trial: HpoTrial) -> tuple[int, int, int, int, int, int, int]:
    return (
        _priority_index(trial.context_length, (128, 64, 256)),
        _priority_index(trial.horizon_length, (16, 10, 5, 20)),
        _priority_index(trial.max_steps, (1000, 500, 2000)),
        _priority_index(trial.batch_size, (8, 4, 16)),
        _priority_index_float(trial.learning_rate, (3e-5, 1e-5, 1e-4)),
        _priority_index(trial.lora_r, (8, 4, 16)),
        _priority_index_float(trial.lora_dropout, (0.10, 0.05, 0.15)),
    )


def _priority_index(value: int, preferred: Sequence[int]) -> int:
    return preferred.index(value) if value in preferred else len(preferred)


def _priority_index_float(value: float, preferred: Sequence[float]) -> int:
    for index, candidate in enumerate(preferred):
        if math.isclose(value, candidate, rel_tol=1e-12, abs_tol=1e-12):
            return index
    return len(preferred)


def _parse_symbols(raw: str) -> tuple[str, ...]:
    symbols = tuple(symbol.strip().upper().removeprefix("$") for symbol in raw.split(","))
    return tuple(symbol for symbol in symbols if symbol)


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("integer list cannot be empty")
    return values


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("float list cannot be empty")
    return values


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _get_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        data = response.read()
    if not isinstance(data, bytes):
        raise RuntimeError("HTTP response did not return bytes")
    return data.decode("utf-8")


def _at(values: Any, index: int) -> Any:
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _format_decimal(value: Decimal) -> str:
    return f"{value:.6f}"


def _float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
