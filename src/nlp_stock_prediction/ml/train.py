"""Command line entry point for local technical-model training."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from nlp_stock_prediction.contracts import PriceBar
from nlp_stock_prediction.ml.dataset import TechnicalDatasetConfig, build_technical_dataset
from nlp_stock_prediction.ml.training import (
    TrainingConfig,
    train_technical_model,
    write_training_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    bars = load_price_bars_csv(csv_path, ticker=args.ticker)
    dataset = build_technical_dataset(
        args.ticker,
        bars,
        config=TechnicalDatasetConfig(
            feature_window=args.feature_window,
            label_horizon_sessions=args.label_horizon,
            positive_return_threshold=args.positive_return_threshold,
        ),
    )
    result = train_technical_model(
        dataset,
        config=TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            train_fraction=args.train_fraction,
            requested_device=args.device,
        ),
    )
    paths = write_training_artifacts(result, output_dir)
    print(
        json.dumps(
            {
                "model_path": str(paths.model_path),
                "metrics_path": str(paths.metrics_path),
                "metadata_path": str(paths.metadata_path),
                "dataset_hash": result.dataset_hash,
                "model_hash": result.model.model_hash,
                "selected_device": result.device.selected_device,
                "validation_accuracy": result.validation_metrics.accuracy,
                "usage_limitations": result.usage_limitations,
            },
            sort_keys=True,
        )
    )
    return 0


def load_price_bars_csv(path: Path, *, ticker: str) -> tuple[PriceBar, ...]:
    """Load local OHLCV CSV rows for training.

    Required columns: ``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``.
    Optional column: ``adjusted_close``.
    """

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header row")
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(reader.fieldnames)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"CSV missing required columns: {missing_columns}")
        return tuple(_row_to_bar(row, ticker=ticker) for row in reader)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Local OHLCV CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory for model artifacts.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol for the CSV rows.")
    parser.add_argument("--feature-window", type=int, default=5)
    parser.add_argument("--label-horizon", type=int, default=1)
    parser.add_argument("--positive-return-threshold", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Record CUDA metadata when available; default tests do not require CUDA.",
    )
    return parser


def _row_to_bar(row: dict[str, str | None], *, ticker: str) -> PriceBar:
    adjusted_close_cell = row.get("adjusted_close")
    adjusted_close = (
        Decimal(adjusted_close_cell)
        if adjusted_close_cell is not None and adjusted_close_cell.strip()
        else None
    )
    return PriceBar(
        ticker=ticker,
        timestamp=_parse_timestamp(_require_cell(row, "timestamp")),
        open=Decimal(_require_cell(row, "open")),
        high=Decimal(_require_cell(row, "high")),
        low=Decimal(_require_cell(row, "low")),
        close=Decimal(_require_cell(row, "close")),
        volume=int(_require_cell(row, "volume")),
        adjusted_close=adjusted_close,
    )


def _require_cell(row: dict[str, str | None], column: str) -> str:
    value = row.get(column)
    if value is None or not value.strip():
        raise ValueError(f"CSV row missing {column}")
    return value.strip()


def _parse_timestamp(value: str) -> date | datetime:
    if "T" in value:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("datetime CSV timestamps must include a timezone")
        return parsed
    return date.fromisoformat(value)


if __name__ == "__main__":
    raise SystemExit(main())
