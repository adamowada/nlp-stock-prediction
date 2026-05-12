"""Command line entry point for evaluating a local technical model artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nlp_stock_prediction.ml.dataset import TechnicalDatasetConfig, build_technical_dataset
from nlp_stock_prediction.ml.train import load_price_bars_csv
from nlp_stock_prediction.ml.training import evaluate_model, load_model_artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    model = load_model_artifact(Path(args.model))
    bars = load_price_bars_csv(Path(args.csv), ticker=args.ticker)
    dataset = build_technical_dataset(
        args.ticker,
        bars,
        config=TechnicalDatasetConfig(
            feature_window=model.feature_window,
            label_horizon_sessions=model.label_horizon_sessions,
            positive_return_threshold=model.positive_return_threshold,
        ),
    )
    result = evaluate_model(model, dataset.rows)
    payload = result.model_dump(mode="json")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_hash": result.model_hash,
                "samples": result.metrics.samples,
                "accuracy": result.metrics.accuracy,
                "log_loss": result.metrics.log_loss,
                "usage_limitations": result.usage_limitations,
            },
            sort_keys=True,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to model.json.")
    parser.add_argument("--csv", required=True, help="Local OHLCV CSV path.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol for the CSV rows.")
    parser.add_argument("--output", help="Optional evaluation JSON output path.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
