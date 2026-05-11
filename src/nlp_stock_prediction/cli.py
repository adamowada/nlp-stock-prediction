"""CLI surface for daily report generation."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from nlp_stock_prediction.contracts.enums import RiskProfile
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.pipeline import generate_daily_report

CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE = 3
PHASE_0_NOT_IMPLEMENTED_EXIT_CODE = CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE


def _parse_date(value: str) -> date:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _parse_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("expected a decimal number") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("capital must be a finite decimal number")
    if parsed < 0:
        raise argparse.ArgumentTypeError("capital must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nlp_stock_prediction",
        description="Generate an evidence-grounded daily stock opportunity report.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Generate a daily report.")
    run_parser.add_argument("--date", dest="run_date", required=True, type=_parse_date)
    run_parser.add_argument("--output", dest="output_dir", required=True, type=Path)
    run_parser.add_argument("--capital", dest="capital", type=_parse_decimal)
    run_parser.add_argument(
        "--risk-profile",
        choices=tuple(profile.value for profile in RiskProfile),
        default=RiskProfile.EXPLORATORY.value,
    )
    run_parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Read deterministic provider fixtures instead of live providers.",
    )
    run_parser.add_argument("--cache-dir", type=Path, help="Read/write provider response cache.")
    run_parser.add_argument(
        "--offline",
        action="store_true",
        help="Disallow live network providers; intended for deterministic runs.",
    )
    return parser


def build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        run_date=args.run_date,
        output_dir=args.output_dir,
        capital=args.capital,
        risk_profile=RiskProfile(args.risk_profile),
        fixture_dir=args.fixture_dir,
        cache_dir=args.cache_dir,
        offline=args.offline,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "run":
        try:
            bundle = generate_daily_report(build_run_config(args))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
        print(f"Wrote Markdown report: {bundle.markdown_path}")
        print(f"Wrote JSON report: {bundle.json_path}")
        print(f"Wrote audit artifacts: {bundle.audit_dir}")
        return 0
    parser.error(f"unknown command: {args.command}")


__all__ = [
    "CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE",
    "PHASE_0_NOT_IMPLEMENTED_EXIT_CODE",
    "build_parser",
    "build_run_config",
    "main",
]
