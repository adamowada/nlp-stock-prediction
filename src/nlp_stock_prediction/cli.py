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
from nlp_stock_prediction.environment import load_local_dotenv
from nlp_stock_prediction.pipeline import generate_daily_report

CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE = 3
PHASE_0_NOT_IMPLEMENTED_EXIT_CODE = CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
_CLI_EPILOG = """Examples:
  python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
  python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape
  python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ \\
    --capital 1000 --risk-profile exploratory --offline

Configuration:
  Offline runs are deterministic and do not use network providers.
  Scrape source mode uses public-provider adapters with deterministic fixtures by default.
  Add --live-providers with --source-mode scrape to call configured live providers.
  Add --ml-artifact to attach an evaluated local TimesFM technical-analysis sidecar.
  A local .env file is loaded automatically without overriding exported shell variables.
  Pass --offline to generate the deterministic fixture-backed report bundle.
  Keep provider credentials in environment variables or ignored local .env files;
  see docs/configuration.md.
"""


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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Generate a daily report.",
        description=(
            "Generate one Markdown report, one JSON report, and audit artifacts under "
            "<output>/<YYYY-MM-DD>/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    run_parser.add_argument(
        "--date",
        dest="run_date",
        required=True,
        type=_parse_date,
        help="Report date in YYYY-MM-DD format.",
    )
    run_parser.add_argument(
        "--output",
        dest="output_dir",
        required=True,
        type=Path,
        help="Base output directory; files are written under <output>/<YYYY-MM-DD>/.",
    )
    run_parser.add_argument(
        "--capital",
        dest="capital",
        type=_parse_decimal,
        help="Optional account capital for risk gates; must be a non-negative decimal.",
    )
    run_parser.add_argument(
        "--risk-profile",
        choices=tuple(profile.value for profile in RiskProfile),
        default=RiskProfile.EXPLORATORY.value,
        help="Risk profile for scoring and report context. Default: exploratory.",
    )
    run_parser.add_argument(
        "--fixture-dir",
        type=Path,
        help=(
            "Optional fixture root for future external fixtures; current offline runs record "
            "this path in command metadata and use built-in deterministic fixtures."
        ),
    )
    run_parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Optional provider cache directory for live provider runs; deterministic runs record "
            "this path in command metadata."
        ),
    )
    run_parser.add_argument(
        "--ml-artifact",
        type=Path,
        help=(
            "Optional evaluated local TimesFM artifact to attach as a technical-analysis sidecar; "
            "scoring still requires evidence/risk gates."
        ),
    )
    run_parser.add_argument(
        "--source-mode",
        choices=("offline", "scrape"),
        default=None,
        help=(
            "Explicit source mode. Use 'scrape' for the experimental public-provider "
            "provider path with deterministic fixtures by default; add --live-providers for "
            "real provider calls."
        ),
    )
    run_parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Use the deterministic offline fixture-backed report path; disallows live network "
            "providers."
        ),
    )
    run_parser.add_argument(
        "--live-providers",
        action="store_true",
        help=(
            "Opt into real provider calls for --source-mode scrape. Fixture-backed scrape mode "
            "remains the default."
        ),
    )
    return parser


def build_run_config(args: argparse.Namespace) -> RunConfig:
    source_mode = "offline" if args.offline else args.source_mode or "disabled"
    if args.live_providers and (args.offline or source_mode != "scrape"):
        raise ValueError(
            "--live-providers requires --source-mode scrape and cannot be used with --offline"
        )
    return RunConfig(
        run_date=args.run_date,
        output_dir=args.output_dir,
        capital=args.capital,
        risk_profile=RiskProfile(args.risk_profile),
        fixture_dir=args.fixture_dir,
        cache_dir=args.cache_dir,
        ml_artifact=args.ml_artifact,
        offline=args.offline,
        source_mode=source_mode,
        live_providers=args.live_providers,
    )


def main(argv: Sequence[str] | None = None) -> int:
    load_local_dotenv()
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
