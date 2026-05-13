"""CLI surface for prediction research reports."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.environment import load_local_dotenv
from nlp_stock_prediction.pipeline import generate_daily_report

CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE = 3
PHASE_0_NOT_IMPLEMENTED_EXIT_CODE = CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
_CLI_EPILOG = """Examples:
  python -m nlp_stock_prediction research --date 2026-05-12 --output reports/ --offline

Configuration:
  Offline runs are deterministic and do not use network providers.
  A local .env file is loaded automatically without overriding exported shell variables.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nlp_stock_prediction",
        description="Generate evidence-grounded prediction research artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    subparsers = parser.add_subparsers(dest="command")

    research_parser = subparsers.add_parser(
        "research",
        help="Generate a prediction research report.",
        description=(
            "Generate one Markdown report, one JSON report, and audit artifacts under "
            "<output>/<YYYY-MM-DD>/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    research_parser.add_argument(
        "--date",
        dest="run_date",
        required=True,
        type=_parse_date,
        help="Report date in YYYY-MM-DD format.",
    )
    research_parser.add_argument(
        "--output",
        dest="output_dir",
        required=True,
        type=Path,
        help="Base output directory; files are written under <output>/<YYYY-MM-DD>/.",
    )
    research_parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Optional fixture root recorded in command metadata.",
    )
    research_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional provider cache directory recorded in command metadata.",
    )
    research_parser.add_argument(
        "--offline",
        action="store_true",
        required=True,
        help="Use deterministic offline fixtures.",
    )
    return parser


def build_research_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        run_date=args.run_date,
        output_dir=args.output_dir,
        fixture_dir=args.fixture_dir,
        cache_dir=args.cache_dir,
        offline=args.offline,
        source_mode="offline" if args.offline else "disabled",
    )


def main(argv: Sequence[str] | None = None) -> int:
    load_local_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "research":
        try:
            bundle = generate_daily_report(build_research_config(args))
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
    "build_research_config",
    "main",
]
