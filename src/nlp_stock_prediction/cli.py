"""CLI surface for prediction research reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from rich.console import Console

from nlp_stock_prediction.contracts.providers import BatchRunConfig, RunConfig, WsbBatchRunConfig
from nlp_stock_prediction.environment import load_local_dotenv
from nlp_stock_prediction.evaluation.calibration import DEFAULT_CALIBRATION_BIN_EDGES
from nlp_stock_prediction.orchestration.evaluation_service import EvaluationService
from nlp_stock_prediction.pipeline import (
    generate_daily_report,
    generate_ranked_research_reports,
    generate_wsb_trending_research_reports,
)
from nlp_stock_prediction.terminal_ui import (
    print_batch_research_paths,
    print_research_paths,
    print_wsb_batch_research_paths,
    prompt_for_research_config,
    render_research_error,
    run_research_terminal,
)

CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE = 3
_CLI_EPILOG = """Examples:
  python -m nlp_stock_prediction research \\
    --date 2026-05-12 --symbol TSLA --output reports/ --offline
  python -m nlp_stock_prediction research-batch \\
    --date 2026-05-12 --symbols TSLA MSFT NVDA --output reports/ --offline
  python -m nlp_stock_prediction research-wsb-batch \\
    --date 2026-05-12 --output reports/ --live
  python -m nlp_stock_prediction research \\
    --date 2026-05-12 --symbol TSLA --output reports/ --live

Configuration:
  Offline runs are deterministic and do not use network providers.
  Live runs require --live and use configured live providers without fixture fallback.
  Batch runs fan out independent per-symbol reports with bounded concurrency, then write a
  research-viability ranking artifact. The ranking is not a trading instruction.
  The WSB batch workflow discovers public r/wallstreetbets ticker mentions first, then runs
  the same batch analysis over the top discovered symbols.
  A local .env file is loaded automatically without overriding exported shell variables.
  Keep provider credentials in environment variables or ignored local .env files;
  see docs/configuration.md.
"""
_EVALUATION_EPILOG = """Examples:
  python -m nlp_stock_prediction evaluation --database data/prediction-research.sqlite3 \\
    inspect --run-id research-msft-2026-05-14
  python -m nlp_stock_prediction evaluation --database data/prediction-research.sqlite3 \\
    calibration --run-id research-msft-2026-05-14 --cohort-id msft-swing \\
    --as-of 2026-05-22T00:00:00+00:00 --artifact-root reports/research-msft-2026-05-14/audit

Evaluation commands read an explicit SQLite run database and require --run-id.
Commands that write audit artifacts require --artifact-root and use the repository write policy.
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
            "<output>/<YYYY-MM-DD>/<symbol>/."
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
        help="Base output directory; files are written under <output>/<YYYY-MM-DD>/<symbol>/.",
    )
    research_parser.add_argument(
        "--symbol",
        default="TSLA",
        help="Instrument symbol or pair to research, for example TSLA or BTC-USD.",
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
    mode_group = research_parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic offline fixtures.",
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Use live providers and public-source adapters without fixture fallback.",
    )
    batch_parser = subparsers.add_parser(
        "research-batch",
        help="Generate and rank prediction research reports for multiple symbols.",
        description=(
            "Generate per-symbol Markdown/JSON reports concurrently, then write aggregate "
            "research-viability ranking artifacts under <output>/<YYYY-MM-DD>/batch/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    batch_parser.add_argument(
        "--date",
        dest="run_date",
        required=True,
        type=_parse_date,
        help="Report date in YYYY-MM-DD format.",
    )
    batch_parser.add_argument(
        "--output",
        dest="output_dir",
        required=True,
        type=Path,
        help="Base output directory; per-symbol reports and batch ranking are written below it.",
    )
    batch_parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Instrument symbols to research; accepts repeated values or comma-separated lists.",
    )
    batch_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent per-symbol research runs, from 1 to 16.",
    )
    batch_parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Optional fixture root recorded in command metadata.",
    )
    batch_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional provider cache directory recorded in command metadata.",
    )
    batch_mode_group = batch_parser.add_mutually_exclusive_group(required=True)
    batch_mode_group.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic offline fixtures.",
    )
    batch_mode_group.add_argument(
        "--live",
        action="store_true",
        help="Use live providers and public-source adapters without fixture fallback.",
    )
    wsb_parser = subparsers.add_parser(
        "research-wsb-batch",
        help="Discover WSB-mentioned symbols and batch-rank research viability.",
        description=(
            "Discover the most-mentioned public r/wallstreetbets symbols, write a discovery "
            "artifact, then generate per-symbol reports and a batch viability ranking."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    wsb_parser.add_argument(
        "--date",
        dest="run_date",
        required=True,
        type=_parse_date,
        help="Report date in YYYY-MM-DD format.",
    )
    wsb_parser.add_argument(
        "--output",
        dest="output_dir",
        required=True,
        type=Path,
        help="Base output directory for discovery, per-symbol reports, and ranking artifacts.",
    )
    wsb_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of WSB-mentioned symbols to batch analyze. Defaults to 10.",
    )
    wsb_parser.add_argument(
        "--max-discussion-pages",
        type=int,
        default=10,
        help="Maximum public WSB discussion pages to expand during discovery.",
    )
    wsb_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent per-symbol research runs, from 1 to 16.",
    )
    wsb_parser.add_argument(
        "--source-url",
        default="https://old.reddit.com/r/wallstreetbets/",
        help="Public r/wallstreetbets HTML page to inspect for discovery.",
    )
    wsb_parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Optional fixture root recorded in command metadata.",
    )
    wsb_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional provider cache directory recorded in command metadata.",
    )
    wsb_mode_group = wsb_parser.add_mutually_exclusive_group(required=True)
    wsb_mode_group.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic offline fixtures.",
    )
    wsb_mode_group.add_argument(
        "--live",
        action="store_true",
        help="Use live public Reddit/provider adapters without fixture fallback.",
    )
    tui_parser = subparsers.add_parser(
        "tui",
        help="Launch the Rich terminal UI for guided report generation.",
        description=(
            "Launch a Rich-styled terminal workflow for generating a research report. "
            "Provide options for a non-interactive run, or omit them in an interactive terminal "
            "to be prompted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    tui_parser.add_argument(
        "--date",
        dest="run_date",
        type=_parse_date,
        help="Report date in YYYY-MM-DD format. Prompted when omitted in an interactive terminal.",
    )
    tui_parser.add_argument(
        "--output",
        dest="output_dir",
        type=Path,
        help="Base output directory. Prompted when omitted in an interactive terminal.",
    )
    tui_parser.add_argument(
        "--symbol",
        help="Instrument symbol or pair to research, for example TSLA or BTC-USD.",
    )
    tui_parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Optional fixture root recorded in command metadata.",
    )
    tui_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional provider cache directory recorded in command metadata.",
    )
    tui_mode_group = tui_parser.add_mutually_exclusive_group()
    tui_mode_group.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic offline fixtures.",
    )
    tui_mode_group.add_argument(
        "--live",
        action="store_true",
        help="Use live providers and public-source adapters without fixture fallback.",
    )
    _add_evaluation_parser(subparsers)
    subparsers.add_parser(
        "app",
        help="Launch the persistent terminal app.",
        description="Launch the menu-driven research assistant app.",
    )
    return parser


def _add_evaluation_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    evaluation_parser = subparsers.add_parser(
        "evaluation",
        help="Inspect and harden persisted prediction evaluations.",
        description=("Run public evaluation workflows over an existing research SQLite database."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EVALUATION_EPILOG,
    )
    evaluation_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for write-policy checks. Defaults to the current directory.",
    )
    evaluation_parser.add_argument(
        "--database",
        required=True,
        type=Path,
        help="Existing research SQLite database path, relative to --repo-root unless absolute.",
    )
    evaluation_subparsers = evaluation_parser.add_subparsers(
        dest="evaluation_command",
        required=True,
    )

    inspect_parser = evaluation_subparsers.add_parser(
        "inspect",
        help="Inspect persisted evaluation counts for one run.",
    )
    _add_run_id(inspect_parser)

    materialize_parser = evaluation_subparsers.add_parser(
        "materialize-outcome",
        help="Materialize one outcome from real post-window market data.",
    )
    _add_run_id(materialize_parser)
    materialize_parser.add_argument("--candidate-id", required=True)
    materialize_parser.add_argument("--point-in-time-cutoff", required=True)
    materialize_parser.add_argument("--evaluation-window-start", required=True)
    materialize_parser.add_argument("--evaluation-window-end", required=True)
    _add_artifact_root(materialize_parser)
    materialize_parser.add_argument("--report-date")
    materialize_parser.add_argument("--market-artifact-id", action="append", default=[])
    materialize_parser.add_argument("--created-at")
    materialize_parser.add_argument("--evaluated-at")

    load_parser = evaluation_subparsers.add_parser(
        "load-outcomes",
        help="Load persisted outcome-evaluation artifacts for one run.",
    )
    _add_run_id(load_parser)

    outcome_summary_parser = evaluation_subparsers.add_parser(
        "outcome-summary",
        help="Write outcome review summary artifacts for one run.",
    )
    _add_run_id(outcome_summary_parser)
    _add_artifact_root(outcome_summary_parser)
    outcome_summary_parser.add_argument("--created-at")

    stale_artifacts_parser = evaluation_subparsers.add_parser(
        "stale-artifacts",
        help="Write artifact freshness reviews for one run.",
    )
    _add_run_id(stale_artifacts_parser)
    _add_artifact_root(stale_artifacts_parser)
    stale_artifacts_parser.add_argument("--reviewed-at")

    evidence_aging_parser = evaluation_subparsers.add_parser(
        "evidence-aging",
        help="Write evidence aging reviews for one run.",
    )
    _add_run_id(evidence_aging_parser)
    _add_artifact_root(evidence_aging_parser)
    evidence_aging_parser.add_argument("--reviewed-at")

    source_reliability_parser = evaluation_subparsers.add_parser(
        "source-reliability",
        help="Write source reliability notes for stored live evidence.",
    )
    _add_run_id(source_reliability_parser)
    _add_artifact_root(source_reliability_parser)
    source_reliability_parser.add_argument("--created-at")

    provider_playbook_parser = evaluation_subparsers.add_parser(
        "provider-playbook",
        help="Write provider replacement playbooks.",
    )
    _add_run_id(provider_playbook_parser)
    _add_artifact_root(provider_playbook_parser)
    provider_playbook_parser.add_argument("--created-at")

    calibration_parser = evaluation_subparsers.add_parser(
        "calibration",
        help="Write calibration reliability bins from stored outcomes.",
    )
    _add_run_id(calibration_parser)
    _add_artifact_root(calibration_parser)
    calibration_parser.add_argument("--cohort-id", required=True)
    calibration_parser.add_argument("--as-of", required=True)
    calibration_parser.add_argument("--bin-edge", dest="bin_edges", action="append", type=float)
    calibration_parser.add_argument("--family", dest="families", action="append")
    calibration_parser.add_argument("--prediction-type")
    calibration_parser.add_argument("--horizon")

    walk_forward_parser = evaluation_subparsers.add_parser(
        "walk-forward",
        help="Write chronological walk-forward folds from stored outcomes.",
    )
    _add_run_id(walk_forward_parser)
    _add_artifact_root(walk_forward_parser)
    walk_forward_parser.add_argument("--cohort-id", required=True)
    walk_forward_parser.add_argument("--point-in-time-cutoff", required=True)
    walk_forward_parser.add_argument("--minimum-train-size", required=True, type=int)
    walk_forward_parser.add_argument("--test-size", type=int, default=1)
    walk_forward_parser.add_argument("--step-size", type=int, default=1)
    walk_forward_parser.add_argument("--prediction-type")
    walk_forward_parser.add_argument("--horizon")

    ablation_parser = evaluation_subparsers.add_parser(
        "ablation",
        help="Write signal-family ablation slices from stored outcomes.",
    )
    _add_run_id(ablation_parser)
    _add_artifact_root(ablation_parser)
    ablation_parser.add_argument("--cohort-id", required=True)
    ablation_parser.add_argument("--point-in-time-cutoff", required=True)
    ablation_parser.add_argument("--family", dest="families", action="append")
    ablation_parser.add_argument("--prediction-type")
    ablation_parser.add_argument("--horizon")

    drift_parser = evaluation_subparsers.add_parser(
        "calibration-drift",
        help="Write a calibration drift check comparing two calibration summaries.",
    )
    _add_run_id(drift_parser)
    _add_artifact_root(drift_parser)
    drift_parser.add_argument("--prior-calibration-id", required=True)
    drift_parser.add_argument("--current-calibration-id", required=True)
    drift_parser.add_argument("--as-of", required=True)
    drift_parser.add_argument("--signal-family")
    drift_parser.add_argument("--min-resolved-count", type=int, default=10)
    drift_parser.add_argument("--watch-delta", type=float, default=0.05)
    drift_parser.add_argument("--degraded-delta", type=float, default=0.10)
    drift_parser.add_argument("--improved-delta", type=float, default=0.10)


def _add_run_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True, help="Research run ID to evaluate.")


def _add_artifact_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact-root",
        required=True,
        type=Path,
        help="Audit artifact root to write under; checked against the repository write policy.",
    )


def build_research_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        run_date=args.run_date,
        output_dir=args.output_dir,
        symbol=args.symbol,
        fixture_dir=args.fixture_dir,
        cache_dir=args.cache_dir,
        offline=args.offline,
        source_mode="offline" if args.offline else "live",
        live_providers=args.live,
    )


def _parse_symbol_values(values: Sequence[str]) -> tuple[str, ...]:
    symbols: list[str] = []
    for value in values:
        symbols.extend(part.strip() for part in value.split(",") if part.strip())
    if not symbols:
        raise ValueError("--symbols requires at least one non-empty symbol")
    return tuple(symbols)


def build_batch_research_config(args: argparse.Namespace) -> BatchRunConfig:
    return BatchRunConfig(
        run_date=args.run_date,
        output_dir=args.output_dir,
        symbols=_parse_symbol_values(args.symbols),
        fixture_dir=args.fixture_dir,
        cache_dir=args.cache_dir,
        offline=args.offline,
        source_mode="offline" if args.offline else "live",
        live_providers=args.live,
        max_workers=args.max_workers,
    )


def build_wsb_batch_research_config(args: argparse.Namespace) -> WsbBatchRunConfig:
    return WsbBatchRunConfig(
        run_date=args.run_date,
        output_dir=args.output_dir,
        fixture_dir=args.fixture_dir,
        cache_dir=args.cache_dir,
        offline=args.offline,
        source_mode="offline" if args.offline else "live",
        live_providers=args.live,
        max_workers=args.max_workers,
        limit=args.limit,
        max_discussion_pages=args.max_discussion_pages,
        source_url=args.source_url,
    )


def build_tui_research_config(args: argparse.Namespace) -> RunConfig:
    return prompt_for_research_config(
        run_date=args.run_date,
        output_dir=args.output_dir,
        symbol=args.symbol,
        fixture_dir=args.fixture_dir,
        cache_dir=args.cache_dir,
        offline=args.offline,
        live=args.live,
    )


def build_evaluation_service(args: argparse.Namespace) -> EvaluationService:
    repo_root = args.repo_root.resolve()
    database_path = args.database if args.database.is_absolute() else repo_root / args.database
    if not database_path.exists():
        raise ValueError(
            "--database must reference an existing research SQLite database; "
            f"not found: {database_path}"
        )
    return EvaluationService(repo_root=repo_root, database_path=database_path)


def run_evaluation_command(args: argparse.Namespace) -> int:
    service = build_evaluation_service(args)
    command = args.evaluation_command
    if command == "inspect":
        result = service.evaluation_inspect(run_id=args.run_id)
    elif command == "materialize-outcome":
        result = service.evaluation_materialize_outcome(
            run_id=args.run_id,
            candidate_id=args.candidate_id,
            point_in_time_cutoff=args.point_in_time_cutoff,
            evaluation_window_start=args.evaluation_window_start,
            evaluation_window_end=args.evaluation_window_end,
            artifact_dir=args.artifact_root.as_posix(),
            report_date=args.report_date,
            market_artifact_ids=tuple(args.market_artifact_id),
            created_at=args.created_at,
            evaluated_at=args.evaluated_at,
        )
    elif command == "load-outcomes":
        result = service.evaluation_load_outcomes(run_id=args.run_id)
    elif command == "outcome-summary":
        result = service.evaluation_outcome_summary(
            run_id=args.run_id,
            artifact_dir=args.artifact_root.as_posix(),
            created_at=args.created_at,
        )
    elif command == "stale-artifacts":
        result = service.evaluation_stale_artifacts(
            run_id=args.run_id,
            artifact_dir=args.artifact_root.as_posix(),
            reviewed_at=args.reviewed_at,
        )
    elif command == "evidence-aging":
        result = service.evaluation_evidence_aging(
            run_id=args.run_id,
            artifact_dir=args.artifact_root.as_posix(),
            reviewed_at=args.reviewed_at,
        )
    elif command == "source-reliability":
        result = service.evaluation_source_reliability(
            run_id=args.run_id,
            artifact_dir=args.artifact_root.as_posix(),
            created_at=args.created_at,
        )
    elif command == "provider-playbook":
        result = service.evaluation_provider_playbook(
            run_id=args.run_id,
            artifact_dir=args.artifact_root.as_posix(),
            created_at=args.created_at,
        )
    elif command == "calibration":
        result = service.evaluation_calibration(
            run_id=args.run_id,
            cohort_id=args.cohort_id,
            as_of=args.as_of,
            artifact_dir=args.artifact_root.as_posix(),
            bin_edges=tuple(args.bin_edges) if args.bin_edges else DEFAULT_CALIBRATION_BIN_EDGES,
            families=args.families,
            prediction_type=args.prediction_type,
            horizon=args.horizon,
        )
    elif command == "walk-forward":
        result = service.evaluation_walk_forward(
            run_id=args.run_id,
            cohort_id=args.cohort_id,
            point_in_time_cutoff=args.point_in_time_cutoff,
            minimum_train_size=args.minimum_train_size,
            artifact_dir=args.artifact_root.as_posix(),
            test_size=args.test_size,
            step_size=args.step_size,
            prediction_type=args.prediction_type,
            horizon=args.horizon,
        )
    elif command == "ablation":
        result = service.evaluation_ablation(
            run_id=args.run_id,
            cohort_id=args.cohort_id,
            point_in_time_cutoff=args.point_in_time_cutoff,
            artifact_dir=args.artifact_root.as_posix(),
            families=args.families,
            prediction_type=args.prediction_type,
            horizon=args.horizon,
        )
    elif command == "calibration-drift":
        result = service.evaluation_calibration_drift(
            run_id=args.run_id,
            prior_calibration_id=args.prior_calibration_id,
            current_calibration_id=args.current_calibration_id,
            as_of=args.as_of,
            artifact_dir=args.artifact_root.as_posix(),
            signal_family=args.signal_family,
            min_resolved_count=args.min_resolved_count,
            watch_delta=args.watch_delta,
            degraded_delta=args.degraded_delta,
            improved_delta=args.improved_delta,
        )
    else:  # pragma: no cover - argparse constrains the command set.
        raise ValueError(f"unknown evaluation command: {command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_local_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "research-batch":
        try:
            batch_bundle = generate_ranked_research_reports(build_batch_research_config(args))
            print_batch_research_paths(batch_bundle)
        except ValueError as exc:
            render_research_error(str(exc), console=Console(stderr=True))
            return CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
        return 0
    if args.command == "research-wsb-batch":
        try:
            wsb_bundle = generate_wsb_trending_research_reports(
                build_wsb_batch_research_config(args)
            )
            print_wsb_batch_research_paths(wsb_bundle)
        except ValueError as exc:
            render_research_error(str(exc), console=Console(stderr=True))
            return CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
        return 0
    if args.command == "research":
        console = Console()
        try:
            config = build_research_config(args)
            if console.is_interactive:
                run_research_terminal(
                    config,
                    report_generator=generate_daily_report,
                    console=console,
                )
            else:
                research_bundle = generate_daily_report(config)
                print_research_paths(research_bundle)
        except ValueError as exc:
            render_research_error(str(exc), console=Console(stderr=True))
            return CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
        return 0
    if args.command == "tui":
        try:
            run_research_terminal(
                build_tui_research_config(args),
                report_generator=generate_daily_report,
            )
        except ValueError as exc:
            render_research_error(str(exc), console=Console(stderr=True))
            return CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
        return 0
    if args.command == "evaluation":
        try:
            return run_evaluation_command(args)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
    if args.command == "app":
        from nlp_stock_prediction.app import run_app

        return run_app()
    parser.error(f"unknown command: {args.command}")


__all__ = [
    "CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE",
    "build_batch_research_config",
    "build_evaluation_service",
    "build_parser",
    "build_research_config",
    "build_tui_research_config",
    "build_wsb_batch_research_config",
    "main",
    "run_evaluation_command",
]
