"""Run the opt-in Research Stage real Codex MCP smoke test."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from nlp_stock_prediction.orchestration.codex_smoke import (
    ALLOWED_SMOKE_WRITE_ROOTS,
    MAX_HASHED_RESTRICTED_FILE_BYTES,
    RESTRICTED_SCAN_EXCLUDED_DIRS,
    RUN_CODEX_SMOKE_ENV,
    CodexSmokeConfig,
    RestrictedPathSnapshot,
    build_codex_command,
    build_codex_prompt,
    ensure_restricted_paths_unchanged,
    ensure_tracked_status_unchanged,
    expected_run_id,
    prepare_clean_database,
    prepare_clean_run_dir,
    require_clean_tracked_status,
    resolve_repo_path,
    run_codex_smoke,
    snapshot_restricted_paths,
    verify_smoke_outputs,
    verify_sqlite_run,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", dest="run_date", required=True, type=date.fromisoformat)
    parser.add_argument("--output", dest="output_dir", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--codex", dest="codex_executable", default="codex")
    parser.add_argument(
        "--python", dest="python_executable", type=Path, default=Path(sys.executable)
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = CodexSmokeConfig(
        run_date=args.run_date,
        output_dir=args.output_dir,
        symbol=args.symbol,
        repo_root=args.repo_root.resolve(),
        python_executable=args.python_executable.resolve(),
        codex_executable=args.codex_executable,
    )
    try:
        run_codex_smoke(config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Codex smoke report: {config.run_dir / 'report.md'}")
    print(f"Codex smoke JSON: {config.run_dir / 'report.json'}")
    print(f"Codex smoke final message: {config.final_message_path}")
    return 0


__all__ = [
    "ALLOWED_SMOKE_WRITE_ROOTS",
    "MAX_HASHED_RESTRICTED_FILE_BYTES",
    "RESTRICTED_SCAN_EXCLUDED_DIRS",
    "RUN_CODEX_SMOKE_ENV",
    "CodexSmokeConfig",
    "RestrictedPathSnapshot",
    "build_codex_command",
    "build_codex_prompt",
    "ensure_restricted_paths_unchanged",
    "ensure_tracked_status_unchanged",
    "expected_run_id",
    "main",
    "parse_args",
    "prepare_clean_database",
    "prepare_clean_run_dir",
    "require_clean_tracked_status",
    "resolve_repo_path",
    "run_codex_smoke",
    "snapshot_restricted_paths",
    "verify_smoke_outputs",
    "verify_sqlite_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
