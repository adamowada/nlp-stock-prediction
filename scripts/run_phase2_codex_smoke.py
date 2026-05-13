"""Run the opt-in Phase 2 real Codex MCP smoke test."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

RUN_CODEX_SMOKE_ENV = "NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE"


@dataclass(frozen=True)
class CodexSmokeConfig:
    run_date: date
    output_dir: Path
    symbol: str
    repo_root: Path
    python_executable: Path
    codex_executable: str = "codex"

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_date.isoformat()

    @property
    def final_message_path(self) -> Path:
        return self.run_dir / "codex-final.md"

    @property
    def database_arg_path(self) -> Path:
        normalized_symbol = self.symbol.lower().replace("/", "-")
        filename = f"phase2-codex-smoke-{self.run_date.isoformat()}-{normalized_symbol}.sqlite3"
        return Path("data") / filename

    @property
    def database_path(self) -> Path:
        return _resolve_repo_path(self, self.database_arg_path)


def build_codex_prompt(config: CodexSmokeConfig) -> str:
    """Return the instruction payload for the real Codex smoke run."""

    return "\n".join(
        [
            "You are smoke-testing Phase 2 of nlp-stock-prediction.",
            "Use live web search for 2-3 current sources about the requested symbol.",
            "Use the nlp-stock-prediction MCP tools; do not edit source files.",
            "Do not import project modules directly or run shell fallbacks for MCP tools.",
            "If an MCP tool call is unavailable or cancelled, stop and report smoke failure.",
            f"Run date: {config.run_date.isoformat()}",
            f"Symbol: {config.symbol.upper()}",
            f"Output directory: {config.output_dir.as_posix()}",
            "",
            "Required tool workflow:",
            "1. start_research_run",
            "2. list_research_tool_plan",
            "3. record_codex_search_evidence for each searched source",
            "4. run_dummy_universe_tool",
            "5. run_dummy_analysis_tool",
            "6. synthesize_prediction_candidates",
            "7. render_prediction_report",
            "8. inspect_research_run",
            "",
            "The final response must summarize the report path, JSON path, audit manifest path, "
            "and whether at least one live-search evidence item was recorded.",
        ]
    )


def build_codex_command(config: CodexSmokeConfig) -> list[str]:
    """Build the Codex CLI invocation without launching it."""

    mcp_command = _toml_string(str(config.python_executable))
    mcp_args = json.dumps(
        [
            "-m",
            "nlp_stock_prediction.codex_mcp",
            "--database",
            config.database_arg_path.as_posix(),
        ]
    )
    prompt = build_codex_prompt(config)
    return [
        config.codex_executable,
        "--ask-for-approval",
        "never",
        "--search",
        "-c",
        f"mcp_servers.nlp-stock-prediction.command={mcp_command}",
        "-c",
        f"mcp_servers.nlp-stock-prediction.args={mcp_args}",
        "exec",
        "-s",
        "danger-full-access",
        "-C",
        str(config.repo_root),
        "--output-last-message",
        str(config.final_message_path),
        prompt,
    ]


def verify_smoke_outputs(config: CodexSmokeConfig, *, require_sqlite: bool = True) -> None:
    """Validate the files and minimum JSON shape expected from the smoke run."""

    run_dir = _resolve_repo_path(config, config.run_dir)
    final_message_path = _resolve_repo_path(config, config.final_message_path)
    report_path = run_dir / "report.md"
    json_path = run_dir / "report.json"
    audit_manifest_path = run_dir / "audit" / "audit-manifest.json"
    for path in (report_path, json_path, audit_manifest_path, final_message_path):
        if not path.exists():
            raise RuntimeError(f"Codex smoke did not create expected file: {path}")
    final_message = final_message_path.read_text(encoding="utf-8").lower()
    failure_markers = (
        "mcp tool call was cancelled",
        "user cancelled mcp",
        "not produced",
        "shell fallback",
        "fastmcp stdio transport was blocked",
        "phase2mcpservice",
    )
    if any(marker in final_message for marker in failure_markers):
        raise RuntimeError("Codex smoke final message reported MCP fallback or failure.")

    report_payload = json.loads(json_path.read_text(encoding="utf-8"))
    evidence = report_payload.get("evidence_sources")
    if not isinstance(evidence, list) or not evidence:
        raise RuntimeError("Codex smoke report did not include evidence_sources.")
    if not any(_is_codex_search_evidence(item) for item in evidence):
        raise RuntimeError("Codex smoke report did not include Codex live-search evidence.")

    candidates = report_payload.get("prediction_candidates")
    insufficient = report_payload.get("insufficient_evidence_summary")
    if not candidates and not insufficient:
        raise RuntimeError(
            "Codex smoke report must include prediction_candidates or insufficient evidence."
        )
    if require_sqlite:
        _verify_sqlite_run(config)


def require_clean_tracked_status(repo_root: Path) -> str:
    """Return current tracked status, failing if the smoke starts from a dirty tree."""

    status = _git_status(repo_root)
    if status.strip():
        raise RuntimeError(
            "Codex smoke must start from a clean tracked tree; commit or stash changes first.\n"
            f"{status}"
        )
    return status


def ensure_tracked_status_unchanged(repo_root: Path, expected: str) -> None:
    status = _git_status(repo_root)
    if status != expected:
        raise RuntimeError(
            "Codex smoke changed tracked files. Review before continuing.\n"
            f"Before:\n{expected or '<clean>'}\nAfter:\n{status or '<clean>'}"
        )


def run_codex_smoke(config: CodexSmokeConfig) -> None:
    if os.environ.get(RUN_CODEX_SMOKE_ENV) != "1":
        raise RuntimeError(f"Set {RUN_CODEX_SMOKE_ENV}=1 to run the real Codex smoke test.")
    if shutil.which(config.codex_executable) is None:
        raise RuntimeError("Codex CLI is not available on PATH.")

    _prepare_clean_run_dir(config)
    _prepare_clean_database(config)
    baseline_status = require_clean_tracked_status(config.repo_root)
    completed = subprocess.run(
        build_codex_command(config),
        cwd=config.repo_root,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Codex smoke failed with exit code {completed.returncode}.")
    verify_smoke_outputs(config)
    ensure_tracked_status_unchanged(config.repo_root, baseline_status)


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


def _is_codex_search_evidence(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    provenance = item.get("provenance")
    metadata = item.get("metadata")
    if isinstance(metadata, dict) and metadata.get("codex_search") is True:
        return True
    return isinstance(provenance, dict) and provenance.get("provider_name") == "codex-web-search"


def _verify_sqlite_run(config: CodexSmokeConfig) -> None:
    from nlp_stock_prediction.storage import initialize_research_database

    store = initialize_research_database(config.database_path)
    run_id = _expected_run_id(config)
    run = store.get_research_run(run_id)
    if run is None:
        raise RuntimeError(f"Codex smoke did not persist research run {run_id!r}.")

    tool_runs = store.list_tool_runs_for_run(run_id)
    if not tool_runs:
        raise RuntimeError("Codex smoke did not persist tool_runs.")

    artifacts = store.list_artifacts_for_run(run_id)
    if not artifacts:
        raise RuntimeError("Codex smoke did not persist artifacts.")

    evidence = store.list_evidence_for_run(run_id)
    if not evidence:
        raise RuntimeError("Codex smoke did not persist search evidence.")
    if not any(_is_codex_sqlite_evidence(item) for item in evidence):
        raise RuntimeError("Codex smoke SQLite evidence did not include Codex live search.")

    candidates = store.list_prediction_candidates_for_run(run_id)
    if not candidates:
        raise RuntimeError("Codex smoke did not persist prediction candidates.")


def _prepare_clean_run_dir(config: CodexSmokeConfig) -> None:
    run_dir = _resolve_repo_path(config, config.run_dir).resolve()
    allowed_roots = tuple(
        (config.repo_root / root).resolve() for root in ("reports", "artifacts", "data", "cache")
    )
    if not any(_is_relative_to(run_dir, root) or run_dir == root for root in allowed_roots):
        roots = ", ".join(root.as_posix() for root in allowed_roots)
        raise RuntimeError(f"Refusing to clean smoke output outside ignored roots: {roots}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _prepare_clean_database(config: CodexSmokeConfig) -> None:
    database_path = config.database_path.resolve()
    data_root = (config.repo_root / "data").resolve()
    if not (_is_relative_to(database_path, data_root) or database_path == data_root):
        raise RuntimeError("Refusing to clean smoke database outside data/.")
    for path in (
        database_path,
        database_path.with_name(f"{database_path.name}-wal"),
        database_path.with_name(f"{database_path.name}-shm"),
        database_path.with_name(f"{database_path.name}-journal"),
    ):
        if path.exists():
            path.unlink()


def _expected_run_id(config: CodexSmokeConfig) -> str:
    normalized_symbol = config.symbol.lower().replace("/", "-")
    return f"codex-smoke-{config.run_date.isoformat()}-{normalized_symbol}"


def _is_codex_sqlite_evidence(item: object) -> bool:
    provider = getattr(item, "provider", None)
    metadata = getattr(item, "metadata", None)
    provenance = getattr(item, "provenance_json", None)
    if provider == "codex-web-search":
        return True
    if isinstance(metadata, dict) and metadata.get("codex_search") is True:
        return True
    return isinstance(provenance, dict) and provenance.get("provider_name") == "codex-web-search"


def _git_status(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _resolve_repo_path(config: CodexSmokeConfig, path: Path) -> Path:
    return path if path.is_absolute() else config.repo_root / path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _toml_string(value: str) -> str:
    return json.dumps(value)


if __name__ == "__main__":
    raise SystemExit(main())
