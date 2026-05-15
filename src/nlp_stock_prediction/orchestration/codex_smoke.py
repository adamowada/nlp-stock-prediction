"""Implementation for the opt-in Phase 4 real Codex MCP smoke test."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from nlp_stock_prediction.orchestration.phase2_common import (
    ALLOWED_WRITE_ROOTS,
    is_relative_to,
    symbol_slug,
)

RUN_CODEX_SMOKE_ENV = "NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE"
ALLOWED_SMOKE_WRITE_ROOTS = ALLOWED_WRITE_ROOTS
RESTRICTED_SCAN_EXCLUDED_DIRS = {".git"}
MAX_HASHED_RESTRICTED_FILE_BYTES = 5 * 1024 * 1024

type RestrictedPathSnapshot = dict[str, tuple[int, int, str | None]]


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
        return self.output_dir / self.run_date.isoformat() / symbol_slug(self.symbol)

    @property
    def final_message_path(self) -> Path:
        return self.run_dir / "codex-final.md"

    @property
    def database_arg_path(self) -> Path:
        filename = (
            f"phase4-codex-smoke-{self.run_date.isoformat()}-{symbol_slug(self.symbol)}.sqlite3"
        )
        return Path("data") / filename

    @property
    def database_path(self) -> Path:
        return resolve_repo_path(self, self.database_arg_path)


def build_codex_prompt(config: CodexSmokeConfig) -> str:
    """Return the instruction payload for the real Codex smoke run."""

    return "\n".join(
        [
            "You are smoke-testing Phase 4 of nlp-stock-prediction.",
            "Use the nlp-stock-prediction Phase 4 MCP tools; do not edit source files.",
            "Do not import project modules directly or run shell fallbacks for MCP tools.",
            "If an MCP tool call is unavailable or cancelled, stop and report smoke failure.",
            f"Run date: {config.run_date.isoformat()}",
            f"Symbol: {config.symbol.upper()}",
            f"Output directory: {config.output_dir.as_posix()}",
            "",
            "Required tool workflow:",
            "1. start_research_run",
            "2. list_research_tool_plan",
            "3. phase4_universe_discovery",
            "4. phase4_market_data",
            "5. phase4_technical_package",
            "6. phase4_social_evidence",
            "7. phase4_news_catalyst",
            "8. phase4_fundamentals",
            "9. phase4_sector_macro",
            "10. phase4_prediction_candidate_synthesis",
            "11. phase4_prediction_evaluation",
            "12. render_prediction_report",
            "13. inspect_research_run",
            "",
            "The final response must summarize the report path, JSON path, audit manifest path, "
            "and whether at least one source evidence item was recorded.",
        ]
    )


def build_codex_command(config: CodexSmokeConfig) -> list[str]:
    """Build the Codex CLI invocation without launching it."""

    mcp_command = _toml_string(str(config.python_executable))
    mcp_args = json.dumps(
        [
            "-B",
            "-m",
            "nlp_stock_prediction.codex_mcp",
            "--repo-root",
            str(config.repo_root),
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

    run_dir = resolve_repo_path(config, config.run_dir)
    final_message_path = resolve_repo_path(config, config.final_message_path)
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
        "record_codex_search_evidence",
        "run_dummy_universe_tool",
        "run_dummy_analysis_tool",
        "synthesize_prediction_candidates",
    )
    if any(marker in final_message for marker in failure_markers):
        raise RuntimeError("Codex smoke final message reported MCP fallback or failure.")

    report_payload = json.loads(json_path.read_text(encoding="utf-8"))
    report_text = json.dumps(report_payload, sort_keys=True).lower()
    data_failure_markers = (
        "phase2mcpservice",
        "record_codex_search_evidence",
        "run_dummy_universe_tool",
        "run_dummy_analysis_tool",
        "synthesize_prediction_candidates",
        '"provider": "dummy',
        '"tool_name": "dummy',
    )
    if any(marker in report_text for marker in data_failure_markers):
        raise RuntimeError("Codex smoke report JSON contains fallback or dummy provenance.")
    evidence = report_payload.get("evidence_sources")
    if not isinstance(evidence, list) or not evidence:
        raise RuntimeError("Codex smoke report did not include evidence_sources.")
    candidates = report_payload.get("prediction_candidates")
    insufficient = report_payload.get("insufficient_evidence_summary")
    if not candidates and not insufficient:
        raise RuntimeError(
            "Codex smoke report must include prediction_candidates or insufficient evidence."
        )
    if require_sqlite:
        verify_sqlite_run(config)


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
    if not _python_has_mcp(config.python_executable, cwd=config.repo_root):
        raise RuntimeError(
            'Install the optional smoke extra first: python -m pip install -e ".[codex-smoke]"'
        )

    prepare_clean_run_dir(config)
    prepare_clean_database(config)
    baseline_status = require_clean_tracked_status(config.repo_root)
    baseline_restricted_paths = snapshot_restricted_paths(config.repo_root)
    codex_env = os.environ.copy()
    codex_env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        build_codex_command(config),
        cwd=config.repo_root,
        check=False,
        text=True,
        env=codex_env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Codex smoke failed with exit code {completed.returncode}.")
    verify_smoke_outputs(config)
    ensure_tracked_status_unchanged(config.repo_root, baseline_status)
    ensure_restricted_paths_unchanged(config.repo_root, baseline_restricted_paths)


def verify_sqlite_run(config: CodexSmokeConfig) -> None:
    from nlp_stock_prediction.storage import initialize_research_database

    store = initialize_research_database(config.database_path)
    run_id = expected_run_id(config)
    run = store.get_research_run(run_id)
    if run is None:
        raise RuntimeError(f"Codex smoke did not persist research run {run_id!r}.")

    tool_runs = store.list_tool_runs_for_run(run_id)
    if not tool_runs:
        raise RuntimeError("Codex smoke did not persist tool_runs.")
    fallback_tool_names = {
        "run_dummy_universe_tool",
        "run_dummy_analysis_tool",
        "synthesize_prediction_candidates",
        "record_codex_search_evidence",
    }
    if any(record.tool_name in fallback_tool_names for record in tool_runs):
        raise RuntimeError("Codex smoke persisted fallback tool runs.")

    artifacts = store.list_artifacts_for_run(run_id)
    if not artifacts:
        raise RuntimeError("Codex smoke did not persist artifacts.")

    evidence = store.list_evidence_for_run(run_id)
    if not evidence:
        raise RuntimeError("Codex smoke did not persist source evidence.")
    if any("dummy" in record.provider.lower() for record in evidence):
        raise RuntimeError("Codex smoke persisted dummy evidence provenance.")
    candidates = store.list_prediction_candidates_for_run(run_id)
    if not candidates:
        raise RuntimeError("Codex smoke did not persist prediction candidates.")


def prepare_clean_run_dir(config: CodexSmokeConfig) -> None:
    run_dir = resolve_repo_path(config, config.run_dir).resolve()
    allowed_roots = tuple((config.repo_root / root).resolve() for root in ALLOWED_SMOKE_WRITE_ROOTS)
    if not any(is_relative_to(run_dir, root) or run_dir == root for root in allowed_roots):
        roots = ", ".join(root.as_posix() for root in allowed_roots)
        raise RuntimeError(f"Refusing to clean smoke output outside ignored roots: {roots}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def prepare_clean_database(config: CodexSmokeConfig) -> None:
    database_path = config.database_path.resolve()
    data_root = (config.repo_root / "data").resolve()
    if not (is_relative_to(database_path, data_root) or database_path == data_root):
        raise RuntimeError("Refusing to clean smoke database outside data/.")
    for path in (
        database_path,
        database_path.with_name(f"{database_path.name}-wal"),
        database_path.with_name(f"{database_path.name}-shm"),
        database_path.with_name(f"{database_path.name}-journal"),
    ):
        if path.exists():
            path.unlink()
    from nlp_stock_prediction.storage import initialize_research_database

    initialize_research_database(database_path)


def _python_has_mcp(python_executable: Path, *, cwd: Path) -> bool:
    completed = subprocess.run(
        [str(python_executable), "-c", "import mcp"],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def expected_run_id(config: CodexSmokeConfig) -> str:
    return f"phase4-{config.run_date.isoformat()}-{symbol_slug(config.symbol)}"


def snapshot_restricted_paths(repo_root: Path) -> RestrictedPathSnapshot:
    """Fingerprint repo files outside the smoke artifact roots."""

    resolved_repo = repo_root.resolve()
    allowed_roots = tuple((resolved_repo / root).resolve() for root in ALLOWED_SMOKE_WRITE_ROOTS)
    snapshot: RestrictedPathSnapshot = {}
    for current_root, dir_names, file_names in os.walk(resolved_repo):
        current_path = Path(current_root).resolve()
        dir_names[:] = [
            dirname
            for dirname in dir_names
            if dirname not in RESTRICTED_SCAN_EXCLUDED_DIRS
            and not _is_allowed_smoke_path((current_path / dirname).resolve(), allowed_roots)
        ]
        if _is_allowed_smoke_path(current_path, allowed_roots):
            dir_names[:] = []
            continue
        for file_name in file_names:
            path = current_path / file_name
            if _is_allowed_smoke_path(path, allowed_roots):
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            snapshot[path.relative_to(resolved_repo).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
                _restricted_file_hash(path, stat.st_size),
            )
    return snapshot


def ensure_restricted_paths_unchanged(
    repo_root: Path,
    expected: RestrictedPathSnapshot,
) -> None:
    observed = snapshot_restricted_paths(repo_root)
    if observed == expected:
        return
    added = sorted(set(observed).difference(expected))
    removed = sorted(set(expected).difference(observed))
    changed = sorted(
        path for path in set(expected).intersection(observed) if expected[path] != observed[path]
    )
    details = _format_restricted_path_changes(added, removed, changed)
    raise RuntimeError(
        "Codex smoke changed files outside allowed artifact roots "
        f"{ALLOWED_SMOKE_WRITE_ROOTS}.\n{details}"
    )


def resolve_repo_path(config: CodexSmokeConfig, path: Path) -> Path:
    return path if path.is_absolute() else config.repo_root / path


def _is_codex_search_evidence(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    provenance = item.get("provenance")
    metadata = item.get("metadata")
    if isinstance(metadata, dict) and metadata.get("codex_search") is True:
        return True
    return isinstance(provenance, dict) and provenance.get("provider_name") == "codex-web-search"


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


def _format_restricted_path_changes(
    added: list[str],
    removed: list[str],
    changed: list[str],
) -> str:
    parts: list[str] = []
    if added:
        parts.append(f"Added: {', '.join(added[:10])}")
    if removed:
        parts.append(f"Removed: {', '.join(removed[:10])}")
    if changed:
        parts.append(f"Changed: {', '.join(changed[:10])}")
    return "\n".join(parts) if parts else "Only metadata changed."


def _restricted_file_hash(path: Path, size: int) -> str | None:
    if size > MAX_HASHED_RESTRICTED_FILE_BYTES:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_allowed_smoke_path(path: Path, allowed_roots: tuple[Path, ...]) -> bool:
    return any(is_relative_to(path, root) or path == root for root in allowed_roots)


def _toml_string(value: str) -> str:
    return json.dumps(value)


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
    "prepare_clean_database",
    "prepare_clean_run_dir",
    "require_clean_tracked_status",
    "resolve_repo_path",
    "run_codex_smoke",
    "snapshot_restricted_paths",
    "verify_smoke_outputs",
    "verify_sqlite_run",
]
