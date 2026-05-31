"""CLI orchestration for deterministic research report generation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts import DailyReport
from nlp_stock_prediction.contracts.providers import BatchRunConfig, RunConfig
from nlp_stock_prediction.orchestration.orchestration_common import stable_digest
from nlp_stock_prediction.orchestration.report_bundle import BatchReportBundle, ReportBundle
from nlp_stock_prediction.orchestration.research_batch import generate_batch_research_reports
from nlp_stock_prediction.orchestration.research_service import ResearchService

LIVE_ORCHESTRATION_DISABLED_MESSAGE = (
    "Live report generation requires source_mode='live' or live_providers=True; "
    "refusing to use fixture or dummy fallback data."
)


def generate_daily_report(config: RunConfig) -> ReportBundle:
    """Generate a Research Stage research report bundle."""

    live_requested = config.source_mode == "live" or config.live_providers
    if not config.offline and not live_requested:
        raise ValueError(LIVE_ORCHESTRATION_DISABLED_MESSAGE)
    if config.offline and live_requested:
        raise ValueError("RunConfig cannot request both offline fixtures and live providers.")

    project_root = _project_root()
    repo_root = _write_root_for_output(config.output_dir, project_root)
    output_arg = _path_arg(config.output_dir, repo_root)
    output_digest = stable_digest(output_arg)[:8]
    normalized_symbol = config.symbol.strip().upper()
    symbol_digest = stable_digest(normalized_symbol)[:8]
    fixture_root = _fixture_project_root(config.fixture_dir, project_root)
    extra_write_roots = tuple(
        path for path in (config.output_dir, config.cache_dir) if path is not None
    )
    service = ResearchService(
        repo_root=repo_root,
        fixture_root=fixture_root,
        provider_cache_root=config.cache_dir,
        extra_write_roots=extra_write_roots,
        database_path=Path("data")
        / (
            f"research-{'offline' if config.offline else 'live'}-runtime-"
            f"{config.run_date.isoformat()}-{symbol_digest}-{output_digest}.sqlite3"
        ),
    )
    result = (
        service.run_offline_research_flow(
            run_date=config.run_date.isoformat(),
            output_dir=output_arg,
            symbol=normalized_symbol,
        )
        if config.offline
        else service.run_live_research_flow(
            run_date=config.run_date.isoformat(),
            output_dir=output_arg,
            symbol=normalized_symbol,
        )
    )
    report_payload = cast(dict[str, object], result["report"])
    markdown_path = Path(str(report_payload["markdown_path"]))
    json_path = Path(str(report_payload["json_path"]))
    audit_manifest_path = Path(str(report_payload["audit_manifest_path"]))
    report = DailyReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    database_path = Path(str(report_payload.get("database_path", service.database_path)))
    if not database_path.is_absolute():
        database_path = repo_root / database_path
    return ReportBundle(
        report_dir=markdown_path.parent,
        markdown_path=markdown_path,
        json_path=json_path,
        audit_dir=audit_manifest_path.parent,
        audit_manifest_path=audit_manifest_path,
        report=report,
        tool_records=service.store.list_tool_runs_for_run(str(result["run_id"])),
        database_path=database_path,
    )


def generate_ranked_research_reports(config: BatchRunConfig) -> BatchReportBundle:
    """Generate concurrent Research Stage reports and rank follow-up viability."""

    return generate_batch_research_reports(config, report_generator=generate_daily_report)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fixture_project_root(fixture_dir: Path | None, default: Path) -> Path:
    if fixture_dir is None:
        if not (default / "tests" / "fixtures").exists():
            raise ValueError(
                "offline fixture runs require --fixture-dir when repository fixtures are absent"
            )
        return default
    resolved = fixture_dir.resolve()
    if (resolved / "tests" / "fixtures").exists():
        return resolved
    if resolved.name == "fixtures" and resolved.parent.name == "tests":
        return resolved.parent.parent
    raise ValueError(
        "--fixture-dir must point to the repository root or to its tests/fixtures directory"
    )


def _write_root_for_output(output_dir: Path, project_root: Path) -> Path:
    resolved = output_dir.resolve()
    try:
        resolved.relative_to(project_root.resolve())
        return project_root.resolve()
    except ValueError:
        return resolved.parent


def _repo_root_for_output(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if resolved.name in {"reports", "artifacts", "data", "cache"}:
        return resolved.parent
    for parent in resolved.parents:
        if parent.name in {"reports", "artifacts", "data", "cache"}:
            return parent.parent
    return resolved.parent


def _path_arg(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


__all__ = [
    "LIVE_ORCHESTRATION_DISABLED_MESSAGE",
    "BatchReportBundle",
    "ReportBundle",
    "generate_daily_report",
    "generate_ranked_research_reports",
]
