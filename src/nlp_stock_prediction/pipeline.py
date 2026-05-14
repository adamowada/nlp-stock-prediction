"""CLI orchestration for deterministic research report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from nlp_stock_prediction.contracts import DailyReport
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.orchestration import (
    DUMMY_ORCHESTRATION_DISABLED_MESSAGE,
    ReportBundle,
    generate_dummy_report_bundle,
)
from nlp_stock_prediction.orchestration.phase2_common import stable_digest
from nlp_stock_prediction.orchestration.phase4_service import Phase4Service

LIVE_ORCHESTRATION_DISABLED_MESSAGE = DUMMY_ORCHESTRATION_DISABLED_MESSAGE


def generate_daily_report(config: RunConfig) -> ReportBundle:
    """Generate a deterministic Phase 4 research report bundle."""

    if not config.offline:
        return generate_dummy_report_bundle(config)

    repo_root = _repo_root_for_output(config.output_dir)
    output_arg = _path_arg(config.output_dir, repo_root)
    output_digest = stable_digest(output_arg)[:8]
    normalized_symbol = config.symbol.strip().upper()
    symbol_digest = stable_digest(normalized_symbol)[:8]
    service = Phase4Service(
        repo_root=repo_root,
        database_path=Path("data")
        / (f"phase4-runtime-{config.run_date.isoformat()}-{symbol_digest}-{output_digest}.sqlite3"),
    )
    result = service.run_offline_phase4_flow(
        run_date=config.run_date.isoformat(),
        output_dir=output_arg,
        symbol=normalized_symbol,
    )
    report_payload = cast(dict[str, object], result["report"])
    markdown_path = Path(str(report_payload["markdown_path"]))
    json_path = Path(str(report_payload["json_path"]))
    audit_manifest_path = Path(str(report_payload["audit_manifest_path"]))
    report = DailyReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    return ReportBundle(
        report_dir=markdown_path.parent,
        markdown_path=markdown_path,
        json_path=json_path,
        audit_dir=audit_manifest_path.parent,
        audit_manifest_path=audit_manifest_path,
        report=report,
        tool_records=cast(Any, service.store.list_tool_runs_for_run(str(result["run_id"]))),
    )


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


__all__ = ["LIVE_ORCHESTRATION_DISABLED_MESSAGE", "ReportBundle", "generate_daily_report"]
