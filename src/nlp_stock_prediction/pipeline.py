"""CLI orchestration for deterministic research report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts import (
    AuditArtifact,
    AuditManifest,
    DailyReport,
    JsonObject,
)
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.reporting.audit import write_json_artifact
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report

LIVE_ORCHESTRATION_DISABLED_MESSAGE = (
    "No source mode selected; pass --offline for the deterministic fixture-backed report."
)


@dataclass(frozen=True)
class ReportBundle:
    report_dir: Path
    markdown_path: Path
    json_path: Path
    audit_dir: Path
    audit_manifest_path: Path


def generate_daily_report(config: RunConfig) -> ReportBundle:
    """Generate a deterministic research report bundle for the configured run."""

    if not (config.offline or config.source_mode == "offline"):
        raise ValueError(LIVE_ORCHESTRATION_DISABLED_MESSAGE)
    if config.live_providers:
        raise ValueError("live providers are not wired into the report runner yet")

    fixture_bundle = build_offline_fixture_bundle(config)
    report_dir = config.output_dir / config.run_date.isoformat()
    audit_dir = report_dir / "audit"
    markdown_path = report_dir / "report.md"
    json_path = report_dir / "report.json"
    audit_manifest_path = audit_dir / "audit-manifest.json"

    report_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_artifacts = _write_audit_payloads(
        fixture_bundle.audit_payloads,
        report=fixture_bundle.report,
        audit_dir=audit_dir,
    )
    report = _attach_audit_manifest(fixture_bundle.report, audit_artifacts)
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    json_path.write_text(render_json_report(report), encoding="utf-8")

    manifest = report.audit_manifest
    if not isinstance(manifest, AuditManifest):
        raise TypeError("offline fixture reports must include an AuditManifest")
    write_json_artifact(
        audit_manifest_path,
        cast(JsonObject, manifest.model_dump(mode="json")),
    )

    return ReportBundle(
        report_dir=report_dir,
        markdown_path=markdown_path,
        json_path=json_path,
        audit_dir=audit_dir,
        audit_manifest_path=audit_manifest_path,
    )


def _write_audit_payloads(
    audit_payloads: dict[str, JsonObject],
    *,
    report: DailyReport,
    audit_dir: Path,
) -> tuple[AuditArtifact, ...]:
    artifacts: list[AuditArtifact] = []
    for filename, payload in sorted(audit_payloads.items()):
        path = audit_dir / filename
        sha256 = write_json_artifact(path, payload)
        artifacts.append(
            AuditArtifact(
                artifact_id=Path(filename).stem,
                artifact_type=_artifact_type(filename),
                path=path.as_posix(),
                created_at=report.generated_at,
                produced_by="offline-fixture",
                sha256=sha256,
                record_count=_record_count(payload),
            )
        )
    return tuple(artifacts)


def _attach_audit_manifest(
    report: DailyReport,
    artifacts: tuple[AuditArtifact, ...],
) -> DailyReport:
    existing = report.audit_manifest
    if not isinstance(existing, AuditManifest):
        raise TypeError("offline fixture reports must include an AuditManifest")
    manifest = existing.model_copy(
        update={
            "artifacts": artifacts,
            "prediction_trace_ids": tuple(
                candidate.candidate_id for candidate in report.prediction_candidates
            ),
        }
    )
    return report.model_copy(update={"audit_manifest": manifest})


def _artifact_type(filename: str) -> str:
    if filename == "normalized-evidence.json":
        return "normalized_evidence"
    if filename == "analysis-contexts.json":
        return "analysis_context"
    if filename == "prediction-inputs.json":
        return "prediction_input"
    return "provider_result"


def _record_count(payload: JsonObject) -> int | None:
    records = payload.get("records")
    if isinstance(records, list):
        return len(records)
    return None


__all__ = ["LIVE_ORCHESTRATION_DISABLED_MESSAGE", "ReportBundle", "generate_daily_report"]
