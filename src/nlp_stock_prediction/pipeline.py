"""CLI orchestration for deterministic report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from nlp_stock_prediction.analysis.ml_signal import (
    TimesFmMlSignalAttachment,
    apply_technical_ml_signal,
    load_timesfm_ml_signal_attachment,
)
from nlp_stock_prediction.contracts import (
    AuditArtifact,
    AuditManifest,
    DailyReport,
    JsonObject,
    JsonValue,
)
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.reporting.audit import json_payload_sha256, write_json_artifact
from nlp_stock_prediction.reporting.fixtures import (
    OfflineFixtureBundle,
    build_offline_fixture_bundle,
)
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.reporting.scrape_fixtures import (
    ScrapeFixtureBundle,
    build_live_scrape_bundle,
    build_scrape_fixture_bundle,
)

LIVE_ORCHESTRATION_DISABLED_MESSAGE = (
    "No source mode selected; pass --offline for the deterministic fixture-backed report, "
    "--source-mode scrape for the fixture-backed scrape-source provider path, or "
    "--source-mode scrape --live-providers for explicit live provider evidence collection."
)


@dataclass(frozen=True)
class ReportBundle:
    report_dir: Path
    markdown_path: Path
    json_path: Path
    audit_dir: Path
    audit_manifest_path: Path


def generate_daily_report(config: RunConfig) -> ReportBundle:
    """Generate a deterministic report bundle for the configured run."""

    fixture_bundle: OfflineFixtureBundle | ScrapeFixtureBundle
    if config.offline or config.source_mode == "offline":
        fixture_bundle = build_offline_fixture_bundle(config)
    elif config.source_mode == "scrape":
        fixture_bundle = (
            build_live_scrape_bundle(config)
            if config.live_providers
            else build_scrape_fixture_bundle(config)
        )
    else:
        raise ValueError(LIVE_ORCHESTRATION_DISABLED_MESSAGE)
    report = fixture_bundle.report
    report_dir = config.output_dir / config.run_date.isoformat()
    audit_dir = report_dir / "audit"
    markdown_path = report_dir / "report.md"
    json_path = report_dir / "report.json"
    audit_manifest_path = audit_dir / "audit-manifest.json"

    report_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    report = _attach_ml_artifact_if_requested(
        report,
        fixture_bundle.audit_payloads,
        config=config,
        audit_dir=audit_dir,
    )

    for filename, payload in fixture_bundle.audit_payloads.items():
        write_json_artifact(audit_dir / filename, payload)

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


def _attach_ml_artifact_if_requested(
    report: DailyReport,
    audit_payloads: dict[str, JsonObject],
    *,
    config: RunConfig,
    audit_dir: Path,
) -> DailyReport:
    if config.ml_artifact is None:
        return report
    try:
        attachment = load_timesfm_ml_signal_attachment(config.ml_artifact)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Could not load --ml-artifact {config.ml_artifact}: {exc}") from exc
    report, attached = _attach_timesfm_signal_to_report(report, attachment)
    if not attached:
        raise ValueError(
            f"TimesFM ML artifact ticker {attachment.ticker} did not match a report ticker "
            "with technical analysis."
        )
    audit_payloads["ml-artifacts.json"] = _ml_artifacts_payload(report, attachment)
    _refresh_analysis_context_payload(audit_payloads, report)
    return _refresh_audit_manifest(report, audit_payloads, attachment, audit_dir=audit_dir)


def _attach_timesfm_signal_to_report(
    report: DailyReport,
    attachment: TimesFmMlSignalAttachment,
) -> tuple[DailyReport, bool]:
    matched = False
    sections = []
    for section in report.ticker_sections:
        if section.ticker == attachment.ticker and section.technical_analysis is not None:
            matched = True
            data_quality = {
                **dict(section.data_quality),
                "ml_signal": "timesfm_evaluation_artifact",
                "ml_artifact_id": attachment.signal.source_artifact_id,
                "ml_artifact_sha256": attachment.artifact_sha256,
            }
            sections.append(
                section.model_copy(
                    update={
                        "technical_analysis": apply_technical_ml_signal(
                            section.technical_analysis,
                            attachment.signal,
                        ),
                        "data_quality": data_quality,
                    }
                )
            )
        else:
            sections.append(section)
    if not matched:
        return report, False
    return report.model_copy(update={"ticker_sections": tuple(sections)}), True


def _ml_artifacts_payload(
    report: DailyReport,
    attachment: TimesFmMlSignalAttachment,
) -> JsonObject:
    return {
        "schema_version": "audit.ml_artifacts.v1",
        "run_id": report.run_id,
        "records": [
            {
                "artifact_id": attachment.signal.source_artifact_id,
                "artifact_type": "timesfm_evaluation",
                "ticker": attachment.ticker,
                "path": str(attachment.artifact_path),
                "sha256": attachment.artifact_sha256,
                "payload": attachment.artifact_payload,
            }
        ],
    }


def _refresh_analysis_context_payload(
    audit_payloads: dict[str, JsonObject],
    report: DailyReport,
) -> None:
    analysis_contexts = audit_payloads.get("analysis-contexts.json")
    if analysis_contexts is None:
        return
    records_value = analysis_contexts.get("records")
    if not isinstance(records_value, list):
        return
    sections_by_ticker = {section.ticker: section for section in report.ticker_sections}
    refreshed_records: list[JsonValue] = []
    for record in records_value:
        if not isinstance(record, dict):
            refreshed_records.append(record)
            continue
        ticker = record.get("ticker")
        if not isinstance(ticker, str) or ticker not in sections_by_ticker:
            refreshed_records.append(cast(JsonValue, record))
            continue
        section = sections_by_ticker[ticker]
        refreshed = dict(record)
        refreshed["technical"] = (
            section.technical_analysis.model_dump(mode="json")
            if section.technical_analysis is not None
            else None
        )
        refreshed_records.append(cast(JsonValue, refreshed))
    analysis_contexts["records"] = refreshed_records


def _refresh_audit_manifest(
    report: DailyReport,
    audit_payloads: dict[str, JsonObject],
    attachment: TimesFmMlSignalAttachment,
    *,
    audit_dir: Path,
) -> DailyReport:
    manifest = report.audit_manifest
    if not isinstance(manifest, AuditManifest):
        return report
    artifacts = []
    for artifact in manifest.artifacts:
        filename = Path(artifact.path).name
        payload = audit_payloads.get(filename)
        if payload is None:
            artifacts.append(artifact)
            continue
        artifacts.append(
            artifact.model_copy(
                update={
                    "sha256": json_payload_sha256(payload),
                    "record_count": _record_count(payload),
                }
            )
        )
    artifacts.append(
        AuditArtifact(
            artifact_id="ml-artifacts",
            artifact_type="ml_artifact",
            path=(audit_dir / "ml-artifacts.json").as_posix(),
            created_at=report.generated_at,
            produced_by="timesfm-ml-artifact-integrator",
            sha256=json_payload_sha256(audit_payloads["ml-artifacts.json"]),
            record_count=_record_count(audit_payloads["ml-artifacts.json"]),
            metadata={
                "ticker": attachment.ticker,
                "source_artifact_id": attachment.signal.source_artifact_id,
                "source_artifact_sha256": attachment.artifact_sha256,
            },
        )
    )
    model_versions = dict(manifest.model_versions)
    model_versions["timesfm"] = attachment.signal.model_hash
    refreshed_manifest = manifest.model_copy(
        update={
            "artifacts": tuple(artifacts),
            "model_versions": model_versions,
            "command_args": report.command_args,
        }
    )
    return report.model_copy(update={"audit_manifest": refreshed_manifest})


def _record_count(payload: JsonObject) -> int | None:
    records = payload.get("records")
    if isinstance(records, list):
        return len(records)
    return None


__all__ = ["LIVE_ORCHESTRATION_DISABLED_MESSAGE", "ReportBundle", "generate_daily_report"]
