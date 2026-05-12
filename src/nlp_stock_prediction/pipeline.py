"""CLI orchestration for deterministic report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts import AuditManifest, JsonObject
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.reporting.audit import write_json_artifact
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
    "Live-provider report orchestration is not enabled yet; pass --offline for the "
    "deterministic fixture-backed report or --source-mode scrape for the experimental "
    "fixture-backed scrape-source provider path."
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


__all__ = ["LIVE_ORCHESTRATION_DISABLED_MESSAGE", "ReportBundle", "generate_daily_report"]
