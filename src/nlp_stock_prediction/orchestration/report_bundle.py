"""Public report-bundle return type shared by orchestration entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nlp_stock_prediction.contracts import (
    DailyReport,
    ResearchViabilityRankingReport,
    WsbTrendingDiscoveryReport,
)
from nlp_stock_prediction.orchestration.runtime import ToolRunRecord as RuntimeToolRunRecord
from nlp_stock_prediction.storage.records import ToolRunRecord as StorageToolRunRecord

ReportToolRunRecord = RuntimeToolRunRecord | StorageToolRunRecord


@dataclass(frozen=True)
class ReportBundle:
    """Files produced by an orchestration run."""

    report_dir: Path
    markdown_path: Path
    json_path: Path
    audit_dir: Path
    audit_manifest_path: Path
    report: DailyReport
    tool_records: tuple[ReportToolRunRecord, ...]
    database_path: Path | None = None


@dataclass(frozen=True)
class BatchReportBundle:
    """Files produced by a batch research ranking run."""

    output_dir: Path
    ranking_markdown_path: Path
    ranking_json_path: Path
    ranking_report: ResearchViabilityRankingReport
    report_bundles: tuple[ReportBundle, ...]


@dataclass(frozen=True)
class WsbBatchReportBundle:
    """Files produced by WSB discovery plus batch research ranking."""

    output_dir: Path
    discovery_markdown_path: Path
    discovery_json_path: Path
    discovery_report: WsbTrendingDiscoveryReport
    batch_bundle: BatchReportBundle


__all__ = ["BatchReportBundle", "ReportBundle", "ReportToolRunRecord", "WsbBatchReportBundle"]
