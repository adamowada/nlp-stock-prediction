"""Public report-bundle return type shared by orchestration entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nlp_stock_prediction.contracts import DailyReport
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


__all__ = ["ReportBundle", "ReportToolRunRecord"]
