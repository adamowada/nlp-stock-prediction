"""JSON report renderer."""

from __future__ import annotations

from typing import Any

from nlp_stock_prediction.contracts import DailyReport
from nlp_stock_prediction.contracts.report import DEFAULT_JSON_REPORT_CONTRACT


def render_json_report(report: DailyReport) -> str:
    validate_json_report_contract(report)
    return f"{report.model_dump_json(indent=2)}\n"


def load_json_report(payload: str | bytes) -> DailyReport:
    """Load a JSON report and validate the stable machine-readable section contract."""

    report = DailyReport.model_validate_json(payload)
    return validate_json_report_contract(report)


def validate_json_report_contract(report: DailyReport) -> DailyReport:
    """Validate that the report JSON shape covers every material product section."""

    if report.schema_version != DEFAULT_JSON_REPORT_CONTRACT.report_schema_version:
        raise ValueError(
            "JSON report payload schema_version must match the JSON report contract "
            f"({report.schema_version!r} != "
            f"{DEFAULT_JSON_REPORT_CONTRACT.report_schema_version!r})"
        )
    payload = report.model_dump(mode="json")
    missing: list[str] = []
    for section in DEFAULT_JSON_REPORT_CONTRACT.material_sections:
        for pointer in section.json_pointers:
            if not _has_json_pointer(payload, pointer):
                missing.append(f"{section.heading}:{pointer}")
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"JSON report payload is missing material report fields: {missing_text}")
    return report


def _has_json_pointer(payload: Any, pointer: str) -> bool:
    current = payload
    for token in pointer.lstrip("/").split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and key in current:
            current = current[key]
            continue
        return False
    return True


__all__ = ["load_json_report", "render_json_report", "validate_json_report_contract"]
