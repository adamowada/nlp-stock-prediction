"""JSON report renderer."""

from __future__ import annotations

from nlp_stock_prediction.contracts import DailyReport


def render_json_report(report: DailyReport) -> str:
    return f"{report.model_dump_json(indent=2)}\n"


__all__ = ["render_json_report"]
