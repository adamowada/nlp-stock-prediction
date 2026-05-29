"""Report loading and concise terminal summaries for the app shell."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from nlp_stock_prediction.app.state import ReportIndexEntry
from nlp_stock_prediction.contracts import DailyReport
from nlp_stock_prediction.reporting.json import load_json_report


def load_report_for_entry(repo_root: Path, entry: ReportIndexEntry) -> DailyReport:
    return load_json_report(entry.resolve_json_path(repo_root).read_text(encoding="utf-8"))


def concise_summary_lines(report: DailyReport) -> list[str]:
    lines = [
        f"{_symbol(report)} | {report.report_date.isoformat()} | {report.data_freshness.summary}",
    ]
    if report.prediction_candidates:
        for candidate in report.prediction_candidates[:3]:
            direction = candidate.direction.value if candidate.direction is not None else "unknown"
            lines.append(
                f"{candidate.symbol}: {candidate.status.value}, {direction}, "
                f"{candidate.horizon.value}, confidence {candidate.confidence:.2f}"
            )
            lines.append(f"  {candidate.thesis}")
    else:
        summary = (
            report.insufficient_evidence.summary
            if report.insufficient_evidence is not None
            else report.insufficient_evidence_summary
            or "No evidence-backed prediction scenario was produced."
        )
        lines.append(f"Insufficient evidence: {summary}")
    blockers = major_blockers(report)
    if blockers:
        lines.append("Major blockers:")
        lines.extend(f"  - {item}" for item in blockers[:5])
    return lines


def major_blockers(report: DailyReport) -> list[str]:
    blockers: list[str] = []
    if report.insufficient_evidence is not None:
        blockers.extend(report.insufficient_evidence.blocking_reasons)
        blockers.extend(
            f"Missing {item}" for item in report.insufficient_evidence.missing_evidence_types
        )
    for health in report.provider_health:
        for warning in health.warnings:
            blockers.append(f"{health.provider_name}: {warning.message}")
    return list(dict.fromkeys(blockers))


def render_report_summary(
    report: DailyReport,
    *,
    console: Console,
    title: str = "Report Summary",
) -> None:
    console.print(
        Panel(
            "\n".join(concise_summary_lines(report)),
            title=title,
            border_style="cyan",
        )
    )


def report_table(entries: tuple[ReportIndexEntry, ...]) -> Table:
    table = Table(title="Reports", expand=True)
    table.add_column("#", justify="right")
    table.add_column("Symbol")
    table.add_column("Date")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Warnings", justify="right")
    for index, entry in enumerate(entries, start=1):
        table.add_row(
            str(index),
            entry.symbol,
            entry.report_date.isoformat(),
            entry.report_data_mode,
            entry.status_summary,
            str(entry.provider_warning_count),
        )
    return table


def render_report_details(report: DailyReport, *, console: Console) -> None:
    table = Table(title="Evidence And Provider Detail", expand=True)
    table.add_column("Area")
    table.add_column("Value")
    table.add_row("Evidence records", str(len(report.evidence_sources)))
    table.add_row("Provider health records", str(len(report.provider_health)))
    table.add_row("Prediction scenarios", str(len(report.prediction_candidates)))
    table.add_row("Source references", str(len(report.source_references)))
    table.add_row("Material claim traces", str(len(report.material_claim_traces)))
    console.print(table)
    blockers = major_blockers(report)
    if blockers:
        console.print(Panel("\n".join(f"- {item}" for item in blockers), title="Warnings"))


def render_markdown_report(markdown_path: Path, *, console: Console) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    if console.is_interactive:
        with console.pager(styles=True):
            console.print(Markdown(text))
    else:
        console.print(Markdown(text))


def _symbol(report: DailyReport) -> str:
    return report.instruments[0].symbol if report.instruments else "UNKNOWN"


__all__ = [
    "concise_summary_lines",
    "load_report_for_entry",
    "major_blockers",
    "render_markdown_report",
    "render_report_details",
    "render_report_summary",
    "report_table",
]
