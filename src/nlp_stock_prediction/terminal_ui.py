"""Rich-powered terminal UI for human-facing research workflows."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.contracts.report import AuditManifest, DailyReport
from nlp_stock_prediction.orchestration.report_bundle import ReportBundle
from nlp_stock_prediction.reporting.view import ReportView

ReportGenerator = Callable[[RunConfig], ReportBundle]


def prompt_for_research_config(
    *,
    run_date: date | None,
    output_dir: Path | None,
    symbol: str | None,
    fixture_dir: Path | None,
    cache_dir: Path | None,
    offline: bool,
    live: bool,
    console: Console | None = None,
) -> RunConfig:
    """Collect missing research options through a Rich prompt flow."""

    active_console = console or Console()
    missing_required = []
    if run_date is None:
        missing_required.append("--date")
    if output_dir is None:
        missing_required.append("--output")
    if not (offline or live):
        missing_required.append("--offline or --live")
    if missing_required and not active_console.is_interactive:
        missing = ", ".join(missing_required)
        raise ValueError(f"tui requires {missing} when stdin is not interactive")

    resolved_date = run_date or _ask_date(active_console)
    resolved_output = output_dir or Path(
        Prompt.ask("Output directory", default="reports", console=active_console)
    )
    if symbol is not None:
        resolved_symbol = symbol
    elif active_console.is_interactive:
        resolved_symbol = Prompt.ask("Symbol", default="TSLA", console=active_console)
    else:
        resolved_symbol = "TSLA"
    resolved_offline = offline
    resolved_live = live
    if not (resolved_offline or resolved_live):
        mode = Prompt.ask(
            "Research mode",
            choices=["offline", "live"],
            default="offline",
            console=active_console,
        )
        resolved_offline = mode == "offline"
        resolved_live = mode == "live"

    return RunConfig(
        run_date=resolved_date,
        output_dir=resolved_output,
        symbol=resolved_symbol,
        fixture_dir=fixture_dir,
        cache_dir=cache_dir,
        offline=resolved_offline,
        source_mode="offline" if resolved_offline else "live",
        live_providers=resolved_live,
    )


def run_research_terminal(
    config: RunConfig,
    *,
    report_generator: ReportGenerator,
    console: Console | None = None,
) -> ReportBundle:
    """Run report generation with a Rich terminal shell around the workflow."""

    active_console = console or Console()
    render_research_start(config, console=active_console)
    status_message = (
        "[bold cyan]Collecting evidence, assembling report, and writing audit artifacts..."
    )
    with active_console.status(status_message, spinner="dots"):
        bundle = report_generator(config)
    render_research_complete(bundle, console=active_console)
    return bundle


def render_research_start(config: RunConfig, *, console: Console | None = None) -> None:
    """Render the starting state for a research run."""

    active_console = console or Console()
    active_console.print(_header_panel())
    active_console.print(_request_panel(config))


def render_research_complete(bundle: ReportBundle, *, console: Console | None = None) -> None:
    """Render a full terminal dashboard for a completed research report."""

    active_console = console or Console()
    report = bundle.report
    view = ReportView.from_report(report)
    active_console.print(_header_panel())
    active_console.print(_completion_panel(bundle))
    active_console.print(_files_table(bundle))
    active_console.print(_provider_health_table(report))
    active_console.print(_scenario_table(report))
    active_console.print(_tool_runs_table(bundle))
    active_console.print(_evidence_table(report))
    if view.provider_warnings:
        active_console.print(_warnings_panel(view))
    _print_compatibility_file_lines(bundle, console=active_console)


def render_research_error(message: str, *, console: Console | None = None) -> None:
    """Render a research workflow error without exposing a traceback."""

    active_console = console or Console(stderr=True)
    active_console.print(
        Panel(
            Text(message, style="bold red"),
            title="Research run blocked",
            border_style="red",
            box=box.ROUNDED,
        )
    )


def _ask_date(console: Console) -> date:
    while True:
        value = Prompt.ask("Report date (YYYY-MM-DD)", console=console)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            try:
                return date.fromisoformat(value)
            except ValueError:
                pass
        console.print("[red]Expected a valid date in YYYY-MM-DD format.[/red]")


def _header_panel() -> Panel:
    title = Text("Prediction Research Terminal", style="bold cyan")
    subtitle = Text("Evidence-backed research, not trading instructions.", style="dim")
    return Panel(
        Group(Align.center(title), Align.center(subtitle)),
        border_style="cyan",
        box=box.DOUBLE,
    )


def _request_panel(config: RunConfig) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Symbol", config.symbol)
    table.add_row("Report date", config.run_date.isoformat())
    table.add_row("Mode", "offline fixtures" if config.offline else "live providers")
    table.add_row("Output", str(config.output_dir))
    table.add_row("Fixture root", _optional_path(config.fixture_dir))
    table.add_row("Cache root", _optional_path(config.cache_dir))
    return Panel(table, title="Research request", border_style="blue", box=box.ROUNDED)


def _completion_panel(bundle: ReportBundle) -> Panel:
    report = bundle.report
    manifest = report.audit_manifest
    artifact_count = len(manifest.artifacts) if isinstance(manifest, AuditManifest) else 0
    body = Table.grid(padding=(0, 2))
    body.add_column(style="bold")
    body.add_column()
    body.add_row("Run ID", report.run_id)
    body.add_row("Report date", report.report_date.isoformat())
    body.add_row("Generated", report.generated_at.isoformat())
    body.add_row("Freshness", report.data_freshness.summary)
    body.add_row("Instruments", str(len(report.instruments)))
    body.add_row("Evidence records", str(len(report.evidence_sources)))
    body.add_row("Prediction scenarios", str(len(report.prediction_candidates)))
    body.add_row("Audit artifacts", str(artifact_count))
    return Panel(body, title="Run complete", border_style="green", box=box.ROUNDED)


def _files_table(bundle: ReportBundle) -> Table:
    table = Table(title="Report files", box=box.ROUNDED, border_style="cyan", expand=True)
    table.add_column("Artifact", style="bold")
    table.add_column("Path")
    table.add_row("Markdown report", str(bundle.markdown_path))
    table.add_row("JSON report", str(bundle.json_path))
    table.add_row("Audit manifest", str(bundle.audit_manifest_path))
    table.add_row("Audit directory", str(bundle.audit_dir))
    return table


def _provider_health_table(report: DailyReport) -> Table:
    table = Table(title="Provider health", box=box.ROUNDED, border_style="blue", expand=True)
    table.add_column("Provider", style="bold")
    table.add_column("Status")
    table.add_column("Credentials")
    table.add_column("Warnings", justify="right")
    table.add_column("Checked")
    if not report.provider_health:
        table.add_row("none", "unknown", "unknown", "0", "none")
        return table
    for health in report.provider_health:
        status_style = _status_style(_enum_value(health.status))
        table.add_row(
            health.provider_name,
            Text(_enum_value(health.status), style=status_style),
            _enum_value(health.credential_state),
            str(len(health.warnings)),
            health.checked_at.isoformat() if health.checked_at else "none",
        )
    return table


def _scenario_table(report: DailyReport) -> Panel | Table:
    if not report.prediction_candidates:
        summary = (
            report.insufficient_evidence.summary
            if report.insufficient_evidence is not None
            else report.insufficient_evidence_summary
            or "No evidence-backed prediction scenario was produced."
        )
        return Panel(
            Markdown(summary),
            title="Insufficient evidence",
            border_style="yellow",
            box=box.ROUNDED,
        )

    table = Table(
        title="Prediction scenarios", box=box.ROUNDED, border_style="magenta", expand=True
    )
    table.add_column("Scenario", style="bold")
    table.add_column("Symbol")
    table.add_column("Status")
    table.add_column("Direction")
    table.add_column("Horizon")
    table.add_column("Confidence", justify="right")
    table.add_column("Evidence", justify="right")
    for candidate in report.prediction_candidates:
        table.add_row(
            candidate.candidate_id,
            candidate.symbol,
            Text(_enum_value(candidate.status), style=_status_style(_enum_value(candidate.status))),
            _enum_value(candidate.direction),
            _enum_value(candidate.horizon),
            f"{candidate.confidence:.2f}",
            str(len(candidate.evidence_for) + len(candidate.evidence_against)),
        )
    return table


def _tool_runs_table(bundle: ReportBundle) -> Table:
    table = Table(title="Tool runs", box=box.ROUNDED, border_style="green", expand=True)
    table.add_column("Tool", style="bold")
    table.add_column("Status")
    table.add_column("Warnings", justify="right")
    table.add_column("Error")
    if not bundle.tool_records:
        table.add_row("none", "not recorded", "0", "none")
        return table
    for record in bundle.tool_records[:12]:
        status = _record_status(record)
        table.add_row(
            _record_name(record),
            Text(status, style=_status_style(status)),
            str(_record_warning_count(record)),
            _record_error(record) or "none",
        )
    remaining = len(bundle.tool_records) - 12
    if remaining > 0:
        table.add_row(f"{remaining} more", "omitted", "0", "see audit database")
    return table


def _evidence_table(report: DailyReport) -> Table:
    table = Table(
        title="Evidence ledger preview", box=box.ROUNDED, border_style="cyan", expand=True
    )
    table.add_column("Evidence ID", style="bold")
    table.add_column("Kind")
    table.add_column("Provider")
    table.add_column("Freshness")
    if not report.evidence_sources:
        table.add_row("none", "none", "none", "unknown")
        return table
    for evidence in report.evidence_sources[:8]:
        provenance = evidence.provenance
        table.add_row(
            evidence.evidence_id,
            _enum_value(evidence.source_kind),
            provenance.provider_name,
            _enum_value(provenance.freshness_status),
        )
    remaining = len(report.evidence_sources) - 8
    if remaining > 0:
        table.add_row(f"{remaining} more", "omitted", "see report.json", "mixed")
    return table


def _warnings_panel(view: ReportView) -> Panel:
    lines = []
    for warning in view.provider_warnings[:8]:
        provider = warning.provider_name or "unknown-provider"
        lines.append(
            f"- `{provider}` {warning.severity.value} [{warning.code.value}] {warning.message}"
        )
    remaining = len(view.provider_warnings) - 8
    if remaining > 0:
        lines.append(f"- {remaining} more provider warnings are in the report JSON.")
    return Panel(
        Markdown("\n".join(lines)),
        title="Provider warnings",
        border_style="yellow",
        box=box.ROUNDED,
    )


def _print_compatibility_file_lines(bundle: ReportBundle, *, console: Console) -> None:
    console.print(f"Wrote Markdown report: {bundle.markdown_path}")
    console.print(f"Wrote JSON report: {bundle.json_path}")
    console.print(f"Wrote audit artifacts: {bundle.audit_dir}")


def _enum_value(value: Any) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value)


def _optional_path(path: Path | None) -> str:
    return str(path) if path is not None else "none"


def _status_style(status: str) -> str:
    normalized = status.lower()
    if normalized in {"ok", "succeeded", "successful", "completed", "evidence_supported", "fresh"}:
        return "green"
    if normalized in {"empty", "partial", "stale", "insufficient_evidence", "unknown"}:
        return "yellow"
    if normalized in {"failed", "rate_limited", "unconfigured", "unauthorized", "malformed"}:
        return "red"
    return "white"


def _record_name(record: object) -> str:
    value = getattr(record, "tool_name", None) or getattr(record, "tool_id", None)
    return str(value or "unknown")


def _record_status(record: object) -> str:
    value = getattr(record, "status", None)
    return str(value or "unknown")


def _record_warning_count(record: object) -> int:
    warnings = getattr(record, "warnings", ())
    return len(warnings) if isinstance(warnings, tuple) else 0


def _record_error(record: object) -> str | None:
    value = getattr(record, "error_message", None)
    return value if isinstance(value, str) and value else None


__all__ = [
    "prompt_for_research_config",
    "render_research_complete",
    "render_research_error",
    "render_research_start",
    "run_research_terminal",
]
