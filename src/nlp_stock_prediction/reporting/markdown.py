"""Markdown report renderer."""

from __future__ import annotations

from decimal import Decimal

from nlp_stock_prediction.contracts import (
    AnalysisComponent,
    AuditManifest,
    DailyReport,
    EvidenceReference,
    TradeCandidate,
)


def render_markdown_report(report: DailyReport) -> str:
    lines: list[str] = [
        "# Daily Stock Opportunity Report",
        "",
        f"Report date: {report.report_date.isoformat()}",
        f"Generated at: {report.generated_at.isoformat()}",
        f"Run ID: `{report.run_id}`",
        f"Risk profile: {report.risk_profile.value}",
    ]
    if report.account_capital is not None:
        lines.append(f"Account capital: {report.account_capital}")
    lines.extend(
        [
            f"Data freshness: {report.data_freshness.summary}",
            f"Disclaimer: {report.disclaimer.text}",
            "",
            "## Data Freshness",
            "",
            f"- As of: {report.data_freshness.as_of.isoformat()}",
            f"- Summary: {report.data_freshness.summary}",
            f"- Stale providers: {_format_list(report.data_freshness.stale_provider_names)}",
            f"- Missing providers: {_format_list(report.data_freshness.missing_provider_names)}",
            "",
            "## Provider Warnings",
            "",
        ]
    )
    warnings = [warning for health in report.provider_health for warning in health.warnings]
    if warnings:
        for warning in warnings:
            provider = warning.provider_name or "unknown-provider"
            lines.append(
                f"- `{provider}` {warning.severity.value}: [{warning.code.value}] {warning.message}"
            )
    else:
        lines.append("- No provider warnings.")

    lines.extend(["", "## Ticker Sections", ""])
    for section in report.ticker_sections:
        lines.extend(
            [
                f"### {section.ticker}",
                "",
                f"Company: {section.company_name or 'Unknown'}",
                "",
                "#### Observed Discussion",
                "",
                section.observed_discussion_summary or "No observed discussion summary available.",
                "",
                "#### Social And News",
                "",
                section.social_news_summary or "No social or news summary available.",
                "",
                "#### Strategy Clusters",
                "",
            ]
        )
        if section.strategy_clusters:
            for cluster in section.strategy_clusters:
                lines.append(
                    f"- `{cluster.cluster_id}`: {cluster.direction.value} "
                    f"{cluster.instrument.value} over {cluster.time_horizon.value}; "
                    f"confidence {cluster.confidence:.2f}; evidence "
                    f"{_format_evidence_ids(cluster.evidence)}."
                )
        else:
            lines.append("- No evidence-backed Reddit strategies extracted for this ticker.")
        lines.extend(["", "#### Technical Analysis", ""])
        lines.extend(_render_analysis(section.technical_analysis))
        lines.extend(["", "#### Fundamental Analysis", ""])
        lines.extend(_render_analysis(section.fundamental_analysis))
        lines.extend(["", "#### Sector Context", ""])
        lines.extend(_render_analysis(section.sector_context))
        lines.extend(["", "#### Macro Context", ""])
        lines.extend(_render_analysis(section.macro_context))
        lines.extend(["", "#### Opportunity Notes", ""])
        if section.opportunity_notes:
            lines.extend(f"- {note}" for note in section.opportunity_notes)
        else:
            lines.append("- No opportunity notes available.")
        lines.extend(["", "#### Evidence References", ""])
        if section.evidence:
            lines.extend(_render_evidence_refs(section.evidence))
        else:
            lines.append("- No evidence references available.")
        lines.append("")

    lines.extend(["## Qualified Trading Strategies Or No-Trade Summary", ""])
    if report.trade_candidates:
        lines.extend(["### Qualified Trading Strategies", ""])
        for candidate in report.trade_candidates:
            lines.extend(_render_candidate(candidate))
    else:
        lines.extend(
            [
                "### No-Trade Summary",
                "",
                report.no_trade_summary or "No qualified strategies passed the report gates.",
                "",
            ]
        )

    lines.extend(
        [
            "## Disclaimer",
            "",
            report.disclaimer.text,
            "",
            "## Audit Artifacts",
            "",
        ]
    )
    manifest = report.audit_manifest
    if isinstance(manifest, AuditManifest) and manifest.artifacts:
        for artifact in manifest.artifacts:
            digest = f", sha256 `{artifact.sha256}`" if artifact.sha256 else ""
            count = (
                f", records {artifact.record_count}" if artifact.record_count is not None else ""
            )
            lines.append(
                f"- `{artifact.artifact_id}` ({artifact.artifact_type}): "
                f"{artifact.path}{count}{digest}"
            )
    else:
        lines.append("- Audit manifest unavailable.")
    lines.append("")
    return "\n".join(lines)


def _render_candidate(candidate: TradeCandidate) -> list[str]:
    lines = [
        f"- `{candidate.candidate_id}` ({candidate.ticker}): {candidate.thesis}",
        f"  - Action: {candidate.action.value}; instrument: {candidate.instrument.value}; "
        f"horizon: {candidate.time_horizon.value}",
        f"  - Score: {_format_score(candidate.score.overall_score)} "
        f"(confidence {_format_score(candidate.score.confidence)}, "
        f"threshold {_format_score(candidate.score.threshold)})",
        f"  - Entry: {candidate.entry_logic}",
        f"  - Invalidation: {candidate.invalidation_criteria}",
        f"  - Risk: {candidate.risk_plan.sizing_basis or 'Defined-risk sizing required.'}",
        f"  - Evidence: {_format_evidence_ids(candidate.evidence)}",
        "",
    ]
    return lines


def _render_analysis(component: AnalysisComponent | None) -> list[str]:
    if component is None:
        return ["No analysis available."]
    lines = [
        f"- Summary: {component.summary}",
        f"- Signal: {component.signal.value}; confidence {component.confidence:.2f}",
    ]
    trend = getattr(component, "trend", None)
    if isinstance(trend, str) and trend:
        lines.append(f"- Trend: {trend}")
    valuation_summary = getattr(component, "valuation_summary", None)
    if isinstance(valuation_summary, str) and valuation_summary:
        lines.append(f"- Valuation: {valuation_summary}")
    sector = getattr(component, "sector", None)
    if isinstance(sector, str) and sector:
        lines.append(f"- Sector: {sector}")
    supportive = getattr(component, "supportive_factors", ())
    if supportive:
        formatted = _format_list(tuple(str(item) for item in supportive))
        lines.append(f"- Supportive factors: {formatted}")
    conflicting = getattr(component, "conflicting_factors", ())
    if conflicting:
        formatted = _format_list(tuple(str(item) for item in conflicting))
        lines.append(f"- Conflicting factors: {formatted}")
    return lines


def _render_evidence_refs(evidence: tuple[EvidenceReference, ...]) -> list[str]:
    lines: list[str] = []
    for reference in evidence:
        quote = f" - {reference.quote}" if reference.quote else ""
        lines.append(f"- `{reference.evidence_id}`{quote}")
    return lines


def _format_evidence_ids(evidence: tuple[EvidenceReference, ...]) -> str:
    if not evidence:
        return "none"
    return ", ".join(f"`{reference.evidence_id}`" for reference in evidence)


def _format_list(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"


def _format_score(value: float | Decimal) -> str:
    return f"{float(value):.2f}"


__all__ = ["render_markdown_report"]
