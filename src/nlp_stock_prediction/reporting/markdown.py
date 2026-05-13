"""Markdown report renderer."""

from __future__ import annotations

from nlp_stock_prediction.contracts import (
    AnalysisComponent,
    AuditManifest,
    DailyReport,
    EvidenceReference,
    PredictionCandidate,
    TechnicalMlSignal,
)


def render_markdown_report(report: DailyReport) -> str:
    lines: list[str] = [
        "# Prediction Research Report",
        "",
        f"Report date: {report.report_date.isoformat()}",
        f"Generated at: {report.generated_at.isoformat()}",
        f"Run ID: `{report.run_id}`",
        f"Objective: {report.objective}",
        f"Universe: {report.universe}",
        f"Data freshness: {report.data_freshness.summary}",
        "",
        "## Research Objective",
        "",
        report.objective,
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
    warnings = [warning for health in report.provider_health for warning in health.warnings]
    if warnings:
        for warning in warnings:
            provider = warning.provider_name or "unknown-provider"
            lines.append(
                f"- `{provider}` {warning.severity.value}: [{warning.code.value}] {warning.message}"
            )
    else:
        lines.append("- No provider warnings.")

    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in report.prediction_candidates
    }
    lines.extend(["", "## Instrument Sections", ""])
    for section in report.instrument_sections:
        lines.extend(
            [
                f"### {section.symbol}",
                "",
                f"Name: {section.display_name or 'Unknown'}",
                "",
                "#### Observed Evidence",
                "",
                section.observed_discussion_summary or "No observed discussion summary available.",
                "",
                "#### Analysis",
                "",
                section.analysis_summary or "No analysis summary available.",
                "",
                "Technical:",
                "",
            ]
        )
        lines.extend(_render_analysis(section.technical_analysis))
        lines.extend(["", "Fundamental:", ""])
        lines.extend(_render_analysis(section.fundamental_analysis))
        lines.extend(["", "Sector:", ""])
        lines.extend(_render_analysis(section.sector_context))
        lines.extend(["", "Macro:", ""])
        lines.extend(_render_analysis(section.macro_context))
        lines.extend(["", "#### Prediction Scenarios", ""])
        section_candidates = [
            candidates_by_id[candidate_id]
            for candidate_id in section.prediction_candidate_ids
            if candidate_id in candidates_by_id
        ]
        if section_candidates:
            for candidate in section_candidates:
                lines.extend(_render_candidate(candidate))
        else:
            lines.append("- No evidence-backed prediction scenario for this instrument.")
        lines.extend(["", "#### Evidence References", ""])
        if section.evidence:
            lines.extend(_render_evidence_refs(section.evidence))
        else:
            lines.append("- No evidence references available.")
        lines.append("")

    lines.extend(["## Prediction Scenarios Or Insufficient-Evidence Summary", ""])
    if report.prediction_candidates:
        for candidate in report.prediction_candidates:
            lines.extend(_render_candidate(candidate))
    else:
        lines.extend(
            [
                report.insufficient_evidence_summary
                or "No prediction scenarios have enough evidence for this report.",
                "",
            ]
        )

    lines.extend(["## Audit Artifacts", ""])
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


def _render_candidate(candidate: PredictionCandidate) -> list[str]:
    lines = [
        f"- `{candidate.candidate_id}` ({candidate.symbol}): {candidate.thesis}",
        (
            f"  - Status: {candidate.status.value}; direction: {candidate.direction.value}; "
            f"horizon: {candidate.horizon.value}; confidence {candidate.confidence:.2f}"
        ),
        f"  - Baseline: {candidate.baseline}",
        f"  - Evidence for: {_format_evidence_ids(candidate.evidence_for)}",
        f"  - Evidence against: {_format_evidence_ids(candidate.evidence_against)}",
        f"  - Assumptions: {_format_list(candidate.assumptions)}",
        f"  - Uncertainties: {_format_list(candidate.uncertainties)}",
    ]
    if candidate.signal_artifact_ids:
        lines.append(f"  - Signal artifacts: {_format_list(candidate.signal_artifact_ids)}")
    lines.append("")
    return lines


def _render_analysis(component: AnalysisComponent | None) -> list[str]:
    if component is None:
        return ["- No analysis available."]
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
    ml_signal = getattr(component, "ml_signal", None)
    if ml_signal is not None:
        lines.extend(_render_ml_signal(ml_signal))
    supportive = getattr(component, "supportive_factors", ())
    if supportive:
        formatted = _format_list(tuple(str(item) for item in supportive))
        lines.append(f"- Supportive factors: {formatted}")
    conflicting = getattr(component, "conflicting_factors", ())
    if conflicting:
        formatted = _format_list(tuple(str(item) for item in conflicting))
        lines.append(f"- Conflicting factors: {formatted}")
    if component.evidence:
        lines.append(f"- Evidence: {_format_evidence_ids(component.evidence)}")
    return lines


def _render_evidence_refs(evidence: tuple[EvidenceReference, ...]) -> list[str]:
    lines: list[str] = []
    for reference in evidence:
        quote = f" - {reference.quote}" if reference.quote else ""
        lines.append(f"- `{reference.evidence_id}`{quote}")
    return lines


def _render_ml_signal(ml_signal: TechnicalMlSignal) -> list[str]:
    return [
        "- ML signal: "
        f"{ml_signal.signal.value}; "
        f"probability {ml_signal.probability_positive:.2f}; "
        f"confidence {ml_signal.calibrated_confidence:.2f}; "
        f"status {ml_signal.status}."
    ]


def _format_evidence_ids(evidence: tuple[EvidenceReference, ...]) -> str:
    if not evidence:
        return "none"
    return ", ".join(f"`{reference.evidence_id}`" for reference in evidence)


def _format_list(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"


__all__ = ["render_markdown_report"]
