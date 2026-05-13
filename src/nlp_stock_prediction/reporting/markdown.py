"""Markdown report renderer."""

from __future__ import annotations

from datetime import datetime

from nlp_stock_prediction.contracts import (
    AnalysisComponent,
    AuditManifest,
    DailyReport,
    EvidenceReference,
    Instrument,
    InstrumentResolution,
    PredictionCandidate,
    SourceEvidence,
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
    instruments_by_id = {instrument.instrument_id: instrument for instrument in report.instruments}

    if report.instrument_resolutions:
        lines.extend(["", "## Universe Resolution", ""])
        lines.extend(_render_universe_resolution(report.instrument_resolutions))

    lines.extend(["", "## Instrument Sections", ""])
    for section in report.instrument_sections:
        instrument = instruments_by_id.get(section.instrument_id)
        lines.extend(
            [
                f"### {section.symbol}",
                "",
                f"Name: {section.display_name or 'Unknown'}",
                "",
                "#### Identity & Availability",
                "",
            ]
        )
        if instrument is not None:
            lines.extend(_render_instrument_identity(instrument))
        else:
            lines.append(f"- Instrument ID: `{section.instrument_id}`")
        lines.extend(
            [
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

    lines.extend(["## Evidence Ledger", ""])
    if report.evidence_sources:
        lines.extend(_render_evidence_ledger(report.evidence_sources))
    else:
        lines.append("- No source evidence records available.")
    lines.append("")

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


def _render_universe_resolution(
    resolutions: tuple[InstrumentResolution, ...],
) -> list[str]:
    lines: list[str] = []
    for resolution in resolutions:
        selected = (
            f"; selected `{resolution.selected_instrument_id}`"
            if resolution.selected_instrument_id
            else ""
        )
        lines.append(f"- `{resolution.query}`: {resolution.status.value}{selected}")
        if resolution.matches:
            lines.append(f"  - Matches: {_format_resolution_matches(resolution.matches)}")
        else:
            lines.append("  - Matches: none")
        lines.append(f"  - Warnings: {_format_list(resolution.warnings)}")
    return lines


def _render_instrument_identity(instrument: Instrument) -> list[str]:
    lines = [
        f"- Instrument ID: `{instrument.instrument_id}`",
        f"- Asset class: {instrument.asset_class.value}",
        f"- Venue: {instrument.venue or 'none'}",
        f"- Aliases: {_format_code_list(instrument.aliases)}",
        "- Provider IDs:",
    ]
    if instrument.provider_ids:
        for provider_id in instrument.provider_ids:
            namespace = f"/{provider_id.namespace}" if provider_id.namespace else ""
            url = f"; URL: {provider_id.url}" if provider_id.url else ""
            lines.append(f"  - {provider_id.provider}{namespace}: `{provider_id.identifier}`{url}")
    else:
        lines.append("  - none")

    lines.append("- Data availability:")
    if instrument.data_availability:
        for availability in instrument.data_availability:
            provider_identifier = (
                f"; provider ID `{availability.provider_identifier}`"
                if availability.provider_identifier
                else ""
            )
            notes = f"; notes: {availability.notes}" if availability.notes else ""
            lines.append(
                f"  - {availability.provider} {availability.data_type}: "
                f"{availability.status.value}; checked {availability.checked_at.isoformat()}"
                f"{provider_identifier}{notes}"
            )
    else:
        lines.append("  - none")

    lines.append("- Tradability/access evidence:")
    if instrument.tradability_evidence:
        for evidence in instrument.tradability_evidence:
            trace_parts = _format_trace_parts(
                (
                    ("URL", evidence.source_url),
                    ("permalink", evidence.permalink),
                    ("raw", evidence.raw_identifier),
                )
            )
            notes = f"; notes: {evidence.notes}" if evidence.notes else ""
            lines.append(
                f"  - {evidence.provider}: {evidence.status.value}; "
                f"retrieved {evidence.retrieved_at.isoformat()}{trace_parts}{notes}"
            )
    else:
        lines.append("  - none")

    lines.append("- Related instruments/proxies:")
    if instrument.related_instruments:
        for related in instrument.related_instruments:
            lines.append(
                f"  - `{related.instrument_id}` {related.relationship}: "
                f"{related.rationale}; evidence {_format_code_list(related.evidence_ids)}"
            )
    else:
        lines.append("  - none")
    return lines


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


def _render_evidence_ledger(evidence_sources: tuple[SourceEvidence, ...]) -> list[str]:
    lines: list[str] = []
    for evidence in evidence_sources:
        provenance = evidence.provenance
        title = f": {evidence.title}" if evidence.title else ""
        source_url = provenance.source_url or evidence.permalink
        permalink = provenance.permalink or evidence.permalink
        raw_artifact_ids = _evidence_raw_artifact_ids(evidence)
        lines.extend(
            [
                (
                    f"- `{evidence.evidence_id}` {evidence.source_kind.value} "
                    f"via {provenance.provider_name}{title}"
                ),
                f"  - URL: {source_url or 'none'}",
                f"  - Permalink: {permalink or 'none'}",
                f"  - Fetched: {provenance.fetched_at.isoformat()}",
                f"  - Observed: {_format_optional_datetime(provenance.observed_at)}",
                f"  - Freshness: {provenance.freshness_status.value}",
                f"  - Instrument IDs: {_format_code_list(_evidence_instrument_ids(evidence))}",
                f"  - Tickers: {_format_code_list(_evidence_tickers(evidence))}",
                f"  - Raw identifier: {_format_optional_code(provenance.raw_identifier)}",
                f"  - Raw snapshot/artifact IDs: {_format_code_list(raw_artifact_ids)}",
            ]
        )
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


def _format_code_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "none"


def _format_optional_code(value: str | None) -> str:
    return f"`{value}`" if value else "none"


def _format_optional_datetime(value: datetime | None) -> str:
    if value is not None:
        return value.isoformat()
    return "none"


def _format_resolution_matches(matches: tuple[Instrument, ...]) -> str:
    return ", ".join(
        f"`{match.instrument_id}` {match.symbol} ({match.asset_class.value})" for match in matches
    )


def _format_trace_parts(parts: tuple[tuple[str, str | None], ...]) -> str:
    formatted = [
        f"{label} {value}" if label != "raw" else f"raw `{value}`"
        for label, value in parts
        if value
    ]
    return f"; {'; '.join(formatted)}" if formatted else ""


def _evidence_instrument_ids(evidence: SourceEvidence) -> tuple[str, ...]:
    values = (
        *((evidence.instrument_id,) if evidence.instrument_id else ()),
        *evidence.matched_instrument_ids,
    )
    return tuple(dict.fromkeys(values))


def _evidence_tickers(evidence: SourceEvidence) -> tuple[str, ...]:
    values = (
        *((evidence.ticker,) if evidence.ticker else ()),
        *evidence.matched_tickers,
    )
    return tuple(dict.fromkeys(values))


def _evidence_raw_artifact_ids(evidence: SourceEvidence) -> tuple[str, ...]:
    values: list[str] = []
    if evidence.provenance.raw_snapshot_id:
        values.append(evidence.provenance.raw_snapshot_id)
    for metadata in (evidence.metadata, evidence.provenance.provider_metadata):
        for key in (
            "artifact_id",
            "artifact_ids",
            "raw_artifact_id",
            "raw_artifact_ids",
            "raw_snapshot_id",
        ):
            value = metadata.get(key)
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, (tuple, list)):
                values.extend(str(item) for item in value if item)
    return tuple(dict.fromkeys(values))


__all__ = ["render_markdown_report"]
