"""Markdown report renderer."""

from __future__ import annotations

from datetime import datetime
from html import escape

from nlp_stock_prediction.contracts.analysis import (
    AnalysisComponent,
    FundamentalAnalysis,
    MacroContext,
    SectorContext,
    TechnicalAnalysis,
    TechnicalMlSignal,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.instruments import Instrument, InstrumentResolution
from nlp_stock_prediction.contracts.provenance import EvidenceReference
from nlp_stock_prediction.contracts.report import DailyReport, PredictionCandidate
from nlp_stock_prediction.reporting.view import ReportView


def render_markdown_report(report: DailyReport) -> str:
    view = ReportView.from_report(report)
    lines: list[str] = [
        "# Prediction Research Report",
        "",
        f"Report date: {report.report_date.isoformat()}",
        f"Generated at: {report.generated_at.isoformat()}",
        f"Run ID: `{_markdown_code(report.run_id)}`",
        f"Objective: {_markdown_text(report.objective)}",
        f"Universe: {_markdown_text(report.universe)}",
        f"Data freshness: {_markdown_text(report.data_freshness.summary)}",
        "",
        "## Research Objective",
        "",
        _markdown_text(report.objective),
        "",
        "## Data Freshness",
        "",
        f"- As of: {report.data_freshness.as_of.isoformat()}",
        f"- Summary: {_markdown_text(report.data_freshness.summary)}",
        f"- Stale providers: {_format_list(report.data_freshness.stale_provider_names)}",
        f"- Missing providers: {_format_list(report.data_freshness.missing_provider_names)}",
        "",
        "## Provider Warnings",
        "",
    ]
    if view.provider_warnings:
        for warning in view.provider_warnings:
            provider = warning.provider_name or "unknown-provider"
            lines.append(
                f"- `{_markdown_code(provider)}` {warning.severity.value}: "
                f"[{warning.code.value}] {_markdown_text(warning.message)}"
            )
    else:
        lines.append("- No provider warnings.")

    if report.instrument_resolutions:
        lines.extend(["", "## Universe Resolution", ""])
        lines.extend(_render_universe_resolution(report.instrument_resolutions))

    lines.extend(["", "## Instrument Sections", ""])
    for section in report.instrument_sections:
        instrument = view.instrument_for_section(section)
        lines.extend(
            [
                f"### {_markdown_text(section.symbol)}",
                "",
                f"Name: {_markdown_text(section.display_name or 'Unknown')}",
                "",
                "#### Identity & Availability",
                "",
            ]
        )
        if instrument is not None:
            lines.extend(_render_instrument_identity(instrument))
        else:
            lines.append(f"- Instrument ID: `{_markdown_code(section.instrument_id)}`")
        lines.extend(
            [
                "",
                "#### Observed Evidence",
                "",
                _markdown_text(
                    section.observed_discussion_summary
                    or "No observed discussion summary available."
                ),
                "",
                "Social/news:",
                "",
                _markdown_text(section.social_news_summary or "No social/news summary available."),
                "",
                "Strategy clusters:",
                "",
            ]
        )
        if section.strategy_clusters:
            for cluster in section.strategy_clusters:
                lines.append(
                    f"- `{_markdown_code(cluster.cluster_id)}` "
                    f"{cluster.direction.value}/{cluster.instrument.value}/"
                    f"{cluster.time_horizon.value}; confidence {cluster.confidence:.2f}; "
                    f"members {_format_code_list(cluster.member_strategy_ids)}; "
                    f"evidence {_format_evidence_ids(cluster.evidence)}"
                )
                if cluster.catalyst_summary:
                    lines.append(f"  - Catalyst: {_markdown_text(cluster.catalyst_summary)}")
        else:
            lines.append("- No strategy clusters available.")
        lines.extend(
            [
                "",
                f"Data quality: {_format_metadata(section.data_quality)}",
                "",
                "#### Analysis",
                "",
                _markdown_text(section.analysis_summary or "No analysis summary available."),
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
        section_candidates = view.candidates_for_section(section)
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
                _markdown_text(report.insufficient_evidence_summary)
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
    if view.audit_artifacts:
        for artifact in view.audit_artifacts:
            digest = f", sha256 `{_markdown_code(artifact.sha256)}`" if artifact.sha256 else ""
            count = (
                f", records {artifact.record_count}" if artifact.record_count is not None else ""
            )
            lines.append(
                f"- `{_markdown_code(artifact.artifact_id)}` "
                f"({_markdown_text(artifact.artifact_type)}): "
                f"{_markdown_text(artifact.path)}{count}{digest}"
            )
    elif view.audit_reference is not None:
        reference = view.audit_reference
        digest = f", sha256 `{_markdown_code(reference.sha256)}`" if reference.sha256 else ""
        path = f", path {_markdown_text(reference.path)}" if reference.path else ""
        lines.append(
            f"- `{_markdown_code(reference.reference_id)}` "
            f"({_markdown_text(reference.reference_type)}){path}{digest}"
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
            f"; selected `{_markdown_code(resolution.selected_instrument_id)}`"
            if resolution.selected_instrument_id
            else ""
        )
        lines.append(f"- `{_markdown_code(resolution.query)}`: {resolution.status.value}{selected}")
        if resolution.matches:
            lines.append(f"  - Matches: {_format_resolution_matches(resolution.matches)}")
        else:
            lines.append("  - Matches: none")
        lines.append(f"  - Warnings: {_format_list(resolution.warnings)}")
    return lines


def _render_instrument_identity(instrument: Instrument) -> list[str]:
    lines = [
        f"- Instrument ID: `{_markdown_code(instrument.instrument_id)}`",
        f"- Asset class: {instrument.asset_class.value}",
        f"- Venue: {instrument.venue or 'none'}",
        f"- Aliases: {_format_code_list(instrument.aliases)}",
        "- Provider IDs:",
    ]
    if instrument.provider_ids:
        for provider_id in instrument.provider_ids:
            namespace = f"/{_markdown_text(provider_id.namespace)}" if provider_id.namespace else ""
            url = f"; URL: {_markdown_text(provider_id.url)}" if provider_id.url else ""
            lines.append(
                f"  - {_markdown_text(provider_id.provider)}{namespace}: "
                f"`{_markdown_code(provider_id.identifier)}`{url}"
            )
    else:
        lines.append("  - none")

    lines.append("- Data availability:")
    if instrument.data_availability:
        for availability in instrument.data_availability:
            provider_identifier = (
                f"; provider ID `{_markdown_code(availability.provider_identifier)}`"
                if availability.provider_identifier
                else ""
            )
            notes = f"; notes: {_markdown_text(availability.notes)}" if availability.notes else ""
            lines.append(
                f"  - {_markdown_text(availability.provider)} "
                f"{_markdown_text(availability.data_type)}: "
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
            notes = f"; notes: {_markdown_text(evidence.notes)}" if evidence.notes else ""
            lines.append(
                f"  - {_markdown_text(evidence.provider)}: {evidence.status.value}; "
                f"retrieved {evidence.retrieved_at.isoformat()}{trace_parts}{notes}"
            )
    else:
        lines.append("  - none")

    lines.append("- Related instruments/proxies:")
    if instrument.related_instruments:
        for related in instrument.related_instruments:
            lines.append(
                f"  - `{_markdown_code(related.instrument_id)}` "
                f"{_markdown_text(related.relationship)}: "
                f"{_markdown_text(related.rationale)}; "
                f"evidence {_format_code_list(related.evidence_ids)}"
            )
    else:
        lines.append("  - none")
    return lines


def _render_candidate(candidate: PredictionCandidate) -> list[str]:
    lines = [
        f"- `{_markdown_code(candidate.candidate_id)}` "
        f"({_markdown_text(candidate.symbol)}): {_markdown_text(candidate.thesis)}",
        (
            f"  - Status: {candidate.status.value}; direction: {candidate.direction.value}; "
            f"horizon: {candidate.horizon.value}; confidence {candidate.confidence:.2f}"
        ),
        f"  - Baseline: {_markdown_text(candidate.baseline)}",
        f"  - Evidence for: {_format_evidence_ids(candidate.evidence_for)}",
        f"  - Evidence against: {_format_evidence_ids(candidate.evidence_against)}",
        f"  - Assumptions: {_format_list(candidate.assumptions)}",
        f"  - Uncertainties: {_format_list(candidate.uncertainties)}",
    ]
    if candidate.signal_artifact_ids:
        lines.append(f"  - Signal artifacts: {_format_code_list(candidate.signal_artifact_ids)}")
    lines.extend(_render_candidate_evaluation_metadata(candidate))
    lines.append("")
    return lines


def _render_candidate_evaluation_metadata(candidate: PredictionCandidate) -> list[str]:
    metadata = candidate.metadata.get("prediction_evaluation")
    if not isinstance(metadata, dict):
        return []
    label = metadata.get("quality_label")
    score = metadata.get("score")
    baseline_verdict = metadata.get("baseline_verdict")
    artifact_id = metadata.get("artifact_id")
    parts: list[str] = []
    if isinstance(label, str) and label.strip():
        parts.append(_markdown_text(label))
    if isinstance(score, int | float):
        parts.append(f"score {score:.2f}")
    if isinstance(baseline_verdict, str) and baseline_verdict.strip():
        parts.append(f"baseline {baseline_verdict}")
    if not parts:
        return []
    line = f"  - Evaluation quality: {'; '.join(parts)}"
    if isinstance(artifact_id, str) and artifact_id.strip():
        line = f"{line}; artifact `{_markdown_code(artifact_id)}`"
    return [line]


def _render_evidence_ledger(evidence_sources: tuple[SourceEvidence, ...]) -> list[str]:
    lines: list[str] = []
    for evidence in evidence_sources:
        provenance = evidence.provenance
        title = f": {_markdown_text(evidence.title)}" if evidence.title else ""
        source_url = provenance.source_url or evidence.permalink
        permalink = provenance.permalink or evidence.permalink
        raw_artifact_ids = _evidence_raw_artifact_ids(evidence)
        lines.extend(
            [
                (
                    f"- `{_markdown_code(evidence.evidence_id)}` {evidence.source_kind.value} "
                    f"via {_markdown_text(provenance.provider_name)}{title}"
                ),
                f"  - URL: {_markdown_text(source_url or 'none')}",
                f"  - Permalink: {_markdown_text(permalink or 'none')}",
                f"  - Fetched: {provenance.fetched_at.isoformat()}",
                f"  - Published/created: {_format_optional_datetime(evidence.created_at)}",
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
        f"- Summary: {_markdown_text(component.summary)}",
        f"- Signal: {component.signal.value}; confidence {component.confidence:.2f}",
    ]
    if isinstance(component, TechnicalAnalysis):
        if component.trend:
            lines.append(f"- Trend: {_markdown_text(component.trend)}")
        if component.ml_signal is not None:
            lines.extend(_render_ml_signal(component.ml_signal))
    elif isinstance(component, FundamentalAnalysis):
        if component.valuation_summary:
            lines.append(f"- Valuation: {_markdown_text(component.valuation_summary)}")
    elif isinstance(component, SectorContext):
        if component.sector:
            lines.append(f"- Sector: {_markdown_text(component.sector)}")
    elif isinstance(component, MacroContext):
        if component.supportive_factors:
            lines.append(f"- Supportive factors: {_format_list(component.supportive_factors)}")
        if component.conflicting_factors:
            lines.append(f"- Conflicting factors: {_format_list(component.conflicting_factors)}")
    if component.evidence:
        lines.append(f"- Evidence: {_format_evidence_ids(component.evidence)}")
    return lines


def _render_evidence_refs(evidence: tuple[EvidenceReference, ...]) -> list[str]:
    lines: list[str] = []
    for reference in evidence:
        quote = f" - {_markdown_text(reference.quote)}" if reference.quote else ""
        lines.append(f"- `{_markdown_code(reference.evidence_id)}`{quote}")
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
    return ", ".join(f"`{_markdown_code(reference.evidence_id)}`" for reference in evidence)


def _format_list(values: tuple[str, ...]) -> str:
    return ", ".join(_markdown_text(value) for value in values) if values else "none"


def _format_code_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"`{_markdown_code(value)}`" for value in values) if values else "none"


def _format_optional_code(value: str | None) -> str:
    return f"`{_markdown_code(value)}`" if value else "none"


def _format_optional_datetime(value: datetime | None) -> str:
    if value is not None:
        return value.isoformat()
    return "none"


def _format_metadata(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(
        f"{_markdown_text(key)}={_markdown_text(metadata_value)}"
        for key, metadata_value in sorted(value.items())
    )


def _format_resolution_matches(matches: tuple[Instrument, ...]) -> str:
    return ", ".join(
        f"`{_markdown_code(match.instrument_id)}` "
        f"{_markdown_text(match.symbol)} ({match.asset_class.value})"
        for match in matches
    )


def _format_trace_parts(parts: tuple[tuple[str, str | None], ...]) -> str:
    formatted = [
        f"{label} {_markdown_text(value)}" if label != "raw" else f"raw `{_markdown_code(value)}`"
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


def _markdown_text(value: object | None) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return escape(text.replace("`", "'"), quote=False)


def _markdown_code(value: object) -> str:
    return escape(str(value).replace("`", "'"), quote=False)


__all__ = ["render_markdown_report"]
