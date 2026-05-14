"""Markdown report renderer."""

from __future__ import annotations

from datetime import datetime
from html import escape

from nlp_stock_prediction.contracts.analysis import (
    AnalysisComponent,
    FundamentalAnalysis,
    MacroContext,
    MetricValue,
    SectorContext,
    TechnicalAnalysis,
    TechnicalMlSignal,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.instruments import Instrument, InstrumentResolution
from nlp_stock_prediction.contracts.provenance import (
    EvidenceReference,
    ProviderHealth,
    ProviderWarning,
)
from nlp_stock_prediction.contracts.report import (
    AuditArtifact,
    AuditManifest,
    DailyReport,
    MaterialClaimTrace,
    PredictionCandidate,
    PriorOutcomeReview,
    ReportSourceReference,
)
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
        "## Report Metadata",
        "",
        f"- Report schema: `{_markdown_code(report.schema_version)}`",
        f"- Timezone: `{_markdown_code(report.timezone)}`",
        f"- App version: {_format_optional_code(report.app_version)}",
        f"- Git SHA: {_format_optional_code(report.git_sha)}",
        f"- Config hash: {_format_optional_code(report.config_hash)}",
        f"- Command args: {_format_metadata(report.command_args)}",
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
        "## Provider Health",
        "",
    ]
    if report.provider_health:
        for health in report.provider_health:
            lines.extend(_render_provider_health(health))
    else:
        lines.append("- No provider health records available.")
    lines.extend(["", "## Provider Warnings", ""])
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
        lines.extend(_render_insufficient_evidence(report))

    lines.extend(["## Prior-Outcome Review", ""])
    if report.prior_outcome_reviews:
        for review in report.prior_outcome_reviews:
            lines.extend(_render_prior_outcome_review(review))
    else:
        lines.append("- No prior-outcome review records available.")
    lines.append("")

    lines.extend(["## Material Claim Traceability", ""])
    if report.material_claim_traces:
        for trace in report.material_claim_traces:
            lines.extend(_render_material_claim_trace(trace))
    else:
        lines.append("- No material claim trace records available.")
    lines.append("")

    lines.extend(["## Report Source References", ""])
    if report.source_references:
        for source_reference in report.source_references:
            lines.extend(_render_source_reference(source_reference))
    else:
        lines.append("- No report source references available.")
    lines.append("")

    lines.extend(["## Evidence Ledger", ""])
    if report.evidence_sources:
        lines.extend(_render_evidence_ledger(report.evidence_sources))
    else:
        lines.append("- No source evidence records available.")
    lines.append("")

    lines.extend(["## Audit Artifacts", ""])
    manifest = report.audit_manifest
    if isinstance(manifest, AuditManifest):
        lines.extend(_render_audit_manifest_details(manifest))
    if view.audit_artifacts:
        for artifact in view.audit_artifacts:
            lines.extend(_render_audit_artifact(artifact))
    elif isinstance(manifest, AuditManifest):
        lines.append("- No audit artifact files listed in manifest.")
    elif view.audit_reference is not None:
        audit_reference = view.audit_reference
        digest = (
            f", sha256 `{_markdown_code(audit_reference.sha256)}`" if audit_reference.sha256 else ""
        )
        path = f", path {_markdown_text(audit_reference.path)}" if audit_reference.path else ""
        lines.append(
            f"- `{_markdown_code(audit_reference.reference_id)}` "
            f"({_markdown_text(audit_reference.reference_type)}){path}{digest}"
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
        f"- `{_markdown_code(candidate.candidate_id)}` ({_markdown_text(candidate.symbol)})",
        (
            f"  - Status: {candidate.status.value}; direction: {candidate.direction.value}; "
            f"type: {candidate.prediction_type.value}; horizon: {candidate.horizon.value}; "
            f"confidence {candidate.confidence:.2f}"
        ),
        f"  - Report-authored scenario: {_markdown_text(candidate.thesis)}",
        f"  - Baseline: {_markdown_text(candidate.baseline)}",
        (
            f"  - Evidence for: {_format_evidence_ids(candidate.evidence_for)} "
            "(observed source claims)"
        ),
        (
            f"  - Evidence against: {_format_evidence_ids(candidate.evidence_against)} "
            "(observed source claims)"
        ),
        f"  - Dissenting evidence: {_format_dissenting_evidence(candidate)}",
        f"  - Assumptions: {_format_list(candidate.assumptions)}",
        f"  - Uncertainties: {_format_list(candidate.uncertainties)}",
        f"  - Uncertainty drivers: {_format_uncertainty_drivers(candidate)}",
        f"  - What would change: {_format_change_triggers(candidate)}",
        f"  - Change trigger limitations: {_format_list(candidate.change_trigger_limitations)}",
        f"  - Prior outcome reviews: {_format_code_list(candidate.prior_outcome_review_ids)}",
    ]
    signal_artifacts = _format_signal_artifacts(candidate)
    if signal_artifacts != "none":
        lines.append(f"  - Signal artifacts: {signal_artifacts}")
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


def _render_insufficient_evidence(report: DailyReport) -> list[str]:
    if report.insufficient_evidence is None:
        return [
            _markdown_text(report.insufficient_evidence_summary)
            or "No prediction scenarios have enough evidence for this report.",
            "",
        ]
    insufficient = report.insufficient_evidence
    lines = [
        _markdown_text(insufficient.summary),
        "",
        f"- Blocking reasons: {_format_list(insufficient.blocking_reasons)}",
        f"- Missing evidence types: {_format_list(insufficient.missing_evidence_types)}",
        f"- Providers: {_format_code_list(insufficient.provider_names)}",
        f"- Evidence: {_format_evidence_ids(insufficient.evidence)}",
        f"- Artifacts: {_format_code_list(insufficient.artifact_ids)}",
        f"- Metadata: {_format_metadata(insufficient.metadata)}",
        "",
    ]
    return lines


def _render_prior_outcome_review(review: PriorOutcomeReview) -> list[str]:
    original_report_date = (
        review.original_report_date.isoformat() if review.original_report_date else "none"
    )
    return [
        f"- `{_markdown_code(review.review_id)}` "
        f"{_markdown_text(review.status)}: "
        f"{_markdown_text(review.summary)}",
        f"  - Candidate: {_format_optional_code(review.candidate_id)}",
        f"  - Instrument: {_format_optional_code(review.instrument_id)}",
        f"  - Original report date: {original_report_date}",
        f"  - Horizon: {review.horizon.value}",
        f"  - Reviewed at: {_format_optional_datetime(review.reviewed_at)}",
        f"  - Outcome evidence: {_format_evidence_ids(review.outcome_evidence)}",
        f"  - Artifacts: {_format_code_list(review.artifact_ids)}",
        f"  - Limitations: {_format_list(review.limitations)}",
        f"  - Metadata: {_format_metadata(review.metadata)}",
    ]


def _render_material_claim_trace(trace: MaterialClaimTrace) -> list[str]:
    return [
        f"- `{_markdown_code(trace.claim_id)}` "
        f"{_markdown_text(trace.claim_type)}: "
        f"{_markdown_text(trace.claim)}",
        f"  - Evidence: {_format_evidence_ids(trace.evidence)}",
        f"  - Artifacts: {_format_code_list(trace.artifact_ids)}",
        f"  - Source references: {_format_code_list(trace.source_reference_ids)}",
        f"  - Candidates: {_format_code_list(trace.candidate_ids)}",
        (f"  - Prior outcome reviews: {_format_code_list(trace.prior_outcome_review_ids)}"),
        f"  - Providers: {_format_code_list(trace.provider_names)}",
        f"  - Rationale: {_markdown_text(trace.rationale or 'none')}",
        f"  - Metadata: {_format_metadata(trace.metadata)}",
    ]


def _render_source_reference(reference: ReportSourceReference) -> list[str]:
    return [
        f"- `{_markdown_code(reference.reference_id)}` "
        f"{_markdown_text(reference.reference_type)}: "
        f"{_markdown_text(reference.label)}",
        f"  - Evidence IDs: {_format_code_list(reference.evidence_ids)}",
        f"  - Artifact IDs: {_format_code_list(reference.artifact_ids)}",
        f"  - Providers: {_format_code_list(reference.provider_names)}",
        f"  - Candidates: {_format_code_list(reference.candidate_ids)}",
        (f"  - Prior outcome reviews: {_format_code_list(reference.prior_outcome_review_ids)}"),
        f"  - Metadata: {_format_metadata(reference.metadata)}",
    ]


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
                f"  - Provider metadata: {_format_metadata(provenance.provider_metadata)}",
                f"  - Evidence metadata: {_format_metadata(evidence.metadata)}",
            ]
        )
    return lines


def _render_provider_health(health: ProviderHealth) -> list[str]:
    lines = [
        f"- `{_markdown_code(health.provider_name)}`: status {health.status.value}; "
        f"credentials {health.credential_state.value}",
        f"  - Checked at: {_format_optional_datetime(health.checked_at)}",
        f"  - Last success: {_format_optional_datetime(health.last_success_at)}",
        f"  - Rate limit remaining: {_format_optional_int(health.rate_limit_remaining)}",
        f"  - Rate limit reset: {_format_optional_datetime(health.rate_limit_reset_at)}",
        f"  - Warning count: {len(health.warnings)}",
    ]
    for warning in health.warnings:
        lines.append(
            f"  - Warning `{_markdown_code(warning.code.value)}` "
            f"{warning.severity.value}: {_markdown_text(warning.message)}"
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
    if component.metrics:
        lines.append(f"- Metrics: {_format_metrics(component.metrics)}")
    if component.warnings:
        lines.append(f"- Warnings: {_format_provider_warnings(component.warnings)}")
    if component.assumptions:
        lines.append(f"- Assumptions: {_format_list(component.assumptions)}")
    return lines


def _render_evidence_refs(evidence: tuple[EvidenceReference, ...]) -> list[str]:
    lines: list[str] = []
    for reference in evidence:
        quote = f" - {_markdown_text(reference.quote)}" if reference.quote else ""
        lines.append(f"- `{_markdown_code(reference.evidence_id)}`{quote}")
    return lines


def _render_ml_signal(ml_signal: TechnicalMlSignal) -> list[str]:
    line = (
        "- ML signal: "
        f"{ml_signal.signal.value}; "
        f"probability {ml_signal.probability_positive:.2f}; "
        f"confidence {ml_signal.calibrated_confidence:.2f}; "
        f"status {ml_signal.status}; "
        f"freshness {ml_signal.freshness_status.value}; "
        f"as of {_markdown_text(ml_signal.as_of)}; "
        f"feature end {_markdown_text(ml_signal.feature_end)}; "
        f"horizon sessions {ml_signal.prediction_horizon_sessions}."
    )
    lines = [
        line,
        f"- ML model hash: `{_markdown_code(ml_signal.model_hash)}`; "
        f"dataset hash `{_markdown_code(ml_signal.dataset_hash)}`",
    ]
    if ml_signal.source_artifact_id or ml_signal.source_artifact_sha256:
        lines.append(
            f"- ML source artifact: {_format_optional_code(ml_signal.source_artifact_id)}; "
            f"sha256 {_format_optional_code(ml_signal.source_artifact_sha256)}"
        )
    if ml_signal.expected_return is not None:
        lines.append(f"- ML expected return: {ml_signal.expected_return}")
    if ml_signal.forecast_interval_width is not None:
        lines.append(f"- ML forecast interval width: {ml_signal.forecast_interval_width}")
    if ml_signal.validation_accuracy is not None:
        lines.append(f"- ML validation accuracy: {ml_signal.validation_accuracy:.2f}")
    if ml_signal.validation_brier_score is not None:
        lines.append(f"- ML validation brier score: {ml_signal.validation_brier_score:.4f}")
    if ml_signal.warning_ids:
        lines.append(f"- ML warning IDs: {_format_code_list(ml_signal.warning_ids)}")
    if ml_signal.metadata:
        lines.append(f"- ML metadata: {_format_metadata(ml_signal.metadata)}")
    return lines


def _format_evidence_ids(evidence: tuple[EvidenceReference, ...]) -> str:
    if not evidence:
        return "none"
    return ", ".join(f"`{_markdown_code(reference.evidence_id)}`" for reference in evidence)


def _format_dissenting_evidence(candidate: PredictionCandidate) -> str:
    if not candidate.dissenting_evidence:
        return "none"
    parts = []
    for dissent in candidate.dissenting_evidence:
        parts.append(
            f"{dissent.impact}: {_markdown_text(dissent.summary)} "
            f"({_format_evidence_ids(dissent.evidence)})"
        )
    return "; ".join(parts)


def _format_uncertainty_drivers(candidate: PredictionCandidate) -> str:
    if not candidate.uncertainty_drivers:
        return "none"
    return "; ".join(
        f"`{_markdown_code(driver.driver_id)}` {driver.severity}: {_markdown_text(driver.summary)}"
        for driver in candidate.uncertainty_drivers
    )


def _format_change_triggers(candidate: PredictionCandidate) -> str:
    if not candidate.change_triggers:
        return "none"
    return "; ".join(
        f"`{_markdown_code(trigger.trigger_id)}` {trigger.trigger_type}: "
        f"{_markdown_text(trigger.summary)}"
        for trigger in candidate.change_triggers
    )


def _format_signal_artifacts(candidate: PredictionCandidate) -> str:
    typed = {reference.artifact_id: reference for reference in candidate.signal_artifacts}
    parts = [
        f"{reference.family.value}: `{_markdown_code(reference.artifact_id)}`"
        for reference in candidate.signal_artifacts
    ]
    for artifact_id in candidate.signal_artifact_ids:
        if artifact_id not in typed:
            parts.append(f"legacy: `{_markdown_code(artifact_id)}`")
    return "; ".join(parts) if parts else "none"


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


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _format_metadata(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(
        f"{_markdown_text(key)}={_markdown_text(metadata_value)}"
        for key, metadata_value in sorted(value.items())
    )


def _format_metrics(metrics: tuple[MetricValue, ...]) -> str:
    if not metrics:
        return "none"
    parts: list[str] = []
    for metric in metrics:
        unit = f" {metric.unit}" if metric.unit else ""
        as_of = f" as of {_markdown_text(metric.as_of)}" if metric.as_of is not None else ""
        parts.append(f"{_markdown_text(metric.name)}={_markdown_text(metric.value)}{unit}{as_of}")
    return "; ".join(parts)


def _format_provider_warnings(warnings: tuple[ProviderWarning, ...]) -> str:
    if not warnings:
        return "none"
    return "; ".join(
        f"{warning.severity.value} {warning.code.value}: {_markdown_text(warning.message)}"
        for warning in warnings
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


def _render_audit_manifest_details(manifest: AuditManifest) -> list[str]:
    lines = [
        f"- Manifest schema: `{_markdown_code(manifest.schema_version)}`",
        f"- Created at: {manifest.created_at.isoformat()}",
        f"- Provider run IDs: {_format_code_list(manifest.provider_run_ids)}",
        f"- Provider health: {_format_audit_provider_health(manifest.provider_health)}",
        f"- Model versions: {_format_metadata(manifest.model_versions)}",
        f"- Prompt versions: {_format_metadata(manifest.prompt_versions)}",
        f"- Config hash: {_format_optional_code(manifest.config_hash)}",
        f"- Command args: {_format_metadata(manifest.command_args)}",
        f"- Prediction trace IDs: {_format_code_list(manifest.prediction_trace_ids)}",
    ]
    return lines


def _format_audit_provider_health(provider_health: tuple[ProviderHealth, ...]) -> str:
    if not provider_health:
        return "none"
    return "; ".join(
        f"`{_markdown_code(health.provider_name)}` {health.status.value} "
        f"warnings={len(health.warnings)}"
        for health in provider_health
    )


def _render_audit_artifact(artifact: AuditArtifact) -> list[str]:
    digest = f", sha256 `{_markdown_code(artifact.sha256)}`" if artifact.sha256 else ""
    count = f", records {artifact.record_count}" if artifact.record_count is not None else ""
    return [
        (
            f"- `{_markdown_code(artifact.artifact_id)}` "
            f"({_markdown_text(artifact.artifact_type)}): "
            f"{_markdown_text(artifact.path)}{count}{digest}"
        ),
        f"  - Created at: {artifact.created_at.isoformat()}",
        f"  - Produced by: {_markdown_text(artifact.produced_by)}",
        f"  - Metadata: {_format_metadata(artifact.metadata)}",
    ]


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
