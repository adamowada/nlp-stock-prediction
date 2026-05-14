"""Report and audit artifact contracts."""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.analysis import (
    FundamentalAnalysis,
    MacroContext,
    SectorContext,
    TechnicalAnalysis,
)
from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    Confidence,
    ContractModel,
    JsonObject,
    NonEmptyStr,
)
from nlp_stock_prediction.contracts.enums import (
    Direction,
    PredictionStatus,
    PredictionType,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.evaluation import SignalArtifactReference
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.extraction import StrategyCluster
from nlp_stock_prediction.contracts.instruments import (
    Instrument,
    InstrumentResolution,
    InstrumentSymbol,
)
from nlp_stock_prediction.contracts.provenance import (
    DataReference,
    EvidenceReference,
    ProviderHealth,
)


class DataFreshnessSummary(ContractModel):
    """Freshness summary shown in report headers and JSON."""

    as_of: AwareDatetime
    summary: NonEmptyStr
    stale_provider_names: tuple[str, ...] = Field(default_factory=tuple)
    missing_provider_names: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_report_authored_language(self) -> DataFreshnessSummary:
        _validate_report_authored_language(self.summary)
        return self


class DissentingEvidence(ContractModel):
    """Source-backed evidence that weakens, limits, or contradicts a candidate."""

    summary: NonEmptyStr
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    impact: Literal["weakens", "contradicts", "limits", "neutral_context"] = "limits"
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dissenting_evidence(self) -> DissentingEvidence:
        if not (self.evidence or self.artifact_ids):
            raise ValueError("dissenting evidence requires evidence or artifact references")
        _validate_report_authored_language(self.summary)
        return self


class UncertaintyDriver(ContractModel):
    """One explicit driver of uncertainty in a prediction scenario."""

    driver_id: NonEmptyStr
    summary: NonEmptyStr
    severity: Literal["low", "medium", "high"] = "medium"
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_uncertainty_language(self) -> UncertaintyDriver:
        _validate_report_authored_language(self.summary)
        return self


class PredictionChangeTrigger(ContractModel):
    """Condition or new evidence that would materially change a prediction."""

    trigger_id: NonEmptyStr
    summary: NonEmptyStr
    trigger_type: Literal[
        "new_source_evidence",
        "provider_refresh",
        "dissent_resolved",
        "baseline_change",
        "outcome_data",
        "data_quality",
        "instrument_resolution",
    ]
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    rationale: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_change_trigger(self) -> PredictionChangeTrigger:
        if not (self.evidence or self.artifact_ids or self.rationale):
            raise ValueError("prediction change triggers require evidence, artifacts, or rationale")
        _validate_report_authored_language(self.summary, self.rationale)
        return self


class PriorOutcomeReview(ContractModel):
    """Review of a prior prediction scenario using later available evidence."""

    review_id: NonEmptyStr
    status: Literal["available", "not_available", "pending", "stale", "unavailable"]
    summary: NonEmptyStr
    candidate_id: str | None = None
    instrument_id: str | None = None
    original_report_date: date | None = None
    reviewed_at: AwareDatetime | None = None
    horizon: TimeHorizon = TimeHorizon.UNKNOWN
    outcome_evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prior_outcome_review(self) -> PriorOutcomeReview:
        if self.status == "available" and not (
            self.candidate_id and (self.outcome_evidence or self.artifact_ids)
        ):
            raise ValueError(
                "available prior outcome reviews require a candidate and outcome references"
            )
        if self.status != "available" and not self.limitations:
            raise ValueError("unavailable prior outcome reviews require limitations")
        _validate_report_authored_language(self.summary, *self.limitations)
        return self


class InsufficientEvidenceReport(ContractModel):
    """Structured no-candidate report outcome."""

    status: Literal["insufficient_evidence"] = "insufficient_evidence"
    summary: NonEmptyStr
    blocking_reasons: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    missing_evidence_types: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    provider_names: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_insufficient_evidence(self) -> InsufficientEvidenceReport:
        if not (self.blocking_reasons or self.missing_evidence_types or self.provider_names):
            raise ValueError(
                "insufficient-evidence reports require reasons, missing evidence types, "
                "or provider names"
            )
        _validate_report_authored_language(
            self.summary,
            *self.blocking_reasons,
            *self.missing_evidence_types,
        )
        return self


class ReportSourceReference(ContractModel):
    """Report-level reference to source evidence, artifacts, providers, or prior reviews."""

    reference_id: NonEmptyStr
    label: NonEmptyStr
    reference_type: Literal[
        "source_evidence",
        "tool_artifact",
        "baseline",
        "prediction_evaluation",
        "prior_outcome",
        "provider_health",
        "instrument_resolution",
    ]
    evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    provider_names: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    candidate_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    prior_outcome_review_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference_targets(self) -> ReportSourceReference:
        if not (
            self.evidence_ids
            or self.artifact_ids
            or self.provider_names
            or self.prior_outcome_review_ids
        ):
            raise ValueError("report source references require at least one source target")
        _validate_report_authored_language(self.label)
        return self


class MaterialClaimTrace(ContractModel):
    """Trace for one material report-authored claim."""

    claim_id: NonEmptyStr
    claim: NonEmptyStr
    claim_type: Literal[
        "source_observation",
        "analysis",
        "baseline",
        "prediction_evaluation",
        "prior_outcome",
        "provider_health",
        "instrument_resolution",
        "labeled_inference",
        "risk_policy",
    ]
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    source_reference_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    candidate_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    prior_outcome_review_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    provider_names: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    rationale: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_claim_trace(self) -> MaterialClaimTrace:
        has_references = bool(
            self.evidence
            or self.artifact_ids
            or self.source_reference_ids
            or self.candidate_ids
            or self.prior_outcome_review_ids
            or self.provider_names
        )
        if self.claim_type == "labeled_inference":
            if not self.rationale:
                raise ValueError("labeled inference claim traces require rationale")
        elif not has_references:
            raise ValueError("material claim traces require trace references")
        _validate_report_authored_language(self.claim, self.rationale)
        return self


class PredictionCandidate(ContractModel):
    """Evidence-backed prediction scenario, not a trade instruction."""

    candidate_id: NonEmptyStr
    instrument_id: NonEmptyStr
    symbol: InstrumentSymbol
    prediction_type: PredictionType = PredictionType.DIRECTIONAL
    horizon: TimeHorizon = TimeHorizon.UNKNOWN
    direction: Direction = Direction.UNKNOWN
    status: PredictionStatus
    thesis: NonEmptyStr
    baseline: NonEmptyStr
    confidence: Confidence
    evidence_for: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    evidence_against: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    dissenting_evidence: tuple[DissentingEvidence, ...] = Field(default_factory=tuple)
    assumptions: tuple[str, ...] = Field(default_factory=tuple)
    uncertainties: tuple[str, ...] = Field(default_factory=tuple)
    uncertainty_drivers: tuple[UncertaintyDriver, ...] = Field(default_factory=tuple)
    change_triggers: tuple[PredictionChangeTrigger, ...] = Field(default_factory=tuple)
    change_trigger_limitations: tuple[str, ...] = Field(default_factory=tuple)
    prior_outcome_review_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    signal_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    signal_artifacts: tuple[SignalArtifactReference, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evidence_shape(self) -> PredictionCandidate:
        if self.status == PredictionStatus.EVIDENCE_SUPPORTED and not self.evidence_for:
            raise ValueError("evidence-supported predictions require evidence_for")
        if self.status in {
            PredictionStatus.CONTRADICTED,
            PredictionStatus.INSUFFICIENT_EVIDENCE,
            PredictionStatus.UNAVAILABLE,
        } and not (
            self.evidence_for
            or self.evidence_against
            or self.dissenting_evidence
            or self.uncertainties
            or self.uncertainty_drivers
        ):
            raise ValueError("non-supported predictions require evidence or uncertainty context")
        if not (
            self.evidence_for
            or self.evidence_against
            or self.dissenting_evidence
            or self.uncertainties
            or self.uncertainty_drivers
        ):
            raise ValueError(
                "prediction candidates require supporting/dissenting context or uncertainty"
            )
        if not (self.change_triggers or self.change_trigger_limitations):
            raise ValueError(
                "prediction candidates require change triggers or change trigger limitations"
            )
        typed_signal_ids = tuple(reference.artifact_id for reference in self.signal_artifacts)
        if len(set(typed_signal_ids)) != len(typed_signal_ids):
            raise ValueError("prediction candidate signal artifact references must be unique")
        _validate_report_authored_language(
            self.thesis,
            self.baseline,
            *self.assumptions,
            *self.uncertainties,
            *self.change_trigger_limitations,
        )
        return self


class InstrumentReportSection(ContractModel):
    """One instrument section in the Markdown and JSON report."""

    instrument_id: NonEmptyStr
    symbol: InstrumentSymbol
    display_name: str | None = None
    observed_discussion_summary: str | None = None
    social_news_summary: str | None = None
    strategy_clusters: tuple[StrategyCluster, ...] = Field(default_factory=tuple)
    technical_analysis: TechnicalAnalysis | None = None
    fundamental_analysis: FundamentalAnalysis | None = None
    sector_context: SectorContext | None = None
    macro_context: MacroContext | None = None
    analysis_summary: str | None = None
    prediction_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    warning_ids: tuple[str, ...] = Field(default_factory=tuple)
    data_quality: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_report_authored_language(self) -> InstrumentReportSection:
        _validate_report_authored_language(
            self.observed_discussion_summary,
            self.social_news_summary,
            self.analysis_summary,
        )
        return self


class AuditArtifact(ContractModel):
    """One file written to the report audit directory."""

    artifact_id: NonEmptyStr
    artifact_type: Literal[
        "raw_snapshot",
        "normalized_evidence",
        "extraction_output",
        "analysis_context",
        "prediction_input",
        "markdown_report",
        "json_report",
        "provider_result",
        "market_data",
        "technical_package",
        "ml_forecast",
        "instrument_universe",
        "prediction_evaluation",
        "prediction_outcome",
        "prediction_outcome_evaluation",
        "calibration_summary",
        "signal_family_ablation",
        "walk_forward_evaluation",
        "audit_manifest",
    ]
    path: NonEmptyStr
    created_at: AwareDatetime
    produced_by: NonEmptyStr
    sha256: str | None = None
    record_count: int | None = Field(default=None, ge=0)
    metadata: JsonObject = Field(default_factory=dict)


class AuditManifest(ContractModel):
    """Audit spine for a generated report."""

    run_id: NonEmptyStr
    schema_version: NonEmptyStr
    created_at: AwareDatetime
    artifacts: tuple[AuditArtifact, ...] = Field(default_factory=tuple)
    provider_run_ids: tuple[str, ...] = Field(default_factory=tuple)
    model_versions: JsonObject = Field(default_factory=dict)
    prompt_versions: JsonObject = Field(default_factory=dict)
    config_hash: str | None = None
    command_args: JsonObject = Field(default_factory=dict)
    prediction_trace_ids: tuple[str, ...] = Field(default_factory=tuple)


class MarkdownReportOutline(ContractModel):
    """Markdown renderer contract."""

    schema_version: NonEmptyStr
    heading_order: tuple[NonEmptyStr, ...]
    instrument_section_heading_template: NonEmptyStr
    required_instrument_subsections: tuple[NonEmptyStr, ...]
    final_section_headings: tuple[NonEmptyStr, ...]
    required_footer_headings: tuple[NonEmptyStr, ...]
    require_evidence_references: bool = True
    require_audit_artifacts: bool = True

    @model_validator(mode="after")
    def validate_outline(self) -> MarkdownReportOutline:
        if len(set(self.heading_order)) != len(self.heading_order):
            raise ValueError("Markdown heading_order entries must be unique")
        if "{symbol}" not in self.instrument_section_heading_template:
            raise ValueError("instrument section heading template must include {symbol}")
        if not self.required_instrument_subsections:
            raise ValueError("Markdown outline requires instrument subsections")
        if not self.final_section_headings:
            raise ValueError("Markdown outline requires final section headings")
        return self


DEFAULT_MARKDOWN_REPORT_OUTLINE = MarkdownReportOutline(
    schema_version="markdown-report.v2",
    heading_order=(
        "Prediction Research Report",
        "Research Objective",
        "Data Freshness",
        "Provider Warnings",
        "Instrument Sections",
        "Prediction Scenarios Or Insufficient-Evidence Summary",
        "Prior-Outcome Review",
        "Material Claim Traceability",
        "Report Source References",
        "Evidence Ledger",
        "Audit Artifacts",
    ),
    instrument_section_heading_template="{symbol}",
    required_instrument_subsections=(
        "Observed Evidence",
        "Analysis",
        "Prediction Scenarios",
        "Evidence References",
    ),
    final_section_headings=(
        "Prediction Scenarios",
        "Insufficient-Evidence Summary",
        "Prior-Outcome Review",
        "Material Claim Traceability",
        "Report Source References",
    ),
    required_footer_headings=("Audit Artifacts",),
)


class DailyReport(ContractModel):
    """Top-level Markdown/JSON report contract."""

    schema_version: NonEmptyStr
    run_id: NonEmptyStr
    report_date: date
    generated_at: AwareDatetime
    timezone: NonEmptyStr
    objective: NonEmptyStr
    universe: NonEmptyStr
    app_version: str | None = None
    git_sha: str | None = None
    config_hash: str | None = None
    command_args: JsonObject = Field(default_factory=dict)
    instruments: tuple[Instrument, ...]
    data_freshness: DataFreshnessSummary
    provider_health: tuple[ProviderHealth, ...] = Field(default_factory=tuple)
    evidence_sources: tuple[SourceEvidence, ...] = Field(default_factory=tuple)
    instrument_resolutions: tuple[InstrumentResolution, ...] = Field(default_factory=tuple)
    instrument_sections: tuple[InstrumentReportSection, ...]
    prediction_candidates: tuple[PredictionCandidate, ...] = Field(default_factory=tuple)
    insufficient_evidence: InsufficientEvidenceReport | None = None
    insufficient_evidence_summary: str | None = None
    source_references: tuple[ReportSourceReference, ...] = Field(default_factory=tuple)
    material_claim_traces: tuple[MaterialClaimTrace, ...] = Field(default_factory=tuple)
    prior_outcome_reviews: tuple[PriorOutcomeReview, ...] = Field(default_factory=tuple)
    audit_manifest: AuditManifest | DataReference | None = None

    @model_validator(mode="after")
    def validate_report_shape(self) -> DailyReport:
        _validate_report_authored_language(
            self.objective,
            self.universe,
            self.insufficient_evidence_summary,
        )
        instrument_ids = tuple(instrument.instrument_id for instrument in self.instruments)
        if not instrument_ids:
            raise ValueError("daily reports require at least one instrument")
        if len(set(instrument_ids)) != len(instrument_ids):
            raise ValueError("report instrument ids must be unique")

        section_instrument_ids = tuple(
            section.instrument_id for section in self.instrument_sections
        )
        if len(section_instrument_ids) != len(instrument_ids):
            raise ValueError("instrument sections must cover each report instrument exactly once")
        if section_instrument_ids != instrument_ids:
            raise ValueError("instrument sections must match report instruments in order")
        symbol_by_instrument_id = {
            instrument.instrument_id: instrument.symbol for instrument in self.instruments
        }
        for section in self.instrument_sections:
            if section.symbol != symbol_by_instrument_id[section.instrument_id]:
                raise ValueError("instrument section symbol must match report instrument symbol")

        candidate_ids = tuple(candidate.candidate_id for candidate in self.prediction_candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("prediction candidate ids must be unique")
        if not self.prediction_candidates and self.insufficient_evidence is None:
            raise ValueError(
                "reports without prediction candidates require structured insufficient_evidence"
            )
        if self.prediction_candidates and not self.material_claim_traces:
            raise ValueError("reports with prediction candidates require material_claim_traces")

        source_evidence_ids = tuple(evidence.evidence_id for evidence in self.evidence_sources)
        if len(set(source_evidence_ids)) != len(source_evidence_ids):
            raise ValueError("report evidence_sources ids must be unique")
        evidence_by_id = {evidence.evidence_id: evidence for evidence in self.evidence_sources}
        artifact_ids = _audit_artifact_ids(self.audit_manifest)
        referenced_artifact_ids: set[str] = set()

        selected_resolution_ids = {
            resolution.selected_instrument_id
            for resolution in self.instrument_resolutions
            if resolution.selected_instrument_id is not None
        }
        if selected_resolution_ids.difference(instrument_ids):
            raise ValueError(
                "instrument_resolutions selected ids must reference report instruments"
            )

        candidate_by_id = {
            candidate.candidate_id: candidate for candidate in self.prediction_candidates
        }
        source_reference_ids = tuple(reference.reference_id for reference in self.source_references)
        if len(set(source_reference_ids)) != len(source_reference_ids):
            raise ValueError("report source reference ids must be unique")
        prior_outcome_review_ids = tuple(review.review_id for review in self.prior_outcome_reviews)
        if len(set(prior_outcome_review_ids)) != len(prior_outcome_review_ids):
            raise ValueError("prior outcome review ids must be unique")
        claim_trace_ids = tuple(trace.claim_id for trace in self.material_claim_traces)
        if len(set(claim_trace_ids)) != len(claim_trace_ids):
            raise ValueError("material claim trace ids must be unique")

        cited_evidence_ids: set[str] = set()
        related_instrument_evidence_ids: set[str] = set()
        for instrument in self.instruments:
            for related_instrument in instrument.related_instruments:
                related_instrument_evidence_ids.update(related_instrument.evidence_ids)
        cited_evidence_ids.update(related_instrument_evidence_ids)

        section_references: dict[str, str] = {}
        for section in self.instrument_sections:
            cited_evidence_ids.update(reference.evidence_id for reference in section.evidence)
            for cluster in section.strategy_clusters:
                cited_evidence_ids.update(reference.evidence_id for reference in cluster.evidence)
            for component in (
                section.technical_analysis,
                section.fundamental_analysis,
                section.sector_context,
                section.macro_context,
            ):
                if component is not None:
                    cited_evidence_ids.update(_analysis_evidence_ids(component))
            for candidate_id in section.prediction_candidate_ids:
                if candidate_id not in candidate_by_id:
                    raise ValueError(
                        "instrument section prediction_candidate_ids must reference candidates"
                    )
                if candidate_id in section_references:
                    raise ValueError(
                        "prediction candidates must be referenced by exactly one instrument section"
                    )
                section_references[candidate_id] = section.instrument_id

        valid_instrument_ids = set(instrument_ids)
        for candidate in self.prediction_candidates:
            if candidate.instrument_id not in valid_instrument_ids:
                raise ValueError("prediction candidates must reference report instruments")
            if candidate.symbol != symbol_by_instrument_id[candidate.instrument_id]:
                raise ValueError("prediction candidate symbol must match report instrument symbol")
            _validate_candidate_evaluation_metadata(candidate)
            if section_references.get(candidate.candidate_id) != candidate.instrument_id:
                raise ValueError(
                    "instrument section prediction_candidate_ids must match candidate instrument_id"
                )
            cited_evidence_ids.update(reference.evidence_id for reference in candidate.evidence_for)
            cited_evidence_ids.update(
                reference.evidence_id for reference in candidate.evidence_against
            )
            for dissent in candidate.dissenting_evidence:
                cited_evidence_ids.update(reference.evidence_id for reference in dissent.evidence)
                referenced_artifact_ids.update(dissent.artifact_ids)
            for driver in candidate.uncertainty_drivers:
                cited_evidence_ids.update(reference.evidence_id for reference in driver.evidence)
                referenced_artifact_ids.update(driver.artifact_ids)
            for trigger in candidate.change_triggers:
                cited_evidence_ids.update(reference.evidence_id for reference in trigger.evidence)
                referenced_artifact_ids.update(trigger.artifact_ids)
            referenced_artifact_ids.update(candidate.signal_artifact_ids)
            for signal_artifact in candidate.signal_artifacts:
                referenced_artifact_ids.add(signal_artifact.artifact_id)
                cited_evidence_ids.update(signal_artifact.source_evidence_ids)
            missing_prior_ids = set(candidate.prior_outcome_review_ids).difference(
                prior_outcome_review_ids
            )
            if missing_prior_ids:
                raise ValueError("candidate prior_outcome_review_ids must reference report reviews")

        for source_reference in self.source_references:
            cited_evidence_ids.update(source_reference.evidence_ids)
            referenced_artifact_ids.update(source_reference.artifact_ids)
            if set(source_reference.candidate_ids).difference(candidate_ids):
                raise ValueError("source reference candidate_ids must reference candidates")
            if set(source_reference.prior_outcome_review_ids).difference(prior_outcome_review_ids):
                raise ValueError(
                    "source reference prior_outcome_review_ids must reference report reviews"
                )

        for review in self.prior_outcome_reviews:
            cited_evidence_ids.update(
                reference.evidence_id for reference in review.outcome_evidence
            )
            referenced_artifact_ids.update(review.artifact_ids)
            if review.instrument_id and review.instrument_id not in valid_instrument_ids:
                raise ValueError("prior outcome instrument_id must reference report instruments")

        if self.insufficient_evidence is not None:
            cited_evidence_ids.update(
                reference.evidence_id for reference in self.insufficient_evidence.evidence
            )
            referenced_artifact_ids.update(self.insufficient_evidence.artifact_ids)

        for trace in self.material_claim_traces:
            cited_evidence_ids.update(reference.evidence_id for reference in trace.evidence)
            referenced_artifact_ids.update(trace.artifact_ids)
            if set(trace.source_reference_ids).difference(source_reference_ids):
                raise ValueError(
                    "material claim trace source_reference_ids must reference source_references"
                )
            if set(trace.candidate_ids).difference(candidate_ids):
                raise ValueError("material claim trace candidate_ids must reference candidates")
            if set(trace.prior_outcome_review_ids).difference(prior_outcome_review_ids):
                raise ValueError(
                    "material claim trace prior_outcome_review_ids must reference report reviews"
                )
        traced_candidate_ids = {
            candidate_id
            for trace in self.material_claim_traces
            for candidate_id in trace.candidate_ids
        }
        if set(candidate_ids).difference(traced_candidate_ids):
            raise ValueError("every prediction candidate must have a material claim trace")

        missing_evidence_ids = cited_evidence_ids.difference(source_evidence_ids)
        if missing_evidence_ids:
            missing_related_evidence_ids = related_instrument_evidence_ids.difference(
                source_evidence_ids
            )
            if missing_related_evidence_ids:
                raise ValueError(
                    "report evidence_sources must include related instrument evidence_ids"
                )
            raise ValueError("report evidence_sources must include every cited evidence_id")
        if referenced_artifact_ids.difference(artifact_ids):
            raise ValueError("report audit artifacts must include every cited artifact_id")
        _validate_evidence_references_against_sources(self, evidence_by_id)
        return self


def _analysis_evidence_ids(
    component: TechnicalAnalysis | FundamentalAnalysis | SectorContext | MacroContext,
) -> tuple[str, ...]:
    ids = [reference.evidence_id for reference in component.evidence]
    if isinstance(component, FundamentalAnalysis) and component.agent_signal is not None:
        ids.extend(component.agent_signal.source_evidence_ids)
    return tuple(ids)


def _validate_evidence_references_against_sources(
    report: DailyReport,
    evidence_by_id: dict[str, SourceEvidence],
) -> None:
    for reference in _iter_report_evidence_references(report):
        evidence = evidence_by_id.get(reference.evidence_id)
        if evidence is None:
            continue
        if reference.quote is not None and reference.quote not in evidence.text:
            raise ValueError("evidence reference quote must appear in source evidence text")
        if reference.start_char is not None and reference.end_char is not None:
            if reference.end_char > len(evidence.text):
                raise ValueError("evidence reference span must stay within source evidence text")
            if evidence.text[reference.start_char : reference.end_char] != reference.quote:
                raise ValueError("evidence reference span must match source evidence text")


def _audit_artifact_ids(manifest: AuditManifest | DataReference | None) -> set[str]:
    if not isinstance(manifest, AuditManifest):
        return set()
    return {artifact.artifact_id for artifact in manifest.artifacts}


def _iter_report_evidence_references(report: DailyReport) -> tuple[EvidenceReference, ...]:
    references: list[EvidenceReference] = []
    for section in report.instrument_sections:
        references.extend(section.evidence)
        for cluster in section.strategy_clusters:
            references.extend(cluster.evidence)
        for component in (
            section.technical_analysis,
            section.fundamental_analysis,
            section.sector_context,
            section.macro_context,
        ):
            if component is not None:
                references.extend(component.evidence)
    for candidate in report.prediction_candidates:
        references.extend(candidate.evidence_for)
        references.extend(candidate.evidence_against)
        for dissent in candidate.dissenting_evidence:
            references.extend(dissent.evidence)
        for driver in candidate.uncertainty_drivers:
            references.extend(driver.evidence)
        for trigger in candidate.change_triggers:
            references.extend(trigger.evidence)
    if report.insufficient_evidence is not None:
        references.extend(report.insufficient_evidence.evidence)
    for review in report.prior_outcome_reviews:
        references.extend(review.outcome_evidence)
    for trace in report.material_claim_traces:
        references.extend(trace.evidence)
    return tuple(references)


_TRADING_INSTRUCTION_PATTERNS = (
    r"\b(buy|sell)\s+(?!or\b|instruction\b|guidance\b|language\b)"
    r"(?-i:[A-Z][A-Z0-9./-]{0,12})\b",
    r"\b(buy|sell|short)\s+the\s+(stock|shares?|coin|token|etf|contract|instrument)\b",
    r"\b(should|must|need to|time to)\s+(buy|sell|short|go long|go short)\b",
    r"\b(you|we|investors?|traders?)\s+"
    r"(should|must|need to|ought to)\s+(buy|sell|short|go long|go short|enter|exit)\b",
    r"\b(recommend|recommendation|advice)\s+(to\s+)?(buy|sell|short|go long|go short)\b",
    r"\brecommendation\s*:\s*(buy|sell|short|hold)\b",
    r"\b(go|stay)\s+(long|short)\b",
    r"\b(enter|exit|open|close)\s+(a\s+)?(long|short\s+)?position\b",
    r"\b(set|use)\s+(a\s+)?stop[-\s]?loss\b",
    r"\bposition\s+sizing?\b",
    r"\b(take\s+profits?|profit\s+target)\b",
)


def _validate_report_authored_language(*values: str | None) -> None:
    for value in values:
        if value is None:
            continue
        normalized = " ".join(value.split())
        for pattern in _TRADING_INSTRUCTION_PATTERNS:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                raise ValueError(
                    "report-authored fields must not contain imperative trading language "
                    "or trading instructions"
                )


def _validate_candidate_evaluation_metadata(candidate: PredictionCandidate) -> None:
    metadata = candidate.metadata.get("prediction_evaluation")
    if not isinstance(metadata, dict):
        return
    status = metadata.get("status")
    if isinstance(status, str) and status != candidate.status.value:
        raise ValueError("prediction evaluation status must match rendered candidate status")
    candidate_id = metadata.get("candidate_id")
    if isinstance(candidate_id, str) and candidate_id != candidate.candidate_id:
        raise ValueError("prediction evaluation candidate_id must match candidate_id")
    instrument_id = metadata.get("instrument_id")
    if isinstance(instrument_id, str) and instrument_id != candidate.instrument_id:
        raise ValueError("prediction evaluation instrument_id must match candidate instrument_id")
    symbol = metadata.get("symbol")
    if isinstance(symbol, str) and symbol.upper() != candidate.symbol.upper():
        raise ValueError("prediction evaluation symbol must match candidate symbol")


__all__ = [
    "DEFAULT_MARKDOWN_REPORT_OUTLINE",
    "AuditArtifact",
    "AuditManifest",
    "DailyReport",
    "DataFreshnessSummary",
    "DissentingEvidence",
    "InstrumentReportSection",
    "InsufficientEvidenceReport",
    "MarkdownReportOutline",
    "MaterialClaimTrace",
    "PredictionCandidate",
    "PredictionChangeTrigger",
    "PriorOutcomeReview",
    "ReportSourceReference",
    "SignalArtifactReference",
    "UncertaintyDriver",
]
