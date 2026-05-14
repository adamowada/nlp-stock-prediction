"""Report rendering for Phase 2 Codex smoke runs."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal, cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import (
    CredentialState,
    ProviderStatus,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.instruments import Instrument
from nlp_stock_prediction.contracts.provenance import EvidenceReference, ProviderHealth
from nlp_stock_prediction.contracts.report import (
    AuditArtifact,
    AuditManifest,
    DailyReport,
    DataFreshnessSummary,
    InstrumentReportSection,
    InsufficientEvidenceReport,
    MaterialClaimTrace,
    PredictionCandidate,
    PriorOutcomeReview,
    ReportSourceReference,
)
from nlp_stock_prediction.instruments.repository import instrument_from_record
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex, ArtifactType
from nlp_stock_prediction.orchestration.phase2_common import (
    Phase2RunPaths,
    phase2_instrument,
    stable_digest,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase2_evidence import source_evidence_from_record
from nlp_stock_prediction.orchestration.report_candidates import prediction_candidate_from_record
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    ResearchRunRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PriorOutcomeReviewStatus = Literal[
    "available",
    "not_available",
    "pending",
    "stale",
    "unavailable",
]


def render_phase2_prediction_report(
    *,
    store: SQLiteStore,
    repo_root: Path,
    run: ResearchRunRecord,
    paths: Phase2RunPaths,
    run_date: date,
    symbol: str,
    tool_run_id: str | None = None,
    record_tool_run: bool = True,
    tool_name: str = "render_prediction_report",
    tool_version: str = "phase2.v1",
    tool_status: str = "ok",
    tool_warnings: tuple[str, ...] = (),
    produced_by: str = "render_prediction_report",
    artifact_schema_version: str = "phase2-report.v1",
    insufficient_evidence_summary: str | None = None,
) -> JsonObject:
    now = utc_now()
    is_phase4_report = artifact_schema_version.startswith("phase4")
    evidence_records = store.list_evidence_for_run(run.run_id)
    evidence_sources = tuple(source_evidence_from_record(record) for record in evidence_records)
    instrument = _primary_instrument(store, symbol=symbol, fallback_generated_at=now)
    candidates = store.list_prediction_candidates_for_run(run.run_id)
    prediction_candidates = tuple(
        prediction_candidate_from_record(
            candidate,
            evidence_sources,
            prefer_evaluated_references=True,
        )
        for candidate in candidates
    )
    audit_artifacts = _report_input_audit_artifacts(
        tuple(
            _audit_artifact_from_record(record, repo_root)
            for record in store.list_artifacts_for_run(run.run_id)
        )
    )
    stored_prior_outcome_reviews = _stored_prior_outcome_reviews(
        store=store,
        run_id=run.run_id,
        candidate_ids=tuple(candidate.candidate_id for candidate in prediction_candidates),
        evidence_source_ids=tuple(evidence.evidence_id for evidence in evidence_sources),
        audit_artifact_ids=tuple(artifact.artifact_id for artifact in audit_artifacts),
    )
    prior_review_ids_by_candidate = _prior_review_ids_by_candidate(stored_prior_outcome_reviews)
    prediction_candidates = tuple(
        _with_prior_outcome_review_ids(
            candidate,
            prior_review_ids_by_candidate.get(candidate.candidate_id, ()),
        )
        for candidate in prediction_candidates
    )
    prior_outcome_reviews = (
        *stored_prior_outcome_reviews,
        *_missing_prior_outcome_reviews(
            prediction_candidates,
            existing_review_ids=tuple(review.review_id for review in stored_prior_outcome_reviews),
        ),
    )
    section_refs = tuple(
        EvidenceReference(
            evidence_id=record.evidence_id,
            quote=record.text[:160],
            relevance=0.75,
        )
        for record in evidence_sources[:5]
    )
    if is_phase4_report:
        observed_discussion_summary = (
            "Phase 4 fixture-backed research tools imported evidence and context."
        )
        analysis_summary = (
            "Phase 4 report rendering consumes stored tool evidence, analysis artifacts, "
            "prediction candidates, and prediction-quality evaluations."
        )
        universe_summary = f"Phase 4 fixture-backed research universe for {symbol.upper()}"
        freshness_summary = (
            "Phase 4 used deterministic fixture-backed tool outputs."
            if evidence_sources
            else "Phase 4 has no imported evidence for this run."
        )
    else:
        observed_discussion_summary = (
            "Codex live search evidence was imported through MCP and combined with "
            "dummy structural tools."
        )
        analysis_summary = (
            "Phase 2 validates orchestration and provenance; dummy tools keep the "
            "prediction conservative."
        )
        universe_summary = f"Phase 2 Codex smoke universe for {symbol.upper()}"
        freshness_summary = (
            "Codex smoke used live web search plus deterministic dummy tools."
            if evidence_sources
            else "Codex smoke has no imported live-search evidence for this run."
        )
    instrument_section = InstrumentReportSection(
        instrument_id=instrument.instrument_id,
        symbol=instrument.symbol,
        display_name=instrument.display_name,
        observed_discussion_summary=observed_discussion_summary,
        analysis_summary=analysis_summary,
        prediction_candidate_ids=tuple(
            candidate.candidate_id for candidate in prediction_candidates
        ),
        evidence=section_refs,
        data_quality={"codex_search_evidence_count": len(evidence_sources)},
    )
    source_references = _report_source_references(
        evidence_sources=evidence_sources,
        prediction_candidates=prediction_candidates,
        audit_artifacts=audit_artifacts,
        prior_outcome_reviews=prior_outcome_reviews,
    )
    material_claim_traces = _material_claim_traces(
        prediction_candidates=prediction_candidates,
        source_references=source_references,
    )
    provider_name = "phase4-fixture-tools" if is_phase4_report else "codex-web-search"
    insufficient_evidence = (
        None
        if prediction_candidates
        else InsufficientEvidenceReport(
            summary=insufficient_evidence_summary or "No candidate could be synthesized.",
            blocking_reasons=(
                insufficient_evidence_summary or "No candidate could be synthesized.",
            ),
            provider_names=(provider_name,),
        )
    )
    has_codex_search_evidence = bool(evidence_sources)
    stale_provider_names = _stale_provider_names(evidence_sources)
    provider_status = (
        ProviderStatus.STALE
        if stale_provider_names and has_codex_search_evidence
        else ProviderStatus.OK
        if has_codex_search_evidence
        else ProviderStatus.EMPTY
    )
    report = DailyReport(
        schema_version="daily-report.v2",
        run_id=run.run_id,
        report_date=run_date,
        generated_at=now,
        timezone="UTC",
        objective=run.objective,
        universe=universe_summary,
        command_args={"run_id": run.run_id, "symbol": symbol.upper()},
        instruments=(instrument,),
        data_freshness=DataFreshnessSummary(
            as_of=now,
            summary=freshness_summary,
            stale_provider_names=stale_provider_names,
            missing_provider_names=() if has_codex_search_evidence else (provider_name,),
        ),
        provider_health=(
            ProviderHealth(
                provider_name=provider_name,
                status=provider_status,
                checked_at=now,
                credential_state=CredentialState.NOT_REQUIRED,
            ),
        ),
        evidence_sources=evidence_sources,
        instrument_sections=(instrument_section,),
        prediction_candidates=prediction_candidates,
        insufficient_evidence=insufficient_evidence,
        insufficient_evidence_summary=None
        if prediction_candidates
        else insufficient_evidence_summary or "No candidate could be synthesized.",
        source_references=source_references,
        material_claim_traces=material_claim_traces,
        prior_outcome_reviews=prior_outcome_reviews,
        audit_manifest=AuditManifest(
            run_id=run.run_id,
            schema_version="audit-manifest.v2",
            created_at=now,
            artifacts=audit_artifacts,
            command_args={"run_id": run.run_id, "symbol": symbol.upper()},
            prediction_trace_ids=tuple(
                candidate.candidate_id for candidate in prediction_candidates
            ),
        ),
    )
    manifest = report.audit_manifest
    if not isinstance(manifest, AuditManifest):
        raise TypeError("Codex smoke reports must include an audit manifest")
    report_tool_run_id = tool_run_id or f"tool-render-report-{run.run_id}"
    if record_tool_run:
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=report_tool_run_id,
                run_id=run.run_id,
                tool_name=tool_name,
                tool_version=tool_version,
                status=tool_status,
                started_at=now,
                completed_at=now,
                inputs={"symbol": symbol},
                warnings=tool_warnings,
            )
        )
    report_index = ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=paths.run_dir,
        created_at=now,
        produced_by=produced_by,
        tool_run_id=report_tool_run_id,
        schema_version=artifact_schema_version,
    )
    markdown_artifact = report_index.write_text(
        artifact_id=f"artifact-report-md-{stable_digest(run.run_id)}",
        artifact_type="markdown_report",
        filename=paths.report_path.name,
        content=render_markdown_report(report),
        metadata={"run_id": run.run_id},
    )
    json_artifact = report_index.write_text(
        artifact_id=f"artifact-report-json-{stable_digest(run.run_id)}",
        artifact_type="json_report",
        filename=paths.json_path.name,
        content=render_json_report(report),
        metadata={"run_id": run.run_id},
    )
    final_manifest = manifest.model_copy(
        update={"artifacts": (*manifest.artifacts, markdown_artifact, json_artifact)}
    )
    ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=paths.audit_dir,
        created_at=now,
        produced_by=produced_by,
        tool_run_id=report_tool_run_id,
        schema_version=artifact_schema_version,
    ).write_json(
        artifact_id=f"artifact-audit-manifest-{stable_digest(run.run_id)}",
        artifact_type="audit_manifest",
        filename=paths.audit_manifest_path.name,
        payload=cast(JsonObject, final_manifest.model_dump(mode="json")),
        metadata={"run_id": run.run_id},
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=run.run_id,
            run_kind=run.run_kind,
            objective=run.objective,
            status="completed",
            started_at=run.started_at,
            completed_at=now,
            metadata=run.metadata,
        )
    )
    return {
        "run_id": run.run_id,
        "markdown_path": paths.report_path.as_posix(),
        "json_path": paths.json_path.as_posix(),
        "audit_manifest_path": paths.audit_manifest_path.as_posix(),
    }


def _report_source_references(
    *,
    evidence_sources: tuple[object, ...],
    prediction_candidates: tuple[object, ...],
    audit_artifacts: tuple[AuditArtifact, ...],
    prior_outcome_reviews: tuple[PriorOutcomeReview, ...],
) -> tuple[ReportSourceReference, ...]:
    references: list[ReportSourceReference] = []
    candidate_ids = tuple(
        candidate_id
        for candidate_id in (
            getattr(candidate, "candidate_id", None) for candidate in prediction_candidates
        )
        if isinstance(candidate_id, str) and candidate_id
    )
    for evidence in evidence_sources:
        evidence_id = getattr(evidence, "evidence_id", None)
        if not isinstance(evidence_id, str) or not evidence_id:
            continue
        references.append(
            ReportSourceReference(
                reference_id=f"source-ref-{evidence_id}",
                label=f"Source evidence {evidence_id}",
                reference_type="source_evidence",
                evidence_ids=(evidence_id,),
                candidate_ids=_candidate_ids_for_evidence(
                    evidence_id,
                    prediction_candidates,
                ),
            )
        )
    for artifact in audit_artifacts:
        if artifact.artifact_type not in {
            "prediction_evaluation",
            "technical_package",
            "calibration_summary",
            "signal_family_ablation",
            "walk_forward_evaluation",
        }:
            continue
        reference_type = (
            "prediction_evaluation"
            if artifact.artifact_type == "prediction_evaluation"
            else "tool_artifact"
        )
        references.append(
            ReportSourceReference(
                reference_id=f"source-ref-{artifact.artifact_id}",
                label=f"Artifact {artifact.artifact_id}",
                reference_type=reference_type,
                artifact_ids=(artifact.artifact_id,),
                candidate_ids=_candidate_ids_for_artifact(
                    artifact,
                    prediction_candidates,
                    fallback_candidate_ids=candidate_ids,
                ),
                metadata={"artifact_type": artifact.artifact_type},
            )
        )
    for review in prior_outcome_reviews:
        references.append(
            ReportSourceReference(
                reference_id=f"source-ref-{review.review_id}",
                label=f"Prior outcome review {review.review_id}",
                reference_type="prior_outcome",
                evidence_ids=tuple(reference.evidence_id for reference in review.outcome_evidence),
                artifact_ids=review.artifact_ids,
                candidate_ids=((review.candidate_id,) if review.candidate_id else ()),
                prior_outcome_review_ids=(review.review_id,),
                metadata={"status": review.status},
            )
        )
    return tuple(references)


def _material_claim_traces(
    *,
    prediction_candidates: tuple[object, ...],
    source_references: tuple[ReportSourceReference, ...],
) -> tuple[MaterialClaimTrace, ...]:
    if not prediction_candidates:
        return ()
    traces: list[MaterialClaimTrace] = []
    for candidate in prediction_candidates:
        candidate_id = getattr(candidate, "candidate_id", None)
        thesis = getattr(candidate, "thesis", None)
        evidence_for = getattr(candidate, "evidence_for", ())
        evidence_against = getattr(candidate, "evidence_against", ())
        signal_artifact_ids = getattr(candidate, "signal_artifact_ids", ())
        if not isinstance(candidate_id, str) or not isinstance(thesis, str):
            continue
        prior_outcome_review_ids = _candidate_prior_outcome_review_ids(candidate)
        source_reference_ids = _source_reference_ids_for_candidate(
            candidate,
            source_references,
        )
        has_trace_references = bool(
            evidence_for
            or evidence_against
            or signal_artifact_ids
            or source_reference_ids
            or prior_outcome_review_ids
        )
        traces.append(
            MaterialClaimTrace(
                claim_id=f"claim-{candidate_id}",
                claim=thesis,
                claim_type="analysis" if has_trace_references else "labeled_inference",
                evidence=tuple(evidence_for) + tuple(evidence_against),
                artifact_ids=tuple(signal_artifact_ids),
                source_reference_ids=source_reference_ids,
                candidate_ids=(candidate_id,),
                prior_outcome_review_ids=prior_outcome_review_ids,
                rationale=(
                    None
                    if has_trace_references
                    else "Candidate is carried as structured insufficient-evidence context."
                ),
            )
        )
    return tuple(traces)


def _report_input_audit_artifacts(
    artifacts: tuple[AuditArtifact, ...],
) -> tuple[AuditArtifact, ...]:
    return tuple(
        artifact
        for artifact in artifacts
        if artifact.artifact_type not in {"markdown_report", "json_report", "audit_manifest"}
    )


def _candidate_ids_for_evidence(
    evidence_id: str,
    prediction_candidates: tuple[object, ...],
) -> tuple[str, ...]:
    candidate_ids: list[str] = []
    for candidate in prediction_candidates:
        candidate_id = getattr(candidate, "candidate_id", None)
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        if evidence_id in _candidate_evidence_ids(candidate):
            candidate_ids.append(candidate_id)
    return tuple(dict.fromkeys(candidate_ids))


def _candidate_ids_for_artifact(
    artifact: AuditArtifact,
    prediction_candidates: tuple[object, ...],
    *,
    fallback_candidate_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if artifact.artifact_type in {
        "calibration_summary",
        "signal_family_ablation",
        "walk_forward_evaluation",
    }:
        return fallback_candidate_ids
    candidate_ids: list[str] = []
    for candidate in prediction_candidates:
        candidate_id = getattr(candidate, "candidate_id", None)
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        if artifact.artifact_id in _candidate_artifact_ids(candidate):
            candidate_ids.append(candidate_id)
    return tuple(dict.fromkeys(candidate_ids))


def _source_reference_ids_for_candidate(
    candidate: object,
    source_references: tuple[ReportSourceReference, ...],
) -> tuple[str, ...]:
    candidate_id = getattr(candidate, "candidate_id", None)
    if not isinstance(candidate_id, str) or not candidate_id:
        return ()
    evidence_ids = set(_candidate_evidence_ids(candidate))
    artifact_ids = set(_candidate_artifact_ids(candidate))
    prior_review_ids = set(_candidate_prior_outcome_review_ids(candidate))
    reference_ids: list[str] = []
    for reference in source_references:
        if candidate_id in reference.candidate_ids:
            reference_ids.append(reference.reference_id)
            continue
        if evidence_ids.intersection(reference.evidence_ids):
            reference_ids.append(reference.reference_id)
            continue
        if artifact_ids.intersection(reference.artifact_ids):
            reference_ids.append(reference.reference_id)
            continue
        if prior_review_ids.intersection(reference.prior_outcome_review_ids):
            reference_ids.append(reference.reference_id)
    return tuple(dict.fromkeys(reference_ids))


def _candidate_evidence_ids(candidate: object) -> tuple[str, ...]:
    ids: list[str] = []
    for field_name in ("evidence_for", "evidence_against"):
        for reference in getattr(candidate, field_name, ()):
            evidence_id = getattr(reference, "evidence_id", None)
            if isinstance(evidence_id, str) and evidence_id:
                ids.append(evidence_id)
    return tuple(dict.fromkeys(ids))


def _candidate_artifact_ids(candidate: object) -> tuple[str, ...]:
    ids: list[str] = []
    for artifact_id in getattr(candidate, "signal_artifact_ids", ()):
        if isinstance(artifact_id, str) and artifact_id:
            ids.append(artifact_id)
    for reference in getattr(candidate, "signal_artifacts", ()):
        artifact_id = getattr(reference, "artifact_id", None)
        if isinstance(artifact_id, str) and artifact_id:
            ids.append(artifact_id)
    return tuple(dict.fromkeys(ids))


def _candidate_prior_outcome_review_ids(candidate: object) -> tuple[str, ...]:
    return tuple(
        review_id
        for review_id in getattr(candidate, "prior_outcome_review_ids", ())
        if isinstance(review_id, str) and review_id
    )


def _stored_prior_outcome_reviews(
    *,
    store: SQLiteStore,
    run_id: str,
    candidate_ids: tuple[str, ...],
    evidence_source_ids: tuple[str, ...],
    audit_artifact_ids: tuple[str, ...],
) -> tuple[PriorOutcomeReview, ...]:
    if not candidate_ids:
        return ()
    candidate_id_set = set(candidate_ids)
    evidence_source_id_set = set(evidence_source_ids)
    audit_artifact_id_set = set(audit_artifact_ids)
    reviews: list[PriorOutcomeReview] = []
    for record in store.list_outcome_evaluations_for_run(run_id):
        if record.candidate_id not in candidate_id_set:
            continue
        outcome = store.get_prediction_outcome(record.outcome_id)
        artifact_ids = _stored_prior_review_artifact_ids(store, record.outcome_evaluation_id)
        if record.artifact_id is not None:
            artifact_ids = (record.artifact_id, *artifact_ids)
        artifact_ids = tuple(
            artifact_id
            for artifact_id in dict.fromkeys(artifact_ids)
            if artifact_id in audit_artifact_id_set
        )
        evidence = tuple(
            EvidenceReference(evidence_id=link.evidence_id)
            for link in store.list_outcome_evaluation_evidence_links(record.outcome_evaluation_id)
            if link.evidence_id in evidence_source_id_set
        )
        review_status = _prior_review_status(record.status)
        limitations = record.limitations
        if review_status != "available" and not limitations:
            limitations = ("Stored outcome evaluation is not resolved as available.",)
        if review_status == "available" and not (evidence or artifact_ids):
            review_status = "not_available"
            limitations = (
                "Stored outcome evaluation is missing report-visible evidence and artifacts.",
            )
        reviews.append(
            PriorOutcomeReview(
                review_id=record.outcome_evaluation_id,
                status=review_status,
                summary=_prior_review_summary(record.status, record.quality_score),
                candidate_id=record.candidate_id,
                instrument_id=record.instrument_id,
                reviewed_at=record.evaluated_at,
                horizon=_review_horizon(outcome.horizon if outcome is not None else None),
                outcome_evidence=evidence,
                artifact_ids=artifact_ids,
                limitations=limitations,
                metadata={
                    "outcome_id": record.outcome_id,
                    "quality_score": record.quality_score,
                    "baseline_comparison": dict(record.baseline_comparison),
                    "source": "prediction_outcome_evaluations",
                },
            )
        )
    return tuple(reviews)


def _stored_prior_review_artifact_ids(
    store: SQLiteStore,
    outcome_evaluation_id: str,
) -> tuple[str, ...]:
    return tuple(
        link.artifact_id
        for link in store.list_outcome_evaluation_artifact_links(outcome_evaluation_id)
    )


def _prior_review_status(
    status: str,
) -> PriorOutcomeReviewStatus:
    if status in {"confirmed", "missed", "mixed", "inconclusive"}:
        return "available"
    if status == "pending":
        return "pending"
    if status == "stale":
        return "stale"
    if status in {"unavailable", "not_evaluable"}:
        return "unavailable"
    return "not_available"


def _prior_review_summary(status: str, quality_score: float | None) -> str:
    if quality_score is None:
        return f"Stored outcome evaluation is {status} for this prediction candidate."
    return (
        f"Stored outcome evaluation is {status} for this prediction candidate with "
        f"quality score {quality_score:.2f}."
    )


def _review_horizon(value: str | None) -> TimeHorizon:
    if value is None:
        return TimeHorizon.UNKNOWN
    try:
        return TimeHorizon(value)
    except ValueError:
        return TimeHorizon.UNKNOWN


def _prior_review_ids_by_candidate(
    reviews: tuple[PriorOutcomeReview, ...],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for review in reviews:
        if review.candidate_id is None:
            continue
        grouped.setdefault(review.candidate_id, []).append(review.review_id)
    return {
        candidate_id: tuple(dict.fromkeys(review_ids))
        for candidate_id, review_ids in grouped.items()
    }


def _with_prior_outcome_review_ids(
    candidate: PredictionCandidate,
    review_ids: tuple[str, ...],
) -> PredictionCandidate:
    existing = getattr(candidate, "prior_outcome_review_ids", ())
    if not isinstance(existing, tuple):
        existing = ()
    merged = tuple(dict.fromkeys((*existing, *review_ids)))
    if merged == existing:
        return candidate
    return candidate.model_copy(update={"prior_outcome_review_ids": merged})


def _missing_prior_outcome_reviews(
    prediction_candidates: tuple[object, ...],
    *,
    existing_review_ids: tuple[str, ...],
) -> tuple[PriorOutcomeReview, ...]:
    existing_review_id_set = set(existing_review_ids)
    reviews: list[PriorOutcomeReview] = []
    for candidate in prediction_candidates:
        candidate_id = getattr(candidate, "candidate_id", None)
        prior_ids = getattr(candidate, "prior_outcome_review_ids", ())
        if not isinstance(candidate_id, str):
            continue
        for review_id in prior_ids:
            if not isinstance(review_id, str) or not review_id:
                continue
            if review_id in existing_review_id_set:
                continue
            reviews.append(
                PriorOutcomeReview(
                    review_id=review_id,
                    status="not_available",
                    summary="No prior outcome review is available for this run.",
                    candidate_id=candidate_id,
                    limitations=("No stored prior outcome artifact was linked to this candidate.",),
                )
            )
    return tuple(reviews)


def _primary_instrument(
    store: SQLiteStore,
    *,
    symbol: str,
    fallback_generated_at: datetime,
) -> Instrument:
    instrument_id = f"instrument:codex:{symbol.upper()}"
    record = store.get_instrument(instrument_id)
    if record is not None:
        return instrument_from_record(record)
    discovered = store.find_instruments_by_symbol_or_alias(symbol.upper())
    if discovered:
        return instrument_from_record(discovered[0])
    return phase2_instrument(symbol.upper(), fallback_generated_at)


def _audit_artifact_from_record(record: ArtifactRecord, repo_root: Path) -> AuditArtifact:
    path = record.path if record.path.is_absolute() else repo_root / record.path
    artifact_type = record.artifact_type
    if artifact_type not in {
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
    }:
        raise ValueError(f"unknown artifact type: {artifact_type}")
    return AuditArtifact(
        artifact_id=record.artifact_id,
        artifact_type=cast(ArtifactType, artifact_type),
        path=path.as_posix(),
        created_at=record.created_at or utc_now(),
        produced_by=record.produced_by or "phase2-mcp",
        sha256=record.sha256,
        record_count=record.record_count,
        metadata=record.metadata,
    )


def _stale_provider_names(evidence_sources: tuple[object, ...]) -> tuple[str, ...]:
    provider_names: list[str] = []
    for evidence in evidence_sources:
        provenance = getattr(evidence, "provenance", None)
        freshness_status = getattr(provenance, "freshness_status", None)
        if getattr(freshness_status, "value", freshness_status) != "stale":
            continue
        provider_name = getattr(provenance, "provider_name", None)
        if isinstance(provider_name, str) and provider_name:
            provider_names.append(provider_name)
    return tuple(dict.fromkeys(provider_names))


__all__ = ["render_phase2_prediction_report"]
