"""Prior report review and prediction change-tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts import (
    DailyReport,
    EvidenceReference,
    Instrument,
    MaterialClaimTrace,
    PredictionCandidate,
    PredictionChangeTrigger,
    PredictionStatus,
    PriorOutcomeReview,
    ReportSourceReference,
    SourceEvidence,
)
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
)
from nlp_stock_prediction.contracts.evaluation import (
    PredictionOutcome,
    PredictionOutcomeEvaluation,
    PredictionOutcomeEvaluationArtifactPayload,
)
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.orchestration.artifact_policy import ALLOWED_ARTIFACT_TYPES, ArtifactType
from nlp_stock_prediction.orchestration.codex_smoke_evidence import source_evidence_from_record
from nlp_stock_prediction.orchestration.orchestration_common import file_sha256, stable_digest
from nlp_stock_prediction.orchestration.report_data_modes import ReportDataMode
from nlp_stock_prediction.reporting.json import load_json_report
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    ReportArtifactRecord,
    ResearchRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PRIOR_REPORT_STALE_AFTER_DAYS = 30


@dataclass(frozen=True)
class PriorOutcomeContext:
    """Prior outcome review additions for one rendered report."""

    prediction_candidates: tuple[PredictionCandidate, ...]
    prior_outcome_reviews: tuple[PriorOutcomeReview, ...]
    audit_artifacts: tuple[AuditArtifact, ...]
    source_references: tuple[ReportSourceReference, ...]
    material_claim_traces: tuple[MaterialClaimTrace, ...]
    evidence_sources: tuple[SourceEvidence, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _LoadedPriorReport:
    artifact: ReportArtifactRecord
    report: DailyReport | None
    audit_artifact: AuditArtifact | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _PriorReviewBuildResult:
    review: PriorOutcomeReview
    audit_artifacts: tuple[AuditArtifact, ...] = ()
    evidence_sources: tuple[SourceEvidence, ...] = ()


def apply_prior_outcome_context(
    *,
    store: SQLiteStore,
    repo_root: Path,
    run: ResearchRunRecord,
    report_date: date,
    reviewed_at: datetime,
    instrument: Instrument,
    prediction_candidates: tuple[PredictionCandidate, ...],
    missing_provider_names: tuple[str, ...] = (),
    stale_provider_names: tuple[str, ...] = (),
    report_data_mode: ReportDataMode | None = None,
) -> PriorOutcomeContext:
    """Attach prior-report review IDs and concrete change triggers to candidates."""

    if not prediction_candidates:
        return PriorOutcomeContext(
            prediction_candidates=(),
            prior_outcome_reviews=(),
            audit_artifacts=(),
            source_references=(),
            material_claim_traces=(),
        )

    prior = _load_prior_report(
        store=store,
        repo_root=repo_root,
        run=run,
        report_date=report_date,
        instrument=instrument,
        report_data_mode=report_data_mode,
    )
    updated_candidates: list[PredictionCandidate] = []
    reviews: list[PriorOutcomeReview] = []
    source_references: list[ReportSourceReference] = []
    traces: list[MaterialClaimTrace] = []
    audit_artifacts: list[AuditArtifact] = []
    evidence_sources: list[SourceEvidence] = []

    for candidate in prediction_candidates:
        review_result = _prior_outcome_review(
            store=store,
            repo_root=repo_root,
            candidate=candidate,
            prior=prior,
            report_date=report_date,
            reviewed_at=reviewed_at,
        )
        review = review_result.review
        review = _review_with_outcome_contracts(
            review=review,
            candidate=candidate,
            prior=prior,
            report_date=report_date,
            reviewed_at=reviewed_at,
        )
        audit_artifacts.extend(review_result.audit_artifacts)
        evidence_sources.extend(review_result.evidence_sources)
        reviews.append(review)
        change_triggers = _change_triggers(
            candidate=candidate,
            prior=prior,
            review=review,
            missing_provider_names=missing_provider_names,
            stale_provider_names=stale_provider_names,
        )
        updated_candidates.append(
            candidate.model_copy(
                update={
                    "prior_outcome_review_ids": tuple(
                        dict.fromkeys((*candidate.prior_outcome_review_ids, review.review_id))
                    ),
                    "change_triggers": _dedupe_change_triggers(
                        (*candidate.change_triggers, *change_triggers)
                    ),
                    "metadata": {
                        **candidate.metadata,
                        "prior_outcome_review": _prior_metadata(prior, review),
                    },
                }
            )
        )
        source_references.append(_prior_source_reference(review))
        traces.append(_prior_material_claim_trace(review))

    if prior and prior.audit_artifact is not None:
        audit_artifacts.append(prior.audit_artifact)
    warnings = prior.warnings if prior else ()
    return PriorOutcomeContext(
        prediction_candidates=tuple(updated_candidates),
        prior_outcome_reviews=tuple(reviews),
        audit_artifacts=_dedupe_audit_artifacts(audit_artifacts),
        source_references=tuple(source_references),
        material_claim_traces=tuple(traces),
        evidence_sources=_dedupe_source_evidence(evidence_sources),
        warnings=warnings,
    )


def _load_prior_report(
    *,
    store: SQLiteStore,
    repo_root: Path,
    run: ResearchRunRecord,
    report_date: date,
    instrument: Instrument,
    report_data_mode: ReportDataMode | None,
) -> _LoadedPriorReport | None:
    explicit_artifact_id = _explicit_prior_report_artifact_id(run.metadata)
    artifact = (
        store.get_report_artifact(explicit_artifact_id)
        if explicit_artifact_id is not None
        else None
    )
    if artifact is None:
        artifact = store.get_latest_prior_report_artifact(
            before_report_date=report_date,
            artifact_type="json_report",
            instrument_id=instrument.instrument_id,
        )
    if artifact is None:
        artifact = store.get_latest_prior_report_artifact(
            before_report_date=report_date,
            artifact_type="json_report",
            symbol=instrument.symbol,
        )
    if artifact is None:
        return None
    if report_data_mode is not None and artifact.report_data_mode != report_data_mode:
        return _LoadedPriorReport(
            artifact=artifact,
            report=None,
            audit_artifact=None,
            warnings=(
                f"Prior report artifact {artifact.artifact_id} uses report_data_mode "
                f"{artifact.report_data_mode!r}, which is incompatible with current "
                f"report_data_mode {report_data_mode!r}.",
            ),
        )

    artifact_path = artifact.path if artifact.path.is_absolute() else repo_root / artifact.path
    audit_artifact = AuditArtifact(
        artifact_id=artifact.artifact_id,
        artifact_type="json_report",
        path=artifact_path.as_posix(),
        created_at=artifact.created_at
        or artifact.source_run_completed_at
        or artifact.source_run_started_at,
        produced_by="prior_report_index",
        sha256=artifact.sha256,
        metadata={
            "prior_outcome_source": True,
            "source_run_id": artifact.run_id,
            "report_date": artifact.report_date.isoformat(),
            "report_data_mode": artifact.report_data_mode,
        },
    )
    warnings: list[str] = []
    if not artifact_path.exists():
        return _LoadedPriorReport(
            artifact=artifact,
            report=None,
            audit_artifact=audit_artifact,
            warnings=(f"Prior report artifact file is missing: {artifact.artifact_id}.",),
        )
    actual_sha256 = file_sha256(artifact_path)
    if actual_sha256 != artifact.sha256:
        warnings.append(
            f"Prior report artifact sha256 mismatch for {artifact.artifact_id}: "
            f"expected {artifact.sha256}, observed {actual_sha256}."
        )
        return _LoadedPriorReport(
            artifact=artifact,
            report=None,
            audit_artifact=audit_artifact,
            warnings=tuple(warnings),
        )
    try:
        report = load_json_report(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"Prior report artifact is malformed: {artifact.artifact_id}: {exc}.")
        report = None
    return _LoadedPriorReport(
        artifact=artifact,
        report=report,
        audit_artifact=audit_artifact,
        warnings=tuple(warnings),
    )


def _prior_outcome_review(
    *,
    store: SQLiteStore,
    repo_root: Path,
    candidate: PredictionCandidate,
    prior: _LoadedPriorReport | None,
    report_date: date,
    reviewed_at: datetime,
) -> _PriorReviewBuildResult:
    review_id = f"prior-outcome-{stable_digest(candidate.candidate_id)}"
    if prior is None:
        return _PriorReviewBuildResult(
            PriorOutcomeReview(
                review_id=review_id,
                status="not_available",
                summary="No prior stored report artifact is available for this instrument.",
                candidate_id=candidate.candidate_id,
                instrument_id=candidate.instrument_id,
                reviewed_at=reviewed_at,
                horizon=candidate.horizon,
                limitations=("This appears to be the first indexed report for the instrument.",),
                metadata={"prior_report_available": False},
            )
        )
    artifact_ids = (prior.artifact.artifact_id,) if prior.audit_artifact is not None else ()
    if prior.report is None:
        return _PriorReviewBuildResult(
            PriorOutcomeReview(
                review_id=review_id,
                status="unavailable",
                summary="A prior stored report artifact exists but could not be reviewed.",
                candidate_id=candidate.candidate_id,
                instrument_id=candidate.instrument_id,
                original_report_date=prior.artifact.report_date,
                reviewed_at=reviewed_at,
                horizon=candidate.horizon,
                artifact_ids=artifact_ids,
                limitations=prior.warnings
                or ("The prior report artifact could not be loaded as a valid JSON report.",),
                metadata=_review_metadata(prior=prior, prior_candidate=None),
            )
        )
    prior_candidate = _matching_prior_candidate(prior.report, candidate)
    if prior_candidate is None:
        return _PriorReviewBuildResult(
            PriorOutcomeReview(
                review_id=review_id,
                status="not_available",
                summary="The prior report did not include a matching prediction scenario.",
                candidate_id=candidate.candidate_id,
                instrument_id=candidate.instrument_id,
                original_report_date=prior.report.report_date,
                reviewed_at=reviewed_at,
                horizon=candidate.horizon,
                artifact_ids=artifact_ids,
                limitations=(
                    "No prior candidate matched the current instrument, horizon, and "
                    "prediction type.",
                ),
                metadata=_review_metadata(prior=prior, prior_candidate=None),
            )
        )
    stale_reasons = _prior_stale_reasons(prior.report, current_report_date=report_date)
    if stale_reasons:
        return _PriorReviewBuildResult(
            PriorOutcomeReview(
                review_id=review_id,
                status="stale",
                summary="A prior matching prediction exists, but its report context is stale.",
                candidate_id=candidate.candidate_id,
                instrument_id=candidate.instrument_id,
                original_report_date=prior.report.report_date,
                reviewed_at=reviewed_at,
                horizon=candidate.horizon,
                artifact_ids=artifact_ids,
                limitations=stale_reasons,
                metadata=_review_metadata(prior=prior, prior_candidate=prior_candidate),
            )
        )
    persisted = _persisted_outcome_review_for_prior_candidate(
        store=store,
        repo_root=repo_root,
        candidate=candidate,
        prior=prior,
        prior_candidate=prior_candidate,
        reviewed_at=reviewed_at,
    )
    if persisted is not None:
        return persisted
    outcome_evidence = tuple(dict.fromkeys((*candidate.evidence_for, *candidate.evidence_against)))
    if not outcome_evidence:
        return _PriorReviewBuildResult(
            PriorOutcomeReview(
                review_id=review_id,
                status="pending",
                summary=(
                    "A prior matching prediction exists, but no follow-up evidence is available."
                ),
                candidate_id=candidate.candidate_id,
                instrument_id=candidate.instrument_id,
                original_report_date=prior.report.report_date,
                reviewed_at=reviewed_at,
                horizon=candidate.horizon,
                artifact_ids=artifact_ids,
                limitations=("Current report evidence did not cite follow-up source evidence.",),
                metadata=_review_metadata(prior=prior, prior_candidate=prior_candidate),
            )
        )
    return _PriorReviewBuildResult(
        PriorOutcomeReview(
            review_id=review_id,
            status="available",
            summary=_available_review_summary(candidate),
            candidate_id=candidate.candidate_id,
            instrument_id=candidate.instrument_id,
            original_report_date=prior.report.report_date,
            reviewed_at=reviewed_at,
            horizon=candidate.horizon,
            outcome_evidence=outcome_evidence,
            artifact_ids=artifact_ids,
            metadata={
                **_review_metadata(prior=prior, prior_candidate=prior_candidate),
                "current_status": candidate.status.value,
                "current_direction": candidate.direction.value,
            },
        )
    )


def _persisted_outcome_review_for_prior_candidate(
    *,
    store: SQLiteStore,
    repo_root: Path,
    candidate: PredictionCandidate,
    prior: _LoadedPriorReport,
    prior_candidate: PredictionCandidate,
    reviewed_at: datetime,
) -> _PriorReviewBuildResult | None:
    records = store.list_outcome_evaluations_for_candidate(prior_candidate.candidate_id)
    if not records:
        return None
    record = sorted(records, key=lambda item: item.evaluated_at, reverse=True)[0]
    review_id = record.outcome_evaluation_id
    artifact, payload, load_limitations = _load_outcome_evaluation_payload(
        store=store,
        repo_root=repo_root,
        outcome_evaluation_id=record.outcome_evaluation_id,
        artifact_id=record.artifact_id,
    )
    audit_artifacts, artifact_limitations = _audit_artifacts_for_outcome_evaluation(
        store=store,
        repo_root=repo_root,
        outcome_evaluation_id=record.outcome_evaluation_id,
        primary_artifact=artifact,
    )
    base_metadata = _persisted_review_base_metadata(
        prior=prior,
        prior_candidate=prior_candidate,
        record_outcome_evaluation_id=record.outcome_evaluation_id,
        artifact=artifact,
    )
    if payload is None:
        artifact_ids = tuple(artifact.artifact_id for artifact in audit_artifacts)
        return _PriorReviewBuildResult(
            review=PriorOutcomeReview(
                review_id=review_id,
                status="unavailable",
                summary="A persisted outcome evaluation exists but its artifact was unavailable.",
                candidate_id=candidate.candidate_id,
                instrument_id=candidate.instrument_id,
                original_report_date=prior.report.report_date if prior.report else None,
                reviewed_at=reviewed_at,
                horizon=candidate.horizon,
                artifact_ids=artifact_ids,
                limitations=tuple(dict.fromkeys((*load_limitations, *artifact_limitations))),
                metadata=base_metadata,
            ),
            audit_artifacts=audit_artifacts,
        )

    compatibility_limitations = _outcome_payload_compatibility_limitations(
        payload=payload,
        prior_candidate=prior_candidate,
    )
    if compatibility_limitations:
        return _PriorReviewBuildResult(
            review=PriorOutcomeReview(
                review_id=review_id,
                status="unavailable",
                summary=(
                    "A persisted outcome evaluation exists but is incompatible with the "
                    "matched prior prediction."
                ),
                candidate_id=candidate.candidate_id,
                instrument_id=candidate.instrument_id,
                original_report_date=prior.report.report_date if prior.report else None,
                reviewed_at=reviewed_at,
                horizon=candidate.horizon,
                artifact_ids=tuple(artifact.artifact_id for artifact in audit_artifacts),
                limitations=tuple(
                    dict.fromkeys((*compatibility_limitations, *artifact_limitations))
                ),
                metadata={
                    **base_metadata,
                    "prediction_outcome_evaluation_artifact_id": (
                        artifact.artifact_id if artifact is not None else None
                    ),
                },
            ),
            audit_artifacts=audit_artifacts,
        )

    evidence_refs, evidence_sources, evidence_limitations = _outcome_evidence_sources(
        store=store,
        payload=payload,
    )
    reliability_freshness = _freshness_metadata(payload)
    artifact_ids = tuple(dict.fromkeys(artifact.artifact_id for artifact in audit_artifacts))
    status = _persisted_prior_review_status(payload.outcome_evaluation.status.value)
    limitations = tuple(
        dict.fromkeys(
            (
                *payload.limitations,
                *payload.outcome_evaluation.limitations,
                *evidence_limitations,
                *artifact_limitations,
                *_reliability_freshness_limitations(reliability_freshness),
            )
        )
    )
    if status != "available" and not limitations:
        limitations = ("Persisted outcome evaluation is unresolved.",)
    return _PriorReviewBuildResult(
        review=PriorOutcomeReview(
            review_id=review_id,
            status=status,
            summary=_persisted_outcome_review_summary(payload),
            candidate_id=candidate.candidate_id,
            instrument_id=candidate.instrument_id,
            original_report_date=prior.report.report_date if prior.report else None,
            reviewed_at=payload.outcome_evaluation.evaluated_at,
            horizon=candidate.horizon,
            outcome_evidence=evidence_refs if status == "available" else (),
            artifact_ids=artifact_ids,
            limitations=limitations,
            metadata={
                **base_metadata,
                "prediction_outcome": payload.outcome_evaluation.outcome.model_dump(mode="json"),
                "prediction_outcome_evaluation": payload.outcome_evaluation.model_dump(mode="json"),
                "prediction_evaluation_target": payload.target.model_dump(mode="json"),
                "reliability_freshness": reliability_freshness,
                "source_evidence_ids": list(payload.evidence_ids),
                "source_artifact_ids": list(payload.artifact_ids),
            },
        ),
        audit_artifacts=audit_artifacts,
        evidence_sources=evidence_sources,
    )


def _load_outcome_evaluation_payload(
    *,
    store: SQLiteStore,
    repo_root: Path,
    outcome_evaluation_id: str,
    artifact_id: str | None,
) -> tuple[
    ArtifactRecord | None,
    PredictionOutcomeEvaluationArtifactPayload | None,
    tuple[str, ...],
]:
    if artifact_id is None:
        return (
            None,
            None,
            (f"Persisted outcome evaluation {outcome_evaluation_id} has no artifact_id.",),
        )
    artifact = store.get_artifact(artifact_id)
    if artifact is None:
        return None, None, (f"Persisted outcome artifact is missing: {artifact_id}.",)
    if artifact.artifact_type != "prediction_outcome_evaluation":
        return (
            artifact,
            None,
            (
                f"Persisted outcome artifact {artifact_id} has unsupported type "
                f"{artifact.artifact_type!r}.",
            ),
        )
    path = artifact.path if artifact.path.is_absolute() else repo_root / artifact.path
    if not path.exists():
        return artifact, None, (f"Persisted outcome artifact file is missing: {artifact_id}.",)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != artifact.sha256:
        return (
            artifact,
            None,
            (
                f"Persisted outcome artifact sha256 mismatch for {artifact_id}: expected "
                f"{artifact.sha256}, observed {actual_sha256}.",
            ),
        )
    try:
        payload = PredictionOutcomeEvaluationArtifactPayload.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return artifact, None, (f"Persisted outcome artifact is malformed: {artifact_id}: {exc}.",)
    if payload.outcome_evaluation.outcome_evaluation_id != outcome_evaluation_id:
        return (
            artifact,
            None,
            (
                "Persisted outcome artifact outcome_evaluation_id does not match storage row: "
                f"{payload.outcome_evaluation.outcome_evaluation_id} != {outcome_evaluation_id}.",
            ),
        )
    return artifact, payload, ()


def _audit_artifacts_for_outcome_evaluation(
    *,
    store: SQLiteStore,
    repo_root: Path,
    outcome_evaluation_id: str,
    primary_artifact: ArtifactRecord | None,
) -> tuple[tuple[AuditArtifact, ...], tuple[str, ...]]:
    records: list[ArtifactRecord] = []
    limitations: list[str] = []
    if primary_artifact is not None:
        records.append(primary_artifact)
    for link in store.list_outcome_evaluation_artifact_links(outcome_evaluation_id):
        artifact = store.get_artifact(link.artifact_id)
        if artifact is None:
            limitations.append(
                "Persisted outcome evaluation artifact is missing from storage: "
                f"{link.artifact_id}."
            )
            continue
        records.append(artifact)
    artifacts: list[AuditArtifact] = []
    for record in records:
        audit_artifact = _audit_artifact_from_record(record, repo_root=repo_root)
        if audit_artifact is None:
            limitations.append(
                "Persisted outcome evaluation artifact has unsupported type: "
                f"{record.artifact_id} ({record.artifact_type})."
            )
            continue
        artifacts.append(audit_artifact)
    return _dedupe_audit_artifacts(artifacts), tuple(dict.fromkeys(limitations))


def _audit_artifact_from_record(
    record: ArtifactRecord,
    *,
    repo_root: Path,
) -> AuditArtifact | None:
    if record.artifact_type not in ALLOWED_ARTIFACT_TYPES:
        return None
    path = record.path if record.path.is_absolute() else repo_root / record.path
    return AuditArtifact(
        artifact_id=record.artifact_id,
        artifact_type=cast(ArtifactType, record.artifact_type),
        path=path.as_posix(),
        created_at=record.created_at or datetime.now(UTC),
        produced_by=record.produced_by or "persisted_outcome_review",
        sha256=record.sha256,
        record_count=record.record_count,
        metadata={
            **record.metadata,
            "prior_outcome_source": True,
            "source": "persisted_prediction_outcome_evaluation",
        },
    )


def _persisted_review_base_metadata(
    *,
    prior: _LoadedPriorReport,
    prior_candidate: PredictionCandidate,
    record_outcome_evaluation_id: str,
    artifact: ArtifactRecord | None,
) -> JsonObject:
    metadata: JsonObject = {
        **_review_metadata(prior=prior, prior_candidate=prior_candidate),
        "source": "persisted_prediction_outcome_evaluation",
        "prior_candidate_id": prior_candidate.candidate_id,
        "source_outcome_evaluation_id": record_outcome_evaluation_id,
    }
    if artifact is not None:
        metadata["source_outcome_evaluation_artifact_id"] = artifact.artifact_id
    return metadata


def _outcome_payload_compatibility_limitations(
    *,
    payload: PredictionOutcomeEvaluationArtifactPayload,
    prior_candidate: PredictionCandidate,
) -> tuple[str, ...]:
    target = payload.target
    limitations: list[str] = []
    if target.candidate_id != prior_candidate.candidate_id:
        limitations.append(
            "Persisted outcome target candidate_id does not match the prior report candidate."
        )
    if target.instrument_id != prior_candidate.instrument_id:
        limitations.append(
            "Persisted outcome target instrument_id does not match the prior report candidate."
        )
    if target.horizon != prior_candidate.horizon:
        limitations.append("Persisted outcome target horizon does not match the prior candidate.")
    if target.prediction_type != prior_candidate.prediction_type:
        limitations.append(
            "Persisted outcome target prediction_type does not match the prior candidate."
        )
    return tuple(limitations)


def _outcome_evidence_sources(
    *,
    store: SQLiteStore,
    payload: PredictionOutcomeEvaluationArtifactPayload,
) -> tuple[tuple[EvidenceReference, ...], tuple[SourceEvidence, ...], tuple[str, ...]]:
    references = _dedupe_evidence_references(
        (
            *payload.outcome_evaluation.evidence,
            *payload.outcome_evaluation.outcome.outcome_evidence,
        )
    )
    evidence_sources: list[SourceEvidence] = []
    limitations: list[str] = []
    kept_references: list[EvidenceReference] = []
    for reference in references:
        record = store.get_evidence(reference.evidence_id)
        if record is None:
            limitations.append(
                f"Persisted outcome evidence is missing from storage: {reference.evidence_id}."
            )
            continue
        evidence_sources.append(source_evidence_from_record(record))
        kept_references.append(reference)
    return tuple(kept_references), _dedupe_source_evidence(evidence_sources), tuple(limitations)


def _persisted_prior_review_status(status: str) -> str:
    if status in {"confirmed", "missed", "mixed", "inconclusive"}:
        return "available"
    if status == "pending":
        return "pending"
    if status == "stale":
        return "stale"
    if status in {"unavailable", "not_evaluable"}:
        return "unavailable"
    return "not_available"


def _persisted_outcome_review_summary(
    payload: PredictionOutcomeEvaluationArtifactPayload,
) -> str:
    outcome = payload.outcome_evaluation.outcome
    if outcome.result_summary:
        return outcome.result_summary
    score = payload.outcome_evaluation.quality_score
    if score is None:
        return (
            f"Persisted outcome evaluation is {payload.outcome_evaluation.status.value} for the "
            "prior prediction scenario."
        )
    return (
        f"Persisted outcome evaluation is {payload.outcome_evaluation.status.value} for the "
        f"prior prediction scenario with quality score {score:.2f}."
    )


def _freshness_metadata(
    payload: PredictionOutcomeEvaluationArtifactPayload,
) -> JsonObject:
    metadata = payload.target.metadata.get("reliability_freshness")
    if isinstance(metadata, dict):
        return dict(metadata)
    snapshot_metadata = payload.target.candidate_snapshot.get("reliability_freshness")
    if isinstance(snapshot_metadata, dict):
        return dict(snapshot_metadata)
    return {}


def _reliability_freshness_limitations(metadata: JsonObject) -> tuple[str, ...]:
    limitations: list[str] = []
    limitations.extend(
        _freshness_record_limitations(
            metadata.get("evidence_aging_records"),
            id_key="evidence_id",
            status_keys=("age_status", "freshness_status"),
            label="Prior evidence freshness review",
        )
    )
    limitations.extend(
        _freshness_record_limitations(
            metadata.get("artifact_freshness_reviews"),
            id_key="artifact_id",
            status_keys=("freshness_status", "status"),
            label="Outcome artifact freshness review",
        )
    )
    return tuple(dict.fromkeys(limitations))


def _freshness_record_limitations(
    value: object,
    *,
    id_key: str,
    status_keys: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    limitations: list[str] = []
    for record in value:
        if not isinstance(record, dict):
            continue
        record_id = record.get(id_key)
        if not isinstance(record_id, str) or not record_id.strip():
            continue
        status = next(
            (
                str(record[key])
                for key in status_keys
                if isinstance(record.get(key), str) and str(record[key]).strip()
            ),
            "unknown",
        )
        if status == "fresh":
            continue
        limitations.append(f"{label} marked {record_id} as {status}.")
    return tuple(limitations)


def _review_with_outcome_contracts(
    *,
    review: PriorOutcomeReview,
    candidate: PredictionCandidate,
    prior: _LoadedPriorReport | None,
    report_date: date,
    reviewed_at: datetime,
) -> PriorOutcomeReview:
    if review.metadata.get("source") == "persisted_prediction_outcome_evaluation":
        return review
    outcome = _prediction_outcome_from_review(
        review=review,
        candidate=candidate,
        prior=prior,
        report_date=report_date,
        reviewed_at=reviewed_at,
    )
    outcome_evaluation = _prediction_outcome_evaluation_from_review(
        review=review,
        candidate=candidate,
        outcome=outcome,
        reviewed_at=reviewed_at,
    )
    return review.model_copy(
        update={
            "metadata": {
                **review.metadata,
                "prediction_outcome": outcome.model_dump(mode="json"),
                "prediction_outcome_evaluation": outcome_evaluation.model_dump(mode="json"),
            }
        }
    )


def _prediction_outcome_from_review(
    *,
    review: PriorOutcomeReview,
    candidate: PredictionCandidate,
    prior: _LoadedPriorReport | None,
    report_date: date,
    reviewed_at: datetime,
) -> PredictionOutcome:
    status = _outcome_status(review.status)
    observed_result = (
        _outcome_result(candidate) if status == PredictionOutcomeStatus.OBSERVED else None
    )
    window_start = _outcome_window_start(prior, report_date)
    window_end = (
        reviewed_at if reviewed_at > window_start else _date_start(report_date) + timedelta(days=1)
    )
    return PredictionOutcome(
        outcome_id=f"outcome-{stable_digest(review.review_id)}",
        candidate_id=candidate.candidate_id,
        instrument_id=candidate.instrument_id,
        symbol=candidate.symbol,
        prediction_type=candidate.prediction_type,
        horizon=candidate.horizon,
        evaluation_window_start=window_start,
        evaluation_window_end=window_end,
        status=status,
        observed_result=observed_result,
        observed_at=reviewed_at if status == PredictionOutcomeStatus.OBSERVED else None,
        result_summary=review.summary if status == PredictionOutcomeStatus.OBSERVED else None,
        outcome_evidence=review.outcome_evidence
        if status == PredictionOutcomeStatus.OBSERVED
        else (),
        artifact_ids=review.artifact_ids,
        limitations=review.limitations,
        metadata={
            "prior_outcome_review_id": review.review_id,
            "prior_outcome_review_status": review.status,
        },
    )


def _prediction_outcome_evaluation_from_review(
    *,
    review: PriorOutcomeReview,
    candidate: PredictionCandidate,
    outcome: PredictionOutcome,
    reviewed_at: datetime,
) -> PredictionOutcomeEvaluation:
    status = _outcome_evaluation_status(review.status, candidate)
    resolved_statuses = {
        PredictionOutcomeEvaluationStatus.CONFIRMED,
        PredictionOutcomeEvaluationStatus.MISSED,
        PredictionOutcomeEvaluationStatus.MIXED,
        PredictionOutcomeEvaluationStatus.INCONCLUSIVE,
    }
    return PredictionOutcomeEvaluation(
        outcome_evaluation_id=f"outcome-evaluation-{stable_digest(review.review_id)}",
        outcome_id=outcome.outcome_id,
        candidate_id=candidate.candidate_id,
        instrument_id=candidate.instrument_id,
        symbol=candidate.symbol,
        evaluated_at=reviewed_at,
        status=status,
        outcome=outcome,
        quality_score=candidate.confidence if status in resolved_statuses else None,
        evidence=review.outcome_evidence if status in resolved_statuses else (),
        artifact_ids=review.artifact_ids,
        limitations=review.limitations
        or (() if status in resolved_statuses else ("Outcome could not be evaluated.",)),
        metadata={"prior_outcome_review_id": review.review_id},
    )


def _outcome_status(review_status: str) -> PredictionOutcomeStatus:
    if review_status == "available":
        return PredictionOutcomeStatus.OBSERVED
    if review_status == "pending":
        return PredictionOutcomeStatus.PENDING
    if review_status == "stale":
        return PredictionOutcomeStatus.STALE
    return PredictionOutcomeStatus.UNAVAILABLE


def _outcome_result(candidate: PredictionCandidate) -> PredictionOutcomeResult:
    if candidate.status == PredictionStatus.CONTRADICTED or candidate.evidence_against:
        return PredictionOutcomeResult.CONTRADICTED
    if candidate.status == PredictionStatus.EVIDENCE_SUPPORTED and candidate.evidence_for:
        return PredictionOutcomeResult.SUPPORTED
    if candidate.status == PredictionStatus.INSUFFICIENT_EVIDENCE:
        return PredictionOutcomeResult.INSUFFICIENT_DATA
    return PredictionOutcomeResult.MIXED


def _outcome_evaluation_status(
    review_status: str,
    candidate: PredictionCandidate,
) -> PredictionOutcomeEvaluationStatus:
    if review_status == "available":
        if candidate.status == PredictionStatus.CONTRADICTED or candidate.evidence_against:
            return PredictionOutcomeEvaluationStatus.MISSED
        if candidate.status == PredictionStatus.EVIDENCE_SUPPORTED and candidate.evidence_for:
            return PredictionOutcomeEvaluationStatus.CONFIRMED
        return PredictionOutcomeEvaluationStatus.INCONCLUSIVE
    if review_status == "pending":
        return PredictionOutcomeEvaluationStatus.PENDING
    if review_status == "stale":
        return PredictionOutcomeEvaluationStatus.STALE
    return PredictionOutcomeEvaluationStatus.NOT_EVALUABLE


def _outcome_window_start(prior: _LoadedPriorReport | None, report_date: date) -> datetime:
    if prior is not None:
        return _date_start(prior.artifact.report_date)
    return _date_start(report_date)


def _date_start(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=UTC)


def _change_triggers(
    *,
    candidate: PredictionCandidate,
    prior: _LoadedPriorReport | None,
    review: PriorOutcomeReview,
    missing_provider_names: tuple[str, ...],
    stale_provider_names: tuple[str, ...],
) -> tuple[PredictionChangeTrigger, ...]:
    triggers: list[PredictionChangeTrigger] = []
    if candidate.evidence_against:
        triggers.append(
            PredictionChangeTrigger(
                trigger_id=f"change-{stable_digest(candidate.candidate_id)}-contradictory-evidence",
                summary=(
                    "Additional opposing source evidence matching the current dissenting evidence "
                    "would reduce confidence or move the scenario toward contradicted."
                ),
                trigger_type="new_source_evidence",
                evidence=candidate.evidence_against,
            )
        )
    if candidate.evidence_for:
        triggers.append(
            PredictionChangeTrigger(
                trigger_id=f"change-{stable_digest(candidate.candidate_id)}-supporting-evidence",
                summary=(
                    "Additional supporting source evidence matching the current cited evidence "
                    "would increase confidence or keep the scenario evidence-supported."
                ),
                trigger_type="new_source_evidence",
                evidence=candidate.evidence_for,
            )
        )
    if review.status == "available":
        triggers.append(
            PredictionChangeTrigger(
                trigger_id=f"change-{stable_digest(candidate.candidate_id)}-outcome-data",
                summary=(
                    "Follow-up outcome evidence from the prior report review would alter "
                    "confidence if it weakens or contradicts the current scenario."
                ),
                trigger_type="outcome_data",
                evidence=review.outcome_evidence,
                rationale=f"Prior review {review.review_id} is available.",
            )
        )
    prior_candidate = (
        _matching_prior_candidate(prior.report, candidate) if prior and prior.report else None
    )
    if prior_candidate is not None and prior_candidate.baseline != candidate.baseline:
        triggers.append(
            PredictionChangeTrigger(
                trigger_id=f"change-{stable_digest(candidate.candidate_id)}-baseline",
                summary=(
                    "A baseline comparison change from the prior report would alter confidence "
                    "or scenario status."
                ),
                trigger_type="baseline_change",
                rationale="Current and prior baseline summaries differ.",
            )
        )
    provider_names = tuple(dict.fromkeys((*missing_provider_names, *stale_provider_names)))
    if provider_names:
        triggers.append(
            PredictionChangeTrigger(
                trigger_id=f"change-{stable_digest(candidate.candidate_id)}-provider-refresh",
                summary=(
                    "A provider refresh would alter confidence if missing or stale inputs change."
                ),
                trigger_type="provider_refresh",
                rationale="Providers needing refresh: " + ", ".join(provider_names),
                metadata={"provider_names": list(provider_names)},
            )
        )
    return tuple(triggers)


def _prior_source_reference(review: PriorOutcomeReview) -> ReportSourceReference:
    return ReportSourceReference(
        reference_id=f"source-ref-{review.review_id}",
        label=f"Prior outcome review {review.review_id}",
        reference_type="prior_outcome",
        evidence_ids=tuple(reference.evidence_id for reference in review.outcome_evidence),
        artifact_ids=review.artifact_ids,
        candidate_ids=((review.candidate_id,) if review.candidate_id else ()),
        prior_outcome_review_ids=(review.review_id,),
        metadata={"status": review.status},
    )


def _prior_material_claim_trace(review: PriorOutcomeReview) -> MaterialClaimTrace:
    return MaterialClaimTrace(
        claim_id=f"claim-{review.review_id}",
        claim=review.summary,
        claim_type="prior_outcome",
        evidence=review.outcome_evidence,
        artifact_ids=review.artifact_ids,
        candidate_ids=((review.candidate_id,) if review.candidate_id else ()),
        prior_outcome_review_ids=(review.review_id,),
        metadata={"status": review.status},
    )


def _matching_prior_candidate(
    prior_report: DailyReport,
    candidate: PredictionCandidate,
) -> PredictionCandidate | None:
    same_instrument = [
        item
        for item in prior_report.prediction_candidates
        if item.instrument_id == candidate.instrument_id
    ]
    for prior_candidate in same_instrument:
        if (
            prior_candidate.horizon == candidate.horizon
            and prior_candidate.prediction_type == candidate.prediction_type
        ):
            return prior_candidate
    return same_instrument[0] if same_instrument else None


def _prior_stale_reasons(
    prior_report: DailyReport,
    *,
    current_report_date: date,
) -> tuple[str, ...]:
    reasons: list[str] = []
    stale_after = prior_report.report_date + timedelta(days=PRIOR_REPORT_STALE_AFTER_DAYS)
    if stale_after < current_report_date:
        reasons.append(
            "Prior report is older than "
            f"{PRIOR_REPORT_STALE_AFTER_DAYS} days for this review window."
        )
    return tuple(reasons)


def _available_review_summary(candidate: PredictionCandidate) -> str:
    if candidate.status == PredictionStatus.CONTRADICTED or candidate.evidence_against:
        return "Follow-up source evidence contradicts or weakens the prior prediction scenario."
    if candidate.status == PredictionStatus.EVIDENCE_SUPPORTED and candidate.evidence_for:
        return "Follow-up source evidence supports the prior prediction scenario."
    return "Follow-up source evidence is mixed or inconclusive relative to the prior scenario."


def _dedupe_change_triggers(
    triggers: tuple[PredictionChangeTrigger, ...],
) -> tuple[PredictionChangeTrigger, ...]:
    deduped: dict[str, PredictionChangeTrigger] = {}
    for trigger in triggers:
        deduped.setdefault(trigger.trigger_id, trigger)
    return tuple(deduped.values())


def _dedupe_audit_artifacts(
    artifacts: tuple[AuditArtifact, ...] | list[AuditArtifact],
) -> tuple[AuditArtifact, ...]:
    deduped: dict[str, AuditArtifact] = {}
    for artifact in artifacts:
        deduped.setdefault(artifact.artifact_id, artifact)
    return tuple(deduped.values())


def _dedupe_source_evidence(
    evidence_sources: tuple[SourceEvidence, ...] | list[SourceEvidence],
) -> tuple[SourceEvidence, ...]:
    deduped: dict[str, SourceEvidence] = {}
    for evidence in evidence_sources:
        deduped.setdefault(evidence.evidence_id, evidence)
    return tuple(deduped.values())


def _dedupe_evidence_references(
    references: tuple[EvidenceReference, ...],
) -> tuple[EvidenceReference, ...]:
    deduped: dict[str, EvidenceReference] = {}
    for reference in references:
        deduped.setdefault(reference.evidence_id, reference)
    return tuple(deduped.values())


def _explicit_prior_report_artifact_id(metadata: JsonObject) -> str | None:
    for key in ("prior_report_artifact_id", "prior_json_report_artifact_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    lineage = metadata.get("report_lineage")
    if isinstance(lineage, dict):
        value = lineage.get("prior_report_artifact_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _review_metadata(
    *,
    prior: _LoadedPriorReport,
    prior_candidate: PredictionCandidate | None,
) -> JsonObject:
    metadata: JsonObject = {
        "prior_report_artifact_id": prior.artifact.artifact_id,
        "prior_report_run_id": prior.artifact.run_id,
        "prior_report_sha256": prior.artifact.sha256,
        "prior_report_date": prior.artifact.report_date.isoformat(),
    }
    if prior_candidate is not None:
        metadata.update(
            {
                "prior_candidate_id": prior_candidate.candidate_id,
                "prior_status": prior_candidate.status.value,
                "prior_direction": prior_candidate.direction.value,
            }
        )
    if prior.warnings:
        metadata["prior_report_warnings"] = list(prior.warnings)
    return metadata


def _prior_metadata(
    prior: _LoadedPriorReport | None,
    review: PriorOutcomeReview,
) -> JsonObject:
    metadata: JsonObject = {"review_id": review.review_id, "status": review.status}
    if prior is not None:
        metadata.update(
            {
                "prior_report_artifact_id": prior.artifact.artifact_id,
                "prior_report_run_id": prior.artifact.run_id,
                "prior_report_date": prior.artifact.report_date.isoformat(),
            }
        )
    return metadata


__all__ = [
    "PRIOR_REPORT_STALE_AFTER_DAYS",
    "PriorOutcomeContext",
    "apply_prior_outcome_context",
]
