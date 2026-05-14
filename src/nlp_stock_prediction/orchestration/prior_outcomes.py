"""Prior report review and prediction change-tracking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from nlp_stock_prediction.contracts import (
    DailyReport,
    Instrument,
    MaterialClaimTrace,
    PredictionCandidate,
    PredictionChangeTrigger,
    PredictionStatus,
    PriorOutcomeReview,
    ReportSourceReference,
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
)
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.orchestration.phase2_common import file_sha256, stable_digest
from nlp_stock_prediction.reporting.json import load_json_report
from nlp_stock_prediction.storage.records import ReportArtifactRecord, ResearchRunRecord
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
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _LoadedPriorReport:
    artifact: ReportArtifactRecord
    report: DailyReport | None
    audit_artifact: AuditArtifact | None
    warnings: tuple[str, ...]


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
    )
    updated_candidates: list[PredictionCandidate] = []
    reviews: list[PriorOutcomeReview] = []
    source_references: list[ReportSourceReference] = []
    traces: list[MaterialClaimTrace] = []

    for candidate in prediction_candidates:
        review = _prior_outcome_review(
            candidate=candidate,
            prior=prior,
            report_date=report_date,
            reviewed_at=reviewed_at,
        )
        review = _review_with_outcome_contracts(
            review=review,
            candidate=candidate,
            prior=prior,
            report_date=report_date,
            reviewed_at=reviewed_at,
        )
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

    audit_artifacts = (prior.audit_artifact,) if prior and prior.audit_artifact is not None else ()
    warnings = prior.warnings if prior else ()
    return PriorOutcomeContext(
        prediction_candidates=tuple(updated_candidates),
        prior_outcome_reviews=tuple(reviews),
        audit_artifacts=audit_artifacts,
        source_references=tuple(source_references),
        material_claim_traces=tuple(traces),
        warnings=warnings,
    )


def _load_prior_report(
    *,
    store: SQLiteStore,
    repo_root: Path,
    run: ResearchRunRecord,
    report_date: date,
    instrument: Instrument,
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
    candidate: PredictionCandidate,
    prior: _LoadedPriorReport | None,
    report_date: date,
    reviewed_at: datetime,
) -> PriorOutcomeReview:
    review_id = f"prior-outcome-{stable_digest(candidate.candidate_id)}"
    if prior is None:
        return PriorOutcomeReview(
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
    artifact_ids = (prior.artifact.artifact_id,)
    if prior.report is None:
        return PriorOutcomeReview(
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
    prior_candidate = _matching_prior_candidate(prior.report, candidate)
    if prior_candidate is None:
        return PriorOutcomeReview(
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
                "No prior candidate matched the current instrument, horizon, and prediction type.",
            ),
            metadata=_review_metadata(prior=prior, prior_candidate=None),
        )
    stale_reasons = _prior_stale_reasons(prior.report, current_report_date=report_date)
    if stale_reasons:
        return PriorOutcomeReview(
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
    outcome_evidence = tuple(dict.fromkeys((*candidate.evidence_for, *candidate.evidence_against)))
    if not outcome_evidence:
        return PriorOutcomeReview(
            review_id=review_id,
            status="pending",
            summary="A prior matching prediction exists, but no follow-up evidence is available.",
            candidate_id=candidate.candidate_id,
            instrument_id=candidate.instrument_id,
            original_report_date=prior.report.report_date,
            reviewed_at=reviewed_at,
            horizon=candidate.horizon,
            artifact_ids=artifact_ids,
            limitations=("Current report evidence did not cite follow-up source evidence.",),
            metadata=_review_metadata(prior=prior, prior_candidate=prior_candidate),
        )
    return PriorOutcomeReview(
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


def _review_with_outcome_contracts(
    *,
    review: PriorOutcomeReview,
    candidate: PredictionCandidate,
    prior: _LoadedPriorReport | None,
    report_date: date,
    reviewed_at: datetime,
) -> PriorOutcomeReview:
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
