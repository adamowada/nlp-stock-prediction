"""Point-in-time prediction outcome evaluation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject, JsonValue
from nlp_stock_prediction.contracts.enums import (
    Direction,
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
    PredictionType,
    SignalArtifactFamily,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.evaluation import (
    BaselineComparison,
    PredictionEvaluationTarget,
    PredictionOutcome,
    PredictionOutcomeArtifactPayload,
    PredictionOutcomeEvaluation,
    PredictionOutcomeEvaluationArtifactPayload,
    SignalArtifactReference,
)
from nlp_stock_prediction.contracts.provenance import EvidenceReference
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase4_common import safe_phase4_tool_execution
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    OutcomeArtifactLinkRecord,
    OutcomeEvaluationArtifactLinkRecord,
    OutcomeEvaluationEvidenceLinkRecord,
    OutcomeEvidenceLinkRecord,
    PredictionCandidateRecord,
    PredictionOutcomeEvaluationRecord,
    PredictionOutcomeRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE6_OUTCOME_TOOL_NAME = "phase6_point_in_time_outcome_evaluation"
PHASE6_OUTCOME_TOOL_VERSION = "phase6.outcome-evaluation.v1"

_SIGNAL_ARTIFACT_TYPES = frozenset(
    {
        "market_data",
        "technical_package",
        "ml_forecast",
        "normalized_evidence",
        "analysis_context",
    }
)


@dataclass(frozen=True)
class PointInTimeOutcomeEvaluationArtifacts:
    """Artifacts and contracts produced by one outcome-evaluation write."""

    target: PredictionEvaluationTarget
    outcome: PredictionOutcome
    outcome_evaluation: PredictionOutcomeEvaluation
    outcome_artifact: AuditArtifact
    outcome_evaluation_artifact: AuditArtifact
    tool_run_id: str


def build_prediction_evaluation_target(
    *,
    store: SQLiteStore,
    run_id: str,
    candidate_id: str,
    point_in_time_cutoff: datetime,
    evaluation_window_start: datetime,
    evaluation_window_end: datetime,
    target_id: str | None = None,
    report_date: date | None = None,
) -> PredictionEvaluationTarget:
    """Freeze a stored prediction candidate using only records available by cutoff."""

    run = store.get_research_run(run_id)
    if run is None:
        raise ValueError(f"research run does not exist: {run_id}")
    candidate = store.get_prediction_candidate(candidate_id)
    if candidate is None:
        raise ValueError(f"prediction candidate does not exist: {candidate_id}")
    if candidate.run_id not in {None, run_id}:
        raise ValueError("prediction candidate run_id does not match target run_id")
    instrument = store.get_instrument(candidate.instrument_id)
    if instrument is None:
        raise ValueError(f"instrument does not exist: {candidate.instrument_id}")

    cutoff = _aware_utc(point_in_time_cutoff, "point_in_time_cutoff")
    window_start = _aware_utc(evaluation_window_start, "evaluation_window_start")
    window_end = _aware_utc(evaluation_window_end, "evaluation_window_end")
    prediction_created_at = _aware_utc(run.started_at, "run.started_at")
    eligible_evidence_ids, source_artifact_ids, evidence_limitations = _eligible_evidence_ids(
        store=store,
        candidate=candidate,
        cutoff=cutoff,
    )
    signal_artifacts, report_artifact_ids, artifact_limitations = _eligible_artifacts(
        store=store,
        candidate=candidate,
        cutoff=cutoff,
    )
    baseline = _baseline_comparison_from_candidate(candidate)
    limitations = [
        *evidence_limitations,
        *artifact_limitations,
    ]
    if baseline is None:
        limitations.append("No stored baseline comparison was available at the cutoff.")

    resolved_target_id = target_id or _target_id(
        run_id=run_id,
        candidate_id=candidate_id,
        cutoff=cutoff,
        evaluation_window_start=window_start,
        evaluation_window_end=window_end,
    )
    return PredictionEvaluationTarget(
        target_id=resolved_target_id,
        run_id=run_id,
        candidate_id=candidate_id,
        instrument_id=candidate.instrument_id,
        symbol=instrument.symbol,
        prediction_type=_prediction_type(candidate.prediction_type),
        horizon=_time_horizon(candidate.prediction_horizon),
        direction=_direction(candidate.direction),
        report_date=report_date,
        prediction_created_at=prediction_created_at,
        point_in_time_cutoff=cutoff,
        evaluation_window_start=window_start,
        evaluation_window_end=window_end,
        candidate_snapshot=_candidate_snapshot(
            candidate=candidate,
            symbol=instrument.symbol,
            evidence_ids=eligible_evidence_ids,
            signal_artifacts=signal_artifacts,
            source_artifact_ids=source_artifact_ids,
            report_artifact_ids=report_artifact_ids,
        ),
        baseline_comparison=baseline,
        evidence_ids=eligible_evidence_ids,
        signal_artifacts=signal_artifacts,
        report_artifact_ids=report_artifact_ids,
        source_artifact_ids=source_artifact_ids,
        limitations=tuple(dict.fromkeys(limitations)),
        metadata={
            "source": PHASE6_OUTCOME_TOOL_NAME,
            "stored_candidate_run_id": candidate.run_id,
        },
    )


def build_prediction_outcome(
    *,
    target: PredictionEvaluationTarget,
    status: PredictionOutcomeStatus | str | None = None,
    observed_result: PredictionOutcomeResult | str | None = None,
    observed_at: datetime | None = None,
    result_summary: str | None = None,
    result_value: float | None = None,
    baseline_value: float | None = None,
    outcome_evidence: tuple[EvidenceReference, ...] = (),
    artifact_ids: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    outcome_id: str | None = None,
    metadata: JsonObject | None = None,
) -> PredictionOutcome:
    """Build a typed observed, pending, stale, or unavailable outcome."""

    resolved_result = _prediction_outcome_result(observed_result)
    resolved_status = _prediction_outcome_status(status, resolved_result)
    resolved_observed_at = None if observed_at is None else _aware_utc(observed_at, "observed_at")
    resolved_limitations = limitations
    if resolved_status != PredictionOutcomeStatus.OBSERVED and not resolved_limitations:
        resolved_limitations = ("Outcome was not observed at evaluation time.",)

    return PredictionOutcome(
        outcome_id=outcome_id
        or _outcome_id(
            target=target,
            observed_result=resolved_result,
            observed_at=resolved_observed_at,
            status=resolved_status,
        ),
        candidate_id=target.candidate_id,
        instrument_id=target.instrument_id,
        symbol=target.symbol,
        prediction_type=target.prediction_type,
        horizon=target.horizon,
        evaluation_window_start=target.evaluation_window_start,
        evaluation_window_end=target.evaluation_window_end,
        status=resolved_status,
        observed_result=resolved_result,
        observed_at=resolved_observed_at,
        result_summary=result_summary,
        result_value=result_value,
        baseline_value=baseline_value,
        outcome_evidence=outcome_evidence,
        artifact_ids=artifact_ids,
        limitations=resolved_limitations,
        metadata={} if metadata is None else metadata,
    )


def evaluate_prediction_outcome(
    *,
    target: PredictionEvaluationTarget,
    outcome: PredictionOutcome,
    evaluated_at: datetime,
    outcome_evaluation_id: str | None = None,
    evidence: tuple[EvidenceReference, ...] = (),
    artifact_ids: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    metadata: JsonObject | None = None,
) -> PredictionOutcomeEvaluation:
    """Evaluate an outcome against the frozen prediction target."""

    resolved_evaluated_at = _aware_utc(evaluated_at, "evaluated_at")
    status = _outcome_evaluation_status(target=target, outcome=outcome)
    quality_score = _quality_score(status)
    resolved_limitations = limitations
    if status in {
        PredictionOutcomeEvaluationStatus.PENDING,
        PredictionOutcomeEvaluationStatus.STALE,
        PredictionOutcomeEvaluationStatus.NOT_EVALUABLE,
    }:
        resolved_limitations = tuple(
            dict.fromkeys((*limitations, *outcome.limitations, *target.limitations))
        )
        if not resolved_limitations:
            resolved_limitations = ("Outcome evaluation is not resolved.",)

    resolved_evidence = _dedupe_evidence_references((*outcome.outcome_evidence, *evidence))
    resolved_artifact_ids = tuple(dict.fromkeys((*outcome.artifact_ids, *artifact_ids)))
    return PredictionOutcomeEvaluation(
        outcome_evaluation_id=outcome_evaluation_id
        or _outcome_evaluation_id(
            target=target,
            outcome=outcome,
            evaluated_at=resolved_evaluated_at,
        ),
        outcome_id=outcome.outcome_id,
        candidate_id=target.candidate_id,
        instrument_id=target.instrument_id,
        symbol=target.symbol,
        evaluated_at=resolved_evaluated_at,
        status=status,
        outcome=outcome,
        quality_score=quality_score,
        baseline_comparison=target.baseline_comparison,
        evidence=resolved_evidence,
        artifact_ids=resolved_artifact_ids,
        limitations=resolved_limitations,
        metadata={} if metadata is None else metadata,
    )


def write_point_in_time_outcome_evaluation_artifacts(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    target: PredictionEvaluationTarget,
    status: PredictionOutcomeStatus | str | None = None,
    observed_result: PredictionOutcomeResult | str | None = None,
    observed_at: datetime | None = None,
    result_summary: str | None = None,
    result_value: float | None = None,
    baseline_value: float | None = None,
    outcome_evidence: tuple[EvidenceReference, ...] = (),
    market_artifact_ids: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    metadata: JsonObject | None = None,
    created_at: datetime | None = None,
    evaluated_at: datetime | None = None,
    outcome_artifact_filename: str | None = None,
    outcome_evaluation_artifact_filename: str | None = None,
    tool_run_id: str | None = None,
    record_tool_run: bool = True,
) -> PointInTimeOutcomeEvaluationArtifacts:
    """Write and persist point-in-time outcome and review artifacts."""

    if target.run_id != run_id:
        raise ValueError("outcome target run_id must match run_id")
    created = _aware_utc(created_at or datetime.now(UTC), "created_at")
    evaluated = _aware_utc(evaluated_at or created, "evaluated_at")
    observed_artifact_ids = tuple(dict.fromkeys(market_artifact_ids))
    outcome = build_prediction_outcome(
        target=target,
        status=status,
        observed_result=observed_result,
        observed_at=observed_at,
        result_summary=result_summary,
        result_value=result_value,
        baseline_value=baseline_value,
        outcome_evidence=outcome_evidence,
        artifact_ids=observed_artifact_ids,
        limitations=limitations,
        metadata={
            **({} if metadata is None else metadata),
            "target_id": target.target_id,
        },
    )
    outcome_evaluation = evaluate_prediction_outcome(
        target=target,
        outcome=outcome,
        evaluated_at=evaluated,
        artifact_ids=observed_artifact_ids,
        limitations=limitations,
        metadata={
            "target_id": target.target_id,
            "source": PHASE6_OUTCOME_TOOL_NAME,
        },
    )
    digest = _digest(
        "|".join(
            (
                run_id,
                target.target_id,
                outcome.outcome_id,
                outcome_evaluation.outcome_evaluation_id,
            )
        )
    )
    resolved_tool_run_id = (
        tool_run_id or f"tool-phase6-outcome-{_slug(target.candidate_id)}-{digest[:12]}"
    )
    inputs: JsonObject = {
        "target_id": target.target_id,
        "candidate_id": target.candidate_id,
        "outcome_status": outcome.status.value,
        "observed_result": outcome.observed_result.value if outcome.observed_result else None,
        "market_artifact_ids": list(market_artifact_ids),
        "outcome_evidence_ids": [reference.evidence_id for reference in outcome_evidence],
    }

    def write_artifacts() -> PointInTimeOutcomeEvaluationArtifacts:
        if record_tool_run:
            store.record_tool_run(
                ToolRunRecord(
                    tool_run_id=resolved_tool_run_id,
                    run_id=run_id,
                    tool_name=PHASE6_OUTCOME_TOOL_NAME,
                    tool_version=PHASE6_OUTCOME_TOOL_VERSION,
                    status="successful",
                    started_at=created,
                    completed_at=evaluated,
                    inputs=inputs,
                    warnings=outcome_evaluation.limitations,
                )
            )
        outcome_payload = PredictionOutcomeArtifactPayload(
            run_id=run_id,
            created_at=created,
            target=target,
            outcome=outcome,
            source_evidence_ids=tuple(
                reference.evidence_id for reference in outcome.outcome_evidence
            ),
            market_artifact_ids=observed_artifact_ids,
            limitations=outcome.limitations,
            metadata={
                "target_id": target.target_id,
                "outcome_evaluation_status": outcome_evaluation.status.value,
            },
        )
        outcome_artifact_id = (
            f"artifact-prediction-outcome-{_slug(target.candidate_id)}-{digest[:8]}"
        )
        outcome_artifact = ArtifactIndex.for_directory(
            store=store,
            repo_root=repo_root,
            base_dir=artifact_dir,
            created_at=created,
            produced_by=PHASE6_OUTCOME_TOOL_NAME,
            tool_run_id=resolved_tool_run_id,
            schema_version=outcome_payload.schema_version,
        ).write_json(
            artifact_id=outcome_artifact_id,
            artifact_type="prediction_outcome",
            filename=outcome_artifact_filename
            or f"prediction-outcomes/{_slug(target.candidate_id)}.json",
            payload=cast(JsonObject, outcome_payload.model_dump(mode="json")),
            record_count=1,
            metadata={
                "run_id": run_id,
                "target_id": target.target_id,
                "outcome_id": outcome.outcome_id,
                "candidate_id": target.candidate_id,
                "status": outcome.status.value,
                "observed_result": (
                    outcome.observed_result.value if outcome.observed_result else None
                ),
            },
        )
        outcome_evaluation_with_artifact = outcome_evaluation.model_copy(
            update={
                "artifact_ids": tuple(
                    dict.fromkeys((*outcome_evaluation.artifact_ids, outcome_artifact.artifact_id))
                )
            }
        )
        review_payload = PredictionOutcomeEvaluationArtifactPayload(
            run_id=run_id,
            created_at=evaluated,
            target=target,
            outcome_evaluation=outcome_evaluation_with_artifact,
            baseline_comparison=target.baseline_comparison,
            evidence_ids=tuple(
                reference.evidence_id for reference in outcome_evaluation_with_artifact.evidence
            ),
            artifact_ids=outcome_evaluation_with_artifact.artifact_ids,
            limitations=outcome_evaluation_with_artifact.limitations,
            metadata={
                "target_id": target.target_id,
                "outcome_id": outcome.outcome_id,
            },
        )
        review_artifact_id = (
            f"artifact-prediction-outcome-evaluation-{_slug(target.candidate_id)}-{digest[:8]}"
        )
        review_artifact = ArtifactIndex.for_directory(
            store=store,
            repo_root=repo_root,
            base_dir=artifact_dir,
            created_at=evaluated,
            produced_by=PHASE6_OUTCOME_TOOL_NAME,
            tool_run_id=resolved_tool_run_id,
            schema_version=review_payload.schema_version,
        ).write_json(
            artifact_id=review_artifact_id,
            artifact_type="prediction_outcome_evaluation",
            filename=outcome_evaluation_artifact_filename
            or f"prediction-outcome-evaluations/{_slug(target.candidate_id)}.json",
            payload=cast(JsonObject, review_payload.model_dump(mode="json")),
            record_count=1,
            metadata={
                "run_id": run_id,
                "target_id": target.target_id,
                "outcome_id": outcome.outcome_id,
                "outcome_evaluation_id": outcome_evaluation.outcome_evaluation_id,
                "candidate_id": target.candidate_id,
                "status": outcome_evaluation.status.value,
                "quality_score": outcome_evaluation.quality_score,
            },
        )
        _persist_outcome(
            store=store,
            target=target,
            outcome=outcome,
            outcome_artifact=outcome_artifact,
            market_artifact_ids=observed_artifact_ids,
            created_at=created,
        )
        _persist_outcome_evaluation(
            store=store,
            target=target,
            outcome_evaluation=outcome_evaluation_with_artifact,
            review_artifact=review_artifact,
            created_at=evaluated,
        )
        return PointInTimeOutcomeEvaluationArtifacts(
            target=target,
            outcome=outcome,
            outcome_evaluation=outcome_evaluation_with_artifact,
            outcome_artifact=outcome_artifact,
            outcome_evaluation_artifact=review_artifact,
            tool_run_id=resolved_tool_run_id,
        )

    if not record_tool_run:
        return write_artifacts()

    with safe_phase4_tool_execution(
        store=store,
        artifact_roots=(artifact_dir,),
        tool_run_id=resolved_tool_run_id,
        run_id=run_id,
        tool_name=PHASE6_OUTCOME_TOOL_NAME,
        tool_version=PHASE6_OUTCOME_TOOL_VERSION,
        started_at=created,
        inputs=inputs,
    ):
        return write_artifacts()


def _persist_outcome(
    *,
    store: SQLiteStore,
    target: PredictionEvaluationTarget,
    outcome: PredictionOutcome,
    outcome_artifact: AuditArtifact,
    market_artifact_ids: tuple[str, ...],
    created_at: datetime,
) -> None:
    store.upsert_prediction_outcome(
        PredictionOutcomeRecord(
            outcome_id=outcome.outcome_id,
            candidate_id=outcome.candidate_id,
            instrument_id=outcome.instrument_id,
            symbol=outcome.symbol,
            prediction_type=outcome.prediction_type.value,
            horizon=outcome.horizon.value,
            evaluation_window_start=outcome.evaluation_window_start,
            evaluation_window_end=outcome.evaluation_window_end,
            status=outcome.status.value,
            observed_result=outcome.observed_result.value if outcome.observed_result else None,
            observed_at=outcome.observed_at,
            result_summary=outcome.result_summary,
            result_value=outcome.result_value,
            baseline_value=outcome.baseline_value,
            limitations=outcome.limitations,
            metadata={
                "target_id": target.target_id,
                "outcome_artifact_id": outcome_artifact.artifact_id,
                **dict(outcome.metadata),
            },
        )
    )
    for reference in outcome.outcome_evidence:
        if store.get_evidence(reference.evidence_id) is None:
            continue
        store.link_outcome_evidence(
            OutcomeEvidenceLinkRecord(
                outcome_id=outcome.outcome_id,
                evidence_id=reference.evidence_id,
                relationship="observes_outcome",
                metadata={"target_id": target.target_id},
                created_at=created_at,
            )
        )
    for artifact_id in market_artifact_ids:
        if store.get_artifact(artifact_id) is None:
            continue
        store.link_outcome_artifact(
            OutcomeArtifactLinkRecord(
                outcome_id=outcome.outcome_id,
                artifact_id=artifact_id,
                relationship="outcome_source",
                metadata={"target_id": target.target_id},
                created_at=created_at,
            )
        )
    store.link_outcome_artifact(
        OutcomeArtifactLinkRecord(
            outcome_id=outcome.outcome_id,
            artifact_id=outcome_artifact.artifact_id,
            relationship="outcome_payload",
            metadata={"target_id": target.target_id},
            created_at=created_at,
        )
    )


def _persist_outcome_evaluation(
    *,
    store: SQLiteStore,
    target: PredictionEvaluationTarget,
    outcome_evaluation: PredictionOutcomeEvaluation,
    review_artifact: AuditArtifact,
    created_at: datetime,
) -> None:
    store.upsert_prediction_outcome_evaluation(
        PredictionOutcomeEvaluationRecord(
            outcome_evaluation_id=outcome_evaluation.outcome_evaluation_id,
            run_id=target.run_id,
            outcome_id=outcome_evaluation.outcome_id,
            candidate_id=outcome_evaluation.candidate_id,
            instrument_id=outcome_evaluation.instrument_id,
            symbol=outcome_evaluation.symbol,
            evaluated_at=outcome_evaluation.evaluated_at,
            status=outcome_evaluation.status.value,
            quality_score=outcome_evaluation.quality_score,
            baseline_comparison=(
                {}
                if outcome_evaluation.baseline_comparison is None
                else cast(
                    JsonObject,
                    outcome_evaluation.baseline_comparison.model_dump(mode="json"),
                )
            ),
            artifact_id=review_artifact.artifact_id,
            limitations=outcome_evaluation.limitations,
            metadata={
                "target_id": target.target_id,
                "outcome_id": outcome_evaluation.outcome_id,
            },
        )
    )
    for reference in outcome_evaluation.evidence:
        if store.get_evidence(reference.evidence_id) is None:
            continue
        store.link_outcome_evaluation_evidence(
            OutcomeEvaluationEvidenceLinkRecord(
                outcome_evaluation_id=outcome_evaluation.outcome_evaluation_id,
                evidence_id=reference.evidence_id,
                relationship="supports_outcome_review",
                metadata={"target_id": target.target_id},
                created_at=created_at,
            )
        )
    for artifact_id in outcome_evaluation.artifact_ids:
        if store.get_artifact(artifact_id) is None:
            continue
        store.link_outcome_evaluation_artifact(
            OutcomeEvaluationArtifactLinkRecord(
                outcome_evaluation_id=outcome_evaluation.outcome_evaluation_id,
                artifact_id=artifact_id,
                relationship="supports_outcome_review",
                metadata={"target_id": target.target_id},
                created_at=created_at,
            )
        )
    store.link_outcome_evaluation_artifact(
        OutcomeEvaluationArtifactLinkRecord(
            outcome_evaluation_id=outcome_evaluation.outcome_evaluation_id,
            artifact_id=review_artifact.artifact_id,
            relationship="outcome_evaluation_payload",
            metadata={"target_id": target.target_id},
            created_at=created_at,
        )
    )


def _eligible_evidence_ids(
    *,
    store: SQLiteStore,
    candidate: PredictionCandidateRecord,
    cutoff: datetime,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    included: list[str] = []
    source_artifacts: list[str] = []
    excluded: list[str] = []
    excluded_artifacts: list[str] = []
    for evidence_id in dict.fromkeys((*candidate.evidence_for, *candidate.evidence_against)):
        evidence = store.get_evidence(evidence_id)
        if evidence is None:
            excluded.append(f"{evidence_id} (missing)")
            continue
        retrieved_at = _aware_utc(evidence.retrieved_at, "evidence.retrieved_at")
        published_at = (
            None
            if evidence.published_at is None
            else _aware_utc(evidence.published_at, "evidence.published_at")
        )
        if retrieved_at > cutoff or (published_at is not None and published_at > cutoff):
            excluded.append(f"{evidence_id} (after cutoff)")
            continue
        included.append(evidence_id)
        if evidence.artifact_id is not None:
            artifact = store.get_artifact(evidence.artifact_id)
            if artifact is None:
                excluded_artifacts.append(f"{evidence.artifact_id} (missing)")
            elif (
                artifact.created_at is not None
                and _aware_utc(
                    artifact.created_at,
                    "artifact.created_at",
                )
                > cutoff
            ):
                excluded_artifacts.append(f"{evidence.artifact_id} (after cutoff)")
            else:
                source_artifacts.append(evidence.artifact_id)
    limitations = []
    if excluded:
        limitations.append(f"Excluded evidence unavailable before cutoff: {', '.join(excluded)}")
    if excluded_artifacts:
        limitations.append(
            "Excluded source artifacts unavailable before cutoff: " + ", ".join(excluded_artifacts)
        )
    return tuple(included), tuple(dict.fromkeys(source_artifacts)), tuple(limitations)


def _eligible_artifacts(
    *,
    store: SQLiteStore,
    candidate: PredictionCandidateRecord,
    cutoff: datetime,
) -> tuple[tuple[SignalArtifactReference, ...], tuple[str, ...], tuple[str, ...]]:
    signal_ids = set(candidate.signal_artifacts)
    signals: list[SignalArtifactReference] = []
    report_artifacts: list[str] = []
    excluded: list[str] = []
    for link in store.list_candidate_artifact_links(candidate.candidate_id):
        artifact = store.get_artifact(link.artifact_id)
        if artifact is None:
            excluded.append(f"{link.artifact_id} (missing)")
            continue
        if (
            artifact.created_at is not None
            and _aware_utc(artifact.created_at, "artifact.created_at") > cutoff
        ):
            excluded.append(f"{link.artifact_id} (after cutoff)")
            continue
        if link.relationship in {"prediction_input", "prediction_evaluation"}:
            report_artifacts.append(link.artifact_id)
        if link.artifact_id in signal_ids or link.relationship == "signal":
            reference = _signal_artifact_reference(artifact, cutoff=cutoff)
            if reference is not None:
                signals.append(reference)
    for artifact_id in signal_ids:
        if artifact_id in {reference.artifact_id for reference in signals}:
            continue
        artifact = store.get_artifact(artifact_id)
        if artifact is None:
            excluded.append(f"{artifact_id} (missing)")
            continue
        if (
            artifact.created_at is not None
            and _aware_utc(artifact.created_at, "artifact.created_at") > cutoff
        ):
            excluded.append(f"{artifact_id} (after cutoff)")
            continue
        reference = _signal_artifact_reference(artifact, cutoff=cutoff)
        if reference is not None:
            signals.append(reference)
    limitations = (
        (f"Excluded artifacts unavailable before cutoff: {', '.join(excluded)}",)
        if excluded
        else ()
    )
    return _dedupe_signal_artifacts(signals), tuple(dict.fromkeys(report_artifacts)), limitations


def _signal_artifact_reference(
    artifact: ArtifactRecord,
    *,
    cutoff: datetime,
) -> SignalArtifactReference | None:
    if artifact.artifact_type not in _SIGNAL_ARTIFACT_TYPES:
        return None
    family = _signal_family(artifact)
    if family is None:
        return None
    created_at = artifact.created_at
    return SignalArtifactReference(
        artifact_id=artifact.artifact_id,
        family=family,
        artifact_type=artifact.artifact_type,
        schema_version=artifact.schema_version,
        tool_run_id=artifact.tool_run_id,
        produced_by=artifact.produced_by,
        created_at=created_at,
        as_of=_artifact_as_of(artifact, cutoff),
        sha256=artifact.sha256,
        metadata={
            "record_count": artifact.record_count,
            "artifact_path": artifact.path.as_posix(),
        },
    )


def _dedupe_signal_artifacts(
    references: list[SignalArtifactReference],
) -> tuple[SignalArtifactReference, ...]:
    by_id: dict[str, SignalArtifactReference] = {}
    for reference in references:
        by_id.setdefault(reference.artifact_id, reference)
    return tuple(by_id.values())


def _dedupe_evidence_references(
    references: tuple[EvidenceReference, ...],
) -> tuple[EvidenceReference, ...]:
    by_id: dict[str, EvidenceReference] = {}
    for reference in references:
        by_id.setdefault(reference.evidence_id, reference)
    return tuple(by_id.values())


def _artifact_as_of(artifact: ArtifactRecord, cutoff: datetime) -> datetime | None:
    raw = artifact.metadata.get("as_of") or artifact.metadata.get("latest_usable_bar")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return artifact.created_at
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=UTC)
        return min(parsed.astimezone(UTC), cutoff)
    return artifact.created_at


def _signal_family(artifact: ArtifactRecord) -> SignalArtifactFamily | None:
    raw_family = artifact.metadata.get("signal_family")
    if isinstance(raw_family, str):
        try:
            return SignalArtifactFamily(raw_family)
        except ValueError:
            return None
    if artifact.artifact_type in {"market_data", "technical_package"}:
        return SignalArtifactFamily.TECHNICALS
    if artifact.artifact_type == "ml_forecast":
        return SignalArtifactFamily.TIMESFM
    if artifact.artifact_type == "normalized_evidence":
        return SignalArtifactFamily.NEWS
    if artifact.artifact_type == "analysis_context":
        return SignalArtifactFamily.FUNDAMENTALS
    return None


def _baseline_comparison_from_candidate(
    candidate: PredictionCandidateRecord,
) -> BaselineComparison | None:
    if not candidate.baseline:
        return None
    baseline_score = _score_from_json(candidate.baseline, ("baseline_score", "score"), 0.5)
    candidate_score = (
        candidate.confidence
        if candidate.confidence is not None
        else _score_from_json(candidate.baseline, ("candidate_score",), 0.5)
    )
    candidate_score = _bounded_score(candidate_score)
    delta = round(candidate_score - baseline_score, 6)
    return BaselineComparison(
        baseline_id=_json_text(candidate.baseline, ("baseline_id", "id", "comparison"))
        or "stored_baseline",
        baseline_summary=_json_text(candidate.baseline, ("summary", "description", "comparison"))
        or "Stored baseline context",
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        score_delta=delta,
        verdict=_baseline_verdict(candidate.baseline, delta),
    )


def _candidate_snapshot(
    *,
    candidate: PredictionCandidateRecord,
    symbol: str,
    evidence_ids: tuple[str, ...],
    signal_artifacts: tuple[SignalArtifactReference, ...],
    source_artifact_ids: tuple[str, ...],
    report_artifact_ids: tuple[str, ...],
) -> JsonObject:
    return {
        "candidate_id": candidate.candidate_id,
        "run_id": candidate.run_id,
        "instrument_id": candidate.instrument_id,
        "symbol": symbol,
        "prediction_horizon": candidate.prediction_horizon,
        "prediction_type": candidate.prediction_type,
        "scenario": candidate.scenario,
        "direction": candidate.direction,
        "confidence": candidate.confidence,
        "status": candidate.status,
        "original_evidence_for": list(candidate.evidence_for),
        "original_evidence_against": list(candidate.evidence_against),
        "included_evidence_ids": list(evidence_ids),
        "signal_artifact_ids": [reference.artifact_id for reference in signal_artifacts],
        "source_artifact_ids": list(source_artifact_ids),
        "report_artifact_ids": list(report_artifact_ids),
        "baseline": _json_ready(candidate.baseline),
        "uncertainty": candidate.uncertainty,
        "metadata": _json_ready(candidate.metadata),
    }


def _outcome_evaluation_status(
    *,
    target: PredictionEvaluationTarget,
    outcome: PredictionOutcome,
) -> PredictionOutcomeEvaluationStatus:
    if outcome.status == PredictionOutcomeStatus.PENDING:
        return PredictionOutcomeEvaluationStatus.PENDING
    if outcome.status == PredictionOutcomeStatus.STALE:
        return PredictionOutcomeEvaluationStatus.STALE
    if outcome.status == PredictionOutcomeStatus.UNAVAILABLE:
        return PredictionOutcomeEvaluationStatus.NOT_EVALUABLE
    if outcome.observed_result == PredictionOutcomeResult.SUPPORTED:
        return PredictionOutcomeEvaluationStatus.CONFIRMED
    if outcome.observed_result in {
        PredictionOutcomeResult.NOT_SUPPORTED,
        PredictionOutcomeResult.CONTRADICTED,
    }:
        return PredictionOutcomeEvaluationStatus.MISSED
    if outcome.observed_result == PredictionOutcomeResult.NEUTRAL:
        if (
            target.direction == Direction.NEUTRAL
            or target.prediction_type == PredictionType.NEUTRAL
        ):
            return PredictionOutcomeEvaluationStatus.CONFIRMED
        return PredictionOutcomeEvaluationStatus.MIXED
    if outcome.observed_result == PredictionOutcomeResult.MIXED:
        return PredictionOutcomeEvaluationStatus.MIXED
    return PredictionOutcomeEvaluationStatus.INCONCLUSIVE


def _quality_score(status: PredictionOutcomeEvaluationStatus) -> float | None:
    if status == PredictionOutcomeEvaluationStatus.CONFIRMED:
        return 1.0
    if status == PredictionOutcomeEvaluationStatus.MISSED:
        return 0.0
    if status in {
        PredictionOutcomeEvaluationStatus.MIXED,
        PredictionOutcomeEvaluationStatus.INCONCLUSIVE,
    }:
        return 0.5
    return None


def _prediction_type(value: str) -> PredictionType:
    try:
        return PredictionType(value)
    except ValueError:
        return PredictionType.UNCERTAIN


def _time_horizon(value: str) -> TimeHorizon:
    try:
        return TimeHorizon(value)
    except ValueError:
        return TimeHorizon.UNKNOWN


def _direction(value: str | None) -> Direction:
    if value is None:
        return Direction.UNKNOWN
    try:
        return Direction(value)
    except ValueError:
        return Direction.UNKNOWN


def _prediction_outcome_status(
    value: PredictionOutcomeStatus | str | None,
    observed_result: PredictionOutcomeResult | None,
) -> PredictionOutcomeStatus:
    if value is None:
        return (
            PredictionOutcomeStatus.OBSERVED
            if observed_result is not None
            else PredictionOutcomeStatus.PENDING
        )
    if isinstance(value, PredictionOutcomeStatus):
        return value
    return PredictionOutcomeStatus(value)


def _prediction_outcome_result(
    value: PredictionOutcomeResult | str | None,
) -> PredictionOutcomeResult | None:
    if value is None:
        return None
    if isinstance(value, PredictionOutcomeResult):
        return value
    return PredictionOutcomeResult(value)


def _json_text(data: JsonObject, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _score_from_json(data: JsonObject, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (float, int)):
            return _bounded_score(float(value))
    return default


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def _baseline_verdict(data: JsonObject, delta: float) -> str:
    raw = data.get("verdict")
    if raw in {
        "above_baseline",
        "near_baseline",
        "below_baseline",
        "baseline_unavailable",
    }:
        return cast(str, raw)
    if delta > 0.05:
        return "above_baseline"
    if delta < -0.05:
        return "below_baseline"
    return "near_baseline"


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _target_id(
    *,
    run_id: str,
    candidate_id: str,
    cutoff: datetime,
    evaluation_window_start: datetime,
    evaluation_window_end: datetime,
) -> str:
    digest = _digest(
        "|".join(
            (
                run_id,
                candidate_id,
                cutoff.isoformat(),
                evaluation_window_start.isoformat(),
                evaluation_window_end.isoformat(),
            )
        )
    )
    return f"target-{_slug(candidate_id)}-{digest[:8]}"


def _outcome_id(
    *,
    target: PredictionEvaluationTarget,
    observed_result: PredictionOutcomeResult | None,
    observed_at: datetime | None,
    status: PredictionOutcomeStatus,
) -> str:
    digest = _digest(
        "|".join(
            (
                target.target_id,
                status.value,
                observed_result.value if observed_result else "unresolved",
                observed_at.isoformat() if observed_at else "unobserved",
            )
        )
    )
    return f"outcome-{_slug(target.candidate_id)}-{digest[:8]}"


def _outcome_evaluation_id(
    *,
    target: PredictionEvaluationTarget,
    outcome: PredictionOutcome,
    evaluated_at: datetime,
) -> str:
    digest = _digest("|".join((target.target_id, outcome.outcome_id, evaluated_at.isoformat())))
    return f"outcome-evaluation-{_slug(target.candidate_id)}-{digest[:8]}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "prediction"


def _json_ready(value: object) -> JsonValue:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "PHASE6_OUTCOME_TOOL_NAME",
    "PHASE6_OUTCOME_TOOL_VERSION",
    "PointInTimeOutcomeEvaluationArtifacts",
    "build_prediction_evaluation_target",
    "build_prediction_outcome",
    "evaluate_prediction_outcome",
    "write_point_in_time_outcome_evaluation_artifacts",
]
