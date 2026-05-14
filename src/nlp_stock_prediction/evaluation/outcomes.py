"""Point-in-time prediction outcome evaluation."""

from __future__ import annotations

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
from nlp_stock_prediction.evaluation.common import aware_utc, digest, slug
from nlp_stock_prediction.evaluation.freshness import (
    artifact_reference_as_of,
    freshness_limitations,
    phase7_freshness_metadata,
    review_candidate_artifact_freshness,
    review_candidate_evidence_aging,
)
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase4_common import safe_phase4_tool_execution
from nlp_stock_prediction.storage.outcome_repository import persist_prediction_outcome_records
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    PredictionCandidateRecord,
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

    cutoff = aware_utc(point_in_time_cutoff, "point_in_time_cutoff")
    window_start = aware_utc(evaluation_window_start, "evaluation_window_start")
    window_end = aware_utc(evaluation_window_end, "evaluation_window_end")
    prediction_created_at = aware_utc(run.started_at, "run.started_at")
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
    evidence_aging_records = review_candidate_evidence_aging(
        store=store,
        candidate=candidate,
        reviewed_at=cutoff,
    )
    artifact_freshness_reviews = review_candidate_artifact_freshness(
        store=store,
        candidate=candidate,
        reviewed_at=cutoff,
    )
    freshness_metadata = phase7_freshness_metadata(
        reviewed_at=cutoff,
        evidence_aging_records=evidence_aging_records,
        artifact_freshness_reviews=artifact_freshness_reviews,
    )
    baseline = _baseline_comparison_from_candidate(candidate)
    limitations = [
        *evidence_limitations,
        *artifact_limitations,
        *freshness_limitations(
            evidence_aging_records=evidence_aging_records,
            artifact_freshness_reviews=artifact_freshness_reviews,
        ),
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
            phase7_freshness=freshness_metadata,
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
            "phase7_freshness": freshness_metadata,
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
    resolved_observed_at = None if observed_at is None else aware_utc(observed_at, "observed_at")
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

    resolved_evaluated_at = aware_utc(evaluated_at, "evaluated_at")
    if outcome.observed_at is not None and resolved_evaluated_at < outcome.observed_at:
        raise ValueError("outcome evaluation must not occur before observed_at")
    status = _outcome_evaluation_status(target=target, outcome=outcome)
    quality_score = _quality_score(status)
    resolved_limitations = tuple(
        dict.fromkeys((*limitations, *outcome.limitations, *target.limitations))
    )
    if (
        status
        in {
            PredictionOutcomeEvaluationStatus.PENDING,
            PredictionOutcomeEvaluationStatus.STALE,
            PredictionOutcomeEvaluationStatus.NOT_EVALUABLE,
        }
        and not resolved_limitations
    ):
        resolved_limitations = ("Outcome evaluation is not resolved.",)
    if target.baseline_comparison is None and not resolved_limitations:
        resolved_limitations = ("No stored baseline comparison was available at the cutoff.",)

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
    created = aware_utc(created_at or datetime.now(UTC), "created_at")
    evaluated = aware_utc(evaluated_at or created, "evaluated_at")
    observed_artifact_ids = _validated_market_artifact_ids(
        store=store,
        target=target,
        market_artifact_ids=tuple(dict.fromkeys(market_artifact_ids)),
        evaluated_at=evaluated,
    )
    validated_outcome_evidence = _validated_outcome_evidence(
        store=store,
        target=target,
        outcome_evidence=_dedupe_evidence_references(outcome_evidence),
        evaluated_at=evaluated,
    )
    outcome = build_prediction_outcome(
        target=target,
        status=status,
        observed_result=observed_result,
        observed_at=observed_at,
        result_summary=result_summary,
        result_value=result_value,
        baseline_value=baseline_value,
        outcome_evidence=validated_outcome_evidence,
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
    artifact_digest = digest(
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
        tool_run_id
        or "tool-phase6-outcome-"
        f"{slug(target.candidate_id, allow_file_safe_punctuation=True)}-{artifact_digest[:12]}"
    )
    inputs: JsonObject = {
        "target_id": target.target_id,
        "candidate_id": target.candidate_id,
        "outcome_status": outcome.status.value,
        "observed_result": outcome.observed_result.value if outcome.observed_result else None,
        "market_artifact_ids": list(observed_artifact_ids),
        "outcome_evidence_ids": [reference.evidence_id for reference in validated_outcome_evidence],
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
            "artifact-prediction-outcome-"
            f"{slug(target.candidate_id, allow_file_safe_punctuation=True)}-{artifact_digest[:8]}"
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
            or (
                "prediction-outcomes/"
                f"{slug(target.candidate_id, allow_file_safe_punctuation=True)}-"
                f"{artifact_digest[:8]}.json"
            ),
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
            "artifact-prediction-outcome-evaluation-"
            f"{slug(target.candidate_id, allow_file_safe_punctuation=True)}-{artifact_digest[:8]}"
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
            or (
                "prediction-outcome-evaluations/"
                f"{slug(target.candidate_id, allow_file_safe_punctuation=True)}-"
                f"{artifact_digest[:8]}.json"
            ),
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
        persist_prediction_outcome_records(
            store=store,
            target=target,
            outcome=outcome,
            outcome_evaluation=outcome_evaluation_with_artifact,
            outcome_artifact=outcome_artifact,
            review_artifact=review_artifact,
            market_artifact_ids=observed_artifact_ids,
            outcome_created_at=created,
            review_created_at=evaluated,
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


def _validated_outcome_evidence(
    *,
    store: SQLiteStore,
    target: PredictionEvaluationTarget,
    outcome_evidence: tuple[EvidenceReference, ...],
    evaluated_at: datetime,
) -> tuple[EvidenceReference, ...]:
    for reference in outcome_evidence:
        record = store.get_evidence(reference.evidence_id)
        if record is None:
            raise ValueError(f"outcome evidence does not exist: {reference.evidence_id}")
        retrieved_at = aware_utc(record.retrieved_at, "evidence.retrieved_at")
        published_at = (
            None
            if record.published_at is None
            else aware_utc(record.published_at, "evidence.published_at")
        )
        if retrieved_at > evaluated_at or (
            published_at is not None and published_at > evaluated_at
        ):
            raise ValueError(
                f"outcome evidence must be available by evaluated_at: {reference.evidence_id}"
            )
        if retrieved_at < target.evaluation_window_end or (
            published_at is not None and published_at < target.evaluation_window_end
        ):
            raise ValueError(
                f"outcome evidence must observe the evaluation window: {reference.evidence_id}"
            )
        if record.instruments and target.instrument_id not in record.instruments:
            raise ValueError(
                f"outcome evidence instrument does not match target: {reference.evidence_id}"
            )
    return outcome_evidence


def _validated_market_artifact_ids(
    *,
    store: SQLiteStore,
    target: PredictionEvaluationTarget,
    market_artifact_ids: tuple[str, ...],
    evaluated_at: datetime,
) -> tuple[str, ...]:
    for artifact_id in market_artifact_ids:
        artifact = store.get_artifact(artifact_id)
        if artifact is None:
            raise ValueError(f"outcome market artifact does not exist: {artifact_id}")
        if artifact.artifact_type != "market_data":
            raise ValueError(f"outcome market artifact must be market_data: {artifact_id}")
        if (
            artifact.created_at is not None
            and aware_utc(
                artifact.created_at,
                "artifact.created_at",
            )
            > evaluated_at
        ):
            raise ValueError(
                f"outcome market artifact must be available by evaluated_at: {artifact_id}"
            )
        if artifact.created_at is None:
            raise ValueError(f"outcome market artifact requires created_at: {artifact_id}")
        if aware_utc(artifact.created_at, "artifact.created_at") < target.evaluation_window_end:
            raise ValueError(
                f"outcome market artifact must observe the evaluation window: {artifact_id}"
            )
        instrument_id = artifact.metadata.get("instrument_id")
        instrument_ids = artifact.metadata.get("instrument_ids")
        if instrument_id is not None and instrument_id != target.instrument_id:
            raise ValueError(
                f"outcome market artifact instrument does not match target: {artifact_id}"
            )
        if isinstance(instrument_ids, list) and target.instrument_id not in instrument_ids:
            raise ValueError(
                f"outcome market artifact instruments do not include target: {artifact_id}"
            )
    return market_artifact_ids


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
        retrieved_at = aware_utc(evidence.retrieved_at, "evidence.retrieved_at")
        published_at = (
            None
            if evidence.published_at is None
            else aware_utc(evidence.published_at, "evidence.published_at")
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
                and aware_utc(
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
    signal_ids = tuple(dict.fromkeys(candidate.signal_artifacts))
    signal_id_set = set(signal_ids)
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
            and aware_utc(artifact.created_at, "artifact.created_at") > cutoff
        ):
            excluded.append(f"{link.artifact_id} (after cutoff)")
            continue
        if link.relationship in {"prediction_input", "prediction_evaluation"}:
            report_artifacts.append(link.artifact_id)
        if link.artifact_id in signal_id_set or link.relationship == "signal":
            artifact_as_of, reason = _artifact_reference_as_of(artifact, cutoff)
            if reason is not None:
                excluded.append(f"{link.artifact_id} ({reason})")
                continue
            reference = _signal_artifact_reference(artifact, as_of=artifact_as_of)
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
            and aware_utc(artifact.created_at, "artifact.created_at") > cutoff
        ):
            excluded.append(f"{artifact_id} (after cutoff)")
            continue
        artifact_as_of, reason = _artifact_reference_as_of(artifact, cutoff)
        if reason is not None:
            excluded.append(f"{artifact_id} ({reason})")
            continue
        reference = _signal_artifact_reference(artifact, as_of=artifact_as_of)
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
    as_of: datetime | None,
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
        as_of=as_of,
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


def _artifact_reference_as_of(
    artifact: ArtifactRecord,
    cutoff: datetime,
) -> tuple[datetime | None, str | None]:
    artifact_as_of, reason = artifact_reference_as_of(artifact)
    if reason is not None:
        return None, reason
    if artifact_as_of is not None and artifact_as_of > cutoff:
        return None, "as_of after cutoff"
    return artifact_as_of, None


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
    phase7_freshness: JsonObject,
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
        "phase7_freshness": phase7_freshness,
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
    if outcome.observed_result == PredictionOutcomeResult.INSUFFICIENT_DATA:
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
    except ValueError as exc:
        allowed = ", ".join(item.value for item in PredictionType)
        raise ValueError(f"prediction_type must be one of: {allowed}") from exc


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


def _target_id(
    *,
    run_id: str,
    candidate_id: str,
    cutoff: datetime,
    evaluation_window_start: datetime,
    evaluation_window_end: datetime,
) -> str:
    target_digest = digest(
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
    return (
        f"target-{slug(candidate_id, fallback='prediction', allow_file_safe_punctuation=True)}-"
        f"{target_digest[:8]}"
    )


def _outcome_id(
    *,
    target: PredictionEvaluationTarget,
    observed_result: PredictionOutcomeResult | None,
    observed_at: datetime | None,
    status: PredictionOutcomeStatus,
) -> str:
    outcome_digest = digest(
        "|".join(
            (
                target.target_id,
                status.value,
                observed_result.value if observed_result else "unresolved",
                observed_at.isoformat() if observed_at else "unobserved",
            )
        )
    )
    return (
        "outcome-"
        f"{slug(target.candidate_id, fallback='prediction', allow_file_safe_punctuation=True)}-"
        f"{outcome_digest[:8]}"
    )


def _outcome_evaluation_id(
    *,
    target: PredictionEvaluationTarget,
    outcome: PredictionOutcome,
    evaluated_at: datetime,
) -> str:
    evaluation_digest = digest(
        "|".join((target.target_id, outcome.outcome_id, evaluated_at.isoformat()))
    )
    return (
        f"outcome-evaluation-"
        f"{slug(target.candidate_id, fallback='prediction', allow_file_safe_punctuation=True)}-"
        f"{evaluation_digest[:8]}"
    )


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
