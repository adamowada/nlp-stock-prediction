"""Walk-forward evaluation for Phase 6 calibration cohorts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
)
from nlp_stock_prediction.contracts.enums import (
    PredictionType,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.evaluation import PredictionOutcomeEvaluation
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.evaluation.common import (
    aware_utc,
    digest,
    filter_outcome_evaluations,
    is_resolved_outcome_evaluation,
    outcome_status_counts,
    slug,
    source_artifact_ids_from_outcome_evaluations,
    source_outcome_evaluation_ids,
    validate_unique,
    write_calibration_slices,
)
from nlp_stock_prediction.evaluation.outcomes import PointInTimeOutcomeEvaluationArtifacts
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase4_common import safe_phase4_tool_execution
from nlp_stock_prediction.storage.records import (
    CalibrationRunRecord,
    CalibrationSliceRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE6_WALK_FORWARD_TOOL_NAME = "phase6_walk_forward_evaluation"
PHASE6_WALK_FORWARD_TOOL_VERSION = "phase6.walk-forward-evaluation.v1"


class WalkForwardFold(ContractModel):
    """One chronological train/test fold for prediction-quality evaluation."""

    schema_version: NonEmptyStr = "walk-forward-fold.v1"
    fold_id: NonEmptyStr
    cohort_id: NonEmptyStr
    created_at: AwareDatetime
    train_window_start: AwareDatetime
    train_window_end: AwareDatetime
    test_window_start: AwareDatetime
    test_window_end: AwareDatetime
    train_sample_count: int = Field(ge=0)
    test_sample_count: int = Field(ge=0)
    train_resolved_count: int = Field(ge=0)
    test_resolved_count: int = Field(ge=0)
    test_pending_count: int = Field(default=0, ge=0)
    test_stale_count: int = Field(default=0, ge=0)
    test_unavailable_count: int = Field(default=0, ge=0)
    test_not_evaluable_count: int = Field(default=0, ge=0)
    train_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    test_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    generalization_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    source_train_outcome_evaluation_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    source_test_outcome_evaluation_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fold(self) -> WalkForwardFold:
        if self.train_window_end < self.train_window_start:
            raise ValueError("walk-forward train window end must not precede start")
        if self.test_window_end < self.test_window_start:
            raise ValueError("walk-forward test window end must not precede start")
        if self.train_window_end > self.test_window_start:
            raise ValueError("walk-forward train window must not overlap test window")
        if self.train_resolved_count > self.train_sample_count:
            raise ValueError("walk-forward train_resolved_count cannot exceed train_sample_count")
        if self.test_resolved_count > self.test_sample_count:
            raise ValueError("walk-forward test_resolved_count cannot exceed test_sample_count")
        test_status_total = (
            self.test_resolved_count
            + self.test_pending_count
            + self.test_stale_count
            + self.test_unavailable_count
            + self.test_not_evaluable_count
        )
        if test_status_total != self.test_sample_count:
            raise ValueError("walk-forward test_sample_count must equal test status counts")
        if len(self.source_train_outcome_evaluation_ids) != self.train_sample_count:
            raise ValueError("walk-forward train source IDs must match train_sample_count")
        if len(self.source_test_outcome_evaluation_ids) != self.test_sample_count:
            raise ValueError("walk-forward test source IDs must match test_sample_count")
        if self.train_resolved_count == 0 and self.train_quality_score is not None:
            raise ValueError("unresolved train folds must not report train_quality_score")
        if self.test_resolved_count == 0 and self.test_quality_score is not None:
            raise ValueError("unresolved test folds must not report test_quality_score")
        if self.generalization_delta is not None and (
            self.train_quality_score is None or self.test_quality_score is None
        ):
            raise ValueError("walk-forward generalization_delta requires train and test metrics")
        if (
            self.train_quality_score is None or self.test_quality_score is None
        ) and not self.limitations:
            raise ValueError("metric-free walk-forward folds require limitations")
        return self


class WalkForwardEvaluationArtifactPayload(ContractModel):
    """Stable JSON payload for persisted walk-forward evaluation folds."""

    schema_version: NonEmptyStr = "walk-forward-evaluation-artifact.v1"
    run_id: NonEmptyStr
    calibration_id: NonEmptyStr
    cohort_id: NonEmptyStr
    created_at: AwareDatetime
    point_in_time_cutoff: AwareDatetime
    minimum_train_size: int = Field(ge=1)
    test_size: int = Field(ge=1)
    step_size: int = Field(ge=1)
    prediction_type: PredictionType | None = None
    horizon: TimeHorizon | None = None
    sample_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    pending_count: int = Field(default=0, ge=0)
    stale_count: int = Field(default=0, ge=0)
    unavailable_count: int = Field(default=0, ge=0)
    not_evaluable_count: int = Field(default=0, ge=0)
    folds: tuple[WalkForwardFold, ...] = Field(default_factory=tuple)
    source_outcome_evaluation_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    source_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    excluded_outcome_evaluation_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> WalkForwardEvaluationArtifactPayload:
        status_total = (
            self.resolved_count
            + self.pending_count
            + self.stale_count
            + self.unavailable_count
            + self.not_evaluable_count
        )
        if status_total != self.sample_count:
            raise ValueError("walk-forward sample_count must equal status counts")
        if len(self.source_outcome_evaluation_ids) != self.sample_count:
            raise ValueError("walk-forward source IDs must match sample_count")
        validate_unique("walk-forward fold ids", tuple(fold.fold_id for fold in self.folds))
        validate_unique(
            "walk-forward source outcome evaluation ids",
            self.source_outcome_evaluation_ids,
        )
        validate_unique("walk-forward source artifact ids", self.source_artifact_ids)
        validate_unique(
            "walk-forward excluded outcome evaluation ids",
            self.excluded_outcome_evaluation_ids,
        )
        if not self.folds and not self.limitations:
            raise ValueError("walk-forward payloads without folds require limitations")
        return self


@dataclass(frozen=True)
class WalkForwardEvaluationArtifacts:
    """Artifacts and storage records produced by a walk-forward evaluation write."""

    calibration_id: str
    calibration_run: CalibrationRunRecord
    calibration_slices: tuple[CalibrationSliceRecord, ...]
    folds: tuple[WalkForwardFold, ...]
    artifact: AuditArtifact
    artifact_payload: WalkForwardEvaluationArtifactPayload
    tool_run_id: str


def walk_forward_evaluations_from_outcome_artifacts(
    outcome_artifacts: Iterable[PointInTimeOutcomeEvaluationArtifacts],
) -> tuple[PredictionOutcomeEvaluation, ...]:
    """Extract outcome evaluations from Stage 3 point-in-time outcome artifacts."""

    return tuple(item.outcome_evaluation for item in outcome_artifacts)


def compute_walk_forward_folds(
    *,
    cohort_id: str,
    created_at: datetime,
    outcome_evaluations: Sequence[PredictionOutcomeEvaluation],
    minimum_train_size: int,
    test_size: int = 1,
    step_size: int = 1,
    prediction_type: PredictionType | None = None,
    horizon: TimeHorizon | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Build chronological train/test folds from eligible outcome evaluations."""

    _validate_window_parameters(
        minimum_train_size=minimum_train_size,
        test_size=test_size,
        step_size=step_size,
    )
    created = aware_utc(created_at, "created_at")
    eligible = _sorted_outcome_evaluations(
        _filtered_outcome_evaluations(
            tuple(outcome_evaluations),
            prediction_type=prediction_type,
            horizon=horizon,
        )
    )
    folds: list[WalkForwardFold] = []
    fold_index = 1
    test_start = minimum_train_size
    while test_start < len(eligible):
        test_end = min(test_start + test_size, len(eligible))
        train = eligible[:test_start]
        test = eligible[test_start:test_end]
        if not test:
            break
        folds.append(
            _fold_from_partitions(
                cohort_id=cohort_id,
                created_at=created,
                fold_index=fold_index,
                train=train,
                test=test,
            )
        )
        fold_index += 1
        test_start += step_size
    return tuple(folds)


def write_walk_forward_evaluation_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    cohort_id: str,
    outcome_evaluations: Sequence[PredictionOutcomeEvaluation],
    point_in_time_cutoff: datetime,
    minimum_train_size: int,
    test_size: int = 1,
    step_size: int = 1,
    created_at: datetime | None = None,
    prediction_type: PredictionType | None = None,
    horizon: TimeHorizon | None = None,
    calibration_id: str | None = None,
    tool_run_id: str | None = None,
    artifact_filename: str | None = None,
    record_tool_run: bool = True,
) -> WalkForwardEvaluationArtifacts:
    """Persist walk-forward folds as an audit artifact and calibration slices."""

    _validate_window_parameters(
        minimum_train_size=minimum_train_size,
        test_size=test_size,
        step_size=step_size,
    )
    created = aware_utc(created_at or datetime.now(UTC), "created_at")
    cutoff = aware_utc(point_in_time_cutoff, "point_in_time_cutoff")
    requested = tuple(outcome_evaluations)
    eligible, excluded_ids, filter_limitations = _eligible_outcome_evaluations(
        requested,
        point_in_time_cutoff=cutoff,
        prediction_type=prediction_type,
        horizon=horizon,
    )
    folds = compute_walk_forward_folds(
        cohort_id=cohort_id,
        created_at=created,
        outcome_evaluations=eligible,
        minimum_train_size=minimum_train_size,
        test_size=test_size,
        step_size=step_size,
    )
    source_outcome_ids = source_outcome_evaluation_ids(eligible)
    source_artifacts = source_artifact_ids_from_outcome_evaluations(eligible)
    status_counts = outcome_status_counts(eligible)
    limitations = _payload_limitations(
        folds=folds,
        eligible=eligible,
        excluded_ids=excluded_ids,
        filter_limitations=filter_limitations,
        minimum_train_size=minimum_train_size,
        test_size=test_size,
    )
    resolved_calibration_id = calibration_id or _calibration_id(
        run_id=run_id,
        cohort_id=cohort_id,
        point_in_time_cutoff=cutoff,
        minimum_train_size=minimum_train_size,
        test_size=test_size,
        step_size=step_size,
        outcome_evaluation_ids=source_outcome_ids,
    )
    artifact_digest = digest(
        "|".join(
            (
                run_id,
                cohort_id,
                resolved_calibration_id,
                ",".join(source_outcome_ids),
                str(minimum_train_size),
                str(test_size),
                str(step_size),
            )
        )
    )
    resolved_tool_run_id = tool_run_id or f"tool-walk-forward-evaluation-{artifact_digest[:12]}"
    payload = WalkForwardEvaluationArtifactPayload(
        run_id=run_id,
        calibration_id=resolved_calibration_id,
        cohort_id=cohort_id,
        created_at=created,
        point_in_time_cutoff=cutoff,
        minimum_train_size=minimum_train_size,
        test_size=test_size,
        step_size=step_size,
        prediction_type=prediction_type,
        horizon=horizon,
        sample_count=len(eligible),
        resolved_count=status_counts["resolved_count"],
        pending_count=status_counts["pending_count"],
        stale_count=status_counts["stale_count"],
        unavailable_count=status_counts["unavailable_count"],
        not_evaluable_count=status_counts["not_evaluable_count"],
        folds=folds,
        source_outcome_evaluation_ids=source_outcome_ids,
        source_artifact_ids=source_artifacts,
        excluded_outcome_evaluation_ids=excluded_ids,
        limitations=limitations,
        metadata={
            "fold_count": len(folds),
            "requested_outcome_evaluation_count": len(requested),
            "eligible_outcome_evaluation_count": len(eligible),
        },
    )
    tool_inputs: JsonObject = {
        "cohort_id": cohort_id,
        "calibration_id": resolved_calibration_id,
        "point_in_time_cutoff": cutoff.isoformat(),
        "minimum_train_size": minimum_train_size,
        "test_size": test_size,
        "step_size": step_size,
        "prediction_type": prediction_type.value if prediction_type else None,
        "horizon": horizon.value if horizon else None,
        "source_outcome_evaluation_ids": list(source_outcome_ids),
        "excluded_outcome_evaluation_ids": list(excluded_ids),
    }

    def write_records() -> WalkForwardEvaluationArtifacts:
        if record_tool_run:
            store.record_tool_run(
                ToolRunRecord(
                    tool_run_id=resolved_tool_run_id,
                    run_id=run_id,
                    tool_name=PHASE6_WALK_FORWARD_TOOL_NAME,
                    tool_version=PHASE6_WALK_FORWARD_TOOL_VERSION,
                    status="successful",
                    started_at=created,
                    completed_at=created,
                    inputs=tool_inputs,
                    warnings=limitations,
                )
            )
        artifact_id = f"artifact-walk-forward-evaluation-{slug(cohort_id)}-{artifact_digest[:12]}"
        artifact = ArtifactIndex.for_directory(
            store=store,
            repo_root=repo_root,
            base_dir=artifact_dir,
            created_at=created,
            produced_by=PHASE6_WALK_FORWARD_TOOL_NAME,
            tool_run_id=resolved_tool_run_id if record_tool_run or tool_run_id else None,
            schema_version=payload.schema_version,
        ).write_json(
            artifact_id=artifact_id,
            artifact_type="walk_forward_evaluation",
            filename=artifact_filename
            or f"calibration/walk-forward-evaluations/{slug(cohort_id)}.json",
            payload=cast(JsonObject, payload.model_dump(mode="json")),
            record_count=len(folds),
            metadata={
                "run_id": run_id,
                "calibration_id": resolved_calibration_id,
                "cohort_id": cohort_id,
                "fold_count": len(folds),
                "sample_count": len(eligible),
            },
        )
        calibration_run = CalibrationRunRecord(
            calibration_id=resolved_calibration_id,
            run_id=run_id,
            method_version=PHASE6_WALK_FORWARD_TOOL_VERSION,
            created_at=created,
            point_in_time_cutoff=cutoff,
            tool_run_id=resolved_tool_run_id if record_tool_run or tool_run_id else None,
            cohort_query={
                "cohort_id": cohort_id,
                "minimum_train_size": minimum_train_size,
                "test_size": test_size,
                "step_size": step_size,
                "prediction_type": prediction_type.value if prediction_type else None,
                "horizon": horizon.value if horizon else None,
            },
            source_outcome_evaluation_ids=source_outcome_ids,
            artifact_id=artifact.artifact_id,
            limitations=limitations,
            metadata={
                "excluded_outcome_evaluation_ids": list(excluded_ids),
                "source_artifact_ids": list(source_artifacts),
                "fold_count": len(folds),
            },
        )
        store.record_calibration_run(calibration_run)
        slices = write_calibration_slices(
            store,
            calibration_id=resolved_calibration_id,
            slices=(
                _calibration_slice_record(calibration_id=resolved_calibration_id, fold=fold)
                for fold in folds
            ),
        )
        return WalkForwardEvaluationArtifacts(
            calibration_id=resolved_calibration_id,
            calibration_run=calibration_run,
            calibration_slices=slices,
            folds=folds,
            artifact=artifact,
            artifact_payload=payload,
            tool_run_id=resolved_tool_run_id,
        )

    if not record_tool_run:
        return write_records()

    with safe_phase4_tool_execution(
        store=store,
        artifact_roots=(artifact_dir,),
        tool_run_id=resolved_tool_run_id,
        run_id=run_id,
        tool_name=PHASE6_WALK_FORWARD_TOOL_NAME,
        tool_version=PHASE6_WALK_FORWARD_TOOL_VERSION,
        started_at=created,
        inputs=tool_inputs,
    ):
        return write_records()


def _fold_from_partitions(
    *,
    cohort_id: str,
    created_at: datetime,
    fold_index: int,
    train: tuple[PredictionOutcomeEvaluation, ...],
    test: tuple[PredictionOutcomeEvaluation, ...],
) -> WalkForwardFold:
    train_status_counts = outcome_status_counts(train)
    test_status_counts = outcome_status_counts(test)
    train_score = _quality_average(train)
    test_score = _quality_average(test)
    delta = (
        round(test_score - train_score, 6)
        if train_score is not None and test_score is not None
        else None
    )
    limitations = _fold_limitations(
        train=train,
        test=test,
        train_score=train_score,
        test_score=test_score,
    )
    return WalkForwardFold(
        fold_id=f"walk-forward-{slug(cohort_id)}-fold-{fold_index:03d}",
        cohort_id=cohort_id,
        created_at=created_at,
        train_window_start=train[0].evaluated_at,
        train_window_end=train[-1].evaluated_at,
        test_window_start=test[0].evaluated_at,
        test_window_end=test[-1].evaluated_at,
        train_sample_count=len(train),
        test_sample_count=len(test),
        train_resolved_count=train_status_counts["resolved_count"],
        test_resolved_count=test_status_counts["resolved_count"],
        test_pending_count=test_status_counts["pending_count"],
        test_stale_count=test_status_counts["stale_count"],
        test_unavailable_count=test_status_counts["unavailable_count"],
        test_not_evaluable_count=test_status_counts["not_evaluable_count"],
        train_quality_score=train_score,
        test_quality_score=test_score,
        generalization_delta=delta,
        source_train_outcome_evaluation_ids=source_outcome_evaluation_ids(train),
        source_test_outcome_evaluation_ids=source_outcome_evaluation_ids(test),
        limitations=limitations,
        metadata={
            "train_status_counts": dict(train_status_counts),
            "test_status_counts": dict(test_status_counts),
        },
    )


def _calibration_slice_record(
    *,
    calibration_id: str,
    fold: WalkForwardFold,
) -> CalibrationSliceRecord:
    return CalibrationSliceRecord(
        slice_id=f"slice-{calibration_id}-{fold.fold_id}",
        calibration_id=calibration_id,
        cohort_label=f"walk_forward:{fold.fold_id}",
        sample_count=fold.test_sample_count,
        resolved_count=fold.test_resolved_count,
        pending_count=fold.test_pending_count,
        stale_count=fold.test_stale_count,
        unavailable_count=fold.test_unavailable_count,
        not_evaluable_count=fold.test_not_evaluable_count,
        metrics={
            "train_sample_count": fold.train_sample_count,
            "test_sample_count": fold.test_sample_count,
            "train_resolved_count": fold.train_resolved_count,
            "test_resolved_count": fold.test_resolved_count,
            "train_quality_score": fold.train_quality_score,
            "test_quality_score": fold.test_quality_score,
            "generalization_delta": fold.generalization_delta,
        },
        provenance=cast(
            JsonObject,
            {
                "fold_id": fold.fold_id,
                "train_window_start": fold.train_window_start.isoformat(),
                "train_window_end": fold.train_window_end.isoformat(),
                "test_window_start": fold.test_window_start.isoformat(),
                "test_window_end": fold.test_window_end.isoformat(),
                "source_train_outcome_evaluation_ids": list(
                    fold.source_train_outcome_evaluation_ids
                ),
                "source_test_outcome_evaluation_ids": list(fold.source_test_outcome_evaluation_ids),
            },
        ),
        metadata=cast(JsonObject, {"limitations": list(fold.limitations)}),
    )


def _eligible_outcome_evaluations(
    outcome_evaluations: tuple[PredictionOutcomeEvaluation, ...],
    *,
    point_in_time_cutoff: datetime,
    prediction_type: PredictionType | None,
    horizon: TimeHorizon | None,
) -> tuple[tuple[PredictionOutcomeEvaluation, ...], tuple[str, ...], tuple[str, ...]]:
    cohort = filter_outcome_evaluations(
        outcome_evaluations,
        as_of=point_in_time_cutoff,
        prediction_type=prediction_type,
        horizon=horizon,
        cutoff_reason="Excluded outcome evaluation after the point-in-time cutoff",
    )
    return (
        _sorted_outcome_evaluations(cohort.eligible),
        cohort.excluded_outcome_evaluation_ids,
        cohort.limitations,
    )


def _payload_limitations(
    *,
    folds: tuple[WalkForwardFold, ...],
    eligible: tuple[PredictionOutcomeEvaluation, ...],
    excluded_ids: tuple[str, ...],
    filter_limitations: tuple[str, ...],
    minimum_train_size: int,
    test_size: int,
) -> tuple[str, ...]:
    limitations = list(filter_limitations)
    if excluded_ids and not filter_limitations:
        limitations.append(
            "Some outcome evaluations were excluded by the requested cohort filters."
        )
    if not eligible:
        limitations.append("No outcome evaluations matched the requested walk-forward cohort.")
    elif len(eligible) < minimum_train_size + test_size:
        limitations.append(
            "Insufficient chronological outcome evaluations for the requested "
            "minimum_train_size and test_size."
        )
    if not folds:
        limitations.append("No walk-forward folds were produced.")
    for fold in folds:
        limitations.extend(fold.limitations)
    return tuple(dict.fromkeys(limitations))


def _fold_limitations(
    *,
    train: tuple[PredictionOutcomeEvaluation, ...],
    test: tuple[PredictionOutcomeEvaluation, ...],
    train_score: float | None,
    test_score: float | None,
) -> tuple[str, ...]:
    limitations: list[str] = []
    if not train:
        limitations.append("Walk-forward fold has no training outcomes.")
    elif train_score is None:
        limitations.append("Walk-forward fold has no resolved training outcomes.")
    if not test:
        limitations.append("Walk-forward fold has no held-out outcomes.")
    elif test_score is None:
        limitations.append("Walk-forward fold has no resolved held-out outcomes.")
    return tuple(limitations)


def _filtered_outcome_evaluations(
    outcome_evaluations: tuple[PredictionOutcomeEvaluation, ...],
    *,
    prediction_type: PredictionType | None,
    horizon: TimeHorizon | None,
) -> tuple[PredictionOutcomeEvaluation, ...]:
    return tuple(
        item
        for item in outcome_evaluations
        if (prediction_type is None or item.outcome.prediction_type == prediction_type)
        and (horizon is None or item.outcome.horizon == horizon)
    )


def _sorted_outcome_evaluations(
    outcome_evaluations: tuple[PredictionOutcomeEvaluation, ...],
) -> tuple[PredictionOutcomeEvaluation, ...]:
    return tuple(
        sorted(
            outcome_evaluations,
            key=lambda item: (item.evaluated_at, item.outcome_evaluation_id),
        )
    )


def _quality_average(outcome_evaluations: tuple[PredictionOutcomeEvaluation, ...]) -> float | None:
    scores = [
        item.quality_score
        for item in outcome_evaluations
        if is_resolved_outcome_evaluation(item) and item.quality_score is not None
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 6)


def _validate_window_parameters(
    *,
    minimum_train_size: int,
    test_size: int,
    step_size: int,
) -> None:
    if minimum_train_size < 1:
        raise ValueError("minimum_train_size must be at least 1")
    if test_size < 1:
        raise ValueError("test_size must be at least 1")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")


def _calibration_id(
    *,
    run_id: str,
    cohort_id: str,
    point_in_time_cutoff: datetime,
    minimum_train_size: int,
    test_size: int,
    step_size: int,
    outcome_evaluation_ids: tuple[str, ...],
) -> str:
    calibration_digest = digest(
        "|".join(
            (
                run_id,
                cohort_id,
                point_in_time_cutoff.isoformat(),
                str(minimum_train_size),
                str(test_size),
                str(step_size),
                ",".join(outcome_evaluation_ids),
            )
        )
    )
    return f"calibration-walk-forward-{slug(cohort_id)}-{calibration_digest[:12]}"


__all__ = [
    "PHASE6_WALK_FORWARD_TOOL_NAME",
    "PHASE6_WALK_FORWARD_TOOL_VERSION",
    "WalkForwardEvaluationArtifactPayload",
    "WalkForwardEvaluationArtifacts",
    "WalkForwardFold",
    "compute_walk_forward_folds",
    "walk_forward_evaluations_from_outcome_artifacts",
    "write_walk_forward_evaluation_artifact",
]
