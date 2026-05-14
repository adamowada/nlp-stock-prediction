"""Shared Phase 6 evaluation cohort and persistence helpers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TypeVar

from nlp_stock_prediction.contracts.enums import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeStatus,
    PredictionType,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.evaluation import (
    PredictionEvaluationTarget,
    PredictionOutcomeEvaluation,
)
from nlp_stock_prediction.storage.records import CalibrationSliceRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

RESOLVED_OUTCOME_EVALUATION_STATUSES = frozenset(
    {
        PredictionOutcomeEvaluationStatus.CONFIRMED,
        PredictionOutcomeEvaluationStatus.MISSED,
        PredictionOutcomeEvaluationStatus.MIXED,
        PredictionOutcomeEvaluationStatus.INCONCLUSIVE,
    }
)


class TargetedOutcomeInput(Protocol):
    """Protocol for cohort inputs that pair a target with an outcome evaluation."""

    @property
    def target(self) -> PredictionEvaluationTarget: ...

    @property
    def outcome_evaluation(self) -> PredictionOutcomeEvaluation: ...


TItem = TypeVar("TItem")


@dataclass(frozen=True)
class FilteredOutcomeCohort[TItem]:
    """Outcome inputs kept and excluded by one point-in-time cohort query."""

    eligible: tuple[TItem, ...]
    excluded_outcome_evaluation_ids: tuple[str, ...]
    limitations: tuple[str, ...]


def filter_targeted_outcome_inputs[TTargetedInput: TargetedOutcomeInput](
    inputs: Sequence[TTargetedInput],
    *,
    as_of: datetime,
    prediction_type: PredictionType | None,
    horizon: TimeHorizon | None,
    cutoff_reason: str,
    score_available: Callable[[TTargetedInput], bool] | None = None,
    scoreless_reason: str | None = None,
) -> FilteredOutcomeCohort[TTargetedInput]:
    """Apply the common no-lookahead and cohort filters for target/evaluation pairs."""

    eligible: list[TTargetedInput] = []
    excluded_ids: list[str] = []
    limitations: list[str] = []
    for item in inputs:
        outcome_evaluation = item.outcome_evaluation
        if outcome_evaluation.evaluated_at > as_of:
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            limitations.append(f"{cutoff_reason}: {outcome_evaluation.outcome_evaluation_id}.")
            continue
        if prediction_type is not None and item.target.prediction_type != prediction_type:
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            continue
        if horizon is not None and item.target.horizon != horizon:
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            continue
        if score_available is not None and not score_available(item):
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            if scoreless_reason is not None:
                limitations.append(
                    f"{scoreless_reason}: {outcome_evaluation.outcome_evaluation_id}."
                )
            continue
        eligible.append(item)
    return FilteredOutcomeCohort(
        eligible=tuple(eligible),
        excluded_outcome_evaluation_ids=tuple(dict.fromkeys(excluded_ids)),
        limitations=tuple(dict.fromkeys(limitations)),
    )


def filter_outcome_evaluations(
    outcome_evaluations: Sequence[PredictionOutcomeEvaluation],
    *,
    as_of: datetime,
    prediction_type: PredictionType | None,
    horizon: TimeHorizon | None,
    cutoff_reason: str,
) -> FilteredOutcomeCohort[PredictionOutcomeEvaluation]:
    """Apply the common cohort filters to raw outcome evaluations."""

    eligible: list[PredictionOutcomeEvaluation] = []
    excluded_ids: list[str] = []
    limitations: list[str] = []
    for outcome_evaluation in outcome_evaluations:
        if outcome_evaluation.evaluated_at > as_of:
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            limitations.append(f"{cutoff_reason}: {outcome_evaluation.outcome_evaluation_id}.")
            continue
        if (
            prediction_type is not None
            and outcome_evaluation.outcome.prediction_type != prediction_type
        ):
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            continue
        if horizon is not None and outcome_evaluation.outcome.horizon != horizon:
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            continue
        eligible.append(outcome_evaluation)
    return FilteredOutcomeCohort(
        eligible=tuple(eligible),
        excluded_outcome_evaluation_ids=tuple(dict.fromkeys(excluded_ids)),
        limitations=tuple(dict.fromkeys(limitations)),
    )


def outcome_status_counts(
    outcome_evaluations: Iterable[PredictionOutcomeEvaluation],
) -> dict[str, int]:
    """Count Phase 6 outcome statuses using the shared resolved-status policy."""

    counts = {
        "resolved_count": 0,
        "pending_count": 0,
        "stale_count": 0,
        "unavailable_count": 0,
        "not_evaluable_count": 0,
    }
    for outcome_evaluation in outcome_evaluations:
        status = outcome_evaluation.status
        outcome_status = outcome_evaluation.outcome.status
        if is_resolved_outcome_evaluation(outcome_evaluation):
            counts["resolved_count"] += 1
        elif status == PredictionOutcomeEvaluationStatus.PENDING:
            counts["pending_count"] += 1
        elif status == PredictionOutcomeEvaluationStatus.STALE:
            counts["stale_count"] += 1
        elif outcome_status == PredictionOutcomeStatus.UNAVAILABLE:
            counts["unavailable_count"] += 1
        else:
            counts["not_evaluable_count"] += 1
    return counts


def is_resolved_outcome_evaluation(
    outcome_evaluation: PredictionOutcomeEvaluation,
    *,
    require_quality_score: bool = False,
) -> bool:
    """Return whether an outcome evaluation is resolved for cohort metrics."""

    if outcome_evaluation.status not in RESOLVED_OUTCOME_EVALUATION_STATUSES:
        return False
    return outcome_evaluation.quality_score is not None if require_quality_score else True


def source_outcome_evaluation_ids(
    outcome_evaluations: Iterable[PredictionOutcomeEvaluation],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.outcome_evaluation_id for item in outcome_evaluations))


def source_target_ids(inputs: Iterable[TargetedOutcomeInput]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.target.target_id for item in inputs))


def source_artifact_ids_from_targeted_inputs(
    inputs: Iterable[TargetedOutcomeInput],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            artifact_id
            for item in inputs
            for artifact_id in (
                *tuple(artifact.artifact_id for artifact in item.target.signal_artifacts),
                *item.outcome_evaluation.outcome.artifact_ids,
                *item.outcome_evaluation.artifact_ids,
            )
        )
    )


def source_artifact_ids_from_outcome_evaluations(
    outcome_evaluations: Iterable[PredictionOutcomeEvaluation],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            artifact_id
            for item in outcome_evaluations
            for artifact_id in (*item.outcome.artifact_ids, *item.artifact_ids)
        )
    )


def single_prediction_type(inputs: Iterable[TargetedOutcomeInput]) -> PredictionType | None:
    values = {item.target.prediction_type for item in inputs}
    return next(iter(values)) if len(values) == 1 else None


def single_horizon(inputs: Iterable[TargetedOutcomeInput]) -> TimeHorizon | None:
    values = {item.target.horizon for item in inputs}
    return next(iter(values)) if len(values) == 1 else None


def write_calibration_slices(
    store: SQLiteStore,
    *,
    calibration_id: str,
    slices: Iterable[CalibrationSliceRecord],
) -> tuple[CalibrationSliceRecord, ...]:
    """Replace calibration slices as one small persistence Interface."""

    records = tuple(slices)
    store.delete_calibration_slices(calibration_id)
    for record in records:
        store.record_calibration_slice(record)
    return records


def aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def duplicate_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def validate_unique(label: str, values: tuple[object, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def slug(
    value: str,
    *,
    fallback: str = "cohort",
    allow_file_safe_punctuation: bool = False,
) -> str:
    if allow_file_safe_punctuation:
        normalized = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower())
        normalized = re.sub(r"-+", "-", normalized).strip("-._")
        return normalized or fallback
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or fallback


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
