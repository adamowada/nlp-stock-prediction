"""Signal-family attribution and ablation for Phase 6 calibration."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import Field

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
)
from nlp_stock_prediction.contracts.enums import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeStatus,
    PredictionType,
    SignalArtifactFamily,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.evaluation import (
    BaselineComparison,
    PredictionEvaluationTarget,
    PredictionOutcomeEvaluation,
    SignalFamilyAblation,
)
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.evaluation.outcomes import PointInTimeOutcomeEvaluationArtifacts
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase4_common import safe_phase4_tool_execution
from nlp_stock_prediction.storage.records import (
    CalibrationRunRecord,
    CalibrationSliceRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE6_ABLATION_TOOL_NAME = "phase6_signal_family_ablation"
PHASE6_ABLATION_TOOL_VERSION = "phase6.signal-family-ablation.v1"

_RESOLVED_STATUSES = frozenset(
    {
        PredictionOutcomeEvaluationStatus.CONFIRMED,
        PredictionOutcomeEvaluationStatus.MISSED,
        PredictionOutcomeEvaluationStatus.MIXED,
        PredictionOutcomeEvaluationStatus.INCONCLUSIVE,
    }
)
_SIGNAL_FAMILY_ORDER = tuple(SignalArtifactFamily)


@dataclass(frozen=True)
class SignalFamilyAblationInput:
    """A frozen target paired with its point-in-time outcome evaluation."""

    target: PredictionEvaluationTarget
    outcome_evaluation: PredictionOutcomeEvaluation

    def __post_init__(self) -> None:
        if self.target.candidate_id != self.outcome_evaluation.candidate_id:
            raise ValueError("ablation input target and outcome candidate_id must match")
        if self.target.instrument_id != self.outcome_evaluation.instrument_id:
            raise ValueError("ablation input target and outcome instrument_id must match")
        if self.target.symbol.upper() != self.outcome_evaluation.symbol.upper():
            raise ValueError("ablation input target and outcome symbol must match")
        outcome = self.outcome_evaluation.outcome
        if self.target.evaluation_window_start != outcome.evaluation_window_start:
            raise ValueError("ablation input evaluation_window_start must match target")
        if self.target.evaluation_window_end != outcome.evaluation_window_end:
            raise ValueError("ablation input evaluation_window_end must match target")


class SignalFamilyAblationArtifactPayload(ContractModel):
    """Stable JSON payload for persisted signal-family ablations."""

    schema_version: NonEmptyStr = "signal-family-ablation-artifact.v1"
    run_id: NonEmptyStr
    calibration_id: NonEmptyStr
    cohort_id: NonEmptyStr
    created_at: AwareDatetime
    point_in_time_cutoff: AwareDatetime
    ablations: tuple[SignalFamilyAblation, ...] = Field(default_factory=tuple)
    source_target_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    source_outcome_evaluation_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    source_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)


@dataclass(frozen=True)
class SignalFamilyAblationArtifacts:
    """Artifacts and storage records produced by a signal-family ablation write."""

    calibration_id: str
    calibration_run: CalibrationRunRecord
    calibration_slices: tuple[CalibrationSliceRecord, ...]
    ablations: tuple[SignalFamilyAblation, ...]
    artifact: AuditArtifact
    artifact_payload: SignalFamilyAblationArtifactPayload
    tool_run_id: str


def signal_family_ablation_inputs_from_outcome_artifacts(
    outcome_artifacts: Iterable[PointInTimeOutcomeEvaluationArtifacts],
) -> tuple[SignalFamilyAblationInput, ...]:
    """Build ablation inputs from Stage 3 point-in-time outcome artifacts."""

    return tuple(
        SignalFamilyAblationInput(
            target=item.target,
            outcome_evaluation=item.outcome_evaluation,
        )
        for item in outcome_artifacts
    )


def compute_signal_family_ablations(
    *,
    cohort_id: str,
    created_at: datetime,
    inputs: Sequence[SignalFamilyAblationInput],
    families: Sequence[SignalArtifactFamily] | None = None,
    prediction_type: PredictionType | None = None,
    horizon: TimeHorizon | None = None,
) -> tuple[SignalFamilyAblation, ...]:
    """Compute included-versus-excluded outcome quality for each signal family."""

    created = _aware_utc(created_at, "created_at")
    requested_families = tuple(families) if families is not None else _SIGNAL_FAMILY_ORDER
    filtered_inputs = _filtered_inputs(
        tuple(inputs),
        prediction_type=prediction_type,
        horizon=horizon,
    )
    inferred_prediction_type = prediction_type or _single_prediction_type(filtered_inputs)
    inferred_horizon = horizon or _single_horizon(filtered_inputs)

    return tuple(
        _compute_family_ablation(
            cohort_id=cohort_id,
            created_at=created,
            family=family,
            inputs=filtered_inputs,
            prediction_type=inferred_prediction_type,
            horizon=inferred_horizon,
        )
        for family in requested_families
    )


def write_signal_family_ablation_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    cohort_id: str,
    inputs: Sequence[SignalFamilyAblationInput],
    point_in_time_cutoff: datetime,
    created_at: datetime | None = None,
    families: Sequence[SignalArtifactFamily] | None = None,
    prediction_type: PredictionType | None = None,
    horizon: TimeHorizon | None = None,
    calibration_id: str | None = None,
    tool_run_id: str | None = None,
    artifact_filename: str | None = None,
    record_tool_run: bool = True,
) -> SignalFamilyAblationArtifacts:
    """Persist signal-family ablation metrics as an audit artifact and calibration slices."""

    created = _aware_utc(created_at or datetime.now(UTC), "created_at")
    cutoff = _aware_utc(point_in_time_cutoff, "point_in_time_cutoff")
    ablation_inputs = tuple(inputs)
    _validate_inputs_for_run(ablation_inputs, run_id)
    requested_families = tuple(families) if families is not None else _SIGNAL_FAMILY_ORDER
    source_outcome_evaluation_ids = _source_outcome_evaluation_ids(ablation_inputs)
    source_target_ids = _source_target_ids(ablation_inputs)
    source_artifact_ids = _source_artifact_ids(ablation_inputs)
    resolved_calibration_id = calibration_id or _calibration_id(
        run_id=run_id,
        cohort_id=cohort_id,
        point_in_time_cutoff=cutoff,
        families=requested_families,
        outcome_evaluation_ids=source_outcome_evaluation_ids,
    )
    digest = _digest(
        "|".join(
            (
                run_id,
                cohort_id,
                resolved_calibration_id,
                ",".join(source_outcome_evaluation_ids),
                ",".join(family.value for family in requested_families),
            )
        )
    )
    resolved_tool_run_id = tool_run_id or f"tool-signal-family-ablation-{digest[:12]}"
    filtered_inputs = _filtered_inputs(
        ablation_inputs,
        prediction_type=prediction_type,
        horizon=horizon,
    )
    ablations = compute_signal_family_ablations(
        cohort_id=cohort_id,
        created_at=created,
        inputs=ablation_inputs,
        families=requested_families,
        prediction_type=prediction_type,
        horizon=horizon,
    )
    limitations = _payload_limitations(ablations, filtered_inputs, requested_families)
    payload = SignalFamilyAblationArtifactPayload(
        run_id=run_id,
        calibration_id=resolved_calibration_id,
        cohort_id=cohort_id,
        created_at=created,
        point_in_time_cutoff=cutoff,
        ablations=ablations,
        source_target_ids=source_target_ids,
        source_outcome_evaluation_ids=source_outcome_evaluation_ids,
        source_artifact_ids=source_artifact_ids,
        limitations=limitations,
        metadata={
            "prediction_type": prediction_type.value if prediction_type else None,
            "horizon": horizon.value if horizon else None,
            "family_count": len(requested_families),
            "sample_count": len(filtered_inputs),
            "resolved_count": _status_counts(filtered_inputs)["resolved_count"],
        },
    )
    tool_inputs: JsonObject = {
        "cohort_id": cohort_id,
        "calibration_id": resolved_calibration_id,
        "point_in_time_cutoff": cutoff.isoformat(),
        "families": [family.value for family in requested_families],
        "prediction_type": prediction_type.value if prediction_type else None,
        "horizon": horizon.value if horizon else None,
        "source_outcome_evaluation_ids": list(source_outcome_evaluation_ids),
    }

    def write_records() -> SignalFamilyAblationArtifacts:
        if record_tool_run:
            store.record_tool_run(
                ToolRunRecord(
                    tool_run_id=resolved_tool_run_id,
                    run_id=run_id,
                    tool_name=PHASE6_ABLATION_TOOL_NAME,
                    tool_version=PHASE6_ABLATION_TOOL_VERSION,
                    status="successful",
                    started_at=created,
                    completed_at=created,
                    inputs=tool_inputs,
                    warnings=limitations,
                )
            )
        artifact_id = f"artifact-signal-family-ablation-{_slug(cohort_id)}-{digest[:12]}"
        artifact = ArtifactIndex.for_directory(
            store=store,
            repo_root=repo_root,
            base_dir=artifact_dir,
            created_at=created,
            produced_by=PHASE6_ABLATION_TOOL_NAME,
            tool_run_id=resolved_tool_run_id,
            schema_version=payload.schema_version,
        ).write_json(
            artifact_id=artifact_id,
            artifact_type="signal_family_ablation",
            filename=artifact_filename
            or f"calibration/signal-family-ablations/{_slug(cohort_id)}.json",
            payload=cast(JsonObject, payload.model_dump(mode="json")),
            record_count=len(ablations),
            metadata={
                "run_id": run_id,
                "calibration_id": resolved_calibration_id,
                "cohort_id": cohort_id,
                "source_outcome_evaluation_count": len(source_outcome_evaluation_ids),
                "family_count": len(requested_families),
            },
        )
        calibration_run = CalibrationRunRecord(
            calibration_id=resolved_calibration_id,
            run_id=run_id,
            method_version=PHASE6_ABLATION_TOOL_VERSION,
            created_at=created,
            point_in_time_cutoff=cutoff,
            tool_run_id=resolved_tool_run_id,
            cohort_query={
                "cohort_id": cohort_id,
                "prediction_type": prediction_type.value if prediction_type else None,
                "horizon": horizon.value if horizon else None,
                "families": [family.value for family in requested_families],
            },
            source_outcome_evaluation_ids=source_outcome_evaluation_ids,
            artifact_id=artifact.artifact_id,
            limitations=limitations,
            metadata={
                "source_target_ids": list(source_target_ids),
                "source_artifact_ids": list(source_artifact_ids),
                "sample_count": len(filtered_inputs),
            },
        )
        store.record_calibration_run(calibration_run)
        slices = tuple(
            _calibration_slice_record(
                calibration_id=resolved_calibration_id,
                ablation=ablation,
                inputs=filtered_inputs,
            )
            for ablation in ablations
        )
        for calibration_slice in slices:
            store.record_calibration_slice(calibration_slice)
        return SignalFamilyAblationArtifacts(
            calibration_id=resolved_calibration_id,
            calibration_run=calibration_run,
            calibration_slices=slices,
            ablations=ablations,
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
        tool_name=PHASE6_ABLATION_TOOL_NAME,
        tool_version=PHASE6_ABLATION_TOOL_VERSION,
        started_at=created,
        inputs=tool_inputs,
    ):
        return write_records()


def _compute_family_ablation(
    *,
    cohort_id: str,
    created_at: datetime,
    family: SignalArtifactFamily,
    inputs: tuple[SignalFamilyAblationInput, ...],
    prediction_type: PredictionType | None,
    horizon: TimeHorizon | None,
) -> SignalFamilyAblation:
    included = tuple(item for item in inputs if _family_artifact_ids(item.target, family))
    excluded = tuple(item for item in inputs if not _family_artifact_ids(item.target, family))
    included_score, resolved_included = _quality_average(included)
    excluded_score, resolved_excluded = _quality_average(excluded)
    delta = (
        round(included_score - excluded_score, 6)
        if included_score is not None and excluded_score is not None
        else None
    )
    baseline_comparison = _baseline_comparison_for_ablation(
        family=family,
        included_quality_score=included_score,
        excluded_quality_score=excluded_score,
        quality_score_delta=delta,
    )
    limitations = _ablation_limitations(
        family=family,
        inputs=inputs,
        included=included,
        excluded=excluded,
        resolved_included=resolved_included,
        resolved_excluded=resolved_excluded,
    )
    return SignalFamilyAblation(
        ablation_id=f"ablation-{_slug(cohort_id)}-{family.value}",
        cohort_id=cohort_id,
        created_at=created_at,
        family=family,
        prediction_type=prediction_type,
        horizon=horizon,
        included_prediction_count=len(included),
        excluded_prediction_count=len(excluded),
        resolved_included_count=resolved_included,
        resolved_excluded_count=resolved_excluded,
        included_quality_score=included_score,
        excluded_quality_score=excluded_score,
        quality_score_delta=delta,
        baseline_comparison=baseline_comparison,
        limitations=limitations,
        metadata=_ablation_metadata(family=family, included=included, excluded=excluded),
    )


def _baseline_comparison_for_ablation(
    *,
    family: SignalArtifactFamily,
    included_quality_score: float | None,
    excluded_quality_score: float | None,
    quality_score_delta: float | None,
) -> BaselineComparison | None:
    if (
        included_quality_score is None
        or excluded_quality_score is None
        or quality_score_delta is None
    ):
        return None
    if quality_score_delta > 0.05:
        verdict = "above_baseline"
    elif quality_score_delta < -0.05:
        verdict = "below_baseline"
    else:
        verdict = "near_baseline"
    return BaselineComparison(
        baseline_id=f"without_{family.value}_signals",
        baseline_summary=(
            f"Average resolved outcome quality for predictions without {family.value} "
            "signal artifacts."
        ),
        baseline_score=excluded_quality_score,
        candidate_score=included_quality_score,
        score_delta=quality_score_delta,
        verdict=verdict,
    )


def _ablation_limitations(
    *,
    family: SignalArtifactFamily,
    inputs: tuple[SignalFamilyAblationInput, ...],
    included: tuple[SignalFamilyAblationInput, ...],
    excluded: tuple[SignalFamilyAblationInput, ...],
    resolved_included: int,
    resolved_excluded: int,
) -> tuple[str, ...]:
    limitations: list[str] = []
    if not inputs:
        limitations.append("No outcome evaluations were available in the requested cohort.")
    if not included:
        limitations.append(f"No predictions in the cohort used {family.value} signal artifacts.")
    elif resolved_included == 0:
        limitations.append(
            f"No resolved included outcomes were available for {family.value} signal artifacts."
        )
    if not excluded:
        limitations.append(
            f"No comparison predictions without {family.value} signal artifacts were available."
        )
    elif resolved_excluded == 0:
        limitations.append(
            f"No resolved excluded outcomes were available for {family.value} signal artifacts."
        )
    return tuple(limitations)


def _ablation_metadata(
    *,
    family: SignalArtifactFamily,
    included: tuple[SignalFamilyAblationInput, ...],
    excluded: tuple[SignalFamilyAblationInput, ...],
) -> JsonObject:
    return cast(
        JsonObject,
        {
            "included_target_ids": [item.target.target_id for item in included],
            "excluded_target_ids": [item.target.target_id for item in excluded],
            "included_outcome_evaluation_ids": [
                item.outcome_evaluation.outcome_evaluation_id for item in included
            ],
            "excluded_outcome_evaluation_ids": [
                item.outcome_evaluation.outcome_evaluation_id for item in excluded
            ],
            "resolved_included_outcome_evaluation_ids": [
                item.outcome_evaluation.outcome_evaluation_id
                for item in included
                if _is_resolved(item.outcome_evaluation)
            ],
            "resolved_excluded_outcome_evaluation_ids": [
                item.outcome_evaluation.outcome_evaluation_id
                for item in excluded
                if _is_resolved(item.outcome_evaluation)
            ],
            "included_signal_artifact_ids": _dedupe(
                artifact_id
                for item in included
                for artifact_id in _family_artifact_ids(item.target, family)
            ),
        },
    )


def _calibration_slice_record(
    *,
    calibration_id: str,
    ablation: SignalFamilyAblation,
    inputs: tuple[SignalFamilyAblationInput, ...],
) -> CalibrationSliceRecord:
    status_counts = _status_counts(inputs)
    baseline_comparison = (
        {}
        if ablation.baseline_comparison is None
        else cast(JsonObject, ablation.baseline_comparison.model_dump(mode="json"))
    )
    return CalibrationSliceRecord(
        slice_id=f"slice-{calibration_id}-{ablation.family.value}",
        calibration_id=calibration_id,
        signal_family=ablation.family.value,
        prediction_type=ablation.prediction_type.value if ablation.prediction_type else None,
        horizon=ablation.horizon.value if ablation.horizon else None,
        cohort_label=f"signal_family_ablation:{ablation.family.value}",
        sample_count=ablation.included_prediction_count + ablation.excluded_prediction_count,
        resolved_count=status_counts["resolved_count"],
        pending_count=status_counts["pending_count"],
        stale_count=status_counts["stale_count"],
        unavailable_count=status_counts["unavailable_count"],
        not_evaluable_count=status_counts["not_evaluable_count"],
        metrics={
            "included_prediction_count": ablation.included_prediction_count,
            "excluded_prediction_count": ablation.excluded_prediction_count,
            "resolved_included_count": ablation.resolved_included_count,
            "resolved_excluded_count": ablation.resolved_excluded_count,
            "included_quality_score": ablation.included_quality_score,
            "excluded_quality_score": ablation.excluded_quality_score,
            "quality_score_delta": ablation.quality_score_delta,
        },
        baseline_comparison=baseline_comparison,
        provenance=cast(
            JsonObject,
            {
                "included_target_ids": _metadata_list(ablation, "included_target_ids"),
                "excluded_target_ids": _metadata_list(ablation, "excluded_target_ids"),
                "included_outcome_evaluation_ids": _metadata_list(
                    ablation,
                    "included_outcome_evaluation_ids",
                ),
                "excluded_outcome_evaluation_ids": _metadata_list(
                    ablation,
                    "excluded_outcome_evaluation_ids",
                ),
                "included_signal_artifact_ids": _metadata_list(
                    ablation,
                    "included_signal_artifact_ids",
                ),
            },
        ),
        metadata=cast(JsonObject, {"limitations": list(ablation.limitations)}),
    )


def _payload_limitations(
    ablations: tuple[SignalFamilyAblation, ...],
    inputs: tuple[SignalFamilyAblationInput, ...],
    families: tuple[SignalArtifactFamily, ...],
) -> tuple[str, ...]:
    limitations: list[str] = []
    if not families:
        limitations.append("No signal families were requested for ablation.")
    if not inputs:
        limitations.append("No outcome evaluations matched the requested cohort filters.")
    for ablation in ablations:
        limitations.extend(ablation.limitations)
    return tuple(dict.fromkeys(limitations))


def _filtered_inputs(
    inputs: tuple[SignalFamilyAblationInput, ...],
    *,
    prediction_type: PredictionType | None,
    horizon: TimeHorizon | None,
) -> tuple[SignalFamilyAblationInput, ...]:
    return tuple(
        item
        for item in inputs
        if (prediction_type is None or item.target.prediction_type == prediction_type)
        and (horizon is None or item.target.horizon == horizon)
    )


def _single_prediction_type(
    inputs: tuple[SignalFamilyAblationInput, ...],
) -> PredictionType | None:
    values = {item.target.prediction_type for item in inputs}
    return next(iter(values)) if len(values) == 1 else None


def _single_horizon(inputs: tuple[SignalFamilyAblationInput, ...]) -> TimeHorizon | None:
    values = {item.target.horizon for item in inputs}
    return next(iter(values)) if len(values) == 1 else None


def _quality_average(
    inputs: tuple[SignalFamilyAblationInput, ...],
) -> tuple[float | None, int]:
    scores = [
        item.outcome_evaluation.quality_score
        for item in inputs
        if _is_resolved(item.outcome_evaluation)
        and item.outcome_evaluation.quality_score is not None
    ]
    if not scores:
        return None, 0
    return round(sum(scores) / len(scores), 6), len(scores)


def _is_resolved(outcome_evaluation: PredictionOutcomeEvaluation) -> bool:
    return outcome_evaluation.status in _RESOLVED_STATUSES


def _status_counts(
    inputs: tuple[SignalFamilyAblationInput, ...],
) -> dict[str, int]:
    counts = {
        "resolved_count": 0,
        "pending_count": 0,
        "stale_count": 0,
        "unavailable_count": 0,
        "not_evaluable_count": 0,
    }
    for item in inputs:
        status = item.outcome_evaluation.status
        outcome_status = item.outcome_evaluation.outcome.status
        if status in _RESOLVED_STATUSES:
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


def _family_artifact_ids(
    target: PredictionEvaluationTarget,
    family: SignalArtifactFamily,
) -> tuple[str, ...]:
    return tuple(
        artifact.artifact_id for artifact in target.signal_artifacts if artifact.family == family
    )


def _validate_inputs_for_run(
    inputs: tuple[SignalFamilyAblationInput, ...],
    run_id: str,
) -> None:
    mismatched = tuple(item.target.target_id for item in inputs if item.target.run_id != run_id)
    if mismatched:
        raise ValueError(
            "ablation inputs must belong to the persisted run_id: " + ", ".join(mismatched)
        )


def _source_outcome_evaluation_ids(
    inputs: tuple[SignalFamilyAblationInput, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.outcome_evaluation.outcome_evaluation_id for item in inputs))


def _source_target_ids(inputs: tuple[SignalFamilyAblationInput, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.target.target_id for item in inputs))


def _source_artifact_ids(inputs: tuple[SignalFamilyAblationInput, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            artifact.artifact_id for item in inputs for artifact in item.target.signal_artifacts
        )
    )


def _metadata_list(ablation: SignalFamilyAblation, key: str) -> list[str]:
    value = ablation.metadata.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _calibration_id(
    *,
    run_id: str,
    cohort_id: str,
    point_in_time_cutoff: datetime,
    families: tuple[SignalArtifactFamily, ...],
    outcome_evaluation_ids: tuple[str, ...],
) -> str:
    digest = _digest(
        "|".join(
            (
                run_id,
                cohort_id,
                point_in_time_cutoff.isoformat(),
                ",".join(family.value for family in families),
                ",".join(outcome_evaluation_ids),
            )
        )
    )
    return f"calibration-signal-family-ablation-{_slug(cohort_id)}-{digest[:12]}"


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "cohort"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "PHASE6_ABLATION_TOOL_NAME",
    "PHASE6_ABLATION_TOOL_VERSION",
    "SignalFamilyAblationArtifactPayload",
    "SignalFamilyAblationArtifacts",
    "SignalFamilyAblationInput",
    "compute_signal_family_ablations",
    "signal_family_ablation_inputs_from_outcome_artifacts",
    "write_signal_family_ablation_artifact",
]
