"""Cohort-level calibration summaries for Phase 6 prediction quality."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
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
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeStatus,
    PredictionType,
    SignalArtifactFamily,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.evaluation import (
    BaselineComparison,
    CalibrationBin,
    CalibrationSummary,
    PredictionEvaluationTarget,
    PredictionOutcomeEvaluation,
    SignalArtifactReference,
    SignalFamilyCalibrationSummary,
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

PHASE6_CALIBRATION_TOOL_NAME = "phase6_calibration_summary"
PHASE6_CALIBRATION_TOOL_VERSION = "phase6.calibration-summary.v1"
DEFAULT_CALIBRATION_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

_RESOLVED_STATUSES = frozenset(
    {
        PredictionOutcomeEvaluationStatus.CONFIRMED,
        PredictionOutcomeEvaluationStatus.MISSED,
        PredictionOutcomeEvaluationStatus.MIXED,
        PredictionOutcomeEvaluationStatus.INCONCLUSIVE,
    }
)


@dataclass(frozen=True)
class CalibrationSummaryInput:
    """A frozen point-in-time target paired with its outcome evaluation."""

    target: PredictionEvaluationTarget
    outcome_evaluation: PredictionOutcomeEvaluation

    def __post_init__(self) -> None:
        if self.target.candidate_id != self.outcome_evaluation.candidate_id:
            raise ValueError("calibration input target and outcome candidate_id must match")
        if self.target.instrument_id != self.outcome_evaluation.instrument_id:
            raise ValueError("calibration input target and outcome instrument_id must match")
        if self.target.symbol.upper() != self.outcome_evaluation.symbol.upper():
            raise ValueError("calibration input target and outcome symbol must match")
        outcome = self.outcome_evaluation.outcome
        if self.target.prediction_type != outcome.prediction_type:
            raise ValueError("calibration input prediction_type must match target")
        if self.target.horizon != outcome.horizon:
            raise ValueError("calibration input horizon must match target")
        if self.target.evaluation_window_start != outcome.evaluation_window_start:
            raise ValueError("calibration input evaluation_window_start must match target")
        if self.target.evaluation_window_end != outcome.evaluation_window_end:
            raise ValueError("calibration input evaluation_window_end must match target")


class CalibrationSummaryArtifactPayload(ContractModel):
    """Stable JSON payload for persisted calibration summaries."""

    schema_version: NonEmptyStr = "calibration-summary-artifact.v1"
    run_id: NonEmptyStr
    calibration_id: NonEmptyStr
    cohort_id: NonEmptyStr
    created_at: AwareDatetime
    as_of: AwareDatetime
    bin_edges: tuple[float, ...] = Field(default_factory=tuple)
    summary: CalibrationSummary
    excluded_outcome_evaluation_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> CalibrationSummaryArtifactPayload:
        _validate_bin_edges(self.bin_edges)
        _validate_unique(
            "calibration payload excluded outcome evaluation ids",
            self.excluded_outcome_evaluation_ids,
        )
        if self.summary.calibration_id != self.calibration_id:
            raise ValueError("calibration payload summary calibration_id must match")
        if self.summary.cohort_id != self.cohort_id:
            raise ValueError("calibration payload summary cohort_id must match")
        return self


@dataclass(frozen=True)
class CalibrationSummaryArtifacts:
    """Artifacts and storage records produced by a calibration summary write."""

    calibration_id: str
    calibration_run: CalibrationRunRecord
    calibration_slices: tuple[CalibrationSliceRecord, ...]
    summary: CalibrationSummary
    artifact: AuditArtifact
    artifact_payload: CalibrationSummaryArtifactPayload
    tool_run_id: str


@dataclass(frozen=True)
class _CalibrationItem:
    source: CalibrationSummaryInput
    prediction_score: float


def calibration_summary_inputs_from_outcome_artifacts(
    outcome_artifacts: Iterable[PointInTimeOutcomeEvaluationArtifacts],
) -> tuple[CalibrationSummaryInput, ...]:
    """Build calibration inputs from Stage 3 point-in-time outcome artifacts."""

    return tuple(
        CalibrationSummaryInput(
            target=item.target,
            outcome_evaluation=item.outcome_evaluation,
        )
        for item in outcome_artifacts
    )


def compute_calibration_summary(
    *,
    calibration_id: str,
    cohort_id: str,
    created_at: datetime,
    as_of: datetime,
    inputs: Sequence[CalibrationSummaryInput],
    bin_edges: Sequence[float] = DEFAULT_CALIBRATION_BIN_EDGES,
    prediction_type: PredictionType | None = None,
    horizon: TimeHorizon | None = None,
    families: Sequence[SignalArtifactFamily] | None = None,
) -> CalibrationSummary:
    """Compute reliability bins and cohort-level quality calibration metrics."""

    created = _aware_utc(created_at, "created_at")
    cutoff = _aware_utc(as_of, "as_of")
    edges = tuple(bin_edges)
    _validate_bin_edges(edges)
    filtered = _filtered_inputs(
        tuple(inputs),
        as_of=cutoff,
        prediction_type=prediction_type,
        horizon=horizon,
    )
    items = _scoreable_items(filtered)
    status_counts = _status_counts(items)
    resolved_items = tuple(item for item in items if _is_resolved(item.source.outcome_evaluation))
    resolved_count = len(resolved_items)
    requested_families = _requested_families(items, families)
    brier_score = _brier_score(resolved_items)
    log_loss = _log_loss(resolved_items)
    accuracy = _accuracy(resolved_items)
    expected_calibration_error = _expected_calibration_error(
        _calibration_bins(items, edges),
        resolved_count=resolved_count,
    )
    limitations = _summary_limitations(
        inputs=tuple(inputs),
        filtered=filtered,
        items=items,
        resolved_count=resolved_count,
    )
    inferred_prediction_type = prediction_type or _single_prediction_type(items)
    inferred_horizon = horizon or _single_horizon(items)
    return CalibrationSummary(
        calibration_id=calibration_id,
        cohort_id=cohort_id,
        created_at=created,
        as_of=cutoff,
        prediction_type=inferred_prediction_type,
        horizon=inferred_horizon,
        sample_count=len(items),
        resolved_count=resolved_count,
        pending_count=status_counts["pending_count"],
        stale_count=status_counts["stale_count"],
        unavailable_count=status_counts["unavailable_count"],
        not_evaluable_count=status_counts["not_evaluable_count"],
        bins=_calibration_bins(items, edges),
        brier_score=brier_score,
        log_loss=log_loss,
        accuracy=accuracy,
        expected_calibration_error=expected_calibration_error,
        baseline_comparison=_baseline_comparison(resolved_items, brier_score),
        signal_families=_signal_family_summaries(
            items,
            families=requested_families,
            cohort_observed_frequency=_observed_frequency(resolved_items),
        ),
        source_outcome_evaluation_ids=_source_outcome_evaluation_ids(items),
        source_artifact_ids=_source_artifact_ids(items),
        limitations=limitations,
        metadata={
            "bin_edges": list(edges),
            "scoreless_input_count": len(filtered) - len(items),
            "requested_input_count": len(tuple(inputs)),
        },
    )


def write_calibration_summary_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    cohort_id: str,
    inputs: Sequence[CalibrationSummaryInput],
    as_of: datetime,
    created_at: datetime | None = None,
    bin_edges: Sequence[float] = DEFAULT_CALIBRATION_BIN_EDGES,
    prediction_type: PredictionType | None = None,
    horizon: TimeHorizon | None = None,
    families: Sequence[SignalArtifactFamily] | None = None,
    calibration_id: str | None = None,
    tool_run_id: str | None = None,
    artifact_filename: str | None = None,
    record_tool_run: bool = True,
) -> CalibrationSummaryArtifacts:
    """Persist a calibration summary artifact and its calibration slice records."""

    created = _aware_utc(created_at or datetime.now(UTC), "created_at")
    cutoff = _aware_utc(as_of, "as_of")
    edges = tuple(bin_edges)
    _validate_bin_edges(edges)
    requested = tuple(inputs)
    _validate_inputs_for_run(requested, run_id)
    eligible, excluded_ids, eligibility_limitations = _eligible_inputs(
        requested,
        as_of=cutoff,
        prediction_type=prediction_type,
        horizon=horizon,
    )
    source_ids = _source_outcome_evaluation_ids(_scoreable_items(eligible))
    resolved_calibration_id = calibration_id or _calibration_id(
        run_id=run_id,
        cohort_id=cohort_id,
        as_of=cutoff,
        bin_edges=edges,
        outcome_evaluation_ids=source_ids,
    )
    summary = compute_calibration_summary(
        calibration_id=resolved_calibration_id,
        cohort_id=cohort_id,
        created_at=created,
        as_of=cutoff,
        inputs=eligible,
        bin_edges=edges,
        prediction_type=prediction_type,
        horizon=horizon,
        families=families,
    )
    summary_limitations = tuple(dict.fromkeys((*eligibility_limitations, *summary.limitations)))
    summary = summary.model_copy(update={"limitations": summary_limitations})
    digest = _digest(
        "|".join(
            (
                run_id,
                cohort_id,
                resolved_calibration_id,
                cutoff.isoformat(),
                ",".join(source_ids),
                ",".join(f"{edge:.8f}" for edge in edges),
            )
        )
    )
    resolved_tool_run_id = tool_run_id or f"tool-calibration-summary-{digest[:12]}"
    payload = CalibrationSummaryArtifactPayload(
        run_id=run_id,
        calibration_id=resolved_calibration_id,
        cohort_id=cohort_id,
        created_at=created,
        as_of=cutoff,
        bin_edges=edges,
        summary=summary,
        excluded_outcome_evaluation_ids=excluded_ids,
        limitations=summary.limitations,
        metadata={
            "requested_input_count": len(requested),
            "eligible_input_count": len(eligible),
            "scoreable_input_count": summary.sample_count,
        },
    )
    tool_inputs: JsonObject = {
        "cohort_id": cohort_id,
        "calibration_id": resolved_calibration_id,
        "as_of": cutoff.isoformat(),
        "bin_edges": list(edges),
        "prediction_type": prediction_type.value if prediction_type else None,
        "horizon": horizon.value if horizon else None,
        "source_outcome_evaluation_ids": list(source_ids),
        "excluded_outcome_evaluation_ids": list(excluded_ids),
    }

    def write_records() -> CalibrationSummaryArtifacts:
        if record_tool_run:
            store.record_tool_run(
                ToolRunRecord(
                    tool_run_id=resolved_tool_run_id,
                    run_id=run_id,
                    tool_name=PHASE6_CALIBRATION_TOOL_NAME,
                    tool_version=PHASE6_CALIBRATION_TOOL_VERSION,
                    status="successful",
                    started_at=created,
                    completed_at=created,
                    inputs=tool_inputs,
                    warnings=summary.limitations,
                )
            )
        artifact_id = f"artifact-calibration-summary-{_slug(cohort_id)}-{digest[:12]}"
        artifact = ArtifactIndex.for_directory(
            store=store,
            repo_root=repo_root,
            base_dir=artifact_dir,
            created_at=created,
            produced_by=PHASE6_CALIBRATION_TOOL_NAME,
            tool_run_id=resolved_tool_run_id if record_tool_run or tool_run_id else None,
            schema_version=payload.schema_version,
        ).write_json(
            artifact_id=artifact_id,
            artifact_type="calibration_summary",
            filename=artifact_filename or f"calibration/summaries/{_slug(cohort_id)}.json",
            payload=cast(JsonObject, payload.model_dump(mode="json")),
            record_count=1,
            metadata={
                "run_id": run_id,
                "calibration_id": resolved_calibration_id,
                "cohort_id": cohort_id,
                "sample_count": summary.sample_count,
                "resolved_count": summary.resolved_count,
            },
        )
        calibration_run = CalibrationRunRecord(
            calibration_id=resolved_calibration_id,
            run_id=run_id,
            method_version=PHASE6_CALIBRATION_TOOL_VERSION,
            created_at=created,
            point_in_time_cutoff=cutoff,
            tool_run_id=resolved_tool_run_id if record_tool_run or tool_run_id else None,
            cohort_query={
                "cohort_id": cohort_id,
                "as_of": cutoff.isoformat(),
                "bin_edges": list(edges),
                "prediction_type": prediction_type.value if prediction_type else None,
                "horizon": horizon.value if horizon else None,
            },
            source_outcome_evaluation_ids=summary.source_outcome_evaluation_ids,
            artifact_id=artifact.artifact_id,
            limitations=summary.limitations,
            metadata={
                "excluded_outcome_evaluation_ids": list(excluded_ids),
                "source_artifact_ids": list(summary.source_artifact_ids),
                "summary_schema_version": summary.schema_version,
            },
        )
        store.record_calibration_run(calibration_run)
        slices = _calibration_slice_records(
            calibration_id=resolved_calibration_id,
            summary=summary,
            items=_scoreable_items(eligible),
            bin_edges=edges,
        )
        for calibration_slice in slices:
            store.record_calibration_slice(calibration_slice)
        return CalibrationSummaryArtifacts(
            calibration_id=resolved_calibration_id,
            calibration_run=calibration_run,
            calibration_slices=slices,
            summary=summary,
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
        tool_name=PHASE6_CALIBRATION_TOOL_NAME,
        tool_version=PHASE6_CALIBRATION_TOOL_VERSION,
        started_at=created,
        inputs=tool_inputs,
    ):
        return write_records()


def _calibration_bins(
    items: tuple[_CalibrationItem, ...],
    edges: tuple[float, ...],
) -> tuple[CalibrationBin, ...]:
    if not items:
        return ()
    bins: list[CalibrationBin] = []
    for index, (lower, upper) in enumerate(pairwise(edges), start=1):
        bin_items = tuple(
            item for item in items if _score_in_bin(item.prediction_score, lower, upper)
        )
        resolved_items = tuple(
            item for item in bin_items if _is_resolved(item.source.outcome_evaluation)
        )
        resolved_count = len(resolved_items)
        limitations: list[str] = []
        if bin_items and resolved_count == 0:
            limitations.append("Calibration bin has predictions but no resolved outcomes.")
        bins.append(
            CalibrationBin(
                bin_id=f"bin-{index:02d}-{_edge_label(lower)}-{_edge_label(upper)}",
                lower_bound=lower,
                upper_bound=upper,
                prediction_count=len(bin_items),
                resolved_count=resolved_count,
                average_score=_average_prediction_score(resolved_items),
                observed_frequency=_observed_frequency(resolved_items),
                brier_score=_brier_score(resolved_items),
                limitations=tuple(limitations),
            )
        )
    return tuple(bins)


def _signal_family_summaries(
    items: tuple[_CalibrationItem, ...],
    *,
    families: tuple[SignalArtifactFamily, ...],
    cohort_observed_frequency: float | None,
) -> tuple[SignalFamilyCalibrationSummary, ...]:
    summaries: list[SignalFamilyCalibrationSummary] = []
    for family in families:
        family_items = tuple(
            item for item in items if _family_artifacts(item.source.target, family)
        )
        signal_artifact_count = sum(
            len(_family_artifacts(item.source.target, family)) for item in family_items
        )
        resolved_items = tuple(
            item for item in family_items if _is_resolved(item.source.outcome_evaluation)
        )
        observed_frequency = _observed_frequency(resolved_items)
        limitations: list[str] = []
        if family_items and not resolved_items:
            limitations.append(
                f"No resolved outcomes were available for {family.value} signal calibration."
            )
        if not family_items and families:
            limitations.append(f"No {family.value} signal artifacts were present in the cohort.")
        summaries.append(
            SignalFamilyCalibrationSummary(
                family=family,
                prediction_count=len(family_items),
                resolved_count=len(resolved_items),
                signal_artifact_count=signal_artifact_count,
                average_score=_average_prediction_score(resolved_items),
                observed_frequency=observed_frequency,
                brier_score=_brier_score(resolved_items),
                score_delta_vs_baseline=(
                    round(observed_frequency - cohort_observed_frequency, 6)
                    if observed_frequency is not None and cohort_observed_frequency is not None
                    else None
                ),
                limitations=tuple(limitations),
                metadata={
                    "source_outcome_evaluation_ids": list(
                        _source_outcome_evaluation_ids(family_items)
                    ),
                    "source_signal_artifact_ids": _dedupe(
                        artifact.artifact_id
                        for item in family_items
                        for artifact in _family_artifacts(item.source.target, family)
                    ),
                },
            )
        )
    return tuple(summaries)


def _calibration_slice_records(
    *,
    calibration_id: str,
    summary: CalibrationSummary,
    items: tuple[_CalibrationItem, ...],
    bin_edges: tuple[float, ...],
) -> tuple[CalibrationSliceRecord, ...]:
    slices = [_overall_slice_record(calibration_id=calibration_id, summary=summary)]
    bins = summary.bins
    for bin_model in bins:
        bin_items = tuple(
            item
            for item in items
            if _score_in_bin(item.prediction_score, bin_model.lower_bound, bin_model.upper_bound)
        )
        slices.append(
            _bin_slice_record(
                calibration_id=calibration_id,
                bin_model=bin_model,
                bin_items=bin_items,
                bin_edges=bin_edges,
            )
        )
    for family_summary in summary.signal_families:
        family_items = tuple(
            item for item in items if _family_artifacts(item.source.target, family_summary.family)
        )
        slices.append(
            _family_slice_record(
                calibration_id=calibration_id,
                family_summary=family_summary,
                family_items=family_items,
            )
        )
    return tuple(
        sorted(
            slices,
            key=lambda item: (item.cohort_label, item.signal_family or "", item.slice_id),
        )
    )


def _overall_slice_record(
    *,
    calibration_id: str,
    summary: CalibrationSummary,
) -> CalibrationSliceRecord:
    baseline_comparison = (
        {}
        if summary.baseline_comparison is None
        else cast(JsonObject, summary.baseline_comparison.model_dump(mode="json"))
    )
    return CalibrationSliceRecord(
        slice_id=f"slice-{calibration_id}-overall",
        calibration_id=calibration_id,
        prediction_type=summary.prediction_type.value if summary.prediction_type else None,
        horizon=summary.horizon.value if summary.horizon else None,
        cohort_label="calibration_summary:overall",
        sample_count=summary.sample_count,
        resolved_count=summary.resolved_count,
        pending_count=summary.pending_count,
        stale_count=summary.stale_count,
        unavailable_count=summary.unavailable_count,
        not_evaluable_count=summary.not_evaluable_count,
        metrics=_summary_metrics(summary),
        baseline_comparison=baseline_comparison,
        provenance=cast(
            JsonObject,
            {
                "source_outcome_evaluation_ids": list(summary.source_outcome_evaluation_ids),
                "source_artifact_ids": list(summary.source_artifact_ids),
            },
        ),
        metadata=cast(JsonObject, {"limitations": list(summary.limitations)}),
    )


def _bin_slice_record(
    *,
    calibration_id: str,
    bin_model: CalibrationBin,
    bin_items: tuple[_CalibrationItem, ...],
    bin_edges: tuple[float, ...],
) -> CalibrationSliceRecord:
    status_counts = _status_counts(bin_items)
    return CalibrationSliceRecord(
        slice_id=f"slice-{calibration_id}-{bin_model.bin_id}",
        calibration_id=calibration_id,
        cohort_label=f"calibration_bin:{bin_model.bin_id}",
        sample_count=bin_model.prediction_count,
        resolved_count=bin_model.resolved_count,
        pending_count=status_counts["pending_count"],
        stale_count=status_counts["stale_count"],
        unavailable_count=status_counts["unavailable_count"],
        not_evaluable_count=status_counts["not_evaluable_count"],
        metrics={
            "lower_bound": bin_model.lower_bound,
            "upper_bound": bin_model.upper_bound,
            "average_score": bin_model.average_score,
            "observed_frequency": bin_model.observed_frequency,
            "brier_score": bin_model.brier_score,
        },
        provenance=cast(
            JsonObject,
            {
                "bin_edges": list(bin_edges),
                "source_outcome_evaluation_ids": list(_source_outcome_evaluation_ids(bin_items)),
            },
        ),
        metadata=cast(JsonObject, {"limitations": list(bin_model.limitations)}),
    )


def _family_slice_record(
    *,
    calibration_id: str,
    family_summary: SignalFamilyCalibrationSummary,
    family_items: tuple[_CalibrationItem, ...],
) -> CalibrationSliceRecord:
    status_counts = _status_counts(family_items)
    return CalibrationSliceRecord(
        slice_id=f"slice-{calibration_id}-family-{family_summary.family.value}",
        calibration_id=calibration_id,
        signal_family=family_summary.family.value,
        cohort_label=f"signal_family_calibration:{family_summary.family.value}",
        sample_count=family_summary.prediction_count,
        resolved_count=family_summary.resolved_count,
        pending_count=status_counts["pending_count"],
        stale_count=status_counts["stale_count"],
        unavailable_count=status_counts["unavailable_count"],
        not_evaluable_count=status_counts["not_evaluable_count"],
        metrics={
            "signal_artifact_count": family_summary.signal_artifact_count,
            "average_score": family_summary.average_score,
            "observed_frequency": family_summary.observed_frequency,
            "brier_score": family_summary.brier_score,
            "score_delta_vs_baseline": family_summary.score_delta_vs_baseline,
        },
        provenance=cast(
            JsonObject,
            {
                "source_outcome_evaluation_ids": list(_source_outcome_evaluation_ids(family_items)),
                "source_signal_artifact_ids": family_summary.metadata.get(
                    "source_signal_artifact_ids",
                    [],
                ),
            },
        ),
        metadata=cast(JsonObject, {"limitations": list(family_summary.limitations)}),
    )


def _summary_metrics(summary: CalibrationSummary) -> JsonObject:
    return {
        "brier_score": summary.brier_score,
        "log_loss": summary.log_loss,
        "accuracy": summary.accuracy,
        "expected_calibration_error": summary.expected_calibration_error,
    }


def _eligible_inputs(
    inputs: tuple[CalibrationSummaryInput, ...],
    *,
    as_of: datetime,
    prediction_type: PredictionType | None,
    horizon: TimeHorizon | None,
) -> tuple[tuple[CalibrationSummaryInput, ...], tuple[str, ...], tuple[str, ...]]:
    eligible: list[CalibrationSummaryInput] = []
    excluded_ids: list[str] = []
    limitations: list[str] = []
    for item in inputs:
        outcome_evaluation = item.outcome_evaluation
        if outcome_evaluation.evaluated_at > as_of:
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            limitations.append(
                "Excluded outcome evaluation after the calibration as_of cutoff: "
                f"{outcome_evaluation.outcome_evaluation_id}."
            )
            continue
        if prediction_type is not None and item.target.prediction_type != prediction_type:
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            continue
        if horizon is not None and item.target.horizon != horizon:
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            continue
        if _prediction_score(item) is None:
            excluded_ids.append(outcome_evaluation.outcome_evaluation_id)
            limitations.append(
                "Excluded outcome evaluation without a prediction score: "
                f"{outcome_evaluation.outcome_evaluation_id}."
            )
            continue
        eligible.append(item)
    return (
        tuple(eligible),
        tuple(dict.fromkeys(excluded_ids)),
        tuple(dict.fromkeys(limitations)),
    )


def _filtered_inputs(
    inputs: tuple[CalibrationSummaryInput, ...],
    *,
    as_of: datetime,
    prediction_type: PredictionType | None,
    horizon: TimeHorizon | None,
) -> tuple[CalibrationSummaryInput, ...]:
    return tuple(
        item
        for item in inputs
        if item.outcome_evaluation.evaluated_at <= as_of
        and (prediction_type is None or item.target.prediction_type == prediction_type)
        and (horizon is None or item.target.horizon == horizon)
    )


def _scoreable_items(inputs: tuple[CalibrationSummaryInput, ...]) -> tuple[_CalibrationItem, ...]:
    return tuple(
        _CalibrationItem(source=item, prediction_score=score)
        for item in inputs
        for score in (_prediction_score(item),)
        if score is not None
    )


def _prediction_score(item: CalibrationSummaryInput) -> float | None:
    baseline = item.target.baseline_comparison
    if baseline is not None:
        return _bounded_score(baseline.candidate_score)
    for key in ("score", "confidence", "candidate_score"):
        value = item.target.candidate_snapshot.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return _bounded_score(float(value))
    return None


def _summary_limitations(
    *,
    inputs: tuple[CalibrationSummaryInput, ...],
    filtered: tuple[CalibrationSummaryInput, ...],
    items: tuple[_CalibrationItem, ...],
    resolved_count: int,
) -> tuple[str, ...]:
    limitations: list[str] = []
    if not inputs:
        limitations.append("No outcome evaluations were provided for calibration.")
    if inputs and not filtered:
        limitations.append(
            "No outcome evaluations matched the calibration filters and as_of cutoff."
        )
    if filtered and len(items) < len(filtered):
        limitations.append("Some calibration inputs did not include a prediction score.")
    if not items:
        limitations.append("No scoreable outcome evaluations were available for calibration.")
    elif resolved_count == 0:
        limitations.append("No resolved scoreable outcomes were available for calibration.")
    return tuple(dict.fromkeys(limitations))


def _requested_families(
    items: tuple[_CalibrationItem, ...],
    families: Sequence[SignalArtifactFamily] | None,
) -> tuple[SignalArtifactFamily, ...]:
    if families is not None:
        return tuple(dict.fromkeys(families))
    observed = tuple(
        dict.fromkeys(
            artifact.family for item in items for artifact in item.source.target.signal_artifacts
        )
    )
    return observed


def _status_counts(items: tuple[_CalibrationItem, ...]) -> dict[str, int]:
    counts = {
        "resolved_count": 0,
        "pending_count": 0,
        "stale_count": 0,
        "unavailable_count": 0,
        "not_evaluable_count": 0,
    }
    for item in items:
        outcome_evaluation = item.source.outcome_evaluation
        status = outcome_evaluation.status
        outcome_status = outcome_evaluation.outcome.status
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


def _is_resolved(outcome_evaluation: PredictionOutcomeEvaluation) -> bool:
    return (
        outcome_evaluation.status in _RESOLVED_STATUSES
        and outcome_evaluation.quality_score is not None
    )


def _outcome_quality(item: _CalibrationItem) -> float:
    quality = item.source.outcome_evaluation.quality_score
    if quality is None:
        raise ValueError("resolved calibration items require quality_score")
    return _bounded_score(quality)


def _brier_score(items: tuple[_CalibrationItem, ...]) -> float | None:
    if not items:
        return None
    return round(
        sum((item.prediction_score - _outcome_quality(item)) ** 2 for item in items) / len(items),
        6,
    )


def _log_loss(items: tuple[_CalibrationItem, ...]) -> float | None:
    if not items:
        return None
    epsilon = 1e-6
    total = 0.0
    for item in items:
        probability = min(max(item.prediction_score, epsilon), 1.0 - epsilon)
        outcome = _outcome_quality(item)
        total += -(outcome * math.log(probability) + (1.0 - outcome) * math.log(1.0 - probability))
    return round(total / len(items), 6)


def _accuracy(items: tuple[_CalibrationItem, ...]) -> float | None:
    if not items:
        return None
    correct = sum(
        1 for item in items if (item.prediction_score >= 0.5) == (_outcome_quality(item) >= 0.5)
    )
    return round(correct / len(items), 6)


def _expected_calibration_error(
    bins: tuple[CalibrationBin, ...],
    *,
    resolved_count: int,
) -> float | None:
    if resolved_count == 0:
        return None
    total = 0.0
    for bin_model in bins:
        if (
            bin_model.resolved_count == 0
            or bin_model.average_score is None
            or bin_model.observed_frequency is None
        ):
            continue
        total += (
            bin_model.resolved_count
            / resolved_count
            * abs(bin_model.average_score - bin_model.observed_frequency)
        )
    return round(total, 6)


def _baseline_comparison(
    resolved_items: tuple[_CalibrationItem, ...],
    brier_score: float | None,
) -> BaselineComparison | None:
    if brier_score is None or not resolved_items:
        return None
    baseline_brier = round(
        sum((0.5 - _outcome_quality(item)) ** 2 for item in resolved_items) / len(resolved_items),
        6,
    )
    candidate_score = _bounded_score(1.0 - brier_score)
    baseline_score = _bounded_score(1.0 - baseline_brier)
    delta = round(candidate_score - baseline_score, 6)
    if delta > 0.05:
        verdict = "above_baseline"
    elif delta < -0.05:
        verdict = "below_baseline"
    else:
        verdict = "near_baseline"
    return BaselineComparison(
        baseline_id="no_skill_constant_0_5",
        baseline_summary="Calibration quality for a constant 0.5 no-skill score.",
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        score_delta=delta,
        verdict=verdict,
    )


def _average_prediction_score(items: tuple[_CalibrationItem, ...]) -> float | None:
    if not items:
        return None
    return round(sum(item.prediction_score for item in items) / len(items), 6)


def _observed_frequency(items: tuple[_CalibrationItem, ...]) -> float | None:
    if not items:
        return None
    return round(sum(_outcome_quality(item) for item in items) / len(items), 6)


def _family_artifacts(
    target: PredictionEvaluationTarget,
    family: SignalArtifactFamily,
) -> tuple[SignalArtifactReference, ...]:
    return tuple(artifact for artifact in target.signal_artifacts if artifact.family == family)


def _source_outcome_evaluation_ids(items: tuple[_CalibrationItem, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(item.source.outcome_evaluation.outcome_evaluation_id for item in items)
    )


def _source_artifact_ids(items: tuple[_CalibrationItem, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            artifact_id
            for item in items
            for artifact_id in (
                *tuple(artifact.artifact_id for artifact in item.source.target.signal_artifacts),
                *item.source.outcome_evaluation.outcome.artifact_ids,
                *item.source.outcome_evaluation.artifact_ids,
            )
        )
    )


def _single_prediction_type(items: tuple[_CalibrationItem, ...]) -> PredictionType | None:
    values = {item.source.target.prediction_type for item in items}
    return next(iter(values)) if len(values) == 1 else None


def _single_horizon(items: tuple[_CalibrationItem, ...]) -> TimeHorizon | None:
    values = {item.source.target.horizon for item in items}
    return next(iter(values)) if len(values) == 1 else None


def _validate_inputs_for_run(inputs: tuple[CalibrationSummaryInput, ...], run_id: str) -> None:
    duplicate_ids = _duplicate_ids(
        tuple(item.outcome_evaluation.outcome_evaluation_id for item in inputs)
    )
    if duplicate_ids:
        raise ValueError("calibration inputs must be unique: " + ", ".join(duplicate_ids))
    mismatched = tuple(item.target.target_id for item in inputs if item.target.run_id != run_id)
    if mismatched:
        raise ValueError(
            "calibration inputs must belong to the persisted run_id: " + ", ".join(mismatched)
        )


def _duplicate_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _validate_bin_edges(edges: Sequence[float]) -> None:
    if len(edges) < 2:
        raise ValueError("calibration bin_edges require at least two values")
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("calibration bin_edges must start at 0.0 and end at 1.0")
    previous = -1.0
    for edge in edges:
        if edge < 0.0 or edge > 1.0:
            raise ValueError("calibration bin_edges must stay within [0.0, 1.0]")
        if edge <= previous:
            raise ValueError("calibration bin_edges must be strictly increasing")
        previous = edge


def _validate_unique(label: str, values: tuple[object, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _score_in_bin(score: float, lower: float, upper: float) -> bool:
    if upper == 1.0:
        return lower <= score <= upper
    return lower <= score < upper


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _calibration_id(
    *,
    run_id: str,
    cohort_id: str,
    as_of: datetime,
    bin_edges: tuple[float, ...],
    outcome_evaluation_ids: tuple[str, ...],
) -> str:
    digest = _digest(
        "|".join(
            (
                run_id,
                cohort_id,
                as_of.isoformat(),
                ",".join(f"{edge:.8f}" for edge in bin_edges),
                ",".join(outcome_evaluation_ids),
            )
        )
    )
    return f"calibration-summary-{_slug(cohort_id)}-{digest[:12]}"


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _edge_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "cohort"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_CALIBRATION_BIN_EDGES",
    "PHASE6_CALIBRATION_TOOL_NAME",
    "PHASE6_CALIBRATION_TOOL_VERSION",
    "CalibrationSummaryArtifactPayload",
    "CalibrationSummaryArtifacts",
    "CalibrationSummaryInput",
    "calibration_summary_inputs_from_outcome_artifacts",
    "compute_calibration_summary",
    "write_calibration_summary_artifact",
]
