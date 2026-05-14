"""Calibration drift checks for Phase 7 evaluation hardening."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    JsonValue,
    NonEmptyStr,
)
from nlp_stock_prediction.contracts.enums import (
    SignalArtifactFamily,
)
from nlp_stock_prediction.contracts.evaluation import (
    CalibrationBin,
    CalibrationDriftCheck,
    CalibrationDriftStatus,
    CalibrationSummary,
    SignalFamilyCalibrationSummary,
)
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.evaluation.calibration import CalibrationSummaryArtifactPayload
from nlp_stock_prediction.evaluation.common import aware_utc, digest, slug
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase4_common import safe_phase4_tool_execution
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    CalibrationDriftCheckRecord,
    CalibrationRunRecord,
    CalibrationSliceRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE7_CALIBRATION_DRIFT_TOOL_NAME = "phase7_calibration_drift_check"
PHASE7_CALIBRATION_DRIFT_TOOL_VERSION = "phase7.calibration-drift.v1"


@dataclass(frozen=True, slots=True)
class CalibrationDriftThresholds:
    """Conservative thresholds for classifying calibration movement."""

    min_resolved_count: int = 10
    watch_delta: float = 0.05
    degraded_delta: float = 0.10
    improved_delta: float = 0.10


DEFAULT_CALIBRATION_DRIFT_THRESHOLDS = CalibrationDriftThresholds()


@dataclass(frozen=True)
class CalibrationSummarySource:
    """A persisted calibration summary plus its storage context."""

    calibration_run: CalibrationRunRecord
    artifact: ArtifactRecord
    artifact_path: Path
    payload: CalibrationSummaryArtifactPayload
    calibration_slices: tuple[CalibrationSliceRecord, ...]

    @property
    def summary(self) -> CalibrationSummary:
        return self.payload.summary


class CalibrationDriftArtifactPayload(ContractModel):
    """Stable JSON payload for a persisted calibration drift check."""

    schema_version: NonEmptyStr = "calibration-drift-artifact.v1"
    run_id: NonEmptyStr
    created_at: AwareDatetime
    as_of: AwareDatetime
    drift_check: CalibrationDriftCheck
    prior_calibration_artifact_id: NonEmptyStr
    current_calibration_artifact_id: NonEmptyStr
    prior_calibration_slice_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    current_calibration_slice_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> CalibrationDriftArtifactPayload:
        if self.drift_check.created_at != self.created_at:
            raise ValueError("drift payload created_at must match drift_check")
        if self.drift_check.as_of != self.as_of:
            raise ValueError("drift payload as_of must match drift_check")
        if self.drift_check.limitations != self.limitations:
            raise ValueError("drift payload limitations must match drift_check")
        if self.prior_calibration_artifact_id == self.current_calibration_artifact_id:
            raise ValueError("drift payload requires distinct source calibration artifacts")
        return self


@dataclass(frozen=True)
class CalibrationDriftArtifacts:
    """Artifacts and storage records produced by a drift check write."""

    drift_check_id: str
    drift_check: CalibrationDriftCheck
    record: CalibrationDriftCheckRecord
    artifact: AuditArtifact
    artifact_payload: CalibrationDriftArtifactPayload
    tool_run_id: str


def compute_calibration_drift_check(
    *,
    prior_summary: CalibrationSummary,
    current_summary: CalibrationSummary,
    created_at: datetime,
    as_of: datetime,
    source_calibration_artifact_ids: tuple[str, ...],
    thresholds: CalibrationDriftThresholds = DEFAULT_CALIBRATION_DRIFT_THRESHOLDS,
    signal_family: SignalArtifactFamily | None = None,
    provider_compatibility_note_ids: tuple[str, ...] = (),
    evidence_aging_record_ids: tuple[str, ...] = (),
    artifact_freshness_review_ids: tuple[str, ...] = (),
    prior_calibration_slice_ids: tuple[str, ...] = (),
    current_calibration_slice_ids: tuple[str, ...] = (),
) -> CalibrationDriftCheck:
    """Compare two calibration summaries without introducing lookahead."""

    created = aware_utc(created_at, "created_at")
    cutoff = aware_utc(as_of, "as_of")
    base_kwargs = _base_drift_kwargs(
        prior_summary=prior_summary,
        current_summary=current_summary,
        created_at=created,
        as_of=cutoff,
        source_calibration_artifact_ids=source_calibration_artifact_ids,
        signal_family=signal_family,
        provider_compatibility_note_ids=provider_compatibility_note_ids,
        evidence_aging_record_ids=evidence_aging_record_ids,
        artifact_freshness_review_ids=artifact_freshness_review_ids,
        prior_calibration_slice_ids=prior_calibration_slice_ids,
        current_calibration_slice_ids=current_calibration_slice_ids,
    )
    limitations = list(
        _compatibility_limitations(
            prior_summary=prior_summary,
            current_summary=current_summary,
            as_of=cutoff,
            signal_family=signal_family,
            source_calibration_artifact_ids=source_calibration_artifact_ids,
        )
    )
    if limitations:
        return CalibrationDriftCheck(
            **base_kwargs,
            drift_status="not_evaluable",
            limitations=tuple(dict.fromkeys(limitations)),
        )
    if (
        prior_summary.resolved_count < thresholds.min_resolved_count
        or current_summary.resolved_count < thresholds.min_resolved_count
    ):
        return CalibrationDriftCheck(
            **base_kwargs,
            drift_status="insufficient_history",
            limitations=(
                "Calibration drift requires more resolved history before metric movement "
                f"is evaluable: prior={prior_summary.resolved_count}, "
                f"current={current_summary.resolved_count}, "
                f"minimum={thresholds.min_resolved_count}.",
            ),
        )

    metric_deltas = _metric_deltas(
        prior_summary=prior_summary,
        current_summary=current_summary,
        signal_family=signal_family,
    )
    if not _has_quality_metric_delta(metric_deltas):
        return CalibrationDriftCheck(
            **base_kwargs,
            drift_status="inconclusive",
            limitations=("No comparable calibration metrics were available for drift.",),
        )

    status = _classify_drift(metric_deltas, thresholds=thresholds)
    if status == "inconclusive":
        inconclusive_kwargs = {
            **base_kwargs,
            "metadata": {
                **cast(JsonObject, base_kwargs["metadata"]),
                "suppressed_metric_delta_keys": list(metric_deltas),
            },
        }
        return CalibrationDriftCheck(
            **inconclusive_kwargs,
            drift_status="inconclusive",
            limitations=("Calibration metrics moved in conflicting directions.",),
        )

    return CalibrationDriftCheck(
        **base_kwargs,
        drift_status=status,
        metric_deltas=metric_deltas,
    )


def load_calibration_summary_source(
    *,
    store: SQLiteStore,
    repo_root: Path,
    calibration_id: str,
) -> CalibrationSummarySource:
    """Load and validate one persisted calibration summary artifact and its slices."""

    calibration_run = store.get_calibration_run(calibration_id)
    if calibration_run is None:
        raise ValueError(f"calibration run does not exist: {calibration_id}")
    if calibration_run.artifact_id is None:
        raise ValueError(f"calibration run is missing artifact_id: {calibration_id}")
    artifact = store.get_artifact(calibration_run.artifact_id)
    if artifact is None:
        raise ValueError(
            "calibration run references a missing artifact: "
            f"{calibration_id} -> {calibration_run.artifact_id}"
        )
    if artifact.artifact_type != "calibration_summary":
        raise ValueError(
            "calibration run points to the wrong artifact type: "
            f"{calibration_id} -> {artifact.artifact_type}"
        )
    artifact_path = _artifact_path(repo_root, artifact.path)
    if not artifact_path.exists():
        raise ValueError(
            "calibration summary artifact file is missing: "
            f"{artifact.artifact_id} at {artifact_path.as_posix()}"
        )
    digest_value = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if digest_value != artifact.sha256:
        raise ValueError(
            f"calibration summary artifact hash does not match storage row: {artifact.artifact_id}"
        )
    try:
        payload = CalibrationSummaryArtifactPayload.model_validate(
            json.loads(artifact_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise ValueError(
            f"calibration summary artifact is not a valid Phase 6 payload: {artifact.artifact_id}"
        ) from exc
    _validate_source_payload(calibration_run=calibration_run, payload=payload)
    return CalibrationSummarySource(
        calibration_run=calibration_run,
        artifact=artifact,
        artifact_path=artifact_path,
        payload=payload,
        calibration_slices=store.list_calibration_slices(calibration_id),
    )


def write_calibration_drift_check_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    prior_calibration_id: str,
    current_calibration_id: str,
    as_of: datetime,
    created_at: datetime | None = None,
    thresholds: CalibrationDriftThresholds = DEFAULT_CALIBRATION_DRIFT_THRESHOLDS,
    signal_family: SignalArtifactFamily | None = None,
    provider_compatibility_note_ids: tuple[str, ...] = (),
    evidence_aging_record_ids: tuple[str, ...] = (),
    artifact_freshness_review_ids: tuple[str, ...] = (),
    tool_run_id: str | None = None,
    record_tool_run: bool = True,
) -> CalibrationDriftArtifacts:
    """Persist one calibration drift artifact and SQLite drift row."""

    prior = load_calibration_summary_source(
        store=store,
        repo_root=repo_root,
        calibration_id=prior_calibration_id,
    )
    current = load_calibration_summary_source(
        store=store,
        repo_root=repo_root,
        calibration_id=current_calibration_id,
    )
    if current.calibration_run.run_id != run_id:
        raise ValueError("current calibration run_id must match drift run_id")
    cutoff = aware_utc(as_of, "as_of")
    created = aware_utc(created_at or max(datetime.now(UTC), cutoff), "created_at")
    drift_check = compute_calibration_drift_check(
        prior_summary=prior.summary,
        current_summary=current.summary,
        created_at=created,
        as_of=cutoff,
        source_calibration_artifact_ids=(prior.artifact.artifact_id, current.artifact.artifact_id),
        thresholds=thresholds,
        signal_family=signal_family,
        provider_compatibility_note_ids=provider_compatibility_note_ids,
        evidence_aging_record_ids=evidence_aging_record_ids,
        artifact_freshness_review_ids=artifact_freshness_review_ids,
        prior_calibration_slice_ids=tuple(item.slice_id for item in prior.calibration_slices),
        current_calibration_slice_ids=tuple(item.slice_id for item in current.calibration_slices),
    )
    payload = CalibrationDriftArtifactPayload(
        run_id=run_id,
        created_at=created,
        as_of=cutoff,
        drift_check=drift_check,
        prior_calibration_artifact_id=prior.artifact.artifact_id,
        current_calibration_artifact_id=current.artifact.artifact_id,
        prior_calibration_slice_ids=tuple(item.slice_id for item in prior.calibration_slices),
        current_calibration_slice_ids=tuple(item.slice_id for item in current.calibration_slices),
        limitations=drift_check.limitations,
        metadata={
            "prior_calibration_path": prior.artifact_path.as_posix(),
            "current_calibration_path": current.artifact_path.as_posix(),
            "thresholds": _threshold_metadata(thresholds),
        },
    )
    artifact_digest = digest(
        "|".join(
            (
                run_id,
                prior_calibration_id,
                current_calibration_id,
                cutoff.isoformat(),
                str(signal_family.value if signal_family else "overall"),
                drift_check.drift_status,
            )
        )
    )
    resolved_tool_run_id = tool_run_id or f"tool-calibration-drift-{artifact_digest[:12]}"
    tool_inputs = cast(
        JsonObject,
        {
            "prior_calibration_id": prior_calibration_id,
            "current_calibration_id": current_calibration_id,
            "as_of": cutoff.isoformat(),
            "signal_family": signal_family.value if signal_family else None,
            "thresholds": _threshold_metadata(thresholds),
        },
    )

    def write_records() -> CalibrationDriftArtifacts:
        if record_tool_run:
            store.record_tool_run(
                ToolRunRecord(
                    tool_run_id=resolved_tool_run_id,
                    run_id=run_id,
                    tool_name=PHASE7_CALIBRATION_DRIFT_TOOL_NAME,
                    tool_version=PHASE7_CALIBRATION_DRIFT_TOOL_VERSION,
                    status="successful",
                    started_at=created,
                    completed_at=created,
                    inputs=tool_inputs,
                    warnings=drift_check.limitations,
                )
            )
        artifact = ArtifactIndex.for_directory(
            store=store,
            repo_root=repo_root,
            base_dir=artifact_dir,
            created_at=created,
            produced_by=PHASE7_CALIBRATION_DRIFT_TOOL_NAME,
            tool_run_id=resolved_tool_run_id if record_tool_run or tool_run_id else None,
            schema_version=payload.schema_version,
        ).write_json(
            artifact_id=f"artifact-calibration-drift-{slug(drift_check.cohort_id)}-{artifact_digest[:12]}",
            artifact_type="calibration_drift_check",
            filename=(
                f"calibration/drift/{slug(drift_check.cohort_id)}-{artifact_digest[:12]}.json"
            ),
            payload=cast(JsonObject, payload.model_dump(mode="json")),
            record_count=1,
            metadata={
                "run_id": run_id,
                "drift_check_id": drift_check.drift_check_id,
                "cohort_id": drift_check.cohort_id,
                "drift_status": drift_check.drift_status,
                "prior_calibration_id": prior_calibration_id,
                "current_calibration_id": current_calibration_id,
                "source_calibration_artifact_ids": list(
                    drift_check.source_calibration_artifact_ids
                ),
                "source_outcome_evaluation_ids": list(drift_check.source_outcome_evaluation_ids),
            },
        )
        record = CalibrationDriftCheckRecord(
            drift_check_id=drift_check.drift_check_id,
            run_id=run_id,
            created_at=created,
            as_of=cutoff,
            tool_run_id=resolved_tool_run_id if record_tool_run or tool_run_id else None,
            prior_calibration_id=prior_calibration_id,
            current_calibration_id=current_calibration_id,
            prediction_type=(
                None if drift_check.prediction_type is None else drift_check.prediction_type.value
            ),
            horizon=None if drift_check.horizon is None else drift_check.horizon.value,
            signal_family=(
                None if drift_check.signal_family is None else drift_check.signal_family.value
            ),
            drift_status=drift_check.drift_status,
            metric_deltas=drift_check.metric_deltas,
            source_calibration_artifact_ids=drift_check.source_calibration_artifact_ids,
            source_outcome_evaluation_ids=drift_check.source_outcome_evaluation_ids,
            artifact_id=artifact.artifact_id,
            limitations=drift_check.limitations,
            metadata={
                "provider_compatibility_note_ids": list(
                    drift_check.provider_compatibility_note_ids
                ),
                "evidence_aging_record_ids": list(drift_check.evidence_aging_record_ids),
                "artifact_freshness_review_ids": list(drift_check.artifact_freshness_review_ids),
                "source_membership": drift_check.metadata.get(
                    "source_membership",
                    {},
                ),
                "thresholds": _threshold_metadata(thresholds),
            },
        )
        store.record_calibration_drift_check(record)
        return CalibrationDriftArtifacts(
            drift_check_id=drift_check.drift_check_id,
            drift_check=drift_check,
            record=record,
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
        tool_name=PHASE7_CALIBRATION_DRIFT_TOOL_NAME,
        tool_version=PHASE7_CALIBRATION_DRIFT_TOOL_VERSION,
        started_at=created,
        inputs=tool_inputs,
    ):
        return write_records()


def _base_drift_kwargs(
    *,
    prior_summary: CalibrationSummary,
    current_summary: CalibrationSummary,
    created_at: datetime,
    as_of: datetime,
    source_calibration_artifact_ids: tuple[str, ...],
    signal_family: SignalArtifactFamily | None,
    provider_compatibility_note_ids: tuple[str, ...],
    evidence_aging_record_ids: tuple[str, ...],
    artifact_freshness_review_ids: tuple[str, ...],
    prior_calibration_slice_ids: tuple[str, ...],
    current_calibration_slice_ids: tuple[str, ...],
) -> dict[str, object]:
    source_membership = _source_membership(prior_summary, current_summary)
    return {
        "drift_check_id": _drift_check_id(
            prior_summary=prior_summary,
            current_summary=current_summary,
            as_of=as_of,
            signal_family=signal_family,
        ),
        "cohort_id": current_summary.cohort_id,
        "created_at": created_at,
        "as_of": as_of,
        "prior_calibration_id": prior_summary.calibration_id,
        "current_calibration_id": current_summary.calibration_id,
        "prediction_type": current_summary.prediction_type,
        "horizon": current_summary.horizon,
        "signal_family": signal_family,
        "source_calibration_artifact_ids": tuple(dict.fromkeys(source_calibration_artifact_ids)),
        "source_outcome_evaluation_ids": tuple(
            dict.fromkeys(
                (
                    *prior_summary.source_outcome_evaluation_ids,
                    *current_summary.source_outcome_evaluation_ids,
                )
            )
        ),
        "provider_compatibility_note_ids": tuple(dict.fromkeys(provider_compatibility_note_ids)),
        "evidence_aging_record_ids": tuple(dict.fromkeys(evidence_aging_record_ids)),
        "artifact_freshness_review_ids": tuple(dict.fromkeys(artifact_freshness_review_ids)),
        "metadata": {
            "source_membership": source_membership,
            "prior_calibration_slice_ids": list(dict.fromkeys(prior_calibration_slice_ids)),
            "current_calibration_slice_ids": list(dict.fromkeys(current_calibration_slice_ids)),
            "prior_as_of": prior_summary.as_of.isoformat(),
            "current_as_of": current_summary.as_of.isoformat(),
        },
    }


def _compatibility_limitations(
    *,
    prior_summary: CalibrationSummary,
    current_summary: CalibrationSummary,
    as_of: datetime,
    signal_family: SignalArtifactFamily | None,
    source_calibration_artifact_ids: tuple[str, ...],
) -> tuple[str, ...]:
    limitations: list[str] = []
    if prior_summary.as_of > as_of:
        limitations.append("Prior calibration summary is after the drift as_of cutoff.")
    if current_summary.as_of > as_of:
        limitations.append("Current calibration summary is after the drift as_of cutoff.")
    if prior_summary.as_of >= current_summary.as_of:
        limitations.append("Calibration drift requires prior as_of to be before current as_of.")
    if prior_summary.cohort_id != current_summary.cohort_id:
        limitations.append("Calibration drift requires matching cohort_id.")
    if prior_summary.prediction_type != current_summary.prediction_type:
        limitations.append("Calibration drift requires matching prediction_type.")
    if prior_summary.horizon != current_summary.horizon:
        limitations.append("Calibration drift requires matching horizon.")
    if len(tuple(dict.fromkeys(source_calibration_artifact_ids))) < 2:
        limitations.append("Calibration drift requires distinct source calibration artifacts.")
    prior_bins = _bin_edges(prior_summary.bins)
    current_bins = _bin_edges(current_summary.bins)
    if prior_bins != current_bins:
        limitations.append("Calibration drift requires matching calibration bin edges.")
    if signal_family is not None:
        if _family_summary(prior_summary, signal_family) is None:
            limitations.append(
                f"Prior calibration summary has no {signal_family.value} signal-family slice."
            )
        if _family_summary(current_summary, signal_family) is None:
            limitations.append(
                f"Current calibration summary has no {signal_family.value} signal-family slice."
            )
    return tuple(dict.fromkeys(limitations))


def _metric_deltas(
    *,
    prior_summary: CalibrationSummary,
    current_summary: CalibrationSummary,
    signal_family: SignalArtifactFamily | None,
) -> JsonObject:
    deltas: JsonObject = {}
    metric_pairs = (
        ("brier_score", prior_summary.brier_score, current_summary.brier_score),
        ("log_loss", prior_summary.log_loss, current_summary.log_loss),
        ("accuracy", prior_summary.accuracy, current_summary.accuracy),
        (
            "expected_calibration_error",
            prior_summary.expected_calibration_error,
            current_summary.expected_calibration_error,
        ),
    )
    for metric_name, prior_value, current_value in metric_pairs:
        if prior_value is not None and current_value is not None:
            deltas[f"{metric_name}_delta"] = round(float(current_value - prior_value), 6)
            deltas[f"prior_{metric_name}"] = float(prior_value)
            deltas[f"current_{metric_name}"] = float(current_value)
    deltas["sample_count_delta"] = current_summary.sample_count - prior_summary.sample_count
    deltas["resolved_count_delta"] = current_summary.resolved_count - prior_summary.resolved_count
    bin_deltas = _bin_deltas(prior_summary.bins, current_summary.bins)
    if bin_deltas:
        deltas["bin_deltas"] = cast(JsonValue, bin_deltas)
    family_deltas = (
        _single_family_deltas(prior_summary, current_summary, signal_family)
        if signal_family is not None
        else _family_deltas(prior_summary, current_summary)
    )
    if family_deltas:
        deltas["signal_family_deltas"] = cast(JsonValue, family_deltas)
    return deltas


def _bin_deltas(
    prior_bins: tuple[CalibrationBin, ...],
    current_bins: tuple[CalibrationBin, ...],
) -> list[JsonObject]:
    deltas: list[JsonObject] = []
    for prior_bin, current_bin in zip(prior_bins, current_bins, strict=True):
        item: JsonObject = {
            "bin_id": current_bin.bin_id,
            "lower_bound": current_bin.lower_bound,
            "upper_bound": current_bin.upper_bound,
            "prediction_count_delta": current_bin.prediction_count - prior_bin.prediction_count,
            "resolved_count_delta": current_bin.resolved_count - prior_bin.resolved_count,
        }
        _add_delta(item, "average_score", prior_bin.average_score, current_bin.average_score)
        _add_delta(
            item,
            "observed_frequency",
            prior_bin.observed_frequency,
            current_bin.observed_frequency,
        )
        _add_delta(item, "brier_score", prior_bin.brier_score, current_bin.brier_score)
        deltas.append(item)
    return deltas


def _single_family_deltas(
    prior_summary: CalibrationSummary,
    current_summary: CalibrationSummary,
    family: SignalArtifactFamily,
) -> list[JsonObject]:
    prior_family = _family_summary(prior_summary, family)
    current_family = _family_summary(current_summary, family)
    if prior_family is None or current_family is None:
        return []
    return [_family_delta(prior_family, current_family)]


def _family_deltas(
    prior_summary: CalibrationSummary,
    current_summary: CalibrationSummary,
) -> list[JsonObject]:
    families = tuple(
        family
        for family in dict.fromkeys(
            (
                *(item.family for item in prior_summary.signal_families),
                *(item.family for item in current_summary.signal_families),
            )
        )
        if _family_summary(prior_summary, family) is not None
        and _family_summary(current_summary, family) is not None
    )
    return [
        _family_delta(
            cast(SignalFamilyCalibrationSummary, _family_summary(prior_summary, family)),
            cast(SignalFamilyCalibrationSummary, _family_summary(current_summary, family)),
        )
        for family in families
    ]


def _family_delta(
    prior: SignalFamilyCalibrationSummary,
    current: SignalFamilyCalibrationSummary,
) -> JsonObject:
    item: JsonObject = {
        "family": current.family.value,
        "prediction_count_delta": current.prediction_count - prior.prediction_count,
        "resolved_count_delta": current.resolved_count - prior.resolved_count,
        "signal_artifact_count_delta": current.signal_artifact_count - prior.signal_artifact_count,
    }
    _add_delta(item, "average_score", prior.average_score, current.average_score)
    _add_delta(item, "observed_frequency", prior.observed_frequency, current.observed_frequency)
    _add_delta(item, "brier_score", prior.brier_score, current.brier_score)
    _add_delta(
        item,
        "score_delta_vs_baseline",
        prior.score_delta_vs_baseline,
        current.score_delta_vs_baseline,
    )
    return item


def _add_delta(
    target: JsonObject,
    metric_name: str,
    prior_value: float | None,
    current_value: float | None,
) -> None:
    if prior_value is None or current_value is None:
        return
    target[f"prior_{metric_name}"] = float(prior_value)
    target[f"current_{metric_name}"] = float(current_value)
    target[f"{metric_name}_delta"] = round(float(current_value - prior_value), 6)


def _classify_drift(
    metric_deltas: JsonObject,
    *,
    thresholds: CalibrationDriftThresholds,
) -> CalibrationDriftStatus:
    degraded = _has_degraded_movement(metric_deltas, threshold=thresholds.degraded_delta)
    improved = _has_improved_movement(metric_deltas, threshold=thresholds.improved_delta)
    if degraded and improved:
        return "inconclusive"
    if degraded:
        return "degraded"
    if improved:
        return "improved"
    if _has_watch_movement(metric_deltas, threshold=thresholds.watch_delta):
        return "watch"
    return "stable"


def _has_quality_metric_delta(metric_deltas: JsonObject) -> bool:
    return any(
        key in metric_deltas
        for key in (
            "brier_score_delta",
            "log_loss_delta",
            "accuracy_delta",
            "expected_calibration_error_delta",
        )
    )


def _has_degraded_movement(metric_deltas: JsonObject, *, threshold: float) -> bool:
    return any(
        (
            _number(metric_deltas.get("brier_score_delta")) >= threshold,
            _number(metric_deltas.get("log_loss_delta")) >= threshold,
            _number(metric_deltas.get("expected_calibration_error_delta")) >= threshold,
            _number(metric_deltas.get("accuracy_delta")) <= -threshold,
        )
    )


def _has_improved_movement(metric_deltas: JsonObject, *, threshold: float) -> bool:
    return any(
        (
            _number(metric_deltas.get("brier_score_delta")) <= -threshold,
            _number(metric_deltas.get("log_loss_delta")) <= -threshold,
            _number(metric_deltas.get("expected_calibration_error_delta")) <= -threshold,
            _number(metric_deltas.get("accuracy_delta")) >= threshold,
        )
    )


def _has_watch_movement(metric_deltas: JsonObject, *, threshold: float) -> bool:
    return any(
        abs(_number(metric_deltas.get(key))) >= threshold
        for key in (
            "brier_score_delta",
            "log_loss_delta",
            "expected_calibration_error_delta",
            "accuracy_delta",
        )
    )


def _number(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _source_membership(
    prior_summary: CalibrationSummary,
    current_summary: CalibrationSummary,
) -> JsonObject:
    prior_ids = tuple(prior_summary.source_outcome_evaluation_ids)
    current_ids = tuple(current_summary.source_outcome_evaluation_ids)
    prior_set = set(prior_ids)
    current_set = set(current_ids)
    return {
        "prior_outcome_evaluation_ids": list(prior_ids),
        "current_outcome_evaluation_ids": list(current_ids),
        "shared_outcome_evaluation_ids": [item for item in prior_ids if item in current_set],
        "prior_only_outcome_evaluation_ids": [
            item for item in prior_ids if item not in current_set
        ],
        "current_only_outcome_evaluation_ids": [
            item for item in current_ids if item not in prior_set
        ],
    }


def _bin_edges(bins: tuple[CalibrationBin, ...]) -> tuple[tuple[float, float], ...]:
    return tuple((item.lower_bound, item.upper_bound) for item in bins)


def _family_summary(
    summary: CalibrationSummary,
    family: SignalArtifactFamily,
) -> SignalFamilyCalibrationSummary | None:
    for item in summary.signal_families:
        if item.family == family:
            return item
    return None


def _drift_check_id(
    *,
    prior_summary: CalibrationSummary,
    current_summary: CalibrationSummary,
    as_of: datetime,
    signal_family: SignalArtifactFamily | None,
) -> str:
    family = "overall" if signal_family is None else signal_family.value
    material = "|".join(
        (
            prior_summary.calibration_id,
            current_summary.calibration_id,
            as_of.isoformat(),
            family,
        )
    )
    return f"calibration-drift-{slug(current_summary.cohort_id)}-{digest(material)[:12]}"


def _validate_source_payload(
    *,
    calibration_run: CalibrationRunRecord,
    payload: CalibrationSummaryArtifactPayload,
) -> None:
    if payload.run_id != calibration_run.run_id:
        raise ValueError("calibration summary payload run_id does not match storage row")
    if payload.calibration_id != calibration_run.calibration_id:
        raise ValueError("calibration summary payload calibration_id does not match storage row")
    if payload.summary.calibration_id != calibration_run.calibration_id:
        raise ValueError("calibration summary model calibration_id does not match storage row")
    if payload.as_of != calibration_run.point_in_time_cutoff:
        raise ValueError("calibration summary payload as_of does not match storage row")


def _artifact_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _threshold_metadata(thresholds: CalibrationDriftThresholds) -> JsonObject:
    return {
        "min_resolved_count": thresholds.min_resolved_count,
        "watch_delta": thresholds.watch_delta,
        "degraded_delta": thresholds.degraded_delta,
        "improved_delta": thresholds.improved_delta,
    }


__all__ = [
    "DEFAULT_CALIBRATION_DRIFT_THRESHOLDS",
    "PHASE7_CALIBRATION_DRIFT_TOOL_NAME",
    "PHASE7_CALIBRATION_DRIFT_TOOL_VERSION",
    "CalibrationDriftArtifactPayload",
    "CalibrationDriftArtifacts",
    "CalibrationDriftThresholds",
    "CalibrationSummarySource",
    "compute_calibration_drift_check",
    "load_calibration_summary_source",
    "write_calibration_drift_check_artifact",
]
