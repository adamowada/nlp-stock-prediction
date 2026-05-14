"""Persisted Phase 6 outcome-evaluation source loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.evaluation import (
    PredictionOutcomeEvaluation,
    PredictionOutcomeEvaluationArtifactPayload,
)
from nlp_stock_prediction.evaluation.ablation import SignalFamilyAblationInput
from nlp_stock_prediction.evaluation.calibration import CalibrationSummaryInput
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    PredictionOutcomeEvaluationRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore


@dataclass(frozen=True)
class Phase6OutcomeEvaluationSource:
    """A persisted outcome-evaluation artifact ready for Phase 6 cohort tools."""

    record: PredictionOutcomeEvaluationRecord
    artifact: ArtifactRecord
    artifact_path: Path
    payload: PredictionOutcomeEvaluationArtifactPayload

    @property
    def calibration_input(self) -> CalibrationSummaryInput:
        return CalibrationSummaryInput(
            target=self.payload.target,
            outcome_evaluation=self.payload.outcome_evaluation,
        )

    @property
    def ablation_input(self) -> SignalFamilyAblationInput:
        return SignalFamilyAblationInput(
            target=self.payload.target,
            outcome_evaluation=self.payload.outcome_evaluation,
        )

    @property
    def outcome_evaluation(self) -> PredictionOutcomeEvaluation:
        return self.payload.outcome_evaluation

    def as_summary(self) -> JsonObject:
        return {
            "outcome_evaluation_id": self.record.outcome_evaluation_id,
            "outcome_id": self.record.outcome_id,
            "candidate_id": self.record.candidate_id,
            "instrument_id": self.record.instrument_id,
            "symbol": self.record.symbol,
            "evaluated_at": self.record.evaluated_at.isoformat(),
            "status": self.record.status,
            "quality_score": self.record.quality_score,
            "artifact_id": self.artifact.artifact_id,
            "artifact_path": self.artifact_path.as_posix(),
            "target_id": self.payload.target.target_id,
        }


def load_phase6_outcome_evaluation_sources(
    *,
    store: SQLiteStore,
    repo_root: Path,
    run_id: str,
) -> tuple[Phase6OutcomeEvaluationSource, ...]:
    """Load and validate persisted outcome-evaluation payloads for Phase 6 tooling."""

    records = store.list_outcome_evaluations_for_run(run_id)
    if not records:
        raise ValueError(f"No persisted outcome evaluations were found for run: {run_id}")
    sources = tuple(
        _source_from_record(store=store, repo_root=repo_root, run_id=run_id, record=record)
        for record in records
    )
    return tuple(
        sorted(
            sources,
            key=lambda item: (
                item.outcome_evaluation.evaluated_at,
                item.outcome_evaluation.outcome_evaluation_id,
            ),
        )
    )


def _source_from_record(
    *,
    store: SQLiteStore,
    repo_root: Path,
    run_id: str,
    record: PredictionOutcomeEvaluationRecord,
) -> Phase6OutcomeEvaluationSource:
    if record.artifact_id is None:
        raise ValueError(
            "Persisted outcome evaluation is missing its full artifact payload: "
            f"{record.outcome_evaluation_id}"
        )
    artifact = store.get_artifact(record.artifact_id)
    if artifact is None:
        raise ValueError(
            "Persisted outcome evaluation references a missing artifact: "
            f"{record.outcome_evaluation_id} -> {record.artifact_id}"
        )
    artifact_path = _artifact_path(repo_root, artifact.path)
    if not artifact_path.exists():
        raise ValueError(
            "Persisted outcome-evaluation artifact file is missing: "
            f"{artifact.artifact_id} at {artifact_path.as_posix()}"
        )
    if artifact.sha256 is not None:
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if digest != artifact.sha256:
            raise ValueError(
                "Persisted outcome-evaluation artifact hash does not match storage row: "
                f"{artifact.artifact_id}"
            )
    try:
        payload = PredictionOutcomeEvaluationArtifactPayload.model_validate(
            json.loads(artifact_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise ValueError(
            "Persisted outcome-evaluation artifact is not a valid Phase 6 payload: "
            f"{artifact.artifact_id}"
        ) from exc
    _validate_source_payload(run_id=run_id, record=record, artifact=artifact, payload=payload)
    return Phase6OutcomeEvaluationSource(
        record=record,
        artifact=artifact,
        artifact_path=artifact_path,
        payload=payload,
    )


def _validate_source_payload(
    *,
    run_id: str,
    record: PredictionOutcomeEvaluationRecord,
    artifact: ArtifactRecord,
    payload: PredictionOutcomeEvaluationArtifactPayload,
) -> None:
    if artifact.artifact_type != "prediction_outcome_evaluation":
        raise ValueError(
            "Persisted outcome evaluation points to the wrong artifact type: "
            f"{record.outcome_evaluation_id} -> {artifact.artifact_type}"
        )
    outcome_evaluation = payload.outcome_evaluation
    target = payload.target
    if payload.run_id != run_id or target.run_id != run_id:
        raise ValueError(
            "Persisted outcome-evaluation artifact run_id does not match requested run: "
            f"{artifact.artifact_id}"
        )
    if outcome_evaluation.outcome_evaluation_id != record.outcome_evaluation_id:
        raise ValueError(
            "Persisted outcome-evaluation artifact ID does not match storage row: "
            f"{artifact.artifact_id}"
        )
    if outcome_evaluation.instrument_id != record.instrument_id:
        raise ValueError(
            "Persisted outcome-evaluation artifact instrument_id does not match storage row: "
            f"{artifact.artifact_id}"
        )
    if outcome_evaluation.symbol.upper() != record.symbol.upper():
        raise ValueError(
            "Persisted outcome-evaluation artifact symbol does not match storage row: "
            f"{artifact.artifact_id}"
        )
    if outcome_evaluation.evaluated_at != record.evaluated_at:
        raise ValueError(
            "Persisted outcome-evaluation artifact evaluated_at does not match storage row: "
            f"{artifact.artifact_id}"
        )
    if outcome_evaluation.status.value != record.status:
        raise ValueError(
            "Persisted outcome-evaluation artifact status does not match storage row: "
            f"{artifact.artifact_id}"
        )
    if outcome_evaluation.quality_score != record.quality_score:
        raise ValueError(
            "Persisted outcome-evaluation artifact quality_score does not match storage row: "
            f"{artifact.artifact_id}"
        )
    if outcome_evaluation.candidate_id != record.candidate_id:
        raise ValueError(
            "Persisted outcome-evaluation artifact candidate_id does not match storage row: "
            f"{artifact.artifact_id}"
        )
    if outcome_evaluation.outcome_id != record.outcome_id:
        raise ValueError(
            "Persisted outcome-evaluation artifact outcome_id does not match storage row: "
            f"{artifact.artifact_id}"
        )
    if target.candidate_id != record.candidate_id:
        raise ValueError(
            "Persisted outcome-evaluation target candidate_id does not match storage row: "
            f"{artifact.artifact_id}"
        )
    if target.instrument_id != record.instrument_id:
        raise ValueError(
            "Persisted outcome-evaluation target instrument_id does not match storage row: "
            f"{artifact.artifact_id}"
        )
    if target.symbol.upper() != record.symbol.upper():
        raise ValueError(
            "Persisted outcome-evaluation target symbol does not match storage row: "
            f"{artifact.artifact_id}"
        )


def _artifact_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


__all__ = [
    "Phase6OutcomeEvaluationSource",
    "load_phase6_outcome_evaluation_sources",
]
