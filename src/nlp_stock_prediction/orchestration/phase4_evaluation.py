"""Phase 4 prediction evaluation service adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.orchestration.phase2_evidence import source_evidence_from_record
from nlp_stock_prediction.orchestration.report_candidates import (
    prediction_candidate_from_record,
    update_candidate_record_from_contract,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore


@dataclass(frozen=True)
class Phase4PredictionEvaluationRun:
    """Summary of evaluations written for a Phase 4 run."""

    payload: JsonObject
    artifact_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def evaluate_stored_prediction_candidates(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    tool_run_id: str,
    generated_at: datetime,
) -> Phase4PredictionEvaluationRun:
    """Evaluate all stored candidates for one run using the active tool-run record."""

    from nlp_stock_prediction.evaluation import (
        attach_evaluation_metadata,
        write_prediction_evaluation_artifact,
    )

    candidate_records = store.list_prediction_candidates_for_run(run_id)
    if not candidate_records:
        return Phase4PredictionEvaluationRun(
            payload={
                "candidate_count": 0,
                "evaluation_count": 0,
                "message": "No stored prediction candidates were available to evaluate.",
            },
            artifact_ids=(),
        )

    evidence_sources = tuple(
        source_evidence_from_record(record) for record in store.list_evidence_for_run(run_id)
    )
    evaluations: list[JsonObject] = []
    artifact_ids: list[str] = []
    warnings: list[str] = []
    for candidate_record in candidate_records:
        candidate = prediction_candidate_from_record(
            candidate_record,
            evidence_sources,
            include_missing_references=True,
        )
        evaluation, artifact = write_prediction_evaluation_artifact(
            store=store,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            run_id=run_id,
            candidate=candidate,
            evidence_sources=evidence_sources,
            created_at=generated_at,
            tool_run_id=tool_run_id,
            record_tool_run=False,
        )
        evaluated_candidate = attach_evaluation_metadata(candidate, evaluation, artifact=artifact)
        store.upsert_prediction_candidate(
            update_candidate_record_from_contract(candidate_record, evaluated_candidate)
        )
        artifact_ids.append(artifact.artifact_id)
        if evaluation.evidence_counts.missing_source_references:
            warnings.append(
                f"{candidate.candidate_id}: "
                f"{evaluation.evidence_counts.missing_source_references} attribution gaps"
            )
        evaluations.append(
            {
                "candidate_id": candidate.candidate_id,
                "evaluation_id": evaluation.evaluation_id,
                "status": evaluation.status.value,
                "score": evaluation.score,
                "artifact_id": artifact.artifact_id,
                "artifact_path": artifact.path,
            }
        )

    return Phase4PredictionEvaluationRun(
        payload=cast(
            JsonObject,
            {
                "candidate_count": len(candidate_records),
                "evaluation_count": len(evaluations),
                "evaluations": evaluations,
            },
        ),
        artifact_ids=tuple(artifact_ids),
        warnings=tuple(warnings),
    )


__all__ = ["Phase4PredictionEvaluationRun", "evaluate_stored_prediction_candidates"]
