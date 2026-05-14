"""Persistence helpers for Phase 6 prediction outcomes and reviews."""

from __future__ import annotations

from datetime import datetime
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.evaluation import (
    PredictionEvaluationTarget,
    PredictionOutcome,
    PredictionOutcomeEvaluation,
)
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.storage.records import (
    OutcomeArtifactLinkRecord,
    OutcomeEvaluationArtifactLinkRecord,
    OutcomeEvaluationEvidenceLinkRecord,
    OutcomeEvidenceLinkRecord,
    PredictionOutcomeEvaluationRecord,
    PredictionOutcomeRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore


def persist_prediction_outcome_records(
    *,
    store: SQLiteStore,
    target: PredictionEvaluationTarget,
    outcome: PredictionOutcome,
    outcome_evaluation: PredictionOutcomeEvaluation,
    outcome_artifact: AuditArtifact,
    review_artifact: AuditArtifact,
    market_artifact_ids: tuple[str, ...],
    outcome_created_at: datetime,
    review_created_at: datetime,
) -> None:
    """Persist a Phase 6 outcome, review, and all source links."""

    _persist_outcome(
        store=store,
        target=target,
        outcome=outcome,
        outcome_artifact=outcome_artifact,
        market_artifact_ids=market_artifact_ids,
        created_at=outcome_created_at,
    )
    _persist_outcome_evaluation(
        store=store,
        target=target,
        outcome_evaluation=outcome_evaluation,
        review_artifact=review_artifact,
        created_at=review_created_at,
    )


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
            raise ValueError(f"outcome evidence does not exist: {reference.evidence_id}")
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
            raise ValueError(f"outcome artifact does not exist: {artifact_id}")
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
            raise ValueError(f"outcome evaluation evidence does not exist: {reference.evidence_id}")
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
            raise ValueError(f"outcome evaluation artifact does not exist: {artifact_id}")
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


__all__ = ["persist_prediction_outcome_records"]
