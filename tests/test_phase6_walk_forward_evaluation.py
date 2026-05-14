from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    Direction,
    EvidenceReference,
    PredictionOutcome,
    PredictionOutcomeEvaluation,
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
    PredictionType,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.evaluation import (
    compute_walk_forward_folds,
    write_walk_forward_evaluation_artifact,
)
from nlp_stock_prediction.storage import (
    InstrumentRecord,
    PredictionCandidateRecord,
    PredictionOutcomeEvaluationRecord,
    PredictionOutcomeRecord,
    ResearchRunRecord,
    SQLiteStore,
)

RUN_ID = "run-phase6-walk-forward"
INSTRUMENT_ID = "instrument:equity:us:msft"
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
WINDOW_START = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 18, 20, 0, tzinfo=UTC)
FOLD_CREATED_AT = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 5, 21, 23, 0, tzinfo=UTC)


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "data" / "prediction-research.sqlite3")
    store.initialize()
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id=INSTRUMENT_ID,
            symbol="MSFT",
            asset_class="stock",
            name="Microsoft Corporation",
        )
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=RUN_ID,
            run_kind="phase6_walk_forward_evaluation",
            objective="Evaluate chronological generalization of prediction quality.",
            status="running",
            started_at=NOW,
            metadata={"run_date": "2026-05-13", "symbol": "MSFT"},
        )
    )
    return store


def _persist_evaluations(
    store: SQLiteStore,
    evaluations: tuple[PredictionOutcomeEvaluation, ...],
) -> None:
    for evaluation in evaluations:
        outcome = evaluation.outcome
        store.upsert_prediction_candidate(
            PredictionCandidateRecord(
                candidate_id=evaluation.candidate_id,
                run_id=RUN_ID,
                instrument_id=evaluation.instrument_id,
                prediction_horizon=outcome.horizon.value,
                prediction_type=outcome.prediction_type.value,
                scenario="Source-backed monitored directional scenario.",
                direction=Direction.BULLISH.value,
                confidence=None,
                status="evidence_supported",
                metadata={"symbol": evaluation.symbol},
            )
        )
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
                observed_result=(
                    outcome.observed_result.value if outcome.observed_result else None
                ),
                observed_at=outcome.observed_at,
                result_summary=outcome.result_summary,
                result_value=outcome.result_value,
                baseline_value=outcome.baseline_value,
                limitations=outcome.limitations,
            )
        )
        store.upsert_prediction_outcome_evaluation(
            PredictionOutcomeEvaluationRecord(
                outcome_evaluation_id=evaluation.outcome_evaluation_id,
                run_id=RUN_ID,
                outcome_id=evaluation.outcome_id,
                candidate_id=evaluation.candidate_id,
                instrument_id=evaluation.instrument_id,
                symbol=evaluation.symbol,
                evaluated_at=evaluation.evaluated_at,
                status=evaluation.status.value,
                quality_score=evaluation.quality_score,
                baseline_comparison=cast(
                    JsonObject,
                    evaluation.baseline_comparison.model_dump(mode="json")
                    if evaluation.baseline_comparison
                    else {},
                ),
                artifact_id=None,
                limitations=evaluation.limitations,
                metadata=evaluation.metadata,
            )
        )


def _resolved_evaluation(
    candidate_id: str,
    *,
    evaluated_at: datetime,
    quality_score: float,
    status: PredictionOutcomeEvaluationStatus = PredictionOutcomeEvaluationStatus.CONFIRMED,
) -> PredictionOutcomeEvaluation:
    outcome = PredictionOutcome(
        outcome_id=f"outcome-{candidate_id}",
        candidate_id=candidate_id,
        instrument_id=INSTRUMENT_ID,
        symbol="MSFT",
        prediction_type=PredictionType.DIRECTIONAL,
        horizon=TimeHorizon.SWING,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        status=PredictionOutcomeStatus.OBSERVED,
        observed_result=(
            PredictionOutcomeResult.MIXED
            if status == PredictionOutcomeEvaluationStatus.MIXED
            else PredictionOutcomeResult.SUPPORTED
            if quality_score >= 0.5
            else PredictionOutcomeResult.CONTRADICTED
        ),
        observed_at=evaluated_at - timedelta(minutes=20),
        result_summary="Outcome was reviewed against the frozen prediction target.",
        outcome_evidence=(EvidenceReference(evidence_id=f"evidence-{candidate_id}-outcome"),),
        artifact_ids=(f"artifact-{candidate_id}-outcome",),
    )
    return PredictionOutcomeEvaluation(
        outcome_evaluation_id=f"outcome-evaluation-{candidate_id}",
        outcome_id=outcome.outcome_id,
        candidate_id=candidate_id,
        instrument_id=INSTRUMENT_ID,
        symbol="MSFT",
        evaluated_at=evaluated_at,
        status=status,
        outcome=outcome,
        quality_score=quality_score,
        evidence=(EvidenceReference(evidence_id=f"evidence-{candidate_id}-outcome"),),
        artifact_ids=(f"artifact-{candidate_id}-outcome-review",),
        metadata={"direction": Direction.BULLISH.value},
    )


def _pending_evaluation(
    candidate_id: str,
    *,
    evaluated_at: datetime,
) -> PredictionOutcomeEvaluation:
    outcome = PredictionOutcome(
        outcome_id=f"outcome-{candidate_id}",
        candidate_id=candidate_id,
        instrument_id=INSTRUMENT_ID,
        symbol="MSFT",
        prediction_type=PredictionType.DIRECTIONAL,
        horizon=TimeHorizon.SWING,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        status=PredictionOutcomeStatus.PENDING,
        limitations=("Held-out provider data has not resolved yet.",),
    )
    return PredictionOutcomeEvaluation(
        outcome_evaluation_id=f"outcome-evaluation-{candidate_id}",
        outcome_id=outcome.outcome_id,
        candidate_id=candidate_id,
        instrument_id=INSTRUMENT_ID,
        symbol="MSFT",
        evaluated_at=evaluated_at,
        status=PredictionOutcomeEvaluationStatus.PENDING,
        outcome=outcome,
        limitations=("Held-out provider data has not resolved yet.",),
    )


def _resolved_evaluations() -> tuple[PredictionOutcomeEvaluation, ...]:
    return (
        _resolved_evaluation(
            "candidate-003",
            evaluated_at=datetime(2026, 5, 20, 21, 0, tzinfo=UTC),
            quality_score=1.0,
        ),
        _resolved_evaluation(
            "candidate-001",
            evaluated_at=datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
            quality_score=1.0,
        ),
        _resolved_evaluation(
            "candidate-004",
            evaluated_at=datetime(2026, 5, 21, 21, 0, tzinfo=UTC),
            quality_score=0.5,
            status=PredictionOutcomeEvaluationStatus.MIXED,
        ),
        _resolved_evaluation(
            "candidate-002",
            evaluated_at=datetime(2026, 5, 19, 21, 0, tzinfo=UTC),
            quality_score=0.0,
            status=PredictionOutcomeEvaluationStatus.MISSED,
        ),
    )


@pytest.mark.unit
def test_walk_forward_evaluation_computes_chronological_train_test_folds() -> None:
    folds = compute_walk_forward_folds(
        cohort_id="phase6-evalcal-stage5-msft-swing",
        created_at=FOLD_CREATED_AT,
        outcome_evaluations=_resolved_evaluations(),
        minimum_train_size=2,
        test_size=1,
        step_size=1,
    )

    assert tuple(fold.fold_id for fold in folds) == (
        "walk-forward-phase6-evalcal-stage5-msft-swing-fold-001",
        "walk-forward-phase6-evalcal-stage5-msft-swing-fold-002",
    )
    first, second = folds
    assert first.source_train_outcome_evaluation_ids == (
        "outcome-evaluation-candidate-001",
        "outcome-evaluation-candidate-002",
    )
    assert first.source_test_outcome_evaluation_ids == ("outcome-evaluation-candidate-003",)
    assert first.train_quality_score == pytest.approx(0.5)
    assert first.test_quality_score == pytest.approx(1.0)
    assert first.generalization_delta == pytest.approx(0.5)
    assert second.train_quality_score == pytest.approx(0.666667)
    assert second.test_quality_score == pytest.approx(0.5)
    assert second.generalization_delta == pytest.approx(-0.166667)


@pytest.mark.unit
def test_walk_forward_evaluation_writes_artifact_and_calibration_slices(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    future = _resolved_evaluation(
        "candidate-after-cutoff",
        evaluated_at=datetime(2026, 5, 22, 21, 0, tzinfo=UTC),
        quality_score=1.0,
    )
    evaluations = (*_resolved_evaluations(), future)
    _persist_evaluations(store, evaluations)

    result = write_walk_forward_evaluation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        cohort_id="phase6-evalcal-stage5-msft-swing",
        outcome_evaluations=evaluations,
        point_in_time_cutoff=CUTOFF,
        minimum_train_size=2,
        test_size=1,
        step_size=1,
        created_at=FOLD_CREATED_AT,
    )

    assert result.artifact.artifact_type == "walk_forward_evaluation"
    assert result.calibration_run.artifact_id == result.artifact.artifact_id
    assert result.artifact_payload.excluded_outcome_evaluation_ids == (
        "outcome-evaluation-candidate-after-cutoff",
    )
    assert result.artifact_payload.sample_count == 4
    assert result.artifact_payload.resolved_count == 4
    assert store.get_calibration_run(result.calibration_id) == result.calibration_run
    assert store.list_calibration_slices(result.calibration_id) == result.calibration_slices

    assert len(result.calibration_slices) == 2
    first_slice = result.calibration_slices[0]
    assert first_slice.cohort_label.endswith("fold-001")
    assert first_slice.sample_count == 1
    assert first_slice.resolved_count == 1
    assert first_slice.metrics["train_quality_score"] == 0.5
    assert first_slice.metrics["test_quality_score"] == 1.0
    assert first_slice.metrics["generalization_delta"] == 0.5
    assert first_slice.provenance["source_test_outcome_evaluation_ids"] == [
        "outcome-evaluation-candidate-003"
    ]

    payload = json.loads(Path(result.artifact.path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "walk-forward-evaluation-artifact.v1"
    assert payload["folds"][0]["test_quality_score"] == 1.0
    assert payload["excluded_outcome_evaluation_ids"] == [
        "outcome-evaluation-candidate-after-cutoff"
    ]
    assert any("point-in-time cutoff" in item for item in payload["limitations"])


@pytest.mark.unit
def test_walk_forward_evaluation_marks_unresolved_held_out_fold_metric_free() -> None:
    evaluations = (
        _resolved_evaluation(
            "candidate-001",
            evaluated_at=datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
            quality_score=1.0,
        ),
        _resolved_evaluation(
            "candidate-002",
            evaluated_at=datetime(2026, 5, 19, 21, 0, tzinfo=UTC),
            quality_score=0.0,
            status=PredictionOutcomeEvaluationStatus.MISSED,
        ),
        _pending_evaluation(
            "candidate-003",
            evaluated_at=datetime(2026, 5, 20, 21, 0, tzinfo=UTC),
        ),
    )

    (fold,) = compute_walk_forward_folds(
        cohort_id="phase6-evalcal-stage5-pending",
        created_at=FOLD_CREATED_AT,
        outcome_evaluations=evaluations,
        minimum_train_size=2,
        test_size=1,
    )

    assert fold.train_quality_score == pytest.approx(0.5)
    assert fold.test_quality_score is None
    assert fold.generalization_delta is None
    assert fold.test_pending_count == 1
    assert fold.limitations == ("Walk-forward fold has no resolved held-out outcomes.",)


@pytest.mark.unit
def test_walk_forward_evaluation_persists_limitations_without_enough_history(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    evaluations = (
        _resolved_evaluation(
            "candidate-001",
            evaluated_at=datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
            quality_score=1.0,
        ),
    )
    _persist_evaluations(store, evaluations)

    result = write_walk_forward_evaluation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        cohort_id="phase6-evalcal-stage5-short-history",
        outcome_evaluations=evaluations,
        point_in_time_cutoff=CUTOFF,
        minimum_train_size=2,
        test_size=1,
        created_at=FOLD_CREATED_AT,
    )

    assert result.folds == ()
    assert result.calibration_slices == ()
    assert result.artifact.record_count == 0
    assert result.artifact_payload.sample_count == 1
    assert any("Insufficient chronological" in item for item in result.artifact_payload.limitations)
    assert store.list_calibration_slices(result.calibration_id) == ()


@pytest.mark.unit
def test_walk_forward_evaluation_rejects_invalid_window_parameters() -> None:
    with pytest.raises(ValueError, match="minimum_train_size"):
        compute_walk_forward_folds(
            cohort_id="phase6-evalcal-stage5-invalid",
            created_at=FOLD_CREATED_AT,
            outcome_evaluations=_resolved_evaluations(),
            minimum_train_size=0,
        )
