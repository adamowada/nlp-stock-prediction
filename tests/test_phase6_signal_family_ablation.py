from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    BaselineComparison,
    Direction,
    EvidenceReference,
    PredictionOutcome,
    PredictionOutcomeEvaluation,
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
    PredictionType,
    SignalArtifactFamily,
    SignalArtifactReference,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.evaluation import PredictionEvaluationTarget
from nlp_stock_prediction.evaluation import (
    SignalFamilyAblationInput,
    compute_signal_family_ablations,
    write_signal_family_ablation_artifact,
)
from nlp_stock_prediction.storage import (
    InstrumentRecord,
    PredictionCandidateRecord,
    PredictionOutcomeEvaluationRecord,
    PredictionOutcomeRecord,
    ResearchRunRecord,
    SQLiteStore,
)

RUN_ID = "run-phase6-ablation"
INSTRUMENT_ID = "instrument:equity:us:msft"
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 5, 13, 20, 0, tzinfo=UTC)
WINDOW_START = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 18, 20, 0, tzinfo=UTC)
OBSERVED_AT = datetime(2026, 5, 18, 20, 5, tzinfo=UTC)
EVALUATED_AT = datetime(2026, 5, 18, 21, 0, tzinfo=UTC)


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
            run_kind="phase6_signal_family_ablation",
            objective="Attribute point-in-time outcome quality by signal family.",
            status="running",
            started_at=NOW,
            metadata={"run_date": "2026-05-13", "symbol": "MSFT"},
        )
    )
    return store


def _persist_inputs(
    store: SQLiteStore,
    inputs: tuple[SignalFamilyAblationInput, ...],
) -> None:
    for item in inputs:
        target = item.target
        outcome_evaluation = item.outcome_evaluation
        outcome = outcome_evaluation.outcome
        store.upsert_prediction_candidate(
            PredictionCandidateRecord(
                candidate_id=target.candidate_id,
                run_id=RUN_ID,
                instrument_id=target.instrument_id,
                prediction_horizon=target.horizon.value,
                prediction_type=target.prediction_type.value,
                scenario=str(target.candidate_snapshot["scenario"]),
                direction=target.direction.value if target.direction else None,
                confidence=(
                    target.baseline_comparison.candidate_score
                    if target.baseline_comparison
                    else None
                ),
                status="evidence_supported",
                evidence_for=target.evidence_ids,
                signal_artifacts=tuple(
                    signal_artifact.artifact_id for signal_artifact in target.signal_artifacts
                ),
                baseline=cast(
                    JsonObject,
                    target.baseline_comparison.model_dump(mode="json")
                    if target.baseline_comparison
                    else {},
                ),
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
                metadata={"target_id": target.target_id},
            )
        )
        store.upsert_prediction_outcome_evaluation(
            PredictionOutcomeEvaluationRecord(
                outcome_evaluation_id=outcome_evaluation.outcome_evaluation_id,
                run_id=RUN_ID,
                outcome_id=outcome_evaluation.outcome_id,
                candidate_id=outcome_evaluation.candidate_id,
                instrument_id=outcome_evaluation.instrument_id,
                symbol=outcome_evaluation.symbol,
                evaluated_at=outcome_evaluation.evaluated_at,
                status=outcome_evaluation.status.value,
                quality_score=outcome_evaluation.quality_score,
                baseline_comparison=cast(
                    JsonObject,
                    outcome_evaluation.baseline_comparison.model_dump(mode="json")
                    if outcome_evaluation.baseline_comparison
                    else {},
                ),
                artifact_id=None,
                limitations=outcome_evaluation.limitations,
                metadata={"target_id": target.target_id},
            )
        )


def _baseline(candidate_score: float) -> BaselineComparison:
    delta = round(candidate_score - 0.5, 6)
    return BaselineComparison(
        baseline_id="no_directional_edge",
        baseline_summary="No directional edge without source-backed evidence.",
        baseline_score=0.5,
        candidate_score=candidate_score,
        score_delta=delta,
        verdict="above_baseline" if delta > 0.05 else "near_baseline",
    )


def _target(
    candidate_id: str,
    *,
    confidence: float = 0.64,
    signal_families: tuple[SignalArtifactFamily, ...] = (),
) -> PredictionEvaluationTarget:
    signal_artifacts = tuple(
        SignalArtifactReference(
            artifact_id=f"artifact-{candidate_id}-{family.value}",
            family=family,
            artifact_type="technical_package",
            schema_version="technical-package.v1",
            tool_run_id=f"tool-{candidate_id}-{family.value}",
            produced_by="phase4_technical_package",
            created_at=CUTOFF,
            as_of=CUTOFF,
            sha256="a" * 64,
            metadata={"signal_family": family.value},
        )
        for family in signal_families
    )
    return PredictionEvaluationTarget(
        target_id=f"target-{candidate_id}",
        run_id=RUN_ID,
        candidate_id=candidate_id,
        instrument_id=INSTRUMENT_ID,
        symbol="MSFT",
        prediction_type=PredictionType.DIRECTIONAL,
        horizon=TimeHorizon.SWING,
        direction=Direction.BULLISH,
        prediction_created_at=NOW,
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        candidate_snapshot={
            "candidate_id": candidate_id,
            "scenario": "Source-backed monitored directional scenario.",
            "confidence": confidence,
            "signal_artifact_ids": [item.artifact_id for item in signal_artifacts],
        },
        baseline_comparison=_baseline(confidence),
        evidence_ids=(f"evidence-{candidate_id}-support",),
        signal_artifacts=signal_artifacts,
    )


def _resolved_outcome_evaluation(
    target: PredictionEvaluationTarget,
    *,
    status: PredictionOutcomeEvaluationStatus,
    quality_score: float,
) -> PredictionOutcomeEvaluation:
    outcome = PredictionOutcome(
        outcome_id=f"outcome-{target.candidate_id}",
        candidate_id=target.candidate_id,
        instrument_id=target.instrument_id,
        symbol=target.symbol,
        prediction_type=target.prediction_type,
        horizon=target.horizon,
        evaluation_window_start=target.evaluation_window_start,
        evaluation_window_end=target.evaluation_window_end,
        status=PredictionOutcomeStatus.OBSERVED,
        observed_result=(
            PredictionOutcomeResult.SUPPORTED
            if quality_score >= 0.5
            else PredictionOutcomeResult.CONTRADICTED
        ),
        observed_at=OBSERVED_AT,
        result_summary="Outcome was reviewed against the frozen prediction target.",
        outcome_evidence=(
            EvidenceReference(evidence_id=f"evidence-{target.candidate_id}-outcome"),
        ),
        artifact_ids=(f"artifact-{target.candidate_id}-outcome",),
    )
    return PredictionOutcomeEvaluation(
        outcome_evaluation_id=f"outcome-evaluation-{target.candidate_id}",
        outcome_id=outcome.outcome_id,
        candidate_id=target.candidate_id,
        instrument_id=target.instrument_id,
        symbol=target.symbol,
        evaluated_at=EVALUATED_AT,
        status=status,
        outcome=outcome,
        quality_score=quality_score,
        baseline_comparison=target.baseline_comparison,
        evidence=(EvidenceReference(evidence_id=f"evidence-{target.candidate_id}-outcome"),),
        artifact_ids=(f"artifact-{target.candidate_id}-outcome-review",),
    )


def _pending_outcome_evaluation(
    target: PredictionEvaluationTarget,
) -> PredictionOutcomeEvaluation:
    outcome = PredictionOutcome(
        outcome_id=f"outcome-{target.candidate_id}",
        candidate_id=target.candidate_id,
        instrument_id=target.instrument_id,
        symbol=target.symbol,
        prediction_type=target.prediction_type,
        horizon=target.horizon,
        evaluation_window_start=target.evaluation_window_start,
        evaluation_window_end=target.evaluation_window_end,
        status=PredictionOutcomeStatus.PENDING,
        limitations=("Evaluation window has not resolved with attributable provider data.",),
    )
    return PredictionOutcomeEvaluation(
        outcome_evaluation_id=f"outcome-evaluation-{target.candidate_id}",
        outcome_id=outcome.outcome_id,
        candidate_id=target.candidate_id,
        instrument_id=target.instrument_id,
        symbol=target.symbol,
        evaluated_at=datetime(2026, 5, 15, 20, 0, tzinfo=UTC),
        status=PredictionOutcomeEvaluationStatus.PENDING,
        outcome=outcome,
        limitations=("Evaluation window has not resolved with attributable provider data.",),
    )


def _input(
    candidate_id: str,
    *,
    signal_families: tuple[SignalArtifactFamily, ...] = (),
    status: PredictionOutcomeEvaluationStatus = PredictionOutcomeEvaluationStatus.CONFIRMED,
    quality_score: float = 1.0,
) -> SignalFamilyAblationInput:
    target = _target(candidate_id, signal_families=signal_families)
    return SignalFamilyAblationInput(
        target=target,
        outcome_evaluation=_resolved_outcome_evaluation(
            target,
            status=status,
            quality_score=quality_score,
        ),
    )


@pytest.mark.unit
def test_signal_family_ablation_computes_included_vs_excluded_quality() -> None:
    inputs = (
        _input("candidate-with-technicals", signal_families=(SignalArtifactFamily.TECHNICALS,)),
        _input(
            "candidate-without-technicals",
            status=PredictionOutcomeEvaluationStatus.MISSED,
            quality_score=0.0,
        ),
    )

    (ablation,) = compute_signal_family_ablations(
        cohort_id="phase6-evalcal-stage4-msft-swing",
        created_at=EVALUATED_AT,
        inputs=inputs,
        families=(SignalArtifactFamily.TECHNICALS, SignalArtifactFamily.TECHNICALS),
    )

    assert ablation.family == SignalArtifactFamily.TECHNICALS
    assert ablation.included_prediction_count == 1
    assert ablation.excluded_prediction_count == 1
    assert ablation.resolved_included_count == 1
    assert ablation.resolved_excluded_count == 1
    assert ablation.included_quality_score == pytest.approx(1.0)
    assert ablation.excluded_quality_score == pytest.approx(0.0)
    assert ablation.quality_score_delta == pytest.approx(1.0)
    assert ablation.baseline_comparison is not None
    assert ablation.baseline_comparison.baseline_id == "without_technicals_signals"
    assert ablation.baseline_comparison.verdict == "above_baseline"
    assert ablation.limitations == ()
    assert ablation.metadata["included_signal_artifact_ids"] == [
        "artifact-candidate-with-technicals-technicals"
    ]


@pytest.mark.unit
def test_signal_family_ablation_writes_artifact_and_calibration_slices(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    inputs = (
        _input("candidate-with-technicals", signal_families=(SignalArtifactFamily.TECHNICALS,)),
        _input(
            "candidate-without-technicals",
            status=PredictionOutcomeEvaluationStatus.MISSED,
            quality_score=0.0,
        ),
    )
    _persist_inputs(store, inputs)

    result = write_signal_family_ablation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        cohort_id="phase6-evalcal-stage4-msft-swing",
        inputs=inputs,
        point_in_time_cutoff=EVALUATED_AT,
        created_at=EVALUATED_AT,
        families=(SignalArtifactFamily.TECHNICALS,),
    )

    assert result.artifact.artifact_type == "signal_family_ablation"
    assert result.calibration_run.artifact_id == result.artifact.artifact_id
    assert store.get_tool_run(result.tool_run_id) is not None
    assert store.get_calibration_run(result.calibration_id) == result.calibration_run
    assert store.list_calibration_runs_for_run(RUN_ID) == (result.calibration_run,)
    assert store.list_calibration_slices(result.calibration_id) == result.calibration_slices

    (calibration_slice,) = result.calibration_slices
    assert calibration_slice.signal_family == "technicals"
    assert calibration_slice.sample_count == 2
    assert calibration_slice.resolved_count == 2
    assert calibration_slice.metrics["quality_score_delta"] == 1.0
    assert calibration_slice.baseline_comparison["baseline_id"] == "without_technicals_signals"
    assert calibration_slice.provenance["included_outcome_evaluation_ids"] == [
        "outcome-evaluation-candidate-with-technicals"
    ]

    payload = json.loads(Path(result.artifact.path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "signal-family-ablation-artifact.v1"
    assert payload["source_outcome_evaluation_ids"] == [
        "outcome-evaluation-candidate-with-technicals",
        "outcome-evaluation-candidate-without-technicals",
    ]
    assert set(payload["source_artifact_ids"]) >= {
        "artifact-candidate-with-technicals-technicals",
        "artifact-candidate-with-technicals-outcome",
        "artifact-candidate-with-technicals-outcome-review",
    }
    assert payload["ablations"][0]["family"] == "technicals"
    assert len(payload["ablations"]) == 1


@pytest.mark.unit
def test_signal_family_ablation_keeps_unresolved_samples_metric_free(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _target(
        "candidate-pending-technicals",
        signal_families=(SignalArtifactFamily.TECHNICALS,),
    )
    inputs = (
        SignalFamilyAblationInput(
            target=target,
            outcome_evaluation=_pending_outcome_evaluation(target),
        ),
    )
    _persist_inputs(store, inputs)

    result = write_signal_family_ablation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        cohort_id="phase6-evalcal-stage4-pending",
        inputs=inputs,
        point_in_time_cutoff=datetime(2026, 5, 15, 20, 0, tzinfo=UTC),
        created_at=datetime(2026, 5, 15, 20, 0, tzinfo=UTC),
        families=(SignalArtifactFamily.TECHNICALS,),
    )

    (ablation,) = result.ablations
    assert ablation.included_prediction_count == 1
    assert ablation.excluded_prediction_count == 0
    assert ablation.resolved_included_count == 0
    assert ablation.included_quality_score is None
    assert ablation.quality_score_delta is None
    assert ablation.limitations

    (calibration_slice,) = result.calibration_slices
    assert calibration_slice.sample_count == 1
    assert calibration_slice.resolved_count == 0
    assert calibration_slice.pending_count == 1
    assert calibration_slice.metrics["included_quality_score"] is None


@pytest.mark.unit
def test_signal_family_ablation_excludes_outcomes_after_cutoff(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inputs = (
        _input("candidate-with-technicals", signal_families=(SignalArtifactFamily.TECHNICALS,)),
        _input(
            "candidate-without-technicals",
            status=PredictionOutcomeEvaluationStatus.MISSED,
            quality_score=0.0,
        ),
    )

    result = write_signal_family_ablation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        cohort_id="phase6-evalcal-stage4-cutoff",
        inputs=inputs,
        point_in_time_cutoff=CUTOFF,
        created_at=EVALUATED_AT,
        families=(SignalArtifactFamily.TECHNICALS,),
    )

    assert result.artifact_payload.source_outcome_evaluation_ids == ()
    assert result.artifact_payload.excluded_outcome_evaluation_ids == (
        "outcome-evaluation-candidate-with-technicals",
        "outcome-evaluation-candidate-without-technicals",
    )
    assert result.calibration_slices[0].sample_count == 0
    payload = json.loads(Path(result.artifact.path).read_text(encoding="utf-8"))
    assert payload["excluded_outcome_evaluation_ids"] == [
        "outcome-evaluation-candidate-with-technicals",
        "outcome-evaluation-candidate-without-technicals",
    ]
    assert any("point-in-time cutoff" in item for item in payload["limitations"])


@pytest.mark.unit
def test_signal_family_ablation_without_tool_run_leaves_artifact_unlinked(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    inputs = (
        _input("candidate-with-technicals", signal_families=(SignalArtifactFamily.TECHNICALS,)),
        _input(
            "candidate-without-technicals",
            status=PredictionOutcomeEvaluationStatus.MISSED,
            quality_score=0.0,
        ),
    )
    _persist_inputs(store, inputs)

    result = write_signal_family_ablation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        cohort_id="phase6-evalcal-stage4-no-tool-run",
        inputs=inputs,
        point_in_time_cutoff=EVALUATED_AT,
        created_at=EVALUATED_AT,
        families=(SignalArtifactFamily.TECHNICALS,),
        record_tool_run=False,
    )

    assert store.get_tool_run(result.tool_run_id) is None
    stored_artifact = store.get_artifact(result.artifact.artifact_id)
    assert stored_artifact is not None
    assert stored_artifact.tool_run_id is None
    assert result.calibration_run.tool_run_id is None


@pytest.mark.unit
def test_signal_family_ablation_input_rejects_target_outcome_mismatch() -> None:
    target = _target("candidate-original", signal_families=(SignalArtifactFamily.TECHNICALS,))
    other_target = _target("candidate-other")

    with pytest.raises(ValueError, match="candidate_id must match"):
        SignalFamilyAblationInput(
            target=target,
            outcome_evaluation=_resolved_outcome_evaluation(
                other_target,
                status=PredictionOutcomeEvaluationStatus.CONFIRMED,
                quality_score=1.0,
            ),
        )
