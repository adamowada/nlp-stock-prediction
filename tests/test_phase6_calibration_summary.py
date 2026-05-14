from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
from nlp_stock_prediction.contracts.evaluation import PredictionEvaluationTarget
from nlp_stock_prediction.evaluation import (
    CalibrationSummaryInput,
    compute_calibration_summary,
    write_calibration_summary_artifact,
)
from nlp_stock_prediction.storage import ResearchRunRecord, SQLiteStore

RUN_ID = "run-phase6-calibration"
INSTRUMENT_ID = "instrument:equity:us:msft"
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
WINDOW_START = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 18, 20, 0, tzinfo=UTC)
SUMMARY_CREATED_AT = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
AS_OF = datetime(2026, 5, 22, 0, 0, tzinfo=UTC)
BIN_EDGES = (0.0, 0.5, 1.0)


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "data" / "prediction-research.sqlite3")
    store.initialize()
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=RUN_ID,
            run_kind="phase6_calibration_summary",
            objective="Summarize prediction-quality calibration for a cohort.",
            status="running",
            started_at=NOW,
            metadata={"run_date": "2026-05-13", "symbol": "MSFT"},
        )
    )
    return store


def _baseline(score: float) -> BaselineComparison:
    delta = round(score - 0.5, 6)
    if delta > 0.05:
        verdict = "above_baseline"
    elif delta < -0.05:
        verdict = "below_baseline"
    else:
        verdict = "near_baseline"
    return BaselineComparison(
        baseline_id="no_directional_edge",
        baseline_summary="No directional edge without source-backed evidence.",
        baseline_score=0.5,
        candidate_score=score,
        score_delta=delta,
        verdict=verdict,
    )


def _artifact_type(family: SignalArtifactFamily) -> str:
    if family == SignalArtifactFamily.TECHNICALS:
        return "technical_package"
    if family == SignalArtifactFamily.NEWS:
        return "normalized_evidence"
    return "analysis_context"


def _target(
    candidate_id: str,
    *,
    score: float,
    families: tuple[SignalArtifactFamily, ...] = (),
) -> PredictionEvaluationTarget:
    signal_artifacts = tuple(
        SignalArtifactReference(
            artifact_id=f"artifact-{candidate_id}-{family.value}",
            family=family,
            artifact_type=_artifact_type(family),
            schema_version=f"{_artifact_type(family)}.v1",
            tool_run_id=f"tool-{candidate_id}-{family.value}",
            produced_by=f"phase4_{family.value}",
            created_at=NOW,
            as_of=NOW,
            sha256="a" * 64,
            metadata={"signal_family": family.value},
        )
        for family in families
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
        point_in_time_cutoff=NOW,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        candidate_snapshot={
            "candidate_id": candidate_id,
            "scenario": "Source-backed monitored directional scenario.",
            "confidence": score,
            "signal_artifact_ids": [item.artifact_id for item in signal_artifacts],
        },
        baseline_comparison=_baseline(score),
        evidence_ids=(f"evidence-{candidate_id}-support",),
        signal_artifacts=signal_artifacts,
    )


def _resolved_input(
    candidate_id: str,
    *,
    score: float,
    quality_score: float,
    evaluated_at: datetime,
    families: tuple[SignalArtifactFamily, ...] = (),
) -> CalibrationSummaryInput:
    target = _target(candidate_id, score=score, families=families)
    status = (
        PredictionOutcomeEvaluationStatus.CONFIRMED
        if quality_score >= 0.5
        else PredictionOutcomeEvaluationStatus.MISSED
    )
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
            PredictionOutcomeResult.SUPPORTED
            if quality_score >= 0.5
            else PredictionOutcomeResult.CONTRADICTED
        ),
        observed_at=evaluated_at - timedelta(minutes=20),
        result_summary="Outcome was reviewed against the frozen prediction target.",
        outcome_evidence=(EvidenceReference(evidence_id=f"evidence-{candidate_id}-outcome"),),
        artifact_ids=(f"artifact-{candidate_id}-outcome",),
    )
    return CalibrationSummaryInput(
        target=target,
        outcome_evaluation=PredictionOutcomeEvaluation(
            outcome_evaluation_id=f"outcome-evaluation-{candidate_id}",
            outcome_id=outcome.outcome_id,
            candidate_id=candidate_id,
            instrument_id=INSTRUMENT_ID,
            symbol="MSFT",
            evaluated_at=evaluated_at,
            status=status,
            outcome=outcome,
            quality_score=quality_score,
            baseline_comparison=target.baseline_comparison,
            evidence=(EvidenceReference(evidence_id=f"evidence-{candidate_id}-outcome"),),
            artifact_ids=(f"artifact-{candidate_id}-outcome-review",),
        ),
    )


def _pending_input(candidate_id: str, *, score: float) -> CalibrationSummaryInput:
    target = _target(candidate_id, score=score, families=(SignalArtifactFamily.TECHNICALS,))
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
        limitations=("Outcome provider data has not resolved yet.",),
    )
    return CalibrationSummaryInput(
        target=target,
        outcome_evaluation=PredictionOutcomeEvaluation(
            outcome_evaluation_id=f"outcome-evaluation-{candidate_id}",
            outcome_id=outcome.outcome_id,
            candidate_id=candidate_id,
            instrument_id=INSTRUMENT_ID,
            symbol="MSFT",
            evaluated_at=datetime(2026, 5, 20, 21, 0, tzinfo=UTC),
            status=PredictionOutcomeEvaluationStatus.PENDING,
            outcome=outcome,
            limitations=("Outcome provider data has not resolved yet.",),
        ),
    )


def _calibration_inputs() -> tuple[CalibrationSummaryInput, ...]:
    return (
        _resolved_input(
            "candidate-low-1",
            score=0.2,
            quality_score=0.0,
            evaluated_at=datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
            families=(SignalArtifactFamily.TECHNICALS,),
        ),
        _resolved_input(
            "candidate-low-2",
            score=0.4,
            quality_score=0.0,
            evaluated_at=datetime(2026, 5, 19, 21, 0, tzinfo=UTC),
        ),
        _resolved_input(
            "candidate-high-news",
            score=0.7,
            quality_score=1.0,
            evaluated_at=datetime(2026, 5, 20, 21, 0, tzinfo=UTC),
            families=(SignalArtifactFamily.NEWS,),
        ),
        _resolved_input(
            "candidate-high-tech",
            score=0.8,
            quality_score=1.0,
            evaluated_at=datetime(2026, 5, 21, 21, 0, tzinfo=UTC),
            families=(SignalArtifactFamily.TECHNICALS,),
        ),
    )


@pytest.mark.unit
def test_calibration_summary_computes_reliability_bins_and_metrics() -> None:
    summary = compute_calibration_summary(
        calibration_id="calibration-summary-stage6-msft",
        cohort_id="phase6-evalcal-stage6-msft-swing",
        created_at=SUMMARY_CREATED_AT,
        as_of=AS_OF,
        inputs=_calibration_inputs(),
        bin_edges=BIN_EDGES,
        families=(SignalArtifactFamily.TECHNICALS, SignalArtifactFamily.NEWS),
    )

    assert summary.sample_count == 4
    assert summary.resolved_count == 4
    assert summary.brier_score == pytest.approx(0.0825)
    assert summary.accuracy == pytest.approx(1.0)
    assert summary.expected_calibration_error == pytest.approx(0.275)
    assert summary.baseline_comparison is not None
    assert summary.baseline_comparison.baseline_id == "no_skill_constant_0_5"
    assert summary.baseline_comparison.candidate_score == pytest.approx(0.9175)
    assert summary.baseline_comparison.baseline_score == pytest.approx(0.75)
    assert summary.baseline_comparison.verdict == "above_baseline"

    low_bin, high_bin = summary.bins
    assert low_bin.prediction_count == 2
    assert low_bin.average_score == pytest.approx(0.3)
    assert low_bin.observed_frequency == pytest.approx(0.0)
    assert low_bin.brier_score == pytest.approx(0.1)
    assert high_bin.prediction_count == 2
    assert high_bin.average_score == pytest.approx(0.75)
    assert high_bin.observed_frequency == pytest.approx(1.0)
    assert high_bin.brier_score == pytest.approx(0.065)

    technicals, news = summary.signal_families
    assert technicals.family == SignalArtifactFamily.TECHNICALS
    assert technicals.prediction_count == 2
    assert technicals.signal_artifact_count == 2
    assert technicals.brier_score == pytest.approx(0.04)
    assert technicals.score_delta_vs_baseline == pytest.approx(0.0)
    assert news.family == SignalArtifactFamily.NEWS
    assert news.prediction_count == 1
    assert news.observed_frequency == pytest.approx(1.0)
    assert news.score_delta_vs_baseline == pytest.approx(0.5)


@pytest.mark.unit
def test_calibration_summary_writes_artifact_and_calibration_slices(tmp_path: Path) -> None:
    store = _store(tmp_path)
    future = _resolved_input(
        "candidate-after-as-of",
        score=0.9,
        quality_score=1.0,
        evaluated_at=datetime(2026, 5, 22, 21, 0, tzinfo=UTC),
    )

    result = write_calibration_summary_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        cohort_id="phase6-evalcal-stage6-msft-swing",
        inputs=(*_calibration_inputs(), future),
        as_of=AS_OF,
        created_at=SUMMARY_CREATED_AT,
        bin_edges=BIN_EDGES,
        families=(SignalArtifactFamily.TECHNICALS, SignalArtifactFamily.NEWS),
    )

    assert result.artifact.artifact_type == "calibration_summary"
    assert result.calibration_run.artifact_id == result.artifact.artifact_id
    assert result.summary.brier_score == pytest.approx(0.0825)
    assert result.artifact_payload.excluded_outcome_evaluation_ids == (
        "outcome-evaluation-candidate-after-as-of",
    )
    assert store.get_calibration_run(result.calibration_id) == result.calibration_run
    assert store.list_calibration_slices(result.calibration_id) == result.calibration_slices

    assert len(result.calibration_slices) == 5
    overall = next(
        item
        for item in result.calibration_slices
        if item.cohort_label == "calibration_summary:overall"
    )
    assert overall.cohort_label == "calibration_summary:overall"
    assert overall.sample_count == 4
    assert overall.metrics["brier_score"] == 0.0825
    assert overall.baseline_comparison["baseline_id"] == "no_skill_constant_0_5"
    bin_slice = next(
        item
        for item in result.calibration_slices
        if item.cohort_label.startswith("calibration_bin:")
    )
    assert bin_slice.sample_count == 2
    family_slice = next(item for item in result.calibration_slices if item.signal_family == "news")
    assert family_slice.signal_family == "news"
    assert family_slice.metrics["observed_frequency"] == 1.0

    payload = json.loads(Path(result.artifact.path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "calibration-summary-artifact.v1"
    assert payload["summary"]["brier_score"] == 0.0825
    assert payload["excluded_outcome_evaluation_ids"] == [
        "outcome-evaluation-candidate-after-as-of"
    ]
    assert any("as_of cutoff" in item for item in payload["limitations"])


@pytest.mark.unit
def test_calibration_summary_keeps_unresolved_cohort_metric_free() -> None:
    summary = compute_calibration_summary(
        calibration_id="calibration-summary-pending",
        cohort_id="phase6-evalcal-stage6-pending",
        created_at=SUMMARY_CREATED_AT,
        as_of=AS_OF,
        inputs=(_pending_input("candidate-pending", score=0.6),),
        bin_edges=BIN_EDGES,
        families=(SignalArtifactFamily.TECHNICALS,),
    )

    assert summary.sample_count == 1
    assert summary.resolved_count == 0
    assert summary.pending_count == 1
    assert summary.brier_score is None
    assert summary.log_loss is None
    assert summary.accuracy is None
    assert summary.expected_calibration_error is None
    assert summary.baseline_comparison is None
    assert summary.limitations == (
        "No resolved scoreable outcomes were available for calibration.",
    )
    assert summary.bins[1].prediction_count == 1
    assert summary.bins[1].resolved_count == 0
    assert summary.bins[1].average_score is None
    assert summary.bins[1].limitations == (
        "Calibration bin has predictions but no resolved outcomes.",
    )


@pytest.mark.unit
def test_calibration_summary_input_rejects_target_outcome_mismatch() -> None:
    target = _target("candidate-original", score=0.7)
    other = _resolved_input(
        "candidate-other",
        score=0.7,
        quality_score=1.0,
        evaluated_at=datetime(2026, 5, 20, 21, 0, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="candidate_id must match"):
        CalibrationSummaryInput(target=target, outcome_evaluation=other.outcome_evaluation)
