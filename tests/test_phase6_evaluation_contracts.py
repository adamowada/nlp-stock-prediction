from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    BaselineComparison,
    CalibrationBin,
    CalibrationSummary,
    Direction,
    EvidenceReference,
    PredictionEvaluationTarget,
    PredictionOutcome,
    PredictionOutcomeArtifactPayload,
    PredictionOutcomeEvaluation,
    PredictionOutcomeEvaluationArtifactPayload,
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
    PredictionType,
    SignalArtifactFamily,
    SignalArtifactReference,
    SignalFamilyAblation,
    SignalFamilyCalibrationSummary,
    TimeHorizon,
)

NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
WINDOW_START = NOW + timedelta(hours=1)
WINDOW_END = NOW + timedelta(days=5)


def _baseline() -> BaselineComparison:
    return BaselineComparison(
        baseline_id="no_directional_edge",
        baseline_summary="No directional edge is assumed without source-backed evidence.",
        baseline_score=0.5,
        candidate_score=0.68,
        score_delta=0.18,
        verdict="above_baseline",
    )


def _signal_reference(
    *,
    artifact_id: str = "artifact-news-signal-tsla",
    family: SignalArtifactFamily = SignalArtifactFamily.NEWS,
    as_of: datetime = NOW - timedelta(minutes=10),
    created_at: datetime = NOW - timedelta(minutes=5),
) -> SignalArtifactReference:
    return SignalArtifactReference(
        artifact_id=artifact_id,
        family=family,
        artifact_type="normalized_evidence",
        schema_version="phase6-signal.v1",
        tool_run_id="tool-news-tsla",
        produced_by="public-news-provider",
        created_at=created_at,
        as_of=as_of,
        source_evidence_ids=("evidence-news-tsla-earnings",),
    )


def _target() -> PredictionEvaluationTarget:
    return PredictionEvaluationTarget(
        target_id="target-candidate-tsla-swing-2026-05-13",
        run_id="run-live-tsla-2026-05-13",
        candidate_id="candidate-tsla-swing-2026-05-13",
        instrument_id="instrument:equity:us:tsla",
        symbol="TSLA",
        prediction_type=PredictionType.DIRECTIONAL,
        horizon=TimeHorizon.SWING,
        direction=Direction.BULLISH,
        report_date=date(2026, 5, 13),
        prediction_created_at=NOW,
        point_in_time_cutoff=NOW,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        candidate_snapshot={
            "candidate_id": "candidate-tsla-swing-2026-05-13",
            "status": "evidence_supported",
            "confidence": 0.68,
        },
        baseline_comparison=_baseline(),
        evidence_ids=("evidence-news-tsla-earnings",),
        signal_artifacts=(_signal_reference(),),
        report_artifact_ids=("artifact-report-json-tsla-2026-05-13",),
        source_artifact_ids=("artifact-news-normalized-tsla",),
    )


def _outcome() -> PredictionOutcome:
    return PredictionOutcome(
        outcome_id="outcome-candidate-tsla-swing-2026-05-13",
        candidate_id="candidate-tsla-swing-2026-05-13",
        instrument_id="instrument:equity:us:tsla",
        symbol="TSLA",
        prediction_type=PredictionType.DIRECTIONAL,
        horizon=TimeHorizon.SWING,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        status=PredictionOutcomeStatus.OBSERVED,
        observed_result=PredictionOutcomeResult.SUPPORTED,
        observed_at=WINDOW_END,
        result_summary="The later market and evidence record supported the original scenario.",
        result_value=0.07,
        baseline_value=0.0,
        outcome_evidence=(EvidenceReference(evidence_id="evidence-outcome-tsla"),),
        artifact_ids=("artifact-market-outcome-tsla",),
    )


def _outcome_evaluation() -> PredictionOutcomeEvaluation:
    outcome = _outcome()
    return PredictionOutcomeEvaluation(
        outcome_evaluation_id="outcome-evaluation-candidate-tsla-swing-2026-05-13",
        outcome_id=outcome.outcome_id,
        candidate_id=outcome.candidate_id,
        instrument_id=outcome.instrument_id,
        symbol=outcome.symbol,
        evaluated_at=WINDOW_END + timedelta(minutes=5),
        status=PredictionOutcomeEvaluationStatus.CONFIRMED,
        outcome=outcome,
        quality_score=0.74,
        baseline_comparison=_baseline(),
        evidence=(EvidenceReference(evidence_id="evidence-outcome-tsla"),),
        artifact_ids=("artifact-market-outcome-tsla",),
    )


@pytest.mark.schema
def test_phase6_prediction_evaluation_target_captures_point_in_time_snapshot() -> None:
    target = _target()

    assert target.schema_version == "prediction-evaluation-target.v1"
    assert target.point_in_time_cutoff == NOW
    assert target.baseline_comparison is not None
    assert target.signal_artifacts[0].family == SignalArtifactFamily.NEWS


@pytest.mark.schema
def test_phase6_prediction_evaluation_target_rejects_signal_lookahead() -> None:
    payload = _target().model_dump()
    payload["signal_artifacts"] = (
        _signal_reference(
            as_of=WINDOW_START + timedelta(minutes=1),
            created_at=WINDOW_START + timedelta(minutes=2),
        ).model_dump(mode="json"),
    )

    with pytest.raises(ValidationError, match="point-in-time cutoff"):
        PredictionEvaluationTarget.model_validate(payload)


@pytest.mark.schema
def test_phase6_prediction_evaluation_target_requires_baseline_or_limitation() -> None:
    payload = _target().model_dump()
    payload["baseline_comparison"] = None
    payload["limitations"] = ()

    with pytest.raises(ValidationError, match="baseline comparison"):
        PredictionEvaluationTarget.model_validate(payload)


@pytest.mark.schema
def test_phase6_outcome_artifact_payload_wraps_target_and_outcome() -> None:
    payload = PredictionOutcomeArtifactPayload(
        run_id="run-live-tsla-2026-05-18",
        created_at=WINDOW_END,
        target=_target(),
        outcome=_outcome(),
        source_evidence_ids=("evidence-outcome-tsla",),
        market_artifact_ids=("artifact-market-outcome-tsla",),
        outcome_artifact_ids=("artifact-prediction-outcome-tsla",),
    )

    assert payload.outcome.status == PredictionOutcomeStatus.OBSERVED
    assert payload.target.candidate_id == payload.outcome.candidate_id


@pytest.mark.schema
def test_phase6_outcome_artifact_payload_rejects_target_mismatch() -> None:
    outcome = _outcome().model_copy(update={"symbol": "NVDA"})

    with pytest.raises(ValidationError, match="symbol"):
        PredictionOutcomeArtifactPayload(
            run_id="run-live-tsla-2026-05-18",
            created_at=WINDOW_END,
            target=_target(),
            outcome=outcome,
        )


@pytest.mark.schema
def test_phase6_outcome_evaluation_payload_preserves_review_metadata() -> None:
    payload = PredictionOutcomeEvaluationArtifactPayload(
        run_id="run-live-tsla-2026-05-18",
        created_at=WINDOW_END + timedelta(minutes=10),
        target=_target(),
        outcome_evaluation=_outcome_evaluation(),
        baseline_comparison=_baseline(),
        evidence_ids=("evidence-outcome-tsla",),
        artifact_ids=("artifact-market-outcome-tsla",),
    )

    assert payload.outcome_evaluation.status == PredictionOutcomeEvaluationStatus.CONFIRMED
    assert payload.baseline_comparison is not None


@pytest.mark.schema
def test_phase6_outcome_evaluation_payload_rejects_pre_evaluation_artifact_time() -> None:
    with pytest.raises(ValidationError, match="created before evaluation"):
        PredictionOutcomeEvaluationArtifactPayload(
            run_id="run-live-tsla-2026-05-18",
            created_at=WINDOW_END,
            target=_target(),
            outcome_evaluation=_outcome_evaluation(),
            baseline_comparison=_baseline(),
            evidence_ids=("evidence-outcome-tsla",),
        )


@pytest.mark.schema
def test_phase6_calibration_summary_accepts_resolved_prediction_quality_metrics() -> None:
    summary = CalibrationSummary(
        calibration_id="calibration-tsla-swing-2026-05",
        cohort_id="cohort-directional-swing-tsla-2026-05",
        created_at=WINDOW_END,
        as_of=WINDOW_END,
        prediction_type=PredictionType.DIRECTIONAL,
        horizon=TimeHorizon.SWING,
        sample_count=3,
        resolved_count=2,
        pending_count=1,
        bins=(
            CalibrationBin(
                bin_id="bin-060-080",
                lower_bound=0.6,
                upper_bound=0.8,
                prediction_count=2,
                resolved_count=2,
                average_score=0.68,
                observed_frequency=0.5,
                brier_score=0.25,
            ),
        ),
        brier_score=0.25,
        log_loss=0.69,
        accuracy=0.5,
        expected_calibration_error=0.18,
        baseline_comparison=_baseline(),
        signal_families=(
            SignalFamilyCalibrationSummary(
                family=SignalArtifactFamily.NEWS,
                prediction_count=2,
                resolved_count=2,
                signal_artifact_count=2,
                average_score=0.68,
                observed_frequency=0.5,
                brier_score=0.25,
            ),
        ),
        source_outcome_evaluation_ids=("outcome-evaluation-candidate-tsla-swing-2026-05-13",),
        source_artifact_ids=("artifact-prediction-outcome-evaluation-tsla",),
    )

    assert summary.schema_version == "calibration-summary.v1"
    assert summary.resolved_count == 2
    assert summary.signal_families[0].family == SignalArtifactFamily.NEWS


@pytest.mark.schema
def test_phase6_signal_family_ablation_accepts_resolved_prediction_quality_delta() -> None:
    ablation = SignalFamilyAblation(
        ablation_id="ablation-news-directional-swing-2026-05",
        cohort_id="cohort-directional-swing-tsla-2026-05",
        created_at=WINDOW_END,
        family=SignalArtifactFamily.NEWS,
        prediction_type=PredictionType.DIRECTIONAL,
        horizon=TimeHorizon.SWING,
        included_prediction_count=2,
        excluded_prediction_count=2,
        resolved_included_count=2,
        resolved_excluded_count=2,
        included_quality_score=0.66,
        excluded_quality_score=0.52,
        quality_score_delta=0.14,
        baseline_comparison=_baseline(),
    )

    assert ablation.schema_version == "signal-family-ablation.v1"
    assert ablation.quality_score_delta == 0.14


@pytest.mark.schema
def test_phase6_calibration_summary_requires_limitations_without_resolved_outcomes() -> None:
    with pytest.raises(ValidationError, match="without resolved outcomes require limitations"):
        CalibrationSummary(
            calibration_id="calibration-empty",
            cohort_id="cohort-empty",
            created_at=NOW,
            as_of=NOW,
            sample_count=1,
            resolved_count=0,
            pending_count=1,
        )


@pytest.mark.schema
def test_phase6_signal_family_ablation_requires_resolved_samples_or_limitations() -> None:
    with pytest.raises(ValidationError, match="unresolved signal family ablations"):
        SignalFamilyAblation(
            ablation_id="ablation-news-empty",
            cohort_id="cohort-directional-swing",
            created_at=NOW,
            family=SignalArtifactFamily.NEWS,
            included_prediction_count=1,
            excluded_prediction_count=1,
            resolved_included_count=0,
            resolved_excluded_count=0,
        )
