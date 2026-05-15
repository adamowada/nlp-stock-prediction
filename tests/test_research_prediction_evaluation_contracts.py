from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    AnalysisBundle,
    AnalysisSignal,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    PredictionCandidate,
    PredictionChangeTrigger,
    PredictionEvaluation,
    PredictionOutcome,
    PredictionOutcomeEvaluation,
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
    PredictionStatus,
    PredictionType,
    RetrievalMethod,
    SignalArtifactFamily,
    SignalArtifactReference,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TechnicalAnalysis,
    TechnicalMlSignal,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.evaluation import (
    BaselineComparison,
    EvaluationEvidenceCounts,
    SignalArtifactCounts,
)
from nlp_stock_prediction.evaluation import evaluate_prediction_candidate

NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _source(
    evidence_id: str,
    text: str,
    *,
    ticker: str = "TSLA",
    instrument_id: str = "instrument:equity:us:tsla",
    freshness_status: FreshnessStatus = FreshnessStatus.FRESH,
) -> SourceEvidence:
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker=ticker,
        text=text,
        created_at=NOW,
        permalink=f"https://example.test/{evidence_id}",
        matched_tickers=(ticker,),
        matched_instrument_ids=(instrument_id,),
        instrument_id=instrument_id,
        provenance=SourceProvenance(
            provider_name="fixture-news",
            source_kind=SourceKind.NEWS_ARTICLE,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=NOW,
            observed_at=NOW,
            source_url=f"https://example.test/{evidence_id}",
            permalink=f"https://example.test/{evidence_id}",
            raw_identifier=evidence_id,
            raw_snapshot_id=f"raw-{evidence_id}",
            freshness_status=freshness_status,
        ),
    )


def _candidate(
    *,
    evidence_for: tuple[EvidenceReference, ...] = (),
    evidence_against: tuple[EvidenceReference, ...] = (),
    signal_artifact_ids: tuple[str, ...] = (),
    signal_artifacts: tuple[SignalArtifactReference, ...] = (),
    include_structured_baseline: bool = True,
) -> PredictionCandidate:
    status = (
        PredictionStatus.EVIDENCE_SUPPORTED
        if evidence_for
        else PredictionStatus.INSUFFICIENT_EVIDENCE
    )
    return PredictionCandidate(
        candidate_id="candidate-tsla-quality",
        instrument_id="instrument:equity:us:tsla",
        symbol="TSLA",
        horizon=TimeHorizon.SWING,
        direction=Direction.MIXED,
        status=status,
        thesis="TSLA fixture scenario quality depends on attributed evidence.",
        baseline="No directional edge is assumed without source-backed evidence.",
        confidence=0.62,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        signal_artifact_ids=signal_artifact_ids,
        signal_artifacts=signal_artifacts,
        uncertainties=("Fixture sources are deterministic test inputs.",),
        change_triggers=(
            PredictionChangeTrigger(
                trigger_id="change-tsla-evaluation-source-refresh",
                summary="Fresh source evidence would change the evaluation support.",
                trigger_type="provider_refresh",
                evidence=(*evidence_for, *evidence_against),
                rationale="The evaluation contract uses bounded fixture source evidence.",
            ),
        ),
        metadata=(
            {
                "baseline": {
                    "baseline_id": "no_directional_edge",
                    "summary": "No directional edge is assumed without source-backed evidence.",
                    "provenance": {
                        "provider": "fixture-baseline",
                        "retrieved_at": NOW.isoformat(),
                    },
                }
            }
            if include_structured_baseline
            else {}
        ),
    )


@pytest.mark.unit
def test_evaluation_scores_evidence_supported_candidate_above_baseline() -> None:
    source = _source("evidence-support", "Fixture catalyst supports the TSLA scenario.")
    candidate = _candidate(
        evidence_for=(EvidenceReference(evidence_id=source.evidence_id),),
    )

    evaluation = evaluate_prediction_candidate(
        candidate,
        evidence_sources=(source,),
        created_at=NOW,
    )

    assert evaluation.status == PredictionStatus.EVIDENCE_SUPPORTED
    assert evaluation.evidence_counts.supporting_source_evidence == 1
    assert evaluation.evidence_counts.contradicting_source_evidence == 0
    assert evaluation.baseline_comparison.verdict == "above_baseline"
    assert evaluation.score > evaluation.baseline_comparison.baseline_score
    assert evaluation.quality_language.report_label == "Prediction quality evaluation"


@pytest.mark.unit
def test_evaluation_scores_conflicting_evidence_as_contradicted() -> None:
    support = _source("evidence-support", "Fixture catalyst supports the TSLA scenario.")
    conflict = _source("evidence-conflict", "Fixture margin risk conflicts with the scenario.")
    candidate = _candidate(
        evidence_for=(EvidenceReference(evidence_id=support.evidence_id),),
        evidence_against=(EvidenceReference(evidence_id=conflict.evidence_id),),
    )

    evaluation = evaluate_prediction_candidate(
        candidate,
        evidence_sources=(support, conflict),
        created_at=NOW,
    )

    assert evaluation.status == PredictionStatus.CONTRADICTED
    assert evaluation.evidence_counts.supporting_source_evidence == 1
    assert evaluation.evidence_counts.contradicting_source_evidence == 1
    assert evaluation.baseline_comparison.verdict == "below_baseline"
    assert any("Contradicting source evidence" in item for item in evaluation.uncertainty)


@pytest.mark.unit
def test_evaluation_scores_no_source_evidence_as_insufficient() -> None:
    candidate = _candidate(signal_artifact_ids=("artifact-technical-tsla",))

    evaluation = evaluate_prediction_candidate(
        candidate,
        evidence_sources=(),
        created_at=NOW,
    )

    assert evaluation.status == PredictionStatus.INSUFFICIENT_EVIDENCE
    assert evaluation.evidence_counts.supporting_source_evidence == 0
    assert evaluation.evidence_counts.technical_signal_artifacts == 1
    assert evaluation.evidence_counts.signal_artifacts_by_family.technicals == 1
    assert evaluation.baseline_comparison.verdict == "below_baseline"
    assert any("No attributable source evidence" in item for item in evaluation.uncertainty)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected_context"),
    (
        (
            _source(
                "evidence-stale",
                "Old fixture catalyst should not support the TSLA scenario.",
                freshness_status=FreshnessStatus.STALE,
            ),
            "stale or unavailable",
        ),
        (
            _source(
                "evidence-nvda",
                "NVDA fixture catalyst should not support the TSLA scenario.",
                ticker="NVDA",
                instrument_id="instrument:equity:us:nvda",
            ),
            "wrong instrument",
        ),
    ),
)
def test_evaluation_rejects_stale_or_wrong_instrument_source_support(
    source: SourceEvidence,
    expected_context: str,
) -> None:
    candidate = _candidate(evidence_for=(EvidenceReference(evidence_id=source.evidence_id),))

    evaluation = evaluate_prediction_candidate(
        candidate,
        evidence_sources=(source,),
        created_at=NOW,
    )

    assert evaluation.status == PredictionStatus.INSUFFICIENT_EVIDENCE
    assert evaluation.evidence_counts.supporting_source_evidence == 0
    assert evaluation.evidence_counts.missing_source_references == 1
    assert source.evidence_id in evaluation.evidence_counts.missing_reference_ids
    assert any(expected_context in item for item in evaluation.uncertainty)


@pytest.mark.unit
def test_baseline_comparison_avoids_directional_claim_without_structured_baseline() -> None:
    source = _source("evidence-support", "Fixture catalyst supports the TSLA scenario.")
    candidate = _candidate(
        evidence_for=(EvidenceReference(evidence_id=source.evidence_id),),
        include_structured_baseline=False,
    )

    evaluation = evaluate_prediction_candidate(
        candidate,
        evidence_sources=(source,),
        created_at=NOW,
    )

    assert evaluation.baseline_comparison.verdict == "baseline_unavailable"
    assert evaluation.baseline_comparison.score_delta == 0.0
    assert any("structured baseline" in item for item in evaluation.uncertainty)


@pytest.mark.unit
def test_technical_ml_only_support_cannot_create_evidence_supported_evaluation() -> None:
    ml_signal = TechnicalMlSignal(
        model_hash="model-fixture",
        dataset_hash="dataset-fixture",
        as_of=NOW,
        feature_end=NOW,
        prediction_horizon_sessions=5,
        probability_positive=0.72,
        calibrated_confidence=0.44,
        signal=AnalysisSignal.SUPPORTS,
        freshness_status=FreshnessStatus.FRESH,
    )
    analysis = AnalysisBundle(
        analysis_id="analysis-tsla-technical",
        ticker="TSLA",
        as_of=NOW,
        technical=TechnicalAnalysis(
            ticker="TSLA",
            summary="Technical and ML sidecar context supports the scenario.",
            signal=AnalysisSignal.SUPPORTS,
            confidence=0.74,
            ml_signal=ml_signal,
        ),
        signals=(AnalysisSignal.SUPPORTS,),
    )
    candidate = _candidate(signal_artifact_ids=("artifact-ml-tsla",))

    evaluation = evaluate_prediction_candidate(
        candidate,
        evidence_sources=(),
        analysis_bundle=analysis,
        created_at=NOW,
    )

    assert evaluation.status == PredictionStatus.INSUFFICIENT_EVIDENCE
    assert evaluation.evidence_counts.ml_signal_count == 1
    assert evaluation.evidence_counts.technical_signal_artifacts == 0
    assert evaluation.evidence_counts.signal_artifacts_by_family.timesfm == 1
    assert evaluation.metadata["technical_or_ml_support_is_sidecar_only"] is True


@pytest.mark.unit
def test_typed_signal_artifact_reference_counts_by_family() -> None:
    signal_ref = SignalArtifactReference(
        artifact_id="artifact-social-tsla",
        family=SignalArtifactFamily.SOCIAL,
        artifact_type="normalized_evidence",
        schema_version="research.social-evidence.v1",
        tool_run_id="tool-social-tsla",
        created_at=NOW,
        as_of=NOW,
        source_evidence_ids=("evidence-social",),
    )
    candidate = _candidate(
        signal_artifact_ids=(signal_ref.artifact_id,),
        signal_artifacts=(signal_ref,),
    )

    evaluation = evaluate_prediction_candidate(
        candidate,
        evidence_sources=(),
        created_at=NOW,
    )

    assert evaluation.signal_artifact_ids == ("artifact-social-tsla",)
    assert evaluation.signal_artifacts == (signal_ref,)
    assert evaluation.evidence_counts.signal_artifacts_by_family.social == 1


@pytest.mark.unit
def test_signal_artifact_reference_rejects_wrong_family_artifact_type() -> None:
    with pytest.raises(ValidationError, match="family"):
        SignalArtifactReference(
            artifact_id="artifact-timesfm-news",
            family=SignalArtifactFamily.TIMESFM,
            artifact_type="normalized_evidence",
        )


@pytest.mark.unit
def test_evaluation_contract_rejects_supported_status_without_source_evidence() -> None:
    with pytest.raises(ValidationError, match="attributable supporting source evidence"):
        PredictionEvaluation(
            evaluation_id="evaluation-invalid",
            candidate_id="candidate-invalid",
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            created_at=NOW,
            status=PredictionStatus.EVIDENCE_SUPPORTED,
            score=0.61,
            baseline_comparison=BaselineComparison(
                baseline_id="no_directional_edge",
                baseline_summary="No directional edge baseline.",
                baseline_score=0.5,
                candidate_score=0.61,
                score_delta=0.11,
                verdict="above_baseline",
            ),
            uncertainty=("No source evidence should fail this contract.",),
            evidence_counts=EvaluationEvidenceCounts(
                supporting_source_evidence=0,
                contradicting_source_evidence=0,
                missing_source_references=0,
            ),
        )


@pytest.mark.unit
def test_evaluation_contract_rejects_counts_that_drift_from_evidence_references() -> None:
    support = EvidenceReference(evidence_id="evidence-support")

    with pytest.raises(ValidationError, match="evidence_for"):
        PredictionEvaluation(
            evaluation_id="evaluation-invalid-reference-counts",
            candidate_id="candidate-invalid",
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            created_at=NOW,
            status=PredictionStatus.EVIDENCE_SUPPORTED,
            score=0.61,
            baseline_comparison=BaselineComparison(
                baseline_id="no_directional_edge",
                baseline_summary="No directional edge baseline.",
                baseline_score=0.5,
                candidate_score=0.61,
                score_delta=0.11,
                verdict="above_baseline",
            ),
            uncertainty=("Counts must summarize the same evidence references.",),
            evidence_for=(support,),
            evidence_counts=EvaluationEvidenceCounts(
                supporting_source_evidence=1,
                contradicting_source_evidence=0,
                missing_source_references=0,
                supporting_reference_ids=("evidence-other",),
            ),
        )


@pytest.mark.unit
def test_evaluation_contract_rejects_contradicted_status_without_contradicting_evidence() -> None:
    with pytest.raises(ValidationError, match="contradicting evidence"):
        PredictionEvaluation(
            evaluation_id="evaluation-contradicted-empty",
            candidate_id="candidate-invalid",
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            created_at=NOW,
            status=PredictionStatus.CONTRADICTED,
            score=0.4,
            baseline_comparison=BaselineComparison(
                baseline_id="no_directional_edge",
                baseline_summary="No directional edge baseline.",
                baseline_score=0.5,
                candidate_score=0.4,
                score_delta=-0.1,
                verdict="below_baseline",
            ),
            uncertainty=("Contradicted status must be source-backed.",),
            evidence_counts=EvaluationEvidenceCounts(
                supporting_source_evidence=0,
                contradicting_source_evidence=0,
                missing_source_references=0,
            ),
        )


@pytest.mark.unit
def test_evaluation_contract_rejects_signal_artifact_count_mismatch() -> None:
    signal_ref = SignalArtifactReference(
        artifact_id="artifact-social-tsla",
        family=SignalArtifactFamily.SOCIAL,
        artifact_type="normalized_evidence",
    )

    with pytest.raises(ValidationError, match="signal_artifact_ids"):
        PredictionEvaluation(
            evaluation_id="evaluation-invalid-signal-ids",
            candidate_id="candidate-invalid",
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            created_at=NOW,
            status=PredictionStatus.INSUFFICIENT_EVIDENCE,
            score=0.2,
            baseline_comparison=BaselineComparison(
                baseline_id="no_directional_edge",
                baseline_summary="No directional edge baseline.",
                baseline_score=0.5,
                candidate_score=0.2,
                score_delta=-0.3,
                verdict="below_baseline",
            ),
            uncertainty=("Signal artifact alignment should fail this contract.",),
            evidence_counts=EvaluationEvidenceCounts(
                supporting_source_evidence=0,
                contradicting_source_evidence=0,
                missing_source_references=0,
                technical_signal_artifacts=1,
                signal_artifacts_by_family=SignalArtifactCounts(social=1),
            ),
            signal_artifact_ids=("artifact-other",),
            signal_artifacts=(signal_ref,),
        )


@pytest.mark.unit
def test_baseline_comparison_rejects_inconsistent_delta_and_verdict() -> None:
    with pytest.raises(ValidationError, match="score_delta"):
        BaselineComparison(
            baseline_id="no_directional_edge",
            baseline_summary="No directional edge baseline.",
            baseline_score=0.5,
            candidate_score=0.61,
            score_delta=0.01,
            verdict="above_baseline",
        )

    with pytest.raises(ValidationError, match="verdict"):
        BaselineComparison(
            baseline_id="no_directional_edge",
            baseline_summary="No directional edge baseline.",
            baseline_score=0.5,
            candidate_score=0.61,
            score_delta=0.11,
            verdict="near_baseline",
        )


@pytest.mark.unit
def test_prediction_candidate_requires_uncertainty_for_supported_status() -> None:
    source = _source("evidence-support", "Fixture catalyst supports the TSLA scenario.")
    payload = _candidate(
        evidence_for=(EvidenceReference(evidence_id=source.evidence_id),),
    ).model_dump(mode="python")
    payload["uncertainties"] = []
    payload["uncertainty_drivers"] = []

    with pytest.raises(ValidationError, match="uncertainty"):
        PredictionCandidate.model_validate(payload)


@pytest.mark.unit
def test_prediction_outcome_and_outcome_evaluation_accept_observed_result() -> None:
    evidence = EvidenceReference(evidence_id="evidence-outcome-tsla")
    outcome = PredictionOutcome(
        outcome_id="outcome-candidate-tsla-quality",
        candidate_id="candidate-tsla-quality",
        instrument_id="instrument:equity:us:tsla",
        symbol="TSLA",
        prediction_type=PredictionType.DIRECTIONAL,
        horizon=TimeHorizon.SWING,
        evaluation_window_start=NOW,
        evaluation_window_end=NOW + timedelta(days=5),
        status=PredictionOutcomeStatus.OBSERVED,
        observed_result=PredictionOutcomeResult.SUPPORTED,
        observed_at=NOW + timedelta(days=5),
        result_summary="The later evidence supported the original scenario.",
        outcome_evidence=(evidence,),
    )
    outcome_evaluation = PredictionOutcomeEvaluation(
        outcome_evaluation_id="outcome-evaluation-candidate-tsla-quality",
        outcome_id=outcome.outcome_id,
        candidate_id=outcome.candidate_id,
        instrument_id=outcome.instrument_id,
        symbol=outcome.symbol,
        evaluated_at=NOW + timedelta(days=5),
        status=PredictionOutcomeEvaluationStatus.CONFIRMED,
        outcome=outcome,
        quality_score=0.73,
        evidence=(evidence,),
    )

    assert outcome_evaluation.outcome.observed_result == PredictionOutcomeResult.SUPPORTED
    assert outcome_evaluation.status == PredictionOutcomeEvaluationStatus.CONFIRMED


@pytest.mark.unit
def test_prediction_outcome_requires_limitations_when_unavailable() -> None:
    with pytest.raises(ValidationError, match="limitations"):
        PredictionOutcome(
            outcome_id="outcome-unavailable",
            candidate_id="candidate-tsla-quality",
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            prediction_type=PredictionType.DIRECTIONAL,
            evaluation_window_start=NOW,
            evaluation_window_end=NOW + timedelta(days=5),
            status=PredictionOutcomeStatus.UNAVAILABLE,
        )


@pytest.mark.unit
def test_prediction_outcome_rejects_observed_fields_when_unavailable() -> None:
    with pytest.raises(ValidationError, match="observed_at"):
        PredictionOutcome(
            outcome_id="outcome-unavailable-observed-fields",
            candidate_id="candidate-tsla-quality",
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            prediction_type=PredictionType.DIRECTIONAL,
            evaluation_window_start=NOW,
            evaluation_window_end=NOW + timedelta(days=5),
            status=PredictionOutcomeStatus.UNAVAILABLE,
            observed_at=NOW + timedelta(days=5),
            limitations=("Provider result was unavailable.",),
        )

    with pytest.raises(ValidationError, match="observed values"):
        PredictionOutcome(
            outcome_id="outcome-unavailable-result-value",
            candidate_id="candidate-tsla-quality",
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            prediction_type=PredictionType.DIRECTIONAL,
            evaluation_window_start=NOW,
            evaluation_window_end=NOW + timedelta(days=5),
            status=PredictionOutcomeStatus.UNAVAILABLE,
            result_value=0.3,
            limitations=("Provider result was unavailable.",),
        )


@pytest.mark.unit
def test_resolved_outcome_evaluation_requires_observed_outcome() -> None:
    outcome = PredictionOutcome(
        outcome_id="outcome-pending",
        candidate_id="candidate-tsla-quality",
        instrument_id="instrument:equity:us:tsla",
        symbol="TSLA",
        prediction_type=PredictionType.DIRECTIONAL,
        evaluation_window_start=NOW,
        evaluation_window_end=NOW + timedelta(days=5),
        status=PredictionOutcomeStatus.PENDING,
        limitations=("Evaluation window has not completed.",),
    )

    with pytest.raises(ValidationError, match="observed outcome"):
        PredictionOutcomeEvaluation(
            outcome_evaluation_id="outcome-evaluation-pending",
            outcome_id=outcome.outcome_id,
            candidate_id=outcome.candidate_id,
            instrument_id=outcome.instrument_id,
            symbol=outcome.symbol,
            evaluated_at=NOW,
            status=PredictionOutcomeEvaluationStatus.CONFIRMED,
            outcome=outcome,
            quality_score=0.5,
            evidence=(EvidenceReference(evidence_id="evidence-outcome-tsla"),),
        )
