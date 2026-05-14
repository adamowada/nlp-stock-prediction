from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    AnalysisBundle,
    AnalysisSignal,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    PredictionCandidate,
    PredictionEvaluation,
    PredictionStatus,
    RetrievalMethod,
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
)
from nlp_stock_prediction.evaluation import evaluate_prediction_candidate

NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _source(evidence_id: str, text: str) -> SourceEvidence:
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="TSLA",
        text=text,
        created_at=NOW,
        permalink=f"https://example.test/{evidence_id}",
        matched_tickers=("TSLA",),
        matched_instrument_ids=("instrument:equity:us:tsla",),
        instrument_id="instrument:equity:us:tsla",
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
            freshness_status=FreshnessStatus.FRESH,
        ),
    )


def _candidate(
    *,
    evidence_for: tuple[EvidenceReference, ...] = (),
    evidence_against: tuple[EvidenceReference, ...] = (),
    signal_artifact_ids: tuple[str, ...] = (),
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
        uncertainties=("Fixture sources are deterministic test inputs.",),
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
    assert evaluation.baseline_comparison.verdict == "below_baseline"
    assert any("No attributable source evidence" in item for item in evaluation.uncertainty)


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
    assert evaluation.evidence_counts.technical_signal_artifacts == 1
    assert evaluation.metadata["technical_or_ml_support_is_sidecar_only"] is True


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
