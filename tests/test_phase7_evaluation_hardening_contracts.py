from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    ArtifactFreshnessReview,
    CalibrationDriftCheck,
    Direction,
    EvidenceAgingRecord,
    FreshnessStatus,
    OutcomeReviewSummary,
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeStatus,
    PredictionType,
    ProviderCompatibilityNote,
    ProviderReplacementPlaybook,
    RetrievalMethod,
    SourceKind,
    SourceReliabilityNote,
    TimeHorizon,
)

NOW = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)


def _source_reliability_note(
    *,
    note_id: str = "reliability-evidence-msft-outcome",
    evidence_id: str = "evidence-msft-outcome",
    provider: str = "alpha-vantage",
) -> SourceReliabilityNote:
    return SourceReliabilityNote(
        note_id=note_id,
        evidence_id=evidence_id,
        provider=provider,
        source_type=SourceKind.MARKET_DATA,
        retrieval_method=RetrievalMethod.OFFICIAL_API,
        retrieved_at=NOW,
        observed_at=NOW - timedelta(minutes=5),
        source_url="https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED",
        raw_identifier="MSFT",
        raw_snapshot_id="raw-alpha-vantage-msft-2026-05-14",
        freshness_status=FreshnessStatus.FRESH,
        extraction_confidence=0.99,
        reliability="high",
        evidence_ids=(evidence_id,),
    )


def _artifact_freshness_review() -> ArtifactFreshnessReview:
    return ArtifactFreshnessReview(
        freshness_review_id="freshness-artifact-market-msft",
        artifact_id="artifact-market-msft-outcome",
        artifact_type="market_data",
        provider="alpha-vantage",
        produced_by="phase7_live_outcome_materialization",
        reviewed_at=NOW,
        created_at=NOW - timedelta(minutes=1),
        as_of=NOW - timedelta(minutes=5),
        observed_at=NOW - timedelta(minutes=5),
        freshness_status="fresh",
        sha256="a" * 64,
        source_evidence_ids=("evidence-msft-outcome",),
    )


def _evidence_aging_record() -> EvidenceAgingRecord:
    return EvidenceAgingRecord(
        aging_record_id="aging-evidence-msft-outcome",
        evidence_id="evidence-msft-outcome",
        provider="alpha-vantage",
        source_type=SourceKind.MARKET_DATA,
        retrieved_at=NOW,
        reviewed_at=NOW,
        age_status="fresh",
        freshness_status=FreshnessStatus.FRESH,
        source_reliability_note_id="reliability-evidence-msft-outcome",
        source_artifact_id="artifact-market-msft-outcome",
    )


def _provider_compatibility_note() -> ProviderCompatibilityNote:
    return ProviderCompatibilityNote(
        compatibility_note_id="compat-alpha-vantage-to-candlecharts",
        provider_family="market_data",
        source_provider="alpha-vantage",
        replacement_provider="candlecharts",
        checked_at=NOW,
        compatibility_status="compatible_with_limitations",
        required_fields=(
            "provider",
            "source_url",
            "raw_identifier",
            "retrieved_at",
            "observed_at",
        ),
        preserved_fields=("provider", "source_url", "raw_identifier", "retrieved_at"),
        missing_fields=("observed_at",),
        expected_artifact_type="market_data",
        replacement_artifact_type="market_data",
        limitations=("Candlecharts public payloads may not include close timestamp metadata.",),
    )


@pytest.mark.schema
def test_phase7_outcome_review_summary_preserves_audit_context() -> None:
    summary = OutcomeReviewSummary(
        summary_id="outcome-summary-msft-2026-05-14",
        run_id="run-live-msft-2026-05-14",
        candidate_id="candidate-msft-swing",
        instrument_id="instrument:equity:us:msft",
        symbol="MSFT",
        prediction_type=PredictionType.DIRECTIONAL,
        horizon=TimeHorizon.SWING,
        direction=Direction.BULLISH,
        created_at=NOW,
        prior_run_id="run-live-msft-2026-05-10",
        outcome_id="outcome-msft-swing",
        outcome_evaluation_id="outcome-evaluation-msft-swing",
        outcome_status=PredictionOutcomeStatus.OBSERVED,
        outcome_evaluation_status=PredictionOutcomeEvaluationStatus.CONFIRMED,
        quality_score=1.0,
        outcome_evidence_ids=("evidence-msft-outcome",),
        artifact_ids=("artifact-market-msft-outcome",),
        evidence_aging_records=(_evidence_aging_record(),),
        artifact_freshness_reviews=(_artifact_freshness_review(),),
        source_reliability_notes=(_source_reliability_note(),),
        source_calibration_artifact_ids=("artifact-calibration-msft-swing",),
    )

    assert summary.schema_version == "outcome-review-summary.v1"
    assert summary.quality_score == 1.0
    assert summary.source_reliability_notes[0].reliability == "high"


@pytest.mark.schema
def test_phase7_outcome_review_summary_rejects_unresolved_metrics() -> None:
    with pytest.raises(ValidationError, match="unresolved outcome summaries"):
        OutcomeReviewSummary(
            summary_id="outcome-summary-pending-msft",
            run_id="run-live-msft-2026-05-14",
            candidate_id="candidate-msft-swing",
            instrument_id="instrument:equity:us:msft",
            symbol="MSFT",
            prediction_type=PredictionType.DIRECTIONAL,
            created_at=NOW,
            outcome_id="outcome-msft-swing",
            outcome_evaluation_id="outcome-evaluation-msft-swing",
            outcome_status=PredictionOutcomeStatus.PENDING,
            outcome_evaluation_status=PredictionOutcomeEvaluationStatus.PENDING,
            quality_score=0.5,
            limitations=("Evaluation window has not closed.",),
        )


@pytest.mark.schema
def test_phase7_source_reliability_note_requires_external_traceability() -> None:
    with pytest.raises(ValidationError, match="external source reliability notes"):
        SourceReliabilityNote(
            note_id="reliability-evidence-msft-missing-source",
            evidence_id="evidence-msft-outcome",
            provider="alpha-vantage",
            source_type=SourceKind.MARKET_DATA,
            retrieval_method=RetrievalMethod.OFFICIAL_API,
            retrieved_at=NOW,
            freshness_status=FreshnessStatus.FRESH,
            reliability="high",
        )


@pytest.mark.schema
def test_phase7_contracts_reject_non_live_data_modes() -> None:
    with pytest.raises(ValidationError, match="Input should be 'live'"):
        SourceReliabilityNote(
            note_id="reliability-evidence-msft-fixture",
            evidence_id="evidence-msft-outcome",
            provider="alpha-vantage",
            source_type=SourceKind.MARKET_DATA,
            retrieval_method=RetrievalMethod.OFFICIAL_API,
            retrieved_at=NOW,
            source_url="https://www.alphavantage.co/query",
            raw_identifier="MSFT",
            freshness_status=FreshnessStatus.FRESH,
            reliability="high",
            report_data_mode="offline_fixture",
        )


@pytest.mark.schema
def test_phase7_provider_playbook_requires_compatible_replacement_notes() -> None:
    with pytest.raises(ValidationError, match="compatible playbooks require"):
        ProviderReplacementPlaybook(
            playbook_id="playbook-alpha-vantage-to-bad-provider",
            provider_family="market_data",
            source_provider="alpha-vantage",
            replacement_provider="bad-provider",
            created_at=NOW,
            compatibility_notes=(
                ProviderCompatibilityNote(
                    compatibility_note_id="compat-alpha-vantage-to-bad-provider",
                    provider_family="market_data",
                    source_provider="alpha-vantage",
                    replacement_provider="bad-provider",
                    checked_at=NOW,
                    compatibility_status="incompatible",
                    required_fields=("provider", "source_url", "raw_identifier"),
                    preserved_fields=("provider",),
                    missing_fields=("source_url", "raw_identifier"),
                    expected_artifact_type="market_data",
                    replacement_artifact_type="provider_result",
                    limitations=("Replacement provider lacks source URLs and OHLCV artifacts.",),
                ),
            ),
            required_provenance_fields=("provider", "source_url", "raw_identifier"),
            artifact_schema_versions=("market-data-tool-output.v1",),
        )


@pytest.mark.schema
def test_phase7_provider_playbook_accepts_limited_compatible_swap() -> None:
    playbook = ProviderReplacementPlaybook(
        playbook_id="playbook-alpha-vantage-to-candlecharts",
        provider_family="market_data",
        source_provider="alpha-vantage",
        replacement_provider="candlecharts",
        created_at=NOW,
        compatibility_notes=(_provider_compatibility_note(),),
        required_provenance_fields=(
            "provider",
            "source_url",
            "raw_identifier",
            "retrieved_at",
            "observed_at",
        ),
        artifact_schema_versions=("market-data-tool-output.v1",),
        credential_requirements=("NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT",),
        limitations=("Public scraping replacement has weaker timestamp guarantees.",),
    )

    assert playbook.schema_version == "provider-replacement-playbook.v1"
    assert playbook.compatibility_notes[0].compatibility_status == "compatible_with_limitations"


@pytest.mark.schema
def test_phase7_calibration_drift_check_preserves_separate_artifact_context() -> None:
    drift = CalibrationDriftCheck(
        drift_check_id="drift-msft-swing-2026-05",
        cohort_id="cohort-msft-swing",
        created_at=NOW,
        as_of=NOW,
        prior_calibration_id="calibration-msft-swing-prior",
        current_calibration_id="calibration-msft-swing-current",
        drift_status="watch",
        metric_deltas={
            "expected_calibration_error_delta": 0.08,
            "brier_score_delta": 0.04,
        },
        source_calibration_artifact_ids=(
            "artifact-calibration-msft-swing-prior",
            "artifact-calibration-msft-swing-current",
        ),
        source_outcome_evaluation_ids=("outcome-evaluation-msft-swing",),
        evidence_aging_record_ids=("aging-evidence-msft-outcome",),
        artifact_freshness_review_ids=("freshness-artifact-market-msft",),
    )

    assert drift.schema_version == "calibration-drift-check.v1"
    assert drift.drift_status == "watch"


@pytest.mark.schema
def test_phase7_calibration_drift_rejects_duplicate_source_artifacts() -> None:
    with pytest.raises(ValidationError, match="source_calibration_artifact_ids"):
        CalibrationDriftCheck(
            drift_check_id="drift-msft-duplicate",
            cohort_id="cohort-msft-swing",
            created_at=NOW,
            as_of=NOW,
            prior_calibration_id="calibration-msft-swing-prior",
            current_calibration_id="calibration-msft-swing-current",
            drift_status="degraded",
            metric_deltas={"expected_calibration_error_delta": 0.2},
            source_calibration_artifact_ids=(
                "artifact-calibration-msft-swing",
                "artifact-calibration-msft-swing",
            ),
        )


@pytest.mark.schema
def test_phase7_calibration_drift_rejects_inline_report_coupling() -> None:
    with pytest.raises(ValidationError, match="must remain separate"):
        CalibrationDriftCheck(
            drift_check_id="drift-msft-inline-report",
            cohort_id="cohort-msft-swing",
            created_at=NOW,
            as_of=NOW,
            prior_calibration_id="calibration-msft-swing-prior",
            current_calibration_id="calibration-msft-swing-current",
            drift_status="stable",
            metric_deltas={"expected_calibration_error_delta": 0.0},
            source_calibration_artifact_ids=(
                "artifact-calibration-msft-swing-prior",
                "artifact-calibration-msft-swing-current",
            ),
            metadata={"inline_report_calculation": True},
        )
