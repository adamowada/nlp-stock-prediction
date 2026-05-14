from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import (
    Direction,
    EvidenceReference,
    FreshnessStatus,
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
    PredictionType,
    SourceKind,
    TimeHorizon,
)
from nlp_stock_prediction.evaluation import (
    build_prediction_evaluation_target,
    write_point_in_time_outcome_evaluation_artifacts,
)
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    CandidateArtifactLinkRecord,
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    SQLiteStore,
)

RUN_ID = "run-phase6-outcome"
INSTRUMENT_ID = "instrument:equity:us:msft"
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 5, 13, 20, 0, tzinfo=UTC)
WINDOW_START = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 18, 20, 0, tzinfo=UTC)
OBSERVED_AT = datetime(2026, 5, 18, 20, 5, tzinfo=UTC)


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
            run_kind="phase6_outcome_evaluation",
            objective="Evaluate prior prediction outcome quality.",
            status="running",
            started_at=NOW,
            metadata={"run_date": "2026-05-13", "symbol": "MSFT"},
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-source-news",
            artifact_type="normalized_evidence",
            path=Path("reports/run-phase6-outcome/audit/news.json"),
            sha256="a" * 64,
            schema_version="normalized-evidence.v1",
            created_at=NOW,
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-support",
            source_type=SourceKind.NEWS_ARTICLE.value,
            provider="verified-news",
            retrieved_at=NOW,
            published_at=NOW,
            instruments=(INSTRUMENT_ID,),
            claim="Product demand supported the monitored scenario before the cutoff.",
            freshness_status=FreshnessStatus.FRESH.value,
            artifact_id="artifact-source-news",
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-after-cutoff",
            source_type=SourceKind.NEWS_ARTICLE.value,
            provider="verified-news",
            retrieved_at=datetime(2026, 5, 14, 12, 0, tzinfo=UTC),
            published_at=datetime(2026, 5, 14, 12, 0, tzinfo=UTC),
            instruments=(INSTRUMENT_ID,),
            claim="This source arrived after the prediction cutoff.",
            freshness_status=FreshnessStatus.FRESH.value,
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-source-after-cutoff",
            artifact_type="normalized_evidence",
            path=Path("reports/run-phase6-outcome/audit/news-after-cutoff.json"),
            sha256="f" * 64,
            schema_version="normalized-evidence.v1",
            created_at=datetime(2026, 5, 14, 12, 0, tzinfo=UTC),
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-with-late-artifact",
            source_type=SourceKind.NEWS_ARTICLE.value,
            provider="verified-news",
            retrieved_at=NOW,
            published_at=NOW,
            instruments=(INSTRUMENT_ID,),
            claim="The evidence row is available before the source artifact is indexed.",
            freshness_status=FreshnessStatus.FRESH.value,
            artifact_id="artifact-source-after-cutoff",
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-technical-cutoff",
            artifact_type="technical_package",
            path=Path("reports/run-phase6-outcome/audit/technicals.json"),
            sha256="b" * 64,
            schema_version="technical-package.v1",
            produced_by="phase4_technical_package",
            record_count=12,
            created_at=CUTOFF,
            metadata={"signal_family": "technicals", "as_of": CUTOFF.isoformat()},
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-technical-lookahead",
            artifact_type="technical_package",
            path=Path("reports/run-phase6-outcome/audit/technicals-lookahead.json"),
            sha256="c" * 64,
            schema_version="technical-package.v1",
            produced_by="phase4_technical_package",
            record_count=12,
            created_at=datetime(2026, 5, 14, 12, 0, tzinfo=UTC),
            metadata={"signal_family": "technicals"},
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-technical-future-asof",
            artifact_type="technical_package",
            path=Path("reports/run-phase6-outcome/audit/technicals-future-asof.json"),
            sha256="c" * 64,
            schema_version="technical-package.v1",
            produced_by="phase4_technical_package",
            record_count=12,
            created_at=NOW,
            metadata={
                "signal_family": "technicals",
                "as_of": datetime(2026, 5, 14, 12, 0, tzinfo=UTC).isoformat(),
            },
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-prediction-input",
            artifact_type="prediction_input",
            path=Path("reports/run-phase6-outcome/audit/prediction-input.json"),
            sha256="d" * 64,
            schema_version="prediction-input.v1",
            created_at=NOW,
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-msft-swing",
            run_id=RUN_ID,
            instrument_id=INSTRUMENT_ID,
            prediction_horizon=TimeHorizon.SWING.value,
            prediction_type=PredictionType.DIRECTIONAL.value,
            scenario="MSFT evidence supported a monitored upside scenario.",
            direction=Direction.BULLISH.value,
            confidence=0.64,
            status="evidence_supported",
            evidence_for=(
                "evidence-support",
                "evidence-with-late-artifact",
                "evidence-after-cutoff",
            ),
            baseline={
                "baseline_id": "no_directional_edge",
                "summary": "No directional edge without source-backed evidence.",
                "baseline_score": 0.5,
            },
            signal_artifacts=(
                "artifact-technical-cutoff",
                "artifact-technical-lookahead",
                "artifact-technical-future-asof",
            ),
            uncertainty="Outcome review depends on attributed post-window evidence.",
        )
    )
    store.link_candidate_artifact(
        CandidateArtifactLinkRecord(
            candidate_id="candidate-msft-swing",
            artifact_id="artifact-prediction-input",
            relationship="prediction_input",
            created_at=NOW,
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-market-observation",
            artifact_type="market_data",
            path=Path("reports/run-phase6-outcome/audit/market-observation.json"),
            sha256="e" * 64,
            schema_version="market-data.v1",
            created_at=OBSERVED_AT,
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-outcome-observed",
            source_type=SourceKind.MARKET_DATA.value,
            provider="verified-market-data",
            retrieved_at=OBSERVED_AT,
            published_at=OBSERVED_AT,
            instruments=(INSTRUMENT_ID,),
            claim="MSFT closed above the baseline comparison at the evaluation window.",
            freshness_status=FreshnessStatus.FRESH.value,
            artifact_id="artifact-market-observation",
        )
    )
    return store


@pytest.mark.unit
def test_build_prediction_evaluation_target_freezes_cutoff_state(tmp_path: Path) -> None:
    store = _store(tmp_path)

    target = build_prediction_evaluation_target(
        store=store,
        run_id=RUN_ID,
        candidate_id="candidate-msft-swing",
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
    )

    assert target.run_id == RUN_ID
    assert target.evidence_ids == ("evidence-support", "evidence-with-late-artifact")
    assert tuple(item.artifact_id for item in target.signal_artifacts) == (
        "artifact-technical-cutoff",
    )
    assert target.report_artifact_ids == ("artifact-prediction-input",)
    assert target.source_artifact_ids == ("artifact-source-news",)
    assert target.baseline_comparison is not None
    assert target.baseline_comparison.baseline_id == "no_directional_edge"
    assert target.baseline_comparison.candidate_score == pytest.approx(0.64)
    assert any("evidence-after-cutoff" in item for item in target.limitations)
    assert any("artifact-source-after-cutoff" in item for item in target.limitations)
    assert any("artifact-technical-lookahead" in item for item in target.limitations)
    assert any("artifact-technical-future-asof" in item for item in target.limitations)
    assert target.candidate_snapshot["included_evidence_ids"] == [
        "evidence-support",
        "evidence-with-late-artifact",
    ]


@pytest.mark.unit
def test_point_in_time_outcome_evaluation_writes_artifacts_and_storage(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = build_prediction_evaluation_target(
        store=store,
        run_id=RUN_ID,
        candidate_id="candidate-msft-swing",
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
    )

    result = write_point_in_time_outcome_evaluation_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        target=target,
        observed_result=PredictionOutcomeResult.SUPPORTED,
        observed_at=OBSERVED_AT,
        result_summary="MSFT closed above the comparison baseline.",
        result_value=433.25,
        baseline_value=428.10,
        outcome_evidence=(EvidenceReference(evidence_id="evidence-outcome-observed"),),
        market_artifact_ids=("artifact-market-observation",),
        created_at=OBSERVED_AT,
        evaluated_at=datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
    )

    assert result.outcome.status == PredictionOutcomeStatus.OBSERVED
    assert result.outcome_evaluation.status == PredictionOutcomeEvaluationStatus.CONFIRMED
    assert result.outcome_evaluation.quality_score == pytest.approx(1.0)
    assert result.outcome_artifact.artifact_type == "prediction_outcome"
    assert result.outcome_evaluation_artifact.artifact_type == "prediction_outcome_evaluation"
    assert store.get_prediction_outcome(result.outcome.outcome_id) is not None
    stored_evaluation = store.get_prediction_outcome_evaluation(
        result.outcome_evaluation.outcome_evaluation_id
    )
    assert stored_evaluation is not None
    assert stored_evaluation.artifact_id == result.outcome_evaluation_artifact.artifact_id
    assert tuple(
        item.evidence_id for item in store.list_outcome_evidence_links(result.outcome.outcome_id)
    ) == ("evidence-outcome-observed",)
    assert {
        item.artifact_id for item in store.list_outcome_artifact_links(result.outcome.outcome_id)
    } == {
        "artifact-market-observation",
        result.outcome_artifact.artifact_id,
    }
    assert {
        item.artifact_id
        for item in store.list_outcome_evaluation_artifact_links(
            result.outcome_evaluation.outcome_evaluation_id
        )
    } == {
        "artifact-market-observation",
        result.outcome_artifact.artifact_id,
        result.outcome_evaluation_artifact.artifact_id,
    }
    assert {artifact.artifact_id for artifact in store.list_artifacts_for_run(RUN_ID)}.issuperset(
        {
            result.outcome_artifact.artifact_id,
            result.outcome_evaluation_artifact.artifact_id,
        }
    )

    outcome_payload = json.loads(Path(result.outcome_artifact.path).read_text(encoding="utf-8"))
    review_payload = json.loads(
        Path(result.outcome_evaluation_artifact.path).read_text(encoding="utf-8")
    )
    assert outcome_payload["target"]["target_id"] == target.target_id
    assert outcome_payload["outcome"]["observed_result"] == "supported"
    assert review_payload["outcome_evaluation"]["status"] == "confirmed"
    assert review_payload["outcome_evaluation"]["quality_score"] == 1.0


@pytest.mark.unit
def test_pending_point_in_time_outcome_has_no_quality_score(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = build_prediction_evaluation_target(
        store=store,
        run_id=RUN_ID,
        candidate_id="candidate-msft-swing",
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
    )

    result = write_point_in_time_outcome_evaluation_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        target=target,
        status=PredictionOutcomeStatus.PENDING,
        limitations=("Evaluation window has opened but final provider data is not complete.",),
        created_at=datetime(2026, 5, 15, 20, 0, tzinfo=UTC),
        evaluated_at=datetime(2026, 5, 15, 20, 0, tzinfo=UTC),
    )

    assert result.outcome.status == PredictionOutcomeStatus.PENDING
    assert result.outcome.observed_result is None
    assert result.outcome_evaluation.status == PredictionOutcomeEvaluationStatus.PENDING
    assert result.outcome_evaluation.quality_score is None
    assert result.outcome_evaluation.limitations
    stored = store.get_prediction_outcome_evaluation(
        result.outcome_evaluation.outcome_evaluation_id
    )
    assert stored is not None
    assert stored.quality_score is None


@pytest.mark.unit
def test_observed_outcome_requires_observation_evidence_or_artifact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = build_prediction_evaluation_target(
        store=store,
        run_id=RUN_ID,
        candidate_id="candidate-msft-swing",
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
    )

    with pytest.raises(ValueError, match="observed prediction outcomes require evidence"):
        write_point_in_time_outcome_evaluation_artifacts(
            store=store,
            repo_root=tmp_path,
            artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
            run_id=RUN_ID,
            target=target,
            observed_result=PredictionOutcomeResult.SUPPORTED,
            observed_at=OBSERVED_AT,
            created_at=OBSERVED_AT,
            evaluated_at=OBSERVED_AT,
        )


@pytest.mark.unit
def test_observed_outcome_rejects_unavailable_or_unattributed_sources(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = build_prediction_evaluation_target(
        store=store,
        run_id=RUN_ID,
        candidate_id="candidate-msft-swing",
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-other-instrument",
            source_type=SourceKind.MARKET_DATA.value,
            provider="verified-market-data",
            retrieved_at=OBSERVED_AT,
            published_at=OBSERVED_AT,
            instruments=("instrument:equity:us:aapl",),
            claim="This evidence belongs to a different instrument.",
            freshness_status=FreshnessStatus.FRESH.value,
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-other-instrument",
            artifact_type="market_data",
            path=Path("reports/run-phase6-outcome/audit/market-other.json"),
            sha256="f" * 64,
            schema_version="market-data.v1",
            created_at=OBSERVED_AT,
            metadata={"instrument_id": "instrument:equity:us:aapl"},
        )
    )

    with pytest.raises(ValueError, match="outcome evidence instrument does not match"):
        write_point_in_time_outcome_evaluation_artifacts(
            store=store,
            repo_root=tmp_path,
            artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
            run_id=RUN_ID,
            target=target,
            observed_result=PredictionOutcomeResult.SUPPORTED,
            observed_at=OBSERVED_AT,
            outcome_evidence=(EvidenceReference(evidence_id="evidence-other-instrument"),),
            created_at=OBSERVED_AT,
            evaluated_at=OBSERVED_AT,
        )

    with pytest.raises(ValueError, match="market artifact instrument does not match"):
        write_point_in_time_outcome_evaluation_artifacts(
            store=store,
            repo_root=tmp_path,
            artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
            run_id=RUN_ID,
            target=target,
            observed_result=PredictionOutcomeResult.SUPPORTED,
            observed_at=OBSERVED_AT,
            market_artifact_ids=("artifact-other-instrument",),
            created_at=OBSERVED_AT,
            evaluated_at=OBSERVED_AT,
        )


@pytest.mark.unit
def test_insufficient_data_outcome_is_not_evaluable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = build_prediction_evaluation_target(
        store=store,
        run_id=RUN_ID,
        candidate_id="candidate-msft-swing",
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
    )

    result = write_point_in_time_outcome_evaluation_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        target=target,
        observed_result=PredictionOutcomeResult.INSUFFICIENT_DATA,
        observed_at=OBSERVED_AT,
        result_summary="The outcome could not be evaluated with attributed provider data.",
        outcome_evidence=(EvidenceReference(evidence_id="evidence-outcome-observed"),),
        limitations=("Insufficient attributed provider data to score the outcome.",),
        created_at=OBSERVED_AT,
        evaluated_at=OBSERVED_AT,
    )

    assert result.outcome_evaluation.status == PredictionOutcomeEvaluationStatus.NOT_EVALUABLE
    assert result.outcome_evaluation.quality_score is None
