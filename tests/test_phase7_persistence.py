from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.storage import (
    ArtifactRecord,
    CalibrationDriftCheckRecord,
    CalibrationRunRecord,
    CalibrationSourceOutcomeEvaluationLinkRecord,
    EvaluationAttemptRecord,
    EvaluationDataMode,
    EvidenceRecord,
    InstrumentRecord,
    OutcomeArtifactLinkRecord,
    PredictionCandidateRecord,
    PredictionOutcomeEvaluationRecord,
    PredictionOutcomeRecord,
    ResearchRunRecord,
    SourceQueryRecord,
    SQLiteStore,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import CURRENT_RESEARCH_SCHEMA_VERSION


def _store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "prediction-research.sqlite3")


def _ts(hour: int = 12) -> datetime:
    return datetime(2026, 5, 14, hour, 0, tzinfo=UTC)


def _seed_prediction_and_evaluation_runs(store: SQLiteStore) -> None:
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            asset_class="stock",
            name="Microsoft Corporation",
            venue="NASDAQ",
            provider_ids=(
                {
                    "provider": "NASDAQ Basic",
                    "namespace": "listed_symbol",
                    "identifier": "MSFT",
                },
            ),
        )
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-msft-report-2026-05-07",
            run_kind="daily_prediction_report",
            objective="Generate an evidence-backed MSFT prediction report.",
            status="completed",
            started_at=_ts(9),
            completed_at=_ts(10),
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-msft-evaluation-2026-05-14",
            run_kind="prediction_evaluation",
            objective="Evaluate the completed MSFT prediction window.",
            status="running",
            started_at=_ts(11),
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-msft-outcome-2026-05-14",
            run_id="run-msft-evaluation-2026-05-14",
            tool_name="phase7_outcome_review",
            tool_version="0.1",
            status="ok",
            started_at=_ts(11),
            completed_at=_ts(12),
            inputs={"symbol": "MSFT", "provider_mode": "live"},
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-msft-directional-2026-05-07",
            run_id="run-msft-report-2026-05-07",
            instrument_id="equity:NASDAQ:MSFT",
            prediction_horizon="swing",
            prediction_type="directional",
            scenario="MSFT closes above the market-neutral comparison at window end.",
            direction="bullish",
            confidence=0.64,
            status="evidence_supported",
            baseline={"baseline_id": "market-neutral"},
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    store.record_source_query(
        SourceQueryRecord(
            source_query_id="query-msft-close-2026-05-14",
            tool_run_id="tool-msft-outcome-2026-05-14",
            provider="NASDAQ Basic",
            query="MSFT official close 2026-05-14",
            url="https://www.nasdaq.com/market-activity/stocks/msft",
            retrieved_at=_ts(12),
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-msft-close-2026-05-14",
            tool_run_id="tool-msft-outcome-2026-05-14",
            artifact_type="market_data",
            path=Path("artifacts/evaluation/msft-close-2026-05-14.json"),
            sha256="1" * 64,
            schema_version="market-data.v1",
            produced_by="phase7_outcome_review",
            record_count=1,
            metadata={"report_data_mode": "live", "provider_mode": "live"},
            created_at=_ts(12),
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-msft-observed-close-2026-05-14",
            tool_run_id="tool-msft-outcome-2026-05-14",
            source_query_id="query-msft-close-2026-05-14",
            source_type="market_data",
            provider="NASDAQ Basic",
            retrieved_at=_ts(12),
            claim="MSFT official close was above the market-neutral comparison.",
            url="https://www.nasdaq.com/market-activity/stocks/msft",
            query="MSFT official close 2026-05-14",
            instruments=("equity:NASDAQ:MSFT",),
            extraction_confidence=0.99,
            freshness_status="fresh",
            artifact_id="artifact-msft-close-2026-05-14",
            provenance_json={"source_query_id": "query-msft-close-2026-05-14"},
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    for artifact_id, artifact_type, path in (
        (
            "artifact-msft-outcome-review-pass-1",
            "prediction_outcome_evaluation",
            "artifacts/evaluation/msft-outcome-review-pass-1.json",
        ),
        (
            "artifact-msft-outcome-review-pass-2",
            "prediction_outcome_evaluation",
            "artifacts/evaluation/msft-outcome-review-pass-2.json",
        ),
        (
            "artifact-msft-calibration-2026-05-14",
            "calibration_summary",
            "artifacts/evaluation/msft-calibration-2026-05-14.json",
        ),
        (
            "artifact-msft-prior-calibration-2026-05-07",
            "calibration_summary",
            "artifacts/evaluation/msft-prior-calibration-2026-05-07.json",
        ),
        (
            "artifact-msft-calibration-drift-2026-05-14",
            "calibration_drift_check",
            "artifacts/evaluation/msft-calibration-drift-2026-05-14.json",
        ),
    ):
        store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                tool_run_id="tool-msft-outcome-2026-05-14",
                artifact_type=artifact_type,
                path=Path(path),
                sha256="2" * 64,
                schema_version=f"{artifact_type}.v1",
                produced_by="phase7_outcome_review",
                record_count=1,
                metadata={"report_data_mode": "live", "provider_mode": "live"},
                created_at=_ts(12),
            )
        )


@pytest.mark.unit
def test_phase7_schema_initializes_persistence_tables_idempotently(tmp_path: Path) -> None:
    store = _store(tmp_path)

    store.initialize()
    store.initialize()

    assert store.schema_version() == CURRENT_RESEARCH_SCHEMA_VERSION
    with store.connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        outcome_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(prediction_outcome_evaluations)"
            ).fetchall()
        }
        calibration_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(calibration_runs)").fetchall()
        }

    assert {
        "evaluation_attempts",
        "calibration_source_outcome_evaluation_links",
        "calibration_drift_checks",
    }.issubset(tables)
    assert {"evaluation_attempt_id", "data_mode", "provider_mode"}.issubset(outcome_columns)
    assert {"evaluation_attempt_id", "data_mode", "provider_mode"}.issubset(calibration_columns)


@pytest.mark.unit
def test_phase7_attempts_preserve_repeated_outcome_reviews_without_overwrite(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    _seed_prediction_and_evaluation_runs(store)

    for attempt_id, subject_id in (
        ("attempt-msft-outcome-review-pass-1", "outcome-msft-2026-05-14"),
        ("attempt-msft-outcome-review-pass-2", "outcome-msft-2026-05-14"),
    ):
        store.record_evaluation_attempt(
            EvaluationAttemptRecord(
                evaluation_attempt_id=attempt_id,
                run_id="run-msft-evaluation-2026-05-14",
                source_run_id="run-msft-report-2026-05-07",
                tool_run_id="tool-msft-outcome-2026-05-14",
                attempt_kind="outcome_evaluation",
                subject_id=subject_id,
                candidate_id="candidate-msft-directional-2026-05-07",
                instrument_id="equity:NASDAQ:MSFT",
                symbol="MSFT",
                status="completed",
                started_at=_ts(11),
                completed_at=_ts(12),
            )
        )

    outcome = PredictionOutcomeRecord(
        outcome_id="outcome-msft-2026-05-14",
        evaluation_attempt_id="attempt-msft-outcome-review-pass-1",
        candidate_id="candidate-msft-directional-2026-05-07",
        instrument_id="equity:NASDAQ:MSFT",
        symbol="MSFT",
        prediction_type="directional",
        horizon="swing",
        evaluation_window_start=datetime(2026, 5, 7, 20, 0, tzinfo=UTC),
        evaluation_window_end=datetime(2026, 5, 14, 20, 0, tzinfo=UTC),
        status="observed",
        observed_result="confirmed",
        observed_at=datetime(2026, 5, 14, 20, 0, tzinfo=UTC),
        result_summary="MSFT closed above the comparison value.",
        result_value=425.8,
        baseline_value=421.4,
        metadata={"source_run_id": "run-msft-report-2026-05-07"},
    )
    store.upsert_prediction_outcome(outcome)
    store.link_outcome_artifact(
        OutcomeArtifactLinkRecord(
            outcome_id="outcome-msft-2026-05-14",
            artifact_id="artifact-msft-close-2026-05-14",
            relationship="observed_market_data",
            created_at=_ts(12),
        )
    )

    first_review = PredictionOutcomeEvaluationRecord(
        outcome_evaluation_id="outcome-evaluation-msft-pass-1",
        evaluation_attempt_id="attempt-msft-outcome-review-pass-1",
        run_id="run-msft-evaluation-2026-05-14",
        outcome_id="outcome-msft-2026-05-14",
        candidate_id="candidate-msft-directional-2026-05-07",
        instrument_id="equity:NASDAQ:MSFT",
        symbol="MSFT",
        evaluated_at=datetime(2026, 5, 14, 21, 0, tzinfo=UTC),
        status="confirmed",
        quality_score=0.72,
        baseline_comparison={"verdict": "above_baseline"},
        artifact_id="artifact-msft-outcome-review-pass-1",
    )
    second_review = PredictionOutcomeEvaluationRecord(
        outcome_evaluation_id="outcome-evaluation-msft-pass-2",
        evaluation_attempt_id="attempt-msft-outcome-review-pass-2",
        run_id="run-msft-evaluation-2026-05-14",
        outcome_id="outcome-msft-2026-05-14",
        candidate_id="candidate-msft-directional-2026-05-07",
        instrument_id="equity:NASDAQ:MSFT",
        symbol="MSFT",
        evaluated_at=datetime(2026, 5, 14, 22, 0, tzinfo=UTC),
        status="confirmed",
        quality_score=0.74,
        baseline_comparison={"verdict": "above_baseline", "review_pass": 2},
        artifact_id="artifact-msft-outcome-review-pass-2",
    )

    store.append_prediction_outcome_evaluation(first_review)
    store.append_prediction_outcome_evaluation(second_review)

    with pytest.raises(ValueError, match="already exists"):
        store.append_prediction_outcome_evaluation(
            PredictionOutcomeEvaluationRecord(
                **{
                    **first_review.__dict__,
                    "quality_score": 0.91,
                }
            )
        )

    assert store.get_prediction_outcome("outcome-msft-2026-05-14") == outcome
    assert store.list_outcome_evaluations_for_candidate(
        "candidate-msft-directional-2026-05-07"
    ) == (first_review, second_review)
    assert tuple(
        attempt.evaluation_attempt_id
        for attempt in store.list_evaluation_attempts_for_run("run-msft-evaluation-2026-05-14")
    ) == ("attempt-msft-outcome-review-pass-1", "attempt-msft-outcome-review-pass-2")


@pytest.mark.unit
def test_phase7_calibration_source_links_and_drift_records_are_queryable(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    _seed_prediction_and_evaluation_runs(store)
    store.record_evaluation_attempt(
        EvaluationAttemptRecord(
            evaluation_attempt_id="attempt-msft-outcome-review-pass-1",
            run_id="run-msft-evaluation-2026-05-14",
            source_run_id="run-msft-report-2026-05-07",
            tool_run_id="tool-msft-outcome-2026-05-14",
            attempt_kind="outcome_evaluation",
            subject_id="outcome-msft-2026-05-14",
            candidate_id="candidate-msft-directional-2026-05-07",
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            status="completed",
            started_at=_ts(11),
        )
    )
    store.upsert_prediction_outcome(
        PredictionOutcomeRecord(
            outcome_id="outcome-msft-2026-05-14",
            evaluation_attempt_id="attempt-msft-outcome-review-pass-1",
            candidate_id="candidate-msft-directional-2026-05-07",
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            prediction_type="directional",
            horizon="swing",
            evaluation_window_start=datetime(2026, 5, 7, 20, 0, tzinfo=UTC),
            evaluation_window_end=datetime(2026, 5, 14, 20, 0, tzinfo=UTC),
            status="observed",
        )
    )
    store.append_prediction_outcome_evaluation(
        PredictionOutcomeEvaluationRecord(
            outcome_evaluation_id="outcome-evaluation-msft-pass-1",
            evaluation_attempt_id="attempt-msft-outcome-review-pass-1",
            run_id="run-msft-evaluation-2026-05-14",
            outcome_id="outcome-msft-2026-05-14",
            candidate_id="candidate-msft-directional-2026-05-07",
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            evaluated_at=datetime(2026, 5, 14, 21, 0, tzinfo=UTC),
            status="confirmed",
            quality_score=0.72,
            baseline_comparison={"verdict": "above_baseline"},
            artifact_id="artifact-msft-outcome-review-pass-1",
        )
    )
    store.record_evaluation_attempt(
        EvaluationAttemptRecord(
            evaluation_attempt_id="attempt-msft-calibration-2026-05-14",
            run_id="run-msft-evaluation-2026-05-14",
            source_run_id="run-msft-report-2026-05-07",
            tool_run_id="tool-msft-outcome-2026-05-14",
            attempt_kind="calibration_summary",
            subject_id="calibration-msft-2026-05-14",
            status="completed",
            started_at=_ts(12),
        )
    )
    calibration = CalibrationRunRecord(
        calibration_id="calibration-msft-2026-05-14",
        evaluation_attempt_id="attempt-msft-calibration-2026-05-14",
        run_id="run-msft-evaluation-2026-05-14",
        tool_run_id="tool-msft-outcome-2026-05-14",
        method_version="phase7-evaluation-hardening.v1",
        created_at=datetime(2026, 5, 14, 22, 0, tzinfo=UTC),
        point_in_time_cutoff=datetime(2026, 5, 14, 21, 0, tzinfo=UTC),
        cohort_query={"prediction_type": "directional", "horizon": "swing"},
        source_outcome_evaluation_ids=("outcome-evaluation-msft-pass-1",),
        artifact_id="artifact-msft-calibration-2026-05-14",
    )

    store.record_calibration_run(calibration)
    store.record_calibration_run(calibration)

    assert store.list_calibration_source_outcome_evaluation_links(
        "calibration-msft-2026-05-14"
    ) == (
        CalibrationSourceOutcomeEvaluationLinkRecord(
            calibration_id="calibration-msft-2026-05-14",
            outcome_evaluation_id="outcome-evaluation-msft-pass-1",
            relationship="source_outcome_evaluation",
            created_at=datetime(2026, 5, 14, 22, 0, tzinfo=UTC),
        ),
    )

    drift = CalibrationDriftCheckRecord(
        drift_check_id="drift-msft-2026-05-14",
        evaluation_attempt_id="attempt-msft-calibration-2026-05-14",
        run_id="run-msft-evaluation-2026-05-14",
        tool_run_id="tool-msft-outcome-2026-05-14",
        created_at=datetime(2026, 5, 14, 22, 30, tzinfo=UTC),
        as_of=datetime(2026, 5, 14, 22, 30, tzinfo=UTC),
        prior_calibration_id="calibration-msft-2026-05-07",
        current_calibration_id="calibration-msft-2026-05-14",
        prediction_type="directional",
        horizon="swing",
        signal_family="technical",
        drift_status="stable",
        metric_deltas={"brier_score": 0.01},
        source_calibration_artifact_ids=(
            "artifact-msft-prior-calibration-2026-05-07",
            "artifact-msft-calibration-2026-05-14",
        ),
        source_outcome_evaluation_ids=("outcome-evaluation-msft-pass-1",),
        artifact_id="artifact-msft-calibration-drift-2026-05-14",
    )
    store.record_calibration_drift_check(drift)

    assert store.get_calibration_run("calibration-msft-2026-05-14") == calibration
    assert store.get_calibration_drift_check("drift-msft-2026-05-14") == drift
    assert store.list_calibration_drift_checks_for_run("run-msft-evaluation-2026-05-14") == (drift,)
    assert "artifact-msft-calibration-drift-2026-05-14" in {
        artifact.artifact_id
        for artifact in store.list_artifacts_for_run("run-msft-evaluation-2026-05-14")
    }


@pytest.mark.unit
def test_phase7_live_persistence_rejects_non_live_modes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize()
    _seed_prediction_and_evaluation_runs(store)

    with pytest.raises(ValueError, match="data_mode must be live"):
        store.record_evaluation_attempt(
            EvaluationAttemptRecord(
                evaluation_attempt_id="attempt-msft-non-live",
                run_id="run-msft-evaluation-2026-05-14",
                attempt_kind="outcome_evaluation",
                subject_id="outcome-msft-2026-05-14",
                status="blocked",
                started_at=_ts(12),
                data_mode=cast(EvaluationDataMode, "offline_fixture"),
            )
        )

    with pytest.raises(ValueError, match="provider_mode must be live"):
        store.upsert_prediction_outcome(
            PredictionOutcomeRecord(
                outcome_id="outcome-msft-non-live",
                candidate_id="candidate-msft-directional-2026-05-07",
                instrument_id="equity:NASDAQ:MSFT",
                symbol="MSFT",
                prediction_type="directional",
                horizon="swing",
                evaluation_window_start=datetime(2026, 5, 7, 20, 0, tzinfo=UTC),
                evaluation_window_end=datetime(2026, 5, 14, 20, 0, tzinfo=UTC),
                status="not_evaluable",
                provider_mode=cast(EvaluationDataMode, "dummy_smoke"),
            )
        )


@pytest.mark.unit
def test_phase7_failed_tool_cleanup_keeps_prior_successful_attempts(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize()
    _seed_prediction_and_evaluation_runs(store)
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-msft-outcome-failed-2026-05-14",
            run_id="run-msft-evaluation-2026-05-14",
            tool_name="phase7_outcome_review",
            tool_version="0.1",
            status="failed",
            started_at=_ts(13),
            completed_at=_ts(13),
            error_message="Provider returned a transient unavailable response.",
        )
    )
    for attempt_id, tool_run_id, status in (
        ("attempt-msft-successful-2026-05-14", "tool-msft-outcome-2026-05-14", "completed"),
        ("attempt-msft-failed-2026-05-14", "tool-msft-outcome-failed-2026-05-14", "failed"),
    ):
        store.record_evaluation_attempt(
            EvaluationAttemptRecord(
                evaluation_attempt_id=attempt_id,
                run_id="run-msft-evaluation-2026-05-14",
                source_run_id="run-msft-report-2026-05-07",
                tool_run_id=tool_run_id,
                attempt_kind="outcome_evaluation",
                subject_id="outcome-msft-2026-05-14",
                candidate_id="candidate-msft-directional-2026-05-07",
                instrument_id="equity:NASDAQ:MSFT",
                symbol="MSFT",
                status=status,
                started_at=_ts(13),
            )
        )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-msft-failed-review-2026-05-14",
            tool_run_id="tool-msft-outcome-failed-2026-05-14",
            artifact_type="prediction_outcome_evaluation",
            path=Path("artifacts/evaluation/msft-failed-review-2026-05-14.json"),
            sha256="3" * 64,
            schema_version="prediction_outcome_evaluation.v1",
            produced_by="phase7_outcome_review",
            record_count=0,
            metadata={"report_data_mode": "live", "provider_mode": "live"},
            created_at=_ts(13),
        )
    )
    store.upsert_prediction_outcome(
        PredictionOutcomeRecord(
            outcome_id="outcome-msft-2026-05-14",
            evaluation_attempt_id="attempt-msft-successful-2026-05-14",
            candidate_id="candidate-msft-directional-2026-05-07",
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            prediction_type="directional",
            horizon="swing",
            evaluation_window_start=datetime(2026, 5, 7, 20, 0, tzinfo=UTC),
            evaluation_window_end=datetime(2026, 5, 14, 20, 0, tzinfo=UTC),
            status="observed",
        )
    )
    store.append_prediction_outcome_evaluation(
        PredictionOutcomeEvaluationRecord(
            outcome_evaluation_id="outcome-evaluation-msft-successful",
            evaluation_attempt_id="attempt-msft-successful-2026-05-14",
            run_id="run-msft-evaluation-2026-05-14",
            outcome_id="outcome-msft-2026-05-14",
            candidate_id="candidate-msft-directional-2026-05-07",
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            evaluated_at=_ts(13),
            status="confirmed",
            quality_score=0.7,
            artifact_id="artifact-msft-outcome-review-pass-1",
        )
    )
    store.append_prediction_outcome_evaluation(
        PredictionOutcomeEvaluationRecord(
            outcome_evaluation_id="outcome-evaluation-msft-failed",
            evaluation_attempt_id="attempt-msft-failed-2026-05-14",
            run_id="run-msft-evaluation-2026-05-14",
            outcome_id="outcome-msft-2026-05-14",
            candidate_id="candidate-msft-directional-2026-05-07",
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            evaluated_at=_ts(13),
            status="unavailable",
            artifact_id="artifact-msft-failed-review-2026-05-14",
            limitations=("Provider returned unavailable during this attempt.",),
        )
    )

    store.delete_tool_run_outputs("tool-msft-outcome-failed-2026-05-14")

    assert store.get_evaluation_attempt("attempt-msft-successful-2026-05-14") is not None
    assert store.get_evaluation_attempt("attempt-msft-failed-2026-05-14") is None
    assert store.get_prediction_outcome_evaluation("outcome-evaluation-msft-successful") is not None
    assert store.get_prediction_outcome_evaluation("outcome-evaluation-msft-failed") is None
    assert store.get_artifact("artifact-msft-failed-review-2026-05-14") is None
