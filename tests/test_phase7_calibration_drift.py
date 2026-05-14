from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    BaselineComparison,
    CalibrationBin,
    CalibrationSummary,
    SignalArtifactFamily,
    SignalFamilyCalibrationSummary,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.evaluation.calibration import CalibrationSummaryArtifactPayload
from nlp_stock_prediction.evaluation.drift import (
    CalibrationDriftThresholds,
    compute_calibration_drift_check,
    write_calibration_drift_check_artifact,
)
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase6_service import Phase6Service
from nlp_stock_prediction.storage import (
    CalibrationRunRecord,
    CalibrationSliceRecord,
    ResearchRunRecord,
    SQLiteStore,
    ToolRunRecord,
    initialize_database,
)

RUN_ID = "run-phase7-calibration-drift"
COHORT_ID = "phase7-drift-msft-swing"
PRIOR_AS_OF = datetime(2026, 5, 10, 0, 0, tzinfo=UTC)
CURRENT_AS_OF = datetime(2026, 5, 20, 0, 0, tzinfo=UTC)
DRIFT_AS_OF = datetime(2026, 5, 21, 0, 0, tzinfo=UTC)
CREATED_AT = datetime(2026, 5, 21, 0, 0, tzinfo=UTC)
BIN_EDGES = (0.0, 0.5, 1.0)
THRESHOLDS = CalibrationDriftThresholds(min_resolved_count=4)


def test_calibration_drift_stable_preserves_source_artifacts_and_membership() -> None:
    prior = _summary(
        "calibration-prior-stable",
        as_of=PRIOR_AS_OF,
        brier_score=0.08,
        expected_calibration_error=0.07,
        accuracy=0.82,
        outcome_ids=("outcome-eval-shared-1", "outcome-eval-prior-only"),
    )
    current = _summary(
        "calibration-current-stable",
        as_of=CURRENT_AS_OF,
        brier_score=0.09,
        expected_calibration_error=0.075,
        accuracy=0.81,
        outcome_ids=("outcome-eval-shared-1", "outcome-eval-current-only"),
    )

    drift = compute_calibration_drift_check(
        prior_summary=prior,
        current_summary=current,
        created_at=CREATED_AT,
        as_of=DRIFT_AS_OF,
        source_calibration_artifact_ids=(
            "artifact-calibration-prior-stable",
            "artifact-calibration-current-stable",
        ),
        thresholds=THRESHOLDS,
    )

    assert drift.drift_status == "stable"
    assert drift.metric_deltas["brier_score_delta"] == pytest.approx(0.01)
    assert drift.metric_deltas["expected_calibration_error_delta"] == pytest.approx(0.005)
    assert drift.source_calibration_artifact_ids == (
        "artifact-calibration-prior-stable",
        "artifact-calibration-current-stable",
    )
    assert drift.source_outcome_evaluation_ids == (
        "outcome-eval-shared-1",
        "outcome-eval-prior-only",
        "outcome-eval-current-only",
    )
    membership = cast(dict[str, object], drift.metadata["source_membership"])
    assert membership["shared_outcome_evaluation_ids"] == ["outcome-eval-shared-1"]
    assert membership["prior_only_outcome_evaluation_ids"] == ["outcome-eval-prior-only"]
    assert membership["current_only_outcome_evaluation_ids"] == ["outcome-eval-current-only"]


def test_calibration_drift_degraded_when_current_errors_worsen() -> None:
    drift = compute_calibration_drift_check(
        prior_summary=_summary(
            "calibration-prior-good",
            as_of=PRIOR_AS_OF,
            brier_score=0.05,
            expected_calibration_error=0.03,
            accuracy=0.91,
        ),
        current_summary=_summary(
            "calibration-current-degraded",
            as_of=CURRENT_AS_OF,
            brier_score=0.18,
            expected_calibration_error=0.16,
            accuracy=0.61,
        ),
        created_at=CREATED_AT,
        as_of=DRIFT_AS_OF,
        source_calibration_artifact_ids=(
            "artifact-calibration-prior-good",
            "artifact-calibration-current-degraded",
        ),
        thresholds=THRESHOLDS,
    )

    assert drift.drift_status == "degraded"
    assert drift.metric_deltas["brier_score_delta"] == pytest.approx(0.13)
    assert drift.metric_deltas["accuracy_delta"] == pytest.approx(-0.3)


def test_calibration_drift_conflicting_metric_movement_is_inconclusive() -> None:
    drift = compute_calibration_drift_check(
        prior_summary=_summary(
            "calibration-prior-conflict",
            as_of=PRIOR_AS_OF,
            brier_score=0.08,
            expected_calibration_error=0.12,
            accuracy=0.6,
        ),
        current_summary=_summary(
            "calibration-current-conflict",
            as_of=CURRENT_AS_OF,
            brier_score=0.2,
            expected_calibration_error=0.04,
            accuracy=0.78,
        ),
        created_at=CREATED_AT,
        as_of=DRIFT_AS_OF,
        source_calibration_artifact_ids=(
            "artifact-calibration-prior-conflict",
            "artifact-calibration-current-conflict",
        ),
        thresholds=THRESHOLDS,
    )

    assert drift.drift_status == "inconclusive"
    assert drift.metric_deltas == {}
    assert any("conflicting directions" in limitation for limitation in drift.limitations)


def test_calibration_drift_without_enough_resolved_history_is_metric_free() -> None:
    drift = compute_calibration_drift_check(
        prior_summary=_summary(
            "calibration-prior-small",
            as_of=PRIOR_AS_OF,
            sample_count=2,
            resolved_count=2,
            brier_score=0.08,
            expected_calibration_error=0.07,
            accuracy=0.75,
        ),
        current_summary=_summary(
            "calibration-current-small",
            as_of=CURRENT_AS_OF,
            sample_count=2,
            resolved_count=2,
            brier_score=0.2,
            expected_calibration_error=0.17,
            accuracy=0.5,
        ),
        created_at=CREATED_AT,
        as_of=DRIFT_AS_OF,
        source_calibration_artifact_ids=(
            "artifact-calibration-prior-small",
            "artifact-calibration-current-small",
        ),
        thresholds=THRESHOLDS,
    )

    assert drift.drift_status == "insufficient_history"
    assert drift.metric_deltas == {}
    assert any("resolved history" in limitation for limitation in drift.limitations)


def test_calibration_drift_rejects_incompatible_cohort_shape() -> None:
    drift = compute_calibration_drift_check(
        prior_summary=_summary(
            "calibration-prior-shape",
            as_of=PRIOR_AS_OF,
            horizon=TimeHorizon.SWING,
        ),
        current_summary=_summary(
            "calibration-current-shape",
            as_of=CURRENT_AS_OF,
            horizon=TimeHorizon.INTRADAY,
        ),
        created_at=CREATED_AT,
        as_of=DRIFT_AS_OF,
        source_calibration_artifact_ids=(
            "artifact-calibration-prior-shape",
            "artifact-calibration-current-shape",
        ),
        thresholds=THRESHOLDS,
    )

    assert drift.drift_status == "not_evaluable"
    assert drift.metric_deltas == {}
    assert any("horizon" in limitation for limitation in drift.limitations)


def test_calibration_drift_preserves_provider_replacement_and_aging_context() -> None:
    drift = compute_calibration_drift_check(
        prior_summary=_summary("calibration-prior-provenance", as_of=PRIOR_AS_OF),
        current_summary=_summary("calibration-current-provenance", as_of=CURRENT_AS_OF),
        created_at=CREATED_AT,
        as_of=DRIFT_AS_OF,
        source_calibration_artifact_ids=(
            "artifact-calibration-prior-provenance",
            "artifact-calibration-current-provenance",
        ),
        provider_compatibility_note_ids=("provider-compatibility-alpha-to-candle",),
        evidence_aging_record_ids=("aging-evidence-msft-old",),
        artifact_freshness_review_ids=("freshness-artifact-calibration-prior",),
        thresholds=THRESHOLDS,
    )

    assert drift.provider_compatibility_note_ids == ("provider-compatibility-alpha-to-candle",)
    assert drift.evidence_aging_record_ids == ("aging-evidence-msft-old",)
    assert drift.artifact_freshness_review_ids == ("freshness-artifact-calibration-prior",)


def test_calibration_drift_writes_artifact_and_sqlite_record(tmp_path: Path) -> None:
    store = initialize_database(tmp_path / "data" / "prediction-research.sqlite3")
    _seed_run(store)
    prior = _persist_calibration_source(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        summary=_summary(
            "calibration-prior-persisted",
            as_of=PRIOR_AS_OF,
            brier_score=0.06,
            expected_calibration_error=0.04,
            accuracy=0.88,
        ),
        created_at=PRIOR_AS_OF,
    )
    current = _persist_calibration_source(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        summary=_summary(
            "calibration-current-persisted",
            as_of=CURRENT_AS_OF,
            brier_score=0.17,
            expected_calibration_error=0.14,
            accuracy=0.7,
        ),
        created_at=CURRENT_AS_OF,
    )

    written = write_calibration_drift_check_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        prior_calibration_id=prior.calibration_id,
        current_calibration_id=current.calibration_id,
        as_of=DRIFT_AS_OF,
        created_at=CREATED_AT,
        thresholds=THRESHOLDS,
    )

    assert written.artifact.artifact_type == "calibration_drift_check"
    assert written.drift_check.drift_status == "degraded"
    assert written.record.artifact_id == written.artifact.artifact_id
    assert store.get_calibration_drift_check(written.drift_check_id) == written.record
    payload = json.loads(Path(written.artifact.path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "calibration-drift-artifact.v1"
    assert payload["drift_check"]["drift_status"] == "degraded"
    assert payload["drift_check"]["source_calibration_artifact_ids"] == [
        prior.artifact_id,
        current.artifact_id,
    ]


def test_phase6_service_writes_calibration_drift_check(tmp_path: Path) -> None:
    store = initialize_database(tmp_path / "data" / "prediction-research.sqlite3")
    _seed_run(store)
    prior = _persist_calibration_source(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        summary=_summary(
            "calibration-prior-service",
            as_of=PRIOR_AS_OF,
            brier_score=0.06,
            expected_calibration_error=0.04,
            accuracy=0.88,
        ),
        created_at=PRIOR_AS_OF,
    )
    current = _persist_calibration_source(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        summary=_summary(
            "calibration-current-service",
            as_of=CURRENT_AS_OF,
            brier_score=0.17,
            expected_calibration_error=0.14,
            accuracy=0.7,
        ),
        created_at=CURRENT_AS_OF,
    )

    service = Phase6Service(repo_root=tmp_path)
    result = service.phase7_calibration_drift_check(
        run_id=RUN_ID,
        prior_calibration_id=prior.calibration_id,
        current_calibration_id=current.calibration_id,
        as_of=DRIFT_AS_OF.isoformat(),
        artifact_dir=(tmp_path / "reports" / RUN_ID / "audit").as_posix(),
        min_resolved_count=4,
    )
    inspection = service.inspect_phase6_run(run_id=RUN_ID)

    assert result["drift_status"] == "degraded"
    assert Path(str(result["artifact_path"])).exists()
    assert inspection["calibration_drift_check_count"] == 1


def _summary(
    calibration_id: str,
    *,
    as_of: datetime,
    sample_count: int = 8,
    resolved_count: int = 8,
    brier_score: float = 0.08,
    expected_calibration_error: float = 0.06,
    accuracy: float = 0.8,
    horizon: TimeHorizon = TimeHorizon.SWING,
    outcome_ids: tuple[str, ...] = (
        "outcome-eval-1",
        "outcome-eval-2",
        "outcome-eval-3",
        "outcome-eval-4",
    ),
) -> CalibrationSummary:
    created_at = as_of
    return CalibrationSummary(
        calibration_id=calibration_id,
        cohort_id=COHORT_ID,
        created_at=created_at,
        as_of=as_of,
        horizon=horizon,
        sample_count=sample_count,
        resolved_count=resolved_count,
        bins=(
            CalibrationBin(
                bin_id="bin-01-0p00-0p50",
                lower_bound=0.0,
                upper_bound=0.5,
                prediction_count=max(1, sample_count // 2),
                resolved_count=max(1, resolved_count // 2),
                average_score=0.3,
                observed_frequency=0.25,
                brier_score=round(brier_score * 1.2, 6),
            ),
            CalibrationBin(
                bin_id="bin-02-0p50-1p00",
                lower_bound=0.5,
                upper_bound=1.0,
                prediction_count=sample_count - max(1, sample_count // 2),
                resolved_count=resolved_count - max(1, resolved_count // 2),
                average_score=0.7,
                observed_frequency=0.75,
                brier_score=round(brier_score * 0.8, 6),
            ),
        ),
        brier_score=brier_score,
        log_loss=round(brier_score + 0.2, 6),
        accuracy=accuracy,
        expected_calibration_error=expected_calibration_error,
        baseline_comparison=_baseline_comparison(brier_score),
        signal_families=(
            SignalFamilyCalibrationSummary(
                family=SignalArtifactFamily.TECHNICALS,
                prediction_count=sample_count,
                resolved_count=resolved_count,
                signal_artifact_count=sample_count,
                average_score=0.62,
                observed_frequency=0.58,
                brier_score=round(brier_score * 1.1, 6),
                score_delta_vs_baseline=0.02,
            ),
        ),
        source_outcome_evaluation_ids=outcome_ids,
        source_artifact_ids=(f"artifact-source-{calibration_id}",),
    )


def _baseline_comparison(brier_score: float) -> BaselineComparison:
    score_delta = round(0.25 - brier_score, 6)
    if score_delta > 0.05:
        verdict = "above_baseline"
    elif score_delta < -0.05:
        verdict = "below_baseline"
    else:
        verdict = "near_baseline"
    return BaselineComparison(
        baseline_id="no_skill_constant_0_5",
        baseline_summary="Calibration quality for a constant 0.5 no-skill score.",
        baseline_score=0.75,
        candidate_score=round(1.0 - brier_score, 6),
        score_delta=score_delta,
        verdict=verdict,
    )


def _seed_run(store: SQLiteStore) -> None:
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=RUN_ID,
            run_kind="phase7_calibration_drift",
            objective="Persist calibration drift checks.",
            status="running",
            started_at=PRIOR_AS_OF,
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )


def _persist_calibration_source(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    summary: CalibrationSummary,
    created_at: datetime,
) -> CalibrationRunRecord:
    tool_run_id = f"tool-{summary.calibration_id}"
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=RUN_ID,
            tool_name="phase6_calibration_summary",
            tool_version="phase6.calibration-summary.v1",
            status="successful",
            started_at=created_at,
            completed_at=created_at,
            inputs={"calibration_id": summary.calibration_id},
        )
    )
    payload = CalibrationSummaryArtifactPayload(
        run_id=RUN_ID,
        calibration_id=summary.calibration_id,
        cohort_id=summary.cohort_id,
        created_at=created_at,
        as_of=summary.as_of,
        bin_edges=BIN_EDGES,
        summary=summary,
        metadata={"source": "phase7_drift_test"},
    )
    artifact = ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=artifact_dir,
        created_at=created_at,
        produced_by="phase6_calibration_summary",
        tool_run_id=tool_run_id,
        schema_version=payload.schema_version,
    ).write_json(
        artifact_id=f"artifact-{summary.calibration_id}",
        artifact_type="calibration_summary",
        filename=f"calibration/summaries/{summary.calibration_id}.json",
        payload=cast(JsonObject, payload.model_dump(mode="json")),
        record_count=1,
        metadata={
            "run_id": RUN_ID,
            "calibration_id": summary.calibration_id,
            "cohort_id": summary.cohort_id,
            "as_of": summary.as_of.isoformat(),
        },
    )
    record = CalibrationRunRecord(
        calibration_id=summary.calibration_id,
        run_id=RUN_ID,
        method_version="phase6.calibration-summary.v1",
        created_at=created_at,
        point_in_time_cutoff=summary.as_of,
        tool_run_id=tool_run_id,
        cohort_query={
            "cohort_id": summary.cohort_id,
            "as_of": summary.as_of.isoformat(),
            "bin_edges": list(BIN_EDGES),
        },
        artifact_id=artifact.artifact_id,
        metadata={"source_artifact_ids": list(summary.source_artifact_ids)},
    )
    store.record_calibration_run(record)
    store.record_calibration_slice(
        CalibrationSliceRecord(
            slice_id=f"slice-{summary.calibration_id}-overall",
            calibration_id=summary.calibration_id,
            cohort_label="calibration_summary:overall",
            sample_count=summary.sample_count,
            resolved_count=summary.resolved_count,
            metrics={
                "brier_score": summary.brier_score,
                "log_loss": summary.log_loss,
                "accuracy": summary.accuracy,
                "expected_calibration_error": summary.expected_calibration_error,
            },
            provenance={
                "source_outcome_evaluation_ids": list(summary.source_outcome_evaluation_ids),
                "source_artifact_ids": list(summary.source_artifact_ids),
            },
        )
    )
    return record
