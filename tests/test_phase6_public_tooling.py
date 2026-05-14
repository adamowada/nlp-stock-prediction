from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

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
from nlp_stock_prediction.contracts.evaluation import (
    PredictionEvaluationTarget,
    PredictionOutcomeEvaluationArtifactPayload,
)
from nlp_stock_prediction.evaluation.outcomes import PHASE6_OUTCOME_TOOL_NAME
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase6_service import (
    Phase6Service,
    build_phase6_tool_registry,
)
from nlp_stock_prediction.storage import (
    InstrumentRecord,
    PredictionCandidateRecord,
    PredictionOutcomeEvaluationRecord,
    PredictionOutcomeRecord,
    ResearchRunRecord,
    SQLiteStore,
)

RUN_ID = "run-phase6-public-tools"
INSTRUMENT_ID = "instrument:equity:us:msft"
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 5, 13, 20, 0, tzinfo=UTC)
WINDOW_START = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 18, 20, 0, tzinfo=UTC)
AS_OF = datetime(2026, 5, 22, 0, 0, tzinfo=UTC)


class _FakeMcpServer:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def tool(self) -> Any:
        def decorator(func: Any) -> Any:
            self.registered.append(func.__name__)
            return func

        return decorator


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
            run_kind="phase6_evaluation_and_calibration",
            objective="Run Phase 6 live tooling over persisted outcome evaluations.",
            status="running",
            started_at=NOW,
            metadata={"run_date": "2026-05-13", "symbol": "MSFT"},
        )
    )
    return store


def _baseline(score: float) -> BaselineComparison:
    delta = round(score - 0.5, 6)
    return BaselineComparison(
        baseline_id="no_directional_edge",
        baseline_summary="No directional edge without source-backed evidence.",
        baseline_score=0.5,
        candidate_score=score,
        score_delta=delta,
        verdict=(
            "above_baseline"
            if delta > 0.05
            else "below_baseline"
            if delta < -0.05
            else "near_baseline"
        ),
    )


def _artifact_type(family: SignalArtifactFamily) -> str:
    if family == SignalArtifactFamily.TECHNICALS:
        return "technical_package"
    return "normalized_evidence"


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
            created_at=CUTOFF,
            as_of=CUTOFF,
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
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        candidate_snapshot={
            "candidate_id": candidate_id,
            "scenario": "Stored source-backed monitored directional scenario.",
            "confidence": score,
            "signal_artifact_ids": [item.artifact_id for item in signal_artifacts],
        },
        baseline_comparison=_baseline(score),
        evidence_ids=(f"evidence-{candidate_id}-support",),
        signal_artifacts=signal_artifacts,
    )


def _outcome_evaluation(
    target: PredictionEvaluationTarget,
    *,
    quality_score: float,
    evaluated_at: datetime,
) -> PredictionOutcomeEvaluation:
    status = (
        PredictionOutcomeEvaluationStatus.CONFIRMED
        if quality_score >= 0.5
        else PredictionOutcomeEvaluationStatus.MISSED
    )
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
        observed_at=evaluated_at - timedelta(minutes=20),
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
        evaluated_at=evaluated_at,
        status=status,
        outcome=outcome,
        quality_score=quality_score,
        baseline_comparison=target.baseline_comparison,
        evidence=(EvidenceReference(evidence_id=f"evidence-{target.candidate_id}-outcome"),),
        artifact_ids=(f"artifact-{target.candidate_id}-outcome-review",),
    )


def _persist_source_payload(
    *,
    store: SQLiteStore,
    repo_root: Path,
    audit_dir: Path,
    target: PredictionEvaluationTarget,
    outcome_evaluation: PredictionOutcomeEvaluation,
) -> None:
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id=target.candidate_id,
            run_id=RUN_ID,
            instrument_id=INSTRUMENT_ID,
            prediction_horizon=target.horizon.value,
            prediction_type=target.prediction_type.value,
            scenario=str(target.candidate_snapshot["scenario"]),
            direction=target.direction.value if target.direction else None,
            confidence=target.baseline_comparison.candidate_score
            if target.baseline_comparison
            else None,
            status="evidence_supported",
            signal_artifacts=tuple(item.artifact_id for item in target.signal_artifacts),
            baseline=cast(
                JsonObject,
                target.baseline_comparison.model_dump(mode="json")
                if target.baseline_comparison
                else {},
            ),
        )
    )
    outcome = outcome_evaluation.outcome
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
            observed_result=outcome.observed_result.value if outcome.observed_result else None,
            observed_at=outcome.observed_at,
            result_summary=outcome.result_summary,
            limitations=outcome.limitations,
            metadata={"target_id": target.target_id},
        )
    )
    payload = PredictionOutcomeEvaluationArtifactPayload(
        run_id=RUN_ID,
        created_at=outcome_evaluation.evaluated_at,
        target=target,
        outcome_evaluation=outcome_evaluation,
        baseline_comparison=target.baseline_comparison,
        evidence_ids=tuple(reference.evidence_id for reference in outcome_evaluation.evidence),
        artifact_ids=outcome_evaluation.artifact_ids,
        limitations=outcome_evaluation.limitations,
        metadata={"target_id": target.target_id},
    )
    artifact = ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=audit_dir,
        created_at=outcome_evaluation.evaluated_at,
        produced_by=PHASE6_OUTCOME_TOOL_NAME,
        tool_run_id=None,
        schema_version=payload.schema_version,
    ).write_json(
        artifact_id=f"artifact-review-{target.candidate_id}",
        artifact_type="prediction_outcome_evaluation",
        filename=f"prediction-outcome-evaluations/{target.candidate_id}.json",
        payload=cast(JsonObject, payload.model_dump(mode="json")),
        record_count=1,
        metadata={
            "run_id": RUN_ID,
            "target_id": target.target_id,
            "outcome_evaluation_id": outcome_evaluation.outcome_evaluation_id,
        },
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
            artifact_id=artifact.artifact_id,
            limitations=outcome_evaluation.limitations,
            metadata={"target_id": target.target_id},
        )
    )


def _persist_outcome_sources(tmp_path: Path) -> Path:
    store = _store(tmp_path)
    audit_dir = tmp_path / "reports" / RUN_ID / "audit"
    samples = (
        (
            "candidate-001",
            0.2,
            0.0,
            datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
            (SignalArtifactFamily.TECHNICALS,),
        ),
        ("candidate-002", 0.4, 0.0, datetime(2026, 5, 19, 21, 0, tzinfo=UTC), ()),
        (
            "candidate-003",
            0.7,
            1.0,
            datetime(2026, 5, 20, 21, 0, tzinfo=UTC),
            (SignalArtifactFamily.NEWS,),
        ),
        (
            "candidate-004",
            0.8,
            1.0,
            datetime(2026, 5, 21, 21, 0, tzinfo=UTC),
            (SignalArtifactFamily.TECHNICALS,),
        ),
    )
    for candidate_id, score, quality_score, evaluated_at, families in samples:
        target = _target(candidate_id, score=score, families=families)
        _persist_source_payload(
            store=store,
            repo_root=tmp_path,
            audit_dir=audit_dir,
            target=target,
            outcome_evaluation=_outcome_evaluation(
                target,
                quality_score=quality_score,
                evaluated_at=evaluated_at,
            ),
        )
    return audit_dir


@pytest.mark.unit
def test_phase6_registry_exposes_live_evaluation_tool_suite() -> None:
    plan = build_phase6_tool_registry().as_plan()
    tools = cast(list[dict[str, object]], plan["tools"])
    tool_names = [str(tool["tool_name"]) for tool in tools]

    assert tool_names == [
        "phase6_load_outcome_evaluations",
        "phase6_signal_family_ablation",
        "phase6_walk_forward_evaluation",
        "phase6_calibration_summary",
        "inspect_phase6_run",
    ]
    assert not any("dummy" in tool_name for tool_name in tool_names)
    assert {str(tool["requires_network"]) for tool in tools} == {"False"}


@pytest.mark.unit
def test_phase6_mcp_registration_matches_public_tool_names(tmp_path: Path) -> None:
    from nlp_stock_prediction.orchestration.phase6_mcp_registration import (
        PHASE6_MCP_TOOL_NAMES,
        register_phase6_mcp_tools,
    )

    server = _FakeMcpServer()
    service = Phase6Service(repo_root=tmp_path)

    register_phase6_mcp_tools(server, service)

    assert list(PHASE6_MCP_TOOL_NAMES) == server.registered
    assert "run_dummy_universe_tool" not in server.registered
    assert "run_dummy_analysis_tool" not in server.registered


@pytest.mark.unit
def test_phase6_service_runs_live_tools_from_persisted_outcome_artifacts(
    tmp_path: Path,
) -> None:
    audit_dir = _persist_outcome_sources(tmp_path)
    service = Phase6Service(repo_root=tmp_path)

    loaded = service.phase6_load_outcome_evaluations(run_id=RUN_ID)
    ablation = service.phase6_signal_family_ablation(
        run_id=RUN_ID,
        cohort_id="phase6-evalcal-stage7-msft-swing",
        point_in_time_cutoff=AS_OF.isoformat(),
        artifact_dir=audit_dir.as_posix(),
        families=("technicals",),
    )
    walk_forward = service.phase6_walk_forward_evaluation(
        run_id=RUN_ID,
        cohort_id="phase6-evalcal-stage7-msft-swing",
        point_in_time_cutoff=AS_OF.isoformat(),
        artifact_dir=audit_dir.as_posix(),
        minimum_train_size=2,
        test_size=1,
        step_size=1,
    )
    calibration = service.phase6_calibration_summary(
        run_id=RUN_ID,
        cohort_id="phase6-evalcal-stage7-msft-swing",
        as_of=AS_OF.isoformat(),
        artifact_dir=audit_dir.as_posix(),
        bin_edges=(0.0, 0.5, 1.0),
        families=("technicals", "news"),
    )
    inspection = service.inspect_phase6_run(run_id=RUN_ID)

    assert loaded["outcome_evaluation_count"] == 4
    assert loaded["source_outcome_evaluation_ids"] == [
        "outcome-evaluation-candidate-001",
        "outcome-evaluation-candidate-002",
        "outcome-evaluation-candidate-003",
        "outcome-evaluation-candidate-004",
    ]
    assert ablation["ablation_count"] == 1
    assert ablation["calibration_slice_count"] == 1
    assert Path(str(ablation["artifact_path"])).exists()
    assert walk_forward["fold_count"] == 2
    assert walk_forward["sample_count"] == 4
    assert Path(str(walk_forward["artifact_path"])).exists()
    assert calibration["sample_count"] == 4
    assert calibration["resolved_count"] == 4
    assert calibration["brier_score"] == pytest.approx(0.0825)
    assert calibration["expected_calibration_error"] == pytest.approx(0.275)
    assert Path(str(calibration["artifact_path"])).exists()
    assert inspection["outcome_evaluation_count"] == 4
    assert inspection["calibration_run_count"] == 3
    assert inspection["calibration_slice_count"] == 8


@pytest.mark.unit
def test_phase6_service_requires_persisted_outcome_artifacts(tmp_path: Path) -> None:
    _store(tmp_path)
    service = Phase6Service(repo_root=tmp_path)

    with pytest.raises(ValueError, match="No persisted outcome evaluations"):
        service.phase6_load_outcome_evaluations(run_id=RUN_ID)
