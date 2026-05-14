from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from nlp_stock_prediction.contracts import (
    BaselineComparison,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    PredictionOutcomeResult,
    PredictionType,
    SignalArtifactFamily,
    SignalArtifactReference,
    SourceKind,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.evaluation import (
    PredictionEvaluationTarget,
)
from nlp_stock_prediction.evaluation.outcomes import (
    write_point_in_time_outcome_evaluation_artifacts,
)
from nlp_stock_prediction.orchestration.phase6_service import (
    Phase6Service,
    build_phase6_tool_registry,
)
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    SQLiteStore,
    ToolRunRecord,
)

RUN_ID = "run-phase6-public-tools"
INSTRUMENT_ID = "instrument:equity:us:msft"
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 5, 13, 20, 0, tzinfo=UTC)
WINDOW_START = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 18, 20, 0, tzinfo=UTC)
OBSERVED_AT = datetime(2026, 5, 18, 20, 5, tzinfo=UTC)
EVALUATED_AT = datetime(2026, 5, 18, 21, 0, tzinfo=UTC)
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


def _persist_candidate_inputs(
    *,
    store: SQLiteStore,
    target: PredictionEvaluationTarget,
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
    for evidence_id in target.evidence_ids:
        store.record_evidence(
            EvidenceRecord(
                evidence_id=evidence_id,
                source_type=SourceKind.NEWS_ARTICLE.value,
                provider="verified-news",
                retrieved_at=target.point_in_time_cutoff,
                published_at=target.point_in_time_cutoff,
                instruments=(target.instrument_id,),
                claim=f"{target.symbol} support evidence existed before the cutoff.",
                freshness_status=FreshnessStatus.FRESH.value,
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
        _persist_candidate_inputs(
            store=store,
            target=target,
        )
        store.record_evidence(
            EvidenceRecord(
                evidence_id=f"evidence-{target.candidate_id}-outcome",
                source_type=SourceKind.MARKET_DATA.value,
                provider="verified-market-data",
                retrieved_at=evaluated_at,
                published_at=evaluated_at,
                instruments=(target.instrument_id,),
                claim=f"{target.symbol} outcome evidence was observed after the window.",
                freshness_status=FreshnessStatus.FRESH.value,
            )
        )
        write_point_in_time_outcome_evaluation_artifacts(
            store=store,
            repo_root=tmp_path,
            artifact_dir=audit_dir,
            run_id=RUN_ID,
            target=target,
            observed_result=(
                PredictionOutcomeResult.SUPPORTED
                if quality_score >= 0.5
                else PredictionOutcomeResult.CONTRADICTED
            ),
            observed_at=evaluated_at - timedelta(minutes=20),
            result_summary="Outcome was reviewed against the frozen prediction target.",
            outcome_evidence=(EvidenceReference(evidence_id=f"evidence-{candidate_id}-outcome"),),
            created_at=evaluated_at - timedelta(minutes=20),
            evaluated_at=evaluated_at,
        )
    return audit_dir


@pytest.mark.unit
def test_phase6_registry_exposes_live_evaluation_tool_suite() -> None:
    plan = build_phase6_tool_registry().as_plan()
    tools = cast(list[dict[str, object]], plan["tools"])
    tool_names = [str(tool["tool_name"]) for tool in tools]

    assert tool_names == [
        "evaluation_materialize_outcome",
        "evaluation_load_outcomes",
        "evaluation_ablation",
        "evaluation_walk_forward",
        "evaluation_outcome_summary",
        "evaluation_stale_artifacts",
        "evaluation_source_reliability",
        "evaluation_provider_playbook",
        "evaluation_calibration",
        "evaluation_calibration_drift",
        "evaluation_inspect",
    ]
    assert not any("dummy" in tool_name for tool_name in tool_names)
    requires_network = {str(tool["tool_name"]): bool(tool["requires_network"]) for tool in tools}
    assert requires_network == {
        "evaluation_materialize_outcome": True,
        "evaluation_load_outcomes": False,
        "evaluation_ablation": False,
        "evaluation_walk_forward": False,
        "evaluation_outcome_summary": False,
        "evaluation_stale_artifacts": False,
        "evaluation_source_reliability": False,
        "evaluation_provider_playbook": False,
        "evaluation_calibration": False,
        "evaluation_calibration_drift": False,
        "evaluation_inspect": False,
    }


@pytest.mark.unit
def test_orchestration_facade_lazily_exports_phase6_service() -> None:
    from nlp_stock_prediction.orchestration import (
        Phase6Service as ExportedPhase6Service,
    )
    from nlp_stock_prediction.orchestration import (
        build_phase6_tool_registry as exported_registry_factory,
    )

    assert ExportedPhase6Service is Phase6Service
    assert exported_registry_factory().as_plan()["tools"]


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
def test_codex_mcp_server_registers_phase4_and_phase6_tooling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlp_stock_prediction.codex_mcp import build_server
    from nlp_stock_prediction.orchestration.phase4_mcp_registration import PHASE4_MCP_TOOL_NAMES
    from nlp_stock_prediction.orchestration.phase6_mcp_registration import PHASE6_MCP_TOOL_NAMES

    fastmcp_module = ModuleType("mcp.server.fastmcp")

    class _FastMCP(_FakeMcpServer):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    fastmcp_module.FastMCP = _FastMCP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp", ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)

    server = build_server(repo_root=tmp_path, database_path=Path("data/test.sqlite3"))

    assert server.registered == [*PHASE4_MCP_TOOL_NAMES, *PHASE6_MCP_TOOL_NAMES]


@pytest.mark.unit
def test_phase6_public_outcome_tool_writes_persisted_review(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = _target("candidate-service-outcome", score=0.64)
    _persist_candidate_inputs(store=store, target=target)
    store.record_evidence(
        EvidenceRecord(
            evidence_id=f"evidence-{target.candidate_id}-outcome",
            source_type=SourceKind.MARKET_DATA.value,
            provider="verified-market-data",
            retrieved_at=EVALUATED_AT,
            published_at=EVALUATED_AT,
            instruments=(target.instrument_id,),
            claim=f"{target.symbol} outcome evidence was observed after the window.",
            freshness_status=FreshnessStatus.FRESH.value,
        )
    )
    service = Phase6Service(repo_root=tmp_path)

    result = service.phase6_point_in_time_outcome_evaluation(
        run_id=RUN_ID,
        candidate_id=target.candidate_id,
        point_in_time_cutoff=CUTOFF.isoformat(),
        evaluation_window_start=WINDOW_START.isoformat(),
        evaluation_window_end=WINDOW_END.isoformat(),
        observed_result="supported",
        observed_at=OBSERVED_AT.isoformat(),
        result_summary="MSFT closed above the comparison baseline.",
        outcome_evidence_ids=(f"evidence-{target.candidate_id}-outcome",),
        created_at=OBSERVED_AT.isoformat(),
        evaluated_at=EVALUATED_AT.isoformat(),
    )

    assert result["status"] == "confirmed"
    assert result["quality_score"] == 1.0
    assert Path(str(result["outcome_evaluation_artifact_path"])).exists()
    assert store.get_prediction_outcome_evaluation(str(result["outcome_evaluation_id"])) is not None


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
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-phase6-report-integration-evidence",
            run_id=RUN_ID,
            tool_name="phase6_report_integration_evidence",
            tool_version="test.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={},
        )
    )
    inspection = service.inspect_phase6_run(run_id=RUN_ID)

    assert loaded["outcome_evaluation_count"] == 4
    assert [
        str(item).split("-")[-2]
        for item in cast(list[object], loaded["source_outcome_evaluation_ids"])
    ] == ["001", "002", "003", "004"]
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
    assert inspection["phase6_tool_run_count"] == 7


@pytest.mark.unit
def test_phase6_service_requires_persisted_outcome_artifacts(tmp_path: Path) -> None:
    _store(tmp_path)
    service = Phase6Service(repo_root=tmp_path)

    with pytest.raises(ValueError, match="No persisted outcome evaluations"):
        service.phase6_load_outcome_evaluations(run_id=RUN_ID)


@pytest.mark.unit
def test_phase6_inferred_artifact_dir_still_uses_write_policy(tmp_path: Path) -> None:
    _persist_outcome_sources(tmp_path)
    service = Phase6Service(repo_root=tmp_path)
    loaded = service.phase6_load_outcome_evaluations(run_id=RUN_ID)
    source_artifact_id = str(cast(list[object], loaded["source_artifact_ids"])[0])
    source_artifact = service.store.get_artifact(source_artifact_id)
    assert source_artifact is not None
    bad_audit_dir = tmp_path / "src" / "audit"
    bad_audit_dir.mkdir(parents=True)
    bad_artifact_path = bad_audit_dir / Path(source_artifact.path).name
    original_artifact_path = (
        source_artifact.path
        if source_artifact.path.is_absolute()
        else tmp_path / source_artifact.path
    )
    bad_artifact_path.write_bytes(original_artifact_path.read_bytes())
    service.store.record_artifact(
        ArtifactRecord(
            **{
                **source_artifact.__dict__,
                "path": bad_artifact_path.relative_to(tmp_path),
            }
        )
    )

    with pytest.raises(ValueError, match="Research artifact writes are limited"):
        service.phase6_calibration_summary(
            run_id=RUN_ID,
            cohort_id="phase6-evalcal-disallowed-dir",
            as_of=AS_OF.isoformat(),
            bin_edges=(0.0, 0.5, 1.0),
        )
