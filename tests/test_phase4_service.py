from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    FreshnessStatus,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
)
from nlp_stock_prediction.orchestration import (
    PHASE4_STAGE_ORDER,
    Phase4Service,
    Phase4ToolExecutionError,
    Phase4ToolMetadata,
    Phase4ToolRunContext,
    Phase4ToolRunOutcome,
    execute_phase4_tool,
)
from nlp_stock_prediction.orchestration.phase4_service import (
    MISSING_CANDIDATE_WARNING,
    Phase4ToolRunStatus,
)
from nlp_stock_prediction.storage import (
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ToolRunRecord,
)

RUN_DATE = date(2026, 5, 13)
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _started_service(tmp_path: Path) -> tuple[Phase4Service, str]:
    service = Phase4Service(repo_root=tmp_path)
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/phase4-runtime-report",
        symbol="TSLA",
        objective="Phase 4 runtime/report foundation test.",
    )
    return service, str(started["run_id"])


@pytest.mark.unit
def test_phase4_tool_plan_is_registry_derived_with_canonical_stages(tmp_path: Path) -> None:
    service = Phase4Service(repo_root=tmp_path)

    plan = service.list_research_tool_plan()
    tools = cast(list[dict[str, object]], plan["tools"])

    assert plan["stage_order"] == list(PHASE4_STAGE_ORDER)
    assert [tool["tool_name"] for tool in tools] == [
        "phase4_universe_discovery",
        "phase4_market_data",
        "phase4_social_evidence",
        "phase4_news_catalyst",
        "phase4_fundamentals",
        "phase4_technical_package",
        "phase4_sector_macro",
        "phase4_prediction_candidate_synthesis",
        "phase4_prediction_evaluation",
        "render_prediction_report",
    ]
    assert [tool["stage"] for tool in tools] == [
        "discover",
        "collect",
        "collect",
        "collect",
        "collect",
        "analyze",
        "analyze",
        "evaluate",
        "evaluate",
        "report",
    ]
    for tool in tools:
        assert {
            "tool_id",
            "tool_name",
            "version",
            "tool_version",
            "stage",
            "description",
            "artifact_kinds",
            "offline_capable",
            "live_capable",
            "requires_network",
            "dependencies",
        }.issubset(tool)
        assert tool["version"] == tool["tool_version"]

    report_tool = tools[-1]
    assert report_tool["tool_name"] == "render_prediction_report"
    assert report_tool["artifact_kinds"] == [
        "markdown_report",
        "json_report",
        "audit_manifest",
    ]
    assert cast(dict[str, object], report_tool["metadata"])["final_only"] is True


@pytest.mark.unit
def test_phase4_start_research_run_rejects_blank_symbol(tmp_path: Path) -> None:
    service = Phase4Service(repo_root=tmp_path)

    with pytest.raises(ValueError, match="symbol must be non-empty"):
        service.start_research_run(
            run_date=RUN_DATE.isoformat(),
            output_dir="reports/phase4-runtime-report",
            symbol="   ",
        )


@pytest.mark.unit
def test_phase4_tool_execution_helper_records_terminal_statuses(tmp_path: Path) -> None:
    service, run_id = _started_service(tmp_path)
    paths = service.write_policy.run_paths(RUN_DATE, "reports/phase4-runtime-report")
    statuses = ("successful", "partial", "empty", "skipped")

    for status in statuses:
        tool = Phase4ToolMetadata(
            tool_id=f"phase4.test.{status}",
            tool_name=f"{status}_tool",
            tool_version="unit.v1",
            stage="analyze",
            description=f"Return a {status} outcome.",
        )
        warnings = ("partial warning",) if status == "partial" else ()

        def action(
            _context: Phase4ToolRunContext,
            *,
            status: str = status,
            warnings: tuple[str, ...] = warnings,
        ) -> Phase4ToolRunOutcome:
            return Phase4ToolRunOutcome(
                status=cast(Phase4ToolRunStatus, status),
                payload={"case": status},
                warnings=warnings,
            )

        outcome = execute_phase4_tool(
            store=service.store,
            repo_root=tmp_path,
            paths=paths,
            run_id=run_id,
            tool=tool,
            inputs={"case": status},
            action=action,
        )

        assert outcome.payload["status"] == status
        assert outcome.payload["tool_run_id"]

    records = service.store.list_tool_runs_for_run(run_id)
    assert len(records) == len(statuses)
    assert {record.status for record in records} == set(statuses)
    assert all(record.completed_at is not None for record in records)


@pytest.mark.unit
def test_phase4_tool_execution_rolls_back_new_and_overwritten_artifacts_but_records_failure(
    tmp_path: Path,
) -> None:
    service, run_id = _started_service(tmp_path)
    paths = service.write_policy.run_paths(RUN_DATE, "reports/phase4-runtime-report")
    existing_file = paths.audit_dir / "preexisting.txt"
    existing_file.parent.mkdir(parents=True)
    existing_file.write_text("keep\n", encoding="utf-8")
    failed_file = paths.audit_dir / "failed-artifact.json"
    tool = Phase4ToolMetadata(
        tool_id="phase4.test.failing",
        tool_name="failing_tool",
        tool_version="unit.v1",
        stage="analyze",
        description="Write an artifact and then fail.",
        artifact_kinds=("provider_result",),
    )

    def action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
        context.artifact_index(schema_version="unit-artifact.v1").write_json(
            artifact_id="artifact-phase4-failed",
            artifact_type="provider_result",
            filename=failed_file.name,
            payload={"will": "rollback"},
        )
        context.artifact_index(schema_version="unit-artifact.v1").write_text(
            artifact_id="artifact-phase4-overwrite",
            artifact_type="provider_result",
            filename=existing_file.name,
            content="overwritten\n",
        )
        assert failed_file.exists()
        assert existing_file.read_text(encoding="utf-8") == "overwritten\n"
        raise RuntimeError("deterministic phase4 failure")

    with pytest.raises(Phase4ToolExecutionError) as exc_info:
        execute_phase4_tool(
            store=service.store,
            repo_root=tmp_path,
            paths=paths,
            run_id=run_id,
            tool=tool,
            inputs={"case": "failure"},
            action=action,
        )

    assert existing_file.exists()
    assert existing_file.read_text(encoding="utf-8") == "keep\n"
    assert not failed_file.exists()
    assert service.store.get_artifact("artifact-phase4-failed") is None
    assert service.store.get_artifact("artifact-phase4-overwrite") is None
    failed_record = service.store.get_tool_run(exc_info.value.tool_run_id)
    assert failed_record is not None
    assert failed_record.status == "failed"
    assert failed_record.error_message == "deterministic phase4 failure"


@pytest.mark.unit
def test_phase4_tool_failed_retry_preserves_previous_success_outputs(tmp_path: Path) -> None:
    service, run_id = _started_service(tmp_path)
    paths = service.write_policy.run_paths(RUN_DATE, "reports/phase4-runtime-report")
    tool = Phase4ToolMetadata(
        tool_id="phase4.test.retry",
        tool_name="retry_tool",
        tool_version="unit.v1",
        stage="analyze",
        description="Exercise deterministic retry behavior.",
        artifact_kinds=("provider_result",),
    )

    def success_action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
        artifact = context.artifact_index(schema_version="unit-artifact.v1").write_json(
            artifact_id="artifact-phase4-retry-success",
            artifact_type="provider_result",
            filename="retry-success.json",
            payload={"attempt": "success"},
        )
        return Phase4ToolRunOutcome(
            status="successful",
            payload={"attempt": "success"},
            artifact_ids=(artifact.artifact_id,),
        )

    first = execute_phase4_tool(
        store=service.store,
        repo_root=tmp_path,
        paths=paths,
        run_id=run_id,
        tool=tool,
        inputs={"case": "retry"},
        action=success_action,
    )
    first_tool_run_id = str(first.payload["tool_run_id"])

    def failing_action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
        context.artifact_index(schema_version="unit-artifact.v1").write_json(
            artifact_id="artifact-phase4-retry-failed",
            artifact_type="provider_result",
            filename="retry-failed.json",
            payload={"attempt": "failed"},
        )
        raise RuntimeError("retry failed after prior success")

    with pytest.raises(Phase4ToolExecutionError) as exc_info:
        execute_phase4_tool(
            store=service.store,
            repo_root=tmp_path,
            paths=paths,
            run_id=run_id,
            tool=tool,
            inputs={"case": "retry"},
            action=failing_action,
        )

    failed_tool_run_id = exc_info.value.tool_run_id
    assert failed_tool_run_id != first_tool_run_id
    first_record = service.store.get_tool_run(first_tool_run_id)
    assert first_record is not None
    assert first_record.status == "successful"
    failed_record = service.store.get_tool_run(failed_tool_run_id)
    assert failed_record is not None
    assert failed_record.status == "failed"
    assert service.store.get_artifact("artifact-phase4-retry-success") is not None
    assert service.store.get_artifact("artifact-phase4-retry-failed") is None


@pytest.mark.integration
def test_phase4_report_is_final_only_and_warns_without_synthesizing_candidates(
    tmp_path: Path,
) -> None:
    service, run_id = _started_service(tmp_path)
    service.phase4_universe_discovery(run_id=run_id, symbol="TSLA")

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    assert rendered["status"] == "empty"
    assert rendered["warnings"] == [MISSING_CANDIDATE_WARNING]
    assert service.store.list_prediction_candidates_for_run(run_id) == ()
    tool_runs = service.store.list_tool_runs_for_run(run_id)
    assert "synthesize_prediction_candidates" not in {tool_run.tool_name for tool_run in tool_runs}
    report_tool_run = tool_runs[-1]
    assert report_tool_run.tool_name == "render_prediction_report"
    assert report_tool_run.status == "empty"
    assert report_tool_run.warnings == (MISSING_CANDIDATE_WARNING,)

    payload = json.loads(Path(str(rendered["json_path"])).read_text(encoding="utf-8"))
    assert payload["prediction_candidates"] == []
    assert payload["insufficient_evidence_summary"] == MISSING_CANDIDATE_WARNING


@pytest.mark.integration
def test_phase4_prediction_evaluation_service_writes_artifact_and_metadata(
    tmp_path: Path,
) -> None:
    service, run_id = _started_service(tmp_path)
    service.store.upsert_instrument(
        InstrumentRecord(
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            asset_class="stock",
            name="Tesla Inc.",
        )
    )
    evidence = SourceEvidence(
        evidence_id="evidence-phase4-service-support",
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="TSLA",
        text="Fixture catalyst supports scenario quality.",
        created_at=NOW,
        permalink="https://example.test/evidence-phase4-service-support",
        matched_tickers=("TSLA",),
        matched_instrument_ids=("instrument:equity:us:tsla",),
        instrument_id="instrument:equity:us:tsla",
        provenance=SourceProvenance(
            provider_name="fixture-news",
            source_kind=SourceKind.NEWS_ARTICLE,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=NOW,
            observed_at=NOW,
            source_url="https://example.test/evidence-phase4-service-support",
            permalink="https://example.test/evidence-phase4-service-support",
            raw_identifier="evidence-phase4-service-support",
            raw_snapshot_id="raw-evidence-phase4-service-support",
            freshness_status=FreshnessStatus.FRESH,
        ),
    )
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-phase4-service-evidence",
            run_id=run_id,
            tool_name="phase4_test_evidence",
            tool_version="test.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "TSLA"},
        )
    )
    service.store.record_evidence(
        EvidenceRecord(
            evidence_id=evidence.evidence_id,
            tool_run_id="tool-phase4-service-evidence",
            source_type=SourceKind.NEWS_ARTICLE.value,
            provider="fixture-news",
            retrieved_at=NOW,
            published_at=NOW,
            instruments=("instrument:equity:us:tsla",),
            claim=evidence.text,
            freshness_status=FreshnessStatus.FRESH.value,
            metadata={"source_evidence": evidence.model_dump(mode="json")},
        )
    )
    service.store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-phase4-service-quality",
            run_id=run_id,
            instrument_id="instrument:equity:us:tsla",
            prediction_horizon="swing",
            prediction_type="directional",
            scenario="TSLA fixture scenario quality depends on attributed evidence.",
            status="evidence_supported",
            confidence=0.62,
            direction="mixed",
            evidence_for=(evidence.evidence_id,),
            baseline={"summary": "No directional edge is assumed without source-backed evidence."},
            uncertainty="Fixture sources are deterministic test inputs.",
            metadata={"symbol": "TSLA"},
        )
    )

    result = service.phase4_prediction_evaluation(run_id=run_id, symbol="TSLA")

    assert result["status"] == "successful"
    assert result["evaluation_count"] == 1
    artifact_id = cast(list[str], result["artifact_ids"])[0]
    artifact = service.store.get_artifact(artifact_id)
    assert artifact is not None
    assert artifact.artifact_type == "prediction_evaluation"
    assert artifact.tool_run_id == result["tool_run_id"]
    candidate = service.store.get_prediction_candidate("candidate-phase4-service-quality")
    assert candidate is not None
    assert "prediction_evaluation" in candidate.metadata
