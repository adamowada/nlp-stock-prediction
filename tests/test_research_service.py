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
    RESEARCH_STAGE_ORDER,
    ResearchService,
    ResearchToolExecutionError,
    ResearchToolMetadata,
    ResearchToolRunContext,
    ResearchToolRunOutcome,
    execute_research_tool,
)
from nlp_stock_prediction.orchestration.research_service import (
    MISSING_CANDIDATE_WARNING,
    ResearchToolRunStatus,
)
from nlp_stock_prediction.storage import (
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ToolRunRecord,
)

RUN_DATE = date(2026, 5, 13)
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _started_service(tmp_path: Path) -> tuple[ResearchService, str]:
    service = ResearchService(repo_root=tmp_path)
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/research-runtime-report",
        symbol="TSLA",
        objective="Research Stage runtime/report foundation test.",
    )
    return service, str(started["run_id"])


@pytest.mark.unit
def test_research_tool_plan_is_registry_derived_with_canonical_stages(tmp_path: Path) -> None:
    service = ResearchService(repo_root=tmp_path)

    plan = service.list_research_tool_plan()
    tools = cast(list[dict[str, object]], plan["tools"])

    assert plan["stage_order"] == list(RESEARCH_STAGE_ORDER)
    assert [tool["tool_name"] for tool in tools] == [
        "research_universe_discovery",
        "research_market_data",
        "research_social_evidence",
        "research_news_catalyst",
        "research_fundamentals",
        "research_technical_package",
        "research_sector_macro",
        "research_prediction_candidate_synthesis",
        "research_prediction_evaluation",
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
def test_research_start_research_run_rejects_blank_symbol(tmp_path: Path) -> None:
    service = ResearchService(repo_root=tmp_path)

    with pytest.raises(ValueError, match="symbol must be non-empty"):
        service.start_research_run(
            run_date=RUN_DATE.isoformat(),
            output_dir="reports/research-runtime-report",
            symbol="   ",
        )


@pytest.mark.unit
def test_research_tool_execution_helper_records_terminal_statuses(tmp_path: Path) -> None:
    service, run_id = _started_service(tmp_path)
    paths = service.write_policy.run_paths(RUN_DATE, "reports/research-runtime-report")
    statuses = ("successful", "partial", "empty", "skipped")

    for status in statuses:
        tool = ResearchToolMetadata(
            tool_id=f"research.test.{status}",
            tool_name=f"{status}_tool",
            tool_version="unit.v1",
            stage="analyze",
            description=f"Return a {status} outcome.",
        )
        warnings = ("partial warning",) if status == "partial" else ()

        def action(
            _context: ResearchToolRunContext,
            *,
            status: str = status,
            warnings: tuple[str, ...] = warnings,
        ) -> ResearchToolRunOutcome:
            return ResearchToolRunOutcome(
                status=cast(ResearchToolRunStatus, status),
                payload={"case": status},
                warnings=warnings,
            )

        outcome = execute_research_tool(
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
def test_research_tool_execution_rolls_back_new_and_overwritten_artifacts_but_records_failure(
    tmp_path: Path,
) -> None:
    service, run_id = _started_service(tmp_path)
    paths = service.write_policy.run_paths(RUN_DATE, "reports/research-runtime-report")
    existing_file = paths.audit_dir / "preexisting.txt"
    existing_file.parent.mkdir(parents=True)
    existing_file.write_text("keep\n", encoding="utf-8")
    failed_file = paths.audit_dir / "failed-artifact.json"
    tool = ResearchToolMetadata(
        tool_id="research.test.failing",
        tool_name="failing_tool",
        tool_version="unit.v1",
        stage="analyze",
        description="Write an artifact and then fail.",
        artifact_kinds=("provider_result",),
    )

    def action(context: ResearchToolRunContext) -> ResearchToolRunOutcome:
        context.artifact_index(schema_version="unit-artifact.v1").write_json(
            artifact_id="artifact-research-failed",
            artifact_type="provider_result",
            filename=failed_file.name,
            payload={"will": "rollback"},
        )
        context.artifact_index(schema_version="unit-artifact.v1").write_text(
            artifact_id="artifact-research-overwrite",
            artifact_type="provider_result",
            filename=existing_file.name,
            content="overwritten\n",
        )
        assert failed_file.exists()
        assert existing_file.read_text(encoding="utf-8") == "overwritten\n"
        raise RuntimeError("deterministic research failure")

    with pytest.raises(ResearchToolExecutionError) as exc_info:
        execute_research_tool(
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
    assert service.store.get_artifact("artifact-research-failed") is None
    assert service.store.get_artifact("artifact-research-overwrite") is None
    failed_record = service.store.get_tool_run(exc_info.value.tool_run_id)
    assert failed_record is not None
    assert failed_record.status == "failed"
    assert failed_record.error_message == "deterministic research failure"


@pytest.mark.unit
def test_research_tool_failed_retry_preserves_previous_success_outputs(tmp_path: Path) -> None:
    service, run_id = _started_service(tmp_path)
    paths = service.write_policy.run_paths(RUN_DATE, "reports/research-runtime-report")
    tool = ResearchToolMetadata(
        tool_id="research.test.retry",
        tool_name="retry_tool",
        tool_version="unit.v1",
        stage="analyze",
        description="Exercise deterministic retry behavior.",
        artifact_kinds=("provider_result",),
    )

    def success_action(context: ResearchToolRunContext) -> ResearchToolRunOutcome:
        artifact = context.artifact_index(schema_version="unit-artifact.v1").write_json(
            artifact_id="artifact-research-retry-success",
            artifact_type="provider_result",
            filename="retry-success.json",
            payload={"attempt": "success"},
        )
        return ResearchToolRunOutcome(
            status="successful",
            payload={"attempt": "success"},
            artifact_ids=(artifact.artifact_id,),
        )

    first = execute_research_tool(
        store=service.store,
        repo_root=tmp_path,
        paths=paths,
        run_id=run_id,
        tool=tool,
        inputs={"case": "retry"},
        action=success_action,
    )
    first_tool_run_id = str(first.payload["tool_run_id"])

    def failing_action(context: ResearchToolRunContext) -> ResearchToolRunOutcome:
        context.artifact_index(schema_version="unit-artifact.v1").write_json(
            artifact_id="artifact-research-retry-failed",
            artifact_type="provider_result",
            filename="retry-failed.json",
            payload={"attempt": "failed"},
        )
        raise RuntimeError("retry failed after prior success")

    with pytest.raises(ResearchToolExecutionError) as exc_info:
        execute_research_tool(
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
    assert service.store.get_artifact("artifact-research-retry-success") is not None
    assert service.store.get_artifact("artifact-research-retry-failed") is None


@pytest.mark.integration
def test_research_report_is_final_only_and_warns_without_synthesizing_candidates(
    tmp_path: Path,
) -> None:
    service, run_id = _started_service(tmp_path)
    service.research_universe_discovery(run_id=run_id, symbol="TSLA")

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    assert rendered["status"] == "empty"
    assert rendered["warnings"] == [MISSING_CANDIDATE_WARNING]
    assert service.store.list_prediction_candidates_for_run(run_id) == ()
    tool_runs = service.store.list_tool_runs_for_run(run_id)
    assert "synthesize_prediction_candidates" not in {tool_run.tool_name for tool_run in tool_runs}
    report_tool_run = next(
        tool_run for tool_run in tool_runs if tool_run.tool_name == "render_prediction_report"
    )
    assert report_tool_run.tool_name == "render_prediction_report"
    assert report_tool_run.status == "empty"
    assert report_tool_run.warnings == (MISSING_CANDIDATE_WARNING,)

    payload = json.loads(Path(str(rendered["json_path"])).read_text(encoding="utf-8"))
    assert payload["prediction_candidates"] == []
    assert payload["insufficient_evidence_summary"] == MISSING_CANDIDATE_WARNING


@pytest.mark.integration
def test_research_prediction_evaluation_service_writes_artifact_and_metadata(
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
        evidence_id="evidence-research-service-support",
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="TSLA",
        text="Fixture catalyst supports scenario quality.",
        created_at=NOW,
        permalink="https://example.test/evidence-research-service-support",
        matched_tickers=("TSLA",),
        matched_instrument_ids=("instrument:equity:us:tsla",),
        instrument_id="instrument:equity:us:tsla",
        provenance=SourceProvenance(
            provider_name="fixture-news",
            source_kind=SourceKind.NEWS_ARTICLE,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=NOW,
            observed_at=NOW,
            source_url="https://example.test/evidence-research-service-support",
            permalink="https://example.test/evidence-research-service-support",
            raw_identifier="evidence-research-service-support",
            raw_snapshot_id="raw-evidence-research-service-support",
            freshness_status=FreshnessStatus.FRESH,
        ),
    )
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-research-service-evidence",
            run_id=run_id,
            tool_name="research_test_evidence",
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
            tool_run_id="tool-research-service-evidence",
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
            candidate_id="candidate-research-service-quality",
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

    result = service.research_prediction_evaluation(run_id=run_id, symbol="TSLA")

    assert result["status"] == "successful"
    assert result["evaluation_count"] == 1
    artifact_id = cast(list[str], result["artifact_ids"])[0]
    artifact = service.store.get_artifact(artifact_id)
    assert artifact is not None
    assert artifact.artifact_type == "prediction_evaluation"
    assert artifact.tool_run_id == result["tool_run_id"]
    candidate = service.store.get_prediction_candidate("candidate-research-service-quality")
    assert candidate is not None
    assert "prediction_evaluation" in candidate.metadata


@pytest.mark.integration
def test_offline_research_flow_degrades_unresolved_symbols_to_unavailable_candidate(
    tmp_path: Path,
) -> None:
    service = ResearchService(
        repo_root=tmp_path,
        fixture_root=Path(__file__).resolve().parents[1],
    )

    result = service.run_offline_research_flow(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/research-unresolved-symbol",
        symbol="NVDA",
    )

    run_id = str(result["run_id"])
    candidates = service.store.list_prediction_candidates_for_run(run_id)

    assert candidates
    assert candidates[0].status == "unavailable"
    assert candidates[0].instrument_id == "instrument:codex:NVDA"
    instrument = service.store.get_instrument("instrument:codex:NVDA")
    assert instrument is not None
    assert instrument.asset_class == "unknown"
    assert Path(str(cast(dict[str, object], result["report"])["json_path"])).exists()
