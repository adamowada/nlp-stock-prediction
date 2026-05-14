from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import cast

import pytest

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

RUN_DATE = date(2026, 5, 13)


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
    assert [tool["stage"] for tool in tools] == list(PHASE4_STAGE_ORDER)
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
def test_phase4_tool_execution_rolls_back_new_artifacts_but_records_failure(
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
        assert failed_file.exists()
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
    assert not failed_file.exists()
    assert service.store.get_artifact("artifact-phase4-failed") is None
    failed_record = service.store.get_tool_run(exc_info.value.tool_run_id)
    assert failed_record is not None
    assert failed_record.status == "failed"
    assert failed_record.error_message == "deterministic phase4 failure"


@pytest.mark.integration
def test_phase4_report_is_final_only_and_warns_without_synthesizing_candidates(
    tmp_path: Path,
) -> None:
    service, run_id = _started_service(tmp_path)
    service.run_dummy_universe_tool(run_id=run_id, symbol="TSLA")

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
