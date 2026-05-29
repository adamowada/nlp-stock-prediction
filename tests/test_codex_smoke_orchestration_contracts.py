from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    CodexEvidenceImport,
    DataReference,
    FreshnessStatus,
    OrchestratorRunSummary,
    ResearchObjective,
    ResearchToolSpec,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TimeHorizon,
    ToolExecutionResult,
    ToolInvocation,
)

pytestmark = pytest.mark.schema

RUN_DATE = date(2026, 5, 13)
NOW = datetime(2026, 5, 13, 15, 0, tzinfo=UTC)
LATER = datetime(2026, 5, 13, 15, 1, tzinfo=UTC)


def _objective(*, offline: bool = True) -> ResearchObjective:
    return ResearchObjective(
        objective_id="objective-tsla-2026-05-13",
        run_date=RUN_DATE,
        prompt="Research TSLA prediction scenarios for the next weekly horizon.",
        requested_symbols=("tsla",),
        time_horizon=TimeHorizon.WEEKLY,
        offline=offline,
    )


def _tool_spec() -> ResearchToolSpec:
    return ResearchToolSpec(
        tool_name="fixture-news-screen",
        tool_version="1.0",
        description="Read recorded news fixtures and write normalized evidence artifacts.",
        output_contract="tuple[SourceEvidence, ...]",
        artifact_kinds=("raw_snapshot", "normalized_evidence"),
    )


def _invocation(
    *,
    invocation_id: str = "invoke-news-screen",
    objective_id: str = "objective-tsla-2026-05-13",
    mode: str = "offline",
) -> ToolInvocation:
    return ToolInvocation(
        invocation_id=invocation_id,
        objective_id=objective_id,
        tool_name="fixture-news-screen",
        tool_version="1.0",
        requested_at=NOW,
        arguments={"query": "TSLA", "limit": 5},
        mode=mode,
    )


def _artifact(
    reference_id: str = "artifact-news-normalized",
    reference_type: str = "normalized_evidence",
) -> DataReference:
    return DataReference(
        reference_id=reference_id,
        reference_type=reference_type,
        path=f"artifacts/{reference_id}.json",
        sha256="0" * 64,
        metadata={"fixture": True},
    )


def _provenance() -> SourceProvenance:
    return SourceProvenance(
        provider_name="codex-fixture-search",
        source_kind=SourceKind.NEWS_ARTICLE,
        retrieval_method=RetrievalMethod.FIXTURE,
        fetched_at=NOW,
        observed_at=NOW,
        source_url="https://example.invalid/news/tsla-catalyst",
        permalink="https://example.invalid/news/tsla-catalyst",
        raw_identifier="fixture-news-tsla-catalyst",
        raw_snapshot_id="artifact-news-raw",
        query="TSLA catalyst",
        freshness_status=FreshnessStatus.FRESH,
        provider_metadata={"fixture": True},
    )


def _evidence(evidence_id: str = "evidence-news-tsla-catalyst") -> SourceEvidence:
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="TSLA",
        title="TSLA fixture catalyst note",
        text="Fixture article states that analysts debated TSLA delivery expectations.",
        created_at=NOW,
        permalink="https://example.invalid/news/tsla-catalyst",
        matched_tickers=("TSLA",),
        provenance=_provenance(),
        metadata={"source_rank": 1},
    )


def _result(
    invocation: ToolInvocation | None = None,
    *,
    status: str = "succeeded",
) -> ToolExecutionResult:
    return ToolExecutionResult(
        result_id=f"result-{status}",
        invocation=invocation or _invocation(),
        status=status,
        started_at=NOW,
        completed_at=LATER,
        artifact_refs=(_artifact(),),
        evidence_ids=("evidence-news-tsla-catalyst",),
        warnings=("Fixture path skipped one malformed article.",) if status == "partial" else (),
        error_message="fixture tool failed" if status == "failed" else None,
    )


def _evidence_import() -> CodexEvidenceImport:
    return CodexEvidenceImport(
        import_id="import-codex-news-evidence",
        objective_id="objective-tsla-2026-05-13",
        imported_at=LATER,
        evidence=(_evidence(),),
        artifact_refs=(_artifact(),),
        originating_invocation_id="invoke-news-screen",
    )


def test_objective_and_tool_spec_default_to_offline_deterministic_contracts() -> None:
    objective = ResearchObjective(
        objective_id="objective-broad-symbols",
        run_date=RUN_DATE,
        prompt="Research TSLA and BTC/USD without live calls.",
        requested_symbols=("tsla", "btc/usd"),
    )
    tool = _tool_spec()
    invocation = _invocation()

    assert objective.requested_symbols == ("TSLA", "BTC/USD")
    assert objective.offline is True
    assert objective.max_tool_calls == 8
    assert objective.require_citations is True
    assert tool.offline_capable is True
    assert tool.deterministic is True
    assert tool.live_capable is False
    assert invocation.mode == "offline"
    assert ToolInvocation.model_validate_json(invocation.model_dump_json()) == invocation

    with pytest.raises(ValidationError, match="network-dependent tools"):
        ResearchToolSpec(
            tool_name="live-search",
            tool_version="1.0",
            description="Network search.",
            requires_network=True,
        )


def test_tool_execution_result_preserves_artifacts_and_status_semantics() -> None:
    result = _result()

    assert result.invocation.mode == "offline"
    assert result.artifact_refs[0].reference_type == "normalized_evidence"
    assert result.evidence_ids == ("evidence-news-tsla-catalyst",)

    with pytest.raises(ValidationError, match="artifact_refs or evidence_ids"):
        ToolExecutionResult(
            result_id="result-empty-success",
            invocation=_invocation(),
            status="succeeded",
            started_at=NOW,
            completed_at=LATER,
        )

    with pytest.raises(ValidationError, match="partial tool results require warnings"):
        ToolExecutionResult(
            result_id="result-partial-no-warning",
            invocation=_invocation(),
            status="partial",
            started_at=NOW,
            completed_at=LATER,
            artifact_refs=(_artifact(),),
        )

    with pytest.raises(ValidationError, match="failed tool results require error_message"):
        ToolExecutionResult(
            result_id="result-failed-no-error",
            invocation=_invocation(),
            status="failed",
            started_at=NOW,
            completed_at=LATER,
        )

    with pytest.raises(ValidationError, match="completed_at"):
        ToolExecutionResult(
            result_id="result-time-travel",
            invocation=_invocation(),
            status="empty",
            started_at=LATER,
            completed_at=NOW,
        )

    with pytest.raises(ValidationError, match="evidence_ids must be unique"):
        ToolExecutionResult(
            result_id="result-duplicate-evidence",
            invocation=_invocation(),
            status="succeeded",
            started_at=NOW,
            completed_at=LATER,
            evidence_ids=("evidence-1", "evidence-1"),
        )


def test_codex_evidence_import_requires_evidence_and_artifact_traceability() -> None:
    evidence_import = _evidence_import()
    dumped = evidence_import.model_dump(mode="json")

    assert dumped["evidence"][0]["provenance"]["provider_name"] == "codex-fixture-search"
    assert dumped["artifact_refs"][0]["reference_id"] == "artifact-news-normalized"

    with pytest.raises(ValidationError, match="require evidence"):
        CodexEvidenceImport(
            import_id="import-without-evidence",
            objective_id="objective-tsla-2026-05-13",
            imported_at=NOW,
            evidence=(),
            artifact_refs=(_artifact(),),
        )

    with pytest.raises(ValidationError, match="require artifact_refs"):
        CodexEvidenceImport(
            import_id="import-without-artifacts",
            objective_id="objective-tsla-2026-05-13",
            imported_at=NOW,
            evidence=(_evidence(),),
            artifact_refs=(),
        )

    with pytest.raises(ValidationError, match="evidence ids must be unique"):
        CodexEvidenceImport(
            import_id="import-duplicate-evidence",
            objective_id="objective-tsla-2026-05-13",
            imported_at=NOW,
            evidence=(_evidence("evidence-1"), _evidence("evidence-1")),
            artifact_refs=(_artifact(),),
        )


def test_orchestrator_run_summary_links_objective_tools_results_and_imports() -> None:
    objective = _objective()
    tool = _tool_spec()
    invocation = _invocation()
    result = _result(invocation)
    evidence_import = _evidence_import()
    report_artifact = _artifact("artifact-report-markdown", "report")

    summary = OrchestratorRunSummary(
        run_id="run-tsla-2026-05-13",
        objective=objective,
        started_at=NOW,
        completed_at=LATER,
        status="succeeded",
        selected_tools=(tool,),
        invocations=(invocation,),
        tool_results=(result,),
        evidence_imports=(evidence_import,),
        report_artifacts=(report_artifact,),
        prediction_candidate_ids=("candidate-tsla-weekly",),
    )

    assert summary.objective.objective_id == "objective-tsla-2026-05-13"
    assert summary.tool_results[0].invocation == invocation
    assert summary.model_dump(mode="json")["status"] == "succeeded"

    unknown_invocation = _invocation(invocation_id="invoke-unknown")
    with pytest.raises(ValidationError, match="tool results must reference run invocations"):
        OrchestratorRunSummary(
            run_id="run-unknown-result",
            objective=objective,
            started_at=NOW,
            completed_at=LATER,
            status="succeeded",
            selected_tools=(tool,),
            invocations=(invocation,),
            tool_results=(_result(unknown_invocation),),
        )

    live_invocation = _invocation(invocation_id="invoke-live", mode="live")
    with pytest.raises(ValidationError, match="offline objectives cannot use live"):
        OrchestratorRunSummary(
            run_id="run-live-offline",
            objective=objective,
            started_at=NOW,
            completed_at=LATER,
            status="skipped",
            selected_tools=(tool,),
            invocations=(live_invocation,),
        )

    with pytest.raises(ValidationError, match="prediction candidates require evidence"):
        OrchestratorRunSummary(
            run_id="run-candidate-without-trace",
            objective=objective,
            started_at=NOW,
            completed_at=LATER,
            status="succeeded",
            prediction_candidate_ids=("candidate-tsla-weekly",),
        )
