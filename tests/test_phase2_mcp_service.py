from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.orchestration import Phase2McpService


@pytest.mark.integration
def test_phase2_mcp_service_records_search_evidence_and_renders_report(tmp_path: Path) -> None:
    service = Phase2McpService(repo_root=tmp_path)

    started = service.start_research_run(
        run_date="2026-05-13",
        output_dir="reports/phase2-codex-smoke",
        symbol="TSLA",
        objective="Smoke-test TSLA with real Codex search evidence.",
    )
    run_id = str(started["run_id"])
    plan = service.list_research_tool_plan()
    tools = cast(list[dict[str, object]], plan["tools"])
    evidence = service.record_codex_search_evidence(
        run_id=run_id,
        symbol="TSLA",
        title="TSLA smoke source",
        url="https://example.com/tsla-smoke",
        claim="Example source says TSLA investors debated delivery expectations.",
        query="TSLA delivery expectations",
        published_at="2026-05-13T12:00:00Z",
    )
    universe = service.run_dummy_universe_tool(run_id=run_id, symbol="TSLA")
    analysis = service.run_dummy_analysis_tool(run_id=run_id, symbol="TSLA")
    candidate = service.synthesize_prediction_candidates(run_id=run_id, symbol="TSLA")
    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")
    inspected = service.inspect_research_run(run_id=run_id)

    assert [tool["tool_name"] for tool in tools] == [
        "record_codex_search_evidence",
        "run_dummy_universe_tool",
        "run_dummy_analysis_tool",
        "synthesize_prediction_candidates",
        "render_prediction_report",
    ]
    assert str(evidence["evidence_id"]).startswith("evidence-codex-search-")
    assert universe["instrument_ids"] == ["instrument:codex:TSLA"]
    assert str(analysis["artifact_id"]).startswith("artifact-dummy-analysis-")
    assert str(candidate["candidate_id"]).startswith("candidate-tsla-")
    assert inspected["has_codex_search_evidence"] is True
    assert inspected["evidence_count"] == 1
    assert inspected["candidate_count"] == 1

    markdown_path = Path(str(rendered["markdown_path"]))
    json_path = Path(str(rendered["json_path"]))
    audit_manifest_path = Path(str(rendered["audit_manifest_path"]))
    assert markdown_path.exists()
    assert json_path.exists()
    assert audit_manifest_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert payload["evidence_sources"][0]["metadata"]["codex_search"] is True
    assert (
        payload["prediction_candidates"][0]["evidence_for"][0]["evidence_id"]
        == evidence["evidence_id"]
    )


@pytest.mark.unit
def test_phase2_mcp_service_restricts_write_roots(tmp_path: Path) -> None:
    service = Phase2McpService(repo_root=tmp_path)

    with pytest.raises(ValueError, match="writes are limited"):
        service.start_research_run(
            run_date=date(2026, 5, 13).isoformat(),
            output_dir="../outside",
            symbol="TSLA",
        )
