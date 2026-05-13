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
    assert str(universe["tool_run_id"]) == f"tool-dummy-universe-{run_id}"
    assert str(universe["universe_id"]).startswith("phase3-fixture-universe-")
    assert "instrument:codex:TSLA" in universe["instrument_ids"]
    assert "instrument:etf:us:spy" in universe["instrument_ids"]
    assert "instrument:crypto:btc-usd" in universe["instrument_ids"]
    assert "instrument:futures:cme:esm6" in universe["instrument_ids"]
    assert universe["warnings"]
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
    audit_payload = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert payload["evidence_sources"][0]["metadata"]["codex_search"] is True
    assert (
        payload["prediction_candidates"][0]["evidence_for"][0]["evidence_id"]
        == evidence["evidence_id"]
    )
    assert any(
        artifact["artifact_type"] == "instrument_universe"
        for artifact in audit_payload["artifacts"]
    )

    universe_artifact = service.store.get_artifact(str(universe["artifact_id"]))
    assert universe_artifact is not None
    universe_payload = json.loads((tmp_path / universe_artifact.path).read_text(encoding="utf-8"))
    assert universe_payload["universe"]["request_id"] == universe["universe_id"]
    assert universe_payload["universe"]["warnings"] == universe["warnings"]
    assert service.store.get_instrument("instrument:codex:TSLA") is not None
    assert service.store.get_instrument("instrument:futures:cme:esm6") is not None


@pytest.mark.unit
def test_phase2_mcp_service_restricts_write_roots(tmp_path: Path) -> None:
    service = Phase2McpService(repo_root=tmp_path)

    with pytest.raises(ValueError, match="writes are limited"):
        service.start_research_run(
            run_date=date(2026, 5, 13).isoformat(),
            output_dir="../outside",
            symbol="TSLA",
        )


@pytest.mark.integration
def test_phase2_mcp_service_synthesizes_contradictory_evidence(tmp_path: Path) -> None:
    service = Phase2McpService(repo_root=tmp_path)
    started = service.start_research_run(
        run_date="2026-05-13",
        output_dir="reports/phase2-codex-smoke",
        symbol="BTC:USD",
    )
    run_id = str(started["run_id"])

    service.record_codex_search_evidence(
        run_id=run_id,
        symbol="BTC:USD",
        title="Supportive BTC source",
        url="https://example.com/btc-support",
        claim="Example source says BTC liquidity improved.",
        query="BTC liquidity",
        stance="supports",
    )
    against = service.record_codex_search_evidence(
        run_id=run_id,
        symbol="BTC:USD",
        title="Contradictory BTC source",
        url="https://example.com/btc-risk",
        claim="Example source says BTC downside risk increased.",
        query="BTC downside risk",
        stance="contradicts",
    )
    candidate = service.synthesize_prediction_candidates(run_id=run_id, symbol="BTC:USD")
    rendered = service.render_prediction_report(run_id=run_id, symbol="BTC:USD")

    payload = json.loads(Path(str(rendered["json_path"])).read_text(encoding="utf-8"))
    report_candidate = payload["prediction_candidates"][0]

    assert run_id == "codex-smoke-2026-05-13-btc-usd"
    assert str(candidate["candidate_id"]).startswith("candidate-btc-usd-")
    assert report_candidate["symbol"] == "BTC:USD"
    assert report_candidate["status"] == "contradicted"
    assert report_candidate["evidence_against"][0]["evidence_id"] == against["evidence_id"]
