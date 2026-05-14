from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.contracts import RunConfig
from nlp_stock_prediction.orchestration.phase4_service import (
    Phase4Service,
    build_phase4_tool_registry,
)
from nlp_stock_prediction.pipeline import generate_daily_report


class _FakeMcpServer:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def tool(self) -> Any:
        def decorator(func: Any) -> Any:
            self.registered.append(func.__name__)
            return func

        return decorator


@pytest.mark.unit
def test_phase4_registry_exposes_real_public_tool_suite() -> None:
    plan = build_phase4_tool_registry().as_plan()
    tools = cast(list[dict[str, object]], plan["tools"])
    tool_names = [str(tool["tool_name"]) for tool in tools]

    assert tool_names == [
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
    assert not any("dummy" in tool_name for tool_name in tool_names)


@pytest.mark.unit
def test_phase4_service_plan_and_mcp_registration_are_not_phase2_dummy_only(
    tmp_path: Path,
) -> None:
    from nlp_stock_prediction.orchestration.phase4_mcp_registration import (
        PHASE4_MCP_TOOL_NAMES,
        register_phase4_mcp_tools,
    )

    service = Phase4Service(repo_root=tmp_path)
    tools = cast(list[dict[str, object]], service.list_research_tool_plan()["tools"])
    plan_names = {str(tool["tool_name"]) for tool in tools}
    server = _FakeMcpServer()

    register_phase4_mcp_tools(server, service)

    assert {
        "phase4_universe_discovery",
        "phase4_market_data",
        "phase4_technical_package",
        "phase4_social_evidence",
        "phase4_news_catalyst",
        "phase4_fundamentals",
        "phase4_sector_macro",
        "phase4_prediction_candidate_synthesis",
        "phase4_prediction_evaluation",
        "render_prediction_report",
    }.issubset(plan_names)
    assert list(PHASE4_MCP_TOOL_NAMES) == server.registered
    assert "run_dummy_universe_tool" not in server.registered
    assert "run_dummy_analysis_tool" not in server.registered


@pytest.mark.integration
def test_offline_pipeline_runs_real_phase4_public_flow(tmp_path: Path) -> None:
    bundle = generate_daily_report(
        RunConfig(run_date="2026-05-11", output_dir=tmp_path / "reports", offline=True)
    )

    tool_records = cast(Any, bundle.tool_records)
    tool_names = {str(record.tool_name) for record in tool_records}
    payload = json.loads(bundle.json_path.read_text(encoding="utf-8"))

    assert {
        "phase4_universe_discovery",
        "phase4_market_data",
        "phase4_technical_package",
        "phase4_social_evidence",
        "phase4_news_catalyst",
        "phase4_fundamentals",
        "phase4_sector_macro",
        "phase4_prediction_candidate_synthesis",
        "phase4_prediction_evaluation",
        "render_prediction_report",
    }.issubset(tool_names)
    assert not any("dummy" in tool_name for tool_name in tool_names)
    assert payload["run_id"].startswith("phase4-")
    assert payload["prediction_candidates"]


@pytest.mark.integration
def test_offline_pipeline_supports_custom_output_and_symbol_isolation(tmp_path: Path) -> None:
    output_dir = tmp_path / "custom-output"
    tsla = generate_daily_report(
        RunConfig(
            run_date="2026-05-11",
            output_dir=output_dir,
            symbol="TSLA",
            offline=True,
        )
    )
    btc = generate_daily_report(
        RunConfig(
            run_date="2026-05-11",
            output_dir=output_dir,
            symbol="BTC:USD",
            offline=True,
        )
    )

    assert tsla.json_path.exists()
    assert btc.json_path.exists()
    assert tsla.json_path != btc.json_path
    assert tsla.json_path.parent.name == "tsla"
    assert btc.json_path.parent.name == "btc-usd"


@pytest.mark.integration
def test_pipeline_rejects_live_mode_until_production_orchestration_exists(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="live providers are not wired"):
        generate_daily_report(
            RunConfig(run_date="2026-05-11", output_dir=tmp_path / "reports", offline=False)
        )
