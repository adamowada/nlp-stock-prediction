from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.contracts import EvidenceRequest, RetrievalMethod, RunConfig
from nlp_stock_prediction.orchestration.research_fixture_providers import (
    ResearchFixtureProviderFactory,
    _StaticJsonTransport,
)
from nlp_stock_prediction.orchestration.research_service import (
    ResearchService,
    build_research_tool_registry,
)
from nlp_stock_prediction.pipeline import generate_daily_report
from nlp_stock_prediction.providers._base import ProviderTransportError


class _FakeMcpServer:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def tool(self) -> Any:
        def decorator(func: Any) -> Any:
            self.registered.append(func.__name__)
            return func

        return decorator


@pytest.mark.unit
def test_research_registry_exposes_real_public_tool_suite() -> None:
    plan = build_research_tool_registry().as_plan()
    tools = cast(list[dict[str, object]], plan["tools"])
    tool_names = [str(tool["tool_name"]) for tool in tools]

    assert tool_names == [
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
    assert not any("dummy" in tool_name for tool_name in tool_names)


@pytest.mark.unit
def test_research_service_plan_and_mcp_registration_are_not_codex_smoke_dummy_only(
    tmp_path: Path,
) -> None:
    from nlp_stock_prediction.orchestration.research_mcp_registration import (
        RESEARCH_MCP_TOOL_NAMES,
        register_research_mcp_tools,
    )

    service = ResearchService(repo_root=tmp_path)
    tools = cast(list[dict[str, object]], service.list_research_tool_plan()["tools"])
    plan_names = {str(tool["tool_name"]) for tool in tools}
    server = _FakeMcpServer()

    register_research_mcp_tools(server, service)

    assert {
        "research_universe_discovery",
        "research_market_data",
        "research_technical_package",
        "research_social_evidence",
        "research_news_catalyst",
        "research_fundamentals",
        "research_sector_macro",
        "research_prediction_candidate_synthesis",
        "research_prediction_evaluation",
        "render_prediction_report",
    }.issubset(plan_names)
    assert list(RESEARCH_MCP_TOOL_NAMES) == server.registered
    assert "research_candidate_synthesis" not in server.registered
    assert "run_dummy_universe_tool" not in server.registered
    assert "run_dummy_analysis_tool" not in server.registered


@pytest.mark.unit
def test_research_mcp_start_research_run_modes_are_explicit() -> None:
    from nlp_stock_prediction.orchestration.research_mcp_registration import _mcp_report_data_mode

    assert _mcp_report_data_mode("live") == "live"
    assert _mcp_report_data_mode("offline") == "offline_fixture"
    assert _mcp_report_data_mode("offline-fixture") == "offline_fixture"
    with pytest.raises(ValueError, match="live"):
        _mcp_report_data_mode("demo")


@pytest.mark.integration
def test_offline_pipeline_runs_real_research_public_flow(tmp_path: Path) -> None:
    bundle = generate_daily_report(
        RunConfig(run_date="2026-05-11", output_dir=tmp_path / "reports", offline=True)
    )

    tool_records = cast(Any, bundle.tool_records)
    tool_names = {str(record.tool_name) for record in tool_records}
    payload = json.loads(bundle.json_path.read_text(encoding="utf-8"))

    assert {
        "research_universe_discovery",
        "research_market_data",
        "research_technical_package",
        "research_social_evidence",
        "research_news_catalyst",
        "research_fundamentals",
        "research_sector_macro",
        "research_prediction_candidate_synthesis",
        "research_prediction_evaluation",
        "render_prediction_report",
    }.issubset(tool_names)
    assert not any("dummy" in tool_name for tool_name in tool_names)
    assert payload["run_id"].startswith("research-")
    assert payload["prediction_candidates"]


@pytest.mark.unit
def test_research_fixture_providers_mark_fixture_provenance() -> None:
    factory = ResearchFixtureProviderFactory(Path.cwd())
    request = EvidenceRequest(
        request_id="fixture-provenance-tsla",
        run_date="2026-05-11",
        tickers=("TSLA",),
        query="$TSLA",
        options={"reddit_search_terms": ["$TSLA"]},
        limit=10,
    )
    reddit_provider = factory.reddit_provider()
    news_provider = factory.news_providers("TSLA")[0]

    assert reddit_provider is not None
    reddit_result = reddit_provider.fetch_discussion(request)
    news_result = news_provider.fetch_articles(request)

    assert reddit_result.data is not None
    assert news_result.data is not None
    assert reddit_result.data[0].provenance.retrieval_method == RetrievalMethod.PUBLIC_SCRAPE
    assert news_result.data[0].provenance.retrieval_method == RetrievalMethod.FIXTURE


@pytest.mark.unit
def test_static_fixture_transport_rejects_unregistered_urls() -> None:
    transport = _StaticJsonTransport({"registered.example/v1": {"ok": True}})

    with pytest.raises(ProviderTransportError, match="No fixture JSON response"):
        transport.get_json("https://other.example/v1")


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
def test_pipeline_requires_explicit_live_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="source_mode='live' or live_providers=True"):
        generate_daily_report(
            RunConfig(run_date="2026-05-11", output_dir=tmp_path / "reports", offline=False)
        )


@pytest.mark.integration
def test_pipeline_rejects_invalid_fixture_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--fixture-dir"):
        generate_daily_report(
            RunConfig(
                run_date="2026-05-11",
                output_dir=tmp_path / "reports",
                fixture_dir=tmp_path / "missing-fixtures",
                offline=True,
            )
        )
