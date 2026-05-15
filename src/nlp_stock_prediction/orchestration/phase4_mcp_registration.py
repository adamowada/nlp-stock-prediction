"""FastMCP registration adapter for Phase 4 research tools."""

from __future__ import annotations

from typing import Any

from nlp_stock_prediction.orchestration.phase4_service import Phase4Service
from nlp_stock_prediction.orchestration.report_data_modes import (
    LIVE_REPORT_DATA_MODE,
    OFFLINE_FIXTURE_REPORT_DATA_MODE,
    ReportDataMode,
)

PHASE4_MCP_TOOL_NAMES: tuple[str, ...] = (
    "start_research_run",
    "list_research_tool_plan",
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
    "inspect_research_run",
)


def register_phase4_mcp_tools(server: Any, service: Phase4Service) -> None:
    """Register Phase 4 tools on a FastMCP-compatible server."""

    def start_research_run(
        run_date: str,
        output_dir: str,
        symbol: str,
        objective: str | None = None,
        mode: str = OFFLINE_FIXTURE_REPORT_DATA_MODE,
    ) -> dict[str, object]:
        """Start a Phase 4 research run in live or offline_fixture mode."""

        return dict(
            service.start_research_run(
                run_date=run_date,
                output_dir=output_dir,
                symbol=symbol,
                objective=objective,
                report_data_mode=_mcp_report_data_mode(mode),
            )
        )

    def list_research_tool_plan() -> dict[str, object]:
        """List the real Phase 4 tools Codex should call."""

        return dict(service.list_research_tool_plan())

    def phase4_universe_discovery(run_id: str, symbol: str) -> dict[str, object]:
        """Resolve the instrument universe using the run's configured data mode."""

        return dict(service.phase4_universe_discovery(run_id=run_id, symbol=symbol))

    def phase4_market_data(run_id: str, symbol: str) -> dict[str, object]:
        """Write market-data artifacts using the run's configured data mode."""

        return dict(service.phase4_market_data(run_id=run_id, symbol=symbol))

    def phase4_technical_package(run_id: str, symbol: str) -> dict[str, object]:
        """Compute deterministic technical context from market data."""

        return dict(service.phase4_technical_package(run_id=run_id, symbol=symbol))

    def phase4_social_evidence(run_id: str, symbol: str) -> dict[str, object]:
        """Normalize social evidence using the run's configured data mode."""

        return dict(service.phase4_social_evidence(run_id=run_id, symbol=symbol))

    def phase4_news_catalyst(run_id: str, symbol: str) -> dict[str, object]:
        """Normalize news and catalyst evidence using the run's configured data mode."""

        return dict(service.phase4_news_catalyst(run_id=run_id, symbol=symbol))

    def phase4_fundamentals(run_id: str, symbol: str) -> dict[str, object]:
        """Write fundamentals evidence and analysis artifacts."""

        return dict(service.phase4_fundamentals(run_id=run_id, symbol=symbol))

    def phase4_sector_macro(run_id: str, symbol: str) -> dict[str, object]:
        """Write sector and macro baseline context artifacts."""

        return dict(service.phase4_sector_macro(run_id=run_id, symbol=symbol))

    def phase4_prediction_candidate_synthesis(run_id: str, symbol: str) -> dict[str, object]:
        """Synthesize conservative prediction candidates from stored evidence."""

        return dict(service.phase4_candidate_synthesis(run_id=run_id, symbol=symbol))

    def phase4_prediction_evaluation(run_id: str, symbol: str) -> dict[str, object]:
        """Evaluate stored prediction candidates as prediction-quality records."""

        return dict(service.phase4_prediction_evaluation(run_id=run_id, symbol=symbol))

    def render_prediction_report(run_id: str, symbol: str) -> dict[str, object]:
        """Render Markdown, JSON, and audit manifest files."""

        return dict(service.render_prediction_report(run_id=run_id, symbol=symbol))

    def inspect_research_run(run_id: str) -> dict[str, object]:
        """Inspect stored run graph counts for verification."""

        return dict(service.inspect_research_run(run_id=run_id))

    functions = {
        "start_research_run": start_research_run,
        "list_research_tool_plan": list_research_tool_plan,
        "phase4_universe_discovery": phase4_universe_discovery,
        "phase4_market_data": phase4_market_data,
        "phase4_technical_package": phase4_technical_package,
        "phase4_social_evidence": phase4_social_evidence,
        "phase4_news_catalyst": phase4_news_catalyst,
        "phase4_fundamentals": phase4_fundamentals,
        "phase4_sector_macro": phase4_sector_macro,
        "phase4_prediction_candidate_synthesis": phase4_prediction_candidate_synthesis,
        "phase4_prediction_evaluation": phase4_prediction_evaluation,
        "render_prediction_report": render_prediction_report,
        "inspect_research_run": inspect_research_run,
    }
    for tool_name in PHASE4_MCP_TOOL_NAMES:
        server.tool()(functions[tool_name])


def _mcp_report_data_mode(value: str) -> ReportDataMode:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"live", LIVE_REPORT_DATA_MODE}:
        return LIVE_REPORT_DATA_MODE
    if normalized in {"offline", "fixture", "offline_fixture", OFFLINE_FIXTURE_REPORT_DATA_MODE}:
        return OFFLINE_FIXTURE_REPORT_DATA_MODE
    raise ValueError("mode must be 'live' or 'offline_fixture'")


__all__ = ["PHASE4_MCP_TOOL_NAMES", "register_phase4_mcp_tools"]
