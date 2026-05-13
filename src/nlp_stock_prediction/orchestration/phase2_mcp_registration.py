"""FastMCP registration adapter for Phase 2 tools."""

from __future__ import annotations

from typing import Any

from nlp_stock_prediction.orchestration.phase2_service import Phase2McpService
from nlp_stock_prediction.orchestration.phase2_tool_plan import PHASE2_MCP_TOOL_NAMES


def register_phase2_mcp_tools(server: Any, service: Phase2McpService) -> None:
    """Register Phase 2 tools on a FastMCP-compatible server."""

    def start_research_run(
        run_date: str,
        output_dir: str,
        symbol: str,
        objective: str | None = None,
    ) -> dict[str, object]:
        """Start a Phase 2 smoke research run and return run paths."""

        return dict(
            service.start_research_run(
                run_date=run_date,
                output_dir=output_dir,
                symbol=symbol,
                objective=objective,
            )
        )

    def list_research_tool_plan() -> dict[str, object]:
        """List the staged dummy tools Codex should call for Phase 2 smoke."""

        return dict(service.list_research_tool_plan())

    def record_codex_search_evidence(
        run_id: str,
        symbol: str,
        title: str,
        url: str,
        claim: str,
        query: str,
        published_at: str | None = None,
        stance: str | None = None,
    ) -> dict[str, object]:
        """Record one live-search source as normalized evidence.

        stance may be supports, contradicts, or neutral relative to the candidate thesis.
        """

        return dict(
            service.record_codex_search_evidence(
                run_id=run_id,
                symbol=symbol,
                title=title,
                url=url,
                claim=claim,
                query=query,
                published_at=published_at,
                stance=stance,
            )
        )

    def run_dummy_universe_tool(run_id: str, symbol: str) -> dict[str, object]:
        """Write deterministic dummy instrument-universe artifacts."""

        return dict(service.run_dummy_universe_tool(run_id=run_id, symbol=symbol))

    def run_dummy_analysis_tool(run_id: str, symbol: str) -> dict[str, object]:
        """Write deterministic dummy analysis artifacts."""

        return dict(service.run_dummy_analysis_tool(run_id=run_id, symbol=symbol))

    def synthesize_prediction_candidates(run_id: str, symbol: str) -> dict[str, object]:
        """Synthesize conservative prediction candidates from stored evidence."""

        return dict(service.synthesize_prediction_candidates(run_id=run_id, symbol=symbol))

    def render_prediction_report(run_id: str, symbol: str) -> dict[str, object]:
        """Render Markdown, JSON, and audit manifest files."""

        return dict(service.render_prediction_report(run_id=run_id, symbol=symbol))

    def inspect_research_run(run_id: str) -> dict[str, object]:
        """Inspect stored run graph counts for smoke verification."""

        return dict(service.inspect_research_run(run_id=run_id))

    functions = {
        "start_research_run": start_research_run,
        "list_research_tool_plan": list_research_tool_plan,
        "record_codex_search_evidence": record_codex_search_evidence,
        "run_dummy_universe_tool": run_dummy_universe_tool,
        "run_dummy_analysis_tool": run_dummy_analysis_tool,
        "synthesize_prediction_candidates": synthesize_prediction_candidates,
        "render_prediction_report": render_prediction_report,
        "inspect_research_run": inspect_research_run,
    }
    for tool_name in PHASE2_MCP_TOOL_NAMES:
        server.tool()(functions[tool_name])


__all__ = ["register_phase2_mcp_tools"]
