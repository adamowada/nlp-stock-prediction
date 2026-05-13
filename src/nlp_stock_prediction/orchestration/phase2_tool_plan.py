"""Phase 2 MCP tool plan definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from nlp_stock_prediction.contracts import JsonObject


@dataclass(frozen=True)
class Phase2ToolDefinition:
    tool_name: str
    stage: str
    description: str

    def as_json(self) -> JsonObject:
        return {
            "tool_name": self.tool_name,
            "stage": self.stage,
            "description": self.description,
        }


PHASE2_RESEARCH_TOOLS: tuple[Phase2ToolDefinition, ...] = (
    Phase2ToolDefinition(
        tool_name="record_codex_search_evidence",
        stage="collect",
        description="Record one live-search source as normalized evidence.",
    ),
    Phase2ToolDefinition(
        tool_name="run_dummy_universe_tool",
        stage="discover",
        description="Write a deterministic retail-accessible instrument universe.",
    ),
    Phase2ToolDefinition(
        tool_name="run_dummy_analysis_tool",
        stage="analyze",
        description="Write deterministic dummy analysis context.",
    ),
    Phase2ToolDefinition(
        tool_name="synthesize_prediction_candidates",
        stage="synthesize",
        description="Create evidence-backed or insufficient-evidence candidates.",
    ),
    Phase2ToolDefinition(
        tool_name="render_prediction_report",
        stage="report",
        description="Render Markdown, JSON, and audit manifest artifacts.",
    ),
)

PHASE2_MCP_TOOL_NAMES: tuple[str, ...] = (
    "start_research_run",
    "list_research_tool_plan",
    *(tool.tool_name for tool in PHASE2_RESEARCH_TOOLS),
    "inspect_research_run",
)


def phase2_research_tool_plan() -> JsonObject:
    return cast(
        JsonObject,
        {
            "tools": [tool.as_json() for tool in PHASE2_RESEARCH_TOOLS],
            "stage_order": ["discover", "collect", "analyze", "synthesize", "report"],
        },
    )


__all__ = [
    "PHASE2_MCP_TOOL_NAMES",
    "PHASE2_RESEARCH_TOOLS",
    "Phase2ToolDefinition",
    "phase2_research_tool_plan",
]
