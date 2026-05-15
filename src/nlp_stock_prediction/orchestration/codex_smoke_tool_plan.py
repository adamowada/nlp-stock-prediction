"""Codex Smoke Stage MCP tool plan definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from nlp_stock_prediction.contracts import JsonObject


@dataclass(frozen=True)
class CodexSmokeToolDefinition:
    tool_name: str
    stage: str
    description: str

    def as_json(self) -> JsonObject:
        return {
            "tool_name": self.tool_name,
            "stage": self.stage,
            "description": self.description,
        }


CODEX_SMOKE_RESEARCH_TOOLS: tuple[CodexSmokeToolDefinition, ...] = (
    CodexSmokeToolDefinition(
        tool_name="record_codex_search_evidence",
        stage="collect",
        description="Record one live-search source as normalized evidence.",
    ),
    CodexSmokeToolDefinition(
        tool_name="run_dummy_universe_tool",
        stage="discover",
        description="Write a deterministic Instrument-Universe Stage mixed-asset fixture universe.",
    ),
    CodexSmokeToolDefinition(
        tool_name="run_dummy_analysis_tool",
        stage="analyze",
        description="Write deterministic dummy analysis context.",
    ),
    CodexSmokeToolDefinition(
        tool_name="synthesize_prediction_candidates",
        stage="synthesize",
        description="Create evidence-backed or insufficient-evidence candidates.",
    ),
    CodexSmokeToolDefinition(
        tool_name="render_prediction_report",
        stage="report",
        description="Render Markdown, JSON, and audit manifest artifacts.",
    ),
)

CODEX_SMOKE_MCP_TOOL_NAMES: tuple[str, ...] = (
    "start_research_run",
    "list_research_tool_plan",
    *(tool.tool_name for tool in CODEX_SMOKE_RESEARCH_TOOLS),
    "inspect_research_run",
)


def codex_smoke_research_tool_plan() -> JsonObject:
    return cast(
        JsonObject,
        {
            "tools": [tool.as_json() for tool in CODEX_SMOKE_RESEARCH_TOOLS],
            "stage_order": ["discover", "collect", "analyze", "synthesize", "report"],
        },
    )


__all__ = [
    "CODEX_SMOKE_MCP_TOOL_NAMES",
    "CODEX_SMOKE_RESEARCH_TOOLS",
    "CodexSmokeToolDefinition",
    "codex_smoke_research_tool_plan",
]
