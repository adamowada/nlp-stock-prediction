"""Deterministic orchestration runtime."""

from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex, ArtifactType, ArtifactWriter
from nlp_stock_prediction.orchestration.context import RunContext, deterministic_generated_at
from nlp_stock_prediction.orchestration.dummy import (
    DEFAULT_STAGE_ORDER,
    DUMMY_ORCHESTRATION_DISABLED_MESSAGE,
    ReportBundle,
    build_dummy_tool_registry,
    generate_dummy_report_bundle,
)
from nlp_stock_prediction.orchestration.phase2_service import Phase2McpService
from nlp_stock_prediction.orchestration.phase4_common import Phase4ToolResult
from nlp_stock_prediction.orchestration.phase4_fundamentals import (
    Phase4FundamentalsTool,
    run_phase4_fundamentals_tool,
)
from nlp_stock_prediction.orchestration.phase4_news import (
    Phase4NewsCatalystTool,
    run_phase4_news_catalyst_tool,
)
from nlp_stock_prediction.orchestration.phase4_sector_macro import (
    Phase4SectorMacroTool,
    run_phase4_sector_macro_tool,
)
from nlp_stock_prediction.orchestration.phase4_social import (
    Phase4SocialEvidenceTool,
    run_phase4_social_evidence_tool,
)
from nlp_stock_prediction.orchestration.runtime import (
    OrchestrationExecutionError,
    OrchestrationState,
    StagedExecutionResult,
    StagedExecutor,
    ToolRunRecord,
)
from nlp_stock_prediction.orchestration.tools import (
    OrchestrationTool,
    ToolRegistry,
    ToolRunResult,
    ToolSpec,
)

__all__ = [
    "DEFAULT_STAGE_ORDER",
    "DUMMY_ORCHESTRATION_DISABLED_MESSAGE",
    "ArtifactIndex",
    "ArtifactType",
    "ArtifactWriter",
    "OrchestrationExecutionError",
    "OrchestrationState",
    "OrchestrationTool",
    "Phase2McpService",
    "Phase4FundamentalsTool",
    "Phase4NewsCatalystTool",
    "Phase4SectorMacroTool",
    "Phase4SocialEvidenceTool",
    "Phase4ToolResult",
    "ReportBundle",
    "RunContext",
    "StagedExecutionResult",
    "StagedExecutor",
    "ToolRegistry",
    "ToolRunRecord",
    "ToolRunResult",
    "ToolSpec",
    "build_dummy_tool_registry",
    "deterministic_generated_at",
    "generate_dummy_report_bundle",
    "run_phase4_fundamentals_tool",
    "run_phase4_news_catalyst_tool",
    "run_phase4_sector_macro_tool",
    "run_phase4_social_evidence_tool",
]
