"""Deterministic orchestration runtime."""

from typing import Any

from nlp_stock_prediction.orchestration.artifacts import (
    ArtifactFileTransaction,
    ArtifactIndex,
    ArtifactType,
    ArtifactWriter,
)
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
from nlp_stock_prediction.orchestration.phase4_market_data import (
    MarketDataToolResult,
    Phase4MarketDataArtifact,
    Phase4MarketDataTool,
)
from nlp_stock_prediction.orchestration.phase4_news import (
    Phase4NewsCatalystTool,
    run_phase4_news_catalyst_tool,
)
from nlp_stock_prediction.orchestration.phase4_sector_macro import (
    Phase4SectorMacroTool,
    run_phase4_sector_macro_tool,
)
from nlp_stock_prediction.orchestration.phase4_service import (
    PHASE4_STAGE_ORDER,
    Phase4Service,
    Phase4ToolExecutionError,
    Phase4ToolMetadata,
    Phase4ToolRegistry,
    Phase4ToolRunContext,
    Phase4ToolRunOutcome,
    build_phase4_tool_registry,
    execute_phase4_tool,
    phase4_research_tool_plan,
)
from nlp_stock_prediction.orchestration.phase4_social import (
    Phase4SocialEvidenceTool,
    run_phase4_social_evidence_tool,
)
from nlp_stock_prediction.orchestration.phase4_technical_package import (
    Phase4TechnicalPackageArtifact,
    Phase4TechnicalPackageTool,
    TechnicalPackageToolResult,
)
from nlp_stock_prediction.orchestration.phase4_universe_discovery import (
    Phase4UniverseDiscoveryTool,
    Phase4UniverseDiscoveryToolResult,
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

_PHASE6_EXPORTS = {
    "PHASE6_ABLATION_TOOL_ID",
    "PHASE6_CALIBRATION_TOOL_ID",
    "PHASE6_INSPECT_TOOL_ID",
    "PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_ID",
    "PHASE6_OUTCOME_EVALUATION_TOOL_ID",
    "PHASE6_STAGE_ORDER",
    "PHASE6_WALK_FORWARD_TOOL_ID",
    "Phase6Service",
    "Phase6ToolMetadata",
    "Phase6ToolRegistry",
    "build_phase6_tool_registry",
    "phase6_evaluation_tool_plan",
}


def __getattr__(name: str) -> Any:
    if name in _PHASE6_EXPORTS:
        from nlp_stock_prediction.orchestration import phase6_service

        value = getattr(phase6_service, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_STAGE_ORDER",
    "DUMMY_ORCHESTRATION_DISABLED_MESSAGE",
    "PHASE4_STAGE_ORDER",
    "PHASE6_ABLATION_TOOL_ID",
    "PHASE6_CALIBRATION_TOOL_ID",
    "PHASE6_INSPECT_TOOL_ID",
    "PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_ID",
    "PHASE6_OUTCOME_EVALUATION_TOOL_ID",
    "PHASE6_STAGE_ORDER",
    "PHASE6_WALK_FORWARD_TOOL_ID",
    "ArtifactFileTransaction",
    "ArtifactIndex",
    "ArtifactType",
    "ArtifactWriter",
    "MarketDataToolResult",
    "OrchestrationExecutionError",
    "OrchestrationState",
    "OrchestrationTool",
    "Phase2McpService",
    "Phase4FundamentalsTool",
    "Phase4MarketDataArtifact",
    "Phase4MarketDataTool",
    "Phase4NewsCatalystTool",
    "Phase4SectorMacroTool",
    "Phase4Service",
    "Phase4SocialEvidenceTool",
    "Phase4TechnicalPackageArtifact",
    "Phase4TechnicalPackageTool",
    "Phase4ToolExecutionError",
    "Phase4ToolMetadata",
    "Phase4ToolRegistry",
    "Phase4ToolResult",
    "Phase4ToolRunContext",
    "Phase4ToolRunOutcome",
    "Phase4UniverseDiscoveryTool",
    "Phase4UniverseDiscoveryToolResult",
    "Phase6Service",
    "Phase6ToolMetadata",
    "Phase6ToolRegistry",
    "ReportBundle",
    "RunContext",
    "StagedExecutionResult",
    "StagedExecutor",
    "TechnicalPackageToolResult",
    "ToolRegistry",
    "ToolRunRecord",
    "ToolRunResult",
    "ToolSpec",
    "build_dummy_tool_registry",
    "build_phase4_tool_registry",
    "build_phase6_tool_registry",
    "deterministic_generated_at",
    "execute_phase4_tool",
    "generate_dummy_report_bundle",
    "phase4_research_tool_plan",
    "phase6_evaluation_tool_plan",
    "run_phase4_fundamentals_tool",
    "run_phase4_news_catalyst_tool",
    "run_phase4_sector_macro_tool",
    "run_phase4_social_evidence_tool",
]
