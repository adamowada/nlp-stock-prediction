"""Deterministic orchestration runtime."""

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
from nlp_stock_prediction.orchestration.phase4_live_providers import (
    Phase4LiveProviderFactory,
    Phase4LiveProviderFactoryProtocol,
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
from nlp_stock_prediction.orchestration.report_data_modes import (
    CODEX_SMOKE_REPORT_DATA_MODE,
    DUMMY_SMOKE_REPORT_DATA_MODE,
    LIVE_REPORT_DATA_MODE,
    OFFLINE_FIXTURE_REPORT_DATA_MODE,
    REPORT_DATA_MODE_KEY,
    ReportDataMode,
    ReportInputBoundaryViolation,
    enforce_live_report_input_boundary,
    find_non_live_report_input_violations,
    report_data_mode_from_run,
    report_data_mode_metadata,
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
    "CODEX_SMOKE_REPORT_DATA_MODE",
    "DEFAULT_STAGE_ORDER",
    "DUMMY_ORCHESTRATION_DISABLED_MESSAGE",
    "DUMMY_SMOKE_REPORT_DATA_MODE",
    "LIVE_REPORT_DATA_MODE",
    "OFFLINE_FIXTURE_REPORT_DATA_MODE",
    "PHASE4_STAGE_ORDER",
    "REPORT_DATA_MODE_KEY",
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
    "Phase4LiveProviderFactory",
    "Phase4LiveProviderFactoryProtocol",
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
    "ReportBundle",
    "ReportDataMode",
    "ReportInputBoundaryViolation",
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
    "deterministic_generated_at",
    "enforce_live_report_input_boundary",
    "execute_phase4_tool",
    "find_non_live_report_input_violations",
    "generate_dummy_report_bundle",
    "phase4_research_tool_plan",
    "report_data_mode_from_run",
    "report_data_mode_metadata",
    "run_phase4_fundamentals_tool",
    "run_phase4_news_catalyst_tool",
    "run_phase4_sector_macro_tool",
    "run_phase4_social_evidence_tool",
]
