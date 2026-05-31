"""Deterministic orchestration runtime."""

from typing import Any

from nlp_stock_prediction.orchestration.artifacts import (
    ArtifactFileTransaction,
    ArtifactIndex,
    ArtifactType,
    ArtifactWriter,
)
from nlp_stock_prediction.orchestration.codex_smoke_service import CodexSmokeMcpService
from nlp_stock_prediction.orchestration.context import RunContext, deterministic_generated_at
from nlp_stock_prediction.orchestration.dummy import (
    DEFAULT_STAGE_ORDER,
    DUMMY_ORCHESTRATION_DISABLED_MESSAGE,
    build_dummy_tool_registry,
    generate_dummy_report_bundle,
)
from nlp_stock_prediction.orchestration.report_bundle import (
    BatchReportBundle,
    ReportBundle,
    WsbBatchReportBundle,
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
from nlp_stock_prediction.orchestration.research_batch import (
    generate_batch_research_reports,
    rank_report_viability,
)
from nlp_stock_prediction.orchestration.research_common import ResearchToolResult
from nlp_stock_prediction.orchestration.research_fundamentals import (
    ResearchFundamentalsTool,
    run_research_fundamentals_tool,
)
from nlp_stock_prediction.orchestration.research_live_providers import (
    ResearchLiveProviderFactory,
    ResearchLiveProviderFactoryProtocol,
)
from nlp_stock_prediction.orchestration.research_market_data import (
    MarketDataToolResult,
    ResearchMarketDataArtifact,
    ResearchMarketDataTool,
)
from nlp_stock_prediction.orchestration.research_news import (
    ResearchNewsCatalystTool,
    run_research_news_catalyst_tool,
)
from nlp_stock_prediction.orchestration.research_sector_macro import (
    ResearchSectorMacroTool,
    run_research_sector_macro_tool,
)
from nlp_stock_prediction.orchestration.research_service import (
    RESEARCH_STAGE_ORDER,
    ResearchService,
    ResearchToolExecutionError,
    ResearchToolMetadata,
    ResearchToolRegistry,
    ResearchToolRunContext,
    ResearchToolRunOutcome,
    build_research_tool_registry,
    execute_research_tool,
    research_tool_plan,
)
from nlp_stock_prediction.orchestration.research_social import (
    ResearchSocialEvidenceTool,
    run_research_social_evidence_tool,
)
from nlp_stock_prediction.orchestration.research_technical_package import (
    ResearchTechnicalPackageArtifact,
    ResearchTechnicalPackageTool,
    TechnicalPackageToolResult,
)
from nlp_stock_prediction.orchestration.research_universe_discovery import (
    ResearchUniverseDiscoveryTool,
    ResearchUniverseDiscoveryToolResult,
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
from nlp_stock_prediction.orchestration.wsb_trending import (
    DEFAULT_WSB_SOURCE_URL,
    WSB_TRENDING_PROVIDER,
    discover_wsb_trending_stocks,
    generate_wsb_batch_research_reports,
)

_EVALUATION_EXPORTS = {
    "EVALUATION_ABLATION_TOOL_ID",
    "EVALUATION_CALIBRATION_TOOL_ID",
    "EVALUATION_INSPECT_TOOL_ID",
    "EVALUATION_LOAD_OUTCOME_EVALUATIONS_TOOL_ID",
    "EVALUATION_OUTCOME_EVALUATION_TOOL_ID",
    "EVALUATION_STAGE_ORDER",
    "EVALUATION_WALK_FORWARD_TOOL_ID",
    "EvaluationService",
    "EvaluationToolMetadata",
    "EvaluationToolRegistry",
    "build_evaluation_tool_registry",
    "evaluation_tool_plan",
}


def __getattr__(name: str) -> Any:
    if name in _EVALUATION_EXPORTS:
        from nlp_stock_prediction.orchestration import evaluation_service

        value = getattr(evaluation_service, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CODEX_SMOKE_REPORT_DATA_MODE",
    "DEFAULT_STAGE_ORDER",
    "DEFAULT_WSB_SOURCE_URL",
    "DUMMY_ORCHESTRATION_DISABLED_MESSAGE",
    "DUMMY_SMOKE_REPORT_DATA_MODE",
    "EVALUATION_ABLATION_TOOL_ID",
    "EVALUATION_CALIBRATION_TOOL_ID",
    "EVALUATION_INSPECT_TOOL_ID",
    "EVALUATION_LOAD_OUTCOME_EVALUATIONS_TOOL_ID",
    "EVALUATION_OUTCOME_EVALUATION_TOOL_ID",
    "EVALUATION_STAGE_ORDER",
    "EVALUATION_WALK_FORWARD_TOOL_ID",
    "LIVE_REPORT_DATA_MODE",
    "OFFLINE_FIXTURE_REPORT_DATA_MODE",
    "REPORT_DATA_MODE_KEY",
    "RESEARCH_STAGE_ORDER",
    "WSB_TRENDING_PROVIDER",
    "ArtifactFileTransaction",
    "ArtifactIndex",
    "ArtifactType",
    "ArtifactWriter",
    "BatchReportBundle",
    "CodexSmokeMcpService",
    "EvaluationService",
    "EvaluationToolMetadata",
    "EvaluationToolRegistry",
    "MarketDataToolResult",
    "OrchestrationExecutionError",
    "OrchestrationState",
    "OrchestrationTool",
    "ReportBundle",
    "ReportDataMode",
    "ReportInputBoundaryViolation",
    "ResearchFundamentalsTool",
    "ResearchLiveProviderFactory",
    "ResearchLiveProviderFactoryProtocol",
    "ResearchMarketDataArtifact",
    "ResearchMarketDataTool",
    "ResearchNewsCatalystTool",
    "ResearchSectorMacroTool",
    "ResearchService",
    "ResearchSocialEvidenceTool",
    "ResearchTechnicalPackageArtifact",
    "ResearchTechnicalPackageTool",
    "ResearchToolExecutionError",
    "ResearchToolMetadata",
    "ResearchToolRegistry",
    "ResearchToolResult",
    "ResearchToolRunContext",
    "ResearchToolRunOutcome",
    "ResearchUniverseDiscoveryTool",
    "ResearchUniverseDiscoveryToolResult",
    "RunContext",
    "StagedExecutionResult",
    "StagedExecutor",
    "TechnicalPackageToolResult",
    "ToolRegistry",
    "ToolRunRecord",
    "ToolRunResult",
    "ToolSpec",
    "WsbBatchReportBundle",
    "build_dummy_tool_registry",
    "build_evaluation_tool_registry",
    "build_research_tool_registry",
    "deterministic_generated_at",
    "discover_wsb_trending_stocks",
    "enforce_live_report_input_boundary",
    "evaluation_tool_plan",
    "execute_research_tool",
    "find_non_live_report_input_violations",
    "generate_batch_research_reports",
    "generate_dummy_report_bundle",
    "generate_wsb_batch_research_reports",
    "rank_report_viability",
    "report_data_mode_from_run",
    "report_data_mode_metadata",
    "research_tool_plan",
    "run_research_fundamentals_tool",
    "run_research_news_catalyst_tool",
    "run_research_sector_macro_tool",
    "run_research_social_evidence_tool",
]
