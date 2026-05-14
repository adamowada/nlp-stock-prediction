"""Phase 4 runtime and final-report service foundation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, NonEmptyStr
from nlp_stock_prediction.contracts.enums import (
    Direction,
    PredictionType,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.instruments import InstrumentQuery, InstrumentUniverseRequest
from nlp_stock_prediction.contracts.providers import FundamentalsSnapshot
from nlp_stock_prediction.orchestration.artifacts import (
    ArtifactFileTransaction,
    ArtifactIndex,
    ArtifactWriter,
)
from nlp_stock_prediction.orchestration.context import RunContext, deterministic_generated_at
from nlp_stock_prediction.orchestration.phase2_common import (
    ALLOWED_WRITE_ROOTS,
    Phase2RunPaths,
    Phase2WritePolicy,
    run_date_from_run,
    stable_digest,
    symbol_slug,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase2_evidence import evidence_stance_from_record
from nlp_stock_prediction.orchestration.phase2_report import (
    PredictionReportBuildRequest,
    ReportBundleBuilder,
)
from nlp_stock_prediction.orchestration.phase4_common import Phase4ToolResult
from nlp_stock_prediction.orchestration.phase4_evaluation import (
    evaluate_stored_prediction_candidates,
)
from nlp_stock_prediction.orchestration.phase4_fixture_providers import (
    Phase4FixtureProviderFactory,
)
from nlp_stock_prediction.orchestration.phase4_fundamentals import (
    PHASE4_FUNDAMENTALS_TOOL_NAME,
    run_phase4_fundamentals_tool,
)
from nlp_stock_prediction.orchestration.phase4_live_providers import (
    Phase4LiveProviderFactory,
    Phase4LiveProviderFactoryProtocol,
)
from nlp_stock_prediction.orchestration.phase4_market_data import (
    PHASE4_MARKET_DATA_TOOL_NAME,
    Phase4MarketDataTool,
)
from nlp_stock_prediction.orchestration.phase4_news import (
    PHASE4_NEWS_TOOL_NAME,
    run_phase4_news_catalyst_tool,
)
from nlp_stock_prediction.orchestration.phase4_run_modes import Phase4RunModeAdapter
from nlp_stock_prediction.orchestration.phase4_sector_macro import (
    PHASE4_SECTOR_MACRO_TOOL_NAME,
    run_phase4_sector_macro_tool,
)
from nlp_stock_prediction.orchestration.phase4_social import (
    PHASE4_SOCIAL_TOOL_NAME,
    run_phase4_social_evidence_tool,
)
from nlp_stock_prediction.orchestration.phase4_technical_package import (
    PHASE4_TECHNICAL_PACKAGE_TOOL_NAME,
    Phase4TechnicalPackageTool,
)
from nlp_stock_prediction.orchestration.phase4_universe_discovery import (
    PHASE4_TOOL_NAME as PHASE4_UNIVERSE_TOOL_NAME,
)
from nlp_stock_prediction.orchestration.phase4_universe_discovery import (
    Phase4UniverseDiscoveryTool,
)
from nlp_stock_prediction.orchestration.report_data_modes import (
    LIVE_REPORT_DATA_MODE,
    OFFLINE_FIXTURE_REPORT_DATA_MODE,
    REPORT_DATA_MODE_KEY,
    ReportDataMode,
    normalize_report_data_mode,
    report_data_mode_from_run,
    report_data_mode_metadata,
    report_data_mode_metadata_from_run,
)
from nlp_stock_prediction.orchestration.signal_artifacts import (
    signal_artifact_metadata,
    signal_artifact_references_for_records,
)
from nlp_stock_prediction.reporting.audit import stable_json_bytes
from nlp_stock_prediction.storage.records import (
    CandidateArtifactLinkRecord,
    CandidateEvidenceLinkRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore, initialize_research_database

PHASE4_STAGE_ORDER: tuple[str, ...] = ("discover", "collect", "analyze", "evaluate", "report")
Phase4Stage = Literal["discover", "collect", "analyze", "evaluate", "report"]
Phase4ToolRunStatus = Literal["successful", "partial", "empty", "skipped", "failed"]

PHASE4_COLLECT_TOOL_ID = "phase4.collect_codex_search_evidence"
PHASE4_EVALUATE_TOOL_ID = "phase4.evaluate_prediction_candidates"
PHASE4_REPORT_TOOL_ID = "phase4.render_final_report"
PHASE4_UNIVERSE_TOOL_ID = "phase4.universe_discovery"
PHASE4_MARKET_DATA_TOOL_ID = "phase4.market_data"
PHASE4_TECHNICAL_TOOL_ID = "phase4.technical_package"
PHASE4_SOCIAL_TOOL_ID = "phase4.social_evidence"
PHASE4_NEWS_TOOL_ID = "phase4.news_catalyst"
PHASE4_FUNDAMENTALS_TOOL_ID = "phase4.fundamentals"
PHASE4_SECTOR_MACRO_TOOL_ID = "phase4.sector_macro"
PHASE4_CANDIDATE_SYNTHESIS_TOOL_ID = "phase4.prediction_candidate_synthesis"
PHASE4_PREDICTION_EVALUATION_TOOL_ID = "phase4.prediction_evaluation"

MISSING_CANDIDATE_WARNING = (
    "No stored prediction candidates were available; Phase 4 report rendering is final-only "
    "and did not synthesize candidates."
)


class Phase4ToolMetadata(ContractModel):
    """Canonical metadata for one Phase 4 research tool."""

    tool_id: NonEmptyStr
    tool_name: NonEmptyStr
    tool_version: NonEmptyStr
    stage: Phase4Stage
    description: NonEmptyStr
    artifact_kinds: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    offline_capable: bool = True
    live_capable: bool = False
    requires_network: bool = False
    dependencies: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tool_metadata(self) -> Phase4ToolMetadata:
        if self.requires_network and not self.live_capable:
            raise ValueError("network-dependent tools must be live_capable")
        if self.tool_id in self.dependencies:
            raise ValueError("phase4 tools cannot depend on themselves")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("phase4 tool dependencies must be unique")
        return self

    def as_plan_item(self) -> JsonObject:
        return {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "version": self.tool_version,
            "tool_version": self.tool_version,
            "stage": self.stage,
            "description": self.description,
            "artifact_kinds": list(self.artifact_kinds),
            "offline_capable": self.offline_capable,
            "live_capable": self.live_capable,
            "requires_network": self.requires_network,
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }


class Phase4ToolRegistry:
    """Deterministic Phase 4 tool metadata registry."""

    def __init__(self, tools: Iterable[Phase4ToolMetadata] = ()) -> None:
        self._tools: dict[str, Phase4ToolMetadata] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Phase4ToolMetadata) -> None:
        if tool.tool_id in self._tools:
            raise ValueError(f"phase4 tool is already registered: {tool.tool_id}")
        if any(
            item.tool_name == tool.tool_name and item.tool_version == tool.tool_version
            for item in self._tools.values()
        ):
            raise ValueError(
                f"phase4 tool name/version is already registered: "
                f"{tool.tool_name}@{tool.tool_version}"
            )
        self._tools[tool.tool_id] = tool

    def get(self, tool_id: str) -> Phase4ToolMetadata:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"phase4 tool is not registered: {tool_id}") from exc

    def specs(self) -> tuple[Phase4ToolMetadata, ...]:
        return self.ordered_for_stages(PHASE4_STAGE_ORDER)

    def ordered_for_stages(
        self,
        stage_order: Sequence[str] = PHASE4_STAGE_ORDER,
    ) -> tuple[Phase4ToolMetadata, ...]:
        stages = tuple(stage_order)
        if len(set(stages)) != len(stages):
            raise ValueError("stage_order entries must be unique")
        stage_index = {stage: index for index, stage in enumerate(stages)}
        unknown_stages = sorted(
            {tool.stage for tool in self._tools.values() if tool.stage not in stage_index}
        )
        if unknown_stages:
            joined = ", ".join(unknown_stages)
            raise ValueError(f"phase4 tools use stages missing from stage_order: {joined}")
        self._validate_dependencies()
        order_index = {tool_id: index for index, tool_id in enumerate(self._tools)}
        return tuple(
            tool
            for _tool_id, tool in sorted(
                self._tools.items(),
                key=lambda item: (stage_index[item[1].stage], order_index[item[0]]),
            )
        )

    def as_plan(self) -> JsonObject:
        return {
            "tools": [tool.as_plan_item() for tool in self.ordered_for_stages()],
            "stage_order": list(PHASE4_STAGE_ORDER),
        }

    def _validate_dependencies(self) -> None:
        known_ids = set(self._tools)
        missing: dict[str, tuple[str, ...]] = {}
        for tool in self._tools.values():
            unknown = tuple(dep for dep in tool.dependencies if dep not in known_ids)
            if unknown:
                missing[tool.tool_id] = unknown
        if missing:
            details = "; ".join(
                f"{tool_id}: {', '.join(dependencies)}"
                for tool_id, dependencies in sorted(missing.items())
            )
            raise ValueError(f"phase4 tool dependencies are not registered: {details}")


class Phase4ToolRunOutcome(ContractModel):
    """Action result consumed by the durable Phase 4 execution helper."""

    status: Phase4ToolRunStatus = "successful"
    payload: JsonObject = Field(default_factory=dict)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    warnings: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    error_message: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome(self) -> Phase4ToolRunOutcome:
        if self.status == "partial" and not self.warnings:
            raise ValueError("partial Phase 4 tool outcomes require warnings")
        if self.status == "failed" and not self.error_message:
            raise ValueError("failed Phase 4 tool outcomes require error_message")
        return self


@dataclass(frozen=True)
class Phase4ToolRunContext:
    """Context object passed into one durable Phase 4 tool action."""

    store: SQLiteStore
    repo_root: Path
    paths: Phase2RunPaths
    run_id: str
    tool: Phase4ToolMetadata
    tool_run_id: str
    started_at: datetime
    inputs: JsonObject

    def artifact_index(
        self,
        *,
        base_dir: Path | None = None,
        produced_by: str | None = None,
        schema_version: str = "phase4-tool-artifact.v1",
    ) -> ArtifactIndex:
        mode = self.inputs.get(REPORT_DATA_MODE_KEY)
        default_metadata = (
            report_data_mode_metadata(normalize_report_data_mode(mode)) if mode is not None else {}
        )
        return ArtifactIndex.for_directory(
            store=self.store,
            repo_root=self.repo_root,
            base_dir=base_dir or self.paths.audit_dir,
            created_at=self.started_at,
            produced_by=produced_by or self.tool.tool_name,
            tool_run_id=self.tool_run_id,
            schema_version=schema_version,
            default_metadata=default_metadata,
        )


class Phase4ToolExecutionError(RuntimeError):
    """Raised after a failed Phase 4 tool run has been durably recorded."""

    def __init__(
        self,
        *,
        tool_run_id: str,
        tool_id: str,
        tool_name: str,
        original_error: Exception,
    ) -> None:
        super().__init__(f"Phase 4 tool {tool_name!r} failed: {original_error}")
        self.tool_run_id = tool_run_id
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.original_error = original_error


class _Phase4ToolReturnedFailure(RuntimeError):
    """Internal sentinel for actions that return a failed outcome."""


Phase4ToolAction = Callable[[Phase4ToolRunContext], Phase4ToolRunOutcome]


def build_phase4_tool_registry() -> Phase4ToolRegistry:
    """Build the default Phase 4 metadata registry."""

    return Phase4ToolRegistry(
        (
            Phase4ToolMetadata(
                tool_id=PHASE4_UNIVERSE_TOOL_ID,
                tool_name=PHASE4_UNIVERSE_TOOL_NAME,
                tool_version="phase4.universe.v1",
                stage="discover",
                description=(
                    "Resolve the requested instrument universe and write instrument-universe "
                    "artifacts."
                ),
                artifact_kinds=("instrument_universe",),
                offline_capable=True,
                live_capable=True,
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_MARKET_DATA_TOOL_ID,
                tool_name=PHASE4_MARKET_DATA_TOOL_NAME,
                tool_version="phase4.market-data.v1",
                stage="collect",
                description="Fetch daily market data and preserve provider provenance.",
                artifact_kinds=("market_data",),
                offline_capable=True,
                live_capable=True,
                requires_network=True,
                dependencies=(PHASE4_UNIVERSE_TOOL_ID,),
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_TECHNICAL_TOOL_ID,
                tool_name=PHASE4_TECHNICAL_PACKAGE_TOOL_NAME,
                tool_version="phase4.technical-package.v1",
                stage="analyze",
                description=(
                    "Compute deterministic technical context from market-data artifacts; raw "
                    "TimesFM remains sidecar context only."
                ),
                artifact_kinds=("technical_package",),
                live_capable=True,
                dependencies=(PHASE4_MARKET_DATA_TOOL_ID,),
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_SOCIAL_TOOL_ID,
                tool_name=PHASE4_SOCIAL_TOOL_NAME,
                tool_version="phase4.evidence-suite.v1",
                stage="collect",
                description="Normalize social evidence and preserve source/provider provenance.",
                artifact_kinds=("normalized_evidence",),
                live_capable=True,
                requires_network=True,
                dependencies=(PHASE4_UNIVERSE_TOOL_ID,),
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_NEWS_TOOL_ID,
                tool_name=PHASE4_NEWS_TOOL_NAME,
                tool_version="phase4.evidence-suite.v1",
                stage="collect",
                description="Normalize news/catalyst evidence with derived labels.",
                artifact_kinds=("normalized_evidence",),
                live_capable=True,
                requires_network=True,
                dependencies=(PHASE4_UNIVERSE_TOOL_ID,),
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_FUNDAMENTALS_TOOL_ID,
                tool_name=PHASE4_FUNDAMENTALS_TOOL_NAME,
                tool_version="phase4.evidence-suite.v1",
                stage="collect",
                description="Fetch fundamentals evidence and write analysis context artifacts.",
                artifact_kinds=("analysis_context",),
                live_capable=True,
                requires_network=True,
                dependencies=(PHASE4_UNIVERSE_TOOL_ID,),
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_SECTOR_MACRO_TOOL_ID,
                tool_name=PHASE4_SECTOR_MACRO_TOOL_NAME,
                tool_version="phase4.evidence-suite.v1",
                stage="analyze",
                description="Build sector and macro baseline context with provenance.",
                artifact_kinds=("analysis_context",),
                live_capable=True,
                requires_network=True,
                dependencies=(PHASE4_FUNDAMENTALS_TOOL_ID,),
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_CANDIDATE_SYNTHESIS_TOOL_ID,
                tool_name="phase4_prediction_candidate_synthesis",
                tool_version="phase4.v1",
                stage="evaluate",
                description=(
                    "Synthesize conservative prediction candidates from stored, attributable "
                    "Phase 4 evidence."
                ),
                artifact_kinds=("prediction_input",),
                live_capable=True,
                dependencies=(
                    PHASE4_SOCIAL_TOOL_ID,
                    PHASE4_NEWS_TOOL_ID,
                    PHASE4_FUNDAMENTALS_TOOL_ID,
                    PHASE4_SECTOR_MACRO_TOOL_ID,
                ),
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_PREDICTION_EVALUATION_TOOL_ID,
                tool_name="phase4_prediction_evaluation",
                tool_version="phase4.v1",
                stage="evaluate",
                description="Evaluate stored prediction candidates as prediction-quality records.",
                artifact_kinds=("prediction_evaluation",),
                live_capable=True,
                dependencies=(
                    PHASE4_CANDIDATE_SYNTHESIS_TOOL_ID,
                    PHASE4_TECHNICAL_TOOL_ID,
                    PHASE4_SOCIAL_TOOL_ID,
                    PHASE4_NEWS_TOOL_ID,
                    PHASE4_FUNDAMENTALS_TOOL_ID,
                    PHASE4_SECTOR_MACRO_TOOL_ID,
                ),
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_REPORT_TOOL_ID,
                tool_name="render_prediction_report",
                tool_version="phase4.v1",
                stage="report",
                description=(
                    "Final-only report renderer that consumes stored evidence, candidates, "
                    "and artifacts."
                ),
                artifact_kinds=("markdown_report", "json_report", "audit_manifest"),
                live_capable=True,
                dependencies=(PHASE4_PREDICTION_EVALUATION_TOOL_ID,),
                metadata={"final_only": True},
            ),
        )
    )


def phase4_research_tool_plan(
    registry: Phase4ToolRegistry | None = None,
) -> JsonObject:
    """Return a registry-derived Phase 4 tool plan."""

    return (registry or build_phase4_tool_registry()).as_plan()


def phase4_run_id(run_date: date, symbol: str) -> str:
    return f"phase4-{run_date.isoformat()}-{symbol_slug(symbol)}"


def execute_phase4_tool(
    *,
    store: SQLiteStore,
    repo_root: Path,
    paths: Phase2RunPaths,
    run_id: str,
    tool: Phase4ToolMetadata,
    inputs: JsonObject,
    action: Phase4ToolAction,
    tool_run_id: str | None = None,
) -> Phase4ToolRunOutcome:
    """Execute one tool in a SQLite transaction and roll back new files on failure."""

    run = store.get_research_run(run_id)
    started_at = _phase4_timestamp_for_run(run)
    resolved_tool_run_id = tool_run_id or _phase4_tool_run_id(
        run_id=run_id,
        tool=tool,
        inputs=inputs,
    )
    base_inputs = _tool_run_inputs(tool=tool, inputs=inputs)
    previous_tool_run = store.get_tool_run(resolved_tool_run_id)
    file_transaction = ArtifactFileTransaction.begin(paths.run_dir)
    context = Phase4ToolRunContext(
        store=store,
        repo_root=repo_root,
        paths=paths,
        run_id=run_id,
        tool=tool,
        tool_run_id=resolved_tool_run_id,
        started_at=started_at,
        inputs=inputs,
    )
    try:
        final_outcome: Phase4ToolRunOutcome
        with store.transaction():
            store.record_tool_run(
                ToolRunRecord(
                    tool_run_id=resolved_tool_run_id,
                    run_id=run_id,
                    tool_name=tool.tool_name,
                    tool_version=tool.tool_version,
                    status="running",
                    started_at=started_at,
                    inputs=base_inputs,
                )
            )
            outcome = action(context)
            if outcome.status == "failed":
                raise _Phase4ToolReturnedFailure(outcome.error_message or "tool returned failed")
            completed_at = _phase4_timestamp_for_run(run)
            final_outcome = _with_execution_payload(
                outcome=outcome,
                run_id=run_id,
                tool_run_id=resolved_tool_run_id,
            )
            store.record_tool_run(
                ToolRunRecord(
                    tool_run_id=resolved_tool_run_id,
                    run_id=run_id,
                    tool_name=tool.tool_name,
                    tool_version=tool.tool_version,
                    status=final_outcome.status,
                    started_at=started_at,
                    completed_at=completed_at,
                    inputs=_tool_run_inputs(
                        tool=tool,
                        inputs=inputs,
                        outcome=final_outcome,
                    ),
                    warnings=final_outcome.warnings,
                )
            )
        file_transaction.cleanup()
        return final_outcome
    except Exception as exc:
        rollback_errors: list[str] = []
        try:
            file_transaction.rollback_new_files()
        except Exception as rollback_exc:
            rollback_errors.append(str(rollback_exc))
        completed_at = _phase4_timestamp_for_run(run)
        error_message = str(exc)
        if rollback_errors:
            error_message = f"{error_message}; artifact rollback errors: " + "; ".join(
                dict.fromkeys(rollback_errors)
            )
        failed_tool_run_id = _failed_phase4_tool_run_id(
            tool_run_id=resolved_tool_run_id,
            previous_exists=previous_tool_run is not None and previous_tool_run.status != "running",
            completed_at=completed_at,
            error_message=error_message,
        )
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=failed_tool_run_id,
                run_id=run_id,
                tool_name=tool.tool_name,
                tool_version=tool.tool_version,
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                inputs=_tool_run_inputs(tool=tool, inputs=inputs),
                error_message=error_message,
            )
        )
        raise Phase4ToolExecutionError(
            tool_run_id=failed_tool_run_id,
            tool_id=tool.tool_id,
            tool_name=tool.tool_name,
            original_error=exc,
        ) from exc


@dataclass(frozen=True)
class Phase4Service:
    """Stateful Phase 4 service over the research SQLite run graph."""

    repo_root: Path = Path(".")
    database_path: Path = Path("data/prediction-research.sqlite3")
    fixture_root: Path | None = None
    provider_cache_root: Path | None = None
    extra_write_roots: tuple[Path, ...] = ()
    registry: Phase4ToolRegistry = field(default_factory=build_phase4_tool_registry)
    live_provider_factory: Phase4LiveProviderFactoryProtocol | None = field(
        default=None,
        compare=False,
    )
    _store: SQLiteStore = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", self.repo_root.resolve())
        object.__setattr__(
            self,
            "fixture_root",
            (self.fixture_root or self.repo_root).resolve(),
        )
        object.__setattr__(
            self,
            "provider_cache_root",
            None if self.provider_cache_root is None else self.provider_cache_root.resolve(),
        )
        object.__setattr__(
            self,
            "extra_write_roots",
            tuple(path.resolve() for path in self.extra_write_roots),
        )
        if self.live_provider_factory is None:
            object.__setattr__(
                self,
                "live_provider_factory",
                Phase4LiveProviderFactory(cache_root=self.provider_cache_root),
            )
        object.__setattr__(
            self,
            "_store",
            initialize_research_database(self._resolve_write_path(self.database_path)),
        )

    @property
    def write_policy(self) -> Phase2WritePolicy:
        return Phase2WritePolicy(self.repo_root, extra_allowed_roots=self.extra_write_roots)

    @property
    def store(self) -> SQLiteStore:
        return self._store

    @property
    def fixtures(self) -> Phase4FixtureProviderFactory:
        return Phase4FixtureProviderFactory(cast(Path, self.fixture_root))

    def _fixtures_for_run(self, run: ResearchRunRecord) -> Phase4FixtureProviderFactory:
        return Phase4FixtureProviderFactory(
            cast(Path, self.fixture_root),
            fetched_at=_phase4_timestamp_for_run(run),
        )

    @property
    def live_providers(self) -> Phase4LiveProviderFactoryProtocol:
        if self.live_provider_factory is None:
            raise RuntimeError("live provider factory was not initialized")
        return self.live_provider_factory

    def start_research_run(
        self,
        *,
        run_date: str,
        output_dir: str,
        symbol: str,
        objective: str | None = None,
        report_data_mode: ReportDataMode = OFFLINE_FIXTURE_REPORT_DATA_MODE,
    ) -> JsonObject:
        parsed_date = date.fromisoformat(run_date)
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must be non-empty")
        run_id = phase4_run_id(parsed_date, normalized_symbol)
        if self.store.get_research_run(run_id) is not None:
            raise ValueError(f"research run already exists: {run_id}")
        paths = self._paths(parsed_date, output_dir, symbol=normalized_symbol)
        now = (
            deterministic_generated_at(parsed_date)
            if report_data_mode == OFFLINE_FIXTURE_REPORT_DATA_MODE
            else utc_now()
        )
        mode_metadata = report_data_mode_metadata(report_data_mode)
        self.store.upsert_research_run(
            ResearchRunRecord(
                run_id=run_id,
                run_kind="phase4_prediction_report",
                objective=objective or f"Phase 4 prediction research for {normalized_symbol}",
                status="running",
                started_at=now,
                metadata={
                    "run_date": parsed_date.isoformat(),
                    "symbol": normalized_symbol,
                    "output_dir": paths.output_dir.as_posix(),
                    "phase": "phase4_runtime_report",
                    "stage_order": list(PHASE4_STAGE_ORDER),
                    **mode_metadata,
                },
            )
        )
        return {
            "run_id": run_id,
            "run_date": parsed_date.isoformat(),
            "symbol": normalized_symbol,
            "output_dir": paths.output_dir.as_posix(),
            "run_dir": paths.run_dir.as_posix(),
            "audit_dir": paths.audit_dir.as_posix(),
            "database_path": self._resolve_write_path(self.database_path).as_posix(),
            **mode_metadata,
        }

    def list_research_tool_plan(self) -> JsonObject:
        return phase4_research_tool_plan(self.registry)

    def phase4_universe_discovery(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]), symbol=normalized_symbol)
        context = self._run_context(run, paths)
        request = InstrumentUniverseRequest(
            request_id=f"phase4-service-universe-{run_id}-{symbol_slug(normalized_symbol)}",
            as_of=context.generated_at,
            queries=(InstrumentQuery(query=normalized_symbol),),
        )
        universe_provider = self._mode_adapter(run).universe_provider()
        universe_tool = (
            Phase4UniverseDiscoveryTool(provider=universe_provider)
            if universe_provider is not None
            else Phase4UniverseDiscoveryTool()
        )
        result = universe_tool.run(
            request=request,
            context=context,
            store=self.store,
            repo_root=self.repo_root,
        )
        return {
            "run_id": run_id,
            "tool_run_id": result.tool_run_id,
            "artifact_id": result.artifact_id,
            "artifact_path": result.artifact.path,
            "instrument_ids": list(result.instrument_ids),
            "source_query_ids": [record.source_query_id for record in result.source_query_records],
            "warnings": list(result.tool_run_record.warnings),
        }

    def phase4_market_data(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]), symbol=normalized_symbol)
        mode_adapter = self._mode_adapter(run)
        market_data = mode_adapter.market_data_selection(normalized_symbol)
        result = Phase4MarketDataTool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=paths.audit_dir,
            provider=market_data.provider,
            now=lambda: _phase4_timestamp_for_run(run),
        ).run(
            run_id=run_id,
            run_date=run_date,
            symbol=normalized_symbol,
            instrument_id=mode_adapter.instrument_id(normalized_symbol),
            source_url=market_data.source_url,
            options=market_data.options,
        )
        return {
            "run_id": run_id,
            "tool_run_id": result.tool_run_id,
            "artifact_id": result.artifact.artifact_id,
            "artifact_path": result.artifact.path,
            "source_query_id": result.source_query_id,
            "status": result.provider_result.status.value,
            "warnings": [warning.message for warning in result.artifact_payload.warnings],
        }

    def phase4_technical_package(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]), symbol=normalized_symbol)
        market_path = self._latest_artifact_path(run_id, "market_data")
        if market_path is None:
            self.phase4_market_data(run_id=run_id, symbol=normalized_symbol)
            market_path = self._latest_artifact_path(run_id, "market_data")
        if market_path is None:
            raise ValueError("phase4 technical package requires a market-data artifact")
        result = Phase4TechnicalPackageTool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=paths.audit_dir,
            now=lambda: _phase4_timestamp_for_run(run),
        ).run(
            run_id=run_id,
            symbol=normalized_symbol,
            market_data=market_path,
            instrument_id=self._mode_adapter(run).instrument_id(normalized_symbol),
        )
        return {
            "run_id": run_id,
            "tool_run_id": result.tool_run_id,
            "artifact_id": result.artifact.artifact_id,
            "artifact_path": result.artifact.path,
            "status": result.artifact_payload.status,
            "warnings": [warning.message for warning in result.artifact_payload.warnings],
        }

    def phase4_social_evidence(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        providers = self._mode_adapter(run).social_providers(normalized_symbol)
        result = run_phase4_social_evidence_tool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir(run),
            run_id=run_id,
            symbol=normalized_symbol,
            run_date=run_date_from_run(run),
            generated_at=_phase4_timestamp_for_run(run),
            reddit_provider=providers.reddit_provider,
            x_provider=providers.x_provider,
            instrument_id=self._mode_adapter(run).instrument_id(normalized_symbol),
        )
        return _phase4_tool_result_payload(result)

    def phase4_news_catalyst(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        result = run_phase4_news_catalyst_tool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir(run),
            run_id=run_id,
            symbol=normalized_symbol,
            run_date=run_date_from_run(run),
            generated_at=_phase4_timestamp_for_run(run),
            providers=self._mode_adapter(run).news_providers(normalized_symbol),
            instrument_id=self._mode_adapter(run).instrument_id(normalized_symbol),
        )
        return _phase4_tool_result_payload(result)

    def phase4_fundamentals(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        result = run_phase4_fundamentals_tool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir(run),
            run_id=run_id,
            symbol=normalized_symbol,
            run_date=run_date_from_run(run),
            generated_at=_phase4_timestamp_for_run(run),
            providers=self._mode_adapter(run).fundamentals_providers(normalized_symbol),
            instrument_id=self._mode_adapter(run).instrument_id(normalized_symbol),
        )
        return _phase4_tool_result_payload(result)

    def phase4_sector_macro(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        result = run_phase4_sector_macro_tool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir(run),
            run_id=run_id,
            symbol=normalized_symbol,
            run_date=run_date_from_run(run),
            generated_at=_phase4_timestamp_for_run(run),
            target_snapshot=FundamentalsSnapshot(ticker=normalized_symbol),
            macro_providers=self._mode_adapter(run).macro_providers(normalized_symbol),
            instrument_id=self._mode_adapter(run).instrument_id(normalized_symbol),
        )
        return _phase4_tool_result_payload(result)

    def phase4_prediction_evaluation(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        candidates = self.store.list_prediction_candidates_for_run(run_id)

        def action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
            if not candidates:
                return Phase4ToolRunOutcome(
                    status="empty",
                    payload={
                        "candidate_count": 0,
                        "message": "No stored prediction candidates were available to evaluate.",
                    },
                    metadata={"candidate_count": 0},
                )
            result = evaluate_stored_prediction_candidates(
                store=self.store,
                repo_root=self.repo_root,
                artifact_dir=context.paths.audit_dir,
                run_id=run_id,
                tool_run_id=context.tool_run_id,
                generated_at=_phase4_timestamp_for_run(run),
            )
            return Phase4ToolRunOutcome(
                status="partial" if result.warnings else "successful",
                payload=result.payload,
                artifact_ids=result.artifact_ids,
                warnings=result.warnings,
                metadata={
                    "candidate_count": len(candidates),
                    "evaluation_count": result.payload["evaluation_count"],
                },
            )

        return self._execute_symbol_tool(
            run_id=run_id,
            symbol=normalized_symbol,
            tool_id=PHASE4_PREDICTION_EVALUATION_TOOL_ID,
            inputs={"symbol": normalized_symbol, "candidate_count": len(candidates)},
            action=action,
        ).payload

    def run_offline_phase4_flow(self, *, run_date: str, output_dir: str, symbol: str) -> JsonObject:
        return self._run_phase4_flow(
            run_date=run_date,
            output_dir=output_dir,
            symbol=symbol,
            report_data_mode=OFFLINE_FIXTURE_REPORT_DATA_MODE,
            objective_template="Phase 4 fixture-backed prediction research for {symbol}",
            mode_error="offline Phase 4 flow requires an offline_fixture report_data_mode",
        )

    def run_live_phase4_flow(self, *, run_date: str, output_dir: str, symbol: str) -> JsonObject:
        return self._run_phase4_flow(
            run_date=run_date,
            output_dir=output_dir,
            symbol=symbol,
            report_data_mode=LIVE_REPORT_DATA_MODE,
            objective_template="Live provider prediction research for {symbol}",
            mode_error="live Phase 4 flow requires a live report_data_mode",
        )

    def _run_phase4_flow(
        self,
        *,
        run_date: str,
        output_dir: str,
        symbol: str,
        report_data_mode: ReportDataMode,
        objective_template: str,
        mode_error: str,
    ) -> JsonObject:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must be non-empty")
        run_id = phase4_run_id(date.fromisoformat(run_date), normalized_symbol)
        existing_run = self.store.get_research_run(run_id)
        if existing_run is None:
            started = self.start_research_run(
                run_date=run_date,
                output_dir=output_dir,
                symbol=normalized_symbol,
                objective=objective_template.format(symbol=normalized_symbol),
                report_data_mode=report_data_mode,
            )
            run_id = str(started["run_id"])
            normalized_symbol = str(started["symbol"])
        else:
            existing_mode = report_data_mode_from_run(existing_run)
            if existing_mode != report_data_mode:
                raise ValueError(mode_error)
            raise ValueError(
                "research run already exists; Phase 4 full flows do not implicitly resume "
                f"stored evidence: {run_id}"
            )
        self.phase4_universe_discovery(run_id=run_id, symbol=normalized_symbol)
        self.phase4_market_data(run_id=run_id, symbol=normalized_symbol)
        self.phase4_technical_package(run_id=run_id, symbol=normalized_symbol)
        self.phase4_social_evidence(run_id=run_id, symbol=normalized_symbol)
        self.phase4_news_catalyst(run_id=run_id, symbol=normalized_symbol)
        self.phase4_fundamentals(run_id=run_id, symbol=normalized_symbol)
        self.phase4_sector_macro(run_id=run_id, symbol=normalized_symbol)
        self.phase4_candidate_synthesis(run_id=run_id, symbol=normalized_symbol)
        self.phase4_prediction_evaluation(run_id=run_id, symbol=normalized_symbol)
        report = self.render_prediction_report(run_id=run_id, symbol=normalized_symbol)
        return {"run_id": run_id, "symbol": normalized_symbol, "report": report}

    def phase4_candidate_synthesis(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        evidence_count = len(self.store.list_evidence_for_run(run_id))
        mode_adapter = self._mode_adapter(run)

        def action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
            message = mode_adapter.no_evidence_candidate_message(evidence_count)
            if message is not None:
                return Phase4ToolRunOutcome(
                    status="empty",
                    payload={
                        "run_id": run_id,
                        "candidate_count": 0,
                        "warnings": [message],
                    },
                    warnings=(message,),
                    metadata={"candidate_count": 0, "live_no_evidence": True},
                )
            result = self._synthesize_phase4_prediction_candidate(
                run_id=run_id,
                symbol=normalized_symbol,
                context=context,
            )
            raw_warnings = result.get("warnings", [])
            warnings = (
                tuple(str(item) for item in raw_warnings) if isinstance(raw_warnings, list) else ()
            )
            return Phase4ToolRunOutcome(
                status="partial" if warnings else "successful",
                payload=result,
                artifact_ids=(str(result["artifact_id"]),),
                warnings=warnings,
                metadata={
                    "candidate_id": str(result["candidate_id"]),
                    "evidence_count": evidence_count,
                },
            )

        return self._execute_symbol_tool(
            run_id=run_id,
            symbol=normalized_symbol,
            tool_id=PHASE4_CANDIDATE_SYNTHESIS_TOOL_ID,
            inputs={"symbol": normalized_symbol, "evidence_count": evidence_count},
            action=action,
        ).payload

    def render_prediction_report(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        report_data_mode = report_data_mode_from_run(run)
        candidates = self.store.list_prediction_candidates_for_run(run_id)
        initial_warnings = () if candidates else (MISSING_CANDIDATE_WARNING,)

        def action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
            result = ReportBundleBuilder().build(
                PredictionReportBuildRequest(
                    store=self.store,
                    repo_root=self.repo_root,
                    run=run,
                    paths=context.paths,
                    run_date=run_date_from_run(run),
                    symbol=normalized_symbol,
                    tool_run_id=context.tool_run_id,
                    record_tool_run=False,
                    tool_name=context.tool.tool_name,
                    tool_version=context.tool.tool_version,
                    tool_warnings=initial_warnings,
                    produced_by=context.tool.tool_name,
                    artifact_schema_version="phase4-report.v1",
                    insufficient_evidence_summary=MISSING_CANDIDATE_WARNING,
                    report_data_mode=report_data_mode,
                )
            )
            artifact_ids = (
                f"artifact-report-md-{stable_digest(run.run_id)}",
                f"artifact-report-json-{stable_digest(run.run_id)}",
                f"artifact-audit-manifest-{stable_digest(run.run_id)}",
            )
            emitted_candidate_count = _json_int(result.get("candidate_count"))
            result_warnings = _json_string_tuple(result.get("warnings"))
            warnings = tuple(dict.fromkeys((*initial_warnings, *result_warnings)))
            status: Phase4ToolRunStatus = "successful" if emitted_candidate_count > 0 else "empty"
            return Phase4ToolRunOutcome(
                status=status,
                payload=result,
                artifact_ids=artifact_ids,
                warnings=warnings,
                metadata={
                    "candidate_count": len(candidates),
                    "emitted_candidate_count": emitted_candidate_count,
                    "excluded_candidate_ids": list(
                        _json_string_tuple(result.get("excluded_candidate_ids"))
                    ),
                    "final_only": True,
                    **report_data_mode_metadata(report_data_mode),
                },
            )

        return self._execute_symbol_tool(
            run_id=run_id,
            symbol=normalized_symbol,
            tool_id=PHASE4_REPORT_TOOL_ID,
            inputs={
                "symbol": normalized_symbol,
                "candidate_count": len(candidates),
                **report_data_mode_metadata(report_data_mode),
            },
            action=action,
        ).payload

    def inspect_research_run(self, *, run_id: str) -> JsonObject:
        run = self._require_run(run_id)
        tool_runs = self.store.list_tool_runs_for_run(run_id)
        raw_status_counts: dict[str, int] = {}
        for tool_run in tool_runs:
            raw_status_counts[tool_run.status] = raw_status_counts.get(tool_run.status, 0) + 1
        return {
            "run_id": run.run_id,
            "status": run.status,
            "tool_run_count": len(tool_runs),
            "tool_status_counts": cast(JsonObject, raw_status_counts),
            "artifact_count": len(self.store.list_artifacts_for_run(run_id)),
            "source_query_count": len(self.store.list_source_queries_for_run(run_id)),
            "evidence_count": len(self.store.list_evidence_for_run(run_id)),
            "candidate_count": len(self.store.list_prediction_candidates_for_run(run_id)),
        }

    def _synthesize_phase4_prediction_candidate(
        self,
        *,
        run_id: str,
        symbol: str,
        context: Phase4ToolRunContext,
    ) -> JsonObject:
        evidence = self.store.list_evidence_for_run(run_id)
        evidence_for = tuple(
            record.evidence_id
            for record in evidence
            if evidence_stance_from_record(record) == "supports"
        )
        evidence_against = tuple(
            record.evidence_id
            for record in evidence
            if evidence_stance_from_record(record) == "contradicts"
        )
        instrument_id = self._mode_adapter(self._require_run(run_id)).instrument_id(symbol)
        instrument = self.store.get_instrument(instrument_id)
        if instrument is None:
            raise ValueError(f"phase4 candidate synthesis requires instrument: {instrument_id}")
        candidate_symbol = instrument.symbol
        candidate_id = f"candidate-phase4-{symbol_slug(symbol)}-{stable_digest(run_id)[:8]}"
        status = (
            "contradicted"
            if evidence_against
            else "evidence_supported"
            if evidence_for
            else "insufficient_evidence"
        )
        confidence = (
            0.42 if evidence_for and not evidence_against else 0.28 if evidence_for else 0.18
        )
        signal_artifacts = signal_artifact_references_for_records(
            self.store.list_artifacts_for_run(run_id)
        )
        warnings = (
            ()
            if evidence_for or evidence_against
            else ("No attributable directional source evidence was available.",)
        )
        candidate = PredictionCandidateRecord(
            candidate_id=candidate_id,
            run_id=run_id,
            instrument_id=instrument_id,
            prediction_horizon=TimeHorizon.SWING.value,
            prediction_type=PredictionType.DIRECTIONAL.value,
            scenario=_phase4_candidate_scenario(
                symbol=candidate_symbol,
                evidence_for=bool(evidence_for),
                evidence_against=bool(evidence_against),
            ),
            direction=Direction.MIXED.value,
            confidence=confidence,
            status=status,
            evidence_for=evidence_for[:5],
            evidence_against=evidence_against[:5],
            signal_artifacts=tuple(reference.artifact_id for reference in signal_artifacts),
            baseline={
                "summary": "No directional edge is assumed without source-backed evidence.",
                "comparison": "baseline_neutral",
            },
            uncertainty=(
                "Candidate synthesis is conservative and depends on source attribution, "
                "freshness, and contradictory evidence."
            ),
            metadata={
                "phase4_candidate_synthesis": True,
                "symbol": candidate_symbol.upper(),
                "requested_symbol": symbol.upper(),
                "source_evidence_count": len(evidence),
                "signal_artifacts": [*signal_artifact_metadata(signal_artifacts)],
            },
        )
        artifact_id = f"artifact-phase4-prediction-inputs-{stable_digest(run_id)}"
        artifact = context.artifact_index(
            produced_by=context.tool.tool_name,
            schema_version="phase4-candidate-synthesis.v1",
        ).write_json(
            artifact_id=artifact_id,
            artifact_type="prediction_input",
            filename=f"prediction-inputs/{symbol_slug(symbol)}.json",
            payload={
                "schema_version": "phase4-candidate-synthesis.v1",
                "run_id": run_id,
                "candidate_id": candidate_id,
                "symbol": candidate_symbol.upper(),
                "requested_symbol": symbol.upper(),
                "evidence_for": list(candidate.evidence_for),
                "evidence_against": list(candidate.evidence_against),
                "signal_artifacts": [*signal_artifact_metadata(signal_artifacts)],
                "status": status,
                "warnings": list(warnings),
            },
            record_count=1,
            metadata={"candidate_id": candidate_id, "symbol": candidate_symbol.upper()},
        )
        self.store.upsert_prediction_candidate(candidate)
        self.store.link_candidate_artifact(
            CandidateArtifactLinkRecord(
                candidate_id=candidate_id,
                artifact_id=artifact.artifact_id,
                relationship="prediction_input",
                metadata={"source": "phase4_candidate_synthesis"},
                created_at=context.started_at,
            )
        )
        for evidence_id in candidate.evidence_for:
            self.store.link_candidate_evidence(
                CandidateEvidenceLinkRecord(
                    candidate_id=candidate_id,
                    evidence_id=evidence_id,
                    relationship="supports",
                    metadata={"source": "phase4_candidate_synthesis"},
                    created_at=context.started_at,
                )
            )
        for evidence_id in candidate.evidence_against:
            self.store.link_candidate_evidence(
                CandidateEvidenceLinkRecord(
                    candidate_id=candidate_id,
                    evidence_id=evidence_id,
                    relationship="contradicts",
                    metadata={"source": "phase4_candidate_synthesis"},
                    created_at=context.started_at,
                )
            )
        return {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "artifact_id": artifact.artifact_id,
            "artifact_path": artifact.path,
            "evidence_for": list(candidate.evidence_for),
            "evidence_against": list(candidate.evidence_against),
            "warnings": list(warnings),
        }

    def _execute_symbol_tool(
        self,
        *,
        run_id: str,
        symbol: str,
        tool_id: str,
        inputs: JsonObject,
        action: Phase4ToolAction,
    ) -> Phase4ToolRunOutcome:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]), symbol=normalized_symbol)
        return execute_phase4_tool(
            store=self.store,
            repo_root=self.repo_root,
            paths=paths,
            run_id=run_id,
            tool=self.registry.get(tool_id),
            inputs={
                **inputs,
                "symbol": normalized_symbol,
                **report_data_mode_metadata_from_run(run),
            },
            action=action,
        )

    def _require_run(self, run_id: str) -> ResearchRunRecord:
        run = self.store.get_research_run(run_id)
        if run is None:
            raise ValueError(f"research run does not exist: {run_id}")
        return run

    def _validated_symbol(self, run: ResearchRunRecord, symbol: str) -> str:
        stored_symbol = run.metadata.get("symbol")
        if not isinstance(stored_symbol, str) or not stored_symbol.strip():
            raise ValueError(f"research run is missing stored symbol metadata: {run.run_id}")
        normalized_stored_symbol = stored_symbol.strip().upper()
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must be non-empty")
        if normalized_symbol != normalized_stored_symbol:
            raise ValueError(
                f"symbol {normalized_symbol} does not match research run symbol "
                f"{normalized_stored_symbol}"
            )
        return normalized_stored_symbol

    def _paths(
        self,
        run_date: date,
        output_dir: str,
        *,
        symbol: str | None = None,
    ) -> Phase2RunPaths:
        return self.write_policy.run_paths(run_date, output_dir, symbol=symbol)

    def _resolve_write_path(self, path: Path) -> Path:
        return self.write_policy.resolve(path)

    def _run_context(self, run: ResearchRunRecord, paths: Phase2RunPaths) -> RunContext:
        generated_at = _phase4_timestamp_for_run(run)
        return RunContext(
            run_id=run.run_id,
            run_date=run_date_from_run(run),
            generated_at=generated_at,
            timezone="UTC",
            output_dir=paths.output_dir,
            report_dir=paths.run_dir,
            audit_dir=paths.audit_dir,
            command_args={
                "phase": "phase4",
                "output_dir": paths.output_dir.as_posix(),
                **report_data_mode_metadata_from_run(run),
            },
            artifact_writer=ArtifactWriter(
                base_dir=paths.audit_dir,
                created_at=generated_at,
                produced_by=PHASE4_UNIVERSE_TOOL_NAME,
                default_metadata=report_data_mode_metadata_from_run(run),
            ),
        )

    def _mode_adapter(self, run: ResearchRunRecord) -> Phase4RunModeAdapter:
        return Phase4RunModeAdapter(
            report_data_mode=report_data_mode_from_run(run),
            store=self.store,
            fixtures=self._fixtures_for_run(run),
            live_providers=self.live_providers,
        )

    def _artifact_dir(self, run: ResearchRunRecord) -> Path:
        return self._paths(
            run_date_from_run(run),
            str(run.metadata["output_dir"]),
            symbol=cast(str, run.metadata.get("symbol")),
        ).audit_dir

    def _instrument_id(
        self,
        symbol: str,
        *,
        report_data_mode: ReportDataMode | None = None,
    ) -> str:
        mode = report_data_mode or OFFLINE_FIXTURE_REPORT_DATA_MODE
        return Phase4RunModeAdapter(
            report_data_mode=mode,
            store=self.store,
            fixtures=self.fixtures,
            live_providers=self.live_providers,
        ).instrument_id(symbol)

    def _latest_artifact_path(self, run_id: str, artifact_type: str) -> Path | None:
        for artifact in reversed(self.store.list_artifacts_for_run(run_id)):
            if artifact.artifact_type == artifact_type:
                path = Path(artifact.path)
                return path if path.is_absolute() else self.repo_root / path
        return None


def _phase4_tool_run_id(
    *,
    run_id: str,
    tool: Phase4ToolMetadata,
    inputs: JsonObject,
) -> str:
    digest_source = stable_json_bytes(
        {
            "run_id": run_id,
            "tool_id": tool.tool_id,
            "tool_name": tool.tool_name,
            "tool_version": tool.tool_version,
            "inputs": dict(inputs),
        }
    ).decode("utf-8")
    return f"tool-{symbol_slug(tool.tool_name)}-{stable_digest(digest_source)}"


def _phase4_candidate_scenario(
    *,
    symbol: str,
    evidence_for: bool,
    evidence_against: bool,
) -> str:
    normalized = symbol.upper()
    if evidence_for and evidence_against:
        return (
            f"Source evidence for {normalized} is mixed, so the reportable scenario remains "
            "contested and conservative."
        )
    if evidence_against:
        return (
            f"Source evidence for {normalized} is mostly contradictory, so no supported "
            "directional scenario is produced."
        )
    if evidence_for:
        return (
            f"Source evidence for {normalized} supports a monitored prediction scenario, "
            "subject to freshness, attribution, and baseline checks."
        )
    return f"Insufficient attributable source evidence is available for {normalized}."


def _failed_phase4_tool_run_id(
    *,
    tool_run_id: str,
    previous_exists: bool,
    completed_at: datetime,
    error_message: str,
) -> str:
    if not previous_exists:
        return tool_run_id
    digest = hashlib.sha256(
        f"{tool_run_id}|{completed_at.isoformat()}|{error_message}".encode()
    ).hexdigest()[:12]
    return f"{tool_run_id}-failed-{digest}"


def _tool_run_inputs(
    *,
    tool: Phase4ToolMetadata,
    inputs: JsonObject,
    outcome: Phase4ToolRunOutcome | None = None,
) -> JsonObject:
    payload: JsonObject = {
        "phase": "phase4_runtime_report",
        "tool_id": tool.tool_id,
        "stage": tool.stage,
        "dependencies": list(tool.dependencies),
        "inputs": dict(inputs),
    }
    if outcome is not None:
        payload["outcome"] = {
            "status": outcome.status,
            "artifact_ids": list(outcome.artifact_ids),
            "warnings": list(outcome.warnings),
            "metadata": dict(outcome.metadata),
        }
    return payload


def _with_execution_payload(
    *,
    outcome: Phase4ToolRunOutcome,
    run_id: str,
    tool_run_id: str,
) -> Phase4ToolRunOutcome:
    payload = dict(outcome.payload)
    payload["run_id"] = run_id
    payload["tool_run_id"] = tool_run_id
    payload["status"] = outcome.status
    payload["artifact_ids"] = list(outcome.artifact_ids)
    payload["warnings"] = list(outcome.warnings)
    return outcome.model_copy(update={"payload": payload})


def _json_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def _json_string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _phase4_tool_result_payload(result: Phase4ToolResult) -> JsonObject:
    return {
        "run_id": result.run_id,
        "tool_run_id": result.tool_run_id,
        "artifact_id": result.artifact_id,
        "artifact_path": result.artifact_path.as_posix(),
        "status": result.status,
        "evidence_ids": list(result.evidence_ids),
        "source_query_ids": list(result.source_query_ids),
        "warnings": list(result.warnings),
    }


def _phase4_timestamp_for_run(run: ResearchRunRecord | None) -> datetime:
    if run is None:
        return utc_now()
    try:
        mode = report_data_mode_from_run(run)
    except ValueError:
        return utc_now()
    if mode == OFFLINE_FIXTURE_REPORT_DATA_MODE:
        return deterministic_generated_at(run_date_from_run(run))
    return utc_now()


__all__ = [
    "ALLOWED_WRITE_ROOTS",
    "MISSING_CANDIDATE_WARNING",
    "PHASE4_CANDIDATE_SYNTHESIS_TOOL_ID",
    "PHASE4_COLLECT_TOOL_ID",
    "PHASE4_EVALUATE_TOOL_ID",
    "PHASE4_FUNDAMENTALS_TOOL_ID",
    "PHASE4_MARKET_DATA_TOOL_ID",
    "PHASE4_NEWS_TOOL_ID",
    "PHASE4_PREDICTION_EVALUATION_TOOL_ID",
    "PHASE4_REPORT_TOOL_ID",
    "PHASE4_SECTOR_MACRO_TOOL_ID",
    "PHASE4_SOCIAL_TOOL_ID",
    "PHASE4_STAGE_ORDER",
    "PHASE4_TECHNICAL_TOOL_ID",
    "PHASE4_UNIVERSE_TOOL_ID",
    "Phase4Service",
    "Phase4Stage",
    "Phase4ToolExecutionError",
    "Phase4ToolMetadata",
    "Phase4ToolRegistry",
    "Phase4ToolRunContext",
    "Phase4ToolRunOutcome",
    "Phase4ToolRunStatus",
    "build_phase4_tool_registry",
    "execute_phase4_tool",
    "phase4_research_tool_plan",
    "phase4_run_id",
]
