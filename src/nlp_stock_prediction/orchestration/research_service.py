"""Research Stage runtime and final-report service foundation."""

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
from nlp_stock_prediction.orchestration.codex_smoke_evidence import evidence_stance_from_record
from nlp_stock_prediction.orchestration.context import RunContext, deterministic_generated_at
from nlp_stock_prediction.orchestration.orchestration_common import (
    ALLOWED_WRITE_ROOTS,
    ArtifactWritePolicy,
    ReportRunPaths,
    run_date_from_run,
    stable_digest,
    symbol_slug,
    utc_now,
)
from nlp_stock_prediction.orchestration.report_builder import (
    PredictionReportBuildRequest,
    ReportBundleBuilder,
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
from nlp_stock_prediction.orchestration.research_common import ResearchToolResult
from nlp_stock_prediction.orchestration.research_evaluation import (
    evaluate_stored_prediction_candidates,
)
from nlp_stock_prediction.orchestration.research_fixture_providers import (
    ResearchFixtureProviderFactory,
)
from nlp_stock_prediction.orchestration.research_fundamentals import (
    RESEARCH_FUNDAMENTALS_TOOL_NAME,
    run_research_fundamentals_tool,
)
from nlp_stock_prediction.orchestration.research_live_providers import (
    ResearchLiveProviderFactory,
    ResearchLiveProviderFactoryProtocol,
)
from nlp_stock_prediction.orchestration.research_market_data import (
    RESEARCH_MARKET_DATA_TOOL_NAME,
    ResearchMarketDataTool,
)
from nlp_stock_prediction.orchestration.research_news import (
    RESEARCH_NEWS_TOOL_NAME,
    run_research_news_catalyst_tool,
)
from nlp_stock_prediction.orchestration.research_run_modes import ResearchRunModeAdapter
from nlp_stock_prediction.orchestration.research_sector_macro import (
    RESEARCH_SECTOR_MACRO_TOOL_NAME,
    run_research_sector_macro_tool,
)
from nlp_stock_prediction.orchestration.research_social import (
    RESEARCH_SOCIAL_TOOL_NAME,
    run_research_social_evidence_tool,
)
from nlp_stock_prediction.orchestration.research_technical_package import (
    RESEARCH_TECHNICAL_PACKAGE_TOOL_NAME,
    ResearchTechnicalPackageTool,
)
from nlp_stock_prediction.orchestration.research_universe_discovery import (
    RESEARCH_TOOL_NAME as RESEARCH_UNIVERSE_TOOL_NAME,
)
from nlp_stock_prediction.orchestration.research_universe_discovery import (
    ResearchUniverseDiscoveryTool,
)
from nlp_stock_prediction.orchestration.signal_artifacts import (
    signal_artifact_metadata,
    signal_artifact_references_for_records,
)
from nlp_stock_prediction.reporting.audit import stable_json_bytes
from nlp_stock_prediction.storage.records import (
    CandidateArtifactLinkRecord,
    CandidateEvidenceLinkRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore, initialize_research_database

RESEARCH_STAGE_ORDER: tuple[str, ...] = ("discover", "collect", "analyze", "evaluate", "report")
ResearchStage = Literal["discover", "collect", "analyze", "evaluate", "report"]
ResearchToolRunStatus = Literal["successful", "partial", "empty", "skipped", "failed"]

RESEARCH_COLLECT_TOOL_ID = "research.collect_codex_search_evidence"
RESEARCH_EVALUATE_TOOL_ID = "research.evaluate_prediction_candidates"
RESEARCH_REPORT_TOOL_ID = "research.render_final_report"
RESEARCH_UNIVERSE_TOOL_ID = "research.universe_discovery"
RESEARCH_MARKET_DATA_TOOL_ID = "research.market_data"
RESEARCH_TECHNICAL_TOOL_ID = "research.technical_package"
RESEARCH_SOCIAL_TOOL_ID = "research.social_evidence"
RESEARCH_NEWS_TOOL_ID = "research.news_catalyst"
RESEARCH_FUNDAMENTALS_TOOL_ID = "research.fundamentals"
RESEARCH_SECTOR_MACRO_TOOL_ID = "research.sector_macro"
RESEARCH_CANDIDATE_SYNTHESIS_TOOL_ID = "research.prediction_candidate_synthesis"
RESEARCH_PREDICTION_EVALUATION_TOOL_ID = "research.prediction_evaluation"

MISSING_CANDIDATE_WARNING = (
    "No stored prediction candidates were available; Research Stage report rendering is final-only "
    "and did not synthesize candidates."
)


class ResearchToolMetadata(ContractModel):
    """Canonical metadata for one Research Stage research tool."""

    tool_id: NonEmptyStr
    tool_name: NonEmptyStr
    tool_version: NonEmptyStr
    stage: ResearchStage
    description: NonEmptyStr
    artifact_kinds: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    offline_capable: bool = True
    live_capable: bool = False
    requires_network: bool = False
    dependencies: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tool_metadata(self) -> ResearchToolMetadata:
        if self.requires_network and not self.live_capable:
            raise ValueError("network-dependent tools must be live_capable")
        if self.tool_id in self.dependencies:
            raise ValueError("research tools cannot depend on themselves")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("research tool dependencies must be unique")
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


class ResearchToolRegistry:
    """Deterministic Research Stage tool metadata registry."""

    def __init__(self, tools: Iterable[ResearchToolMetadata] = ()) -> None:
        self._tools: dict[str, ResearchToolMetadata] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ResearchToolMetadata) -> None:
        if tool.tool_id in self._tools:
            raise ValueError(f"research tool is already registered: {tool.tool_id}")
        if any(
            item.tool_name == tool.tool_name and item.tool_version == tool.tool_version
            for item in self._tools.values()
        ):
            raise ValueError(
                f"research tool name/version is already registered: "
                f"{tool.tool_name}@{tool.tool_version}"
            )
        self._tools[tool.tool_id] = tool

    def get(self, tool_id: str) -> ResearchToolMetadata:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"research tool is not registered: {tool_id}") from exc

    def specs(self) -> tuple[ResearchToolMetadata, ...]:
        return self.ordered_for_stages(RESEARCH_STAGE_ORDER)

    def ordered_for_stages(
        self,
        stage_order: Sequence[str] = RESEARCH_STAGE_ORDER,
    ) -> tuple[ResearchToolMetadata, ...]:
        stages = tuple(stage_order)
        if len(set(stages)) != len(stages):
            raise ValueError("stage_order entries must be unique")
        stage_index = {stage: index for index, stage in enumerate(stages)}
        unknown_stages = sorted(
            {tool.stage for tool in self._tools.values() if tool.stage not in stage_index}
        )
        if unknown_stages:
            joined = ", ".join(unknown_stages)
            raise ValueError(f"research tools use stages missing from stage_order: {joined}")
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
            "stage_order": list(RESEARCH_STAGE_ORDER),
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
            raise ValueError(f"research tool dependencies are not registered: {details}")


class ResearchToolRunOutcome(ContractModel):
    """Action result consumed by the durable Research Stage execution helper."""

    status: ResearchToolRunStatus = "successful"
    payload: JsonObject = Field(default_factory=dict)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    warnings: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    error_message: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome(self) -> ResearchToolRunOutcome:
        if self.status == "partial" and not self.warnings:
            raise ValueError("partial Research Stage tool outcomes require warnings")
        if self.status == "failed" and not self.error_message:
            raise ValueError("failed Research Stage tool outcomes require error_message")
        return self


@dataclass(frozen=True)
class ResearchToolRunContext:
    """Context object passed into one durable Research Stage tool action."""

    store: SQLiteStore
    repo_root: Path
    paths: ReportRunPaths
    run_id: str
    tool: ResearchToolMetadata
    tool_run_id: str
    started_at: datetime
    inputs: JsonObject

    def artifact_index(
        self,
        *,
        base_dir: Path | None = None,
        produced_by: str | None = None,
        schema_version: str = "research-tool-artifact.v1",
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


class ResearchToolExecutionError(RuntimeError):
    """Raised after a failed Research Stage tool run has been durably recorded."""

    def __init__(
        self,
        *,
        tool_run_id: str,
        tool_id: str,
        tool_name: str,
        original_error: Exception,
    ) -> None:
        super().__init__(f"Research Stage tool {tool_name!r} failed: {original_error}")
        self.tool_run_id = tool_run_id
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.original_error = original_error


class _ResearchToolReturnedFailure(RuntimeError):
    """Internal sentinel for actions that return a failed outcome."""


ResearchToolAction = Callable[[ResearchToolRunContext], ResearchToolRunOutcome]


def build_research_tool_registry() -> ResearchToolRegistry:
    """Build the default Research Stage metadata registry."""

    return ResearchToolRegistry(
        (
            ResearchToolMetadata(
                tool_id=RESEARCH_UNIVERSE_TOOL_ID,
                tool_name=RESEARCH_UNIVERSE_TOOL_NAME,
                tool_version="research.universe.v1",
                stage="discover",
                description=(
                    "Resolve the requested instrument universe and write instrument-universe "
                    "artifacts."
                ),
                artifact_kinds=("instrument_universe",),
                offline_capable=True,
                live_capable=True,
            ),
            ResearchToolMetadata(
                tool_id=RESEARCH_MARKET_DATA_TOOL_ID,
                tool_name=RESEARCH_MARKET_DATA_TOOL_NAME,
                tool_version="research.market-data.v1",
                stage="collect",
                description="Fetch daily market data and preserve provider provenance.",
                artifact_kinds=("market_data",),
                offline_capable=True,
                live_capable=True,
                requires_network=True,
                dependencies=(RESEARCH_UNIVERSE_TOOL_ID,),
            ),
            ResearchToolMetadata(
                tool_id=RESEARCH_TECHNICAL_TOOL_ID,
                tool_name=RESEARCH_TECHNICAL_PACKAGE_TOOL_NAME,
                tool_version="research.technical-package.v1",
                stage="analyze",
                description=(
                    "Compute deterministic technical context from market-data artifacts; raw "
                    "TimesFM remains sidecar context only."
                ),
                artifact_kinds=("technical_package",),
                live_capable=True,
                dependencies=(RESEARCH_MARKET_DATA_TOOL_ID,),
            ),
            ResearchToolMetadata(
                tool_id=RESEARCH_SOCIAL_TOOL_ID,
                tool_name=RESEARCH_SOCIAL_TOOL_NAME,
                tool_version="research.evidence-suite.v1",
                stage="collect",
                description="Normalize social evidence and preserve source/provider provenance.",
                artifact_kinds=("normalized_evidence",),
                live_capable=True,
                requires_network=True,
                dependencies=(RESEARCH_UNIVERSE_TOOL_ID,),
            ),
            ResearchToolMetadata(
                tool_id=RESEARCH_NEWS_TOOL_ID,
                tool_name=RESEARCH_NEWS_TOOL_NAME,
                tool_version="research.evidence-suite.v1",
                stage="collect",
                description="Normalize news/catalyst evidence with derived labels.",
                artifact_kinds=("normalized_evidence",),
                live_capable=True,
                requires_network=True,
                dependencies=(RESEARCH_UNIVERSE_TOOL_ID,),
            ),
            ResearchToolMetadata(
                tool_id=RESEARCH_FUNDAMENTALS_TOOL_ID,
                tool_name=RESEARCH_FUNDAMENTALS_TOOL_NAME,
                tool_version="research.evidence-suite.v1",
                stage="collect",
                description="Fetch fundamentals evidence and write analysis context artifacts.",
                artifact_kinds=("analysis_context",),
                live_capable=True,
                requires_network=True,
                dependencies=(RESEARCH_UNIVERSE_TOOL_ID,),
            ),
            ResearchToolMetadata(
                tool_id=RESEARCH_SECTOR_MACRO_TOOL_ID,
                tool_name=RESEARCH_SECTOR_MACRO_TOOL_NAME,
                tool_version="research.evidence-suite.v1",
                stage="analyze",
                description="Build sector and macro baseline context with provenance.",
                artifact_kinds=("analysis_context",),
                live_capable=True,
                requires_network=True,
                dependencies=(RESEARCH_FUNDAMENTALS_TOOL_ID,),
            ),
            ResearchToolMetadata(
                tool_id=RESEARCH_CANDIDATE_SYNTHESIS_TOOL_ID,
                tool_name="research_prediction_candidate_synthesis",
                tool_version="research.v1",
                stage="evaluate",
                description=(
                    "Synthesize conservative prediction candidates from stored, attributable "
                    "Research Stage evidence."
                ),
                artifact_kinds=("prediction_input",),
                live_capable=True,
                dependencies=(
                    RESEARCH_SOCIAL_TOOL_ID,
                    RESEARCH_NEWS_TOOL_ID,
                    RESEARCH_FUNDAMENTALS_TOOL_ID,
                    RESEARCH_SECTOR_MACRO_TOOL_ID,
                ),
            ),
            ResearchToolMetadata(
                tool_id=RESEARCH_PREDICTION_EVALUATION_TOOL_ID,
                tool_name="research_prediction_evaluation",
                tool_version="research.v1",
                stage="evaluate",
                description="Evaluate stored prediction candidates as prediction-quality records.",
                artifact_kinds=("prediction_evaluation",),
                live_capable=True,
                dependencies=(
                    RESEARCH_CANDIDATE_SYNTHESIS_TOOL_ID,
                    RESEARCH_TECHNICAL_TOOL_ID,
                    RESEARCH_SOCIAL_TOOL_ID,
                    RESEARCH_NEWS_TOOL_ID,
                    RESEARCH_FUNDAMENTALS_TOOL_ID,
                    RESEARCH_SECTOR_MACRO_TOOL_ID,
                ),
            ),
            ResearchToolMetadata(
                tool_id=RESEARCH_REPORT_TOOL_ID,
                tool_name="render_prediction_report",
                tool_version="research.v1",
                stage="report",
                description=(
                    "Final-only report renderer that consumes stored evidence, candidates, "
                    "and artifacts."
                ),
                artifact_kinds=("markdown_report", "json_report", "audit_manifest"),
                live_capable=True,
                dependencies=(RESEARCH_PREDICTION_EVALUATION_TOOL_ID,),
                metadata={"final_only": True},
            ),
        )
    )


def research_tool_plan(
    registry: ResearchToolRegistry | None = None,
) -> JsonObject:
    """Return a registry-derived Research Stage tool plan."""

    return (registry or build_research_tool_registry()).as_plan()


def research_run_id(run_date: date, symbol: str) -> str:
    return f"research-{run_date.isoformat()}-{symbol_slug(symbol)}"


def execute_research_tool(
    *,
    store: SQLiteStore,
    repo_root: Path,
    paths: ReportRunPaths,
    run_id: str,
    tool: ResearchToolMetadata,
    inputs: JsonObject,
    action: ResearchToolAction,
    tool_run_id: str | None = None,
) -> ResearchToolRunOutcome:
    """Execute one tool in a SQLite transaction and roll back new files on failure."""

    run = store.get_research_run(run_id)
    started_at = _research_timestamp_for_run(run)
    resolved_tool_run_id = tool_run_id or _research_tool_run_id(
        run_id=run_id,
        tool=tool,
        inputs=inputs,
    )
    base_inputs = _tool_run_inputs(tool=tool, inputs=inputs)
    previous_tool_run = store.get_tool_run(resolved_tool_run_id)
    file_transaction = ArtifactFileTransaction.begin(paths.run_dir)
    context = ResearchToolRunContext(
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
        final_outcome: ResearchToolRunOutcome
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
                raise _ResearchToolReturnedFailure(outcome.error_message or "tool returned failed")
            completed_at = _research_timestamp_for_run(run)
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
        completed_at = _research_timestamp_for_run(run)
        error_message = str(exc)
        if rollback_errors:
            error_message = f"{error_message}; artifact rollback errors: " + "; ".join(
                dict.fromkeys(rollback_errors)
            )
        failed_tool_run_id = _failed_research_tool_run_id(
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
        raise ResearchToolExecutionError(
            tool_run_id=failed_tool_run_id,
            tool_id=tool.tool_id,
            tool_name=tool.tool_name,
            original_error=exc,
        ) from exc


@dataclass(frozen=True)
class ResearchService:
    """Stateful Research Stage service over the research SQLite run graph."""

    repo_root: Path = Path(".")
    database_path: Path = Path("data/prediction-research.sqlite3")
    fixture_root: Path | None = None
    provider_cache_root: Path | None = None
    extra_write_roots: tuple[Path, ...] = ()
    registry: ResearchToolRegistry = field(default_factory=build_research_tool_registry)
    live_provider_factory: ResearchLiveProviderFactoryProtocol | None = field(
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
                ResearchLiveProviderFactory(cache_root=self.provider_cache_root),
            )
        object.__setattr__(
            self,
            "_store",
            initialize_research_database(self._resolve_write_path(self.database_path)),
        )

    @property
    def write_policy(self) -> ArtifactWritePolicy:
        return ArtifactWritePolicy(self.repo_root, extra_allowed_roots=self.extra_write_roots)

    @property
    def store(self) -> SQLiteStore:
        return self._store

    @property
    def fixtures(self) -> ResearchFixtureProviderFactory:
        return ResearchFixtureProviderFactory(cast(Path, self.fixture_root))

    def _fixtures_for_run(self, run: ResearchRunRecord) -> ResearchFixtureProviderFactory:
        return ResearchFixtureProviderFactory(
            cast(Path, self.fixture_root),
            fetched_at=_research_timestamp_for_run(run),
        )

    @property
    def live_providers(self) -> ResearchLiveProviderFactoryProtocol:
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
        run_id = research_run_id(parsed_date, normalized_symbol)
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
                run_kind="research_prediction_report",
                objective=(
                    objective or f"Research Stage prediction research for {normalized_symbol}"
                ),
                status="running",
                started_at=now,
                metadata={
                    "run_date": parsed_date.isoformat(),
                    "symbol": normalized_symbol,
                    "output_dir": paths.output_dir.as_posix(),
                    "stage": "research_runtime_report",
                    "stage_order": list(RESEARCH_STAGE_ORDER),
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
        return research_tool_plan(self.registry)

    def research_universe_discovery(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]), symbol=normalized_symbol)
        context = self._run_context(run, paths)
        request = InstrumentUniverseRequest(
            request_id=f"research-service-universe-{run_id}-{symbol_slug(normalized_symbol)}",
            as_of=context.generated_at,
            queries=(InstrumentQuery(query=normalized_symbol),),
        )
        universe_provider = self._mode_adapter(run).universe_provider()
        universe_tool = (
            ResearchUniverseDiscoveryTool(provider=universe_provider)
            if universe_provider is not None
            else ResearchUniverseDiscoveryTool()
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

    def research_market_data(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]), symbol=normalized_symbol)
        mode_adapter = self._mode_adapter(run)
        market_data = mode_adapter.market_data_selection(normalized_symbol)
        result = ResearchMarketDataTool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=paths.audit_dir,
            provider=market_data.provider,
            now=lambda: _research_timestamp_for_run(run),
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

    def research_technical_package(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]), symbol=normalized_symbol)
        market_path = self._latest_artifact_path(run_id, "market_data")
        if market_path is None:
            self.research_market_data(run_id=run_id, symbol=normalized_symbol)
            market_path = self._latest_artifact_path(run_id, "market_data")
        if market_path is None:
            raise ValueError("research technical package requires a market-data artifact")
        result = ResearchTechnicalPackageTool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=paths.audit_dir,
            now=lambda: _research_timestamp_for_run(run),
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

    def research_social_evidence(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        mode_adapter = self._mode_adapter(run)
        providers = mode_adapter.social_providers(normalized_symbol)
        instrument_id = mode_adapter.instrument_id(normalized_symbol)
        instrument = self.store.get_instrument(instrument_id)
        result = run_research_social_evidence_tool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir(run),
            run_id=run_id,
            symbol=normalized_symbol,
            run_date=run_date_from_run(run),
            generated_at=_research_timestamp_for_run(run),
            reddit_provider=providers.reddit_provider,
            instrument_id=instrument_id,
            company_name=instrument.name if instrument is not None else None,
            aliases=instrument.aliases if instrument is not None else (),
        )
        return _research_tool_result_payload(result)

    def research_news_catalyst(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        result = run_research_news_catalyst_tool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir(run),
            run_id=run_id,
            symbol=normalized_symbol,
            run_date=run_date_from_run(run),
            generated_at=_research_timestamp_for_run(run),
            providers=self._mode_adapter(run).news_providers(normalized_symbol),
            instrument_id=self._mode_adapter(run).instrument_id(normalized_symbol),
        )
        return _research_tool_result_payload(result)

    def research_fundamentals(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        result = run_research_fundamentals_tool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir(run),
            run_id=run_id,
            symbol=normalized_symbol,
            run_date=run_date_from_run(run),
            generated_at=_research_timestamp_for_run(run),
            providers=self._mode_adapter(run).fundamentals_providers(normalized_symbol),
            instrument_id=self._mode_adapter(run).instrument_id(normalized_symbol),
        )
        return _research_tool_result_payload(result)

    def research_sector_macro(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        result = run_research_sector_macro_tool(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir(run),
            run_id=run_id,
            symbol=normalized_symbol,
            run_date=run_date_from_run(run),
            generated_at=_research_timestamp_for_run(run),
            target_snapshot=FundamentalsSnapshot(ticker=normalized_symbol),
            macro_providers=self._mode_adapter(run).macro_providers(normalized_symbol),
            instrument_id=self._mode_adapter(run).instrument_id(normalized_symbol),
        )
        return _research_tool_result_payload(result)

    def research_prediction_evaluation(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        candidates = self.store.list_prediction_candidates_for_run(run_id)

        def action(context: ResearchToolRunContext) -> ResearchToolRunOutcome:
            if not candidates:
                return ResearchToolRunOutcome(
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
                generated_at=_research_timestamp_for_run(run),
            )
            return ResearchToolRunOutcome(
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
            tool_id=RESEARCH_PREDICTION_EVALUATION_TOOL_ID,
            inputs={"symbol": normalized_symbol, "candidate_count": len(candidates)},
            action=action,
        ).payload

    def run_offline_research_flow(
        self,
        *,
        run_date: str,
        output_dir: str,
        symbol: str,
    ) -> JsonObject:
        return self._run_research_flow(
            run_date=run_date,
            output_dir=output_dir,
            symbol=symbol,
            report_data_mode=OFFLINE_FIXTURE_REPORT_DATA_MODE,
            objective_template="Research Stage fixture-backed prediction research for {symbol}",
            mode_error="offline Research Stage flow requires an offline_fixture report_data_mode",
        )

    def run_live_research_flow(self, *, run_date: str, output_dir: str, symbol: str) -> JsonObject:
        return self._run_research_flow(
            run_date=run_date,
            output_dir=output_dir,
            symbol=symbol,
            report_data_mode=LIVE_REPORT_DATA_MODE,
            objective_template="Live provider prediction research for {symbol}",
            mode_error="live Research Stage flow requires a live report_data_mode",
        )

    def _run_research_flow(
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
        run_id = research_run_id(date.fromisoformat(run_date), normalized_symbol)
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
                "research run already exists; Research Stage full flows do not implicitly resume "
                f"stored evidence: {run_id}"
            )
        try:
            self.research_universe_discovery(run_id=run_id, symbol=normalized_symbol)
            self.research_market_data(run_id=run_id, symbol=normalized_symbol)
            self.research_technical_package(run_id=run_id, symbol=normalized_symbol)
            self.research_social_evidence(run_id=run_id, symbol=normalized_symbol)
            self.research_news_catalyst(run_id=run_id, symbol=normalized_symbol)
            self.research_fundamentals(run_id=run_id, symbol=normalized_symbol)
            self.research_sector_macro(run_id=run_id, symbol=normalized_symbol)
            self.research_candidate_synthesis(run_id=run_id, symbol=normalized_symbol)
            self.research_prediction_evaluation(run_id=run_id, symbol=normalized_symbol)
            report = self.render_prediction_report(run_id=run_id, symbol=normalized_symbol)
        except Exception as exc:
            self._finalize_research_run(run_id, status="failed", error_message=str(exc))
            raise
        self._finalize_research_run(run_id, status="completed")
        return {"run_id": run_id, "symbol": normalized_symbol, "report": report}

    def research_candidate_synthesis(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        evidence_count = len(self.store.list_evidence_for_run(run_id))
        mode_adapter = self._mode_adapter(run)

        def action(context: ResearchToolRunContext) -> ResearchToolRunOutcome:
            message = mode_adapter.no_evidence_candidate_message(evidence_count)
            if message is not None:
                return ResearchToolRunOutcome(
                    status="empty",
                    payload={
                        "run_id": run_id,
                        "candidate_count": 0,
                        "warnings": [message],
                    },
                    warnings=(message,),
                    metadata={"candidate_count": 0, "live_no_evidence": True},
                )
            result = self._synthesize_research_prediction_candidate(
                run_id=run_id,
                symbol=normalized_symbol,
                context=context,
            )
            raw_warnings = result.get("warnings", [])
            warnings = (
                tuple(str(item) for item in raw_warnings) if isinstance(raw_warnings, list) else ()
            )
            return ResearchToolRunOutcome(
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
            tool_id=RESEARCH_CANDIDATE_SYNTHESIS_TOOL_ID,
            inputs={"symbol": normalized_symbol, "evidence_count": evidence_count},
            action=action,
        ).payload

    def _finalize_research_run(
        self,
        run_id: str,
        *,
        status: str,
        error_message: str | None = None,
    ) -> None:
        run = self._require_run(run_id)
        metadata = dict(run.metadata)
        if error_message is not None:
            metadata["error_message"] = error_message
        self.store.upsert_research_run(
            ResearchRunRecord(
                run_id=run.run_id,
                run_kind=run.run_kind,
                objective=run.objective,
                status=status,
                started_at=run.started_at,
                completed_at=_research_timestamp_for_run(run),
                metadata=metadata,
            )
        )

    def render_prediction_report(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        report_data_mode = report_data_mode_from_run(run)
        candidates = self.store.list_prediction_candidates_for_run(run_id)
        initial_warnings = () if candidates else (MISSING_CANDIDATE_WARNING,)

        def action(context: ResearchToolRunContext) -> ResearchToolRunOutcome:
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
                    artifact_schema_version="research-report.v1",
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
            status: ResearchToolRunStatus = "successful" if emitted_candidate_count > 0 else "empty"
            return ResearchToolRunOutcome(
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
            tool_id=RESEARCH_REPORT_TOOL_ID,
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

    def _synthesize_research_prediction_candidate(
        self,
        *,
        run_id: str,
        symbol: str,
        context: ResearchToolRunContext,
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
        instrument_missing = instrument is None
        if instrument is None:
            instrument = InstrumentRecord(
                instrument_id=instrument_id,
                symbol=symbol.upper(),
                asset_class="unknown",
                name=f"{symbol.upper()} unresolved research symbol",
                metadata={
                    "research_candidate_synthesis": True,
                    "instrument_resolution_status": "unavailable",
                    "limitation": (
                        "No stored instrument identity was available; candidate remains "
                        "unavailable pending instrument resolution."
                    ),
                },
            )
            self.store.upsert_instrument(instrument)
        candidate_symbol = instrument.symbol
        candidate_id = f"candidate-research-{symbol_slug(symbol)}-{stable_digest(run_id)[:8]}"
        status = (
            "unavailable"
            if instrument_missing
            else "contradicted"
            if evidence_against
            else "evidence_supported"
            if evidence_for
            else "insufficient_evidence"
        )
        confidence = (
            0.08
            if instrument_missing
            else 0.42
            if evidence_for and not evidence_against
            else 0.28
            if evidence_for
            else 0.18
        )
        signal_artifacts = signal_artifact_references_for_records(
            self.store.list_artifacts_for_run(run_id)
        )
        warnings = tuple(
            warning
            for warning in (
                (
                    "No stored instrument identity was available; synthesized an "
                    "unavailable research placeholder."
                )
                if instrument_missing
                else None,
                None
                if evidence_for or evidence_against
                else "No attributable directional source evidence was available.",
            )
            if warning is not None
        )
        candidate = PredictionCandidateRecord(
            candidate_id=candidate_id,
            run_id=run_id,
            instrument_id=instrument_id,
            prediction_horizon=TimeHorizon.SWING.value,
            prediction_type=PredictionType.DIRECTIONAL.value,
            scenario=_research_candidate_scenario(
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
                "research_candidate_synthesis": True,
                "symbol": candidate_symbol.upper(),
                "requested_symbol": symbol.upper(),
                "source_evidence_count": len(evidence),
                "signal_artifacts": [*signal_artifact_metadata(signal_artifacts)],
            },
        )
        artifact_id = f"artifact-research-prediction-inputs-{stable_digest(run_id)}"
        artifact = context.artifact_index(
            produced_by=context.tool.tool_name,
            schema_version="research-candidate-synthesis.v1",
        ).write_json(
            artifact_id=artifact_id,
            artifact_type="prediction_input",
            filename=f"prediction-inputs/{symbol_slug(symbol)}.json",
            payload={
                "schema_version": "research-candidate-synthesis.v1",
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
                metadata={"source": "research_candidate_synthesis"},
                created_at=context.started_at,
            )
        )
        for evidence_id in candidate.evidence_for:
            self.store.link_candidate_evidence(
                CandidateEvidenceLinkRecord(
                    candidate_id=candidate_id,
                    evidence_id=evidence_id,
                    relationship="supports",
                    metadata={"source": "research_candidate_synthesis"},
                    created_at=context.started_at,
                )
            )
        for evidence_id in candidate.evidence_against:
            self.store.link_candidate_evidence(
                CandidateEvidenceLinkRecord(
                    candidate_id=candidate_id,
                    evidence_id=evidence_id,
                    relationship="contradicts",
                    metadata={"source": "research_candidate_synthesis"},
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
        action: ResearchToolAction,
    ) -> ResearchToolRunOutcome:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]), symbol=normalized_symbol)
        return execute_research_tool(
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
    ) -> ReportRunPaths:
        return self.write_policy.run_paths(run_date, output_dir, symbol=symbol)

    def _resolve_write_path(self, path: Path) -> Path:
        return self.write_policy.resolve(path)

    def _run_context(self, run: ResearchRunRecord, paths: ReportRunPaths) -> RunContext:
        generated_at = _research_timestamp_for_run(run)
        return RunContext(
            run_id=run.run_id,
            run_date=run_date_from_run(run),
            generated_at=generated_at,
            timezone="UTC",
            output_dir=paths.output_dir,
            report_dir=paths.run_dir,
            audit_dir=paths.audit_dir,
            command_args={
                "stage": "research",
                "output_dir": paths.output_dir.as_posix(),
                "database_path": self._resolve_write_path(self.database_path).as_posix(),
                **report_data_mode_metadata_from_run(run),
            },
            artifact_writer=ArtifactWriter(
                base_dir=paths.audit_dir,
                created_at=generated_at,
                produced_by=RESEARCH_UNIVERSE_TOOL_NAME,
                default_metadata=report_data_mode_metadata_from_run(run),
            ),
        )

    def _mode_adapter(self, run: ResearchRunRecord) -> ResearchRunModeAdapter:
        return ResearchRunModeAdapter(
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
        return ResearchRunModeAdapter(
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


def _research_tool_run_id(
    *,
    run_id: str,
    tool: ResearchToolMetadata,
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


def _research_candidate_scenario(
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


def _failed_research_tool_run_id(
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
    tool: ResearchToolMetadata,
    inputs: JsonObject,
    outcome: ResearchToolRunOutcome | None = None,
) -> JsonObject:
    payload: JsonObject = {
        "operation": "research_runtime_report",
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
    outcome: ResearchToolRunOutcome,
    run_id: str,
    tool_run_id: str,
) -> ResearchToolRunOutcome:
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


def _research_tool_result_payload(result: ResearchToolResult) -> JsonObject:
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


def _research_timestamp_for_run(run: ResearchRunRecord | None) -> datetime:
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
    "RESEARCH_CANDIDATE_SYNTHESIS_TOOL_ID",
    "RESEARCH_COLLECT_TOOL_ID",
    "RESEARCH_EVALUATE_TOOL_ID",
    "RESEARCH_FUNDAMENTALS_TOOL_ID",
    "RESEARCH_MARKET_DATA_TOOL_ID",
    "RESEARCH_NEWS_TOOL_ID",
    "RESEARCH_PREDICTION_EVALUATION_TOOL_ID",
    "RESEARCH_REPORT_TOOL_ID",
    "RESEARCH_SECTOR_MACRO_TOOL_ID",
    "RESEARCH_SOCIAL_TOOL_ID",
    "RESEARCH_STAGE_ORDER",
    "RESEARCH_TECHNICAL_TOOL_ID",
    "RESEARCH_UNIVERSE_TOOL_ID",
    "ResearchService",
    "ResearchStage",
    "ResearchToolExecutionError",
    "ResearchToolMetadata",
    "ResearchToolRegistry",
    "ResearchToolRunContext",
    "ResearchToolRunOutcome",
    "ResearchToolRunStatus",
    "build_research_tool_registry",
    "execute_research_tool",
    "research_run_id",
    "research_tool_plan",
]
