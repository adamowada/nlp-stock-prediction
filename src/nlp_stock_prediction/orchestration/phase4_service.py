"""Phase 4 runtime and final-report service foundation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, NonEmptyStr
from nlp_stock_prediction.orchestration.artifacts import ArtifactFileTransaction, ArtifactIndex
from nlp_stock_prediction.orchestration.phase2_common import (
    ALLOWED_WRITE_ROOTS,
    Phase2RunPaths,
    Phase2WritePolicy,
    run_date_from_run,
    stable_digest,
    symbol_slug,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase2_dummy_tools import (
    run_phase2_dummy_analysis_tool,
    run_phase2_dummy_universe_tool,
)
from nlp_stock_prediction.orchestration.phase2_evidence import record_codex_search_evidence
from nlp_stock_prediction.orchestration.phase2_report import render_phase2_prediction_report
from nlp_stock_prediction.orchestration.phase2_synthesis import synthesize_prediction_candidates
from nlp_stock_prediction.reporting.audit import stable_json_bytes
from nlp_stock_prediction.storage.records import ResearchRunRecord, ToolRunRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore, initialize_research_database

PHASE4_STAGE_ORDER: tuple[str, ...] = ("discover", "collect", "analyze", "evaluate", "report")
Phase4Stage = Literal["discover", "collect", "analyze", "evaluate", "report"]
Phase4ToolRunStatus = Literal["successful", "partial", "empty", "skipped", "failed"]

PHASE4_DISCOVER_TOOL_ID = "phase4.discover_instruments"
PHASE4_COLLECT_TOOL_ID = "phase4.collect_codex_search_evidence"
PHASE4_ANALYZE_TOOL_ID = "phase4.analyze_context"
PHASE4_EVALUATE_TOOL_ID = "phase4.evaluate_prediction_candidates"
PHASE4_REPORT_TOOL_ID = "phase4.render_final_report"

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
        return tuple(
            sorted(
                self._tools.values(),
                key=lambda tool: (stage_index[tool.stage], tool.tool_id),
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
        return ArtifactIndex.for_directory(
            store=self.store,
            repo_root=self.repo_root,
            base_dir=base_dir or self.paths.audit_dir,
            created_at=self.started_at,
            produced_by=produced_by or self.tool.tool_name,
            tool_run_id=self.tool_run_id,
            schema_version=schema_version,
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
                tool_id=PHASE4_DISCOVER_TOOL_ID,
                tool_name="run_dummy_universe_tool",
                tool_version="phase4.v1",
                stage="discover",
                description=(
                    "Resolve the requested instrument universe and write instrument-universe "
                    "artifacts."
                ),
                artifact_kinds=("instrument_universe",),
                offline_capable=True,
                live_capable=False,
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_COLLECT_TOOL_ID,
                tool_name="record_codex_search_evidence",
                tool_version="phase4.v1",
                stage="collect",
                description="Record Codex-supplied source material as normalized evidence.",
                artifact_kinds=("normalized_evidence",),
                offline_capable=True,
                live_capable=True,
                dependencies=(PHASE4_DISCOVER_TOOL_ID,),
                metadata={"source_material": "codex_search_or_browse"},
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_ANALYZE_TOOL_ID,
                tool_name="run_dummy_analysis_tool",
                tool_version="phase4.v1",
                stage="analyze",
                description="Write analysis context artifacts from collected evidence.",
                artifact_kinds=("analysis_context",),
                dependencies=(PHASE4_DISCOVER_TOOL_ID, PHASE4_COLLECT_TOOL_ID),
            ),
            Phase4ToolMetadata(
                tool_id=PHASE4_EVALUATE_TOOL_ID,
                tool_name="synthesize_prediction_candidates",
                tool_version="phase4.v1",
                stage="evaluate",
                description="Evaluate stored evidence into persisted prediction candidates.",
                artifact_kinds=("prediction_input",),
                dependencies=(
                    PHASE4_DISCOVER_TOOL_ID,
                    PHASE4_COLLECT_TOOL_ID,
                    PHASE4_ANALYZE_TOOL_ID,
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
                dependencies=(
                    PHASE4_DISCOVER_TOOL_ID,
                    PHASE4_COLLECT_TOOL_ID,
                    PHASE4_ANALYZE_TOOL_ID,
                    PHASE4_EVALUATE_TOOL_ID,
                ),
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

    started_at = utc_now()
    resolved_tool_run_id = tool_run_id or _phase4_tool_run_id(
        run_id=run_id,
        tool=tool,
        inputs=inputs,
    )
    base_inputs = _tool_run_inputs(tool=tool, inputs=inputs)
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
        with store.transaction():
            outcome = action(context)
            if outcome.status == "failed":
                raise _Phase4ToolReturnedFailure(outcome.error_message or "tool returned failed")
            completed_at = utc_now()
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
            return final_outcome
    except Exception as exc:
        file_transaction.rollback_new_files()
        completed_at = utc_now()
        error_message = str(exc)
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=resolved_tool_run_id,
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
            tool_run_id=resolved_tool_run_id,
            tool_id=tool.tool_id,
            tool_name=tool.tool_name,
            original_error=exc,
        ) from exc


@dataclass(frozen=True)
class Phase4Service:
    """Stateful Phase 4 service over the research SQLite run graph."""

    repo_root: Path = Path(".")
    database_path: Path = Path("data/prediction-research.sqlite3")
    registry: Phase4ToolRegistry = field(default_factory=build_phase4_tool_registry)
    _store: SQLiteStore = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", self.repo_root.resolve())
        object.__setattr__(
            self,
            "_store",
            initialize_research_database(self._resolve_write_path(self.database_path)),
        )

    @property
    def write_policy(self) -> Phase2WritePolicy:
        return Phase2WritePolicy(self.repo_root)

    @property
    def store(self) -> SQLiteStore:
        return self._store

    def start_research_run(
        self,
        *,
        run_date: str,
        output_dir: str,
        symbol: str,
        objective: str | None = None,
    ) -> JsonObject:
        parsed_date = date.fromisoformat(run_date)
        normalized_symbol = symbol.strip().upper()
        run_id = phase4_run_id(parsed_date, normalized_symbol)
        if self.store.get_research_run(run_id) is not None:
            raise ValueError(f"research run already exists: {run_id}")
        paths = self._paths(parsed_date, output_dir)
        now = utc_now()
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
        }

    def list_research_tool_plan(self) -> JsonObject:
        return phase4_research_tool_plan(self.registry)

    def record_codex_search_evidence(
        self,
        *,
        run_id: str,
        symbol: str,
        title: str,
        url: str,
        claim: str,
        query: str,
        published_at: str | None = None,
        stance: str | None = None,
    ) -> JsonObject:
        def action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
            result = record_codex_search_evidence(
                store=self.store,
                repo_root=self.repo_root,
                paths=context.paths,
                run_id=run_id,
                symbol=normalized_symbol,
                title=title,
                url=url,
                claim=claim,
                query=query,
                published_at=published_at,
                stance=stance,
                tool_run_id=context.tool_run_id,
                record_tool_run=False,
                tool_name=context.tool.tool_name,
                tool_version=context.tool.tool_version,
            )
            return Phase4ToolRunOutcome(
                status="successful",
                payload=result,
                artifact_ids=(str(result["artifact_id"]),),
                metadata={"evidence_id": str(result["evidence_id"])},
            )

        normalized_symbol = self._validated_symbol(self._require_run(run_id), symbol)
        return self._execute_symbol_tool(
            run_id=run_id,
            symbol=normalized_symbol,
            tool_id=PHASE4_COLLECT_TOOL_ID,
            inputs={
                "symbol": normalized_symbol,
                "title": title,
                "url": url,
                "claim": claim,
                "query": query,
                "published_at": published_at,
                "stance": stance,
            },
            action=action,
        ).payload

    def run_dummy_universe_tool(self, *, run_id: str, symbol: str) -> JsonObject:
        def action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
            result = run_phase2_dummy_universe_tool(
                store=self.store,
                repo_root=self.repo_root,
                paths=context.paths,
                run_id=run_id,
                symbol=normalized_symbol,
                tool_run_id=context.tool_run_id,
                record_tool_run=False,
                tool_name=context.tool.tool_name,
                tool_version=context.tool.tool_version,
            )
            warnings = _string_tuple(result.get("warnings"))
            return Phase4ToolRunOutcome(
                status="partial" if warnings else "successful",
                payload=result,
                artifact_ids=(str(result["artifact_id"]),),
                warnings=warnings,
                metadata={"instrument_count": len(cast(list[object], result["instrument_ids"]))},
            )

        normalized_symbol = self._validated_symbol(self._require_run(run_id), symbol)
        return self._execute_symbol_tool(
            run_id=run_id,
            symbol=normalized_symbol,
            tool_id=PHASE4_DISCOVER_TOOL_ID,
            inputs={"symbol": normalized_symbol},
            action=action,
        ).payload

    def run_dummy_analysis_tool(self, *, run_id: str, symbol: str) -> JsonObject:
        def action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
            result = run_phase2_dummy_analysis_tool(
                store=self.store,
                repo_root=self.repo_root,
                paths=context.paths,
                run_id=run_id,
                symbol=normalized_symbol,
                tool_run_id=context.tool_run_id,
                record_tool_run=False,
                tool_name=context.tool.tool_name,
                tool_version=context.tool.tool_version,
            )
            return Phase4ToolRunOutcome(
                status="successful",
                payload=result,
                artifact_ids=(str(result["artifact_id"]),),
            )

        normalized_symbol = self._validated_symbol(self._require_run(run_id), symbol)
        return self._execute_symbol_tool(
            run_id=run_id,
            symbol=normalized_symbol,
            tool_id=PHASE4_ANALYZE_TOOL_ID,
            inputs={"symbol": normalized_symbol},
            action=action,
        ).payload

    def synthesize_prediction_candidates(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        instrument_id = f"instrument:codex:{normalized_symbol}"
        if self.store.get_instrument(instrument_id) is None:
            self.run_dummy_universe_tool(run_id=run_id, symbol=normalized_symbol)

        def action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
            result = synthesize_prediction_candidates(
                store=self.store,
                repo_root=self.repo_root,
                paths=context.paths,
                run_id=run_id,
                symbol=normalized_symbol,
                ensure_instrument=lambda: None,
                tool_run_id=context.tool_run_id,
                record_tool_run=False,
                tool_name=context.tool.tool_name,
                tool_version=context.tool.tool_version,
            )
            return Phase4ToolRunOutcome(
                status="successful",
                payload=result,
                artifact_ids=(str(result["artifact_id"]),),
                metadata={"candidate_id": str(result["candidate_id"])},
            )

        return self._execute_symbol_tool(
            run_id=run_id,
            symbol=normalized_symbol,
            tool_id=PHASE4_EVALUATE_TOOL_ID,
            inputs={
                "symbol": normalized_symbol,
                "evidence_count": len(self.store.list_evidence_for_run(run_id)),
            },
            action=action,
        ).payload

    def render_prediction_report(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        candidates = self.store.list_prediction_candidates_for_run(run_id)
        warnings = () if candidates else (MISSING_CANDIDATE_WARNING,)
        status: Phase4ToolRunStatus = "successful" if candidates else "empty"

        def action(context: Phase4ToolRunContext) -> Phase4ToolRunOutcome:
            result = render_phase2_prediction_report(
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
                tool_warnings=warnings,
                produced_by=context.tool.tool_name,
                artifact_schema_version="phase4-report.v1",
                insufficient_evidence_summary=MISSING_CANDIDATE_WARNING,
            )
            artifact_ids = (
                f"artifact-report-md-{stable_digest(run.run_id)}",
                f"artifact-report-json-{stable_digest(run.run_id)}",
                f"artifact-audit-manifest-{stable_digest(run.run_id)}",
            )
            return Phase4ToolRunOutcome(
                status=status,
                payload=result,
                artifact_ids=artifact_ids,
                warnings=warnings,
                metadata={"candidate_count": len(candidates), "final_only": True},
            )

        return self._execute_symbol_tool(
            run_id=run_id,
            symbol=normalized_symbol,
            tool_id=PHASE4_REPORT_TOOL_ID,
            inputs={"symbol": normalized_symbol, "candidate_count": len(candidates)},
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
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        return execute_phase4_tool(
            store=self.store,
            repo_root=self.repo_root,
            paths=paths,
            run_id=run_id,
            tool=self.registry.get(tool_id),
            inputs={**inputs, "symbol": normalized_symbol},
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
        if normalized_symbol != normalized_stored_symbol:
            raise ValueError(
                f"symbol {normalized_symbol} does not match research run symbol "
                f"{normalized_stored_symbol}"
            )
        return normalized_stored_symbol

    def _paths(self, run_date: date, output_dir: str) -> Phase2RunPaths:
        return self.write_policy.run_paths(run_date, output_dir)

    def _resolve_write_path(self, path: Path) -> Path:
        return self.write_policy.resolve(path)


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


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if item)
    if isinstance(value, str) and value:
        return (value,)
    return ()


__all__ = [
    "ALLOWED_WRITE_ROOTS",
    "MISSING_CANDIDATE_WARNING",
    "PHASE4_ANALYZE_TOOL_ID",
    "PHASE4_COLLECT_TOOL_ID",
    "PHASE4_DISCOVER_TOOL_ID",
    "PHASE4_EVALUATE_TOOL_ID",
    "PHASE4_REPORT_TOOL_ID",
    "PHASE4_STAGE_ORDER",
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
