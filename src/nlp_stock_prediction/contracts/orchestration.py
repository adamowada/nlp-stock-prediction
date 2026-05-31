"""Codex orchestration public contracts.

These contracts describe the orchestrator's public run shape only. They do not execute tools,
perform live provider calls, or decide what to research.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    Confidence,
    ContractModel,
    JsonObject,
    NonEmptyStr,
    Score,
)
from nlp_stock_prediction.contracts.enums import PredictionStatus, TimeHorizon
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.instruments import InstrumentSymbol
from nlp_stock_prediction.contracts.provenance import DataReference


class ResearchObjective(ContractModel):
    """The parsed research objective Codex should satisfy."""

    objective_id: NonEmptyStr
    run_date: date
    prompt: NonEmptyStr
    requested_symbols: tuple[InstrumentSymbol, ...] = Field(default_factory=tuple)
    time_horizon: TimeHorizon = TimeHorizon.UNKNOWN
    offline: bool = True
    max_tool_calls: int = Field(default=8, ge=0)
    require_citations: bool = True
    metadata: JsonObject = Field(default_factory=dict)


class ResearchToolSpec(ContractModel):
    """Static description of one research tool available to the orchestrator."""

    tool_name: NonEmptyStr
    tool_version: NonEmptyStr
    description: NonEmptyStr
    input_schema: JsonObject = Field(default_factory=dict)
    output_contract: str | None = None
    artifact_kinds: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    offline_capable: bool = True
    deterministic: bool = True
    live_capable: bool = False
    requires_network: bool = False
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_live_capability(self) -> ResearchToolSpec:
        if self.requires_network and not self.live_capable:
            raise ValueError("network-dependent tools must be marked live_capable")
        return self


class ToolInvocation(ContractModel):
    """A planned or requested tool call."""

    invocation_id: NonEmptyStr
    objective_id: NonEmptyStr
    tool_name: NonEmptyStr
    tool_version: NonEmptyStr
    requested_at: AwareDatetime
    arguments: JsonObject = Field(default_factory=dict)
    mode: Literal["offline", "live"] = "offline"
    depends_on_invocation_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dependencies(self) -> ToolInvocation:
        if self.invocation_id in self.depends_on_invocation_ids:
            raise ValueError("tool invocations cannot depend on themselves")
        if len(set(self.depends_on_invocation_ids)) != len(self.depends_on_invocation_ids):
            raise ValueError("tool invocation dependencies must be unique")
        return self


class ToolExecutionResult(ContractModel):
    """Result envelope for an executed or skipped tool invocation."""

    result_id: NonEmptyStr
    invocation: ToolInvocation
    status: Literal["succeeded", "partial", "empty", "failed", "skipped"]
    started_at: AwareDatetime
    completed_at: AwareDatetime
    artifact_refs: tuple[DataReference, ...] = Field(default_factory=tuple)
    evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    warnings: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    error_message: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result_shape(self) -> ToolExecutionResult:
        if self.completed_at < self.started_at:
            raise ValueError("tool completed_at must be greater than or equal to started_at")
        if self.status in {"succeeded", "partial"} and not (
            self.artifact_refs or self.evidence_ids
        ):
            raise ValueError("successful tool results require artifact_refs or evidence_ids")
        if self.status == "partial" and not self.warnings:
            raise ValueError("partial tool results require warnings")
        if self.status == "failed" and not self.error_message:
            raise ValueError("failed tool results require error_message")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("tool result evidence_ids must be unique")
        return self


class CodexEvidenceImport(ContractModel):
    """Evidence imported from a Codex search, browse, or artifact-inspection step."""

    import_id: NonEmptyStr
    objective_id: NonEmptyStr
    imported_at: AwareDatetime
    evidence: tuple[SourceEvidence, ...]
    artifact_refs: tuple[DataReference, ...]
    originating_invocation_id: str | None = None
    notes: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_import_traceability(self) -> CodexEvidenceImport:
        if not self.evidence:
            raise ValueError("Codex evidence imports require evidence")
        if not self.artifact_refs:
            raise ValueError("Codex evidence imports require artifact_refs")
        evidence_ids = tuple(record.evidence_id for record in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Codex evidence import evidence ids must be unique")
        return self


class OrchestratorRunSummary(ContractModel):
    """Summary of a completed Codex orchestration run."""

    run_id: NonEmptyStr
    objective: ResearchObjective
    started_at: AwareDatetime
    completed_at: AwareDatetime
    status: Literal["succeeded", "partial", "failed", "skipped"]
    selected_tools: tuple[ResearchToolSpec, ...] = Field(default_factory=tuple)
    invocations: tuple[ToolInvocation, ...] = Field(default_factory=tuple)
    tool_results: tuple[ToolExecutionResult, ...] = Field(default_factory=tuple)
    evidence_imports: tuple[CodexEvidenceImport, ...] = Field(default_factory=tuple)
    report_artifacts: tuple[DataReference, ...] = Field(default_factory=tuple)
    prediction_candidate_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    warnings: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    error_message: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_run_consistency(self) -> OrchestratorRunSummary:
        if self.completed_at < self.started_at:
            raise ValueError("run completed_at must be greater than or equal to started_at")

        tool_keys = tuple((tool.tool_name, tool.tool_version) for tool in self.selected_tools)
        if len(set(tool_keys)) != len(tool_keys):
            raise ValueError("selected tool specs must be unique by name and version")

        invocation_ids = tuple(invocation.invocation_id for invocation in self.invocations)
        if len(set(invocation_ids)) != len(invocation_ids):
            raise ValueError("orchestrator invocation ids must be unique")

        invocation_by_id = {invocation.invocation_id: invocation for invocation in self.invocations}
        selected_tool_keys = set(tool_keys)
        for invocation in self.invocations:
            if invocation.objective_id != self.objective.objective_id:
                raise ValueError("tool invocations must reference the run objective")
            if self.objective.offline and invocation.mode == "live":
                raise ValueError("offline objectives cannot use live tool invocations")
            if (
                selected_tool_keys
                and (
                    invocation.tool_name,
                    invocation.tool_version,
                )
                not in selected_tool_keys
            ):
                raise ValueError("tool invocations must reference selected tool specs")
            missing_dependencies = set(invocation.depends_on_invocation_ids).difference(
                invocation_by_id
            )
            if missing_dependencies:
                raise ValueError("tool invocation dependencies must reference run invocations")

        result_ids = tuple(result.result_id for result in self.tool_results)
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("tool result ids must be unique")
        result_invocation_ids = tuple(
            result.invocation.invocation_id for result in self.tool_results
        )
        if len(set(result_invocation_ids)) != len(result_invocation_ids):
            raise ValueError("tool results must be unique per invocation")
        unknown_result_invocations = set(result_invocation_ids).difference(invocation_by_id)
        if unknown_result_invocations:
            raise ValueError("tool results must reference run invocations")
        for result in self.tool_results:
            invocation = invocation_by_id[result.invocation.invocation_id]
            if result.invocation != invocation:
                raise ValueError("tool results must embed the matching run invocation")

        import_ids = tuple(evidence_import.import_id for evidence_import in self.evidence_imports)
        if len(set(import_ids)) != len(import_ids):
            raise ValueError("Codex evidence import ids must be unique")
        for evidence_import in self.evidence_imports:
            if evidence_import.objective_id != self.objective.objective_id:
                raise ValueError("Codex evidence imports must reference the run objective")
            if (
                evidence_import.originating_invocation_id is not None
                and evidence_import.originating_invocation_id not in invocation_by_id
            ):
                raise ValueError("Codex evidence imports must reference run invocations")

        if self.prediction_candidate_ids and not (self.evidence_imports and self.report_artifacts):
            raise ValueError("prediction candidates require evidence imports and report artifacts")
        if len(set(self.prediction_candidate_ids)) != len(self.prediction_candidate_ids):
            raise ValueError("prediction candidate ids must be unique")
        if self.status == "failed" and not self.error_message:
            raise ValueError("failed orchestrator runs require error_message")
        has_incomplete_result = any(
            result.status in {"partial", "failed"} for result in self.tool_results
        )
        if self.status == "partial" and not (self.warnings or has_incomplete_result):
            raise ValueError("partial orchestrator runs require warnings or partial tool results")
        if self.status == "succeeded" and any(
            result.status in {"partial", "failed"} for result in self.tool_results
        ):
            raise ValueError("succeeded orchestrator runs cannot include partial or failed results")
        return self


class ResearchViabilityTarget(ContractModel):
    """One researched instrument ranked by follow-up research viability."""

    symbol: InstrumentSymbol
    viability_score: Score
    research_status: Literal["ranked", "failed"] = "ranked"
    rank: int | None = Field(default=None, ge=1)
    report_status: NonEmptyStr | None = None
    candidate_id: str | None = None
    candidate_status: PredictionStatus | None = None
    candidate_confidence: Confidence | None = None
    evaluation_score: Score | None = None
    evidence_for_count: int = Field(default=0, ge=0)
    evidence_against_count: int = Field(default=0, ge=0)
    signal_artifact_count: int = Field(default=0, ge=0)
    markdown_path: str | None = None
    json_path: str | None = None
    database_path: str | None = None
    rationale: tuple[NonEmptyStr, ...]
    warnings: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    error_message: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_viability_target(self) -> ResearchViabilityTarget:
        if self.research_status == "ranked":
            if self.rank is None:
                raise ValueError("ranked research viability targets require rank")
            if self.report_status is None:
                raise ValueError("ranked research viability targets require report_status")
            if self.markdown_path is None or self.json_path is None:
                raise ValueError("ranked research viability targets require report paths")
        if self.research_status == "failed" and not self.error_message:
            raise ValueError("failed research viability targets require error_message")
        return self


class ResearchViabilityRankingReport(ContractModel):
    """Batch artifact that ranks researched instruments by evidence-backed viability."""

    schema_version: Literal["research-batch-ranking.v1"] = "research-batch-ranking.v1"
    generated_at: AwareDatetime
    run_date: date
    mode: Literal["offline_fixture", "live"]
    ranked_targets: tuple[ResearchViabilityTarget, ...] = Field(default_factory=tuple)
    failed_targets: tuple[ResearchViabilityTarget, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ranking_report(self) -> ResearchViabilityRankingReport:
        ranked_symbols = tuple(target.symbol for target in self.ranked_targets)
        failed_symbols = tuple(target.symbol for target in self.failed_targets)
        all_symbols = (*ranked_symbols, *failed_symbols)
        if len(set(all_symbols)) != len(all_symbols):
            raise ValueError("batch research ranking symbols must be unique")
        ranks = tuple(target.rank for target in self.ranked_targets)
        expected = tuple(range(1, len(self.ranked_targets) + 1))
        if ranks != expected:
            raise ValueError("batch research ranks must be consecutive starting at 1")
        if any(target.research_status != "ranked" for target in self.ranked_targets):
            raise ValueError("ranked_targets entries must have research_status='ranked'")
        if any(target.research_status != "failed" for target in self.failed_targets):
            raise ValueError("failed_targets entries must have research_status='failed'")
        return self


class WsbTrendingStock(ContractModel):
    """One WSB-mentioned symbol selected for downstream research."""

    symbol: InstrumentSymbol
    rank: int = Field(ge=1)
    mention_count: int = Field(ge=1)
    cashtag_count: int = Field(default=0, ge=0)
    source_record_count: int = Field(default=0, ge=0)
    source_urls: tuple[NonEmptyStr, ...]
    snippets: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trending_stock(self) -> WsbTrendingStock:
        if not self.source_urls:
            raise ValueError("WSB trending stocks require source URLs")
        if self.cashtag_count > self.mention_count:
            raise ValueError("cashtag_count cannot exceed mention_count")
        return self


class WsbTrendingDiscoveryReport(ContractModel):
    """Public Reddit WSB mention-count discovery artifact."""

    schema_version: Literal["wsb-trending-discovery.v1"] = "wsb-trending-discovery.v1"
    generated_at: AwareDatetime
    run_date: date
    source_url: NonEmptyStr
    provider_name: NonEmptyStr
    mode: Literal["offline_fixture", "live"]
    limit: int = Field(ge=1)
    max_discussion_pages: int = Field(ge=0)
    trending_stocks: tuple[WsbTrendingStock, ...] = Field(default_factory=tuple)
    raw_snapshot_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    source_urls: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    warnings: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_wsb_discovery(self) -> WsbTrendingDiscoveryReport:
        symbols = tuple(stock.symbol for stock in self.trending_stocks)
        if len(set(symbols)) != len(symbols):
            raise ValueError("WSB trending discovery symbols must be unique")
        ranks = tuple(stock.rank for stock in self.trending_stocks)
        expected = tuple(range(1, len(self.trending_stocks) + 1))
        if ranks != expected:
            raise ValueError("WSB trending discovery ranks must be consecutive")
        if len(self.trending_stocks) > self.limit:
            raise ValueError("WSB trending discovery cannot exceed its limit")
        if not self.source_urls:
            raise ValueError("WSB trending discovery requires source URLs")
        return self


__all__ = [
    "CodexEvidenceImport",
    "OrchestratorRunSummary",
    "ResearchObjective",
    "ResearchToolSpec",
    "ResearchViabilityRankingReport",
    "ResearchViabilityTarget",
    "ToolExecutionResult",
    "ToolInvocation",
    "WsbTrendingDiscoveryReport",
    "WsbTrendingStock",
]
