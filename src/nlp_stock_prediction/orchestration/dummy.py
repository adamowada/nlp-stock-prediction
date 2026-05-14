"""Dummy-tool orchestration path for deterministic report bundle generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    AuditArtifact,
    AuditManifest,
    CredentialState,
    DailyReport,
    DataFreshnessSummary,
    Direction,
    DissentingEvidence,
    EvidenceReference,
    Instrument,
    InstrumentReportSection,
    InstrumentResolution,
    JsonObject,
    MaterialClaimTrace,
    PredictionCandidate,
    PredictionChangeTrigger,
    PredictionStatus,
    PriorOutcomeReview,
    ProviderHealth,
    ReportSourceReference,
    RunConfig,
    SourceEvidence,
    TechnicalAnalysis,
    TimeHorizon,
    UncertaintyDriver,
)
from nlp_stock_prediction.orchestration.context import RunContext
from nlp_stock_prediction.orchestration.dummy_fixtures import (
    dummy_evidence_references,
    dummy_evidence_sources,
    dummy_instrument_universe,
)
from nlp_stock_prediction.orchestration.phase3_universe import phase3_universe_artifact_payload
from nlp_stock_prediction.orchestration.report_data_modes import (
    DUMMY_SMOKE_REPORT_DATA_MODE,
    report_data_mode_metadata,
)
from nlp_stock_prediction.orchestration.runtime import (
    OrchestrationState,
    StagedExecutor,
    ToolRunRecord,
)
from nlp_stock_prediction.orchestration.tools import ToolRegistry, ToolRunResult, ToolSpec
from nlp_stock_prediction.reporting.audit import write_json_artifact
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report

DEFAULT_STAGE_ORDER = ("discover", "collect", "analyze", "score", "assemble")
DUMMY_ORCHESTRATION_DISABLED_MESSAGE = (
    "Dummy orchestration is deterministic; pass an offline RunConfig to run it."
)


@dataclass(frozen=True)
class ReportBundle:
    """Files produced by a deterministic orchestration run.

    TODO: Promote this to a public contract when pipeline and orchestrator share a runtime.
    """

    report_dir: Path
    markdown_path: Path
    json_path: Path
    audit_dir: Path
    audit_manifest_path: Path
    report: DailyReport
    tool_records: tuple[ToolRunRecord, ...]


class DummyInstrumentTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id="dummy.instrument-discovery",
            stage="discover",
            description="Return a deterministic mixed-instrument research universe.",
            output_keys=(
                "instrument_universe",
                "instrument_universe_metadata",
                "instrument_resolutions",
                "instruments",
            ),
        )

    def run(self, context: RunContext, _state: OrchestrationState) -> ToolRunResult:
        universe = dummy_instrument_universe(context)
        instruments = universe.instruments
        payload = phase3_universe_artifact_payload(run_id=context.run_id, universe=universe)
        artifact = context.artifact_writer.write_json(
            artifact_id="instrument-universe",
            artifact_type="instrument_universe",
            filename="instrument-universe.json",
            payload=payload,
            record_count=len(instruments),
            metadata={
                "universe_id": universe.request_id,
                "instrument_ids": list(universe.instrument_ids),
                "warning_count": len(universe.warnings),
            },
        )
        return ToolRunResult(
            tool_id=self.spec.tool_id,
            updates={
                "instrument_universe": universe,
                "instrument_universe_metadata": universe.metadata,
                "instrument_resolutions": universe.resolutions,
                "instruments": instruments,
            },
            artifacts=(artifact,),
            metadata={"warnings": list(universe.warnings)},
        )


class DummyEvidenceTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id="dummy.evidence-collection",
            stage="collect",
            description="Return deterministic evidence and provider health.",
            input_keys=("instruments",),
            output_keys=("evidence_sources", "provider_health"),
        )

    def run(self, context: RunContext, state: OrchestrationState) -> ToolRunResult:
        state.require("instruments", tuple)
        evidence = dummy_evidence_sources(context)
        provider_health = (
            ProviderHealth(
                provider_name="dummy-provider",
                status="ok",
                checked_at=context.generated_at,
                credential_state=CredentialState.NOT_REQUIRED,
            ),
        )
        payload: JsonObject = {
            "schema_version": "orchestration.normalized-evidence.v1",
            "run_id": context.run_id,
            "records": [record.model_dump(mode="json") for record in evidence],
        }
        artifact = context.artifact_writer.write_json(
            artifact_id="normalized-evidence",
            artifact_type="normalized_evidence",
            filename="normalized-evidence.json",
            payload=payload,
            record_count=len(evidence),
        )
        return ToolRunResult(
            tool_id=self.spec.tool_id,
            updates={
                "evidence_sources": evidence,
                "provider_health": provider_health,
            },
            artifacts=(artifact,),
        )


class DummyAnalysisTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id="dummy.analysis",
            stage="analyze",
            description="Create deterministic instrument sections from evidence.",
            input_keys=("instruments", "evidence_sources"),
            output_keys=("instrument_sections",),
        )

    def run(self, context: RunContext, state: OrchestrationState) -> ToolRunResult:
        instruments = cast(tuple[Instrument, ...], state.require("instruments", tuple))
        evidence = cast(tuple[SourceEvidence, ...], state.require("evidence_sources", tuple))
        sections = _instrument_sections(instruments, evidence)
        payload: JsonObject = {
            "schema_version": "orchestration.analysis-contexts.v1",
            "run_id": context.run_id,
            "records": [section.model_dump(mode="json") for section in sections],
        }
        artifact = context.artifact_writer.write_json(
            artifact_id="analysis-contexts",
            artifact_type="analysis_context",
            filename="analysis-contexts.json",
            payload=payload,
            record_count=len(sections),
        )
        return ToolRunResult(
            tool_id=self.spec.tool_id,
            updates={"instrument_sections": sections},
            artifacts=(artifact,),
        )


class DummyPredictionTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id="dummy.prediction-scoring",
            stage="score",
            description="Create a deterministic evidence-backed prediction scenario.",
            input_keys=("instrument_sections", "evidence_sources"),
            output_keys=("prediction_candidates", "data_freshness"),
        )

    def run(self, context: RunContext, state: OrchestrationState) -> ToolRunResult:
        state.require("instrument_sections", tuple)
        state.require("evidence_sources", tuple)
        evidence_refs = dummy_evidence_references()
        predictions = (
            PredictionCandidate(
                candidate_id="prediction-tsla-dummy-volatility",
                instrument_id="instrument:equity:us:tsla",
                symbol="TSLA",
                horizon=TimeHorizon.SWING,
                direction=Direction.MIXED,
                status=PredictionStatus.EVIDENCE_SUPPORTED,
                thesis=(
                    "TSLA may remain headline-sensitive over the swing horizon while broad "
                    "market context can offset single-name evidence."
                ),
                baseline="The baseline is no directional edge because the dummy evidence is mixed.",
                confidence=0.37,
                evidence_for=(evidence_refs["dummy-news-tsla-001"],),
                evidence_against=(evidence_refs["dummy-market-spy-001"],),
                dissenting_evidence=(
                    DissentingEvidence(
                        summary="Broad-market dummy evidence limits the single-name scenario.",
                        evidence=(evidence_refs["dummy-market-spy-001"],),
                    ),
                ),
                assumptions=("Dummy tools emit deterministic fixture-like research artifacts.",),
                uncertainties=(
                    "Dummy evidence is not live provider evidence.",
                    "The scenario is research context and not a buy or sell instruction.",
                ),
                uncertainty_drivers=(
                    UncertaintyDriver(
                        driver_id="uncertainty-dummy-live-coverage",
                        summary="Dummy evidence cannot establish live provider freshness.",
                        severity="high",
                        evidence=(evidence_refs["dummy-news-tsla-001"],),
                    ),
                ),
                change_triggers=(
                    PredictionChangeTrigger(
                        trigger_id="change-dummy-live-sources",
                        summary="Live source evidence would be needed to change dummy support.",
                        trigger_type="provider_refresh",
                        evidence=(
                            evidence_refs["dummy-news-tsla-001"],
                            evidence_refs["dummy-market-spy-001"],
                        ),
                        rationale="The dummy path is structural validation only.",
                    ),
                ),
                prior_outcome_review_ids=("prior-outcome-dummy-unavailable",),
            ),
        )
        data_freshness = DataFreshnessSummary(
            as_of=context.generated_at,
            summary="Dummy tool evidence is deterministic and current to the run date.",
        )
        payload: JsonObject = {
            "schema_version": "orchestration.prediction-inputs.v1",
            "run_id": context.run_id,
            "records": [candidate.model_dump(mode="json") for candidate in predictions],
        }
        artifact = context.artifact_writer.write_json(
            artifact_id="prediction-inputs",
            artifact_type="prediction_input",
            filename="prediction-inputs.json",
            payload=payload,
            record_count=len(predictions),
        )
        return ToolRunResult(
            tool_id=self.spec.tool_id,
            updates={
                "prediction_candidates": predictions,
                "data_freshness": data_freshness,
            },
            artifacts=(artifact,),
        )


class DummyReportAssemblyTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            tool_id="dummy.report-assembly",
            stage="assemble",
            description="Assemble public report contracts from prior tool outputs.",
            input_keys=(
                "instruments",
                "instrument_resolutions",
                "instrument_universe_metadata",
                "evidence_sources",
                "provider_health",
                "instrument_sections",
                "prediction_candidates",
                "data_freshness",
            ),
            output_keys=("daily_report",),
        )

    def run(self, context: RunContext, state: OrchestrationState) -> ToolRunResult:
        instruments = cast(tuple[Instrument, ...], state.require("instruments", tuple))
        resolutions = cast(
            tuple[InstrumentResolution, ...],
            state.require("instrument_resolutions", tuple),
        )
        universe_metadata = cast(
            JsonObject,
            state.require("instrument_universe_metadata", dict),
        )
        evidence = cast(tuple[SourceEvidence, ...], state.require("evidence_sources", tuple))
        provider_health = cast(tuple[ProviderHealth, ...], state.require("provider_health", tuple))
        sections = cast(
            tuple[InstrumentReportSection, ...],
            state.require("instrument_sections", tuple),
        )
        predictions = cast(
            tuple[PredictionCandidate, ...],
            state.require("prediction_candidates", tuple),
        )
        data_freshness = state.require("data_freshness", DataFreshnessSummary)
        manifest = AuditManifest(
            run_id=context.run_id,
            schema_version="audit-manifest.v2",
            created_at=context.generated_at,
            artifacts=tuple(state.artifacts),
            command_args=context.command_args,
            prediction_trace_ids=tuple(candidate.candidate_id for candidate in predictions),
        )
        first_prediction = predictions[0]
        universe_summary = universe_metadata.get("summary")
        report = DailyReport(
            schema_version="daily-report.v2",
            run_id=context.run_id,
            report_date=context.run_date,
            generated_at=context.generated_at,
            timezone=context.timezone,
            objective="Generate an evidence-backed prediction research report.",
            universe=(
                universe_summary
                if isinstance(universe_summary, str)
                else "Deterministic dummy-tool Phase 3 fixture universe."
            ),
            command_args=context.command_args,
            instruments=instruments,
            data_freshness=data_freshness,
            provider_health=provider_health,
            evidence_sources=evidence,
            instrument_resolutions=resolutions,
            instrument_sections=sections,
            prediction_candidates=predictions,
            source_references=(
                ReportSourceReference(
                    reference_id="source-ref-dummy-support",
                    label="Dummy supporting evidence",
                    reference_type="source_evidence",
                    evidence_ids=("dummy-news-tsla-001",),
                    candidate_ids=(first_prediction.candidate_id,),
                ),
                ReportSourceReference(
                    reference_id="source-ref-dummy-dissent",
                    label="Dummy dissenting evidence",
                    reference_type="source_evidence",
                    evidence_ids=("dummy-market-spy-001",),
                    candidate_ids=(first_prediction.candidate_id,),
                ),
            ),
            material_claim_traces=(
                MaterialClaimTrace(
                    claim_id="claim-dummy-headline-sensitivity",
                    claim=first_prediction.thesis,
                    claim_type="analysis",
                    evidence=first_prediction.evidence_for,
                    source_reference_ids=("source-ref-dummy-support",),
                    candidate_ids=(first_prediction.candidate_id,),
                ),
            ),
            prior_outcome_reviews=(
                PriorOutcomeReview(
                    review_id="prior-outcome-dummy-unavailable",
                    status="not_available",
                    summary="No prior outcome review exists for the dummy report path.",
                    limitations=("Dummy reports do not track prior live outcomes.",),
                ),
            ),
            audit_manifest=manifest,
        )
        return ToolRunResult(tool_id=self.spec.tool_id, updates={"daily_report": report})


def build_dummy_tool_registry() -> ToolRegistry:
    """Return the complete deterministic dummy-tool registry."""

    return ToolRegistry(
        (
            DummyInstrumentTool(),
            DummyEvidenceTool(),
            DummyAnalysisTool(),
            DummyPredictionTool(),
            DummyReportAssemblyTool(),
        )
    )


def generate_dummy_report_bundle(config: RunConfig) -> ReportBundle:
    """Run deterministic dummy tools and write a complete report bundle."""

    if not (config.offline or config.source_mode == "offline"):
        raise ValueError(DUMMY_ORCHESTRATION_DISABLED_MESSAGE)
    if config.live_providers:
        raise ValueError("live providers are not wired into the dummy orchestrator")

    context = RunContext.from_config(config)
    context.report_dir.mkdir(parents=True, exist_ok=True)
    context.audit_dir.mkdir(parents=True, exist_ok=True)

    result = StagedExecutor(
        registry=build_dummy_tool_registry(),
        stage_order=DEFAULT_STAGE_ORDER,
    ).run(context)
    report = result.state.require("daily_report", DailyReport)

    markdown_path = context.report_dir / "report.md"
    json_path = context.report_dir / "report.json"
    audit_manifest_path = context.audit_dir / "audit-manifest.json"
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    json_path.write_text(render_json_report(report), encoding="utf-8")

    manifest = report.audit_manifest
    if not isinstance(manifest, AuditManifest):
        raise TypeError("dummy orchestration reports must include an AuditManifest")
    final_manifest = manifest.model_copy(
        update={
            "artifacts": (
                *manifest.artifacts,
                _file_audit_artifact(
                    artifact_id="report-markdown",
                    artifact_type="markdown_report",
                    path=markdown_path,
                    created_at=context.generated_at,
                    produced_by="dummy.report-bundle",
                    metadata=report_data_mode_metadata(DUMMY_SMOKE_REPORT_DATA_MODE),
                ),
                _file_audit_artifact(
                    artifact_id="report-json",
                    artifact_type="json_report",
                    path=json_path,
                    created_at=context.generated_at,
                    produced_by="dummy.report-bundle",
                    metadata=report_data_mode_metadata(DUMMY_SMOKE_REPORT_DATA_MODE),
                ),
            )
        }
    )
    write_json_artifact(
        audit_manifest_path,
        cast(JsonObject, final_manifest.model_dump(mode="json")),
    )

    return ReportBundle(
        report_dir=context.report_dir,
        markdown_path=markdown_path,
        json_path=json_path,
        audit_dir=context.audit_dir,
        audit_manifest_path=audit_manifest_path,
        report=report,
        tool_records=result.tool_records,
    )


def _file_audit_artifact(
    *,
    artifact_id: str,
    artifact_type: Literal["markdown_report", "json_report"],
    path: Path,
    created_at: datetime,
    produced_by: str,
    metadata: JsonObject | None = None,
) -> AuditArtifact:
    return AuditArtifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        path=path.as_posix(),
        created_at=created_at,
        produced_by=produced_by,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata={} if metadata is None else metadata,
    )


def _instrument_sections(
    instruments: tuple[Instrument, ...],
    evidence: tuple[SourceEvidence, ...],
) -> tuple[InstrumentReportSection, ...]:
    evidence_refs = dummy_evidence_references()
    evidence_by_ticker = {record.ticker: record for record in evidence if record.ticker}
    sections: list[InstrumentReportSection] = []
    for instrument in instruments:
        if instrument.symbol == "TSLA":
            sections.append(
                InstrumentReportSection(
                    instrument_id=instrument.instrument_id,
                    symbol=instrument.symbol,
                    display_name=instrument.display_name,
                    observed_discussion_summary=(
                        "Dummy evidence notes delivery-cycle uncertainty, margin pressure, "
                        "and headline sensitivity."
                    ),
                    social_news_summary="Dummy evidence is intentionally mixed.",
                    technical_analysis=TechnicalAnalysis(
                        ticker="TSLA",
                        summary=(
                            "Dummy technical context is mixed baseline context, not a "
                            "standalone prediction signal."
                        ),
                        signal=AnalysisSignal.MIXED,
                        confidence=0.35,
                        evidence=(evidence_refs["dummy-news-tsla-001"],),
                        assumptions=("Dummy market data is deterministic.",),
                    ),
                    analysis_summary=(
                        "Evidence supports a monitored volatility scenario with explicit "
                        "uncertainty."
                    ),
                    prediction_candidate_ids=("prediction-tsla-dummy-volatility",),
                    evidence=(evidence_refs["dummy-news-tsla-001"],),
                    data_quality={"mode": "dummy", "evidence_count": 1},
                )
            )
            continue

        evidence_record = evidence_by_ticker.get(instrument.symbol)
        reference = (
            EvidenceReference(
                evidence_id=evidence_record.evidence_id,
                quote=evidence_record.text[:140],
                relevance=0.72,
            )
            if evidence_record is not None
            else evidence_refs["dummy-crypto-btc-001"]
        )
        sections.append(
            InstrumentReportSection(
                instrument_id=instrument.instrument_id,
                symbol=instrument.symbol,
                display_name=instrument.display_name,
                observed_discussion_summary=(
                    f"Dummy evidence records {instrument.symbol} as context for the universe."
                ),
                analysis_summary="No evidence-backed prediction candidate is produced.",
                evidence=(reference,),
                data_quality={"mode": "dummy", "evidence_count": 1},
            )
        )
    return tuple(sections)


__all__ = [
    "DEFAULT_STAGE_ORDER",
    "DUMMY_ORCHESTRATION_DISABLED_MESSAGE",
    "ReportBundle",
    "build_dummy_tool_registry",
    "generate_dummy_report_bundle",
]
