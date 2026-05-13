"""Dummy-tool orchestration path for deterministic report bundle generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    AssetClass,
    AuditManifest,
    CredentialState,
    DailyReport,
    DataFreshnessSummary,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    Instrument,
    InstrumentReportSection,
    JsonObject,
    PredictionCandidate,
    PredictionStatus,
    ProviderHealth,
    RetrievalMethod,
    RunConfig,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TechnicalAnalysis,
    TimeHorizon,
    TradabilityStatus,
)
from nlp_stock_prediction.contracts.instruments import (
    InstrumentDataAvailability,
    ProviderInstrumentId,
    TradabilityEvidence,
)
from nlp_stock_prediction.orchestration.context import RunContext
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
            output_keys=("instruments",),
        )

    def run(self, context: RunContext, _state: OrchestrationState) -> ToolRunResult:
        instruments = _instruments(context)
        payload: JsonObject = {
            "schema_version": "orchestration.dummy-instruments.v1",
            "run_id": context.run_id,
            "records": [instrument.model_dump(mode="json") for instrument in instruments],
        }
        artifact = context.artifact_writer.write_json(
            artifact_id="dummy-instruments",
            artifact_type="provider_result",
            filename="dummy-instruments.json",
            payload=payload,
            record_count=len(instruments),
        )
        return ToolRunResult(
            tool_id=self.spec.tool_id,
            updates={"instruments": instruments},
            artifacts=(artifact,),
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
        evidence = _evidence_sources(context)
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
        evidence_refs = _evidence_references()
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
                assumptions=("Dummy tools emit deterministic fixture-like research artifacts.",),
                uncertainties=(
                    "Dummy evidence is not live provider evidence.",
                    "The scenario is research context and not a buy or sell instruction.",
                ),
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
        report = DailyReport(
            schema_version="daily-report.v2",
            run_id=context.run_id,
            report_date=context.run_date,
            generated_at=context.generated_at,
            timezone=context.timezone,
            objective="Generate an evidence-backed prediction research report.",
            universe="Deterministic dummy-tool universe: one stock, one ETF, one crypto pair.",
            command_args=context.command_args,
            instruments=instruments,
            data_freshness=data_freshness,
            provider_health=provider_health,
            evidence_sources=evidence,
            instrument_sections=sections,
            prediction_candidates=predictions,
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
    write_json_artifact(
        audit_manifest_path,
        cast(JsonObject, manifest.model_dump(mode="json")),
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


def _instruments(context: RunContext) -> tuple[Instrument, ...]:
    return (
        _instrument(
            context=context,
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            display_name="Tesla Inc.",
            asset_class=AssetClass.STOCK,
            provider_identifier="TSLA",
        ),
        _instrument(
            context=context,
            instrument_id="instrument:etf:us:spy",
            symbol="SPY",
            display_name="SPDR S&P 500 ETF Trust",
            asset_class=AssetClass.ETF,
            provider_identifier="SPY",
        ),
        _instrument(
            context=context,
            instrument_id="instrument:crypto:btc-usd",
            symbol="BTC/USD",
            display_name="Bitcoin versus U.S. dollar",
            asset_class=AssetClass.CRYPTO,
            provider_identifier="BTC/USD",
        ),
    )


def _instrument(
    *,
    context: RunContext,
    instrument_id: str,
    symbol: str,
    display_name: str,
    asset_class: AssetClass,
    provider_identifier: str,
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        display_name=display_name,
        asset_class=asset_class,
        provider_ids=(
            ProviderInstrumentId(
                provider="dummy-provider",
                identifier=provider_identifier,
                namespace="dummy-symbol",
            ),
        ),
        tradability_evidence=(
            TradabilityEvidence(
                provider="dummy-provider",
                status=TradabilityStatus.UNKNOWN,
                retrieved_at=context.generated_at,
                raw_identifier=f"{provider_identifier}:dummy-tradability",
                notes="Dummy records availability context only.",
            ),
        ),
        data_availability=(
            InstrumentDataAvailability(
                provider="dummy-provider",
                data_type="dummy-research",
                status=TradabilityStatus.AVAILABLE,
                checked_at=context.generated_at,
                provider_identifier=provider_identifier,
            ),
        ),
        metadata={"dummy": True},
    )


def _evidence_sources(context: RunContext) -> tuple[SourceEvidence, ...]:
    return (
        _evidence(
            context=context,
            evidence_id="dummy-news-tsla-001",
            source_kind=SourceKind.NEWS_ARTICLE,
            ticker="TSLA",
            title="Dummy TSLA delivery and margin context",
            text=(
                "Dummy source describes TSLA delivery uncertainty, margin pressure, and "
                "headline sensitivity without asserting a trade action."
            ),
            source_url="https://example.com/dummy/tsla-delivery-context",
        ),
        _evidence(
            context=context,
            evidence_id="dummy-market-spy-001",
            source_kind=SourceKind.MARKET_DATA,
            ticker="SPY",
            title="Dummy SPY regime context",
            text=(
                "Dummy source describes broad-index regime risk that could outweigh "
                "single-name narratives."
            ),
            source_url="https://example.com/dummy/spy-regime-context",
        ),
        _evidence(
            context=context,
            evidence_id="dummy-crypto-btc-001",
            source_kind=SourceKind.MARKET_DATA,
            ticker=None,
            title="Dummy BTC/USD data availability",
            text=(
                "Dummy source records BTC/USD as a researchable crypto pair when provider "
                "data is available."
            ),
            source_url="https://example.com/dummy/btc-usd-availability",
        ),
    )


def _evidence(
    *,
    context: RunContext,
    evidence_id: str,
    source_kind: SourceKind,
    ticker: str | None,
    title: str,
    text: str,
    source_url: str,
) -> SourceEvidence:
    raw_identifier = evidence_id
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=source_kind,
        ticker=ticker,
        title=title,
        text=text,
        created_at=context.generated_at,
        permalink=source_url,
        matched_tickers=((ticker,) if ticker else ()),
        provenance=SourceProvenance(
            provider_name="dummy-provider",
            source_kind=source_kind,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=context.generated_at,
            observed_at=context.generated_at,
            source_url=source_url,
            permalink=source_url,
            raw_identifier=raw_identifier,
            raw_snapshot_id=f"raw-{raw_identifier}",
            freshness_status=FreshnessStatus.FRESH,
            provider_metadata={"dummy": True},
        ),
        metadata={"dummy": True},
    )


def _instrument_sections(
    instruments: tuple[Instrument, ...],
    evidence: tuple[SourceEvidence, ...],
) -> tuple[InstrumentReportSection, ...]:
    evidence_refs = _evidence_references()
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


def _evidence_references() -> dict[str, EvidenceReference]:
    quotes = {
        "dummy-news-tsla-001": (
            "Dummy source describes TSLA delivery uncertainty, margin pressure, and headline "
            "sensitivity without asserting a trade action."
        ),
        "dummy-market-spy-001": (
            "Dummy source describes broad-index regime risk that could outweigh single-name "
            "narratives."
        ),
        "dummy-crypto-btc-001": (
            "Dummy source records BTC/USD as a researchable crypto pair when provider data is "
            "available."
        ),
    }
    return {
        evidence_id: EvidenceReference(evidence_id=evidence_id, quote=quote, relevance=0.8)
        for evidence_id, quote in quotes.items()
    }


__all__ = [
    "DEFAULT_STAGE_ORDER",
    "DUMMY_ORCHESTRATION_DISABLED_MESSAGE",
    "ReportBundle",
    "build_dummy_tool_registry",
    "generate_dummy_report_bundle",
]
