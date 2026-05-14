"""Report rendering for Phase 2 Codex smoke runs."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import (
    CredentialState,
    Direction,
    PredictionStatus,
    ProviderStatus,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.instruments import Instrument
from nlp_stock_prediction.contracts.provenance import EvidenceReference, ProviderHealth
from nlp_stock_prediction.contracts.report import (
    AuditArtifact,
    AuditManifest,
    DailyReport,
    DataFreshnessSummary,
    InstrumentReportSection,
    PredictionCandidate,
)
from nlp_stock_prediction.instruments.repository import instrument_from_record
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex, ArtifactType
from nlp_stock_prediction.orchestration.phase2_common import (
    Phase2RunPaths,
    phase2_instrument,
    stable_digest,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase2_evidence import source_evidence_from_record
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore


def render_phase2_prediction_report(
    *,
    store: SQLiteStore,
    repo_root: Path,
    run: ResearchRunRecord,
    paths: Phase2RunPaths,
    run_date: date,
    symbol: str,
    tool_run_id: str | None = None,
    record_tool_run: bool = True,
    tool_name: str = "render_prediction_report",
    tool_version: str = "phase2.v1",
    tool_status: str = "ok",
    tool_warnings: tuple[str, ...] = (),
    produced_by: str = "render_prediction_report",
    artifact_schema_version: str = "phase2-report.v1",
    insufficient_evidence_summary: str | None = None,
) -> JsonObject:
    now = utc_now()
    is_phase4_report = artifact_schema_version.startswith("phase4")
    evidence_records = store.list_evidence_for_run(run.run_id)
    evidence_sources = tuple(source_evidence_from_record(record) for record in evidence_records)
    instrument = _primary_instrument(store, symbol=symbol, fallback_generated_at=now)
    candidates = store.list_prediction_candidates_for_run(run.run_id)
    prediction_candidates = tuple(
        _report_candidate(candidate, evidence_sources) for candidate in candidates
    )
    section_refs = tuple(
        EvidenceReference(
            evidence_id=record.evidence_id,
            quote=record.text[:160],
            relevance=0.75,
        )
        for record in evidence_sources[:5]
    )
    if is_phase4_report:
        observed_discussion_summary = (
            "Phase 4 fixture-backed research tools imported evidence and context."
        )
        analysis_summary = (
            "Phase 4 report rendering consumes stored tool evidence, analysis artifacts, "
            "prediction candidates, and prediction-quality evaluations."
        )
        universe_summary = f"Phase 4 fixture-backed research universe for {symbol.upper()}"
        freshness_summary = (
            "Phase 4 used deterministic fixture-backed tool outputs."
            if evidence_sources
            else "Phase 4 has no imported evidence for this run."
        )
    else:
        observed_discussion_summary = (
            "Codex live search evidence was imported through MCP and combined with "
            "dummy structural tools."
        )
        analysis_summary = (
            "Phase 2 validates orchestration and provenance; dummy tools keep the "
            "prediction conservative."
        )
        universe_summary = f"Phase 2 Codex smoke universe for {symbol.upper()}"
        freshness_summary = (
            "Codex smoke used live web search plus deterministic dummy tools."
            if evidence_sources
            else "Codex smoke has no imported live-search evidence for this run."
        )
    instrument_section = InstrumentReportSection(
        instrument_id=instrument.instrument_id,
        symbol=instrument.symbol,
        display_name=instrument.display_name,
        observed_discussion_summary=observed_discussion_summary,
        analysis_summary=analysis_summary,
        prediction_candidate_ids=tuple(
            candidate.candidate_id for candidate in prediction_candidates
        ),
        evidence=section_refs,
        data_quality={"codex_search_evidence_count": len(evidence_sources)},
    )
    audit_artifacts = tuple(
        _audit_artifact_from_record(record, repo_root)
        for record in store.list_artifacts_for_run(run.run_id)
    )
    has_codex_search_evidence = bool(evidence_sources)
    provider_status = ProviderStatus.OK if has_codex_search_evidence else ProviderStatus.EMPTY
    provider_name = "phase4-fixture-tools" if is_phase4_report else "codex-web-search"
    report = DailyReport(
        schema_version="daily-report.v2",
        run_id=run.run_id,
        report_date=run_date,
        generated_at=now,
        timezone="UTC",
        objective=run.objective,
        universe=universe_summary,
        command_args={"run_id": run.run_id, "symbol": symbol.upper()},
        instruments=(instrument,),
        data_freshness=DataFreshnessSummary(
            as_of=now,
            summary=freshness_summary,
            missing_provider_names=() if has_codex_search_evidence else (provider_name,),
        ),
        provider_health=(
            ProviderHealth(
                provider_name=provider_name,
                status=provider_status,
                checked_at=now,
                credential_state=CredentialState.NOT_REQUIRED,
            ),
        ),
        evidence_sources=evidence_sources,
        instrument_sections=(instrument_section,),
        prediction_candidates=prediction_candidates,
        insufficient_evidence_summary=None
        if prediction_candidates
        else insufficient_evidence_summary or "No candidate could be synthesized.",
        audit_manifest=AuditManifest(
            run_id=run.run_id,
            schema_version="audit-manifest.v2",
            created_at=now,
            artifacts=audit_artifacts,
            command_args={"run_id": run.run_id, "symbol": symbol.upper()},
            prediction_trace_ids=tuple(
                candidate.candidate_id for candidate in prediction_candidates
            ),
        ),
    )
    manifest = report.audit_manifest
    if not isinstance(manifest, AuditManifest):
        raise TypeError("Codex smoke reports must include an audit manifest")
    report_tool_run_id = tool_run_id or f"tool-render-report-{run.run_id}"
    if record_tool_run:
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=report_tool_run_id,
                run_id=run.run_id,
                tool_name=tool_name,
                tool_version=tool_version,
                status=tool_status,
                started_at=now,
                completed_at=now,
                inputs={"symbol": symbol},
                warnings=tool_warnings,
            )
        )
    report_index = ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=paths.run_dir,
        created_at=now,
        produced_by=produced_by,
        tool_run_id=report_tool_run_id,
        schema_version=artifact_schema_version,
    )
    markdown_artifact = report_index.write_text(
        artifact_id=f"artifact-report-md-{stable_digest(run.run_id)}",
        artifact_type="markdown_report",
        filename=paths.report_path.name,
        content=render_markdown_report(report),
        metadata={"run_id": run.run_id},
    )
    json_artifact = report_index.write_text(
        artifact_id=f"artifact-report-json-{stable_digest(run.run_id)}",
        artifact_type="json_report",
        filename=paths.json_path.name,
        content=render_json_report(report),
        metadata={"run_id": run.run_id},
    )
    final_manifest = manifest.model_copy(
        update={"artifacts": (*manifest.artifacts, markdown_artifact, json_artifact)}
    )
    ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=paths.audit_dir,
        created_at=now,
        produced_by=produced_by,
        tool_run_id=report_tool_run_id,
        schema_version=artifact_schema_version,
    ).write_json(
        artifact_id=f"artifact-audit-manifest-{stable_digest(run.run_id)}",
        artifact_type="audit_manifest",
        filename=paths.audit_manifest_path.name,
        payload=cast(JsonObject, final_manifest.model_dump(mode="json")),
        metadata={"run_id": run.run_id},
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=run.run_id,
            run_kind=run.run_kind,
            objective=run.objective,
            status="completed",
            started_at=run.started_at,
            completed_at=now,
            metadata=run.metadata,
        )
    )
    return {
        "run_id": run.run_id,
        "markdown_path": paths.report_path.as_posix(),
        "json_path": paths.json_path.as_posix(),
        "audit_manifest_path": paths.audit_manifest_path.as_posix(),
    }


def _primary_instrument(
    store: SQLiteStore,
    *,
    symbol: str,
    fallback_generated_at: datetime,
) -> Instrument:
    instrument_id = f"instrument:codex:{symbol.upper()}"
    record = store.get_instrument(instrument_id)
    if record is not None:
        return instrument_from_record(record)
    discovered = store.find_instruments_by_symbol_or_alias(symbol.upper())
    if discovered:
        return instrument_from_record(discovered[0])
    return phase2_instrument(symbol.upper(), fallback_generated_at)


def _report_candidate(
    candidate: PredictionCandidateRecord,
    evidence_sources: tuple[SourceEvidence, ...],
) -> PredictionCandidate:
    evidence_by_id = {record.evidence_id: record for record in evidence_sources}
    evidence_for_refs = tuple(
        EvidenceReference(
            evidence_id=evidence_id,
            quote=evidence_by_id[evidence_id].text[:180] if evidence_id in evidence_by_id else None,
            relevance=0.76,
        )
        for evidence_id in candidate.evidence_for
    )
    evidence_against_refs = tuple(
        EvidenceReference(
            evidence_id=evidence_id,
            quote=evidence_by_id[evidence_id].text[:180] if evidence_id in evidence_by_id else None,
            relevance=0.76,
        )
        for evidence_id in candidate.evidence_against
    )
    status = (
        PredictionStatus.CONTRADICTED
        if evidence_against_refs
        else PredictionStatus.EVIDENCE_SUPPORTED
        if evidence_for_refs
        else PredictionStatus.INSUFFICIENT_EVIDENCE
    )
    symbol = candidate.metadata.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        symbol = candidate.instrument_id.rsplit(":", 1)[-1]
    baseline = candidate.baseline.get("summary")
    if not isinstance(baseline, str) or not baseline.strip():
        baseline = "No directional edge is assumed without source-backed evidence."
    uncertainty = candidate.uncertainty or "Evidence coverage and freshness may limit confidence."
    return PredictionCandidate(
        candidate_id=candidate.candidate_id,
        instrument_id=candidate.instrument_id,
        symbol=symbol,
        horizon=TimeHorizon.SWING,
        direction=Direction.MIXED,
        status=status,
        thesis=candidate.scenario,
        baseline=baseline,
        confidence=candidate.confidence or 0.0,
        evidence_for=evidence_for_refs,
        evidence_against=evidence_against_refs,
        assumptions=("Source evidence is observed material, not automatically true.",),
        uncertainties=(uncertainty,),
        signal_artifact_ids=candidate.signal_artifacts,
        metadata=candidate.metadata,
    )


def _audit_artifact_from_record(record: ArtifactRecord, repo_root: Path) -> AuditArtifact:
    path = record.path if record.path.is_absolute() else repo_root / record.path
    artifact_type = record.artifact_type
    if artifact_type not in {
        "raw_snapshot",
        "normalized_evidence",
        "extraction_output",
        "analysis_context",
        "prediction_input",
        "markdown_report",
        "json_report",
        "provider_result",
        "market_data",
        "technical_package",
        "ml_forecast",
        "instrument_universe",
        "prediction_evaluation",
        "audit_manifest",
    }:
        raise ValueError(f"unknown artifact type: {artifact_type}")
    return AuditArtifact(
        artifact_id=record.artifact_id,
        artifact_type=cast(ArtifactType, artifact_type),
        path=path.as_posix(),
        created_at=record.created_at or utc_now(),
        produced_by=record.produced_by or "phase2-mcp",
        sha256=record.sha256,
        record_count=record.record_count,
        metadata=record.metadata,
    )


__all__ = ["render_phase2_prediction_report"]
