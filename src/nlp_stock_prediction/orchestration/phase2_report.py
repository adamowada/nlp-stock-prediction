"""Report rendering for Phase 2 Codex smoke runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts import (
    AuditArtifact,
    AuditManifest,
    CredentialState,
    DailyReport,
    DataFreshnessSummary,
    Direction,
    EvidenceReference,
    InstrumentReportSection,
    JsonObject,
    PredictionCandidate,
    PredictionStatus,
    ProviderHealth,
    SourceEvidence,
    TimeHorizon,
)
from nlp_stock_prediction.orchestration.artifacts import ArtifactType
from nlp_stock_prediction.orchestration.phase2_common import (
    Phase2RunPaths,
    file_sha256,
    phase2_instrument,
    stable_digest,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase2_evidence import source_evidence_from_record
from nlp_stock_prediction.reporting.audit import write_json_artifact
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    SQLiteStore,
    ToolRunRecord,
)


def render_phase2_prediction_report(
    *,
    store: SQLiteStore,
    repo_root: Path,
    run: ResearchRunRecord,
    paths: Phase2RunPaths,
    run_date: date,
    symbol: str,
) -> JsonObject:
    now = utc_now()
    evidence_records = store.list_evidence_for_run(run.run_id)
    evidence_sources = tuple(source_evidence_from_record(record) for record in evidence_records)
    instrument = phase2_instrument(symbol.upper(), now)
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
    instrument_section = InstrumentReportSection(
        instrument_id=instrument.instrument_id,
        symbol=instrument.symbol,
        display_name=instrument.display_name,
        observed_discussion_summary=(
            "Codex live search evidence was imported through MCP and combined with "
            "dummy structural tools."
        ),
        analysis_summary=(
            "Phase 2 validates orchestration and provenance; dummy tools keep the "
            "prediction conservative."
        ),
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
    report = DailyReport(
        schema_version="daily-report.v2",
        run_id=run.run_id,
        report_date=run_date,
        generated_at=now,
        timezone="UTC",
        objective=run.objective,
        universe=f"Phase 2 Codex smoke universe for {symbol.upper()}",
        command_args={"run_id": run.run_id, "symbol": symbol.upper()},
        instruments=(instrument,),
        data_freshness=DataFreshnessSummary(
            as_of=now,
            summary="Codex smoke used live web search plus deterministic dummy tools.",
        ),
        provider_health=(
            ProviderHealth(
                provider_name="codex-web-search",
                status="ok",
                checked_at=now,
                credential_state=CredentialState.NOT_REQUIRED,
            ),
        ),
        evidence_sources=evidence_sources,
        instrument_sections=(instrument_section,),
        prediction_candidates=prediction_candidates,
        insufficient_evidence_summary=None
        if prediction_candidates
        else "No candidate could be synthesized.",
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
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.audit_dir.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text(render_markdown_report(report), encoding="utf-8")
    paths.json_path.write_text(render_json_report(report), encoding="utf-8")
    manifest = report.audit_manifest
    if not isinstance(manifest, AuditManifest):
        raise TypeError("Codex smoke reports must include an audit manifest")
    manifest_sha = write_json_artifact(
        paths.audit_manifest_path,
        cast(JsonObject, manifest.model_dump(mode="json")),
    )
    report_sha = file_sha256(paths.report_path)
    json_sha = file_sha256(paths.json_path)
    report_tool_run_id = f"tool-render-report-{run.run_id}"
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=report_tool_run_id,
            run_id=run.run_id,
            tool_name="render_prediction_report",
            tool_version="phase2.v1",
            status="ok",
            started_at=now,
            completed_at=now,
            inputs={"symbol": symbol},
        )
    )
    for artifact_id, artifact_type, path, sha256 in (
        (
            f"artifact-report-md-{stable_digest(run.run_id)}",
            "markdown_report",
            paths.report_path,
            report_sha,
        ),
        (
            f"artifact-report-json-{stable_digest(run.run_id)}",
            "json_report",
            paths.json_path,
            json_sha,
        ),
        (
            f"artifact-audit-manifest-{stable_digest(run.run_id)}",
            "audit_manifest",
            paths.audit_manifest_path,
            manifest_sha,
        ),
    ):
        store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                tool_run_id=report_tool_run_id,
                artifact_type=artifact_type,
                path=path.relative_to(repo_root),
                sha256=sha256,
                schema_version="phase2-report.v1",
                metadata={"run_id": run.run_id},
                created_at=now,
            )
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
    return PredictionCandidate(
        candidate_id=candidate.candidate_id,
        instrument_id=candidate.instrument_id,
        symbol=symbol,
        horizon=TimeHorizon.SWING,
        direction=Direction.MIXED,
        status=status,
        thesis=candidate.scenario,
        baseline="No directional edge is assumed; dummy tools only validate orchestration.",
        confidence=candidate.confidence or 0.0,
        evidence_for=evidence_for_refs,
        evidence_against=evidence_against_refs,
        assumptions=("Codex search evidence is source material, not automatically true.",),
        uncertainties=(candidate.uncertainty or "Phase 2 tools are structural dummies.",),
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
        "ml_forecast",
        "instrument_universe",
        "audit_manifest",
    }:
        raise ValueError(f"unknown artifact type: {artifact_type}")
    return AuditArtifact(
        artifact_id=record.artifact_id,
        artifact_type=cast(ArtifactType, artifact_type),
        path=path.as_posix(),
        created_at=record.created_at or utc_now(),
        produced_by="phase2-mcp",
        sha256=record.sha256,
        metadata=record.metadata,
    )


__all__ = ["render_phase2_prediction_report"]
