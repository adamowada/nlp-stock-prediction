"""Report rendering for Phase 2 Codex smoke runs."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import (
    CredentialState,
    ProviderStatus,
)
from nlp_stock_prediction.contracts.instruments import Instrument
from nlp_stock_prediction.contracts.provenance import EvidenceReference, ProviderHealth
from nlp_stock_prediction.contracts.report import (
    AuditManifest,
    DailyReport,
    DataFreshnessSummary,
    InstrumentReportSection,
    InsufficientEvidenceReport,
    PriorOutcomeReview,
)
from nlp_stock_prediction.instruments.repository import instrument_from_record
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase2_common import (
    Phase2RunPaths,
    phase2_instrument,
    stable_digest,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase2_evidence import source_evidence_from_record
from nlp_stock_prediction.orchestration.report_assembly import (
    material_claim_traces,
    prepare_report_assembly_state,
    report_source_references,
)
from nlp_stock_prediction.orchestration.report_candidates import prediction_candidate_from_record
from nlp_stock_prediction.orchestration.report_data_modes import (
    CODEX_SMOKE_REPORT_DATA_MODE,
    LIVE_REPORT_DATA_MODE,
    OFFLINE_FIXTURE_REPORT_DATA_MODE,
    ReportDataMode,
    enforce_live_report_input_boundary,
    report_data_mode_from_run,
    report_data_mode_metadata,
)
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.storage.records import (
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
    report_data_mode: ReportDataMode | None = None,
) -> JsonObject:
    now = utc_now()
    is_phase4_report = artifact_schema_version.startswith("phase4")
    resolved_report_data_mode = report_data_mode or report_data_mode_from_run(
        run,
        default=(
            OFFLINE_FIXTURE_REPORT_DATA_MODE if is_phase4_report else CODEX_SMOKE_REPORT_DATA_MODE
        ),
    )
    enforce_live_report_input_boundary(store=store, run=run)
    mode_metadata = report_data_mode_metadata(resolved_report_data_mode)
    evidence_records = store.list_evidence_for_run(run.run_id)
    evidence_sources = tuple(source_evidence_from_record(record) for record in evidence_records)
    instrument = _primary_instrument(store, symbol=symbol, fallback_generated_at=now)
    candidate_records = store.list_prediction_candidates_for_run(run.run_id)
    artifact_records = store.list_artifacts_for_run(run.run_id)
    tool_runs = store.list_tool_runs_for_run(run.run_id)
    assembly_state = prepare_report_assembly_state(
        store=store,
        repo_root=repo_root,
        artifact_records=artifact_records,
        evidence_records=evidence_records,
        candidate_records=candidate_records,
        tool_runs=tool_runs,
    )
    prediction_candidates = tuple(
        prediction_candidate_from_record(
            candidate,
            evidence_sources,
            prefer_evaluated_references=True,
        )
        for candidate in assembly_state.usable_candidate_records
    )
    section_refs = tuple(
        EvidenceReference(
            evidence_id=record.evidence_id,
            quote=record.text[:160],
            relevance=0.75,
        )
        for record in evidence_sources[:5]
    )
    if resolved_report_data_mode == LIVE_REPORT_DATA_MODE:
        observed_discussion_summary = "Live provider research evidence was assembled from storage."
        analysis_summary = (
            "Live report rendering consumes stored provider evidence, signal artifacts, "
            "prediction candidates, and prediction-quality evaluations."
        )
        universe_summary = f"Live provider research universe for {symbol.upper()}"
        freshness_summary = (
            "Live provider evidence is present for this run."
            if evidence_sources
            else "Live report rendering found no stored provider evidence for this run."
        )
    elif is_phase4_report:
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
    audit_artifacts = assembly_state.audit_artifacts
    provider_name = _provider_name_for_mode(
        resolved_report_data_mode,
    )
    has_codex_search_evidence = bool(evidence_sources)
    stale_provider_names = _stale_provider_names(evidence_sources)
    provider_status = (
        ProviderStatus.STALE
        if stale_provider_names and has_codex_search_evidence
        else ProviderStatus.OK
        if has_codex_search_evidence
        else ProviderStatus.EMPTY
    )
    provider_health = (
        ProviderHealth(
            provider_name=provider_name,
            status=provider_status,
            checked_at=now,
            credential_state=CredentialState.NOT_REQUIRED,
        ),
        *assembly_state.provider_health,
    )
    source_references = report_source_references(
        evidence_sources=evidence_sources,
        audit_artifacts=audit_artifacts,
        provider_health=provider_health,
        assembly_state=assembly_state,
    )
    material_traces = material_claim_traces(
        prediction_candidates=prediction_candidates,
        source_references=source_references,
        assembly_state=assembly_state,
    )
    prior_outcome_reviews = _prior_outcome_reviews(prediction_candidates)
    insufficient_summary = _insufficient_evidence_summary(
        prediction_candidates=prediction_candidates,
        assembly_state_blocking_reasons=assembly_state.blocking_reasons,
        default_summary=insufficient_evidence_summary,
    )
    insufficient_blocking_reasons = (
        assembly_state.blocking_reasons
        if assembly_state.blocking_reasons
        else (insufficient_summary,)
    )
    insufficient_evidence = (
        None
        if prediction_candidates
        else InsufficientEvidenceReport(
            summary=insufficient_summary,
            blocking_reasons=insufficient_blocking_reasons,
            provider_names=tuple(health.provider_name for health in provider_health),
        )
    )
    instrument_section = instrument_section.model_copy(
        update={
            "data_quality": {
                **instrument_section.data_quality,
                "stored_candidate_count": len(candidate_records),
                "assembled_candidate_count": len(prediction_candidates),
                "excluded_candidate_count": len(assembly_state.excluded_candidate_reasons),
                "audit_artifact_count": len(audit_artifacts),
            }
        }
    )
    report = DailyReport(
        schema_version="daily-report.v2",
        run_id=run.run_id,
        report_date=run_date,
        generated_at=now,
        timezone="UTC",
        objective=run.objective,
        universe=universe_summary,
        command_args={"run_id": run.run_id, "symbol": symbol.upper(), **mode_metadata},
        instruments=(instrument,),
        data_freshness=DataFreshnessSummary(
            as_of=now,
            summary=freshness_summary,
            stale_provider_names=stale_provider_names,
            missing_provider_names=() if has_codex_search_evidence else (provider_name,),
        ),
        provider_health=provider_health,
        evidence_sources=evidence_sources,
        instrument_sections=(instrument_section,),
        prediction_candidates=prediction_candidates,
        insufficient_evidence=insufficient_evidence,
        insufficient_evidence_summary=None if prediction_candidates else insufficient_summary,
        source_references=source_references,
        material_claim_traces=material_traces,
        prior_outcome_reviews=prior_outcome_reviews,
        audit_manifest=AuditManifest(
            run_id=run.run_id,
            schema_version="audit-manifest.v2",
            created_at=now,
            artifacts=audit_artifacts,
            command_args={"run_id": run.run_id, "symbol": symbol.upper(), **mode_metadata},
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
                inputs={"symbol": symbol, **mode_metadata},
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
        metadata={"run_id": run.run_id, **mode_metadata},
    )
    json_artifact = report_index.write_text(
        artifact_id=f"artifact-report-json-{stable_digest(run.run_id)}",
        artifact_type="json_report",
        filename=paths.json_path.name,
        content=render_json_report(report),
        metadata={"run_id": run.run_id, **mode_metadata},
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
        metadata={"run_id": run.run_id, **mode_metadata},
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=run.run_id,
            run_kind=run.run_kind,
            objective=run.objective,
            status="completed",
            started_at=run.started_at,
            completed_at=now,
            metadata={**run.metadata, **mode_metadata},
        )
    )
    return {
        "run_id": run.run_id,
        "markdown_path": paths.report_path.as_posix(),
        "json_path": paths.json_path.as_posix(),
        "audit_manifest_path": paths.audit_manifest_path.as_posix(),
        "candidate_count": len(prediction_candidates),
        "stored_candidate_count": len(candidate_records),
        "excluded_candidate_ids": list(assembly_state.excluded_candidate_ids),
        "warnings": list(assembly_state.warnings),
        **mode_metadata,
    }


def _provider_name_for_mode(
    report_data_mode: ReportDataMode,
) -> str:
    if report_data_mode == LIVE_REPORT_DATA_MODE:
        return "live-providers"
    if report_data_mode == OFFLINE_FIXTURE_REPORT_DATA_MODE:
        return "phase4-fixture-tools"
    if report_data_mode == CODEX_SMOKE_REPORT_DATA_MODE:
        return "codex-web-search"
    return "dummy-smoke-tools"


def _prior_outcome_reviews(
    prediction_candidates: tuple[object, ...],
) -> tuple[PriorOutcomeReview, ...]:
    reviews: list[PriorOutcomeReview] = []
    for candidate in prediction_candidates:
        candidate_id = getattr(candidate, "candidate_id", None)
        prior_ids = getattr(candidate, "prior_outcome_review_ids", ())
        if not isinstance(candidate_id, str):
            continue
        for review_id in prior_ids:
            if not isinstance(review_id, str) or not review_id:
                continue
            reviews.append(
                PriorOutcomeReview(
                    review_id=review_id,
                    status="not_available",
                    summary="No prior outcome review is available for this run.",
                    candidate_id=candidate_id,
                    limitations=("No stored prior outcome artifact was linked to this candidate.",),
                )
            )
    return tuple(reviews)


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


def _stale_provider_names(evidence_sources: tuple[object, ...]) -> tuple[str, ...]:
    provider_names: list[str] = []
    for evidence in evidence_sources:
        provenance = getattr(evidence, "provenance", None)
        freshness_status = getattr(provenance, "freshness_status", None)
        if getattr(freshness_status, "value", freshness_status) != "stale":
            continue
        provider_name = getattr(provenance, "provider_name", None)
        if isinstance(provider_name, str) and provider_name:
            provider_names.append(provider_name)
    return tuple(dict.fromkeys(provider_names))


def _insufficient_evidence_summary(
    *,
    prediction_candidates: tuple[object, ...],
    assembly_state_blocking_reasons: tuple[str, ...],
    default_summary: str | None,
) -> str:
    if prediction_candidates:
        return ""
    if assembly_state_blocking_reasons:
        return (
            "Report assembly could not use stored prediction candidates because required "
            "tool artifacts or evidence were missing, malformed, or not linked through the "
            "stored run graph."
        )
    return default_summary or "No candidate could be synthesized."


__all__ = ["render_phase2_prediction_report"]
