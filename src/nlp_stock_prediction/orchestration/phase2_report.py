"""Report rendering for Phase 2 Codex smoke runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import (
    AssetClass,
    CredentialState,
    InstrumentResolutionStatus,
    ProviderStatus,
    TradabilityStatus,
)
from nlp_stock_prediction.contracts.instruments import (
    Instrument,
    InstrumentDataAvailability,
    InstrumentResolution,
    InstrumentUniverse,
    ProviderInstrumentId,
    TradabilityEvidence,
)
from nlp_stock_prediction.contracts.provenance import EvidenceReference, ProviderHealth
from nlp_stock_prediction.contracts.report import (
    AuditArtifact,
    AuditManifest,
    DailyReport,
    DataFreshnessSummary,
    InstrumentReportSection,
    InsufficientEvidenceReport,
)
from nlp_stock_prediction.instruments.repository import instrument_from_record
from nlp_stock_prediction.orchestration.artifact_policy import FINAL_REPORT_ARTIFACT_TYPES
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase2_common import (
    Phase2RunPaths,
    phase2_instrument,
    stable_digest,
    symbol_slug,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase2_evidence import source_evidence_from_record
from nlp_stock_prediction.orchestration.phase4_universe_discovery import (
    PHASE4_LIVE_SYMBOL_PROVIDER,
)
from nlp_stock_prediction.orchestration.prior_outcomes import apply_prior_outcome_context
from nlp_stock_prediction.orchestration.report_assembly import (
    ReportAssemblyState,
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
    ArtifactRecord,
    PredictionCandidateRecord,
    ReportArtifactRecord,
    ResearchRunRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore


@dataclass(frozen=True)
class PredictionReportBuildRequest:
    """Phase-neutral request for building and indexing a final prediction report bundle."""

    store: SQLiteStore
    repo_root: Path
    run: ResearchRunRecord
    paths: Phase2RunPaths
    run_date: date
    symbol: str
    tool_run_id: str | None = None
    record_tool_run: bool = True
    tool_name: str = "render_prediction_report"
    tool_version: str = "phase2.v1"
    tool_status: str = "ok"
    tool_warnings: tuple[str, ...] = ()
    produced_by: str = "render_prediction_report"
    artifact_schema_version: str = "phase2-report.v1"
    insufficient_evidence_summary: str | None = None
    report_data_mode: ReportDataMode | None = None


@dataclass(frozen=True)
class ReportBundleBuilder:
    """Build the Markdown, JSON, audit manifest, and report-artifact index in one seam."""

    def build(self, request: PredictionReportBuildRequest) -> JsonObject:
        return _render_phase2_prediction_report_core(
            store=request.store,
            repo_root=request.repo_root,
            run=request.run,
            paths=request.paths,
            run_date=request.run_date,
            symbol=request.symbol,
            tool_run_id=request.tool_run_id,
            record_tool_run=request.record_tool_run,
            tool_name=request.tool_name,
            tool_version=request.tool_version,
            tool_status=request.tool_status,
            tool_warnings=request.tool_warnings,
            produced_by=request.produced_by,
            artifact_schema_version=request.artifact_schema_version,
            insufficient_evidence_summary=request.insufficient_evidence_summary,
            report_data_mode=request.report_data_mode,
        )


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
    return ReportBundleBuilder().build(
        PredictionReportBuildRequest(
            store=store,
            repo_root=repo_root,
            run=run,
            paths=paths,
            run_date=run_date,
            symbol=symbol,
            tool_run_id=tool_run_id,
            record_tool_run=record_tool_run,
            tool_name=tool_name,
            tool_version=tool_version,
            tool_status=tool_status,
            tool_warnings=tool_warnings,
            produced_by=produced_by,
            artifact_schema_version=artifact_schema_version,
            insufficient_evidence_summary=insufficient_evidence_summary,
            report_data_mode=report_data_mode,
        )
    )


def _render_phase2_prediction_report_core(
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
    enforce_live_report_input_boundary(
        store=store,
        run=run,
        report_data_mode=resolved_report_data_mode,
    )
    mode_metadata = report_data_mode_metadata(resolved_report_data_mode)
    evidence_records = store.list_evidence_for_run(run.run_id)
    evidence_sources = tuple(source_evidence_from_record(record) for record in evidence_records)
    instrument = _primary_instrument(
        store,
        symbol=symbol,
        fallback_generated_at=now,
        report_data_mode=resolved_report_data_mode,
    )
    candidate_records = store.list_prediction_candidates_for_run(run.run_id)
    artifact_records = tuple(
        record
        for record in store.list_artifacts_for_run(run.run_id)
        if record.artifact_type not in FINAL_REPORT_ARTIFACT_TYPES
    )
    instrument_resolutions = _instrument_resolutions_from_artifacts(
        artifact_records,
        repo_root=repo_root,
        report_instrument_ids=(instrument.instrument_id,),
    )
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
            candidate_evidence_links=store.list_candidate_evidence_links(candidate.candidate_id),
            prefer_evaluated_references=True,
        )
        for candidate in assembly_state.usable_candidate_records
    )
    provider_name = _provider_name_for_mode(
        resolved_report_data_mode,
    )
    has_codex_search_evidence = bool(evidence_sources)
    stale_provider_names = _stale_provider_names(evidence_sources)
    missing_provider_names = () if has_codex_search_evidence else (provider_name,)
    prior_outcome_context = apply_prior_outcome_context(
        store=store,
        repo_root=repo_root,
        run=run,
        report_date=run_date,
        reviewed_at=now,
        instrument=instrument,
        prediction_candidates=prediction_candidates,
        missing_provider_names=missing_provider_names,
        stale_provider_names=stale_provider_names,
        report_data_mode=resolved_report_data_mode,
    )
    prediction_candidates = prior_outcome_context.prediction_candidates
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
    audit_artifacts = (*assembly_state.audit_artifacts, *prior_outcome_context.audit_artifacts)
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
    source_references = (*source_references, *prior_outcome_context.source_references)
    material_traces = material_claim_traces(
        prediction_candidates=prediction_candidates,
        source_references=source_references,
        assembly_state=assembly_state,
    )
    material_traces = (*material_traces, *prior_outcome_context.material_claim_traces)
    prior_outcome_reviews = prior_outcome_context.prior_outcome_reviews
    resolution_blocking_reasons = _instrument_resolution_blocking_reasons(
        instrument_resolutions,
    )
    blocking_reasons = tuple(
        dict.fromkeys((*assembly_state.blocking_reasons, *resolution_blocking_reasons))
    )
    insufficient_summary = _insufficient_evidence_summary(
        prediction_candidates=prediction_candidates,
        blocking_reasons=blocking_reasons,
        default_summary=insufficient_evidence_summary,
    )
    insufficient_blocking_reasons = (
        blocking_reasons if blocking_reasons else (insufficient_summary,)
    )
    insufficient_missing_evidence_types = _insufficient_missing_evidence_types(
        prediction_candidates=prediction_candidates,
        evidence_sources=evidence_sources,
        candidate_records=candidate_records,
        assembly_state=assembly_state,
        provider_health=provider_health,
        stale_provider_names=stale_provider_names,
        instrument_resolutions=instrument_resolutions,
    )
    insufficient_artifact_ids = _insufficient_artifact_ids(
        audit_artifacts=audit_artifacts,
        assembly_state=assembly_state,
    )
    insufficient_metadata = _insufficient_evidence_metadata(
        candidate_records=candidate_records,
        assembly_state=assembly_state,
        provider_health=provider_health,
        instrument_resolutions=instrument_resolutions,
    )
    insufficient_evidence = (
        None
        if prediction_candidates
        else InsufficientEvidenceReport(
            summary=insufficient_summary,
            blocking_reasons=insufficient_blocking_reasons,
            missing_evidence_types=insufficient_missing_evidence_types,
            provider_names=tuple(health.provider_name for health in provider_health),
            evidence=section_refs,
            artifact_ids=insufficient_artifact_ids,
            metadata=insufficient_metadata,
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
                "instrument_resolution_status_counts": _resolution_status_counts(
                    instrument_resolutions
                ),
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
        instrument_resolutions=instrument_resolutions,
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
            provider_health=provider_health,
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
    audit_manifest_artifact = ArtifactIndex.for_directory(
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
    _record_final_report_artifact_index(
        store=store,
        repo_root=repo_root,
        run=run,
        run_completed_at=now,
        report=report,
        instrument=instrument,
        report_data_mode=resolved_report_data_mode,
        tool_run_id=report_tool_run_id,
        artifact_schema_version=artifact_schema_version,
        artifacts=(markdown_artifact, json_artifact, audit_manifest_artifact),
        metadata={
            "candidate_count": len(prediction_candidates),
            "stored_candidate_count": len(candidate_records),
            "excluded_candidate_ids": list(assembly_state.excluded_candidate_ids),
            "provider_names": [health.provider_name for health in provider_health],
            **mode_metadata,
        },
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
        "warnings": list((*assembly_state.warnings, *prior_outcome_context.warnings)),
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


def _record_final_report_artifact_index(
    *,
    store: SQLiteStore,
    repo_root: Path,
    run: ResearchRunRecord,
    run_completed_at: datetime,
    report: DailyReport,
    instrument: Instrument,
    report_data_mode: ReportDataMode,
    tool_run_id: str,
    artifact_schema_version: str,
    artifacts: tuple[AuditArtifact, ...],
    metadata: JsonObject,
) -> None:
    for artifact in artifacts:
        if artifact.sha256 is None:
            raise ValueError("final report artifact index requires artifact hashes")
        store.record_report_artifact(
            ReportArtifactRecord(
                artifact_id=artifact.artifact_id,
                run_id=run.run_id,
                tool_run_id=tool_run_id,
                artifact_type=artifact.artifact_type,
                path=Path(artifact.path).resolve().relative_to(repo_root.resolve())
                if Path(artifact.path).is_absolute()
                else Path(artifact.path),
                sha256=artifact.sha256,
                schema_version=artifact_schema_version,
                report_schema_version=report.schema_version,
                report_date=report.report_date,
                instrument_id=instrument.instrument_id,
                symbol=instrument.symbol,
                report_data_mode=report_data_mode,
                source_run_started_at=run.started_at,
                source_run_completed_at=run_completed_at,
                metadata=metadata,
                created_at=artifact.created_at,
            )
        )


def _primary_instrument(
    store: SQLiteStore,
    *,
    symbol: str,
    fallback_generated_at: datetime,
    report_data_mode: ReportDataMode,
) -> Instrument:
    if report_data_mode == LIVE_REPORT_DATA_MODE:
        record = store.find_instrument_by_provider_id(
            PHASE4_LIVE_SYMBOL_PROVIDER,
            "symbol",
            symbol.upper(),
        )
        if record is not None:
            return instrument_from_record(record)
        for discovered in store.find_instruments_by_symbol_or_alias(symbol.upper()):
            if discovered.metadata.get("live_provider_symbol") is True:
                return instrument_from_record(discovered)
        return _live_unknown_instrument(symbol.upper(), fallback_generated_at)
    instrument_id = f"instrument:codex:{symbol.upper()}"
    record = store.get_instrument(instrument_id)
    if record is not None:
        return instrument_from_record(record)
    discovered_records = store.find_instruments_by_symbol_or_alias(symbol.upper())
    if discovered_records:
        return instrument_from_record(discovered_records[0])
    return phase2_instrument(symbol.upper(), fallback_generated_at)


def _live_unknown_instrument(symbol: str, retrieved_at: datetime) -> Instrument:
    normalized = symbol.upper()
    return Instrument(
        instrument_id=f"instrument:live:unknown:{symbol_slug(normalized)}",
        symbol=normalized,
        display_name=f"{normalized} unresolved live instrument",
        asset_class=AssetClass.UNKNOWN,
        provider_ids=(
            ProviderInstrumentId(
                provider=PHASE4_LIVE_SYMBOL_PROVIDER,
                identifier=normalized,
                namespace="symbol",
            ),
        ),
        tradability_evidence=(
            TradabilityEvidence(
                provider=PHASE4_LIVE_SYMBOL_PROVIDER,
                status=TradabilityStatus.UNKNOWN,
                retrieved_at=retrieved_at,
                raw_identifier=f"{normalized}:live-unresolved",
                notes="Live instrument identity could not be resolved from available providers.",
            ),
        ),
        data_availability=(
            InstrumentDataAvailability(
                provider=PHASE4_LIVE_SYMBOL_PROVIDER,
                data_type="live_provider_lookup",
                status=TradabilityStatus.UNKNOWN,
                checked_at=retrieved_at,
                provider_identifier=normalized,
            ),
        ),
        metadata={"live_provider_symbol": True, "resolution_status": "unavailable"},
    )


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


def _instrument_resolutions_from_artifacts(
    artifact_records: tuple[ArtifactRecord, ...],
    *,
    repo_root: Path,
    report_instrument_ids: tuple[str, ...],
) -> tuple[InstrumentResolution, ...]:
    report_ids = set(report_instrument_ids)
    resolutions: list[InstrumentResolution] = []
    seen: set[tuple[str, str, str | None]] = set()
    for record in artifact_records:
        if record.artifact_type != "instrument_universe":
            continue
        path = _artifact_record_path(record, repo_root)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            universe_payload = payload.get("universe")
            if isinstance(universe_payload, dict):
                universe_payload = dict(universe_payload)
                universe_payload.pop("instrument_ids", None)
            universe = InstrumentUniverse.model_validate(universe_payload)
        except OSError, json.JSONDecodeError, ValidationError, ValueError:
            continue
        for resolution in universe.resolutions:
            if (
                resolution.selected_instrument_id is not None
                and resolution.selected_instrument_id not in report_ids
            ):
                continue
            key = (
                resolution.query,
                resolution.status.value,
                resolution.selected_instrument_id,
            )
            if key in seen:
                continue
            seen.add(key)
            resolutions.append(resolution)
    return tuple(resolutions)


def _instrument_resolution_blocking_reasons(
    instrument_resolutions: tuple[InstrumentResolution, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for resolution in instrument_resolutions:
        if resolution.status == InstrumentResolutionStatus.RESOLVED:
            continue
        warning_text = f" Warnings: {'; '.join(resolution.warnings)}" if resolution.warnings else ""
        if resolution.status == InstrumentResolutionStatus.AMBIGUOUS:
            reasons.append(
                f"Instrument query {resolution.query} is ambiguous; no default instrument was "
                f"selected.{warning_text}"
            )
            continue
        reasons.append(
            f"Instrument query {resolution.query} is {resolution.status.value}; it cannot support "
            f"a reportable prediction candidate.{warning_text}"
        )
    return tuple(dict.fromkeys(reasons))


def _insufficient_missing_evidence_types(
    *,
    prediction_candidates: tuple[object, ...],
    evidence_sources: tuple[object, ...],
    candidate_records: tuple[PredictionCandidateRecord, ...],
    assembly_state: ReportAssemblyState,
    provider_health: tuple[ProviderHealth, ...],
    stale_provider_names: tuple[str, ...],
    instrument_resolutions: tuple[InstrumentResolution, ...],
) -> tuple[str, ...]:
    if prediction_candidates:
        return ()
    missing: list[str] = []
    if not candidate_records:
        missing.append("stored prediction candidates")
    elif assembly_state.excluded_candidate_reasons:
        missing.append("usable prediction candidates")
    if not evidence_sources:
        missing.append("attributable source evidence")
    if stale_provider_names:
        missing.append("fresh source evidence")
    if assembly_state.missing_evidence_ids:
        missing.append("cited source evidence records")
    if assembly_state.missing_artifact_ids:
        missing.append("required tool artifacts")
    if any(_provider_status_blocks_evidence(health.status) for health in provider_health):
        missing.append("complete provider outputs")
    if any(
        resolution.status != InstrumentResolutionStatus.RESOLVED
        for resolution in instrument_resolutions
    ):
        missing.append("resolved supported instrument identity")
    return tuple(dict.fromkeys(missing))


def _insufficient_artifact_ids(
    *,
    audit_artifacts: tuple[AuditArtifact, ...],
    assembly_state: ReportAssemblyState,
) -> tuple[str, ...]:
    if not audit_artifacts:
        return ()
    if not assembly_state.blocking_reasons:
        return tuple(artifact.artifact_id for artifact in audit_artifacts)
    return tuple(
        artifact.artifact_id
        for artifact in audit_artifacts
        if artifact.metadata.get("assembly_required") is True
        or artifact.metadata.get("assembly_status") != "ok"
    )


def _insufficient_evidence_metadata(
    *,
    candidate_records: tuple[PredictionCandidateRecord, ...],
    assembly_state: ReportAssemblyState,
    provider_health: tuple[ProviderHealth, ...],
    instrument_resolutions: tuple[InstrumentResolution, ...],
) -> JsonObject:
    return {
        "stored_candidate_count": len(candidate_records),
        "excluded_candidate_ids": list(assembly_state.excluded_candidate_ids),
        "missing_evidence_ids": list(assembly_state.missing_evidence_ids),
        "missing_artifact_ids": list(assembly_state.missing_artifact_ids),
        "provider_statuses": {
            health.provider_name: health.status.value for health in provider_health
        },
        "instrument_resolution_status_counts": _resolution_status_counts(instrument_resolutions),
    }


def _provider_status_blocks_evidence(status: ProviderStatus) -> bool:
    return status in {
        ProviderStatus.EMPTY,
        ProviderStatus.PARTIAL,
        ProviderStatus.STALE,
        ProviderStatus.UNCONFIGURED,
        ProviderStatus.UNAUTHORIZED,
        ProviderStatus.RATE_LIMITED,
        ProviderStatus.FAILED,
        ProviderStatus.MALFORMED,
    }


def _resolution_status_counts(
    instrument_resolutions: tuple[InstrumentResolution, ...],
) -> JsonObject:
    counts: JsonObject = {}
    for resolution in instrument_resolutions:
        current = counts.get(resolution.status.value, 0)
        counts[resolution.status.value] = int(current) + 1 if isinstance(current, int) else 1
    return counts


def _artifact_record_path(record: ArtifactRecord, repo_root: Path) -> Path:
    return record.path if record.path.is_absolute() else repo_root / record.path


def _insufficient_evidence_summary(
    *,
    prediction_candidates: tuple[object, ...],
    blocking_reasons: tuple[str, ...],
    default_summary: str | None,
) -> str:
    if prediction_candidates:
        return ""
    if blocking_reasons:
        return (
            "Report assembly could not use stored prediction candidates because required "
            "instrument resolution, tool artifacts, evidence, or provider checks blocked a "
            "reliable conclusion."
        )
    return default_summary or "No candidate could be synthesized."


__all__ = [
    "PredictionReportBuildRequest",
    "ReportBundleBuilder",
    "render_phase2_prediction_report",
]
