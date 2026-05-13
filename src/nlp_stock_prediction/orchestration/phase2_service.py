"""MCP-facing Phase 2 service for real Codex smoke runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts import (
    AssetClass,
    AuditArtifact,
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
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TimeHorizon,
    TradabilityStatus,
)
from nlp_stock_prediction.contracts.instruments import (
    InstrumentDataAvailability,
    ProviderInstrumentId,
    TradabilityEvidence,
)
from nlp_stock_prediction.orchestration.artifacts import ArtifactType
from nlp_stock_prediction.reporting.audit import write_json_artifact
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    CandidateArtifactLinkRecord,
    CandidateEvidenceLinkRecord,
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    SourceQueryRecord,
    SQLiteStore,
    ToolRunRecord,
    initialize_research_database,
)

ALLOWED_WRITE_ROOTS = ("reports", "artifacts", "data", "cache")


@dataclass(frozen=True)
class Phase2RunPaths:
    output_dir: Path
    run_dir: Path
    audit_dir: Path
    report_path: Path
    json_path: Path
    audit_manifest_path: Path


@dataclass(frozen=True)
class Phase2McpService:
    """Small stateful service exposed through MCP and tests."""

    repo_root: Path = Path(".")
    database_path: Path = Path("data/prediction-research.sqlite3")

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", self.repo_root.resolve())

    @property
    def store(self) -> SQLiteStore:
        path = self._resolve_write_path(self.database_path)
        return initialize_research_database(path)

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
        run_id = _run_id(parsed_date, normalized_symbol)
        paths = self._paths(parsed_date, output_dir)
        now = _utc_now()
        self.store.upsert_research_run(
            ResearchRunRecord(
                run_id=run_id,
                run_kind="codex_smoke_prediction_report",
                objective=objective or f"Codex smoke prediction research for {normalized_symbol}",
                status="running",
                started_at=now,
                metadata={
                    "run_date": parsed_date.isoformat(),
                    "symbol": normalized_symbol,
                    "output_dir": paths.output_dir.as_posix(),
                    "phase": "phase2_codex_orchestrator",
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
        tools: list[JsonObject] = [
            {
                "tool_name": "record_codex_search_evidence",
                "stage": "collect",
                "description": "Record one live-search source as normalized evidence.",
            },
            {
                "tool_name": "run_dummy_universe_tool",
                "stage": "discover",
                "description": "Write a deterministic retail-accessible instrument universe.",
            },
            {
                "tool_name": "run_dummy_analysis_tool",
                "stage": "analyze",
                "description": "Write deterministic dummy analysis context.",
            },
            {
                "tool_name": "synthesize_prediction_candidates",
                "stage": "synthesize",
                "description": "Create evidence-backed or insufficient-evidence candidates.",
            },
            {
                "tool_name": "render_prediction_report",
                "stage": "report",
                "description": "Render Markdown, JSON, and audit manifest artifacts.",
            },
        ]
        return cast(
            JsonObject,
            {
                "tools": tools,
                "stage_order": ["discover", "collect", "analyze", "synthesize", "report"],
            },
        )

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
    ) -> JsonObject:
        run = self._require_run(run_id)
        run_date = _run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        now = _utc_now()
        normalized_symbol = symbol.strip().upper()
        digest = _stable_digest("|".join([run_id, normalized_symbol, title, url, claim]))
        evidence_id = f"evidence-codex-search-{digest}"
        tool_run_id = f"tool-codex-search-{digest}"
        source_query_id = f"query-codex-search-{digest}"
        artifact_id = f"artifact-codex-search-{digest}"
        artifact_path = paths.audit_dir / f"codex-search-evidence-{digest}.json"

        source_kind = SourceKind.NEWS_ARTICLE
        provenance = SourceProvenance(
            provider_name="codex-web-search",
            source_kind=source_kind,
            retrieval_method=RetrievalMethod.LLM,
            fetched_at=now,
            observed_at=_parse_optional_datetime(published_at) or now,
            source_url=url,
            permalink=url,
            raw_identifier=url,
            raw_snapshot_id=artifact_id,
            query=query,
            freshness_status=FreshnessStatus.FRESH,
            provider_metadata={"codex_search": True},
        )
        evidence = SourceEvidence(
            evidence_id=evidence_id,
            source_kind=source_kind,
            ticker=normalized_symbol,
            title=title,
            text=claim,
            created_at=_parse_optional_datetime(published_at),
            permalink=url,
            matched_tickers=(normalized_symbol,),
            provenance=provenance,
            metadata={"codex_search": True},
        )
        payload: JsonObject = {
            "schema_version": "codex-search-evidence.v1",
            "run_id": run_id,
            "records": [cast(JsonObject, evidence.model_dump(mode="json"))],
        }
        sha256 = write_json_artifact(artifact_path, payload)
        store = self.store
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name="record_codex_search_evidence",
                tool_version="phase2.v1",
                status="ok",
                inputs={"symbol": normalized_symbol, "url": url, "query": query},
                started_at=now,
                completed_at=now,
            )
        )
        store.record_source_query(
            SourceQueryRecord(
                source_query_id=source_query_id,
                tool_run_id=tool_run_id,
                provider="codex-web-search",
                query=query,
                url=url,
                retrieved_at=now,
                metadata={"codex_search": True},
            )
        )
        store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                tool_run_id=tool_run_id,
                artifact_type="normalized_evidence",
                path=artifact_path.relative_to(self.repo_root),
                sha256=sha256,
                schema_version="codex-search-evidence.v1",
                metadata={"codex_search": True, "evidence_id": evidence_id},
                created_at=now,
            )
        )
        store.record_evidence(
            EvidenceRecord(
                evidence_id=evidence_id,
                tool_run_id=tool_run_id,
                source_query_id=source_query_id,
                source_type=source_kind.value,
                provider="codex-web-search",
                url=url,
                query=query,
                retrieved_at=now,
                published_at=_parse_optional_datetime(published_at),
                instruments=(f"instrument:codex:{normalized_symbol}",),
                claim=claim,
                extraction_confidence=0.7,
                source_reliability="codex_search_source",
                freshness_status=FreshnessStatus.FRESH.value,
                artifact_id=artifact_id,
                raw_excerpt=claim[:240],
                provenance_json=cast(JsonObject, provenance.model_dump(mode="json")),
                metadata={
                    "codex_search": True,
                    "source_evidence": cast(JsonObject, evidence.model_dump(mode="json")),
                },
            )
        )
        return {"run_id": run_id, "evidence_id": evidence_id, "artifact_id": artifact_id}

    def run_dummy_universe_tool(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        run_date = _run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        now = _utc_now()
        instruments = (_instrument(symbol.strip().upper(), now),)
        payload: JsonObject = {
            "schema_version": "dummy-universe.v1",
            "run_id": run_id,
            "records": [instrument.model_dump(mode="json") for instrument in instruments],
        }
        artifact_id = f"artifact-dummy-universe-{_stable_digest(run_id)}"
        path = paths.audit_dir / "dummy-universe.json"
        sha256 = write_json_artifact(path, payload)
        tool_run_id = f"tool-dummy-universe-{run_id}"
        store = self.store
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name="run_dummy_universe_tool",
                tool_version="phase2.v1",
                status="ok",
                started_at=now,
                completed_at=now,
                inputs={"symbol": symbol},
            )
        )
        for instrument in instruments:
            store.upsert_instrument(
                InstrumentRecord(
                    instrument_id=instrument.instrument_id,
                    symbol=instrument.symbol,
                    asset_class=instrument.asset_class.value,
                    name=instrument.display_name,
                    venue=instrument.venue,
                    provider_ids=tuple(
                        cast(JsonObject, item.model_dump(mode="json"))
                        for item in instrument.provider_ids
                    ),
                    tradability_evidence=tuple(
                        cast(JsonObject, item.model_dump(mode="json"))
                        for item in instrument.tradability_evidence
                    ),
                    data_availability=tuple(
                        cast(JsonObject, item.model_dump(mode="json"))
                        for item in instrument.data_availability
                    ),
                    metadata={"phase2_mcp": True},
                )
            )
        store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                tool_run_id=tool_run_id,
                artifact_type="provider_result",
                path=path.relative_to(self.repo_root),
                sha256=sha256,
                schema_version="dummy-universe.v1",
                metadata={"instrument_ids": [item.instrument_id for item in instruments]},
                created_at=now,
            )
        )
        return {
            "run_id": run_id,
            "instrument_ids": [instrument.instrument_id for instrument in instruments],
            "artifact_id": artifact_id,
        }

    def run_dummy_analysis_tool(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        run_date = _run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        now = _utc_now()
        artifact_id = f"artifact-dummy-analysis-{_stable_digest(run_id)}"
        tool_run_id = f"tool-dummy-analysis-{run_id}"
        payload: JsonObject = {
            "schema_version": "dummy-analysis.v1",
            "run_id": run_id,
            "records": [
                {
                    "symbol": symbol.upper(),
                    "signal": "mixed",
                    "summary": "Dummy analysis marks live-search evidence as the only real signal.",
                    "confidence": 0.3,
                }
            ],
        }
        path = paths.audit_dir / "dummy-analysis.json"
        sha256 = write_json_artifact(path, payload)
        store = self.store
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name="run_dummy_analysis_tool",
                tool_version="phase2.v1",
                status="ok",
                started_at=now,
                completed_at=now,
                inputs={"symbol": symbol},
            )
        )
        store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                tool_run_id=tool_run_id,
                artifact_type="analysis_context",
                path=path.relative_to(self.repo_root),
                sha256=sha256,
                schema_version="dummy-analysis.v1",
                metadata={"symbol": symbol.upper()},
                created_at=now,
            )
        )
        return {"run_id": run_id, "artifact_id": artifact_id}

    def synthesize_prediction_candidates(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        run_date = _run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        now = _utc_now()
        store = self.store
        evidence = store.list_evidence_for_run(run_id)
        instrument_id = f"instrument:codex:{symbol.upper()}"
        if not store.get_instrument(instrument_id):
            self.run_dummy_universe_tool(run_id=run_id, symbol=symbol)
        evidence_ids = tuple(record.evidence_id for record in evidence)
        candidate_id = f"candidate-{symbol.lower()}-{_stable_digest(run_id)[:8]}"
        scenario = (
            f"Live-search evidence for {symbol.upper()} is sufficient for a monitored "
            "prediction scenario, but dummy tools keep confidence conservative."
            if evidence_ids
            else f"Insufficient evidence for {symbol.upper()} after dummy tool execution."
        )
        candidate = PredictionCandidateRecord(
            candidate_id=candidate_id,
            run_id=run_id,
            instrument_id=instrument_id,
            prediction_horizon=TimeHorizon.SWING.value,
            prediction_type="scenario",
            scenario=scenario,
            direction=Direction.MIXED.value,
            confidence=0.36 if evidence_ids else None,
            status="moderate_confidence" if evidence_ids else "insufficient_evidence",
            evidence_for=evidence_ids[:3],
            baseline={"comparison": "no directional edge"},
            uncertainty="Dummy tools are structural validation only.",
            metadata={"phase2_mcp": True},
        )
        payload: JsonObject = {
            "schema_version": "phase2-candidate-synthesis.v1",
            "run_id": run_id,
            "records": [
                {
                    "candidate_id": candidate.candidate_id,
                    "scenario": candidate.scenario,
                    "evidence_for": list(candidate.evidence_for),
                }
            ],
        }
        artifact_id = f"artifact-prediction-inputs-{_stable_digest(run_id)}"
        tool_run_id = f"tool-candidate-synthesis-{run_id}"
        path = paths.audit_dir / "prediction-inputs.json"
        sha256 = write_json_artifact(path, payload)
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name="synthesize_prediction_candidates",
                tool_version="phase2.v1",
                status="ok",
                started_at=now,
                completed_at=now,
                inputs={"symbol": symbol, "evidence_count": len(evidence)},
            )
        )
        store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                tool_run_id=tool_run_id,
                artifact_type="prediction_input",
                path=path.relative_to(self.repo_root),
                sha256=sha256,
                schema_version="phase2-candidate-synthesis.v1",
                metadata={"candidate_id": candidate_id},
                created_at=now,
            )
        )
        store.upsert_prediction_candidate(candidate)
        for evidence_id in candidate.evidence_for:
            store.link_candidate_evidence(
                CandidateEvidenceLinkRecord(
                    candidate_id=candidate_id,
                    evidence_id=evidence_id,
                    relationship="supports",
                    metadata={"source": "phase2_mcp_synthesis"},
                    created_at=now,
                )
            )
        store.link_candidate_artifact(
            CandidateArtifactLinkRecord(
                candidate_id=candidate_id,
                artifact_id=artifact_id,
                relationship="prediction_input",
                metadata={"source": "phase2_mcp_synthesis"},
                created_at=now,
            )
        )
        return {"run_id": run_id, "candidate_id": candidate_id, "artifact_id": artifact_id}

    def render_prediction_report(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        run_date = _run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        now = _utc_now()
        store = self.store
        candidates = store.list_prediction_candidates_for_run(run_id)
        if not candidates:
            self.synthesize_prediction_candidates(run_id=run_id, symbol=symbol)
            candidates = store.list_prediction_candidates_for_run(run_id)
        evidence_records = store.list_evidence_for_run(run_id)
        evidence_sources = tuple(
            _source_evidence_from_record(record) for record in evidence_records
        )
        instrument = _instrument(symbol.upper(), now)
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
            _audit_artifact_from_record(record, self.repo_root)
            for record in store.list_artifacts_for_run(run_id)
        )
        report = DailyReport(
            schema_version="daily-report.v2",
            run_id=run_id,
            report_date=run_date,
            generated_at=now,
            timezone="UTC",
            objective=run.objective,
            universe=f"Phase 2 Codex smoke universe for {symbol.upper()}",
            command_args={"run_id": run_id, "symbol": symbol.upper()},
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
            insufficient_evidence_summary=(
                None if prediction_candidates else "No candidate could be synthesized."
            ),
            audit_manifest=AuditManifest(
                run_id=run_id,
                schema_version="audit-manifest.v2",
                created_at=now,
                artifacts=audit_artifacts,
                command_args={"run_id": run_id, "symbol": symbol.upper()},
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
        report_sha = _file_sha256(paths.report_path)
        json_sha = _file_sha256(paths.json_path)
        report_tool_run_id = f"tool-render-report-{run_id}"
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=report_tool_run_id,
                run_id=run_id,
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
                f"artifact-report-md-{_stable_digest(run_id)}",
                "markdown_report",
                paths.report_path,
                report_sha,
            ),
            (
                f"artifact-report-json-{_stable_digest(run_id)}",
                "json_report",
                paths.json_path,
                json_sha,
            ),
            (
                f"artifact-audit-manifest-{_stable_digest(run_id)}",
                "provider_result",
                paths.audit_manifest_path,
                manifest_sha,
            ),
        ):
            store.record_artifact(
                ArtifactRecord(
                    artifact_id=artifact_id,
                    tool_run_id=report_tool_run_id,
                    artifact_type=artifact_type,
                    path=path.relative_to(self.repo_root),
                    sha256=sha256,
                    schema_version="phase2-report.v1",
                    metadata={"run_id": run_id},
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
            "run_id": run_id,
            "markdown_path": paths.report_path.as_posix(),
            "json_path": paths.json_path.as_posix(),
            "audit_manifest_path": paths.audit_manifest_path.as_posix(),
        }

    def inspect_research_run(self, *, run_id: str) -> JsonObject:
        run = self._require_run(run_id)
        store = self.store
        return {
            "run_id": run.run_id,
            "status": run.status,
            "tool_run_count": len(store.list_tool_runs_for_run(run_id)),
            "artifact_count": len(store.list_artifacts_for_run(run_id)),
            "source_query_count": len(store.list_source_queries_for_run(run_id)),
            "evidence_count": len(store.list_evidence_for_run(run_id)),
            "candidate_count": len(store.list_prediction_candidates_for_run(run_id)),
            "has_codex_search_evidence": any(
                record.provider == "codex-web-search"
                for record in store.list_evidence_for_run(run_id)
            ),
        }

    def _require_run(self, run_id: str) -> ResearchRunRecord:
        run = self.store.get_research_run(run_id)
        if run is None:
            raise ValueError(f"research run does not exist: {run_id}")
        return run

    def _paths(self, run_date: date, output_dir: str) -> Phase2RunPaths:
        output_path = self._resolve_write_path(Path(output_dir))
        run_dir = output_path / run_date.isoformat()
        audit_dir = run_dir / "audit"
        return Phase2RunPaths(
            output_dir=output_path,
            run_dir=run_dir,
            audit_dir=audit_dir,
            report_path=run_dir / "report.md",
            json_path=run_dir / "report.json",
            audit_manifest_path=audit_dir / "audit-manifest.json",
        )

    def _resolve_write_path(self, path: Path) -> Path:
        resolved = path if path.is_absolute() else self.repo_root / path
        resolved = resolved.resolve()
        allowed_roots = tuple((self.repo_root / root).resolve() for root in ALLOWED_WRITE_ROOTS)
        if not any(_is_relative_to(resolved, root) or resolved == root for root in allowed_roots):
            roots = ", ".join(root.as_posix() for root in allowed_roots)
            raise ValueError(f"Phase 2 MCP writes are limited to: {roots}")
        return resolved


def _instrument(symbol: str, retrieved_at: datetime) -> Instrument:
    instrument_id = f"instrument:codex:{symbol.upper()}"
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        display_name=f"{symbol.upper()} Codex smoke instrument",
        asset_class=AssetClass.STOCK,
        provider_ids=(
            ProviderInstrumentId(
                provider="codex-smoke",
                identifier=symbol.upper(),
                namespace="symbol",
            ),
        ),
        tradability_evidence=(
            TradabilityEvidence(
                provider="codex-smoke",
                status=TradabilityStatus.UNKNOWN,
                retrieved_at=retrieved_at,
                raw_identifier=f"{symbol.upper()}:codex-smoke",
                notes="Smoke records researchability only, not trading availability.",
            ),
        ),
        data_availability=(
            InstrumentDataAvailability(
                provider="codex-smoke",
                data_type="dummy_tool_context",
                status=TradabilityStatus.AVAILABLE,
                checked_at=retrieved_at,
                provider_identifier=symbol.upper(),
            ),
        ),
        metadata={"phase2_mcp": True},
    )


def _source_evidence_from_record(record: EvidenceRecord) -> SourceEvidence:
    source_evidence = record.metadata.get("source_evidence")
    if isinstance(source_evidence, dict):
        return SourceEvidence.model_validate(source_evidence)
    provenance = SourceProvenance.model_validate(record.provenance_json)
    ticker = record.instruments[0].rsplit(":", 1)[-1] if record.instruments else None
    return SourceEvidence(
        evidence_id=record.evidence_id,
        source_kind=SourceKind(record.source_type),
        ticker=ticker,
        text=record.claim,
        created_at=record.published_at,
        permalink=record.url,
        matched_tickers=((ticker,) if ticker else ()),
        provenance=provenance,
        metadata=record.metadata,
    )


def _report_candidate(
    candidate: PredictionCandidateRecord,
    evidence_sources: tuple[SourceEvidence, ...],
) -> PredictionCandidate:
    evidence_by_id = {record.evidence_id: record for record in evidence_sources}
    evidence_refs = tuple(
        EvidenceReference(
            evidence_id=evidence_id,
            quote=evidence_by_id[evidence_id].text[:180] if evidence_id in evidence_by_id else None,
            relevance=0.76,
        )
        for evidence_id in candidate.evidence_for
    )
    status = (
        PredictionStatus.EVIDENCE_SUPPORTED
        if evidence_refs
        else PredictionStatus.INSUFFICIENT_EVIDENCE
    )
    return PredictionCandidate(
        candidate_id=candidate.candidate_id,
        instrument_id=candidate.instrument_id,
        symbol=candidate.instrument_id.rsplit(":", 1)[-1],
        horizon=TimeHorizon.SWING,
        direction=Direction.MIXED,
        status=status,
        thesis=candidate.scenario,
        baseline="No directional edge is assumed; dummy tools only validate orchestration.",
        confidence=candidate.confidence or 0.0,
        evidence_for=evidence_refs,
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
    }:
        artifact_type = "provider_result"
    return AuditArtifact(
        artifact_id=record.artifact_id,
        artifact_type=cast(ArtifactType, artifact_type),
        path=path.as_posix(),
        created_at=record.created_at or _utc_now(),
        produced_by="phase2-mcp",
        sha256=record.sha256,
        metadata=record.metadata,
    )


def _run_id(run_date: date, symbol: str) -> str:
    return f"codex-smoke-{run_date.isoformat()}-{symbol.lower().replace('/', '-')}"


def _run_date_from_run(run: ResearchRunRecord) -> date:
    raw = run.metadata.get("run_date")
    if isinstance(raw, str):
        return date.fromisoformat(raw)
    return run.started_at.date()


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = ["ALLOWED_WRITE_ROOTS", "Phase2McpService", "Phase2RunPaths"]
