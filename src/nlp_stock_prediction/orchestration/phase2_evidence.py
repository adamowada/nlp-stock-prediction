"""Codex search-evidence import for Phase 2 smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts import (
    FreshnessStatus,
    JsonObject,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
)
from nlp_stock_prediction.orchestration.phase2_common import (
    Phase2RunPaths,
    normalize_evidence_stance,
    parse_optional_datetime,
    source_evidence_ticker,
    stable_digest,
    utc_now,
)
from nlp_stock_prediction.reporting.audit import write_json_artifact
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    EvidenceRecord,
    SourceQueryRecord,
    SQLiteStore,
    ToolRunRecord,
)


def record_codex_search_evidence(
    *,
    store: SQLiteStore,
    repo_root: Path,
    paths: Phase2RunPaths,
    run_id: str,
    symbol: str,
    title: str,
    url: str,
    claim: str,
    query: str,
    published_at: str | None = None,
    stance: str | None = None,
) -> JsonObject:
    now = utc_now()
    normalized_symbol = symbol.strip().upper()
    normalized_stance = normalize_evidence_stance(stance, claim)
    digest = stable_digest("|".join([run_id, normalized_symbol, title, url, claim]))
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
        observed_at=parse_optional_datetime(published_at) or now,
        source_url=url,
        permalink=url,
        raw_identifier=url,
        raw_snapshot_id=artifact_id,
        query=query,
        freshness_status=FreshnessStatus.FRESH,
        provider_metadata={"codex_search": True},
    )
    source_ticker = source_evidence_ticker(normalized_symbol)
    evidence = SourceEvidence(
        evidence_id=evidence_id,
        source_kind=source_kind,
        ticker=source_ticker,
        title=title,
        text=claim,
        created_at=parse_optional_datetime(published_at),
        permalink=url,
        matched_tickers=((source_ticker,) if source_ticker else ()),
        provenance=provenance,
        metadata={"codex_search": True, "stance": normalized_stance},
    )
    payload: JsonObject = {
        "schema_version": "codex-search-evidence.v1",
        "run_id": run_id,
        "records": [cast(JsonObject, evidence.model_dump(mode="json"))],
    }
    sha256 = write_json_artifact(artifact_path, payload)
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name="record_codex_search_evidence",
            tool_version="phase2.v1",
            status="ok",
            inputs={
                "symbol": normalized_symbol,
                "url": url,
                "query": query,
                "stance": normalized_stance,
            },
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
            path=artifact_path.relative_to(repo_root),
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
            published_at=parse_optional_datetime(published_at),
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
                "stance": normalized_stance,
                "source_evidence": cast(JsonObject, evidence.model_dump(mode="json")),
            },
        )
    )
    return {"run_id": run_id, "evidence_id": evidence_id, "artifact_id": artifact_id}


def source_evidence_from_record(record: EvidenceRecord) -> SourceEvidence:
    source_evidence = record.metadata.get("source_evidence")
    if isinstance(source_evidence, dict):
        return SourceEvidence.model_validate(source_evidence)
    provenance = SourceProvenance.model_validate(record.provenance_json)
    raw_ticker = (
        record.instruments[0].removeprefix("instrument:codex:") if record.instruments else None
    )
    ticker = source_evidence_ticker(raw_ticker) if raw_ticker else None
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


def evidence_stance_from_record(record: EvidenceRecord) -> str:
    stance = record.metadata.get("stance")
    if isinstance(stance, str) and stance in {"supports", "contradicts", "neutral"}:
        return stance
    source_evidence = record.metadata.get("source_evidence")
    if isinstance(source_evidence, dict):
        metadata = source_evidence.get("metadata")
        if isinstance(metadata, dict):
            nested_stance = metadata.get("stance")
            if isinstance(nested_stance, str) and nested_stance in {
                "supports",
                "contradicts",
                "neutral",
            }:
                return nested_stance
    return normalize_evidence_stance(None, record.claim)


__all__ = [
    "evidence_stance_from_record",
    "record_codex_search_evidence",
    "source_evidence_from_record",
]
