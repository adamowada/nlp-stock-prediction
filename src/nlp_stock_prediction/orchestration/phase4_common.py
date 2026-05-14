"""Shared helpers for Phase 4 evidence and analysis tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import (
    FreshnessStatus,
    ProviderStatus,
    RetrievalMethod,
    SourceKind,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.provenance import (
    EvidenceReference,
    ProviderWarning,
    SourceProvenance,
)
from nlp_stock_prediction.contracts.providers import (
    ProviderMetric,
    ProviderRequest,
    ProviderResult,
)
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex, ArtifactType
from nlp_stock_prediction.orchestration.phase2_common import (
    source_evidence_ticker,
    stable_digest,
)
from nlp_stock_prediction.storage.records import (
    EvidenceRecord,
    SourceQueryRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE4_TOOL_VERSION = "phase4.evidence-suite.v1"


@dataclass(frozen=True)
class Phase4ToolResult:
    """Stable summary returned by Phase 4 tools after SQLite/artifact writes."""

    run_id: str
    tool_run_id: str
    artifact_id: str
    artifact_path: Path
    status: str
    evidence_ids: tuple[str, ...] = ()
    source_query_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    artifact_payload: JsonObject | None = None


def tool_identity(
    *, tool_slug: str, run_id: str, symbol: str | None = None
) -> tuple[str, str, str]:
    """Return stable tool-run, artifact, and filename components."""

    digest = stable_digest("|".join((tool_slug, run_id, symbol or "all")))
    return (
        f"tool-{tool_slug}-{digest}",
        f"artifact-{tool_slug}-{digest}",
        f"{tool_slug}-{digest}.json",
    )


def record_tool_started(
    *,
    store: SQLiteStore,
    tool_run_id: str,
    run_id: str,
    tool_name: str,
    started_at: datetime,
    inputs: JsonObject,
) -> None:
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=tool_name,
            tool_version=PHASE4_TOOL_VERSION,
            status="running",
            started_at=started_at,
            inputs=inputs,
        )
    )


def record_tool_completed(
    *,
    store: SQLiteStore,
    tool_run_id: str,
    run_id: str,
    tool_name: str,
    started_at: datetime,
    completed_at: datetime,
    status: str,
    inputs: JsonObject,
    warnings: Sequence[str] = (),
    error_message: str | None = None,
) -> None:
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=tool_name,
            tool_version=PHASE4_TOOL_VERSION,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            inputs=inputs,
            warnings=tuple(warnings),
            error_message=error_message,
        )
    )


def write_phase4_json_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    created_at: datetime,
    produced_by: str,
    tool_run_id: str,
    schema_version: str,
    artifact_id: str,
    artifact_type: ArtifactType,
    filename: str,
    payload: JsonObject,
    record_count: int | None,
    metadata: JsonObject,
) -> Path:
    artifact = ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=artifact_dir,
        created_at=created_at,
        produced_by=produced_by,
        tool_run_id=tool_run_id,
        schema_version=schema_version,
    ).write_json(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        filename=filename,
        payload=payload,
        record_count=record_count,
        metadata=metadata,
    )
    return Path(artifact.path)


def record_source_query_for_result(
    *,
    store: SQLiteStore,
    tool_slug: str,
    tool_run_id: str,
    result: ProviderResult[object],
    source_url: str | None = None,
) -> SourceQueryRecord:
    query = provider_request_query(result.request)
    url = source_url or source_url_from_result(result)
    source_query_id = (
        f"query-{tool_slug}-"
        f"{stable_digest('|'.join((tool_run_id, result.provider_name, query, url or '')))}"
    )
    record = SourceQueryRecord(
        source_query_id=source_query_id,
        tool_run_id=tool_run_id,
        provider=result.provider_name,
        query=query,
        url=url,
        retrieved_at=result.fetched_at,
        metadata=cast(
            JsonObject,
            {
                "phase4_tool": tool_slug,
                "provider_status": result.status.value,
                "request": model_json(result.request),
                "raw_snapshot_id": result.raw_snapshot_id,
                "cache_key": result.cache_key,
                "warnings": warning_payloads(result.warnings),
            },
        ),
    )
    store.record_source_query(record)
    return record


def provider_request_query(request: ProviderRequest) -> str:
    """Build a non-empty source-query label from a provider request."""

    if request.query:
        return request.query
    if request.tickers:
        return ",".join(request.tickers)
    series_ids = getattr(request, "series_ids", ())
    if isinstance(series_ids, tuple) and series_ids:
        return ",".join(str(series_id) for series_id in series_ids)
    return request.request_id


def source_url_from_result(result: ProviderResult[object]) -> str | None:
    data = result.data
    if isinstance(data, tuple):
        for item in data:
            if isinstance(item, SourceEvidence):
                source_url = item.provenance.source_url or item.provenance.permalink
                if source_url:
                    return source_url
    for warning in result.warnings:
        if warning.source_url:
            return warning.source_url
    return None


def scoped_source_evidence(
    evidence: SourceEvidence,
    *,
    run_id: str,
    tool_slug: str,
) -> SourceEvidence:
    """Scope provider evidence IDs to a run while preserving the provider ID in metadata."""

    evidence_id = f"evidence-{tool_slug}-{stable_digest('|'.join((run_id, evidence.evidence_id)))}"
    payload = evidence.model_dump(mode="python")
    raw_metadata = payload.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    metadata["provider_evidence_id"] = evidence.evidence_id
    metadata["phase4_tool"] = tool_slug
    payload["evidence_id"] = evidence_id
    payload["metadata"] = metadata
    return SourceEvidence.model_validate(payload)


def evidence_record_from_source(
    evidence: SourceEvidence,
    *,
    tool_run_id: str,
    source_query_id: str,
    artifact_id: str,
    fallback_instrument_id: str | None = None,
    derived_analysis: JsonObject | None = None,
) -> EvidenceRecord:
    provenance = evidence.provenance
    extraction_confidence = confidence_from_metadata(evidence.metadata) or 0.75
    source_reliability = text_from_metadata(evidence.metadata, "source_reliability")
    metadata: JsonObject = {
        "phase4_source_record": True,
        "source_evidence": model_json(evidence),
    }
    if derived_analysis is not None:
        metadata["derived_analysis"] = derived_analysis
    return EvidenceRecord(
        evidence_id=evidence.evidence_id,
        tool_run_id=tool_run_id,
        source_query_id=source_query_id,
        source_type=evidence.source_kind.value,
        provider=provenance.provider_name,
        url=provenance.permalink or provenance.source_url or evidence.permalink,
        query=provenance.query,
        retrieved_at=provenance.fetched_at,
        published_at=evidence.created_at or provenance.observed_at,
        instruments=instruments_from_evidence(evidence, fallback_instrument_id),
        claim=evidence.text,
        extraction_confidence=extraction_confidence,
        source_reliability=source_reliability or "provider_normalized",
        freshness_status=provenance.freshness_status.value,
        artifact_id=artifact_id,
        raw_excerpt=excerpt(evidence.text),
        provenance_json=model_json(provenance),
        metadata=metadata,
    )


def metric_source_evidence(
    *,
    run_id: str,
    tool_slug: str,
    fetched_at: datetime,
    provider_name: str,
    source_query_id: str,
    query: str | None = None,
    raw_snapshot_id: str | None,
    cache_key: str | None,
    symbol: str | None,
    metric: ProviderMetric,
    source_kind: SourceKind,
    retrieval_method: RetrievalMethod,
    freshness_status: FreshnessStatus,
    index: int,
) -> SourceEvidence:
    raw_identifier = metric_raw_identifier(
        provider_name=provider_name,
        source_query_id=source_query_id,
        metric=metric,
        index=index,
    )
    observed_at = aware_datetime_from_metric_as_of(metric.as_of)
    source_url = metric_source_url(provider_name, metric, raw_identifier)
    ticker = source_evidence_ticker(symbol or "") if symbol else None
    evidence_id = (
        f"evidence-{tool_slug}-"
        f"{stable_digest('|'.join((run_id, provider_name, source_query_id, raw_identifier)))}"
    )
    claim = metric_claim(symbol=symbol, metric=metric)
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=source_kind,
        ticker=ticker,
        title=f"{provider_name} {metric.name}",
        text=claim,
        created_at=observed_at,
        permalink=source_url,
        matched_tickers=((ticker,) if ticker else ()),
        provenance=SourceProvenance(
            provider_name=provider_name,
            source_kind=source_kind,
            retrieval_method=retrieval_method,
            fetched_at=fetched_at,
            observed_at=observed_at,
            source_url=source_url,
            permalink=source_url,
            raw_identifier=raw_identifier,
            raw_snapshot_id=raw_snapshot_id or f"metric:{stable_digest(raw_identifier)}",
            query=query or source_query_id,
            cache_key=cache_key,
            freshness_status=freshness_status,
            provider_metadata={
                "phase4_tool": tool_slug,
                "metric": model_json(metric),
                "source_query_id": source_query_id,
            },
        ),
        metadata={
            "phase4_tool": tool_slug,
            "provider_metric": model_json(metric),
            "source_query_id": source_query_id,
        },
    )


def metric_record_from_evidence(
    evidence: SourceEvidence,
    *,
    tool_run_id: str,
    source_query_id: str,
    artifact_id: str,
    fallback_instrument_id: str | None = None,
    analysis_reference: JsonObject | None = None,
) -> EvidenceRecord:
    metadata: JsonObject = {
        "phase4_metric_record": True,
        "source_evidence": model_json(evidence),
    }
    if analysis_reference is not None:
        metadata["analysis_reference"] = analysis_reference
    return EvidenceRecord(
        evidence_id=evidence.evidence_id,
        tool_run_id=tool_run_id,
        source_query_id=source_query_id,
        source_type=evidence.source_kind.value,
        provider=evidence.provenance.provider_name,
        url=evidence.provenance.source_url,
        query=evidence.provenance.query,
        retrieved_at=evidence.provenance.fetched_at,
        published_at=evidence.created_at,
        instruments=instruments_from_evidence(evidence, fallback_instrument_id),
        claim=evidence.text,
        extraction_confidence=0.9,
        source_reliability="provider_metric",
        freshness_status=evidence.provenance.freshness_status.value,
        artifact_id=artifact_id,
        raw_excerpt=excerpt(evidence.text),
        provenance_json=model_json(evidence.provenance),
        metadata=metadata,
    )


def evidence_reference(evidence: SourceEvidence, *, relevance: float = 1.0) -> EvidenceReference:
    quote = excerpt(evidence.text, limit=180)
    return EvidenceReference(
        evidence_id=evidence.evidence_id,
        quote=quote,
        start_char=0,
        end_char=len(quote),
        relevance=relevance,
    )


def derived_label_for_text(
    *,
    evidence: SourceEvidence,
    catalysts: Sequence[str] = (),
) -> JsonObject:
    stance = heuristic_stance(evidence.text)
    return cast(
        JsonObject,
        {
            "evidence_id": evidence.evidence_id,
            "stance": stance,
            "catalysts": list(catalysts),
            "analysis_type": "derived_label",
            "evidence_reference": model_json(evidence_reference(evidence, relevance=0.7)),
        },
    )


def heuristic_stance(text: str) -> str:
    normalized = text.lower()
    if "expensive" in normalized:
        return "contradicts"
    supportive = (
        "advance",
        "beat",
        "bull",
        "call spread",
        "calls",
        "catalyst",
        "growth",
        "lead",
        "momentum",
        "rally",
        "rise",
        "rose",
        "strong",
        "support",
    )
    conflicting = (
        "bear",
        "decline",
        "down",
        "expensive",
        "fall",
        "miss",
        "pressure",
        "risk",
        "selloff",
        "stale",
        "weak",
    )
    support_count = sum(1 for token in supportive if token in normalized)
    conflict_count = sum(1 for token in conflicting if token in normalized)
    if support_count > conflict_count:
        return "supports"
    if conflict_count > support_count:
        return "contradicts"
    return "neutral"


def catalyst_labels(text: str) -> tuple[str, ...]:
    normalized = text.lower()
    labels: list[str] = []
    for token, label in (
        ("earnings", "earnings"),
        ("delivery", "deliveries"),
        ("deliveries", "deliveries"),
        ("filing", "filing"),
        ("inflation", "macro"),
        ("product update", "product_update"),
        ("rate", "macro"),
        ("robotaxi", "robotaxi"),
        ("volatility", "volatility"),
    ):
        if token in normalized and label not in labels:
            labels.append(label)
    return tuple(labels)


def provider_result_payload(result: ProviderResult[object]) -> JsonObject:
    return cast(
        JsonObject,
        {
            "provider_name": result.provider_name,
            "status": result.status.value,
            "fetched_at": result.fetched_at.isoformat(),
            "request": model_json(result.request),
            "warnings": warning_payloads(result.warnings),
            "raw_snapshot_id": result.raw_snapshot_id,
            "cache_key": result.cache_key,
            "health": model_json(result.health),
        },
    )


def warning_payloads(warnings: Sequence[ProviderWarning]) -> list[JsonObject]:
    return [model_json(warning) for warning in warnings]


def warning_messages(warnings: Sequence[ProviderWarning]) -> tuple[str, ...]:
    return tuple(f"{warning.code.value}: {warning.message}" for warning in warnings)


def tool_status(*, record_count: int, warnings: Sequence[str]) -> str:
    if record_count and warnings:
        return "partial"
    if record_count:
        return "ok"
    return "warning" if warnings else "empty"


def provider_metric_freshness(
    provider_status: ProviderStatus,
    metric: ProviderMetric,
) -> FreshnessStatus:
    if provider_status == ProviderStatus.STALE:
        return FreshnessStatus.STALE
    if metric.as_of is None:
        return FreshnessStatus.MISSING
    return FreshnessStatus.FRESH


def retrieval_method_for_provider(provider_name: str) -> RetrievalMethod:
    normalized = provider_name.lower()
    if "fixture" in normalized:
        return RetrievalMethod.FIXTURE
    if normalized in {"fred", "sec-edgar", "alpha-vantage-fundamentals", "x-recent-search"}:
        return RetrievalMethod.OFFICIAL_API
    if normalized in {"reddit", "ap-news"}:
        return RetrievalMethod.PUBLIC_SCRAPE
    return RetrievalMethod.DERIVED


def source_kind_for_fundamental_metric(provider_name: str, metric: ProviderMetric) -> SourceKind:
    if provider_name == "sec-edgar" and metric.name.startswith("sec_recent_filing_"):
        return SourceKind.SEC_FILING
    return SourceKind.FUNDAMENTAL_DATA


def model_json(model: BaseModel) -> JsonObject:
    return cast(JsonObject, model.model_dump(mode="json"))


def result_list_json(results: Sequence[ProviderResult[object]]) -> list[JsonObject]:
    return [provider_result_payload(result) for result in results]


def source_evidence_json(records: Sequence[SourceEvidence]) -> list[JsonObject]:
    return [model_json(record) for record in records]


def instruments_from_evidence(
    evidence: SourceEvidence,
    fallback_instrument_id: str | None,
) -> tuple[str, ...]:
    if evidence.matched_instrument_ids:
        return tuple(evidence.matched_instrument_ids)
    if evidence.instrument_id:
        return (evidence.instrument_id,)
    if fallback_instrument_id:
        return (fallback_instrument_id,)
    if evidence.matched_tickers:
        return tuple(f"instrument:codex:{ticker}" for ticker in evidence.matched_tickers)
    if evidence.ticker:
        return (f"instrument:codex:{evidence.ticker}",)
    return ()


def confidence_from_metadata(metadata: JsonObject) -> float | None:
    value = metadata.get("extraction_confidence")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def text_from_metadata(metadata: JsonObject, key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def excerpt(text: str, *, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit]


def aware_datetime_from_metric_as_of(value: date | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def metric_raw_identifier(
    *,
    provider_name: str,
    source_query_id: str,
    metric: ProviderMetric,
    index: int,
) -> str:
    metadata_identifier = text_from_metadata(metric.metadata, "accession_number")
    if metadata_identifier:
        return f"{provider_name}:{metadata_identifier}:{metric.name}"
    as_of = metric.as_of.isoformat() if metric.as_of is not None else "missing-as-of"
    return f"{provider_name}:{source_query_id}:{metric.name}:{as_of}:{index}"


def metric_source_url(
    provider_name: str,
    metric: ProviderMetric,
    raw_identifier: str,
) -> str:
    source_url = text_from_metadata(metric.metadata, "source_url")
    if source_url:
        return source_url
    return f"provider://{provider_name}/{stable_digest(raw_identifier)}"


def metric_claim(*, symbol: str | None, metric: ProviderMetric) -> str:
    subject = symbol.upper() if symbol else "macro series"
    value = metric_value_text(metric.value)
    as_of = metric.as_of.isoformat() if metric.as_of is not None else "unknown date"
    unit = f" {metric.unit}" if metric.unit else ""
    return f"{subject} provider metric {metric.name} was {value}{unit} as of {as_of}."


def metric_value_text(value: Decimal | float | int | str | None) -> str:
    if value is None:
        return "null"
    if isinstance(value, Decimal):
        return f"{value:g}"
    return str(value)


def dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "PHASE4_TOOL_VERSION",
    "Phase4ToolResult",
    "catalyst_labels",
    "dedupe_strings",
    "derived_label_for_text",
    "evidence_record_from_source",
    "evidence_reference",
    "metric_record_from_evidence",
    "metric_source_evidence",
    "model_json",
    "provider_metric_freshness",
    "provider_request_query",
    "record_source_query_for_result",
    "record_tool_completed",
    "record_tool_started",
    "result_list_json",
    "retrieval_method_for_provider",
    "scoped_source_evidence",
    "source_evidence_json",
    "source_kind_for_fundamental_metric",
    "tool_identity",
    "tool_status",
    "warning_messages",
    "warning_payloads",
    "write_phase4_json_artifact",
]
