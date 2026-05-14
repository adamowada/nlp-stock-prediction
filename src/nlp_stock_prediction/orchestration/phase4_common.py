"""Shared helpers for Phase 4 evidence and analysis tools."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
from nlp_stock_prediction.orchestration.phase2_common import (
    source_evidence_ticker,
    stable_digest,
)
from nlp_stock_prediction.orchestration.phase4_execution import (
    PHASE4_RUNNING_TOOL_RUN_STATUS,
    PHASE4_TOOL_VERSION,
    Phase4StoredToolRunStatus,
    record_tool_completed,
    record_tool_started,
    safe_phase4_tool_execution,
    standardize_phase4_tool_run_status,
    write_phase4_json_artifact,
)
from nlp_stock_prediction.orchestration.report_data_modes import (
    report_data_mode_metadata_for_run_id,
)
from nlp_stock_prediction.storage.records import (
    EvidenceRecord,
    SourceQueryRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore


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
    *,
    tool_slug: str,
    run_id: str,
    symbol: str | None = None,
    inputs: JsonObject | None = None,
) -> tuple[str, str, str]:
    """Return stable tool-run, artifact, and filename components."""

    input_fingerprint = "" if inputs is None else repr(sorted(inputs.items()))
    digest = stable_digest("|".join((tool_slug, run_id, symbol or "all", input_fingerprint)))
    return (
        f"tool-{tool_slug}-{digest}",
        f"artifact-{tool_slug}-{digest}",
        f"{tool_slug}-{digest}.json",
    )


def record_source_query_for_result(
    *,
    store: SQLiteStore,
    tool_slug: str,
    tool_run_id: str,
    result: ProviderResult[object],
    source_url: str | None = None,
) -> SourceQueryRecord:
    mode_metadata = _mode_metadata_for_tool_run(store, tool_run_id)
    query = provider_request_query(result.request)
    query_urls = source_query_urls_from_result(result)
    url = (
        sanitize_source_query_url(source_url)
        if source_url
        else (query_urls[0] if query_urls else None)
    )
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
                "source_query_urls": query_urls,
                "evidence_source_urls": evidence_source_urls_from_result(result),
                **mode_metadata,
            },
        ),
    )
    store.record_source_query(record)
    return record


def _mode_metadata_for_tool_run(store: SQLiteStore, tool_run_id: str) -> JsonObject:
    tool_run = store.get_tool_run(tool_run_id)
    if tool_run is None or tool_run.run_id is None:
        return {}
    return report_data_mode_metadata_for_run_id(store, tool_run.run_id)


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


def source_query_urls_from_result(result: ProviderResult[object]) -> tuple[str, ...]:
    """Return provider-query URLs, never article/permalink URLs masquerading as queries."""

    explicit_urls = _urls_from_request_options(result.request.options)
    if explicit_urls:
        return explicit_urls
    provider_name = result.provider_name.lower()
    if provider_name == "fred":
        return _fred_query_urls(result)
    if provider_name == "sec-edgar":
        return _sec_edgar_query_urls(result)
    inferred_urls = _provider_query_urls_from_evidence(result)
    if inferred_urls:
        return inferred_urls
    warning_urls = tuple(
        sanitize_source_query_url(warning.source_url)
        for warning in result.warnings
        if warning.source_url
    )
    return dedupe_strings(tuple(url for url in warning_urls if url))


def evidence_source_urls_from_result(result: ProviderResult[object]) -> tuple[str, ...]:
    data = result.data
    urls: list[str] = []
    if isinstance(data, tuple):
        for item in data:
            if isinstance(item, SourceEvidence):
                source_url = item.provenance.source_url or item.provenance.permalink
                if source_url:
                    sanitized = sanitize_source_query_url(source_url)
                    if sanitized:
                        urls.append(sanitized)
    return dedupe_strings(tuple(urls))


def sanitize_source_query_url(url: str | None) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    split = urlsplit(url.strip())
    sensitive = {"api_key", "apikey", "token", "access_token", "key", "bearer"}
    netloc = split.hostname or ""
    if split.port is not None:
        netloc = f"{netloc}:{split.port}"
    query = urlencode(
        [
            (key, "REDACTED" if key.lower() in sensitive else value)
            for key, value in parse_qsl(split.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((split.scheme, netloc, _sanitize_url_path(split.path), query, split.fragment))


def _urls_from_request_options(options: JsonObject) -> tuple[str, ...]:
    raw_urls: list[object] = []
    for key in ("source_query_url", "query_url"):
        value = options.get(key)
        if isinstance(value, str):
            raw_urls.append(value)
    for key in ("source_query_urls", "query_urls"):
        value = options.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            raw_urls.extend(value)
    return dedupe_strings(
        tuple(
            sanitized
            for raw_url in raw_urls
            if isinstance(raw_url, str)
            for sanitized in (sanitize_source_query_url(raw_url),)
            if sanitized
        )
    )


def _provider_query_urls_from_evidence(result: ProviderResult[object]) -> tuple[str, ...]:
    data = result.data
    if not isinstance(data, tuple):
        return ()
    urls: list[str] = []
    for item in data:
        if not isinstance(item, SourceEvidence):
            continue
        source_url = item.provenance.source_url
        permalink = item.provenance.permalink or item.permalink
        if source_url and source_url != permalink:
            sanitized = sanitize_source_query_url(source_url)
            if sanitized:
                urls.append(sanitized)
        for key in ("source_query_url", "query_url", "hub_url"):
            metadata_url = item.provenance.provider_metadata.get(key)
            if isinstance(metadata_url, str):
                sanitized = sanitize_source_query_url(metadata_url)
                if sanitized:
                    urls.append(sanitized)
    return dedupe_strings(tuple(urls))


def _fred_query_urls(result: ProviderResult[object]) -> tuple[str, ...]:
    series_ids = getattr(result.request, "series_ids", ())
    if not isinstance(series_ids, tuple):
        return ()
    urls: list[str] = []
    for series_id in series_ids:
        if not isinstance(series_id, str) or not series_id.strip():
            continue
        query = urlencode(
            {
                "series_id": series_id.strip().upper(),
                "file_type": "json",
                "observation_end": result.request.run_date.isoformat(),
                "sort_order": "desc",
                "limit": 100,
            }
        )
        urls.append(f"https://api.stlouisfed.org/fred/series/observations?{query}")
    return tuple(urls)


def _sec_edgar_query_urls(result: ProviderResult[object]) -> tuple[str, ...]:
    cik = _sec_cik_from_result(result)
    if cik is None:
        return ()
    normalized_cik = cik.zfill(10)
    return (
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalized_cik}.json",
        f"https://data.sec.gov/submissions/CIK{normalized_cik}.json",
    )


def _sec_cik_from_result(result: ProviderResult[object]) -> str | None:
    data = result.data
    metrics = getattr(data, "metrics", ())
    if not isinstance(metrics, tuple):
        return None
    for metric in metrics:
        if not isinstance(metric, ProviderMetric):
            continue
        source_url = text_from_metadata(metric.metadata, "source_url")
        if source_url is None:
            continue
        match = re.search(r"/Archives/edgar/data/(\d+)/", source_url)
        if match:
            return match.group(1)
    return None


def scoped_source_evidence(
    evidence: SourceEvidence,
    *,
    run_id: str,
    tool_slug: str,
) -> SourceEvidence:
    """Scope provider evidence IDs to a run while preserving the provider ID in metadata."""

    provenance = evidence.provenance
    scope_parts = (
        run_id,
        tool_slug,
        provenance.provider_name,
        provenance.raw_identifier or "",
        provenance.permalink or "",
        evidence.evidence_id,
    )
    evidence_id = f"evidence-{tool_slug}-{stable_digest('|'.join(scope_parts))}"
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
        stance = text_from_metadata(derived_analysis, "stance")
        if stance in {"supports", "contradicts", "neutral"}:
            metadata["stance"] = stance
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
    upstream_provider = text_from_metadata(metric.metadata, "provider_name") or provider_name
    upstream_raw_snapshot_id = text_from_metadata(metric.metadata, "raw_snapshot_id")
    upstream_cache_key = text_from_metadata(metric.metadata, "cache_key")
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
            provider_name=upstream_provider,
            source_kind=source_kind,
            retrieval_method=retrieval_method,
            fetched_at=fetched_at,
            observed_at=observed_at,
            source_url=source_url,
            permalink=source_url,
            raw_identifier=raw_identifier,
            raw_snapshot_id=(
                upstream_raw_snapshot_id
                or raw_snapshot_id
                or f"metric:{stable_digest(raw_identifier)}"
            ),
            query=query or source_query_id,
            cache_key=upstream_cache_key or cache_key,
            freshness_status=freshness_status,
            provider_metadata={
                "phase4_tool": tool_slug,
                "metric": model_json(metric),
                "source_query_id": source_query_id,
                "phase4_metric_provider_name": provider_name,
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
    start_char, end_char, quote = quote_span(evidence.text, limit=180)
    return EvidenceReference(
        evidence_id=evidence.evidence_id,
        quote=quote,
        start_char=start_char,
        end_char=end_char,
        relevance=relevance,
    )


def quote_span(text: str, *, limit: int) -> tuple[int | None, int | None, str | None]:
    if not text:
        return None, None, None
    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    if start == len(text):
        return None, None, None
    end = min(len(text), start + limit)
    while end > start and text[end - 1].isspace():
        end -= 1
    if end == start:
        return None, None, None
    return start, end, text[start:end]


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
        return "successful"
    if warnings:
        return "failed"
    return "empty"


def provider_metric_freshness(
    provider_status: ProviderStatus,
    metric: ProviderMetric,
    *,
    run_date: date | None = None,
    stale_after_days: int = 456,
) -> FreshnessStatus:
    if provider_status == ProviderStatus.STALE:
        return FreshnessStatus.STALE
    if metric.as_of is None:
        return FreshnessStatus.MISSING
    if run_date is not None:
        metric_date = metric.as_of.date() if isinstance(metric.as_of, datetime) else metric.as_of
        age_days = (run_date - metric_date).days
        if age_days < 0:
            return FreshnessStatus.UNKNOWN
        if age_days > stale_after_days:
            return FreshnessStatus.STALE
    return FreshnessStatus.FRESH


def retrieval_method_for_provider(provider_name: str) -> RetrievalMethod:
    normalized = provider_name.lower()
    if "fixture" in normalized:
        return RetrievalMethod.FIXTURE
    if normalized in {
        "fred",
        "sec-edgar",
        "alpha-vantage-fundamentals",
        "alpha-vantage-market-data",
        "x-recent-search",
    }:
        return RetrievalMethod.OFFICIAL_API
    if normalized in {"reddit", "ap-news", "candlecharts-market-data", "yahoo-finance-chart"}:
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
        return sanitize_source_query_url(source_url) or source_url
    query_url = text_from_metadata(metric.metadata, "source_query_url")
    if query_url:
        return sanitize_source_query_url(query_url) or query_url
    return f"provider://{provider_name}/{stable_digest(raw_identifier)}"


def _sanitize_url_path(path: str) -> str:
    if not path:
        return path
    sensitive_markers = ("apikey", "api_key", "access_token", "token", "key", "secret")
    return "/".join(
        "REDACTED" if any(marker in part.lower() for marker in sensitive_markers) else part
        for part in path.split("/")
    )


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
    "PHASE4_RUNNING_TOOL_RUN_STATUS",
    "PHASE4_TOOL_VERSION",
    "Phase4StoredToolRunStatus",
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
    "safe_phase4_tool_execution",
    "scoped_source_evidence",
    "source_evidence_json",
    "source_kind_for_fundamental_metric",
    "source_query_urls_from_result",
    "standardize_phase4_tool_run_status",
    "tool_identity",
    "tool_status",
    "warning_messages",
    "warning_payloads",
    "write_phase4_json_artifact",
]
