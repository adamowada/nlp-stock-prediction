"""Phase 4 fundamentals evidence and analysis tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.analysis.fundamentals import analyze_fundamentals
from nlp_stock_prediction.contracts.analysis import FundamentalAnalysis
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.provenance import ProviderWarning
from nlp_stock_prediction.contracts.providers import (
    FundamentalsProvider,
    FundamentalsRequest,
    FundamentalsSnapshot,
    ProviderMetric,
    ProviderResult,
)
from nlp_stock_prediction.orchestration.phase4_common import (
    Phase4ToolResult,
    dedupe_strings,
    evidence_reference,
    metric_record_from_evidence,
    metric_source_evidence,
    model_json,
    provider_metric_freshness,
    provider_request_query,
    record_source_query_for_result,
    result_list_json,
    retrieval_method_for_provider,
    source_evidence_json,
    source_kind_for_fundamental_metric,
    source_query_urls_from_result,
    tool_identity,
    tool_status,
    warning_messages,
)
from nlp_stock_prediction.orchestration.phase4_execution import (
    record_tool_completed,
    record_tool_started,
    write_phase4_json_artifact,
)
from nlp_stock_prediction.storage.records import EvidenceRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE4_FUNDAMENTALS_SCHEMA_VERSION = "phase4.fundamentals.v1"
PHASE4_FUNDAMENTALS_TOOL_NAME = "phase4_fundamentals"
PHASE4_FUNDAMENTALS_TOOL_SLUG = "phase4-fundamentals"


@dataclass(frozen=True)
class Phase4FundamentalsTool:
    """Fetch provider facts, preserve them as evidence, and synthesize analysis."""

    providers: tuple[FundamentalsProvider, ...]

    def run(
        self,
        *,
        store: SQLiteStore,
        repo_root: Path,
        artifact_dir: Path,
        run_id: str,
        symbol: str,
        run_date: date,
        generated_at: datetime,
        instrument_id: str | None = None,
        fiscal_period: str | None = None,
    ) -> Phase4ToolResult:
        normalized_symbol = symbol.strip().upper()
        request = FundamentalsRequest(
            request_id=f"phase4-fundamentals-{run_id}-{normalized_symbol}",
            run_date=run_date,
            tickers=(normalized_symbol,),
            query=normalized_symbol,
            fiscal_period=fiscal_period,
        )
        tool_run_id, artifact_id, filename = tool_identity(
            tool_slug=PHASE4_FUNDAMENTALS_TOOL_SLUG,
            run_id=run_id,
            symbol=normalized_symbol,
        )
        inputs: JsonObject = {
            "symbol": normalized_symbol,
            "instrument_id": instrument_id,
            "run_date": run_date.isoformat(),
            "request": model_json(request),
        }
        record_tool_started(
            store=store,
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=PHASE4_FUNDAMENTALS_TOOL_NAME,
            started_at=generated_at,
            inputs=inputs,
        )

        provider_results = tuple(
            provider.fetch_fundamentals(request) for provider in self.providers
        )
        source_query_ids: list[str] = []
        warning_text: list[str] = []
        all_warnings: list[ProviderWarning] = []
        metric_evidence: list[SourceEvidence] = []
        evidence_records: list[EvidenceRecord] = []
        merged_metrics: list[ProviderMetric] = []
        company_name: str | None = None

        for provider_result in provider_results:
            source_query = record_source_query_for_result(
                store=store,
                tool_slug=PHASE4_FUNDAMENTALS_TOOL_SLUG,
                tool_run_id=tool_run_id,
                result=cast(ProviderResult[object], provider_result),
            )
            source_query_ids.append(source_query.source_query_id)
            warning_text.extend(warning_messages(provider_result.warnings))
            all_warnings.extend(provider_result.warnings)
            snapshot = provider_result.data
            if snapshot is None:
                continue
            company_name = company_name or snapshot.company_name
            provider_query_urls = source_query_urls_from_result(
                cast(ProviderResult[object], provider_result)
            )
            for index, raw_metric in enumerate(snapshot.metrics):
                metric = _metric_with_provider_metadata(
                    raw_metric,
                    provider_result,
                    source_query_urls=provider_query_urls,
                )
                merged_metrics.append(metric)
                evidence = metric_source_evidence(
                    run_id=run_id,
                    tool_slug=PHASE4_FUNDAMENTALS_TOOL_SLUG,
                    fetched_at=provider_result.fetched_at,
                    provider_name=provider_result.provider_name,
                    source_query_id=source_query.source_query_id,
                    query=provider_request_query(provider_result.request),
                    raw_snapshot_id=provider_result.raw_snapshot_id,
                    cache_key=provider_result.cache_key,
                    symbol=normalized_symbol,
                    metric=metric,
                    source_kind=source_kind_for_fundamental_metric(
                        provider_result.provider_name,
                        metric,
                    ),
                    retrieval_method=retrieval_method_for_provider(provider_result.provider_name),
                    freshness_status=provider_metric_freshness(provider_result.status, metric),
                    index=index,
                )
                metric_evidence.append(evidence)
                evidence_records.append(
                    metric_record_from_evidence(
                        evidence,
                        tool_run_id=tool_run_id,
                        source_query_id=source_query.source_query_id,
                        artifact_id=artifact_id,
                        fallback_instrument_id=instrument_id,
                        analysis_reference={
                            "analysis_type": "fundamental_metric_input",
                            "metric_name": metric.name,
                        },
                    )
                )

        snapshot = FundamentalsSnapshot(
            ticker=normalized_symbol,
            company_name=company_name,
            metrics=tuple(merged_metrics),
        )
        analysis = _analysis_with_references(
            analyze_fundamentals(snapshot, as_of=run_date),
            metric_evidence=metric_evidence,
            warnings=tuple(all_warnings),
        )
        warnings = dedupe_strings(warning_text)
        status = tool_status(record_count=len(metric_evidence), warnings=warnings)
        payload = _artifact_payload(
            run_id=run_id,
            tool_run_id=tool_run_id,
            request=request,
            provider_results=provider_results,
            snapshot=snapshot,
            analysis=analysis,
            metric_evidence=metric_evidence,
            warnings=warnings,
        )
        artifact_path = write_phase4_json_artifact(
            store=store,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            created_at=generated_at,
            produced_by=PHASE4_FUNDAMENTALS_TOOL_NAME,
            tool_run_id=tool_run_id,
            schema_version=PHASE4_FUNDAMENTALS_SCHEMA_VERSION,
            artifact_id=artifact_id,
            artifact_type="analysis_context",
            filename=filename,
            payload=payload,
            record_count=len(metric_evidence),
            metadata=cast(
                JsonObject,
                {
                    "symbol": normalized_symbol,
                    "evidence_ids": [record.evidence_id for record in metric_evidence],
                    "source_query_ids": source_query_ids,
                    "warnings": list(warnings),
                },
            ),
        )
        for evidence_record in evidence_records:
            store.record_evidence(evidence_record)
        record_tool_completed(
            store=store,
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=PHASE4_FUNDAMENTALS_TOOL_NAME,
            started_at=generated_at,
            completed_at=generated_at,
            status=status,
            inputs=inputs,
            warnings=warnings,
        )
        return Phase4ToolResult(
            run_id=run_id,
            tool_run_id=tool_run_id,
            artifact_id=artifact_id,
            artifact_path=artifact_path,
            status=status,
            evidence_ids=tuple(record.evidence_id for record in metric_evidence),
            source_query_ids=tuple(source_query_ids),
            warnings=warnings,
            artifact_payload=payload,
        )


def run_phase4_fundamentals_tool(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    symbol: str,
    run_date: date,
    generated_at: datetime,
    providers: Sequence[FundamentalsProvider],
    instrument_id: str | None = None,
    fiscal_period: str | None = None,
) -> Phase4ToolResult:
    """Convenience function for the fundamentals evidence tool."""

    return Phase4FundamentalsTool(providers=tuple(providers)).run(
        store=store,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        run_id=run_id,
        symbol=symbol,
        run_date=run_date,
        generated_at=generated_at,
        instrument_id=instrument_id,
        fiscal_period=fiscal_period,
    )


def _metric_with_provider_metadata(
    metric: ProviderMetric,
    provider_result: ProviderResult[FundamentalsSnapshot],
    *,
    source_query_urls: Sequence[str] = (),
) -> ProviderMetric:
    payload = metric.model_dump(mode="python")
    raw_metadata = payload.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    metadata["provider_name"] = provider_result.provider_name
    metadata["provider_status"] = provider_result.status.value
    if provider_result.raw_snapshot_id is not None:
        metadata.setdefault("raw_snapshot_id", provider_result.raw_snapshot_id)
    if provider_result.cache_key is not None:
        metadata.setdefault("cache_key", provider_result.cache_key)
    if source_query_urls:
        metadata.setdefault("source_query_url", source_query_urls[0])
        metadata.setdefault("source_query_urls", list(source_query_urls))
    payload["metadata"] = metadata
    return ProviderMetric.model_validate(payload)


def _analysis_with_references(
    analysis: FundamentalAnalysis,
    *,
    metric_evidence: Sequence[SourceEvidence],
    warnings: tuple[ProviderWarning, ...],
) -> FundamentalAnalysis:
    payload = analysis.model_dump(mode="python")
    payload["evidence"] = tuple(
        evidence_reference(evidence, relevance=0.75) for evidence in metric_evidence[:12]
    )
    payload["warnings"] = warnings
    return FundamentalAnalysis.model_validate(payload)


def _artifact_payload(
    *,
    run_id: str,
    tool_run_id: str,
    request: FundamentalsRequest,
    provider_results: Sequence[ProviderResult[FundamentalsSnapshot]],
    snapshot: FundamentalsSnapshot,
    analysis: FundamentalAnalysis,
    metric_evidence: Sequence[SourceEvidence],
    warnings: Sequence[str],
) -> JsonObject:
    return cast(
        JsonObject,
        {
            "schema_version": PHASE4_FUNDAMENTALS_SCHEMA_VERSION,
            "run_id": run_id,
            "tool_run_id": tool_run_id,
            "request": model_json(request),
            "provider_results": result_list_json(
                cast(Sequence[ProviderResult[object]], provider_results)
            ),
            "fundamentals_snapshot": model_json(snapshot),
            "source_evidence": source_evidence_json(metric_evidence),
            "derived_analysis": {
                "analysis_type": "fundamental_analysis",
                "component": model_json(analysis),
                "notes": (
                    "Provider metrics and filings are observed evidence. The fundamental "
                    "component is derived analysis with metric evidence references."
                ),
            },
            "warnings": list(warnings),
        },
    )


__all__ = [
    "PHASE4_FUNDAMENTALS_SCHEMA_VERSION",
    "PHASE4_FUNDAMENTALS_TOOL_NAME",
    "Phase4FundamentalsTool",
    "run_phase4_fundamentals_tool",
]
