"""Phase 4 sector and macro context tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.analysis.macro import analyze_macro_context
from nlp_stock_prediction.analysis.sector import analyze_sector_context
from nlp_stock_prediction.contracts.analysis import MacroContext, SectorContext
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import (
    FreshnessStatus,
    RetrievalMethod,
    SourceKind,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.provenance import ProviderWarning
from nlp_stock_prediction.contracts.providers import (
    FundamentalsSnapshot,
    MacroProvider,
    MacroRequest,
    MacroSeries,
    MacroSnapshot,
    ProviderMetric,
    ProviderResult,
)
from nlp_stock_prediction.orchestration.phase2_common import stable_digest
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
    record_tool_completed,
    record_tool_started,
    result_list_json,
    retrieval_method_for_provider,
    source_evidence_json,
    text_from_metadata,
    tool_identity,
    tool_status,
    warning_messages,
    write_phase4_json_artifact,
)
from nlp_stock_prediction.storage.records import EvidenceRecord, SourceQueryRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE4_SECTOR_MACRO_SCHEMA_VERSION = "phase4.sector-macro.v1"
PHASE4_SECTOR_MACRO_TOOL_NAME = "phase4_sector_macro"
PHASE4_SECTOR_MACRO_TOOL_SLUG = "phase4-sector-macro"
_SECTOR_INPUT_PROVIDER = "phase4-sector-inputs"


@dataclass(frozen=True)
class Phase4SectorMacroTool:
    """Synthesize sector and macro context while indexing metric evidence."""

    macro_providers: tuple[MacroProvider, ...] = ()

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
        target_snapshot: FundamentalsSnapshot,
        instrument_id: str | None = None,
        peers: Sequence[FundamentalsSnapshot] = (),
        sector: str | None = None,
        benchmark_symbol: str | None = None,
        benchmark_metrics: Sequence[ProviderMetric] = (),
        macro_series_ids: Sequence[str] = (),
        horizon: TimeHorizon = TimeHorizon.UNKNOWN,
    ) -> Phase4ToolResult:
        normalized_symbol = symbol.strip().upper()
        tool_run_id, artifact_id, filename = tool_identity(
            tool_slug=PHASE4_SECTOR_MACRO_TOOL_SLUG,
            run_id=run_id,
            symbol=normalized_symbol,
        )
        macro_request = MacroRequest(
            request_id=f"phase4-macro-{run_id}-{normalized_symbol}",
            run_date=run_date,
            series_ids=tuple(macro_series_ids),
            horizon=horizon,
            query=",".join(macro_series_ids) if macro_series_ids else None,
        )
        inputs: JsonObject = {
            "symbol": normalized_symbol,
            "instrument_id": instrument_id,
            "run_date": run_date.isoformat(),
            "sector": sector,
            "benchmark_symbol": benchmark_symbol,
            "horizon": horizon.value,
            "macro_request": model_json(macro_request),
        }
        record_tool_started(
            store=store,
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=PHASE4_SECTOR_MACRO_TOOL_NAME,
            started_at=generated_at,
            inputs=inputs,
        )

        sector_query = _record_sector_source_query(
            store=store,
            tool_run_id=tool_run_id,
            generated_at=generated_at,
            run_id=run_id,
            symbol=normalized_symbol,
            sector=sector,
            benchmark_symbol=benchmark_symbol,
        )
        source_query_ids: list[str] = [sector_query.source_query_id]
        warning_text: list[str] = []
        all_warnings: list[ProviderWarning] = []
        metric_evidence: list[SourceEvidence] = []
        evidence_records: list[EvidenceRecord] = []

        sector_metric_evidence = _sector_metric_evidence(
            run_id=run_id,
            generated_at=generated_at,
            source_query_id=sector_query.source_query_id,
            query=sector_query.query,
            target_snapshot=target_snapshot,
            peers=peers,
            benchmark_symbol=benchmark_symbol,
            benchmark_metrics=benchmark_metrics,
        )
        metric_evidence.extend(sector_metric_evidence)
        evidence_records.extend(
            metric_record_from_evidence(
                evidence,
                tool_run_id=tool_run_id,
                source_query_id=sector_query.source_query_id,
                artifact_id=artifact_id,
                fallback_instrument_id=instrument_id,
                analysis_reference={"analysis_type": "sector_metric_input"},
            )
            for evidence in sector_metric_evidence
        )

        macro_results = tuple(
            provider.fetch_macro(macro_request) for provider in self.macro_providers
        )
        macro_series: list[MacroSeries] = []
        for provider_result in macro_results:
            source_query = record_source_query_for_result(
                store=store,
                tool_slug=PHASE4_SECTOR_MACRO_TOOL_SLUG,
                tool_run_id=tool_run_id,
                result=cast(ProviderResult[object], provider_result),
            )
            source_query_ids.append(source_query.source_query_id)
            warning_text.extend(warning_messages(provider_result.warnings))
            all_warnings.extend(provider_result.warnings)
            snapshot = provider_result.data
            if snapshot is None:
                continue
            macro_series.extend(snapshot.series)
            for series in snapshot.series:
                for index, raw_metric in enumerate(series.values):
                    metric = _macro_metric_with_metadata(raw_metric, provider_result, series)
                    evidence = metric_source_evidence(
                        run_id=run_id,
                        tool_slug=PHASE4_SECTOR_MACRO_TOOL_SLUG,
                        fetched_at=provider_result.fetched_at,
                        provider_name=provider_result.provider_name,
                        source_query_id=source_query.source_query_id,
                        query=provider_request_query(provider_result.request),
                        raw_snapshot_id=provider_result.raw_snapshot_id,
                        cache_key=provider_result.cache_key,
                        symbol=None,
                        metric=metric,
                        source_kind=SourceKind.MACRO_SERIES,
                        retrieval_method=retrieval_method_for_provider(
                            provider_result.provider_name
                        ),
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
                            analysis_reference={
                                "analysis_type": "macro_metric_input",
                                "series_id": series.series_id,
                            },
                        )
                    )

        sector_context = _sector_with_references(
            analyze_sector_context(
                target_snapshot,
                peers=peers,
                sector=sector,
                benchmark_symbol=benchmark_symbol,
                benchmark_metrics=benchmark_metrics,
                as_of=run_date,
            ),
            evidence=sector_metric_evidence,
        )
        macro_snapshot = MacroSnapshot(series=tuple(macro_series))
        macro_context = _macro_with_references(
            analyze_macro_context(macro_snapshot, horizon=horizon, as_of=run_date),
            evidence=tuple(
                item for item in metric_evidence if item.source_kind == SourceKind.MACRO_SERIES
            ),
            warnings=tuple(all_warnings),
        )
        warnings = dedupe_strings(warning_text)
        status = tool_status(record_count=len(metric_evidence), warnings=warnings)
        payload = _artifact_payload(
            run_id=run_id,
            tool_run_id=tool_run_id,
            macro_request=macro_request,
            macro_results=macro_results,
            target_snapshot=target_snapshot,
            peers=peers,
            benchmark_metrics=benchmark_metrics,
            macro_snapshot=macro_snapshot,
            sector_context=sector_context,
            macro_context=macro_context,
            metric_evidence=metric_evidence,
            warnings=warnings,
        )
        artifact_path = write_phase4_json_artifact(
            store=store,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            created_at=generated_at,
            produced_by=PHASE4_SECTOR_MACRO_TOOL_NAME,
            tool_run_id=tool_run_id,
            schema_version=PHASE4_SECTOR_MACRO_SCHEMA_VERSION,
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
            tool_name=PHASE4_SECTOR_MACRO_TOOL_NAME,
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


def run_phase4_sector_macro_tool(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    symbol: str,
    run_date: date,
    generated_at: datetime,
    target_snapshot: FundamentalsSnapshot,
    macro_providers: Sequence[MacroProvider] = (),
    instrument_id: str | None = None,
    peers: Sequence[FundamentalsSnapshot] = (),
    sector: str | None = None,
    benchmark_symbol: str | None = None,
    benchmark_metrics: Sequence[ProviderMetric] = (),
    macro_series_ids: Sequence[str] = (),
    horizon: TimeHorizon = TimeHorizon.UNKNOWN,
) -> Phase4ToolResult:
    """Convenience function for the sector/macro context tool."""

    return Phase4SectorMacroTool(macro_providers=tuple(macro_providers)).run(
        store=store,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        run_id=run_id,
        symbol=symbol,
        run_date=run_date,
        generated_at=generated_at,
        target_snapshot=target_snapshot,
        instrument_id=instrument_id,
        peers=peers,
        sector=sector,
        benchmark_symbol=benchmark_symbol,
        benchmark_metrics=benchmark_metrics,
        macro_series_ids=macro_series_ids,
        horizon=horizon,
    )


def _record_sector_source_query(
    *,
    store: SQLiteStore,
    tool_run_id: str,
    generated_at: datetime,
    run_id: str,
    symbol: str,
    sector: str | None,
    benchmark_symbol: str | None,
) -> SourceQueryRecord:
    query = f"{symbol} sector context"
    source_query_id = (
        f"query-{PHASE4_SECTOR_MACRO_TOOL_SLUG}-"
        f"{stable_digest('|'.join((run_id, query, sector or '', benchmark_symbol or '')))}"
    )
    record = SourceQueryRecord(
        source_query_id=source_query_id,
        tool_run_id=tool_run_id,
        provider=_SECTOR_INPUT_PROVIDER,
        query=query,
        retrieved_at=generated_at,
        metadata=cast(
            JsonObject,
            {
                "phase4_tool": PHASE4_SECTOR_MACRO_TOOL_SLUG,
                "sector": sector,
                "benchmark_symbol": benchmark_symbol,
            },
        ),
    )
    store.record_source_query(record)
    return record


def _sector_metric_evidence(
    *,
    run_id: str,
    generated_at: datetime,
    source_query_id: str,
    query: str,
    target_snapshot: FundamentalsSnapshot,
    peers: Sequence[FundamentalsSnapshot],
    benchmark_symbol: str | None,
    benchmark_metrics: Sequence[ProviderMetric],
) -> tuple[SourceEvidence, ...]:
    evidence: list[SourceEvidence] = []
    index = 0
    for metric in target_snapshot.metrics:
        evidence.append(
            _sector_metric_source_evidence(
                run_id=run_id,
                generated_at=generated_at,
                source_query_id=source_query_id,
                query=query,
                symbol=target_snapshot.ticker,
                metric=_metric_with_input_metadata(
                    metric, role="target", ticker=target_snapshot.ticker
                ),
                source_kind=SourceKind.FUNDAMENTAL_DATA,
                index=index,
            )
        )
        index += 1
    for peer in peers:
        for metric in peer.metrics:
            evidence.append(
                _sector_metric_source_evidence(
                    run_id=run_id,
                    generated_at=generated_at,
                    source_query_id=source_query_id,
                    query=query,
                    symbol=peer.ticker,
                    metric=_metric_with_input_metadata(metric, role="peer", ticker=peer.ticker),
                    source_kind=SourceKind.FUNDAMENTAL_DATA,
                    index=index,
                )
            )
            index += 1
    for metric in benchmark_metrics:
        evidence.append(
            _sector_metric_source_evidence(
                run_id=run_id,
                generated_at=generated_at,
                source_query_id=source_query_id,
                query=query,
                symbol=benchmark_symbol,
                metric=_metric_with_input_metadata(
                    metric,
                    role="benchmark",
                    ticker=benchmark_symbol,
                ),
                source_kind=SourceKind.MARKET_DATA,
                index=index,
            )
        )
        index += 1
    return tuple(evidence)


def _sector_metric_source_evidence(
    *,
    run_id: str,
    generated_at: datetime,
    source_query_id: str,
    query: str,
    symbol: str | None,
    metric: ProviderMetric,
    source_kind: SourceKind,
    index: int,
) -> SourceEvidence:
    upstream_provider = (
        text_from_metadata(metric.metadata, "provider_name") or _SECTOR_INPUT_PROVIDER
    )
    return metric_source_evidence(
        run_id=run_id,
        tool_slug=PHASE4_SECTOR_MACRO_TOOL_SLUG,
        fetched_at=generated_at,
        provider_name=upstream_provider,
        source_query_id=source_query_id,
        query=query,
        raw_snapshot_id=(
            text_from_metadata(metric.metadata, "raw_snapshot_id")
            or f"sector-input:{stable_digest('|'.join((source_query_id, str(index))))}"
        ),
        cache_key=text_from_metadata(metric.metadata, "cache_key"),
        symbol=symbol,
        metric=metric,
        source_kind=source_kind,
        retrieval_method=(
            RetrievalMethod.DERIVED
            if upstream_provider == _SECTOR_INPUT_PROVIDER
            else retrieval_method_for_provider(upstream_provider)
        ),
        freshness_status=(
            FreshnessStatus.MISSING if metric.as_of is None else FreshnessStatus.FRESH
        ),
        index=index,
    )


def _metric_with_input_metadata(
    metric: ProviderMetric,
    *,
    role: str,
    ticker: str | None,
) -> ProviderMetric:
    payload = metric.model_dump(mode="python")
    raw_metadata = payload.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    metadata["input_role"] = role
    if ticker is not None:
        metadata["ticker"] = ticker
    payload["metadata"] = metadata
    return ProviderMetric.model_validate(payload)


def _macro_metric_with_metadata(
    metric: ProviderMetric,
    provider_result: ProviderResult[MacroSnapshot],
    series: MacroSeries,
) -> ProviderMetric:
    payload = metric.model_dump(mode="python")
    raw_metadata = payload.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    metadata["provider_name"] = provider_result.provider_name
    metadata["provider_status"] = provider_result.status.value
    metadata["series_id"] = series.series_id
    metadata["series_name"] = series.name
    if provider_result.raw_snapshot_id is not None:
        metadata.setdefault("raw_snapshot_id", provider_result.raw_snapshot_id)
    payload["metadata"] = metadata
    return ProviderMetric.model_validate(payload)


def _sector_with_references(
    context: SectorContext,
    *,
    evidence: Sequence[SourceEvidence],
) -> SectorContext:
    payload = context.model_dump(mode="python")
    payload["evidence"] = tuple(
        evidence_reference(record, relevance=0.65) for record in evidence[:12]
    )
    return SectorContext.model_validate(payload)


def _macro_with_references(
    context: MacroContext,
    *,
    evidence: Sequence[SourceEvidence],
    warnings: tuple[ProviderWarning, ...],
) -> MacroContext:
    payload = context.model_dump(mode="python")
    payload["evidence"] = tuple(
        evidence_reference(record, relevance=0.65) for record in evidence[:12]
    )
    payload["warnings"] = warnings
    return MacroContext.model_validate(payload)


def _artifact_payload(
    *,
    run_id: str,
    tool_run_id: str,
    macro_request: MacroRequest,
    macro_results: Sequence[ProviderResult[MacroSnapshot]],
    target_snapshot: FundamentalsSnapshot,
    peers: Sequence[FundamentalsSnapshot],
    benchmark_metrics: Sequence[ProviderMetric],
    macro_snapshot: MacroSnapshot,
    sector_context: SectorContext,
    macro_context: MacroContext,
    metric_evidence: Sequence[SourceEvidence],
    warnings: Sequence[str],
) -> JsonObject:
    return cast(
        JsonObject,
        {
            "schema_version": PHASE4_SECTOR_MACRO_SCHEMA_VERSION,
            "run_id": run_id,
            "tool_run_id": tool_run_id,
            "macro_request": model_json(macro_request),
            "macro_provider_results": result_list_json(
                cast(Sequence[ProviderResult[object]], macro_results)
            ),
            "sector_inputs": {
                "target": model_json(target_snapshot),
                "peers": [model_json(peer) for peer in peers],
                "benchmark_metrics": [model_json(metric) for metric in benchmark_metrics],
            },
            "macro_snapshot": model_json(macro_snapshot),
            "source_evidence": source_evidence_json(metric_evidence),
            "derived_analysis": {
                "analysis_type": "sector_macro_context",
                "sector_context": model_json(sector_context),
                "macro_context": model_json(macro_context),
                "notes": (
                    "Sector and macro context are derived analysis. Provider metrics and macro "
                    "series observations are preserved separately as source evidence."
                ),
            },
            "warnings": list(warnings),
        },
    )


__all__ = [
    "PHASE4_SECTOR_MACRO_SCHEMA_VERSION",
    "PHASE4_SECTOR_MACRO_TOOL_NAME",
    "Phase4SectorMacroTool",
    "run_phase4_sector_macro_tool",
]
