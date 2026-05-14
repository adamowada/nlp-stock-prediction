"""Phase 4 news and catalyst evidence tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.providers import EvidenceRequest, NewsProvider, ProviderResult
from nlp_stock_prediction.orchestration.phase2_common import source_evidence_ticker
from nlp_stock_prediction.orchestration.phase4_common import (
    Phase4ToolResult,
    catalyst_labels,
    dedupe_strings,
    derived_label_for_text,
    evidence_record_from_source,
    model_json,
    provider_request_query,
    record_source_query_for_result,
    result_list_json,
    scoped_source_evidence,
    source_evidence_json,
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

PHASE4_NEWS_SCHEMA_VERSION = "phase4.news-catalyst.v1"
PHASE4_NEWS_TOOL_NAME = "phase4_news_catalyst"
PHASE4_NEWS_TOOL_SLUG = "phase4-news-catalyst"


@dataclass(frozen=True)
class Phase4NewsCatalystTool:
    """Aggregate article providers and keep catalysts as derived labels."""

    providers: tuple[NewsProvider, ...]

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
        extra_tickers: Sequence[str] = (),
        query: str | None = None,
        limit: int = 10,
    ) -> Phase4ToolResult:
        normalized_symbol = symbol.strip().upper()
        tickers = _request_tickers(normalized_symbol, extra_tickers)
        request = EvidenceRequest(
            request_id=f"phase4-news-{run_id}-{normalized_symbol}",
            run_date=run_date,
            tickers=tickers,
            limit=limit,
            query=query or f"{normalized_symbol} catalyst news",
            include_posts=False,
            include_comments=False,
        )
        tool_run_id, artifact_id, filename = tool_identity(
            tool_slug=PHASE4_NEWS_TOOL_SLUG,
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
            tool_name=PHASE4_NEWS_TOOL_NAME,
            started_at=generated_at,
            inputs=inputs,
        )

        provider_results = tuple(provider.fetch_articles(request) for provider in self.providers)
        source_query_ids: list[str] = []
        evidence_records: list[SourceEvidence] = []
        sqlite_evidence_records: list[EvidenceRecord] = []
        derived_labels: list[JsonObject] = []
        warning_text: list[str] = []

        for provider_result in provider_results:
            source_query = record_source_query_for_result(
                store=store,
                tool_slug=PHASE4_NEWS_TOOL_SLUG,
                tool_run_id=tool_run_id,
                result=cast(ProviderResult[object], provider_result),
            )
            source_query_ids.append(source_query.source_query_id)
            warning_text.extend(warning_messages(provider_result.warnings))
            for provider_evidence in provider_result.data or ():
                evidence = scoped_source_evidence(
                    provider_evidence,
                    run_id=run_id,
                    tool_slug=PHASE4_NEWS_TOOL_SLUG,
                )
                label = derived_label_for_text(
                    evidence=evidence,
                    catalysts=catalyst_labels(f"{evidence.title or ''}\n{evidence.text}"),
                )
                evidence_records.append(evidence)
                derived_labels.append(label)
                sqlite_evidence_records.append(
                    evidence_record_from_source(
                        evidence,
                        tool_run_id=tool_run_id,
                        source_query_id=source_query.source_query_id,
                        artifact_id=artifact_id,
                        fallback_instrument_id=instrument_id,
                        derived_analysis=label,
                    )
                )

        warnings = dedupe_strings(warning_text)
        status = tool_status(record_count=len(evidence_records), warnings=warnings)
        payload = _artifact_payload(
            run_id=run_id,
            tool_run_id=tool_run_id,
            request=request,
            provider_results=provider_results,
            evidence=evidence_records,
            derived_labels=derived_labels,
            warnings=warnings,
        )
        artifact_path = write_phase4_json_artifact(
            store=store,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            created_at=generated_at,
            produced_by=PHASE4_NEWS_TOOL_NAME,
            tool_run_id=tool_run_id,
            schema_version=PHASE4_NEWS_SCHEMA_VERSION,
            artifact_id=artifact_id,
            artifact_type="normalized_evidence",
            filename=filename,
            payload=payload,
            record_count=len(evidence_records),
            metadata=cast(
                JsonObject,
                {
                    "symbol": normalized_symbol,
                    "evidence_ids": [record.evidence_id for record in evidence_records],
                    "source_query_ids": source_query_ids,
                    "warnings": list(warnings),
                },
            ),
        )
        for evidence_record in sqlite_evidence_records:
            store.record_evidence(evidence_record)
        record_tool_completed(
            store=store,
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=PHASE4_NEWS_TOOL_NAME,
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
            evidence_ids=tuple(record.evidence_id for record in evidence_records),
            source_query_ids=tuple(source_query_ids),
            warnings=warnings,
            artifact_payload=payload,
        )


def run_phase4_news_catalyst_tool(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    symbol: str,
    run_date: date,
    generated_at: datetime,
    providers: Sequence[NewsProvider],
    instrument_id: str | None = None,
    extra_tickers: Sequence[str] = (),
    query: str | None = None,
    limit: int = 10,
) -> Phase4ToolResult:
    """Convenience function for the news/catalyst evidence tool."""

    return Phase4NewsCatalystTool(providers=tuple(providers)).run(
        store=store,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        run_id=run_id,
        symbol=symbol,
        run_date=run_date,
        generated_at=generated_at,
        instrument_id=instrument_id,
        extra_tickers=extra_tickers,
        query=query,
        limit=limit,
    )


def _request_tickers(symbol: str, extra_tickers: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_ticker in (symbol, *extra_tickers):
        ticker = source_evidence_ticker(raw_ticker)
        if ticker is not None and ticker not in normalized:
            normalized.append(ticker)
    return tuple(normalized or [symbol])


def _artifact_payload(
    *,
    run_id: str,
    tool_run_id: str,
    request: EvidenceRequest,
    provider_results: Sequence[ProviderResult[tuple[SourceEvidence, ...]]],
    evidence: Sequence[SourceEvidence],
    derived_labels: Sequence[JsonObject],
    warnings: Sequence[str],
) -> JsonObject:
    return cast(
        JsonObject,
        {
            "schema_version": PHASE4_NEWS_SCHEMA_VERSION,
            "run_id": run_id,
            "tool_run_id": tool_run_id,
            "request": model_json(request),
            "query": provider_request_query(request),
            "provider_results": result_list_json(
                cast(Sequence[ProviderResult[object]], provider_results)
            ),
            "source_evidence": source_evidence_json(evidence),
            "derived_analysis": {
                "analysis_type": "news_catalyst_labels",
                "labels": list(derived_labels),
                "notes": (
                    "Article text is source evidence. Catalyst and stance labels are derived "
                    "analysis and cite source evidence IDs."
                ),
            },
            "warnings": list(warnings),
        },
    )


__all__ = [
    "PHASE4_NEWS_SCHEMA_VERSION",
    "PHASE4_NEWS_TOOL_NAME",
    "Phase4NewsCatalystTool",
    "run_phase4_news_catalyst_tool",
]
