"""Research Stage social-evidence tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.providers import (
    EvidenceRequest,
    ProviderResult,
    RedditProvider,
)
from nlp_stock_prediction.orchestration.orchestration_common import source_evidence_ticker
from nlp_stock_prediction.orchestration.research_common import (
    ResearchToolResult,
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
from nlp_stock_prediction.orchestration.research_execution import (
    record_tool_completed,
    safe_research_tool_execution,
    write_research_json_artifact,
)
from nlp_stock_prediction.storage.records import EvidenceRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

RESEARCH_SOCIAL_SCHEMA_VERSION = "research.social-evidence.v1"
RESEARCH_SOCIAL_TOOL_NAME = "research_social_evidence"
RESEARCH_SOCIAL_TOOL_SLUG = "research-social-evidence"


@dataclass(frozen=True)
class ResearchSocialEvidenceTool:
    """Aggregate social providers into run-scoped source evidence and labels."""

    reddit_provider: RedditProvider | None = None

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
        company_name: str | None = None,
        aliases: Sequence[str] = (),
        extra_tickers: Sequence[str] = (),
        limit: int = 25,
    ) -> ResearchToolResult:
        normalized_symbol = symbol.strip().upper()
        tickers = _request_tickers(normalized_symbol, extra_tickers)
        search_terms = _reddit_search_terms(
            symbol=normalized_symbol,
            tickers=tickers,
            company_name=company_name,
            aliases=aliases,
        )
        request = EvidenceRequest(
            request_id=f"research-social-{run_id}-{normalized_symbol}",
            run_date=run_date,
            tickers=tickers,
            limit=limit,
            query=" OR ".join(search_terms),
            options={
                "social_provider": "reddit-public-search",
                "reddit_search_terms": list(search_terms),
                "company_name": company_name,
                "aliases": list(aliases),
            },
            include_posts=True,
            include_comments=True,
        )
        tool_run_id, artifact_id, filename = tool_identity(
            tool_slug=RESEARCH_SOCIAL_TOOL_SLUG,
            run_id=run_id,
            symbol=normalized_symbol,
        )
        inputs: JsonObject = {
            "symbol": normalized_symbol,
            "instrument_id": instrument_id,
            "run_date": run_date.isoformat(),
            "request": model_json(request),
        }
        with safe_research_tool_execution(
            store=store,
            artifact_roots=(artifact_dir,),
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=RESEARCH_SOCIAL_TOOL_NAME,
            started_at=generated_at,
            inputs=inputs,
        ):
            provider_results = self._fetch_provider_results(request)
            source_query_ids: list[str] = []
            evidence_records: list[SourceEvidence] = []
            sqlite_evidence_records: list[EvidenceRecord] = []
            derived_labels: list[JsonObject] = []
            warning_text: list[str] = []

            for provider_result in provider_results:
                source_query = record_source_query_for_result(
                    store=store,
                    tool_slug=RESEARCH_SOCIAL_TOOL_SLUG,
                    tool_run_id=tool_run_id,
                    result=cast(ProviderResult[object], provider_result),
                )
                source_query_ids.append(source_query.source_query_id)
                warning_text.extend(warning_messages(provider_result.warnings))
                for provider_evidence in provider_result.data or ():
                    evidence = scoped_source_evidence(
                        provider_evidence,
                        run_id=run_id,
                        tool_slug=RESEARCH_SOCIAL_TOOL_SLUG,
                    )
                    label = derived_label_for_text(
                        evidence=evidence,
                        catalysts=catalyst_labels(evidence.text),
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
            artifact_path = write_research_json_artifact(
                store=store,
                repo_root=repo_root,
                artifact_dir=artifact_dir,
                created_at=generated_at,
                produced_by=RESEARCH_SOCIAL_TOOL_NAME,
                tool_run_id=tool_run_id,
                schema_version=RESEARCH_SOCIAL_SCHEMA_VERSION,
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
                tool_name=RESEARCH_SOCIAL_TOOL_NAME,
                started_at=generated_at,
                completed_at=generated_at,
                status=status,
                inputs=inputs,
                warnings=warnings,
            )
            return ResearchToolResult(
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

    def _fetch_provider_results(
        self,
        request: EvidenceRequest,
    ) -> tuple[ProviderResult[tuple[SourceEvidence, ...]], ...]:
        results: list[ProviderResult[tuple[SourceEvidence, ...]]] = []
        if self.reddit_provider is not None:
            results.append(self.reddit_provider.fetch_discussion(request))
        return tuple(results)


def run_research_social_evidence_tool(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    symbol: str,
    run_date: date,
    generated_at: datetime,
    reddit_provider: RedditProvider | None = None,
    instrument_id: str | None = None,
    company_name: str | None = None,
    aliases: Sequence[str] = (),
    extra_tickers: Sequence[str] = (),
    limit: int = 25,
) -> ResearchToolResult:
    """Convenience function for the social-evidence tool."""

    return ResearchSocialEvidenceTool(
        reddit_provider=reddit_provider,
    ).run(
        store=store,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        run_id=run_id,
        symbol=symbol,
        run_date=run_date,
        generated_at=generated_at,
        instrument_id=instrument_id,
        company_name=company_name,
        aliases=aliases,
        extra_tickers=extra_tickers,
        limit=limit,
    )


def _request_tickers(symbol: str, extra_tickers: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_ticker in (symbol, *extra_tickers):
        ticker = source_evidence_ticker(raw_ticker)
        if ticker is not None and ticker not in normalized:
            normalized.append(ticker)
    return tuple(normalized or [symbol])


def _reddit_search_terms(
    *,
    symbol: str,
    tickers: Sequence[str],
    company_name: str | None,
    aliases: Sequence[str],
) -> tuple[str, ...]:
    terms: list[str] = []
    for ticker in tickers:
        _append_unique(terms, f"${ticker}")
        _append_unique(terms, f"{ticker} stock")
    if company_name is not None:
        _append_unique(terms, f"{company_name.strip()} stock")
    for alias in aliases:
        cleaned = alias.strip()
        if cleaned and cleaned.upper() != symbol:
            _append_unique(terms, f"{cleaned} stock")
    return tuple(terms or [f"${symbol}"])


def _append_unique(values: list[str], value: str) -> None:
    normalized = " ".join(value.split())
    if normalized and normalized not in values:
        values.append(normalized)


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
            "schema_version": RESEARCH_SOCIAL_SCHEMA_VERSION,
            "run_id": run_id,
            "tool_run_id": tool_run_id,
            "request": model_json(request),
            "query": provider_request_query(request),
            "provider_results": result_list_json(
                cast(Sequence[ProviderResult[object]], provider_results)
            ),
            "source_evidence": source_evidence_json(evidence),
            "derived_analysis": {
                "analysis_type": "social_stance_labels",
                "labels": list(derived_labels),
                "notes": (
                    "Social posts and comments are observed discussion. Stance labels are derived "
                    "analysis and must cite source evidence."
                ),
            },
            "warnings": list(warnings),
        },
    )


__all__ = [
    "RESEARCH_SOCIAL_SCHEMA_VERSION",
    "RESEARCH_SOCIAL_TOOL_NAME",
    "ResearchSocialEvidenceTool",
    "run_research_social_evidence_tool",
]
