"""Deterministic fixture data for the dummy orchestration path."""

from __future__ import annotations

from nlp_stock_prediction.contracts import (
    EvidenceReference,
    FreshnessStatus,
    Instrument,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
)
from nlp_stock_prediction.contracts.instruments import InstrumentUniverse
from nlp_stock_prediction.orchestration.context import RunContext
from nlp_stock_prediction.orchestration.instrument_universe import build_fixture_universe


def dummy_instrument_universe(context: RunContext) -> InstrumentUniverse:
    return build_fixture_universe(
        request_id=f"instrument_universe-fixture-universe-{context.run_id}",
        generated_at=context.generated_at,
        primary_symbol="TSLA",
    )


def dummy_instruments(context: RunContext) -> tuple[Instrument, ...]:
    return dummy_instrument_universe(context).instruments


def dummy_evidence_sources(context: RunContext) -> tuple[SourceEvidence, ...]:
    return (
        _evidence(
            context=context,
            evidence_id="dummy-news-tsla-001",
            source_kind=SourceKind.NEWS_ARTICLE,
            ticker="TSLA",
            title="Dummy TSLA delivery and margin context",
            text=(
                "Dummy source describes TSLA delivery uncertainty, margin pressure, and "
                "headline sensitivity without asserting a trade action."
            ),
            source_url="https://example.com/dummy/tsla-delivery-context",
        ),
        _evidence(
            context=context,
            evidence_id="dummy-market-spy-001",
            source_kind=SourceKind.MARKET_DATA,
            ticker="SPY",
            title="Dummy SPY regime context",
            text=(
                "Dummy source describes broad-index regime risk that could outweigh "
                "single-name narratives."
            ),
            source_url="https://example.com/dummy/spy-regime-context",
        ),
        _evidence(
            context=context,
            evidence_id="dummy-crypto-btc-001",
            source_kind=SourceKind.MARKET_DATA,
            ticker=None,
            title="Dummy BTC/USD data availability",
            text=(
                "Dummy source records BTC/USD as a researchable crypto pair when provider "
                "data is available."
            ),
            source_url="https://example.com/dummy/btc-usd-availability",
        ),
        _evidence(
            context=context,
            evidence_id="dummy-futures-esm6-001",
            source_kind=SourceKind.MARKET_DATA,
            ticker="ESM6",
            title="Dummy ESM6 futures context availability",
            text=(
                "Dummy source records E-mini S&P 500 futures as restricted context for "
                "macro and proxy analysis, not as a tradeable recommendation."
            ),
            source_url="https://example.com/dummy/esm6-context",
        ),
    )


def dummy_evidence_references() -> dict[str, EvidenceReference]:
    quotes = {
        "dummy-news-tsla-001": (
            "Dummy source describes TSLA delivery uncertainty, margin pressure, and headline "
            "sensitivity without asserting a trade action."
        ),
        "dummy-market-spy-001": (
            "Dummy source describes broad-index regime risk that could outweigh single-name "
            "narratives."
        ),
        "dummy-crypto-btc-001": (
            "Dummy source records BTC/USD as a researchable crypto pair when provider data is "
            "available."
        ),
        "dummy-futures-esm6-001": (
            "Dummy source records E-mini S&P 500 futures as restricted context for macro and "
            "proxy analysis, not as a tradeable recommendation."
        ),
    }
    return {
        evidence_id: EvidenceReference(evidence_id=evidence_id, quote=quote, relevance=0.8)
        for evidence_id, quote in quotes.items()
    }


def _evidence(
    *,
    context: RunContext,
    evidence_id: str,
    source_kind: SourceKind,
    ticker: str | None,
    title: str,
    text: str,
    source_url: str,
) -> SourceEvidence:
    raw_identifier = evidence_id
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=source_kind,
        ticker=ticker,
        title=title,
        text=text,
        created_at=context.generated_at,
        permalink=source_url,
        matched_tickers=((ticker,) if ticker else ()),
        provenance=SourceProvenance(
            provider_name="dummy-provider",
            source_kind=source_kind,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=context.generated_at,
            observed_at=context.generated_at,
            source_url=source_url,
            permalink=source_url,
            raw_identifier=raw_identifier,
            raw_snapshot_id=f"raw-{raw_identifier}",
            freshness_status=FreshnessStatus.FRESH,
            provider_metadata={"dummy": True},
        ),
        metadata={"dummy": True},
    )


__all__ = [
    "dummy_evidence_references",
    "dummy_evidence_sources",
    "dummy_instrument_universe",
    "dummy_instruments",
]
