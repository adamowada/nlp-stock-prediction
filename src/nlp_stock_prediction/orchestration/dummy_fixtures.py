"""Deterministic fixture data for the dummy orchestration path."""

from __future__ import annotations

from nlp_stock_prediction.contracts import (
    AssetClass,
    EvidenceReference,
    FreshnessStatus,
    Instrument,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TradabilityStatus,
)
from nlp_stock_prediction.contracts.instruments import (
    InstrumentDataAvailability,
    ProviderInstrumentId,
    TradabilityEvidence,
)
from nlp_stock_prediction.orchestration.context import RunContext


def dummy_instruments(context: RunContext) -> tuple[Instrument, ...]:
    return (
        _instrument(
            context=context,
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            display_name="Tesla Inc.",
            asset_class=AssetClass.STOCK,
            provider_identifier="TSLA",
        ),
        _instrument(
            context=context,
            instrument_id="instrument:etf:us:spy",
            symbol="SPY",
            display_name="SPDR S&P 500 ETF Trust",
            asset_class=AssetClass.ETF,
            provider_identifier="SPY",
        ),
        _instrument(
            context=context,
            instrument_id="instrument:crypto:btc-usd",
            symbol="BTC/USD",
            display_name="Bitcoin versus U.S. dollar",
            asset_class=AssetClass.CRYPTO,
            provider_identifier="BTC/USD",
        ),
    )


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
    }
    return {
        evidence_id: EvidenceReference(evidence_id=evidence_id, quote=quote, relevance=0.8)
        for evidence_id, quote in quotes.items()
    }


def _instrument(
    *,
    context: RunContext,
    instrument_id: str,
    symbol: str,
    display_name: str,
    asset_class: AssetClass,
    provider_identifier: str,
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        display_name=display_name,
        asset_class=asset_class,
        provider_ids=(
            ProviderInstrumentId(
                provider="dummy-provider",
                identifier=provider_identifier,
                namespace="dummy-symbol",
            ),
        ),
        tradability_evidence=(
            TradabilityEvidence(
                provider="dummy-provider",
                status=TradabilityStatus.UNKNOWN,
                retrieved_at=context.generated_at,
                raw_identifier=f"{provider_identifier}:dummy-tradability",
                notes="Dummy records availability context only.",
            ),
        ),
        data_availability=(
            InstrumentDataAvailability(
                provider="dummy-provider",
                data_type="dummy-research",
                status=TradabilityStatus.AVAILABLE,
                checked_at=context.generated_at,
                provider_identifier=provider_identifier,
            ),
        ),
        metadata={"dummy": True},
    )


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


__all__ = ["dummy_evidence_references", "dummy_evidence_sources", "dummy_instruments"]
