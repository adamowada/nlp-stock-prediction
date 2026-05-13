"""Fixture-backed Phase 3 instrument universe discovery."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import cast

from nlp_stock_prediction.contracts import (
    AssetClass,
    Instrument,
    InstrumentResolution,
    InstrumentResolutionStatus,
    InstrumentUniverse,
    JsonObject,
    TradabilityStatus,
)
from nlp_stock_prediction.contracts.instruments import (
    InstrumentDataAvailability,
    ProviderInstrumentId,
    TradabilityEvidence,
)
from nlp_stock_prediction.storage import InstrumentRecord

PHASE3_UNIVERSE_SCHEMA_VERSION = "phase3.instrument-universe.v1"
PHASE3_FIXTURE_PROVIDER = "phase3-fixture-directory"


def build_phase3_fixture_universe(
    *,
    request_id: str,
    generated_at: datetime,
    primary_symbol: str = "TSLA",
    primary_instrument_id: str | None = None,
) -> InstrumentUniverse:
    """Return a deterministic mixed-asset universe with explicit resolutions."""

    primary = _fixture_instrument(
        instrument_id=primary_instrument_id or "instrument:equity:us:tsla",
        symbol=primary_symbol,
        display_name=_display_name(primary_symbol),
        asset_class=_asset_class_for_symbol(primary_symbol),
        venue=_venue_for_symbol(primary_symbol),
        provider_identifier=primary_symbol,
        generated_at=generated_at,
        aliases=_aliases_for_symbol(primary_symbol),
        notes="Primary fixture instrument requested for this research run.",
    )
    spy = _fixture_instrument(
        instrument_id="instrument:etf:us:spy",
        symbol="SPY",
        display_name="SPDR S&P 500 ETF Trust",
        asset_class=AssetClass.ETF,
        venue="NYSEARCA",
        provider_identifier="SPY",
        generated_at=generated_at,
        aliases=("SPY.US",),
        notes="Broad-market ETF fixture used as index proxy context.",
    )
    btc = _fixture_instrument(
        instrument_id="instrument:crypto:btc-usd",
        symbol="BTC/USD",
        display_name="Bitcoin versus U.S. dollar",
        asset_class=AssetClass.CRYPTO,
        venue="fixture-crypto-venues",
        provider_identifier="BTC/USD",
        generated_at=generated_at,
        aliases=("BTC-USD", "BTC:USD"),
        notes="Crypto pair fixture with provider data availability only.",
    )
    futures = _fixture_instrument(
        instrument_id="instrument:futures:cme:esm6",
        symbol="ESM6",
        display_name="E-mini S&P 500 June 2026 futures context",
        asset_class=AssetClass.FUTURES,
        venue="CME",
        provider_identifier="ESM6",
        generated_at=generated_at,
        aliases=("ESM26",),
        tradability_status=TradabilityStatus.RESTRICTED,
        notes=(
            "Futures record is included as macro/proxy context only; availability does not "
            "imply retail trading access."
        ),
    )
    ai_stock = _fixture_instrument(
        instrument_id="instrument:equity:us:ai",
        symbol="AI",
        display_name="C3.ai Inc.",
        asset_class=AssetClass.STOCK,
        venue="NYSE",
        provider_identifier="AI",
        generated_at=generated_at,
        notes="Ambiguity fixture for the AI query.",
    )
    ai_token = _fixture_instrument(
        instrument_id="instrument:crypto:ai-usd",
        symbol="AI/USD",
        display_name="AI token versus U.S. dollar fixture",
        asset_class=AssetClass.CRYPTO,
        venue="fixture-crypto-venues",
        provider_identifier="AI/USD",
        generated_at=generated_at,
        notes="Ambiguity fixture for the AI query.",
    )
    instruments = _dedupe_instruments((primary, spy, btc, futures))
    warnings = (
        "AI query is ambiguous across an equity and a crypto fixture; no default selection made.",
        "PRIVATE:SPACEX is unsupported because no retail-accessible instrument fixture exists.",
    )
    resolutions = _dedupe_resolutions(
        _resolved_resolution(primary.symbol, primary),
        _resolved_resolution("SPY", spy),
        _resolved_resolution("BTC/USD", btc),
        _resolved_resolution("ESM6", futures),
        InstrumentResolution(
            query="AI",
            status=InstrumentResolutionStatus.AMBIGUOUS,
            matches=(ai_stock, ai_token),
            warnings=(warnings[0],),
            metadata={"fixture": True, "reason": "symbol_collision"},
        ),
        InstrumentResolution(
            query="PRIVATE:SPACEX",
            status=InstrumentResolutionStatus.UNSUPPORTED,
            warnings=(warnings[1],),
            metadata={"fixture": True, "reason": "private_company"},
        ),
    )
    status_counts = Counter(resolution.status.value for resolution in resolutions)
    return InstrumentUniverse(
        request_id=request_id,
        generated_at=generated_at,
        resolutions=resolutions,
        instruments=instruments,
        warnings=warnings,
        metadata={
            "schema_version": PHASE3_UNIVERSE_SCHEMA_VERSION,
            "provider": PHASE3_FIXTURE_PROVIDER,
            "source": "fixture",
            "summary": (
                "Fixture universe includes stock, ETF, crypto, and futures context with "
                "explicit ambiguity and unsupported-query records."
            ),
            "resolution_status_counts": dict(sorted(status_counts.items())),
        },
    )


def phase3_universe_artifact_payload(*, run_id: str, universe: InstrumentUniverse) -> JsonObject:
    """Serialize a full universe artifact with a stable top-level envelope."""

    return cast(
        JsonObject,
        {
            "schema_version": PHASE3_UNIVERSE_SCHEMA_VERSION,
            "run_id": run_id,
            "universe_id": universe.request_id,
            "universe": universe.model_dump(mode="json"),
        },
    )


def instrument_record_from_contract(instrument: Instrument) -> InstrumentRecord:
    """Convert a public Instrument contract into the SQLite registry record."""

    return InstrumentRecord(
        instrument_id=instrument.instrument_id,
        symbol=instrument.symbol,
        asset_class=instrument.asset_class.value,
        name=instrument.display_name,
        venue=instrument.venue,
        aliases=instrument.aliases,
        provider_ids=tuple(
            cast(JsonObject, provider_id.model_dump(mode="json"))
            for provider_id in instrument.provider_ids
        ),
        related_instruments=tuple(
            cast(JsonObject, related.model_dump(mode="json"))
            for related in instrument.related_instruments
        ),
        tradability_evidence=tuple(
            cast(JsonObject, evidence.model_dump(mode="json"))
            for evidence in instrument.tradability_evidence
        ),
        data_availability=tuple(
            cast(JsonObject, availability.model_dump(mode="json"))
            for availability in instrument.data_availability
        ),
        metadata={**instrument.metadata, "phase3_universe": True},
    )


def _resolved_resolution(query: str, instrument: Instrument) -> InstrumentResolution:
    return InstrumentResolution(
        query=query,
        status=InstrumentResolutionStatus.RESOLVED,
        matches=(instrument,),
        selected_instrument_id=instrument.instrument_id,
        metadata={"fixture": True},
    )


def _fixture_instrument(
    *,
    instrument_id: str,
    symbol: str,
    display_name: str,
    asset_class: AssetClass,
    venue: str | None,
    provider_identifier: str,
    generated_at: datetime,
    aliases: tuple[str, ...] = (),
    tradability_status: TradabilityStatus = TradabilityStatus.UNKNOWN,
    notes: str,
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        display_name=display_name,
        asset_class=asset_class,
        venue=venue,
        aliases=aliases,
        provider_ids=(
            ProviderInstrumentId(
                provider=PHASE3_FIXTURE_PROVIDER,
                identifier=provider_identifier.upper(),
                namespace="fixture-symbol",
                url=f"https://example.com/fixtures/instruments/{provider_identifier.lower()}",
                metadata={"source": "phase3_fixture"},
            ),
        ),
        tradability_evidence=(
            TradabilityEvidence(
                provider=PHASE3_FIXTURE_PROVIDER,
                status=tradability_status,
                retrieved_at=generated_at,
                source_url=f"https://example.com/fixtures/instruments/{provider_identifier.lower()}",
                raw_identifier=f"{provider_identifier.upper()}:phase3-fixture",
                notes=notes,
                metadata={"source": "phase3_fixture"},
            ),
        ),
        data_availability=(
            InstrumentDataAvailability(
                provider=PHASE3_FIXTURE_PROVIDER,
                data_type="fixture_research_context",
                status=TradabilityStatus.AVAILABLE,
                checked_at=generated_at,
                provider_identifier=provider_identifier.upper(),
                notes="Fixture data is deterministic and offline.",
                metadata={"source": "phase3_fixture"},
            ),
        ),
        metadata={"phase": "phase3", "fixture": True},
    )


def _asset_class_for_symbol(symbol: str) -> AssetClass:
    normalized = symbol.strip().upper()
    if normalized in {"BTC/USD", "BTC:USD", "BTC-USD"}:
        return AssetClass.CRYPTO
    if normalized in {"EUR/USD", "EUR:USD", "USD/JPY", "USD:JPY"}:
        return AssetClass.CURRENCY
    if normalized in {"GC", "GC/USD", "CL", "CL/USD"}:
        return AssetClass.COMMODITY
    if normalized in {"ESM6", "ESM26"}:
        return AssetClass.FUTURES
    return AssetClass.STOCK


def _venue_for_symbol(symbol: str) -> str | None:
    asset_class = _asset_class_for_symbol(symbol)
    if asset_class == AssetClass.CRYPTO:
        return "fixture-crypto-venues"
    if asset_class == AssetClass.CURRENCY:
        return "fixture-fx-venues"
    if asset_class == AssetClass.FUTURES:
        return "CME"
    if asset_class == AssetClass.COMMODITY:
        return "fixture-commodity-context"
    return "NASDAQ" if symbol.strip().upper() == "TSLA" else None


def _display_name(symbol: str) -> str:
    normalized = symbol.strip().upper()
    names = {
        "TSLA": "Tesla Inc.",
        "BTC/USD": "Bitcoin versus U.S. dollar",
        "BTC:USD": "Bitcoin versus U.S. dollar",
        "EUR/USD": "Euro versus U.S. dollar",
        "ESM6": "E-mini S&P 500 June 2026 futures context",
    }
    return names.get(normalized, f"{normalized} fixture instrument")


def _aliases_for_symbol(symbol: str) -> tuple[str, ...]:
    normalized = symbol.strip().upper()
    if normalized == "TSLA":
        return ("TESLA",)
    if normalized in {"BTC/USD", "BTC:USD"}:
        return ("BTC-USD", "BTC:USD", "BTC/USD")
    return ()


def _dedupe_instruments(instruments: tuple[Instrument, ...]) -> tuple[Instrument, ...]:
    records: dict[str, Instrument] = {}
    for instrument in instruments:
        records.setdefault(instrument.instrument_id, instrument)
    return tuple(records.values())


def _dedupe_resolutions(*resolutions: InstrumentResolution) -> tuple[InstrumentResolution, ...]:
    records: dict[str, InstrumentResolution] = {}
    for resolution in resolutions:
        records.setdefault(resolution.query, resolution)
    return tuple(records.values())


__all__ = [
    "PHASE3_FIXTURE_PROVIDER",
    "PHASE3_UNIVERSE_SCHEMA_VERSION",
    "build_phase3_fixture_universe",
    "instrument_record_from_contract",
    "phase3_universe_artifact_payload",
]
