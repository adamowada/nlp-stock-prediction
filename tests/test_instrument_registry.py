from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import (
    AssetClass,
    Instrument,
    InstrumentDataAvailability,
    InstrumentQuery,
    InstrumentResolutionStatus,
    InstrumentUniverseRequest,
    ProviderInstrumentId,
    RelatedInstrument,
    TradabilityEvidence,
    TradabilityStatus,
    Watchlist,
    WatchlistEntry,
)
from nlp_stock_prediction.instruments import InstrumentRegistry, SQLiteInstrumentRepository
from nlp_stock_prediction.orchestration.instrument_universe import (
    INSTRUMENT_UNIVERSE_SCHEMA_VERSION,
    FixtureUniverseTool,
)
from nlp_stock_prediction.storage import SQLiteStore

pytestmark = pytest.mark.unit


def _now() -> datetime:
    return datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "prediction-research.sqlite3")


def _registry(tmp_path: Path) -> InstrumentRegistry:
    return InstrumentRegistry(_store(tmp_path))


def _instrument(
    *,
    instrument_id: str,
    symbol: str,
    display_name: str,
    asset_class: AssetClass,
    venue: str | None = None,
    aliases: tuple[str, ...] = (),
    provider_ids: tuple[ProviderInstrumentId, ...] = (),
    related_instruments: tuple[RelatedInstrument, ...] = (),
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        display_name=display_name,
        asset_class=asset_class,
        venue=venue,
        aliases=aliases,
        provider_ids=provider_ids,
        related_instruments=related_instruments,
        tradability_evidence=(
            TradabilityEvidence(
                provider="fixture-broker",
                status=TradabilityStatus.AVAILABLE,
                retrieved_at=_now(),
                source_url=f"https://example.test/tradability/{instrument_id}",
                raw_identifier=instrument_id,
            ),
        ),
        data_availability=(
            InstrumentDataAvailability(
                provider="fixture-market",
                data_type="daily_ohlcv",
                status=TradabilityStatus.AVAILABLE,
                checked_at=_now(),
                provider_identifier=symbol,
            ),
        ),
        metadata={"fixture": "instrument_universe-registry"},
    )


def _provider(
    provider: str,
    namespace: str,
    identifier: str,
) -> ProviderInstrumentId:
    return ProviderInstrumentId(
        provider=provider,
        namespace=namespace,
        identifier=identifier,
        url=f"https://example.test/{provider}/{namespace}/{identifier}",
    )


def test_registry_upsert_get_and_lookup_round_trip_contract_instrument(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    instrument = _instrument(
        instrument_id="equity:NASDAQ:TSLA",
        symbol="tsla",
        display_name="Tesla Inc.",
        asset_class=AssetClass.STOCK,
        venue="NASDAQ",
        aliases=("tesla", "TSLA-US"),
        provider_ids=(_provider("alpha-vantage", "symbol", "TSLA"),),
    )

    registry.upsert(instrument)

    expected = instrument.model_copy(update={"symbol": "TSLA", "aliases": ("TESLA", "TSLA-US")})
    stored = registry.get("equity:NASDAQ:TSLA")
    assert stored == expected
    assert registry.find_by_symbol_or_alias("Tesla") == (stored,)
    assert registry.find_by_provider_id("ALPHA-VANTAGE", "symbol", "TSLA") == (stored,)
    assert registry.find_by_provider_id("ALPHA-VANTAGE", None, "TSLA") == (stored,)


def test_sqlite_instrument_repository_is_contract_shaped_seam(tmp_path: Path) -> None:
    repository = SQLiteInstrumentRepository(_store(tmp_path))
    repository.initialize()
    instrument = _instrument(
        instrument_id="currency:EURUSD",
        symbol="EUR/USD",
        display_name="Euro / US Dollar",
        asset_class=AssetClass.CURRENCY,
        aliases=("EUR-USD", "EUR:USD"),
        provider_ids=(_provider("fixture-fx", "pair", "EUR/USD"),),
    )

    repository.upsert(instrument, metadata={"repository_test": True})

    stored = repository.get("currency:EURUSD")
    assert stored is not None
    assert stored.symbol == "EUR/USD"
    assert stored.metadata["repository_test"] is True
    assert repository.find_by_symbol_or_alias("eur-usd") == (stored,)
    assert repository.find_by_provider_id(
        provider="FIXTURE-FX",
        namespace="pair",
        identifier="EUR/USD",
        require_namespace=True,
    ) == (stored,)
    assert repository.find_by_provider_id(
        provider="FIXTURE-FX",
        namespace=None,
        identifier="EUR/USD",
        require_namespace=False,
    ) == (stored,)


def test_instrument_universe_fixture_universe_tool_returns_artifact_and_index_records() -> None:
    result = FixtureUniverseTool().run(
        request_id="fixture-universe-tool",
        generated_at=_now(),
        run_id="run-fixture-universe-tool",
        primary_symbol="BTC/USD",
        primary_instrument_id="instrument:codex:BTC-USD",
    )

    assert result.instrument_ids == result.universe.instrument_ids
    assert result.artifact_payload["schema_version"] == INSTRUMENT_UNIVERSE_SCHEMA_VERSION
    assert result.artifact_payload["run_id"] == "run-fixture-universe-tool"
    assert result.instrument_records[0].instrument_id == "instrument:codex:BTC-USD"
    assert result.instrument_records[0].metadata["instrument_universe"] is True
    assert {record.instrument_id for record in result.instrument_records} == set(
        result.universe.instrument_ids
    )


def test_resolve_returns_ambiguity_and_uses_hints_to_disambiguate(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    stock = _instrument(
        instrument_id="equity:NYSE:AI",
        symbol="AI",
        display_name="C3.ai Inc.",
        asset_class=AssetClass.STOCK,
        venue="NYSE",
        provider_ids=(_provider("fixture-market", "symbol", "AI"),),
    )
    token = _instrument(
        instrument_id="crypto:AI",
        symbol="AI",
        display_name="AI Token",
        asset_class=AssetClass.CRYPTO,
        provider_ids=(_provider("fixture-market", "crypto_symbol", "AI"),),
    )
    registry.upsert(stock)
    registry.upsert(token)

    ambiguous = registry.resolve(InstrumentQuery(query="AI"))
    assert ambiguous.status == InstrumentResolutionStatus.AMBIGUOUS
    assert tuple(instrument.instrument_id for instrument in ambiguous.matches) == (
        "crypto:AI",
        "equity:NYSE:AI",
    )
    assert ambiguous.warnings

    stock_resolution = registry.resolve(
        InstrumentQuery(query="AI", asset_class=AssetClass.STOCK, venue="nyse")
    )
    assert stock_resolution.status == InstrumentResolutionStatus.RESOLVED
    assert stock_resolution.selected_instrument_id == "equity:NYSE:AI"

    provider_ambiguous = registry.resolve(
        InstrumentQuery(
            query="AI",
            provider="fixture-market",
            provider_identifier="AI",
        )
    )
    assert provider_ambiguous.status == InstrumentResolutionStatus.AMBIGUOUS
    assert len(provider_ambiguous.matches) == 2


def test_resolve_universe_includes_watchlists_related_and_stable_deduped_selection(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    xly = _instrument(
        instrument_id="etf:NYSEARCA:XLY",
        symbol="XLY",
        display_name="Consumer Discretionary Select Sector SPDR Fund",
        asset_class=AssetClass.ETF,
        venue="NYSEARCA",
        provider_ids=(_provider("fixture-market", "symbol", "XLY"),),
    )
    tsla = _instrument(
        instrument_id="equity:NASDAQ:TSLA",
        symbol="TSLA",
        display_name="Tesla Inc.",
        asset_class=AssetClass.STOCK,
        venue="NASDAQ",
        aliases=("TESLA",),
        provider_ids=(_provider("fixture-market", "symbol", "TSLA"),),
        related_instruments=(
            RelatedInstrument(
                instrument_id="etf:NYSEARCA:XLY",
                relationship="sector_proxy",
                rationale="Fixture sector context for consumer discretionary exposure.",
            ),
        ),
    )
    qqq = _instrument(
        instrument_id="etf:NASDAQ:QQQ",
        symbol="QQQ",
        display_name="Invesco QQQ Trust",
        asset_class=AssetClass.ETF,
        venue="NASDAQ",
        provider_ids=(_provider("fixture-market", "symbol", "QQQ"),),
    )
    registry.upsert(xly)
    registry.upsert(tsla)
    registry.upsert(qqq)

    universe = registry.resolve_universe(
        InstrumentUniverseRequest(
            request_id="universe-fixture",
            as_of=_now(),
            queries=(InstrumentQuery(query="TSLA"),),
            watchlists=(
                Watchlist(
                    watchlist_id="watchlist-core",
                    name="Core",
                    entries=(
                        WatchlistEntry(
                            query=InstrumentQuery(
                                query="NASDAQ-100",
                                asset_class=AssetClass.ETF,
                            ),
                            requested_instrument_id="etf:NASDAQ:QQQ",
                        ),
                        WatchlistEntry(query=InstrumentQuery(query="TSLA")),
                    ),
                ),
            ),
        )
    )

    assert tuple(resolution.query for resolution in universe.resolutions) == (
        "TSLA",
        "NASDAQ-100",
    )
    assert universe.instrument_ids == (
        "equity:NASDAQ:TSLA",
        "etf:NASDAQ:QQQ",
        "etf:NYSEARCA:XLY",
    )
    assert any("Duplicate watchlist query" in warning for warning in universe.warnings)


def test_unavailable_and_unsupported_outcomes_are_explicit(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    missing = registry.resolve(InstrumentQuery(query="OTC:MISSING"))
    assert missing.status == InstrumentResolutionStatus.UNAVAILABLE
    assert missing.matches == ()
    assert missing.warnings

    registry.upsert(
        _instrument(
            instrument_id="crypto:BTC",
            symbol="BTC/USD",
            display_name="Bitcoin / US Dollar",
            asset_class=AssetClass.CRYPTO,
            provider_ids=(_provider("fixture-market", "pair", "BTC/USD"),),
        )
    )
    universe = registry.resolve_universe(
        InstrumentUniverseRequest(
            request_id="stock-only",
            as_of=_now(),
            queries=(InstrumentQuery(query="BTC/USD"),),
            allowed_asset_classes=(AssetClass.STOCK,),
        )
    )

    assert universe.resolutions[0].status == InstrumentResolutionStatus.UNSUPPORTED
    assert universe.instruments == ()
    assert universe.warnings
