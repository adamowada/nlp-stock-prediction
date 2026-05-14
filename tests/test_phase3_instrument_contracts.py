from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    AssetClass,
    AuditArtifact,
    FreshnessStatus,
    Instrument,
    InstrumentDataAvailability,
    InstrumentQuery,
    InstrumentResolution,
    InstrumentResolutionStatus,
    InstrumentUniverse,
    InstrumentUniverseRequest,
    ProviderInstrumentId,
    RelatedInstrument,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TradabilityEvidence,
    TradabilityStatus,
    Watchlist,
    WatchlistEntry,
)

pytestmark = pytest.mark.schema


def _now() -> datetime:
    return datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _tradability() -> TradabilityEvidence:
    return TradabilityEvidence(
        provider="example-broker-directory",
        status=TradabilityStatus.AVAILABLE,
        retrieved_at=_now(),
        source_url="https://example.test/instruments/TSLA",
        raw_identifier="example:equity:TSLA",
        notes="Retail-searchable equity listing observed.",
    )


def _instrument(
    *,
    instrument_id: str = "instrument-us-equity-tsla",
    symbol: str = "tsla",
    asset_class: AssetClass = AssetClass.STOCK,
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        display_name="Tesla Inc.",
        asset_class=asset_class,
        venue="NASDAQ",
        aliases=("TSLA", "tesla"),
        provider_ids=(
            ProviderInstrumentId(
                provider="alpha-vantage",
                namespace="symbol",
                identifier="TSLA",
                url="https://www.alphavantage.co/query?symbol=TSLA",
            ),
        ),
        related_instruments=(
            RelatedInstrument(
                instrument_id="instrument-etf-xly",
                relationship="sector_proxy",
                rationale="Consumer discretionary ETF can provide sector context.",
                evidence_ids=("evidence-sector-proxy-xly",),
            ),
        ),
        tradability_evidence=(_tradability(),),
        data_availability=(
            InstrumentDataAvailability(
                provider="alpha-vantage",
                data_type="daily_ohlcv",
                status=TradabilityStatus.AVAILABLE,
                checked_at=_now(),
                provider_identifier="TSLA",
            ),
        ),
        metadata={"phase": "phase3"},
    )


@pytest.mark.parametrize(
    ("symbol", "asset_class"),
    (
        ("TSLA", AssetClass.STOCK),
        ("XLY", AssetClass.ETF),
        ("btc/usd", AssetClass.CRYPTO),
        ("eur_usd", AssetClass.CURRENCY),
        ("GC", AssetClass.COMMODITY),
        ("ESM6", AssetClass.FUTURES),
        ("VTSAX", AssetClass.FUND),
        ("SPX", AssetClass.INDEX),
        ("semiconductor-proxy", AssetClass.PROXY),
        ("unknown:fixture", AssetClass.UNKNOWN),
    ),
)
def test_instrument_contract_preserves_broad_identity_and_availability(
    symbol: str,
    asset_class: AssetClass,
) -> None:
    instrument = _instrument(symbol=symbol, asset_class=asset_class)
    dumped = instrument.model_dump(mode="json")

    assert instrument.symbol == symbol.upper()
    assert instrument.asset_class == asset_class
    assert instrument.aliases == ("TSLA", "TESLA")
    assert dumped["tradability_evidence"][0]["status"] == "available"
    assert dumped["data_availability"][0]["data_type"] == "daily_ohlcv"
    assert dumped["related_instruments"][0]["relationship"] == "sector_proxy"


def test_instrument_provider_ids_are_unique_per_provider_namespace() -> None:
    payload = _instrument().model_dump()
    duplicate = ProviderInstrumentId(
        provider="alpha-vantage",
        namespace="symbol",
        identifier="TSLA.US",
    ).model_dump()

    with pytest.raises(ValidationError, match="provider ids must be unique"):
        Instrument.model_validate(
            {
                **payload,
                "provider_ids": (*payload["provider_ids"], duplicate),
            }
        )


def test_provider_identifier_uniqueness_allows_distinct_namespaces() -> None:
    instrument = Instrument.model_validate(
        {
            **_instrument().model_dump(mode="python"),
            "provider_ids": (
                ProviderInstrumentId(
                    provider="sec-edgar",
                    namespace="ticker",
                    identifier="TSLA",
                ),
                ProviderInstrumentId(
                    provider="sec-edgar",
                    namespace="cik",
                    identifier="1318605",
                ),
            ),
        }
    )

    assert {provider_id.namespace for provider_id in instrument.provider_ids} == {"ticker", "cik"}


def test_symbol_normalization_and_alias_dedupe_are_stable() -> None:
    instrument = _instrument(symbol=" brk.b ", asset_class=AssetClass.STOCK).model_copy(
        update={"aliases": ("brk.b", "BRK.B", "brk-b", "BRK-B")}
    )

    instrument = Instrument.model_validate(instrument.model_dump(mode="python"))

    assert instrument.symbol == "BRK.B"
    assert instrument.aliases == ("BRK.B", "BRK-B")


def test_tradability_evidence_requires_traceable_source() -> None:
    with pytest.raises(ValidationError, match="source_url, permalink, or raw_identifier"):
        TradabilityEvidence(
            provider="example-broker-directory",
            status=TradabilityStatus.UNKNOWN,
            retrieved_at=_now(),
        )


def test_instrument_resolution_makes_ambiguity_explicit() -> None:
    stock = _instrument(instrument_id="instrument-us-equity-ai", symbol="AI")
    token = _instrument(
        instrument_id="instrument-crypto-ai",
        symbol="AI",
        asset_class=AssetClass.CRYPTO,
    )

    ambiguous = InstrumentResolution(
        query="AI",
        status=InstrumentResolutionStatus.AMBIGUOUS,
        matches=(stock, token),
        warnings=("AI resolves to multiple retail-accessible instruments.",),
    )

    assert ambiguous.selected_instrument_id is None
    assert len(ambiguous.matches) == 2

    resolved = InstrumentResolution(
        query="TSLA",
        status=InstrumentResolutionStatus.RESOLVED,
        matches=(_instrument(),),
        selected_instrument_id="instrument-us-equity-tsla",
    )

    assert resolved.selected_instrument_id == "instrument-us-equity-tsla"

    with pytest.raises(ValidationError, match="at least two matches"):
        InstrumentResolution(
            query="AI",
            status=InstrumentResolutionStatus.AMBIGUOUS,
            matches=(stock,),
        )

    with pytest.raises(ValidationError, match="selected_instrument_id"):
        InstrumentResolution(
            query="TSLA",
            status=InstrumentResolutionStatus.RESOLVED,
            matches=(_instrument(),),
        )


def test_instrument_resolution_rejects_selected_id_mismatch() -> None:
    with pytest.raises(ValidationError, match="selected_instrument_id"):
        InstrumentResolution(
            query="TSLA",
            status=InstrumentResolutionStatus.RESOLVED,
            matches=(_instrument(),),
            selected_instrument_id="instrument-us-equity-nvda",
        )


@pytest.mark.parametrize(
    "status",
    (InstrumentResolutionStatus.UNSUPPORTED, InstrumentResolutionStatus.UNAVAILABLE),
)
def test_unresolved_resolution_shapes_do_not_select_instruments(
    status: InstrumentResolutionStatus,
) -> None:
    resolution = InstrumentResolution(
        query="OTC:MISSING",
        status=status,
        warnings=("Provider did not return a retail-accessible listing.",),
    )

    assert resolution.matches == ()
    assert resolution.selected_instrument_id is None

    with pytest.raises(ValidationError, match="cannot select"):
        InstrumentResolution(
            query="OTC:MISSING",
            status=status,
            selected_instrument_id="instrument-missing",
        )

    with pytest.raises(ValidationError, match="cannot include matches"):
        InstrumentResolution(
            query="OTC:MISSING",
            status=status,
            matches=(_instrument(),),
        )


def test_watchlist_entries_dedupe_aliases_and_require_entries() -> None:
    watchlist = Watchlist(
        watchlist_id="watchlist-core",
        name="Core research list",
        entries=(
            WatchlistEntry(
                query=InstrumentQuery(
                    query=" tsla ",
                    asset_class=AssetClass.STOCK,
                    aliases=("Tesla", "tesla", "$TSLA"),
                ),
                notes="Retail equity fixture.",
                tags=("ev", "ev", "large-cap"),
            ),
        ),
    )

    assert watchlist.entries[0].query.query == "TSLA"
    assert watchlist.entries[0].query.aliases == ("TESLA", "$TSLA")
    assert watchlist.entries[0].tags == ("ev", "large-cap")

    with pytest.raises(ValidationError, match="at least one entry"):
        Watchlist(watchlist_id="empty", name="Empty", entries=())


def test_universe_request_requires_query_or_watchlist_and_rejects_duplicate_watchlists() -> None:
    request = InstrumentUniverseRequest(
        request_id="universe-request-1",
        as_of=_now(),
        queries=(InstrumentQuery(query="BTC/USD", asset_class=AssetClass.CRYPTO),),
        allowed_asset_classes=(AssetClass.STOCK, AssetClass.CRYPTO),
    )

    assert request.queries[0].query == "BTC/USD"

    with pytest.raises(ValidationError, match="at least one query or watchlist"):
        InstrumentUniverseRequest(request_id="empty", as_of=_now())

    watchlist = Watchlist(
        watchlist_id="dupe",
        name="Duplicate fixture",
        entries=(WatchlistEntry(query=InstrumentQuery(query="TSLA")),),
    )
    with pytest.raises(ValidationError, match="watchlist ids must be unique"):
        InstrumentUniverseRequest(
            request_id="dupe-watchlists",
            as_of=_now(),
            watchlists=(watchlist, watchlist),
        )


def test_universe_result_validates_selected_and_materialized_instrument_ids() -> None:
    instrument = _instrument()
    universe = InstrumentUniverse(
        request_id="universe-request-1",
        generated_at=_now(),
        resolutions=(
            InstrumentResolution(
                query="TSLA",
                status=InstrumentResolutionStatus.RESOLVED,
                matches=(instrument,),
                selected_instrument_id=instrument.instrument_id,
            ),
        ),
        instruments=(instrument,),
    )

    assert universe.instrument_ids == ("instrument-us-equity-tsla",)
    round_tripped = InstrumentUniverse.model_validate(universe.model_dump(mode="json"))
    assert round_tripped.instrument_ids == universe.instrument_ids

    with pytest.raises(ValidationError, match="selected resolution ids must be materialized"):
        InstrumentUniverse(
            request_id="universe-request-1",
            generated_at=_now(),
            resolutions=(
                InstrumentResolution(
                    query="NVDA",
                    status=InstrumentResolutionStatus.RESOLVED,
                    matches=(
                        _instrument(instrument_id="instrument-us-equity-nvda", symbol="NVDA"),
                    ),
                    selected_instrument_id="instrument-us-equity-nvda",
                ),
            ),
            instruments=(instrument,),
        )


def test_universe_result_can_represent_only_unsupported_queries() -> None:
    universe = InstrumentUniverse(
        request_id="universe-request-unsupported",
        generated_at=_now(),
        resolutions=(
            InstrumentResolution(
                query="OTC:MISSING",
                status=InstrumentResolutionStatus.UNSUPPORTED,
                warnings=("No supported adapter for this instrument class.",),
            ),
        ),
        instruments=(),
    )

    assert universe.instrument_ids == ()
    assert universe.resolutions[0].status == InstrumentResolutionStatus.UNSUPPORTED

    with pytest.raises(ValidationError, match="at least one instrument or resolution"):
        InstrumentUniverse(
            request_id="universe-request-empty",
            generated_at=_now(),
            instruments=(),
        )


def test_source_evidence_tracks_generic_instrument_ids_with_ticker_compatibility() -> None:
    evidence = SourceEvidence(
        evidence_id="evidence-tsla-sector",
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="tsla",
        text="Tesla and XLY were both discussed in the same sourced article.",
        matched_tickers=("tsla", "xly", "TSLA"),
        instrument_id="instrument-us-equity-tsla",
        matched_instrument_ids=(
            "instrument-us-equity-tsla",
            "instrument-etf-xly",
            "instrument-etf-xly",
        ),
        provenance=SourceProvenance(
            provider_name="fixture-news",
            source_kind=SourceKind.NEWS_ARTICLE,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=_now(),
            observed_at=_now(),
            source_url="https://example.test/tsla-xly",
            raw_identifier="fixture-tsla-xly",
            raw_snapshot_id="raw-fixture-tsla-xly",
            freshness_status="fresh",
        ),
    )

    assert evidence.ticker == "TSLA"
    assert evidence.matched_tickers == ("TSLA", "XLY")
    assert evidence.instrument_id == "instrument-us-equity-tsla"
    assert evidence.matched_instrument_ids == ("instrument-us-equity-tsla", "instrument-etf-xly")


def test_source_evidence_rejects_source_kind_mismatch() -> None:
    with pytest.raises(ValidationError, match="source_kind must match"):
        SourceEvidence(
            evidence_id="evidence-mismatch",
            source_kind=SourceKind.NEWS_ARTICLE,
            text="A source kind mismatch should not validate.",
            provenance=SourceProvenance(
                provider_name="fixture-reddit",
                source_kind=SourceKind.REDDIT_POST,
                retrieval_method=RetrievalMethod.FIXTURE,
                fetched_at=_now(),
                source_url="https://example.test/mismatch",
                raw_identifier="mismatch",
                raw_snapshot_id="raw-mismatch",
                freshness_status=FreshnessStatus.FRESH,
            ),
        )


def test_json_metadata_rejects_augmented_assignment_mutation() -> None:
    evidence = SourceEvidence(
        evidence_id="evidence-frozen-json",
        source_kind=SourceKind.NEWS_ARTICLE,
        text="Metadata mutation should be rejected.",
        provenance=SourceProvenance(
            provider_name="fixture-news",
            source_kind=SourceKind.NEWS_ARTICLE,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=_now(),
            source_url="https://example.test/frozen",
            raw_identifier="frozen",
            raw_snapshot_id="raw-frozen",
            freshness_status=FreshnessStatus.FRESH,
        ),
        metadata={"tags": ["initial"]},
    )

    with pytest.raises(TypeError, match="immutable"):
        evidence.metadata.__ior__({"extra": True})
    tags = evidence.metadata["tags"]
    assert isinstance(tags, list)
    with pytest.raises(TypeError, match="immutable"):
        tags += ["extra"]


def test_audit_artifact_accepts_instrument_universe_type() -> None:
    artifact = AuditArtifact(
        artifact_id="artifact-instrument-universe",
        artifact_type="instrument_universe",
        path="reports/2026-05-13/instrument-universe.json",
        created_at=_now(),
        produced_by="instrument-universe-tool",
        record_count=1,
    )

    assert artifact.artifact_type == "instrument_universe"
