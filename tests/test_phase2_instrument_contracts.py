from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    AssetClass,
    Instrument,
    InstrumentDataAvailability,
    InstrumentResolution,
    InstrumentResolutionStatus,
    ProviderInstrumentId,
    RelatedInstrument,
    TradabilityEvidence,
    TradabilityStatus,
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
        metadata={"phase": "phase2"},
    )


def test_instrument_contract_preserves_broad_identity_and_availability() -> None:
    instrument = _instrument(symbol="btc/usd", asset_class=AssetClass.CRYPTO)
    dumped = instrument.model_dump(mode="json")

    assert instrument.symbol == "BTC/USD"
    assert instrument.asset_class == AssetClass.CRYPTO
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
