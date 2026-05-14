"""Instrument persistence seam used by registry resolution."""

from __future__ import annotations

from typing import Protocol, cast

from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, JsonValue
from nlp_stock_prediction.contracts.enums import AssetClass
from nlp_stock_prediction.contracts.instruments import (
    Instrument,
    InstrumentDataAvailability,
    ProviderInstrumentId,
    RelatedInstrument,
    TradabilityEvidence,
)
from nlp_stock_prediction.storage.records import InstrumentRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore


class InstrumentRepository(Protocol):
    """Contract-shaped persistence Interface for instrument lookup and indexing."""

    def initialize(self) -> None: ...

    def upsert(self, instrument: Instrument, *, metadata: JsonObject | None = None) -> None: ...

    def get(self, instrument_id: str) -> Instrument | None: ...

    def find_by_symbol_or_alias(self, query: str) -> tuple[Instrument, ...]: ...

    def find_by_provider_id(
        self,
        *,
        provider: str | None,
        namespace: str | None,
        identifier: str,
        require_namespace: bool,
    ) -> tuple[Instrument, ...]: ...

    def list_instruments(self) -> tuple[Instrument, ...]: ...


class SQLiteInstrumentRepository:
    """SQLite Adapter for the instrument repository Interface."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    @property
    def store(self) -> SQLiteStore:
        return self._store

    def initialize(self) -> None:
        self._store.initialize()

    def upsert(self, instrument: Instrument, *, metadata: JsonObject | None = None) -> None:
        self._store.upsert_instrument(instrument_to_record(instrument, metadata=metadata))

    def get(self, instrument_id: str) -> Instrument | None:
        record = self._store.get_instrument(instrument_id)
        if record is None:
            return None
        return instrument_from_record(record)

    def find_by_symbol_or_alias(self, query: str) -> tuple[Instrument, ...]:
        return tuple(
            instrument_from_record(record)
            for record in self._store.find_instruments_by_symbol_or_alias(query)
        )

    def find_by_provider_id(
        self,
        *,
        provider: str | None,
        namespace: str | None,
        identifier: str,
        require_namespace: bool,
    ) -> tuple[Instrument, ...]:
        if provider is not None and require_namespace:
            record = self._store.find_instrument_by_provider_id(
                provider,
                namespace or "default",
                identifier,
            )
            return () if record is None else (instrument_from_record(record),)
        return tuple(
            instrument
            for instrument in self.list_instruments()
            if any(
                _provider_id_matches(
                    provider_id,
                    provider=provider,
                    namespace=namespace,
                    identifier=identifier,
                    require_namespace=require_namespace,
                )
                for provider_id in instrument.provider_ids
            )
        )

    def list_instruments(self) -> tuple[Instrument, ...]:
        return tuple(instrument_from_record(record) for record in self._store.list_instruments())


def instrument_to_record(
    instrument: Instrument,
    *,
    metadata: JsonObject | None = None,
) -> InstrumentRecord:
    """Convert a validated instrument contract into the SQLite persistence record."""

    validated = Instrument.model_validate(instrument)
    return InstrumentRecord(
        instrument_id=validated.instrument_id,
        symbol=validated.symbol,
        asset_class=validated.asset_class.value,
        name=validated.display_name,
        venue=validated.venue,
        aliases=validated.aliases,
        provider_ids=tuple(_dump_contract(item) for item in validated.provider_ids),
        related_instruments=tuple(_dump_contract(item) for item in validated.related_instruments),
        tradability_evidence=tuple(_dump_contract(item) for item in validated.tradability_evidence),
        data_availability=tuple(_dump_contract(item) for item in validated.data_availability),
        metadata=validated.metadata if metadata is None else {**validated.metadata, **metadata},
    )


def instrument_from_record(record: InstrumentRecord) -> Instrument:
    """Convert a SQLite instrument record into the public Instrument contract."""

    return Instrument(
        instrument_id=record.instrument_id,
        symbol=record.symbol,
        display_name=record.name or record.symbol,
        asset_class=AssetClass(record.asset_class),
        venue=record.venue,
        aliases=record.aliases,
        provider_ids=tuple(
            ProviderInstrumentId.model_validate(item) for item in record.provider_ids
        ),
        related_instruments=tuple(
            RelatedInstrument.model_validate(item) for item in record.related_instruments
        ),
        tradability_evidence=tuple(
            TradabilityEvidence.model_validate(_normalize_tradability_record(item))
            for item in record.tradability_evidence
        ),
        data_availability=tuple(
            InstrumentDataAvailability.model_validate(_normalize_availability_record(item))
            for item in record.data_availability
        ),
        metadata=record.metadata,
    )


def _dump_contract(model: ContractModel) -> JsonObject:
    return cast(JsonObject, model.model_dump(mode="json"))


def _normalize_tradability_record(value: JsonObject) -> JsonObject:
    normalized = dict(value)
    if "source_url" not in normalized and isinstance(normalized.get("url"), str):
        normalized["source_url"] = normalized["url"]
    return normalized


def _normalize_availability_record(value: JsonObject) -> JsonObject:
    normalized = dict(value)
    if "checked_at" not in normalized and "as_of" in normalized:
        normalized["checked_at"] = normalized["as_of"]
    return normalized


def _provider_id_matches(
    provider_id: ProviderInstrumentId,
    *,
    provider: str | None,
    namespace: str | None,
    identifier: str | None,
    require_namespace: bool,
) -> bool:
    if provider is not None and _normalize_text(provider_id.provider) != _normalize_text(provider):
        return False
    if require_namespace and _normalize_namespace(provider_id.namespace) != _normalize_namespace(
        namespace
    ):
        return False
    return not (
        identifier is not None and not _identifier_matches(provider_id.identifier, identifier)
    )


def _identifier_matches(left: str, right: str) -> bool:
    return left == right or _normalize_text(left) == _normalize_text(right)


def _normalize_namespace(value: str | None) -> str:
    return _normalize_text(value or "default")


def _normalize_text(value: str | JsonValue | None) -> str:
    return str(value or "").strip().casefold()


__all__ = [
    "InstrumentRepository",
    "SQLiteInstrumentRepository",
    "instrument_from_record",
    "instrument_to_record",
]
