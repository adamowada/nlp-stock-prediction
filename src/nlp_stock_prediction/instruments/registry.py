"""SQLite-backed instrument registry and resolver.

The registry intentionally sits above storage. SQLite records are a persistence shape; this module
normalizes every read and write through the public Pydantic contracts before resolver decisions are
made.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import cast

from nlp_stock_prediction.contracts import (
    AssetClass,
    Instrument,
    InstrumentDataAvailability,
    InstrumentQuery,
    InstrumentResolution,
    InstrumentResolutionStatus,
    InstrumentUniverse,
    InstrumentUniverseRequest,
    ProviderInstrumentId,
    RelatedInstrument,
    TradabilityEvidence,
    WatchlistEntry,
)
from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, JsonValue
from nlp_stock_prediction.storage import InstrumentRecord, SQLiteStore


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


class InstrumentRegistry:
    """Contract-aware registry and resolver backed by ``SQLiteStore``."""

    def __init__(self, store: SQLiteStore, *, initialize: bool = True) -> None:
        self._store = store
        if initialize:
            self._store.initialize()

    @property
    def store(self) -> SQLiteStore:
        return self._store

    def upsert(self, instrument: Instrument) -> Instrument:
        validated = Instrument.model_validate(instrument)
        self._store.upsert_instrument(instrument_to_record(validated))
        return validated

    def get(self, instrument_id: str) -> Instrument | None:
        record = self._store.get_instrument(instrument_id)
        if record is None:
            return None
        return instrument_from_record(record)

    def find_by_symbol_or_alias(self, query: str) -> tuple[Instrument, ...]:
        query_model = InstrumentQuery(query=query)
        return tuple(
            instrument_from_record(record)
            for record in self._store.find_instruments_by_symbol_or_alias(query_model.query)
        )

    def find_by_provider_id(
        self,
        provider: str,
        namespace: str | None,
        identifier: str,
    ) -> tuple[Instrument, ...]:
        provider_id = ProviderInstrumentId(
            provider=provider,
            namespace=namespace,
            identifier=identifier,
        )
        return self._find_by_provider_id(
            provider=provider_id.provider,
            namespace=provider_id.namespace,
            identifier=provider_id.identifier,
            require_namespace=provider_id.namespace is not None,
        )

    def resolve(self, query: InstrumentQuery) -> InstrumentResolution:
        return self._resolve(query)

    def resolve_universe(self, request: InstrumentUniverseRequest) -> InstrumentUniverse:
        validated = InstrumentUniverseRequest.model_validate(request)
        resolutions: list[InstrumentResolution] = []
        selected: list[Instrument] = []
        warnings: list[str] = []
        seen_resolution_queries: set[str] = set()

        for query in _iter_universe_queries(validated):
            resolution = self._resolve(
                query,
                allowed_asset_classes=validated.allowed_asset_classes,
                provider_names=validated.provider_names,
            )
            if resolution.query in seen_resolution_queries:
                warnings.append(
                    f"Duplicate instrument query {resolution.query!r} was ignored in universe "
                    "resolution."
                )
                continue
            seen_resolution_queries.add(resolution.query)
            resolutions.append(resolution)
            selected_instrument = _selected_instrument(resolution)
            if selected_instrument is not None:
                selected.append(selected_instrument)

        for entry in _iter_watchlist_entries(validated):
            resolution = self._resolve_watchlist_entry(
                entry,
                allowed_asset_classes=validated.allowed_asset_classes,
                provider_names=validated.provider_names,
            )
            if resolution.query in seen_resolution_queries:
                warnings.append(
                    f"Duplicate watchlist query {resolution.query!r} was ignored in universe "
                    "resolution."
                )
                continue
            seen_resolution_queries.add(resolution.query)
            resolutions.append(resolution)
            selected_instrument = _selected_instrument(resolution)
            if selected_instrument is not None:
                selected.append(selected_instrument)

        instruments = _dedupe_instruments(selected)
        if validated.include_related_instruments:
            related, related_warnings = self._related_instruments(
                instruments,
                allowed_asset_classes=validated.allowed_asset_classes,
                provider_names=validated.provider_names,
            )
            instruments = _dedupe_instruments((*instruments, *related))
            warnings.extend(related_warnings)

        for resolution in resolutions:
            warnings.extend(resolution.warnings)

        return InstrumentUniverse(
            request_id=validated.request_id,
            generated_at=datetime.now(UTC),
            resolutions=tuple(resolutions),
            instruments=instruments,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata=validated.metadata,
        )

    def _resolve(
        self,
        query: InstrumentQuery,
        *,
        allowed_asset_classes: tuple[AssetClass, ...] = (),
        provider_names: tuple[str, ...] = (),
    ) -> InstrumentResolution:
        validated = InstrumentQuery.model_validate(query)
        if validated.asset_class == AssetClass.UNKNOWN:
            return _unsupported_resolution(
                validated,
                "Instrument query uses the unknown asset class; add a supported asset_class hint.",
            )

        candidates = self._candidate_instruments(validated)
        hinted = tuple(
            instrument for instrument in candidates if _matches_query_hints(instrument, validated)
        )
        asset_filtered = tuple(
            instrument
            for instrument in hinted
            if _matches_allowed_asset_classes(instrument, allowed_asset_classes)
        )
        provider_filtered = tuple(
            instrument
            for instrument in asset_filtered
            if _matches_provider_names(instrument, provider_names)
        )

        if not provider_filtered:
            if candidates and not hinted:
                return _unavailable_resolution(
                    validated,
                    f"Instrument query {validated.query!r} matched registry instruments, but none "
                    "satisfied the supplied hints.",
                )
            if hinted and not asset_filtered and allowed_asset_classes:
                allowed = ", ".join(asset_class.value for asset_class in allowed_asset_classes)
                return _unsupported_resolution(
                    validated,
                    f"Instrument query {validated.query!r} matched registry instruments, but none "
                    f"were in the allowed asset classes: {allowed}.",
                )
            if asset_filtered and provider_names:
                providers = ", ".join(provider_names)
                return _unavailable_resolution(
                    validated,
                    f"Instrument query {validated.query!r} matched registry instruments, but none "
                    f"matched the requested providers: {providers}.",
                )
            return _unavailable_resolution(
                validated,
                f"Instrument query {validated.query!r} did not match the registry.",
            )

        return _resolution_from_matches(validated, _dedupe_instruments(provider_filtered))

    def _resolve_watchlist_entry(
        self,
        entry: WatchlistEntry,
        *,
        allowed_asset_classes: tuple[AssetClass, ...],
        provider_names: tuple[str, ...],
    ) -> InstrumentResolution:
        if entry.requested_instrument_id is None:
            return self._resolve(
                entry.query,
                allowed_asset_classes=allowed_asset_classes,
                provider_names=provider_names,
            )

        instrument = self.get(entry.requested_instrument_id)
        if instrument is None:
            fallback = self._resolve(
                entry.query,
                allowed_asset_classes=allowed_asset_classes,
                provider_names=provider_names,
            )
            return fallback.model_copy(
                update={
                    "warnings": (
                        f"Watchlist requested instrument "
                        f"{entry.requested_instrument_id!r} was not found.",
                        *fallback.warnings,
                    )
                }
            )

        if not (
            _matches_query_hints(instrument, entry.query)
            and _matches_allowed_asset_classes(instrument, allowed_asset_classes)
            and _matches_provider_names(instrument, provider_names)
        ):
            return _unavailable_resolution(
                entry.query,
                f"Watchlist requested instrument {entry.requested_instrument_id!r} did not "
                "satisfy the supplied query or universe hints.",
            )

        return InstrumentResolution(
            query=entry.query.query,
            status=InstrumentResolutionStatus.RESOLVED,
            matches=(instrument,),
            selected_instrument_id=instrument.instrument_id,
            metadata={"requested_instrument_id": entry.requested_instrument_id},
        )

    def _candidate_instruments(self, query: InstrumentQuery) -> tuple[Instrument, ...]:
        if query.provider_identifier is not None:
            provider_matches = self._find_provider_matches_for_query(query)
            if provider_matches:
                return provider_matches

        symbol_matches = _dedupe_instruments(
            instrument
            for lookup in (query.query, *query.aliases)
            for instrument in self.find_by_symbol_or_alias(lookup)
        )
        if symbol_matches:
            return symbol_matches

        if query.provider is not None:
            return self._find_by_provider_id(
                provider=query.provider,
                namespace=query.provider_namespace,
                identifier=query.query,
                require_namespace=query.provider_namespace is not None,
            )
        return ()

    def _find_provider_matches_for_query(self, query: InstrumentQuery) -> tuple[Instrument, ...]:
        identifier = query.provider_identifier
        if identifier is None:
            return ()
        return self._find_by_provider_id(
            provider=query.provider,
            namespace=query.provider_namespace,
            identifier=identifier,
            require_namespace=query.provider_namespace is not None,
        )

    def _find_by_provider_id(
        self,
        *,
        provider: str | None,
        namespace: str | None,
        identifier: str,
        require_namespace: bool,
    ) -> tuple[Instrument, ...]:
        return _dedupe_instruments(
            instrument
            for instrument in self._list_instruments()
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

    def _list_instruments(self) -> tuple[Instrument, ...]:
        return tuple(instrument_from_record(record) for record in self._store.list_instruments())

    def _related_instruments(
        self,
        instruments: tuple[Instrument, ...],
        *,
        allowed_asset_classes: tuple[AssetClass, ...],
        provider_names: tuple[str, ...],
    ) -> tuple[tuple[Instrument, ...], tuple[str, ...]]:
        related: list[Instrument] = []
        warnings: list[str] = []
        for instrument in instruments:
            for related_instrument in instrument.related_instruments:
                found = self.get(related_instrument.instrument_id)
                if found is None:
                    warnings.append(
                        f"Related instrument {related_instrument.instrument_id!r} referenced by "
                        f"{instrument.instrument_id!r} was not found."
                    )
                    continue
                if not (
                    _matches_allowed_asset_classes(found, allowed_asset_classes)
                    and _matches_provider_names(found, provider_names)
                ):
                    warnings.append(
                        f"Related instrument {found.instrument_id!r} was excluded by universe "
                        "filters."
                    )
                    continue
                related.append(found)
        return _dedupe_instruments(related), tuple(warnings)


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


def _dedupe_instruments(instruments: Iterable[Instrument]) -> tuple[Instrument, ...]:
    deduped: list[Instrument] = []
    seen: set[str] = set()
    for instrument in instruments:
        if instrument.instrument_id in seen:
            continue
        seen.add(instrument.instrument_id)
        deduped.append(instrument)
    return tuple(deduped)


def _resolution_from_matches(
    query: InstrumentQuery,
    matches: tuple[Instrument, ...],
) -> InstrumentResolution:
    if len(matches) == 1:
        return InstrumentResolution(
            query=query.query,
            status=InstrumentResolutionStatus.RESOLVED,
            matches=matches,
            selected_instrument_id=matches[0].instrument_id,
        )
    return InstrumentResolution(
        query=query.query,
        status=InstrumentResolutionStatus.AMBIGUOUS,
        matches=matches,
        warnings=(
            f"Instrument query {query.query!r} matched {len(matches)} instruments; add "
            "asset_class, venue, provider, or provider_identifier hints to disambiguate.",
        ),
    )


def _unsupported_resolution(query: InstrumentQuery, warning: str) -> InstrumentResolution:
    return InstrumentResolution(
        query=query.query,
        status=InstrumentResolutionStatus.UNSUPPORTED,
        warnings=(warning,),
    )


def _unavailable_resolution(query: InstrumentQuery, warning: str) -> InstrumentResolution:
    return InstrumentResolution(
        query=query.query,
        status=InstrumentResolutionStatus.UNAVAILABLE,
        warnings=(warning,),
    )


def _selected_instrument(resolution: InstrumentResolution) -> Instrument | None:
    if resolution.selected_instrument_id is None:
        return None
    return next(
        instrument
        for instrument in resolution.matches
        if instrument.instrument_id == resolution.selected_instrument_id
    )


def _iter_universe_queries(request: InstrumentUniverseRequest) -> tuple[InstrumentQuery, ...]:
    return request.queries


def _iter_watchlist_entries(request: InstrumentUniverseRequest) -> tuple[WatchlistEntry, ...]:
    return tuple(entry for watchlist in request.watchlists for entry in watchlist.entries)


def _matches_query_hints(instrument: Instrument, query: InstrumentQuery) -> bool:
    if query.asset_class is not None and instrument.asset_class != query.asset_class:
        return False
    if query.venue is not None and _normalize_text(instrument.venue) != _normalize_text(
        query.venue
    ):
        return False
    if query.provider is not None and not any(
        _provider_id_matches(
            provider_id,
            provider=query.provider,
            namespace=query.provider_namespace,
            identifier=query.provider_identifier,
            require_namespace=query.provider_namespace is not None,
        )
        for provider_id in instrument.provider_ids
    ):
        return False
    return not (
        query.provider is None
        and query.provider_identifier is not None
        and not any(
            _provider_id_matches(
                provider_id,
                provider=None,
                namespace=query.provider_namespace,
                identifier=query.provider_identifier,
                require_namespace=query.provider_namespace is not None,
            )
            for provider_id in instrument.provider_ids
        )
    )


def _matches_allowed_asset_classes(
    instrument: Instrument,
    allowed_asset_classes: tuple[AssetClass, ...],
) -> bool:
    return not allowed_asset_classes or instrument.asset_class in set(allowed_asset_classes)


def _matches_provider_names(
    instrument: Instrument,
    provider_names: tuple[str, ...],
) -> bool:
    if not provider_names:
        return True
    normalized = {_normalize_text(provider_name) for provider_name in provider_names}
    return any(
        _normalize_text(provider_id.provider) in normalized
        for provider_id in instrument.provider_ids
    )


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
    "InstrumentRegistry",
    "instrument_from_record",
    "instrument_to_record",
]
