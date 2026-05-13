"""Instrument-universe contracts for the Phase 3 rebuild.

These models are intentionally broader than ticker-only contracts. They describe what the
assistant is allowed to research, which provider identifiers point at it, and what evidence supports
availability or tradability. They do not imply that any instrument should be bought, sold, or sized.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BeforeValidator,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
)
from nlp_stock_prediction.contracts.enums import (
    AssetClass,
    InstrumentResolutionStatus,
    TradabilityStatus,
)


def _normalize_instrument_symbol(value: object) -> str:
    if value is None:
        raise ValueError("instrument symbol cannot be null")
    if not isinstance(value, str):
        raise ValueError("instrument symbol must be a string")
    return value.strip().upper()


def _normalize_instrument_query(value: object) -> str:
    if value is None:
        raise ValueError("instrument query cannot be null")
    if not isinstance(value, str):
        raise ValueError("instrument query must be a string")
    return value.strip().upper()


type InstrumentSymbol = Annotated[
    str,
    BeforeValidator(_normalize_instrument_symbol),
    StringConstraints(pattern=r"^[A-Z0-9][A-Z0-9./:_-]{0,31}$", min_length=1, max_length=32),
]
type InstrumentQueryText = Annotated[
    str,
    BeforeValidator(_normalize_instrument_query),
    StringConstraints(min_length=1, max_length=128),
]


class ProviderInstrumentId(ContractModel):
    """One provider-specific identifier for an instrument."""

    provider: NonEmptyStr
    identifier: NonEmptyStr
    namespace: str | None = None
    url: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class TradabilityEvidence(ContractModel):
    """Source-backed availability or tradability observation."""

    provider: NonEmptyStr
    status: TradabilityStatus
    retrieved_at: AwareDatetime
    source_url: str | None = None
    permalink: str | None = None
    raw_identifier: str | None = None
    notes: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_traceable_source(self) -> TradabilityEvidence:
        if not (self.source_url or self.permalink or self.raw_identifier):
            raise ValueError(
                "tradability evidence requires source_url, permalink, or raw_identifier"
            )
        return self


class InstrumentDataAvailability(ContractModel):
    """Provider-level statement about data availability for an instrument."""

    provider: NonEmptyStr
    data_type: NonEmptyStr
    status: TradabilityStatus
    checked_at: AwareDatetime
    provider_identifier: str | None = None
    notes: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class RelatedInstrument(ContractModel):
    """A related instrument or proxy with evidence-backed rationale."""

    instrument_id: NonEmptyStr
    relationship: NonEmptyStr
    rationale: NonEmptyStr
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)


class InstrumentQuery(ContractModel):
    """User or provider query to resolve into a researchable instrument."""

    query: InstrumentQueryText
    asset_class: AssetClass | None = None
    venue: str | None = None
    provider: str | None = None
    provider_namespace: str | None = None
    provider_identifier: str | None = None
    aliases: tuple[InstrumentQueryText, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("aliases")
    @classmethod
    def remove_duplicate_aliases(cls, aliases: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(aliases))


class WatchlistEntry(ContractModel):
    """One instrument query and local context from a named watchlist."""

    query: InstrumentQuery
    requested_instrument_id: str | None = None
    notes: str | None = None
    tags: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def remove_duplicate_tags(cls, tags: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(tags))


class Watchlist(ContractModel):
    """Named collection of instrument queries supplied to a universe request."""

    watchlist_id: NonEmptyStr
    name: NonEmptyStr
    entries: tuple[WatchlistEntry, ...]
    description: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_entries(self) -> Watchlist:
        if not self.entries:
            raise ValueError("watchlists require at least one entry")
        return self


class Instrument(ContractModel):
    """Researchable instrument identity with provider and availability context."""

    instrument_id: NonEmptyStr
    symbol: InstrumentSymbol
    display_name: NonEmptyStr
    asset_class: AssetClass
    venue: str | None = None
    aliases: tuple[InstrumentSymbol, ...] = Field(default_factory=tuple)
    provider_ids: tuple[ProviderInstrumentId, ...] = Field(default_factory=tuple)
    related_instruments: tuple[RelatedInstrument, ...] = Field(default_factory=tuple)
    tradability_evidence: tuple[TradabilityEvidence, ...] = Field(default_factory=tuple)
    data_availability: tuple[InstrumentDataAvailability, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("aliases")
    @classmethod
    def remove_duplicate_aliases(cls, aliases: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for alias in aliases:
            if alias not in normalized:
                normalized.append(alias)
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_provider_ids(self) -> Instrument:
        seen: set[tuple[str, str | None]] = set()
        for provider_id in self.provider_ids:
            key = (provider_id.provider.lower(), provider_id.namespace)
            if key in seen:
                raise ValueError("instrument provider ids must be unique per provider namespace")
            seen.add(key)
        return self


class InstrumentResolution(ContractModel):
    """Result of resolving a user/provider query into zero, one, or many instruments."""

    query: NonEmptyStr
    status: InstrumentResolutionStatus
    matches: tuple[Instrument, ...] = Field(default_factory=tuple)
    selected_instrument_id: str | None = None
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_resolution_shape(self) -> InstrumentResolution:
        match_ids = {instrument.instrument_id for instrument in self.matches}
        if self.status == InstrumentResolutionStatus.RESOLVED:
            if self.selected_instrument_id is None:
                raise ValueError("resolved instruments require selected_instrument_id")
            if self.selected_instrument_id not in match_ids:
                raise ValueError("selected_instrument_id must reference one of the matches")
        if self.status == InstrumentResolutionStatus.AMBIGUOUS and len(self.matches) < 2:
            raise ValueError("ambiguous instrument resolutions require at least two matches")
        if (
            self.status
            in {
                InstrumentResolutionStatus.UNSUPPORTED,
                InstrumentResolutionStatus.UNAVAILABLE,
            }
            and self.selected_instrument_id is not None
        ):
            raise ValueError("unsupported or unavailable resolutions cannot select an instrument")
        return self


class InstrumentUniverseRequest(ContractModel):
    """Request for resolving a mixed-asset instrument universe."""

    request_id: NonEmptyStr
    as_of: AwareDatetime
    queries: tuple[InstrumentQuery, ...] = Field(default_factory=tuple)
    watchlists: tuple[Watchlist, ...] = Field(default_factory=tuple)
    allowed_asset_classes: tuple[AssetClass, ...] = Field(default_factory=tuple)
    provider_names: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    include_related_instruments: bool = True
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request_shape(self) -> InstrumentUniverseRequest:
        if not self.queries and not self.watchlists:
            raise ValueError("instrument universe requests require at least one query or watchlist")
        watchlist_ids = tuple(watchlist.watchlist_id for watchlist in self.watchlists)
        if len(set(watchlist_ids)) != len(watchlist_ids):
            raise ValueError("instrument universe request watchlist ids must be unique")
        return self


class InstrumentUniverse(ContractModel):
    """Resolved instrument universe with request-to-instrument traceability."""

    request_id: NonEmptyStr
    generated_at: AwareDatetime
    resolutions: tuple[InstrumentResolution, ...] = Field(default_factory=tuple)
    instruments: tuple[Instrument, ...]
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def instrument_ids(self) -> tuple[str, ...]:
        return tuple(instrument.instrument_id for instrument in self.instruments)

    @model_validator(mode="after")
    def validate_universe_shape(self) -> InstrumentUniverse:
        if not self.instruments and not self.resolutions:
            raise ValueError("instrument universes require at least one instrument or resolution")
        instrument_ids = self.instrument_ids
        if len(set(instrument_ids)) != len(instrument_ids):
            raise ValueError("instrument universe instrument ids must be unique")

        resolution_queries = tuple(resolution.query for resolution in self.resolutions)
        if len(set(resolution_queries)) != len(resolution_queries):
            raise ValueError("instrument universe resolution queries must be unique")

        selected_ids = {
            resolution.selected_instrument_id
            for resolution in self.resolutions
            if resolution.selected_instrument_id is not None
        }
        missing_selected_ids = selected_ids.difference(instrument_ids)
        if missing_selected_ids:
            raise ValueError(
                "instrument universe selected resolution ids must be materialized as instruments"
            )
        return self


__all__ = [
    "Instrument",
    "InstrumentDataAvailability",
    "InstrumentQuery",
    "InstrumentResolution",
    "InstrumentSymbol",
    "InstrumentUniverse",
    "InstrumentUniverseRequest",
    "ProviderInstrumentId",
    "RelatedInstrument",
    "TradabilityEvidence",
    "Watchlist",
    "WatchlistEntry",
]
