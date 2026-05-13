"""Instrument-universe contracts for the Phase 2 rebuild.

These models are intentionally broader than ticker-only contracts. They describe what the
assistant is allowed to research, which provider identifiers point at it, and what evidence supports
availability or tradability. They do not imply that any instrument should be bought, sold, or sized.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator, Field, StringConstraints, field_validator, model_validator

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


type InstrumentSymbol = Annotated[
    str,
    BeforeValidator(_normalize_instrument_symbol),
    StringConstraints(pattern=r"^[A-Z0-9][A-Z0-9./:_-]{0,31}$", min_length=1, max_length=32),
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


__all__ = [
    "Instrument",
    "InstrumentDataAvailability",
    "InstrumentResolution",
    "InstrumentSymbol",
    "ProviderInstrumentId",
    "RelatedInstrument",
    "TradabilityEvidence",
]
