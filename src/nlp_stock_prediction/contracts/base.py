"""Shared contract primitives."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints


def _normalize_ticker(value: object) -> str:
    if value is None:
        raise ValueError("ticker symbol cannot be null")
    return str(value).strip().upper()


def _normalize_non_empty(value: object) -> str:
    if value is None:
        raise ValueError("value cannot be null")
    return str(value).strip()


type TickerSymbol = Annotated[
    str,
    BeforeValidator(_normalize_ticker),
    StringConstraints(pattern=r"^[A-Z][A-Z0-9.\-]{0,9}$", min_length=1, max_length=10),
]
type NonEmptyStr = Annotated[
    str,
    BeforeValidator(_normalize_non_empty),
    StringConstraints(min_length=1),
]
type Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
type Score = Annotated[float, Field(ge=0.0, le=1.0)]
type PositiveInt = Annotated[int, Field(ge=0)]
type PositiveDecimal = Annotated[Decimal, Field(ge=Decimal("0"))]
type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class ContractModel(BaseModel):
    """Base class for immutable public contract models."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def ensure_utc_timestamp(value: datetime) -> datetime:
    """Return a timestamp contract value.

    The contract does not mutate timezone information yet; live providers must supply timezone-aware
    timestamps where the source exposes them.
    """

    return value


__all__ = [
    "Confidence",
    "ContractModel",
    "JsonObject",
    "JsonValue",
    "NonEmptyStr",
    "PositiveDecimal",
    "PositiveInt",
    "Score",
    "TickerSymbol",
    "ensure_utc_timestamp",
]
