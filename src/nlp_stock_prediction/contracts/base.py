"""Shared contract primitives."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from math import isfinite
from typing import Annotated, Any, NoReturn, Self, SupportsIndex, overload

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
)


def _normalize_ticker(value: object) -> str:
    if value is None:
        raise ValueError("ticker symbol cannot be null")
    if not isinstance(value, str):
        raise ValueError("ticker symbol must be a string")
    return str(value).strip().upper()


def _normalize_non_empty(value: object) -> str:
    if value is None:
        raise ValueError("value cannot be null")
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    return str(value).strip()


type TickerSymbol = Annotated[
    str,
    BeforeValidator(_normalize_ticker),
    StringConstraints(pattern=r"^[A-Z0-9][A-Z0-9./:_-]{0,31}$", min_length=1, max_length=32),
]
type NonEmptyStr = Annotated[
    str,
    BeforeValidator(_normalize_non_empty),
    StringConstraints(min_length=1),
]
type Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
type Score = Annotated[float, Field(ge=0.0, le=1.0)]
type PositiveInt = Annotated[int, Field(gt=0)]
type PositiveDecimal = Annotated[Decimal, Field(gt=Decimal("0"))]
type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class FrozenJsonDict(dict[str, JsonValue]):
    """A JSON object that rejects mutation after contract validation."""

    def _immutable(self) -> NoReturn:
        raise TypeError("JSON metadata is immutable")

    def __setitem__(self, key: str, value: JsonValue) -> None:
        self._immutable()

    def __delitem__(self, key: str) -> None:
        self._immutable()

    def clear(self) -> None:
        self._immutable()

    def pop(self, key: str, default: Any = None) -> Any:
        self._immutable()

    def popitem(self) -> tuple[str, JsonValue]:
        self._immutable()

    @overload
    def setdefault(self, key: str, default: None = None) -> None: ...

    @overload
    def setdefault(self, key: str, default: JsonValue) -> JsonValue: ...

    def setdefault(self, key: str, default: JsonValue | None = None) -> JsonValue | None:
        self._immutable()

    def update(self, *args: Any, **kwargs: JsonValue) -> None:
        self._immutable()

    def __ior__(self, other: object, /) -> Self:  # type: ignore[override,misc]
        del other
        self._immutable()


class FrozenJsonList(list[JsonValue]):
    """A JSON array that rejects mutation after contract validation."""

    def _immutable(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError("JSON metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __iadd__(self, values: Iterable[Any], /) -> Self:  # type: ignore[override,misc]
        del values
        self._immutable()

    def __imul__(self, value: SupportsIndex) -> Self:
        del value
        self._immutable()


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("JSON values must not contain non-finite floats")
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError("decimal values must be finite")
    if isinstance(value, dict):
        return FrozenJsonDict(
            {key: _freeze_json_value(nested_value) for key, nested_value in value.items()}
        )
    if isinstance(value, list):
        return FrozenJsonList(_freeze_json_value(nested_value) for nested_value in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(nested_value) for nested_value in value)
    return value


class ContractModel(BaseModel):
    """Base class for immutable public contract models."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        ser_json_inf_nan="constants",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    def model_post_init(self, _context: object) -> None:
        for field_name, value in self.__dict__.items():
            object.__setattr__(self, field_name, _freeze_json_value(value))


def ensure_utc_timestamp(value: datetime) -> datetime:
    """Normalize a contract timestamp to UTC and reject naive datetimes."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


type AwareDatetime = Annotated[datetime, AfterValidator(ensure_utc_timestamp)]


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
