"""Shared live-input validation helpers for evaluation and reporting paths."""

from __future__ import annotations

import re

from nlp_stock_prediction.contracts.base import JsonObject, JsonValue
from nlp_stock_prediction.contracts.enums import RetrievalMethod

LIVE_RETRIEVAL_METHODS = frozenset({RetrievalMethod.OFFICIAL_API, RetrievalMethod.PUBLIC_SCRAPE})
LIVE_MODE_VALUE = "live"
LIVE_MODE_KEYS = ("report_data_mode", "provider_mode", "input_data_mode")
LIVE_METADATA_MODE_KEYS = frozenset(
    {
        *LIVE_MODE_KEYS,
        "source_mode",
        "retrieval_method",
        "report_input_mode",
        "artifact_data_mode",
        "tool_data_mode",
    }
)
NON_LIVE_MODE_MARKERS = frozenset(
    {
        "offline",
        "offline_fixture",
        "fixture",
        "fixtures",
        "fixture_backed",
        "dummy",
        "dummy_smoke",
        "codex_smoke",
        "smoke",
        "test",
    }
)
NON_LIVE_TEXT_MARKERS = ("fixture", "dummy", "smoke")


def require_live_metadata(record_type: str, record_id: str, metadata: JsonObject) -> None:
    """Reject metadata that marks a record as fixture, dummy, smoke, or otherwise non-live."""

    for key in LIVE_MODE_KEYS:
        value = metadata.get(key)
        if value is not None and value != LIVE_MODE_VALUE:
            raise ValueError(
                f"{record_type} {record_id} has {key}={value!r} contains non-live {key} "
                "and is not allowed in a live evaluation path"
            )
    for field, value in _non_live_text_fields(metadata):
        raise ValueError(
            f"{record_type} {record_id} has {field}={value!r} contains non-live {field} "
            "and is not allowed in a live evaluation path"
        )


def require_live_retrieval_method(
    *,
    record_type: str,
    record_id: str,
    retrieval_method: RetrievalMethod,
) -> None:
    if retrieval_method not in LIVE_RETRIEVAL_METHODS:
        raise ValueError(
            f"{record_type} {record_id} must come from an official API or public scraping "
            f"provider: {retrieval_method.value}"
        )


def text_is_non_live(value: str) -> bool:
    return _text_contains_non_live_token(value)


def _non_live_text_fields(metadata: JsonObject) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    for key, value in metadata.items():
        violations.extend(_non_live_text_value_fields(key, value))
    return violations


def _non_live_text_value_fields(field: str, value: JsonValue) -> list[tuple[str, str]]:
    if isinstance(value, str):
        field_key = _field_key(field)
        if field_key in LIVE_METADATA_MODE_KEYS and _mode_value_is_non_live(value):
            return [(field, value)]
        if _text_contains_non_live_token(value):
            return [(field, value)]
    if isinstance(value, dict):
        return [
            nested
            for key, nested_value in value.items()
            for nested in _non_live_text_value_fields(f"{field}.{key}", nested_value)
        ]
    if isinstance(value, list | tuple):
        return [
            nested
            for index, nested_value in enumerate(value)
            for nested in _non_live_text_value_fields(f"{field}[{index}]", nested_value)
        ]
    return []


def _field_key(field: str) -> str:
    return field.rsplit(".", 1)[-1].split("[", 1)[0]


def _mode_value_is_non_live(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_")
    return normalized in NON_LIVE_MODE_MARKERS


def _text_contains_non_live_token(value: str) -> bool:
    tokens = {token for token in re.split(r"[^a-z0-9_]+", value.strip().lower()) if token}
    return any(marker in tokens for marker in NON_LIVE_TEXT_MARKERS)


__all__ = [
    "LIVE_METADATA_MODE_KEYS",
    "LIVE_MODE_KEYS",
    "LIVE_MODE_VALUE",
    "LIVE_RETRIEVAL_METHODS",
    "NON_LIVE_MODE_MARKERS",
    "NON_LIVE_TEXT_MARKERS",
    "require_live_metadata",
    "require_live_retrieval_method",
    "text_is_non_live",
]
