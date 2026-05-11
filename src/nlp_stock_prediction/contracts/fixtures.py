"""Fixture manifest contracts for deterministic tests and audit regeneration."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field

from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, NonEmptyStr
from nlp_stock_prediction.contracts.providers import ProviderRequest


class RawProviderFixture(ContractModel):
    """Raw provider response fixture stored before normalization."""

    fixture_version: int = Field(default=1, ge=1)
    provider_name: NonEmptyStr
    scenario: NonEmptyStr
    recorded_at: datetime
    request: ProviderRequest
    response_path: NonEmptyStr
    content_type: NonEmptyStr = "application/json"
    provider_metadata: JsonObject = Field(default_factory=dict)
    redactions: tuple[str, ...] = Field(default_factory=tuple)


class NormalizedFixture(ContractModel):
    """Normalized contract fixture consumed by contract and e2e tests."""

    fixture_version: int = Field(default=1, ge=1)
    scenario: NonEmptyStr
    layer: Literal[
        "provider_result",
        "evidence",
        "extraction",
        "analysis",
        "scoring",
        "report",
    ]
    path: NonEmptyStr
    source_fixture_ids: tuple[str, ...] = Field(default_factory=tuple)
    expected_record_count: int | None = Field(default=None, ge=0)
    metadata: JsonObject = Field(default_factory=dict)


class FixtureManifest(ContractModel):
    """Scenario manifest that composes raw and normalized fixtures."""

    fixture_version: int = Field(default=1, ge=1)
    scenario: NonEmptyStr
    run_date: date
    raw_fixtures: tuple[RawProviderFixture, ...] = Field(default_factory=tuple)
    normalized_fixtures: tuple[NormalizedFixture, ...] = Field(default_factory=tuple)
    expected_report_json_path: str | None = None
    expected_report_markdown_path: str | None = None
    notes: tuple[str, ...] = Field(default_factory=tuple)


__all__ = ["FixtureManifest", "NormalizedFixture", "RawProviderFixture"]
