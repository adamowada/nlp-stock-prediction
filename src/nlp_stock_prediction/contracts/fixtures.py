"""Fixture manifest contracts for deterministic tests and audit regeneration."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
)
from nlp_stock_prediction.contracts.providers import ProviderRequest


class RawProviderFixture(ContractModel):
    """Raw provider response fixture stored before normalization."""

    fixture_version: int = Field(default=1, ge=1)
    provider_name: NonEmptyStr
    scenario: NonEmptyStr
    recorded_at: AwareDatetime
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

    @model_validator(mode="after")
    def validate_nested_fixture_context(self) -> FixtureManifest:
        for raw_fixture in self.raw_fixtures:
            if raw_fixture.scenario != self.scenario:
                raise ValueError("raw fixture scenario must match manifest scenario")
            if raw_fixture.request.run_date != self.run_date:
                raise ValueError("raw fixture request run_date must match manifest run_date")
        for normalized_fixture in self.normalized_fixtures:
            if normalized_fixture.scenario != self.scenario:
                raise ValueError("normalized fixture scenario must match manifest scenario")
        return self


__all__ = ["FixtureManifest", "NormalizedFixture", "RawProviderFixture"]
