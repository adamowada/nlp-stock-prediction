"""Provenance and provider health contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
)
from nlp_stock_prediction.contracts.enums import (
    CredentialState,
    FreshnessStatus,
    ProviderStatus,
    RetrievalMethod,
    SourceKind,
    WarningCode,
    WarningSeverity,
)


class ProviderWarning(ContractModel):
    """A recoverable provider or data-quality problem surfaced to reports."""

    code: WarningCode
    severity: WarningSeverity
    message: NonEmptyStr
    provider_name: NonEmptyStr | None = None
    retryable: bool = False
    provider_status_code: int | None = None
    provider_error_type: str | None = None
    occurred_at: AwareDatetime
    stale_after: AwareDatetime | None = None
    raw_snapshot_id: str | None = None
    source_url: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class ProviderHealth(ContractModel):
    """Health snapshot for one provider interaction or provider check."""

    provider_name: NonEmptyStr
    status: ProviderStatus
    checked_at: AwareDatetime
    credential_state: CredentialState = CredentialState.NOT_REQUIRED
    latency_ms: int | None = Field(default=None, ge=0)
    rate_limit_remaining: int | None = Field(default=None, ge=0)
    rate_limit_reset_at: AwareDatetime | None = None
    last_success_at: AwareDatetime | None = None
    warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status == ProviderStatus.OK


class SourceProvenance(ContractModel):
    """Traceability block carried by every normalized external datum."""

    provider_name: NonEmptyStr
    source_kind: SourceKind
    retrieval_method: RetrievalMethod
    fetched_at: AwareDatetime
    observed_at: AwareDatetime | None = None
    source_url: str | None = None
    permalink: str | None = None
    raw_identifier: str | None = None
    raw_snapshot_id: str | None = None
    query: str | None = None
    cache_key: str | None = None
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    freshness_seconds: int | None = Field(default=None, ge=0)
    provider_metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_external_traceability(self) -> SourceProvenance:
        if self.source_kind == SourceKind.INTERNAL_ANALYSIS or (
            self.retrieval_method == RetrievalMethod.DERIVED
        ):
            return self
        if not (self.source_url or self.permalink):
            raise ValueError("external provenance requires source_url or permalink")
        if not self.raw_identifier:
            raise ValueError("external provenance requires raw_identifier")
        if not self.raw_snapshot_id:
            raise ValueError("external provenance requires raw_snapshot_id")
        if self.freshness_status == FreshnessStatus.UNKNOWN:
            raise ValueError("external provenance requires explicit freshness_status")
        return self


class EvidenceReference(ContractModel):
    """Stable reference to evidence used by derived claims."""

    evidence_id: NonEmptyStr
    quote: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_span_order(self) -> EvidenceReference:
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char must be greater than or equal to start_char")
        return self


class DataReference(ContractModel):
    """Reference to a derived data artifact or provider output."""

    reference_id: NonEmptyStr
    reference_type: Literal[
        "raw_snapshot",
        "normalized_evidence",
        "extraction",
        "analysis",
        "scoring_input",
        "report",
        "audit_artifact",
    ]
    path: str | None = None
    sha256: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


__all__ = [
    "DataReference",
    "EvidenceReference",
    "ProviderHealth",
    "ProviderWarning",
    "SourceProvenance",
]
