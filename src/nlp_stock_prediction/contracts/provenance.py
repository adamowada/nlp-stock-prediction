"""Provenance and provider health contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, NonEmptyStr
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
    occurred_at: datetime
    stale_after: datetime | None = None
    raw_snapshot_id: str | None = None
    source_url: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class ProviderHealth(ContractModel):
    """Health snapshot for one provider interaction or provider check."""

    provider_name: NonEmptyStr
    status: ProviderStatus
    checked_at: datetime
    credential_state: CredentialState = CredentialState.NOT_REQUIRED
    latency_ms: int | None = Field(default=None, ge=0)
    rate_limit_remaining: int | None = Field(default=None, ge=0)
    rate_limit_reset_at: datetime | None = None
    last_success_at: datetime | None = None
    warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status == ProviderStatus.OK


class SourceProvenance(ContractModel):
    """Traceability block carried by every normalized external datum."""

    provider_name: NonEmptyStr
    source_kind: SourceKind
    retrieval_method: RetrievalMethod
    fetched_at: datetime
    observed_at: datetime | None = None
    source_url: str | None = None
    permalink: str | None = None
    raw_identifier: str | None = None
    raw_snapshot_id: str | None = None
    query: str | None = None
    cache_key: str | None = None
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    freshness_seconds: int | None = Field(default=None, ge=0)
    provider_metadata: JsonObject = Field(default_factory=dict)


class EvidenceReference(ContractModel):
    """Stable reference to evidence used by derived claims."""

    evidence_id: NonEmptyStr
    quote: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)


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
