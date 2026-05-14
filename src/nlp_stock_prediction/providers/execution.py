"""Shared provider execution policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from nlp_stock_prediction.contracts.enums import (
    CredentialState,
    FreshnessStatus,
    ProviderStatus,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.provenance import (
    ProviderWarning,
)
from nlp_stock_prediction.contracts.providers import (
    ProviderRequest,
    ProviderResult,
)
from nlp_stock_prediction.providers._base import (
    malformed_result,
    no_data_result,
    provider_result,
    provider_warning,
)

T = TypeVar("T")


@dataclass(frozen=True)
class ProviderExecutionContext:
    """Result-building context for one provider fetch attempt."""

    provider_name: str
    request: ProviderRequest
    fetched_at: datetime
    credential_state: CredentialState = CredentialState.NOT_REQUIRED
    raw_snapshot_id: str | None = None
    cache_key: str | None = None

    def with_fetch(
        self,
        *,
        raw_snapshot_id: str,
        cache_key: str,
    ) -> ProviderExecutionContext:
        return ProviderExecutionContext(
            provider_name=self.provider_name,
            request=self.request,
            fetched_at=self.fetched_at,
            credential_state=self.credential_state,
            raw_snapshot_id=raw_snapshot_id,
            cache_key=cache_key,
        )

    def no_data(self, message: str) -> ProviderResult[T]:
        return no_data_result(
            provider_name=self.provider_name,
            request=self.request,
            fetched_at=self.fetched_at,
            message=message,
            credential_state=self.credential_state,
            raw_snapshot_id=self.raw_snapshot_id,
            cache_key=self.cache_key,
        )

    def malformed(self, message: str) -> ProviderResult[T]:
        return malformed_result(
            provider_name=self.provider_name,
            request=self.request,
            fetched_at=self.fetched_at,
            message=message,
            credential_state=self.credential_state,
            raw_snapshot_id=self.raw_snapshot_id,
            cache_key=self.cache_key,
        )

    def rate_limited(self, message: str) -> ProviderResult[T]:
        warning = provider_warning(
            provider_name=self.provider_name,
            code=WarningCode.RATE_LIMITED,
            severity=WarningSeverity.ERROR,
            message=message,
            occurred_at=self.fetched_at,
            raw_snapshot_id=self.raw_snapshot_id,
            provider_error_type="rate_limit",
        )
        return provider_result(
            provider_name=self.provider_name,
            status=ProviderStatus.RATE_LIMITED,
            request=self.request,
            fetched_at=self.fetched_at,
            credential_state=self.credential_state,
            warnings=(warning,),
            raw_snapshot_id=self.raw_snapshot_id,
            cache_key=self.cache_key,
        )

    def evidence_result(
        self,
        *,
        evidence: tuple[SourceEvidence, ...],
        warnings: tuple[ProviderWarning, ...],
        no_data_message: str,
        stale_message: str,
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        raw_snapshot_id = self.raw_snapshot_id
        cache_key = self.cache_key
        if not evidence:
            if warnings:
                return provider_result(
                    provider_name=self.provider_name,
                    status=ProviderStatus.MALFORMED,
                    request=self.request,
                    fetched_at=self.fetched_at,
                    credential_state=self.credential_state,
                    warnings=warnings,
                    raw_snapshot_id=self.raw_snapshot_id,
                    cache_key=self.cache_key,
                )
            return self.no_data(no_data_message)
        status = ProviderStatus.PARTIAL if warnings else ProviderStatus.OK
        if any(item.provenance.freshness_status == FreshnessStatus.STALE for item in evidence):
            status = ProviderStatus.STALE
            warnings += (
                provider_warning(
                    provider_name=self.provider_name,
                    code=WarningCode.STALE_DATA,
                    severity=WarningSeverity.WARNING,
                    message=stale_message,
                    occurred_at=self.fetched_at,
                    raw_snapshot_id=raw_snapshot_id,
                ),
            )
        return provider_result(
            provider_name=self.provider_name,
            status=status,
            request=self.request,
            fetched_at=self.fetched_at,
            credential_state=self.credential_state,
            data=evidence,
            warnings=warnings,
            raw_snapshot_id=raw_snapshot_id,
            cache_key=cache_key,
        )

    def partial_item_warning(self, *, index: int, message: str) -> ProviderWarning:
        return provider_warning(
            provider_name=self.provider_name,
            code=WarningCode.PARTIAL_DATA,
            severity=WarningSeverity.WARNING,
            message=message,
            occurred_at=self.fetched_at,
            raw_snapshot_id=self.raw_snapshot_id,
            metadata={"item_index": index},
        )


def evidence_result_from_records(
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    evidence: tuple[SourceEvidence, ...],
    warnings: tuple[ProviderWarning, ...],
    no_data_message: str,
    stale_message: str,
    credential_state: CredentialState,
    raw_snapshot_id: str,
    cache_key: str,
) -> ProviderResult[tuple[SourceEvidence, ...]]:
    """Build a provider result for source evidence with consistent degradation policy."""

    return ProviderExecutionContext(
        provider_name=provider_name,
        request=request,
        fetched_at=fetched_at,
        credential_state=credential_state,
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    ).evidence_result(
        evidence=evidence,
        warnings=warnings,
        no_data_message=no_data_message,
        stale_message=stale_message,
    )


def partial_item_warning(
    *,
    provider_name: str,
    fetched_at: datetime,
    raw_snapshot_id: str,
    index: int,
    message: str,
) -> ProviderWarning:
    return provider_warning(
        provider_name=provider_name,
        code=WarningCode.PARTIAL_DATA,
        severity=WarningSeverity.WARNING,
        message=message,
        occurred_at=fetched_at,
        raw_snapshot_id=raw_snapshot_id,
        metadata={"item_index": index},
    )


def rate_limited_result(
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    message: str,
    raw_snapshot_id: str,
    cache_key: str,
    credential_state: CredentialState = CredentialState.CONFIGURED,
) -> ProviderResult[T]:
    return ProviderExecutionContext(
        provider_name=provider_name,
        request=request,
        fetched_at=fetched_at,
        credential_state=credential_state,
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    ).rate_limited(message)


__all__ = [
    "ProviderExecutionContext",
    "evidence_result_from_records",
    "partial_item_warning",
    "rate_limited_result",
]
