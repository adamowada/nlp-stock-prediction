"""Shared provider execution policies."""

from __future__ import annotations

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
    no_data_result,
    provider_result,
    provider_warning,
)

T = TypeVar("T")


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

    if not evidence:
        return no_data_result(
            provider_name=provider_name,
            request=request,
            fetched_at=fetched_at,
            message=no_data_message,
            credential_state=credential_state,
            raw_snapshot_id=raw_snapshot_id,
            cache_key=cache_key,
        )
    status = ProviderStatus.PARTIAL if warnings else ProviderStatus.OK
    if any(item.provenance.freshness_status == FreshnessStatus.STALE for item in evidence):
        status = ProviderStatus.STALE
        warnings += (
            provider_warning(
                provider_name=provider_name,
                code=WarningCode.STALE_DATA,
                severity=WarningSeverity.WARNING,
                message=stale_message,
                occurred_at=fetched_at,
                raw_snapshot_id=raw_snapshot_id,
            ),
        )
    return provider_result(
        provider_name=provider_name,
        status=status,
        request=request,
        fetched_at=fetched_at,
        credential_state=credential_state,
        data=evidence,
        warnings=warnings,
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
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
    warning = provider_warning(
        provider_name=provider_name,
        code=WarningCode.RATE_LIMITED,
        severity=WarningSeverity.ERROR,
        message=message,
        occurred_at=fetched_at,
        raw_snapshot_id=raw_snapshot_id,
        provider_error_type="rate_limit",
    )
    return provider_result(
        provider_name=provider_name,
        status=ProviderStatus.RATE_LIMITED,
        request=request,
        fetched_at=fetched_at,
        credential_state=credential_state,
        warnings=(warning,),
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


__all__ = ["evidence_result_from_records", "partial_item_warning", "rate_limited_result"]
