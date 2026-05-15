"""Reliability helpers for provider adapters and report degradation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from time import sleep as default_sleep

from nlp_stock_prediction.contracts import (
    CredentialState,
    ProviderHealth,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.reliability.live import (
    DEFAULT_SOURCE_RELIABILITY_POLICY,
    ProviderReplacementSpec,
    SourceReliabilityPolicy,
    build_provider_compatibility_note,
    build_provider_replacement_playbook,
    build_source_reliability_note,
    build_source_reliability_notes,
    default_provider_replacement_playbooks,
    write_provider_replacement_playbook_artifacts,
    write_reliability_audit_artifacts,
    write_source_reliability_note_artifacts,
)


class ProviderError(Exception):
    """Base class for expected provider failures that should degrade gracefully."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        provider_error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.provider_error_type = provider_error_type


class ProviderTimeoutError(ProviderError):
    """Provider call exceeded its configured timeout."""

    def __init__(self, message: str = "Provider request timed out.") -> None:
        super().__init__(message, retryable=True, provider_error_type="timeout")


class ProviderRateLimitError(ProviderError):
    """Provider returned a retryable rate-limit response."""

    def __init__(
        self,
        message: str = "Provider rate limit was reached.",
        *,
        retry_after_seconds: int | None = None,
        reset_at: datetime | None = None,
        quota_exceeded: bool = False,
        status_code: int | None = 429,
    ) -> None:
        super().__init__(
            message,
            retryable=not quota_exceeded,
            status_code=status_code,
            provider_error_type="rate_limited",
        )
        self.retry_after_seconds = retry_after_seconds
        self.reset_at = reset_at
        self.quota_exceeded = quota_exceeded


class ProviderConfigurationError(ProviderError):
    """Provider cannot run because required configuration is missing."""

    def __init__(self, message: str = "Provider credentials or configuration are missing.") -> None:
        super().__init__(message, retryable=False, provider_error_type="missing_credentials")


class ProviderAuthenticationError(ProviderError):
    """Provider rejected configured credentials."""

    def __init__(
        self,
        message: str = "Provider authentication failed.",
        *,
        status_code: int | None = 401,
    ) -> None:
        super().__init__(
            message,
            retryable=False,
            status_code=status_code,
            provider_error_type="auth_failed",
        )


class ProviderMalformedResponseError(ProviderError):
    """Provider response could not be parsed into the expected shape."""

    def __init__(self, message: str = "Provider response was malformed.") -> None:
        super().__init__(message, retryable=False, provider_error_type="malformed_response")


class ProviderNoDataError(ProviderError):
    """Provider call completed but found no matching data."""

    def __init__(self, message: str = "Provider returned no matching data.") -> None:
        super().__init__(message, retryable=False, provider_error_type="no_data")


class ProviderStaleDataError(ProviderError):
    """Provider data is available but outside the freshness target."""

    def __init__(
        self,
        message: str = "Provider data is stale.",
        *,
        stale_after: datetime | None = None,
    ) -> None:
        super().__init__(message, retryable=False, provider_error_type="stale_data")
        self.stale_after = stale_after


class ProviderUpstreamError(ProviderError):
    """Provider or upstream dependency returned an availability failure."""

    def __init__(
        self,
        message: str = "Provider upstream is unavailable.",
        *,
        status_code: int | None = None,
        retryable: bool = True,
        provider_error_type: str | None = "upstream_unavailable",
    ) -> None:
        super().__init__(
            message,
            retryable=retryable,
            status_code=status_code,
            provider_error_type=provider_error_type,
        )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Deterministic retry/backoff policy for provider calls."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be non-negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be greater than or equal to base delay")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")

    def delay_for_attempt(self, failed_attempt_number: int) -> float:
        """Return delay after a failed 1-based attempt number."""

        if failed_attempt_number < 1:
            raise ValueError("failed_attempt_number must be at least 1")
        delay = self.base_delay_seconds * (self.multiplier ** (failed_attempt_number - 1))
        return min(delay, self.max_delay_seconds)


DEFAULT_RETRY_POLICY = RetryPolicy()


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """Classified failure with contract-ready warning and health snapshots."""

    provider_name: str
    status: ProviderStatus
    warning: ProviderWarning
    health: ProviderHealth


def default_should_retry(exception: Exception) -> bool:
    """Return whether the default retry loop should retry an exception."""

    if isinstance(exception, ProviderError):
        return exception.retryable
    return isinstance(exception, TimeoutError)


def retry_call[T](
    operation: Callable[[], T],
    *,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    should_retry: Callable[[Exception], bool] | None = None,
    sleep: Callable[[float], None] = default_sleep,
) -> T:
    """Run a provider operation with deterministic retry/backoff behavior."""

    retry_classifier = should_retry or default_should_retry
    for attempt_number in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as exception:
            if attempt_number >= policy.max_attempts or not retry_classifier(exception):
                raise
            sleep(policy.delay_for_attempt(attempt_number))
    raise RuntimeError("retry loop exhausted without returning or raising")


def warning_severity_for(code: WarningCode) -> WarningSeverity:
    """Return a default report severity for a provider warning code."""

    if code == WarningCode.NO_DATA:
        return WarningSeverity.INFO
    if code in {
        WarningCode.PARTIAL_DATA,
        WarningCode.RATE_LIMITED,
        WarningCode.STALE_DATA,
    }:
        return WarningSeverity.WARNING
    return WarningSeverity.ERROR


def provider_warning(
    *,
    provider_name: str,
    code: WarningCode,
    message: str,
    occurred_at: datetime,
    severity: WarningSeverity | None = None,
    retryable: bool = False,
    provider_status_code: int | None = None,
    provider_error_type: str | None = None,
    stale_after: datetime | None = None,
    raw_snapshot_id: str | None = None,
    source_url: str | None = None,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> ProviderWarning:
    """Build a provider warning with consistent defaults."""

    return ProviderWarning(
        code=code,
        severity=severity or warning_severity_for(code),
        message=message,
        provider_name=provider_name,
        retryable=retryable,
        provider_status_code=provider_status_code,
        provider_error_type=provider_error_type,
        occurred_at=occurred_at,
        stale_after=stale_after,
        raw_snapshot_id=raw_snapshot_id,
        source_url=source_url,
        metadata=metadata or {},
    )


def provider_health(
    *,
    provider_name: str,
    status: ProviderStatus,
    checked_at: datetime,
    credential_state: CredentialState = CredentialState.NOT_REQUIRED,
    latency_ms: int | None = None,
    rate_limit_remaining: int | None = None,
    rate_limit_reset_at: datetime | None = None,
    last_success_at: datetime | None = None,
    warnings: Sequence[ProviderWarning] = (),
) -> ProviderHealth:
    """Build a provider health snapshot matching the result-envelope status."""

    return ProviderHealth(
        provider_name=provider_name,
        status=status,
        checked_at=checked_at,
        credential_state=credential_state,
        latency_ms=latency_ms,
        rate_limit_remaining=rate_limit_remaining,
        rate_limit_reset_at=rate_limit_reset_at,
        last_success_at=last_success_at,
        warnings=tuple(warnings),
    )


def format_provider_health_message(health: ProviderHealth) -> str:
    """Return a concise report/log message for a provider health snapshot."""

    base = f"{health.provider_name}: {health.status.value}"
    if health.credential_state != CredentialState.NOT_REQUIRED:
        base = f"{base} (credentials: {health.credential_state.value})"
    if not health.warnings:
        return base
    warning_messages = "; ".join(warning.message for warning in health.warnings)
    return f"{base} - {warning_messages}"


def ok_result[T](
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    data: T,
    credential_state: CredentialState = CredentialState.NOT_REQUIRED,
    latency_ms: int | None = None,
    raw_snapshot_id: str | None = None,
    cache_key: str | None = None,
) -> ProviderResult[T]:
    """Build a successful provider result envelope."""

    return ProviderResult[T](
        provider_name=provider_name,
        status=ProviderStatus.OK,
        request=request,
        fetched_at=fetched_at,
        data=data,
        health=provider_health(
            provider_name=provider_name,
            status=ProviderStatus.OK,
            checked_at=fetched_at,
            credential_state=credential_state,
            latency_ms=latency_ms,
            last_success_at=fetched_at,
        ),
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


def partial_result[T](
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    data: T,
    warning_code: WarningCode = WarningCode.PARTIAL_DATA,
    message: str = "Provider returned partial data.",
    credential_state: CredentialState = CredentialState.NOT_REQUIRED,
    latency_ms: int | None = None,
    raw_snapshot_id: str | None = None,
    cache_key: str | None = None,
) -> ProviderResult[T]:
    """Build a partial provider result that preserves usable data."""

    warning = provider_warning(
        provider_name=provider_name,
        code=warning_code,
        message=message,
        occurred_at=fetched_at,
        raw_snapshot_id=raw_snapshot_id,
    )
    warnings = (warning,)
    return ProviderResult[T](
        provider_name=provider_name,
        status=ProviderStatus.PARTIAL,
        request=request,
        fetched_at=fetched_at,
        data=data,
        warnings=warnings,
        health=provider_health(
            provider_name=provider_name,
            status=ProviderStatus.PARTIAL,
            checked_at=fetched_at,
            credential_state=credential_state,
            latency_ms=latency_ms,
            last_success_at=fetched_at,
            warnings=warnings,
        ),
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


def empty_result(
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    message: str = "Provider returned no data.",
    credential_state: CredentialState = CredentialState.NOT_REQUIRED,
    latency_ms: int | None = None,
    raw_snapshot_id: str | None = None,
    cache_key: str | None = None,
) -> ProviderResult[None]:
    """Build an empty provider result with the required no-data warning."""

    warning = provider_warning(
        provider_name=provider_name,
        code=WarningCode.NO_DATA,
        severity=WarningSeverity.INFO,
        message=message,
        occurred_at=fetched_at,
        raw_snapshot_id=raw_snapshot_id,
    )
    warnings = (warning,)
    return ProviderResult[None](
        provider_name=provider_name,
        status=ProviderStatus.EMPTY,
        request=request,
        fetched_at=fetched_at,
        data=None,
        warnings=warnings,
        health=provider_health(
            provider_name=provider_name,
            status=ProviderStatus.EMPTY,
            checked_at=fetched_at,
            credential_state=credential_state,
            latency_ms=latency_ms,
            warnings=warnings,
        ),
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


def failed_result(
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    status: ProviderStatus,
    warning_code: WarningCode,
    message: str,
    credential_state: CredentialState = CredentialState.NOT_REQUIRED,
    retryable: bool = False,
    provider_status_code: int | None = None,
    provider_error_type: str | None = None,
    latency_ms: int | None = None,
    rate_limit_reset_at: datetime | None = None,
    raw_snapshot_id: str | None = None,
    cache_key: str | None = None,
) -> ProviderResult[None]:
    """Build a failure provider result with no data and visible health warning."""

    warning = provider_warning(
        provider_name=provider_name,
        code=warning_code,
        message=message,
        occurred_at=fetched_at,
        retryable=retryable,
        provider_status_code=provider_status_code,
        provider_error_type=provider_error_type,
        raw_snapshot_id=raw_snapshot_id,
    )
    warnings = (warning,)
    return ProviderResult[None](
        provider_name=provider_name,
        status=status,
        request=request,
        fetched_at=fetched_at,
        data=None,
        warnings=warnings,
        health=provider_health(
            provider_name=provider_name,
            status=status,
            checked_at=fetched_at,
            credential_state=credential_state,
            latency_ms=latency_ms,
            rate_limit_remaining=0 if status == ProviderStatus.RATE_LIMITED else None,
            rate_limit_reset_at=rate_limit_reset_at,
            warnings=warnings,
        ),
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


def classify_provider_exception(
    exception: Exception,
    *,
    provider_name: str,
    occurred_at: datetime,
    credential_state: CredentialState = CredentialState.NOT_REQUIRED,
    latency_ms: int | None = None,
    last_success_at: datetime | None = None,
) -> ProviderFailure:
    """Classify an expected provider exception into contract-ready health metadata."""

    status = ProviderStatus.FAILED
    code = WarningCode.UPSTREAM_UNAVAILABLE
    severity = WarningSeverity.ERROR
    retryable = False
    status_code: int | None = None
    error_type: str | None = None
    rate_limit_reset_at: datetime | None = None
    stale_after: datetime | None = None

    if isinstance(exception, ProviderRateLimitError):
        status = ProviderStatus.RATE_LIMITED
        code = WarningCode.QUOTA_EXCEEDED if exception.quota_exceeded else WarningCode.RATE_LIMITED
        severity = WarningSeverity.WARNING
        retryable = exception.retryable
        status_code = exception.status_code
        error_type = exception.provider_error_type
        rate_limit_reset_at = exception.reset_at
    elif isinstance(exception, ProviderTimeoutError):
        code = WarningCode.TIMEOUT
        retryable = True
        error_type = exception.provider_error_type
    elif isinstance(exception, ProviderConfigurationError):
        status = ProviderStatus.UNCONFIGURED
        code = WarningCode.MISSING_CREDENTIALS
        error_type = exception.provider_error_type
        credential_state = CredentialState.MISSING
    elif isinstance(exception, ProviderAuthenticationError):
        status = ProviderStatus.UNAUTHORIZED
        code = WarningCode.AUTH_FAILED
        status_code = exception.status_code
        error_type = exception.provider_error_type
        credential_state = CredentialState.INVALID
    elif isinstance(exception, ProviderMalformedResponseError):
        status = ProviderStatus.MALFORMED
        code = WarningCode.MALFORMED_RESPONSE
        error_type = exception.provider_error_type
    elif isinstance(exception, ProviderNoDataError):
        status = ProviderStatus.EMPTY
        code = WarningCode.NO_DATA
        severity = WarningSeverity.INFO
        error_type = exception.provider_error_type
    elif isinstance(exception, ProviderStaleDataError):
        status = ProviderStatus.STALE
        code = WarningCode.STALE_DATA
        severity = WarningSeverity.WARNING
        error_type = exception.provider_error_type
        stale_after = exception.stale_after
    elif isinstance(exception, ProviderUpstreamError):
        status_code = exception.status_code
        retryable = exception.retryable
        error_type = exception.provider_error_type
        if status_code in {401, 403}:
            status = ProviderStatus.UNAUTHORIZED
            code = WarningCode.AUTH_FAILED
            credential_state = CredentialState.INVALID

    warning = provider_warning(
        provider_name=provider_name,
        code=code,
        severity=severity,
        message=str(exception),
        occurred_at=occurred_at,
        retryable=retryable,
        provider_status_code=status_code,
        provider_error_type=error_type,
        stale_after=stale_after,
    )
    health = provider_health(
        provider_name=provider_name,
        status=status,
        checked_at=occurred_at,
        credential_state=credential_state,
        latency_ms=latency_ms,
        rate_limit_remaining=0 if status == ProviderStatus.RATE_LIMITED else None,
        rate_limit_reset_at=rate_limit_reset_at,
        last_success_at=last_success_at,
        warnings=(warning,),
    )
    return ProviderFailure(
        provider_name=provider_name,
        status=status,
        warning=warning,
        health=health,
    )


def result_from_exception(
    *,
    provider_name: str,
    request: ProviderRequest,
    exception: Exception,
    occurred_at: datetime,
    credential_state: CredentialState = CredentialState.NOT_REQUIRED,
    latency_ms: int | None = None,
    raw_snapshot_id: str | None = None,
    cache_key: str | None = None,
) -> ProviderResult[None]:
    """Convert an expected provider exception into a contract-compliant result."""

    failure = classify_provider_exception(
        exception,
        provider_name=provider_name,
        occurred_at=occurred_at,
        credential_state=credential_state,
        latency_ms=latency_ms,
    )
    return ProviderResult[None](
        provider_name=provider_name,
        status=failure.status,
        request=request,
        fetched_at=occurred_at,
        data=None,
        warnings=(failure.warning,),
        health=failure.health,
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


__all__ = [
    "DEFAULT_RETRY_POLICY",
    "DEFAULT_SOURCE_RELIABILITY_POLICY",
    "ProviderAuthenticationError",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderFailure",
    "ProviderMalformedResponseError",
    "ProviderNoDataError",
    "ProviderRateLimitError",
    "ProviderReplacementSpec",
    "ProviderStaleDataError",
    "ProviderTimeoutError",
    "ProviderUpstreamError",
    "RetryPolicy",
    "SourceReliabilityPolicy",
    "build_provider_compatibility_note",
    "build_provider_replacement_playbook",
    "build_source_reliability_note",
    "build_source_reliability_notes",
    "classify_provider_exception",
    "default_provider_replacement_playbooks",
    "default_should_retry",
    "empty_result",
    "failed_result",
    "format_provider_health_message",
    "ok_result",
    "partial_result",
    "provider_health",
    "provider_warning",
    "result_from_exception",
    "retry_call",
    "warning_severity_for",
    "write_provider_replacement_playbook_artifacts",
    "write_reliability_audit_artifacts",
    "write_source_reliability_note_artifacts",
]
