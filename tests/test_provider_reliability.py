from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import pytest

from nlp_stock_prediction.contracts import (
    CredentialState,
    EvidenceRequest,
    FreshnessStatus,
    ProviderStatus,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.providers._base import freshness_status
from nlp_stock_prediction.reliability import (
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUpstreamError,
    RetryPolicy,
    classify_provider_exception,
    empty_result,
    format_provider_health_message,
    partial_result,
    result_from_exception,
    retry_call,
)

pytestmark = pytest.mark.unit

FETCHED_AT = datetime(2026, 5, 11, 12, 0, tzinfo=UTC)
REQUEST = EvidenceRequest(
    request_id="provider-reliability",
    run_date=date(2026, 5, 11),
    tickers=("TSLA",),
    query="TSLA",
)


def test_retry_call_uses_deterministic_exponential_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []

    def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderTimeoutError("fixture provider timed out")
        return "ok"

    value = retry_call(
        flaky_operation,
        policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.25, multiplier=2.0),
        sleep=sleeps.append,
    )

    assert value == "ok"
    assert attempts == 3
    assert sleeps == [0.25, 0.5]


def test_freshness_status_marks_future_observations_unknown() -> None:
    status, seconds = freshness_status(
        observed_at=FETCHED_AT + timedelta(minutes=10),
        fetched_at=FETCHED_AT,
        stale_after_seconds=60,
    )

    assert status == FreshnessStatus.UNKNOWN
    assert seconds == 0


def test_retry_call_does_not_retry_non_retryable_provider_errors() -> None:
    attempts = 0

    def unauthorized_operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderUpstreamError(
            "fixture provider rejected credentials",
            status_code=401,
            retryable=False,
        )

    with pytest.raises(ProviderUpstreamError):
        retry_call(
            unauthorized_operation,
            policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.25),
            sleep=lambda _delay: None,
        )

    assert attempts == 1


def test_classify_provider_exception_maps_rate_limit_timeout_and_upstream() -> None:
    reset_at = FETCHED_AT + timedelta(minutes=5)

    rate_limited = classify_provider_exception(
        ProviderRateLimitError("provider quota window is exhausted", reset_at=reset_at),
        provider_name="fixture-provider",
        occurred_at=FETCHED_AT,
    )
    timeout = classify_provider_exception(
        ProviderTimeoutError("provider timed out"),
        provider_name="fixture-provider",
        occurred_at=FETCHED_AT,
    )
    upstream = classify_provider_exception(
        ProviderUpstreamError("gateway failed", status_code=503, retryable=True),
        provider_name="fixture-provider",
        occurred_at=FETCHED_AT,
    )

    assert rate_limited.status == ProviderStatus.RATE_LIMITED
    assert rate_limited.warning.code == WarningCode.RATE_LIMITED
    assert rate_limited.warning.retryable is True
    assert rate_limited.health.rate_limit_reset_at == reset_at
    assert timeout.status == ProviderStatus.FAILED
    assert timeout.warning.code == WarningCode.TIMEOUT
    assert timeout.warning.severity == WarningSeverity.ERROR
    assert upstream.status == ProviderStatus.FAILED
    assert upstream.warning.code == WarningCode.UPSTREAM_UNAVAILABLE
    assert upstream.warning.provider_status_code == 503


def test_graceful_result_helpers_preserve_data_and_surface_health_messages() -> None:
    partial = partial_result(
        provider_name="fixture-provider",
        request=REQUEST,
        fetched_at=FETCHED_AT,
        data=("evidence-a",),
        warning_code=WarningCode.PARTIAL_DATA,
        message="Only one fixture source returned data.",
    )
    empty = empty_result(
        provider_name="fixture-provider",
        request=REQUEST,
        fetched_at=FETCHED_AT,
        message="No fixture records matched TSLA.",
    )

    assert partial.status == ProviderStatus.PARTIAL
    assert partial.data == ("evidence-a",)
    assert partial.warnings[0].code == WarningCode.PARTIAL_DATA
    assert partial.health.warnings == partial.warnings
    assert "partial" in format_provider_health_message(partial.health).lower()
    assert empty.status == ProviderStatus.EMPTY
    assert empty.data is None
    assert empty.warnings[0].code == WarningCode.NO_DATA
    assert "No fixture records matched TSLA." in format_provider_health_message(empty.health)


def test_result_from_exception_creates_contract_compliant_failure_envelope() -> None:
    exception = ProviderUpstreamError("fixture upstream is unavailable", status_code=503)

    result = result_from_exception(
        provider_name="fixture-provider",
        request=REQUEST,
        exception=exception,
        occurred_at=FETCHED_AT,
        credential_state=CredentialState.NOT_REQUIRED,
    )

    assert result.status == ProviderStatus.FAILED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.UPSTREAM_UNAVAILABLE
    assert result.health.status == result.status
    assert result.health.warnings == result.warnings


def test_result_from_configuration_exception_creates_missing_credentials_envelope() -> None:
    exception = ProviderConfigurationError("fixture API key is not configured")

    result = result_from_exception(
        provider_name="fixture-provider",
        request=REQUEST,
        exception=exception,
        occurred_at=FETCHED_AT,
    )

    assert result.status == ProviderStatus.UNCONFIGURED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.MISSING_CREDENTIALS
    assert result.health.status == result.status
    assert result.health.credential_state == CredentialState.MISSING
    assert "credentials: missing" in format_provider_health_message(result.health)


def test_retry_policy_validates_attempt_and_delay_values() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)

    with pytest.raises(ValueError, match="base_delay_seconds"):
        RetryPolicy(base_delay_seconds=-0.1)

    with pytest.raises(ValueError, match="multiplier"):
        RetryPolicy(multiplier=0.9)


def test_retry_call_accepts_custom_retry_classifier() -> None:
    attempts = 0

    class FixtureError(Exception):
        pass

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FixtureError("retryable fixture error")
        return "ok"

    value = retry_call(
        operation,
        policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.01),
        should_retry=lambda exception: isinstance(exception, FixtureError),
        sleep=lambda _delay: None,
    )

    assert value == "ok"


def test_retry_call_raises_last_exception_after_attempts_are_exhausted() -> None:
    def operation() -> str:
        raise ProviderTimeoutError("still timing out")

    with pytest.raises(ProviderTimeoutError, match="still timing out"):
        retry_call(
            operation,
            policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.01),
            sleep=lambda _delay: None,
        )


def test_retry_call_preserves_callable_type() -> None:
    def operation() -> int:
        return 7

    typed_operation: Callable[[], int] = operation

    assert retry_call(typed_operation, sleep=lambda _delay: None) == 7
