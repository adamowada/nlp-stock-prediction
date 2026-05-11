"""Fixture-backed LLM extractor scaffolding for deterministic tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

from nlp_stock_prediction.contracts import (
    CredentialState,
    ExtractionRequest,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    StrategyExtraction,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.extraction.validation import (
    parse_llm_json_response,
    validate_llm_strategy_payloads,
)


class FixtureLLMExtractor:
    """Deterministic LLM extractor that validates pre-recorded JSON-like payloads."""

    def __init__(
        self,
        *,
        payloads: Iterable[Mapping[str, object]] | str,
        provider_name: str = "fixture-llm-extractor",
        fetched_at: datetime | None = None,
    ) -> None:
        self.provider_name = provider_name
        self._raw_response = payloads if isinstance(payloads, str) else None
        self._payloads = (
            None if isinstance(payloads, str) else tuple(dict(item) for item in payloads)
        )
        self._fetched_at = fetched_at or datetime.now(tz=UTC)

    def extract_strategies(
        self, request: ExtractionRequest
    ) -> ProviderResult[tuple[StrategyExtraction, ...]]:
        try:
            payloads = (
                parse_llm_json_response(self._raw_response)
                if self._raw_response is not None
                else self._payloads or ()
            )
        except ValueError as error:
            parse_warnings = (
                self._warning(
                    code=WarningCode.LLM_SCHEMA_INVALID,
                    severity=WarningSeverity.ERROR,
                    message=f"Fixture LLM response could not be parsed: {error}",
                ),
            )
            return ProviderResult[tuple[StrategyExtraction, ...]](
                provider_name=self.provider_name,
                status=ProviderStatus.MALFORMED,
                request=request,
                fetched_at=self._fetched_at,
                data=None,
                warnings=parse_warnings,
                health=self._health(ProviderStatus.MALFORMED, parse_warnings),
            )

        validation = validate_llm_strategy_payloads(
            payloads,
            evidence=request.evidence,
            provider_name=self.provider_name,
            occurred_at=self._fetched_at,
        )
        data: tuple[StrategyExtraction, ...] | None
        result_warnings: tuple[ProviderWarning, ...]
        if validation.strategies and validation.warnings:
            status = ProviderStatus.PARTIAL
            data = validation.strategies
            result_warnings = validation.warnings
        elif validation.strategies:
            status = ProviderStatus.OK
            data = validation.strategies
            result_warnings = ()
        elif validation.warnings:
            status = ProviderStatus.MALFORMED
            data = None
            result_warnings = validation.warnings
        else:
            status = ProviderStatus.EMPTY
            data = None
            result_warnings = (
                self._warning(
                    code=WarningCode.NO_DATA,
                    severity=WarningSeverity.INFO,
                    message="Fixture LLM extractor produced no strategies.",
                ),
            )

        return ProviderResult[tuple[StrategyExtraction, ...]](
            provider_name=self.provider_name,
            status=status,
            request=request,
            fetched_at=self._fetched_at,
            data=data,
            warnings=result_warnings,
            health=self._health(status, result_warnings),
        )

    def health(self) -> ProviderHealth:
        return self._health(ProviderStatus.OK, ())

    def _health(
        self,
        status: ProviderStatus,
        warnings: tuple[ProviderWarning, ...],
    ) -> ProviderHealth:
        return ProviderHealth(
            provider_name=self.provider_name,
            status=status,
            checked_at=self._fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            last_success_at=(
                self._fetched_at if status in {ProviderStatus.OK, ProviderStatus.PARTIAL} else None
            ),
            warnings=warnings,
        )

    def _warning(
        self,
        *,
        code: WarningCode,
        severity: WarningSeverity,
        message: str,
    ) -> ProviderWarning:
        return ProviderWarning(
            code=code,
            severity=severity,
            message=message,
            provider_name=self.provider_name,
            occurred_at=self._fetched_at,
        )


__all__ = ["FixtureLLMExtractor"]
