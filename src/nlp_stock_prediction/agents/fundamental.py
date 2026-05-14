"""Fixture-backed foundation for optional fundamental-analysis agent output."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

from pydantic import ValidationError

from nlp_stock_prediction.contracts.analysis import (
    FundamentalNlpAnalysisRequest,
    FundamentalNlpAnalysisResponse,
    FundamentalNlpCitation,
)
from nlp_stock_prediction.contracts.base import JsonValue
from nlp_stock_prediction.contracts.enums import (
    CredentialState,
    FreshnessStatus,
    ProviderStatus,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.provenance import ProviderHealth, ProviderWarning
from nlp_stock_prediction.contracts.providers import ProviderResult

AgentRawResponse = Mapping[str, object] | str

_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "company",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "its",
    "of",
    "on",
    "or",
    "reported",
    "reports",
    "says",
    "the",
    "to",
    "with",
}
_POSITIVE_STANCES = {"bullish", "positive", "supportive", "supports", "beat", "growth"}
_NEGATIVE_STANCES = {"bearish", "negative", "conflicts", "risk", "miss", "decline"}
_STANCE_KEYS = (
    "fundamental_signal",
    "sentiment",
    "sentiment_hint",
    "stance",
    "direction",
)


class NoAgentAvailableError(RuntimeError):
    """Raised when no fixture or local supervised agent runner is configured."""


class FundamentalAgentRunner(Protocol):
    """Small runner boundary for fixture, manual, or future local Codex-agent execution."""

    runner_name: str

    def run_fundamental_analysis(self, request: FundamentalNlpAnalysisRequest) -> AgentRawResponse:
        """Return raw JSON-like agent output for one request."""


class NoAgentFundamentalAgentRunner:
    """Runner that makes no external calls and reports the optional lane as unavailable."""

    runner_name = "no-agent-available"

    def run_fundamental_analysis(self, request: FundamentalNlpAnalysisRequest) -> AgentRawResponse:
        raise NoAgentAvailableError(
            f"No fundamental agent runner is configured for request {request.request_id}."
        )


class FixtureFundamentalAgentRunner:
    """Deterministic runner backed by pre-recorded JSON or mapping payloads."""

    runner_name = "fixture-fundamental-agent-runner"

    def __init__(self, payload: AgentRawResponse | None) -> None:
        self._payload = payload

    def run_fundamental_analysis(self, request: FundamentalNlpAnalysisRequest) -> AgentRawResponse:
        if self._payload is None:
            raise NoAgentAvailableError(
                f"No fixture payload is configured for request {request.request_id}."
            )
        return self._payload


@dataclass(frozen=True, slots=True)
class FundamentalAgentValidation:
    """Validation warnings plus whether the output should be rejected."""

    warnings: tuple[ProviderWarning, ...]
    hard_failure: bool


class FundamentalAgentProvider:
    """Provider envelope for optional agent-backed fundamental analysis."""

    def __init__(
        self,
        *,
        runner: FundamentalAgentRunner | None = None,
        provider_name: str = "fundamental-agent",
        fetched_at: datetime | None = None,
    ) -> None:
        self.provider_name = provider_name
        self._runner = runner or NoAgentFundamentalAgentRunner()
        self._fetched_at = _ensure_aware_utc(fetched_at) if fetched_at else None

    def analyze_fundamentals(
        self,
        request: FundamentalNlpAnalysisRequest,
    ) -> ProviderResult[FundamentalNlpAnalysisResponse]:
        fetched_at = self._now()
        try:
            raw_response = self._runner.run_fundamental_analysis(request)
        except NoAgentAvailableError as error:
            return self._no_agent_result(request=request, fetched_at=fetched_at, error=error)

        raw_text = _raw_response_text(raw_response)
        raw_response_id = f"raw-fundamental-agent-{_sha256(raw_text)[:20]}"
        cache_key = _cache_key(request=request, runner_name=self._runner.runner_name)

        try:
            payload = _parse_raw_response(raw_response)
            payload = _with_audit_metadata(
                payload,
                request=request,
                provider_name=self.provider_name,
                runner_name=self._runner.runner_name,
                raw_response_id=raw_response_id,
                raw_text=raw_text,
            )
            response = FundamentalNlpAnalysisResponse.model_validate(payload)
        except (ValueError, ValidationError) as error:
            warning_code = _schema_validation_warning_code(error)
            message = (
                "Fundamental agent response has missing citations or invalid evidence "
                f"references: {error}"
                if warning_code == WarningCode.LLM_EVIDENCE_MISMATCH
                else f"Fundamental agent response failed schema validation: {error}"
            )
            warning = _warning(
                provider_name=self.provider_name,
                code=warning_code,
                severity=WarningSeverity.ERROR,
                message=message,
                occurred_at=fetched_at,
                raw_snapshot_id=raw_response_id,
            )
            return _result(
                provider_name=self.provider_name,
                status=ProviderStatus.MALFORMED,
                request=request,
                fetched_at=fetched_at,
                warnings=(warning,),
                raw_snapshot_id=raw_response_id,
                cache_key=cache_key,
            )

        validation = validate_fundamental_agent_response(
            request=request,
            response=response,
            provider_name=self.provider_name,
            occurred_at=fetched_at,
        )
        if validation.hard_failure:
            return _result(
                provider_name=self.provider_name,
                status=ProviderStatus.MALFORMED,
                request=request,
                fetched_at=fetched_at,
                warnings=validation.warnings,
                raw_snapshot_id=raw_response_id,
                cache_key=cache_key,
            )

        enriched_response = _enrich_response(response, validation.warnings)
        status = ProviderStatus.PARTIAL if validation.warnings else ProviderStatus.OK
        return _result(
            provider_name=self.provider_name,
            status=status,
            request=request,
            fetched_at=fetched_at,
            data=enriched_response,
            warnings=validation.warnings,
            raw_snapshot_id=raw_response_id,
            cache_key=cache_key,
        )

    def health(self) -> ProviderHealth:
        status = (
            ProviderStatus.UNCONFIGURED
            if isinstance(self._runner, NoAgentFundamentalAgentRunner)
            else ProviderStatus.OK
        )
        return _health(
            provider_name=self.provider_name,
            status=status,
            checked_at=self._now(),
            warnings=(),
        )

    def _no_agent_result(
        self,
        *,
        request: FundamentalNlpAnalysisRequest,
        fetched_at: datetime,
        error: NoAgentAvailableError,
    ) -> ProviderResult[FundamentalNlpAnalysisResponse]:
        warning = _warning(
            provider_name=self.provider_name,
            code=WarningCode.NO_DATA,
            severity=WarningSeverity.INFO,
            message=str(error),
            occurred_at=fetched_at,
            provider_error_type="no_agent_available",
        )
        return _result(
            provider_name=self.provider_name,
            status=ProviderStatus.EMPTY,
            request=request,
            fetched_at=fetched_at,
            warnings=(warning,),
        )

    def _now(self) -> datetime:
        return self._fetched_at or datetime.now(UTC)


class FixtureFundamentalAgentProvider(FundamentalAgentProvider):
    """Provider convenience wrapper for fixture-only tests and local JSON imports."""

    def __init__(
        self,
        *,
        payload: AgentRawResponse | None,
        provider_name: str = "fixture-fundamental-agent",
        fetched_at: datetime | None = None,
    ) -> None:
        super().__init__(
            runner=FixtureFundamentalAgentRunner(payload),
            provider_name=provider_name,
            fetched_at=fetched_at,
        )


def validate_fundamental_agent_response(
    *,
    request: FundamentalNlpAnalysisRequest,
    response: FundamentalNlpAnalysisResponse,
    provider_name: str,
    occurred_at: datetime,
) -> FundamentalAgentValidation:
    """Validate agent claims against the evidence packet used to produce them."""

    evidence_by_id = {record.evidence_id: record for record in request.evidence}
    warnings: list[ProviderWarning] = []
    hard_failure = False

    if response.request_id != request.request_id:
        warnings.append(
            _warning(
                provider_name=provider_name,
                code=WarningCode.LLM_SCHEMA_INVALID,
                severity=WarningSeverity.ERROR,
                message="Fundamental agent response request_id does not match the request.",
                occurred_at=occurred_at,
            )
        )
        hard_failure = True
    if response.ticker != request.ticker:
        warnings.append(
            _warning(
                provider_name=provider_name,
                code=WarningCode.LLM_SCHEMA_INVALID,
                severity=WarningSeverity.ERROR,
                message="Fundamental agent response ticker does not match the request.",
                occurred_at=occurred_at,
            )
        )
        hard_failure = True

    for source_evidence_id in response.source_evidence_ids:
        if source_evidence_id not in evidence_by_id:
            warnings.append(
                _warning(
                    provider_name=provider_name,
                    code=WarningCode.LLM_EVIDENCE_MISMATCH,
                    severity=WarningSeverity.ERROR,
                    message="Fundamental agent response declared an unknown source_evidence_id.",
                    occurred_at=occurred_at,
                    metadata={"evidence_id": source_evidence_id},
                )
            )
            hard_failure = True

    cited_records: list[SourceEvidence] = []
    stale_evidence_ids: set[str] = set()
    citation_keys: set[tuple[str, str, int | None, int | None]] = set()

    for citation in response.citations:
        records, citation_warnings, citation_hard_failure = _validate_citations(
            citations=(citation,),
            request=request,
            evidence_by_id=evidence_by_id,
            provider_name=provider_name,
            occurred_at=occurred_at,
            stale_evidence_ids=stale_evidence_ids,
            citation_keys=citation_keys,
            location="summary",
        )
        cited_records.extend(records)
        warnings.extend(citation_warnings)
        hard_failure = hard_failure or citation_hard_failure

    for claim in response.claims:
        if not claim.citations:
            warnings.append(
                _warning(
                    provider_name=provider_name,
                    code=WarningCode.LLM_EVIDENCE_MISMATCH,
                    severity=WarningSeverity.ERROR,
                    message="Fundamental agent claim is missing citations.",
                    occurred_at=occurred_at,
                    metadata={"claim_id": claim.claim_id},
                )
            )
            hard_failure = True
            continue
        records, citation_warnings, citation_hard_failure = _validate_citations(
            citations=claim.citations,
            request=request,
            evidence_by_id=evidence_by_id,
            provider_name=provider_name,
            occurred_at=occurred_at,
            stale_evidence_ids=stale_evidence_ids,
            citation_keys=citation_keys,
            location=f"claim:{claim.claim_id}",
        )
        cited_records.extend(records)
        warnings.extend(citation_warnings)
        hard_failure = hard_failure or citation_hard_failure
        if (
            claim.claim_type == "observed"
            and records
            and not _observed_claim_supported(
                claim.text,
                support_text=_support_text(claim.citations, records),
            )
        ):
            warnings.append(
                _warning(
                    provider_name=provider_name,
                    code=WarningCode.UNSUPPORTED_CLAIM,
                    severity=WarningSeverity.ERROR,
                    message="Fundamental agent observed claim is not supported by cited quotes.",
                    occurred_at=occurred_at,
                    metadata={"claim_id": claim.claim_id},
                )
            )
            hard_failure = True

    for risk in response.risks:
        if not risk.citations:
            warnings.append(
                _warning(
                    provider_name=provider_name,
                    code=WarningCode.LLM_EVIDENCE_MISMATCH,
                    severity=WarningSeverity.ERROR,
                    message="Fundamental agent risk is missing citations.",
                    occurred_at=occurred_at,
                    metadata={"risk_id": risk.risk_id},
                )
            )
            hard_failure = True
            continue
        records, citation_warnings, citation_hard_failure = _validate_citations(
            citations=risk.citations,
            request=request,
            evidence_by_id=evidence_by_id,
            provider_name=provider_name,
            occurred_at=occurred_at,
            stale_evidence_ids=stale_evidence_ids,
            citation_keys=citation_keys,
            location=f"risk:{risk.risk_id}",
        )
        cited_records.extend(records)
        warnings.extend(citation_warnings)
        hard_failure = hard_failure or citation_hard_failure

    contradiction_warning = _contradiction_warning(
        cited_records=tuple(cited_records),
        response=response,
        provider_name=provider_name,
        occurred_at=occurred_at,
    )
    if contradiction_warning is not None:
        warnings.append(contradiction_warning)

    return FundamentalAgentValidation(warnings=tuple(warnings), hard_failure=hard_failure)


def _schema_validation_warning_code(error: ValueError | ValidationError) -> WarningCode:
    message = str(error).lower()
    if "citation" in message or "source_evidence_id" in message:
        return WarningCode.LLM_EVIDENCE_MISMATCH
    return WarningCode.LLM_SCHEMA_INVALID


def _validate_citations(
    *,
    citations: Sequence[FundamentalNlpCitation],
    request: FundamentalNlpAnalysisRequest,
    evidence_by_id: Mapping[str, SourceEvidence],
    provider_name: str,
    occurred_at: datetime,
    stale_evidence_ids: set[str],
    citation_keys: set[tuple[str, str, int | None, int | None]],
    location: str,
) -> tuple[list[SourceEvidence], list[ProviderWarning], bool]:
    records: list[SourceEvidence] = []
    warnings: list[ProviderWarning] = []
    hard_failure = False
    for citation in citations:
        citation_key = (
            citation.evidence_id,
            citation.quote,
            citation.start_char,
            citation.end_char,
        )
        if citation_key in citation_keys:
            duplicate_record = evidence_by_id.get(citation.evidence_id)
            if duplicate_record is not None:
                records.append(duplicate_record)
            continue
        citation_keys.add(citation_key)
        record = evidence_by_id.get(citation.evidence_id)
        if record is None:
            warnings.append(
                _warning(
                    provider_name=provider_name,
                    code=WarningCode.LLM_EVIDENCE_MISMATCH,
                    severity=WarningSeverity.ERROR,
                    message="Fundamental agent citation referenced unknown evidence.",
                    occurred_at=occurred_at,
                    metadata={"evidence_id": citation.evidence_id, "location": location},
                )
            )
            hard_failure = True
            continue
        if not _evidence_mentions_ticker(record, request.ticker):
            warnings.append(
                _warning(
                    provider_name=provider_name,
                    code=WarningCode.LLM_EVIDENCE_MISMATCH,
                    severity=WarningSeverity.ERROR,
                    message="Fundamental agent citation referenced evidence for another ticker.",
                    occurred_at=occurred_at,
                    metadata={
                        "evidence_id": citation.evidence_id,
                        "location": location,
                        "ticker": request.ticker,
                    },
                )
            )
            hard_failure = True
        if not _quote_matches(record.text, citation):
            warnings.append(
                _warning(
                    provider_name=provider_name,
                    code=WarningCode.LLM_EVIDENCE_MISMATCH,
                    severity=WarningSeverity.ERROR,
                    message="Fundamental agent citation quote was not found in source evidence.",
                    occurred_at=occurred_at,
                    metadata={"evidence_id": citation.evidence_id, "location": location},
                )
            )
            hard_failure = True
        if (
            record.provenance.freshness_status == FreshnessStatus.STALE
            and record.evidence_id not in stale_evidence_ids
        ):
            stale_evidence_ids.add(record.evidence_id)
            warnings.append(
                _warning(
                    provider_name=provider_name,
                    code=WarningCode.STALE_DATA,
                    severity=WarningSeverity.WARNING,
                    message="Fundamental agent cited stale source evidence.",
                    occurred_at=occurred_at,
                    stale_after=record.provenance.observed_at,
                    raw_snapshot_id=record.provenance.raw_snapshot_id,
                    source_url=record.provenance.source_url or record.provenance.permalink,
                    metadata={"evidence_id": record.evidence_id, "location": location},
                )
            )
        records.append(record)
    return records, warnings, hard_failure


def _contradiction_warning(
    *,
    cited_records: tuple[SourceEvidence, ...],
    response: FundamentalNlpAnalysisResponse,
    provider_name: str,
    occurred_at: datetime,
) -> ProviderWarning | None:
    positive_ids: list[str] = []
    negative_ids: list[str] = []
    for record in cited_records:
        stance = _evidence_stance(record)
        if stance == "positive" and record.evidence_id not in positive_ids:
            positive_ids.append(record.evidence_id)
        elif stance == "negative" and record.evidence_id not in negative_ids:
            negative_ids.append(record.evidence_id)

    if not positive_ids or not negative_ids or response.contradictions:
        return None
    metadata: dict[str, JsonValue] = {
        "positive_evidence_ids": cast(list[JsonValue], positive_ids.copy()),
        "negative_evidence_ids": cast(list[JsonValue], negative_ids.copy()),
    }
    return _warning(
        provider_name=provider_name,
        code=WarningCode.PARTIAL_DATA,
        severity=WarningSeverity.WARNING,
        message="Fundamental agent cited contradictory evidence without noting a contradiction.",
        occurred_at=occurred_at,
        metadata=metadata,
    )


def _parse_raw_response(raw_response: AgentRawResponse) -> dict[str, object]:
    if isinstance(raw_response, str):
        try:
            loaded = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise ValueError("fundamental agent response is not valid JSON") from error
        if not isinstance(loaded, Mapping):
            raise ValueError("fundamental agent JSON response must be an object")
        return dict(cast(Mapping[str, object], loaded))
    return dict(raw_response)


def _with_audit_metadata(
    payload: dict[str, object],
    *,
    request: FundamentalNlpAnalysisRequest,
    provider_name: str,
    runner_name: str,
    raw_response_id: str,
    raw_text: str,
) -> dict[str, object]:
    normalized = dict(payload)
    normalized.setdefault(
        "audit",
        {
            "provider_name": provider_name,
            "runner_name": runner_name,
            "prompt_version": request.prompt_version,
            "schema_version": request.schema_version,
            "prompt_sha256": _sha256(_request_text(request)),
            "response_sha256": _sha256(raw_text),
            "raw_response_id": raw_response_id,
            "audit_artifact_ids": [raw_response_id],
            "metadata": {
                "request_id": request.request_id,
                "source_evidence_ids": list(request.source_evidence_ids),
            },
        },
    )
    return normalized


def _enrich_response(
    response: FundamentalNlpAnalysisResponse,
    warnings: tuple[ProviderWarning, ...],
) -> FundamentalNlpAnalysisResponse:
    validation_warnings = response.validation_warnings + warnings
    component_warnings = response.warnings + warnings
    evidence = response.evidence or tuple(
        citation.as_evidence_reference() for citation in _all_citations(response)
    )
    return response.model_copy(
        update={
            "validation_warnings": validation_warnings,
            "warnings": component_warnings,
            "evidence": evidence,
        }
    )


def _all_citations(
    response: FundamentalNlpAnalysisResponse,
) -> tuple[FundamentalNlpCitation, ...]:
    citations: list[FundamentalNlpCitation] = list(response.citations)
    for claim in response.claims:
        citations.extend(claim.citations)
    for risk in response.risks:
        citations.extend(risk.citations)
    return tuple(citations)


def _quote_matches(text: str, citation: FundamentalNlpCitation) -> bool:
    if citation.start_char is not None and citation.end_char is not None:
        if citation.end_char > len(text):
            return False
        return text[citation.start_char : citation.end_char] == citation.quote
    return citation.quote in text or citation.quote.lower() in text.lower()


def _evidence_mentions_ticker(record: SourceEvidence, ticker: str) -> bool:
    tickers = {matched_ticker.upper() for matched_ticker in record.matched_tickers}
    if record.ticker is not None:
        tickers.add(record.ticker.upper())
    return not tickers or ticker.upper() in tickers


def _observed_claim_supported(claim_text: str, *, support_text: str) -> bool:
    claim_tokens = _meaningful_tokens(claim_text)
    if not claim_tokens:
        return True
    support = support_text.lower()
    numeric_tokens = tuple(token for token in claim_tokens if any(char.isdigit() for char in token))
    if numeric_tokens and any(token not in support for token in numeric_tokens):
        return False
    required_overlap = min(3, max(1, len(claim_tokens) // 2))
    overlap = sum(1 for token in claim_tokens if token in support)
    return overlap >= required_overlap


def _support_text(
    citations: Sequence[FundamentalNlpCitation],
    records: Sequence[SourceEvidence],
) -> str:
    return " ".join(
        [record.text for record in records] + [citation.quote for citation in citations]
    )


def _meaningful_tokens(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw_token in re.split(r"[\s,/;:()]+", value.lower()):
        token = "".join(
            character for character in raw_token if character.isalnum() or character == "."
        ).strip(".")
        if len(token) >= 2 and token not in _STOPWORDS:
            tokens.append(token)
    return tuple(tokens)


def _evidence_stance(record: SourceEvidence) -> str | None:
    for key in _STANCE_KEYS:
        raw_value = record.metadata.get(key)
        if raw_value is None:
            continue
        value = str(raw_value).strip().lower()
        if value in _POSITIVE_STANCES:
            return "positive"
        if value in _NEGATIVE_STANCES:
            return "negative"
    return None


def _raw_response_text(raw_response: AgentRawResponse) -> str:
    if isinstance(raw_response, str):
        return raw_response
    return json.dumps(raw_response, sort_keys=True, separators=(",", ":"), default=str)


def _request_text(request: FundamentalNlpAnalysisRequest) -> str:
    return json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _cache_key(*, request: FundamentalNlpAnalysisRequest, runner_name: str) -> str:
    payload = {
        "request_id": request.request_id,
        "ticker": request.ticker,
        "run_date": request.run_date.isoformat(),
        "prompt_version": request.prompt_version,
        "schema_version": request.schema_version,
        "runner_name": runner_name,
        "source_evidence_ids": list(request.source_evidence_ids),
    }
    return f"fundamental-agent:{request.ticker}:{_sha256(json.dumps(payload, sort_keys=True))[:16]}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _warning(
    *,
    provider_name: str,
    code: WarningCode,
    severity: WarningSeverity,
    message: str,
    occurred_at: datetime,
    retryable: bool = False,
    provider_status_code: int | None = None,
    provider_error_type: str | None = None,
    stale_after: datetime | None = None,
    raw_snapshot_id: str | None = None,
    source_url: str | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ProviderWarning:
    return ProviderWarning(
        code=code,
        severity=severity,
        message=message,
        provider_name=provider_name,
        retryable=retryable,
        provider_status_code=provider_status_code,
        provider_error_type=provider_error_type,
        occurred_at=_ensure_aware_utc(occurred_at),
        stale_after=_ensure_aware_utc(stale_after) if stale_after else None,
        raw_snapshot_id=raw_snapshot_id,
        source_url=source_url,
        metadata=dict(metadata or {}),
    )


def _health(
    *,
    provider_name: str,
    status: ProviderStatus,
    checked_at: datetime,
    warnings: tuple[ProviderWarning, ...],
) -> ProviderHealth:
    return ProviderHealth(
        provider_name=provider_name,
        status=status,
        checked_at=_ensure_aware_utc(checked_at),
        credential_state=CredentialState.NOT_REQUIRED,
        last_success_at=(
            _ensure_aware_utc(checked_at)
            if status in {ProviderStatus.OK, ProviderStatus.PARTIAL}
            else None
        ),
        warnings=warnings,
    )


def _result(
    *,
    provider_name: str,
    status: ProviderStatus,
    request: FundamentalNlpAnalysisRequest,
    fetched_at: datetime,
    data: FundamentalNlpAnalysisResponse | None = None,
    warnings: tuple[ProviderWarning, ...] = (),
    raw_snapshot_id: str | None = None,
    cache_key: str | None = None,
) -> ProviderResult[FundamentalNlpAnalysisResponse]:
    return ProviderResult[FundamentalNlpAnalysisResponse](
        provider_name=provider_name,
        status=status,
        request=request,
        fetched_at=_ensure_aware_utc(fetched_at),
        data=data,
        warnings=warnings,
        health=_health(
            provider_name=provider_name,
            status=status,
            checked_at=fetched_at,
            warnings=warnings,
        ),
        raw_snapshot_id=raw_snapshot_id,
        cache_key=cache_key,
    )


__all__ = [
    "FixtureFundamentalAgentProvider",
    "FixtureFundamentalAgentRunner",
    "FundamentalAgentProvider",
    "FundamentalAgentRunner",
    "FundamentalAgentValidation",
    "NoAgentAvailableError",
    "NoAgentFundamentalAgentRunner",
    "validate_fundamental_agent_response",
]
