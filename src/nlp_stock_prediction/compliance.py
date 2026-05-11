"""Compliance guardrails for generated v1 reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from nlp_stock_prediction.contracts import (
    DailyReport,
    Disclaimer,
    JsonValue,
    RetrievalMethod,
    SourceKind,
    SourceProvenance,
    TradeCandidate,
)

ComplianceSeverity = Literal["warning", "error"]

DEFAULT_V1_DISCLAIMER_TEXT = (
    "Educational research only; not financial advice; no automatic trading or brokerage "
    "execution is performed by this v1 report."
)

_FORBIDDEN_BROKERAGE_KEYS = frozenset(
    {
        "broker",
        "brokerage",
        "broker_order_id",
        "brokerage_order_id",
        "brokerage_account_id",
    }
)
_FORBIDDEN_AUTO_TRADE_KEYS = frozenset(
    {
        "auto_trade",
        "auto_trading",
        "automated_trading",
        "submit_order",
        "place_order",
    }
)
_FORBIDDEN_EXECUTION_KEYS = frozenset(
    {
        "execution",
        "execution_payload",
        "order",
        "order_payload",
    }
)


@dataclass(frozen=True, slots=True)
class ComplianceIssue:
    """One actionable compliance guardrail issue."""

    code: str
    message: str
    subject_id: str | None = None
    severity: ComplianceSeverity = "error"


@dataclass(frozen=True, slots=True)
class RetrievalSourceClassification:
    """Readable classification of a normalized datum's retrieval route."""

    provider_name: str
    source_kind: SourceKind
    retrieval_method: RetrievalMethod
    label: str
    is_official_api: bool
    is_public_scraping_fallback: bool
    requires_scraping_drift_monitor: bool


class ComplianceError(ValueError):
    """Raised when a report fails v1 guardrails and strict mode is requested."""

    def __init__(self, issues: Iterable[ComplianceIssue]) -> None:
        issue_tuple = tuple(issues)
        self.issues = issue_tuple
        message = ", ".join(issue.code for issue in issue_tuple) or "unknown_compliance_issue"
        super().__init__(message)


def build_v1_disclaimer(
    *,
    disclaimer_id: str = "educational-report-v1",
    version: str = "v1",
    text: str = DEFAULT_V1_DISCLAIMER_TEXT,
) -> Disclaimer:
    """Build the default v1 disclaimer required by generated reports."""

    return Disclaimer(
        disclaimer_id=disclaimer_id,
        version=version,
        text=text,
        educational_only=True,
        not_financial_advice=True,
        no_auto_trading=True,
    )


def validate_disclaimer_guardrails(disclaimer: Disclaimer) -> tuple[ComplianceIssue, ...]:
    """Check that disclaimer flags and text carry all v1 legal/financial guardrails."""

    issues: list[ComplianceIssue] = []
    text = " ".join(disclaimer.text.lower().split())
    if not disclaimer.educational_only:
        issues.append(
            ComplianceIssue(
                code="disclaimer_not_educational_only",
                message="The v1 disclaimer must be educational-only.",
                subject_id=disclaimer.disclaimer_id,
            )
        )
    if "educational" not in text:
        issues.append(
            ComplianceIssue(
                code="missing_educational_only_text",
                message="The disclaimer text must clearly say the report is educational only.",
                subject_id=disclaimer.disclaimer_id,
            )
        )
    if not disclaimer.not_financial_advice:
        issues.append(
            ComplianceIssue(
                code="disclaimer_allows_financial_advice",
                message="The v1 disclaimer must state that it is not financial advice.",
                subject_id=disclaimer.disclaimer_id,
            )
        )
    if "not financial advice" not in text and "not investment advice" not in text:
        issues.append(
            ComplianceIssue(
                code="missing_not_financial_advice_text",
                message="The disclaimer text must explicitly say it is not financial advice.",
                subject_id=disclaimer.disclaimer_id,
            )
        )
    if not disclaimer.no_auto_trading:
        issues.append(
            ComplianceIssue(
                code="disclaimer_allows_auto_trading",
                message="The v1 disclaimer must prohibit automatic trading.",
                subject_id=disclaimer.disclaimer_id,
            )
        )
    if not _mentions_no_auto_trading(text):
        issues.append(
            ComplianceIssue(
                code="missing_no_auto_trading_text",
                message="The disclaimer text must clearly prohibit automatic trading.",
                subject_id=disclaimer.disclaimer_id,
            )
        )
    return tuple(issues)


def _mentions_no_auto_trading(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "no automatic trading",
            "no auto trading",
            "no auto-trading",
            "does not place trades",
            "will not place trades",
        )
    )


def validate_no_auto_trading_metadata(
    metadata: Mapping[str, object],
    *,
    subject_id: str | None = None,
) -> tuple[ComplianceIssue, ...]:
    """Reject metadata fields that look like brokerage execution instructions."""

    issues: list[ComplianceIssue] = []
    flattened_keys = set(_flatten_metadata_keys(metadata))
    if flattened_keys & _FORBIDDEN_BROKERAGE_KEYS:
        issues.append(
            ComplianceIssue(
                code="forbidden_brokerage_metadata",
                message="V1 reports must not include brokerage account or order metadata.",
                subject_id=subject_id,
            )
        )
    if flattened_keys & _FORBIDDEN_AUTO_TRADE_KEYS:
        issues.append(
            ComplianceIssue(
                code="forbidden_auto_trade_metadata",
                message="V1 reports must not include auto-trading instructions.",
                subject_id=subject_id,
            )
        )
    if flattened_keys & _FORBIDDEN_EXECUTION_KEYS:
        issues.append(
            ComplianceIssue(
                code="forbidden_execution_metadata",
                message="V1 reports must not include executable order payloads.",
                subject_id=subject_id,
            )
        )
    return tuple(issues)


def _flatten_metadata_keys(metadata: Mapping[str, object]) -> Iterable[str]:
    for key, value in metadata.items():
        normalized_key = key.lower()
        yield normalized_key
        if isinstance(value, Mapping):
            yield from _flatten_metadata_keys(value)


def validate_trade_candidate_guardrails(
    candidate: TradeCandidate,
) -> tuple[ComplianceIssue, ...]:
    """Check one candidate for no-auto-trading metadata guardrails."""

    return validate_no_auto_trading_metadata(
        candidate.metadata,
        subject_id=candidate.candidate_id,
    )


def validate_report_guardrails(
    report: DailyReport,
    *,
    raise_on_error: bool = False,
) -> tuple[ComplianceIssue, ...]:
    """Validate report-level legal/financial and no-auto-trading guardrails."""

    issues = list(validate_disclaimer_guardrails(report.disclaimer))
    for candidate in report.trade_candidates:
        issues.extend(validate_trade_candidate_guardrails(candidate))
    issue_tuple = tuple(issues)
    if raise_on_error and issue_tuple:
        raise ComplianceError(issue_tuple)
    return issue_tuple


def classify_retrieval_source(provenance: SourceProvenance) -> RetrievalSourceClassification:
    """Distinguish official API data from public scraping fallback data."""

    is_official_api = provenance.retrieval_method == RetrievalMethod.OFFICIAL_API
    is_public_scraping = provenance.retrieval_method == RetrievalMethod.PUBLIC_SCRAPE
    return RetrievalSourceClassification(
        provider_name=provenance.provider_name,
        source_kind=provenance.source_kind,
        retrieval_method=provenance.retrieval_method,
        label=provenance.retrieval_method.value,
        is_official_api=is_official_api,
        is_public_scraping_fallback=is_public_scraping,
        requires_scraping_drift_monitor=is_public_scraping,
    )


def metadata_is_guardrail_clean(metadata: Mapping[str, JsonValue]) -> bool:
    """Return whether metadata contains no v1 execution guardrail issues."""

    return not validate_no_auto_trading_metadata(metadata)


__all__ = [
    "DEFAULT_V1_DISCLAIMER_TEXT",
    "ComplianceError",
    "ComplianceIssue",
    "ComplianceSeverity",
    "RetrievalSourceClassification",
    "build_v1_disclaimer",
    "classify_retrieval_source",
    "metadata_is_guardrail_clean",
    "validate_disclaimer_guardrails",
    "validate_no_auto_trading_metadata",
    "validate_report_guardrails",
    "validate_trade_candidate_guardrails",
]
