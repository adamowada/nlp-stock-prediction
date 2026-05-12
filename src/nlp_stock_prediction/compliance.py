"""Compliance guardrails for generated v1 reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal
from urllib.parse import urlsplit

from nlp_stock_prediction.contracts import (
    CredentialState,
    DailyReport,
    Disclaimer,
    JsonValue,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    RetrievalMethod,
    SourceKind,
    SourceProvenance,
    TradeCandidate,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.providers._base import (
    provider_result,
    provider_warning,
)

ComplianceSeverity = Literal["warning", "error"]
RobotsPolicyStatus = Literal[
    "allowed",
    "partially_allowed",
    "disallowed",
    "unknown",
    "not_applicable",
]
SourceFallbackBehavior = Literal[
    "degraded_result",
    "fixture_only",
    "official_api",
    "feasibility_probe",
    "manual_review",
]
SourcePolicyDecision = Literal["allowed", "blocked", "login_required"]

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


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    """Compliance policy for one external source entrypoint."""

    source_id: str
    provider_name: str
    display_name: str
    base_url: str
    robots_txt_url: str | None
    robots_status: RobotsPolicyStatus
    allowed_paths: tuple[str, ...]
    disallowed_paths: tuple[str, ...] = ()
    crawl_delay_seconds: float | None = None
    login_required: bool = False
    javascript_required: bool = False
    fallback_behavior: SourceFallbackBehavior = "degraded_result"
    last_reviewed: date | None = None
    notes: str | None = None

    def evaluate_url(self, source_url: str) -> SourcePolicyEvaluation:
        """Evaluate whether a URL is within this source's public access policy."""

        split_source = urlsplit(source_url)
        split_base = urlsplit(self.base_url)
        source_host = split_source.netloc.lower()
        base_host = split_base.netloc.lower()
        if not split_source.scheme or not source_host:
            return SourcePolicyEvaluation(
                policy=self,
                source_url=source_url,
                decision="blocked",
                reason="source_url_must_be_absolute_http_url",
                matched_path=None,
            )
        if split_source.scheme not in {"http", "https"}:
            return SourcePolicyEvaluation(
                policy=self,
                source_url=source_url,
                decision="blocked",
                reason="source_url_scheme_not_supported",
                matched_path=None,
            )
        if base_host and source_host != base_host:
            return SourcePolicyEvaluation(
                policy=self,
                source_url=source_url,
                decision="blocked",
                reason="host_outside_source_policy",
                matched_path=None,
            )

        path = _normalize_policy_path(split_source.path)
        disallowed_path = _first_matching_policy_path(path, self.disallowed_paths)
        if disallowed_path is not None:
            return SourcePolicyEvaluation(
                policy=self,
                source_url=source_url,
                decision="blocked",
                reason="path_disallowed_by_source_policy",
                matched_path=disallowed_path,
            )
        allowed_path = _first_matching_policy_path(path, self.allowed_paths)
        if allowed_path is None:
            return SourcePolicyEvaluation(
                policy=self,
                source_url=source_url,
                decision="blocked",
                reason="path_not_allowlisted_by_source_policy",
                matched_path=None,
            )
        if self.login_required:
            return SourcePolicyEvaluation(
                policy=self,
                source_url=source_url,
                decision="login_required",
                reason="source_requires_login",
                matched_path=allowed_path,
            )
        return SourcePolicyEvaluation(
            policy=self,
            source_url=source_url,
            decision="allowed",
            reason="allowed_by_source_policy",
            matched_path=allowed_path,
        )


@dataclass(frozen=True, slots=True)
class SourcePolicyEvaluation:
    """Result of evaluating one URL against a source policy."""

    policy: SourcePolicy
    source_url: str
    decision: SourcePolicyDecision
    reason: str
    matched_path: str | None

    @property
    def allowed(self) -> bool:
        return self.decision == "allowed"

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {
            "source_id": self.policy.source_id,
            "display_name": self.policy.display_name,
            "decision": self.decision,
            "reason": self.reason,
            "matched_path": self.matched_path,
            "robots_status": self.policy.robots_status,
            "robots_txt_url": self.policy.robots_txt_url,
            "fallback_behavior": self.policy.fallback_behavior,
            "login_required": self.policy.login_required,
            "javascript_required": self.policy.javascript_required,
            "crawl_delay_seconds": self.policy.crawl_delay_seconds,
            "last_reviewed": self.policy.last_reviewed.isoformat()
            if self.policy.last_reviewed
            else None,
        }


class ComplianceError(ValueError):
    """Raised when a report fails v1 guardrails and strict mode is requested."""

    def __init__(self, issues: Iterable[ComplianceIssue]) -> None:
        issue_tuple = tuple(issues)
        self.issues = issue_tuple
        message = ", ".join(issue.code for issue in issue_tuple) or "unknown_compliance_issue"
        super().__init__(message)


DEFAULT_SOURCE_POLICIES: Mapping[str, SourcePolicy] = {
    "reddit_wsb": SourcePolicy(
        source_id="reddit_wsb",
        provider_name="reddit-public",
        display_name="r/wallstreetbets public pages",
        base_url="https://www.reddit.com",
        robots_txt_url="https://www.reddit.com/robots.txt",
        robots_status="partially_allowed",
        allowed_paths=("/r/wallstreetbets/",),
        disallowed_paths=(
            "/api/",
            "/search/",
            "/r/wallstreetbets/search/",
            "/svc/",
            "/r/wallstreetbets/.json",
        ),
        login_required=False,
        javascript_required=True,
        fallback_behavior="degraded_result",
        last_reviewed=date(2026, 5, 11),
        notes="Use public subreddit and post HTML only; do not use Reddit API/search endpoints.",
    ),
    "apnews_financial_markets": SourcePolicy(
        source_id="apnews_financial_markets",
        provider_name="apnews-public",
        display_name="AP News financial markets public pages",
        base_url="https://apnews.com",
        robots_txt_url="https://apnews.com/robots.txt",
        robots_status="partially_allowed",
        allowed_paths=("/hub/financial-markets", "/article/"),
        disallowed_paths=("/api/", "/api/v2/feed/", "/search", "/rss"),
        login_required=False,
        javascript_required=False,
        fallback_behavior="degraded_result",
        last_reviewed=date(2026, 5, 11),
        notes="Use hub/article HTML only; avoid AP API, feed, search, and RSS paths.",
    ),
    "candlecharts_live_charts": SourcePolicy(
        source_id="candlecharts_live_charts",
        provider_name="candlecharts-public",
        display_name="Candlecharts live charts page",
        base_url="https://candlecharts.com",
        robots_txt_url="https://candlecharts.com/robots.txt",
        robots_status="unknown",
        allowed_paths=("/live-charts",),
        disallowed_paths=(),
        login_required=False,
        javascript_required=True,
        fallback_behavior="feasibility_probe",
        last_reviewed=date(2026, 5, 11),
        notes="Probe public HTML only; do not scrape embedded TradingView internals.",
    ),
    "x_recent_search": SourcePolicy(
        source_id="x_recent_search",
        provider_name="x-recent-search",
        display_name="X recent-search official API",
        base_url="https://api.x.com",
        robots_txt_url=None,
        robots_status="not_applicable",
        allowed_paths=("/2/tweets/search/recent",),
        disallowed_paths=(),
        login_required=False,
        javascript_required=False,
        fallback_behavior="official_api",
        last_reviewed=date(2026, 5, 11),
        notes="Use official X API recent search; do not browser-scrape X pages.",
    ),
}


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


def list_source_policies(
    registry: Mapping[str, SourcePolicy] = DEFAULT_SOURCE_POLICIES,
) -> tuple[SourcePolicy, ...]:
    """Return the configured source policies in deterministic source_id order."""

    return tuple(registry[source_id] for source_id in sorted(registry))


def get_source_policy(
    source_id: str,
    registry: Mapping[str, SourcePolicy] = DEFAULT_SOURCE_POLICIES,
) -> SourcePolicy:
    """Read one source policy from the registry."""

    return registry[source_id]


def evaluate_source_policy(
    source_id: str,
    source_url: str,
    registry: Mapping[str, SourcePolicy] = DEFAULT_SOURCE_POLICIES,
) -> SourcePolicyEvaluation:
    """Evaluate a URL against a named policy from the registry."""

    return get_source_policy(source_id, registry).evaluate_url(source_url)


def source_policy_warning(
    *,
    provider_name: str,
    evaluation: SourcePolicyEvaluation,
    occurred_at: datetime,
    raw_snapshot_id: str | None = None,
) -> ProviderWarning:
    """Convert a blocked source policy decision into a provider warning."""

    if evaluation.allowed:
        raise ValueError("allowed source policy evaluations do not produce blocked warnings")
    if evaluation.decision == "login_required":
        code = WarningCode.AUTH_FAILED
        provider_error_type = "scraping_login_required"
        message = f"{evaluation.policy.display_name} requires login; public scraping is skipped."
    else:
        code = WarningCode.UPSTREAM_UNAVAILABLE
        provider_error_type = "scraping_blocked_by_policy"
        message = (
            f"{evaluation.policy.display_name} is blocked by source policy: {evaluation.reason}."
        )
    return provider_warning(
        provider_name=provider_name,
        code=code,
        severity=WarningSeverity.ERROR,
        message=message,
        occurred_at=occurred_at,
        provider_error_type=provider_error_type,
        raw_snapshot_id=raw_snapshot_id,
        source_url=evaluation.source_url,
        metadata=evaluation.metadata,
    )


def source_policy_result[T](
    *,
    provider_name: str,
    request: ProviderRequest,
    fetched_at: datetime,
    evaluation: SourcePolicyEvaluation,
    credential_state: CredentialState = CredentialState.NOT_REQUIRED,
) -> ProviderResult[T]:
    """Build a contract-valid degraded result for a blocked source policy decision."""

    warning = source_policy_warning(
        provider_name=provider_name,
        evaluation=evaluation,
        occurred_at=fetched_at,
    )
    status = (
        ProviderStatus.UNAUTHORIZED
        if evaluation.decision == "login_required"
        else ProviderStatus.FAILED
    )
    health_credential_state = (
        CredentialState.INVALID if evaluation.decision == "login_required" else credential_state
    )
    return provider_result(
        provider_name=provider_name,
        status=status,
        request=request,
        fetched_at=fetched_at,
        credential_state=health_credential_state,
        warnings=(warning,),
    )


def scraping_drift_warning(
    *,
    provider_name: str,
    source_url: str,
    selector: str,
    occurred_at: datetime,
    raw_snapshot_id: str | None = None,
    required: bool = True,
    message: str | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ProviderWarning:
    """Create a structured warning for public HTML markup drift."""

    warning_message = message or (
        f"Required HTML selector or section was not found: {selector}."
        if required
        else f"Optional HTML selector or section was not found: {selector}."
    )
    warning_metadata: dict[str, JsonValue] = {
        "selector": selector,
        "required": required,
        "drift_type": "missing_selector",
    }
    warning_metadata.update(dict(metadata or {}))
    return provider_warning(
        provider_name=provider_name,
        code=WarningCode.SCRAPING_DRIFT,
        severity=WarningSeverity.ERROR if required else WarningSeverity.WARNING,
        message=warning_message,
        occurred_at=occurred_at,
        provider_error_type="scraping_drift",
        raw_snapshot_id=raw_snapshot_id,
        source_url=source_url,
        metadata=warning_metadata,
    )


def _normalize_policy_path(path: str) -> str:
    stripped = path.strip() or "/"
    return f"/{stripped.lstrip('/')}"


def _first_matching_policy_path(path: str, patterns: Iterable[str]) -> str | None:
    normalized_path = _normalize_policy_path(path)
    for raw_pattern in patterns:
        pattern = _normalize_policy_path(raw_pattern)
        if pattern.endswith("*") and normalized_path.startswith(pattern[:-1]):
            return raw_pattern
        if pattern == "/":
            return raw_pattern
        if pattern.endswith("/"):
            if normalized_path == pattern.rstrip("/") or normalized_path.startswith(pattern):
                return raw_pattern
        elif normalized_path == pattern or normalized_path.startswith(f"{pattern}/"):
            return raw_pattern
    return None


__all__ = [
    "DEFAULT_SOURCE_POLICIES",
    "DEFAULT_V1_DISCLAIMER_TEXT",
    "ComplianceError",
    "ComplianceIssue",
    "ComplianceSeverity",
    "RetrievalSourceClassification",
    "RobotsPolicyStatus",
    "SourceFallbackBehavior",
    "SourcePolicy",
    "SourcePolicyDecision",
    "SourcePolicyEvaluation",
    "build_v1_disclaimer",
    "classify_retrieval_source",
    "evaluate_source_policy",
    "get_source_policy",
    "list_source_policies",
    "metadata_is_guardrail_clean",
    "scraping_drift_warning",
    "source_policy_result",
    "source_policy_warning",
    "validate_disclaimer_guardrails",
    "validate_no_auto_trading_metadata",
    "validate_report_guardrails",
    "validate_trade_candidate_guardrails",
]
