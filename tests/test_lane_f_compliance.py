from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from nlp_stock_prediction.compliance import (
    ComplianceError,
    SourcePolicy,
    build_v1_disclaimer,
    classify_retrieval_source,
    evaluate_source_policy,
    list_source_policies,
    scraping_drift_warning,
    source_policy_result,
    source_policy_warning,
    validate_disclaimer_guardrails,
    validate_no_auto_trading_metadata,
    validate_report_guardrails,
)
from nlp_stock_prediction.contracts import (
    CredentialState,
    DailyReport,
    DataFreshnessSummary,
    Disclaimer,
    EvidenceRequest,
    FreshnessStatus,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    RetrievalMethod,
    RiskProfile,
    SourceKind,
    SourceProvenance,
    TickerCandidate,
    TickerDiscoveryResult,
    TickerDiscoveryStatus,
    TickerReportSection,
    WarningCode,
    WarningSeverity,
)

pytestmark = pytest.mark.unit

FETCHED_AT = datetime(2026, 5, 11, 12, 0, tzinfo=UTC)
TICKERS = ("TSLA", "NVDA", "AMD", "AAPL", "MU", "SPY")


def _provenance(method: RetrievalMethod = RetrievalMethod.OFFICIAL_API) -> SourceProvenance:
    return SourceProvenance(
        provider_name="fixture-provider",
        source_kind=SourceKind.MARKET_DATA,
        retrieval_method=method,
        fetched_at=FETCHED_AT,
        observed_at=FETCHED_AT,
        source_url="https://api.example.invalid/market/TSLA",
        raw_identifier="raw-tsla-market",
        raw_snapshot_id="raw-market-2026-05-11",
        query="TSLA",
        cache_key="market:TSLA:2026-05-11",
        freshness_status=FreshnessStatus.FRESH,
        provider_metadata={"fixture": True},
    )


def _ticker_discovery() -> TickerDiscoveryResult:
    provenance = SourceProvenance(
        provider_name="fixture-reddit",
        source_kind=SourceKind.REDDIT_TICKER_CARD,
        retrieval_method=RetrievalMethod.FIXTURE,
        fetched_at=FETCHED_AT,
        source_url="https://reddit.example.invalid/wsb",
        raw_identifier="devvit-card",
        raw_snapshot_id="raw-card-2026-05-11",
        query="wallstreetbets daily card",
        cache_key="reddit-card:2026-05-11",
        freshness_status=FreshnessStatus.FRESH,
        provider_metadata={"fixture": True},
    )
    candidates = tuple(
        TickerCandidate(
            symbol=ticker,
            raw_identifier=f"ticker-container-{ticker.lower()}",
            raw_text=ticker,
            first_seen_rank=index,
            source_url="https://reddit.example.invalid/wsb",
            provenance=provenance,
        )
        for index, ticker in enumerate(TICKERS)
    )
    return TickerDiscoveryResult(
        run_date=date(2026, 5, 11),
        status=TickerDiscoveryStatus.VALID,
        candidates=candidates,
        tickers=TICKERS,
        raw_snapshot_id="raw-card-2026-05-11",
    )


def _report(disclaimer: Disclaimer | None = None) -> DailyReport:
    return DailyReport(
        schema_version="report.v1",
        run_id="lane-f-report",
        report_date=date(2026, 5, 11),
        generated_at=FETCHED_AT,
        timezone="America/Los_Angeles",
        risk_profile=RiskProfile.EXPLORATORY,
        disclaimer=disclaimer or build_v1_disclaimer(version="2026-05-11"),
        ticker_discovery=_ticker_discovery(),
        data_freshness=DataFreshnessSummary(
            as_of=FETCHED_AT,
            summary="Fixture data is fresh.",
        ),
        provider_health=(
            ProviderHealth(
                provider_name="fixture-provider",
                status=ProviderStatus.OK,
                checked_at=FETCHED_AT,
                credential_state=CredentialState.NOT_REQUIRED,
                last_success_at=FETCHED_AT,
            ),
        ),
        ticker_sections=tuple(TickerReportSection(ticker=ticker) for ticker in TICKERS),
        no_trade_summary="No qualified setup passed the fixture guardrails.",
    )


def test_build_v1_disclaimer_contains_required_text_and_flags() -> None:
    disclaimer = build_v1_disclaimer(version="2026-05-11")

    assert disclaimer.educational_only is True
    assert disclaimer.not_financial_advice is True
    assert disclaimer.no_auto_trading is True
    assert "educational" in disclaimer.text.lower()
    assert "not financial advice" in disclaimer.text.lower()
    assert "no automatic trading" in disclaimer.text.lower()
    assert validate_disclaimer_guardrails(disclaimer) == ()


def test_disclaimer_text_guardrails_report_missing_required_language() -> None:
    disclaimer = Disclaimer(
        disclaimer_id="weak-disclaimer",
        version="2026-05-11",
        text="Research notes only.",
    )

    issues = validate_disclaimer_guardrails(disclaimer)

    assert {issue.code for issue in issues} == {
        "missing_educational_only_text",
        "missing_not_financial_advice_text",
        "missing_no_auto_trading_text",
    }


def test_report_guardrails_raise_for_non_compliant_disclaimer_text() -> None:
    report = _report(
        Disclaimer(
            disclaimer_id="weak-disclaimer",
            version="2026-05-11",
            text="Research notes only.",
        )
    )

    with pytest.raises(ComplianceError) as exc_info:
        validate_report_guardrails(report, raise_on_error=True)

    assert "missing_not_financial_advice_text" in str(exc_info.value)


def test_no_auto_trading_metadata_guardrail_rejects_execution_payloads() -> None:
    issues = validate_no_auto_trading_metadata(
        {
            "confidence_inputs": {"evidence": 0.7},
            "broker_order_id": "abc123",
            "auto_trade": True,
            "execution": {"venue": "fixture-broker"},
        },
        subject_id="candidate-tsla",
    )

    assert {issue.code for issue in issues} == {
        "forbidden_brokerage_metadata",
        "forbidden_auto_trade_metadata",
        "forbidden_execution_metadata",
    }
    assert all(issue.subject_id == "candidate-tsla" for issue in issues)


def test_retrieval_source_classification_distinguishes_api_from_scraping_fallback() -> None:
    official = classify_retrieval_source(_provenance(RetrievalMethod.OFFICIAL_API))
    scraping = classify_retrieval_source(_provenance(RetrievalMethod.PUBLIC_SCRAPE))
    fixture = classify_retrieval_source(_provenance(RetrievalMethod.FIXTURE))

    assert official.is_official_api is True
    assert official.is_public_scraping_fallback is False
    assert official.requires_scraping_drift_monitor is False
    assert scraping.is_official_api is False
    assert scraping.is_public_scraping_fallback is True
    assert scraping.requires_scraping_drift_monitor is True
    assert fixture.label == "fixture"


def test_compliant_report_has_no_guardrail_issues() -> None:
    assert validate_report_guardrails(_report()) == ()


def test_source_policy_registry_contains_scraping_and_api_entrypoints() -> None:
    policies = list_source_policies()
    source_ids = [policy.source_id for policy in policies]

    assert source_ids == sorted(source_ids)
    assert {policy.source_id for policy in policies} >= {
        "reddit_wsb",
        "apnews_financial_markets",
        "candlecharts_live_charts",
        "x_recent_search",
    }
    x_policy = next(policy for policy in policies if policy.source_id == "x_recent_search")
    assert x_policy.fallback_behavior == "official_api"
    assert x_policy.robots_status == "not_applicable"


def test_source_policy_allows_public_allowlisted_path() -> None:
    evaluation = evaluate_source_policy(
        "apnews_financial_markets",
        "https://apnews.com/hub/financial-markets",
    )

    assert evaluation.allowed is True
    assert evaluation.decision == "allowed"
    assert evaluation.matched_path == "/hub/financial-markets"
    assert evaluation.metadata["source_id"] == "apnews_financial_markets"
    assert evaluation.metadata["fallback_behavior"] == "degraded_result"


def test_source_policy_blocks_disallowed_path_with_structured_warning() -> None:
    evaluation = evaluate_source_policy(
        "reddit_wsb",
        "https://www.reddit.com/r/wallstreetbets/search/?q=TSLA",
    )

    warning = source_policy_warning(
        provider_name="reddit-public",
        evaluation=evaluation,
        occurred_at=FETCHED_AT,
    )

    assert evaluation.allowed is False
    assert evaluation.reason == "path_disallowed_by_source_policy"
    assert warning.code == WarningCode.UPSTREAM_UNAVAILABLE
    assert warning.severity == WarningSeverity.ERROR
    assert warning.provider_error_type == "scraping_blocked_by_policy"
    assert warning.source_url == "https://www.reddit.com/r/wallstreetbets/search/?q=TSLA"
    assert warning.metadata["source_id"] == "reddit_wsb"
    assert warning.metadata["matched_path"] == "/r/wallstreetbets/search/"


def test_source_policy_login_required_becomes_unauthorized_result() -> None:
    policy = SourcePolicy(
        source_id="login_source",
        provider_name="login-provider",
        display_name="Login Source",
        base_url="https://example.invalid",
        robots_txt_url="https://example.invalid/robots.txt",
        robots_status="allowed",
        allowed_paths=("/members/",),
        login_required=True,
        fallback_behavior="degraded_result",
        last_reviewed=date(2026, 5, 11),
    )
    evaluation = policy.evaluate_url("https://example.invalid/members/research")
    request = EvidenceRequest(
        request_id="login-required-source-2026-05-11",
        run_date=date(2026, 5, 11),
        tickers=("TSLA",),
    )

    result: ProviderResult[tuple[str, ...]] = source_policy_result(
        provider_name="login-provider",
        request=request,
        fetched_at=FETCHED_AT,
        evaluation=evaluation,
    )

    assert result.status == ProviderStatus.UNAUTHORIZED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.AUTH_FAILED
    assert result.warnings[0].provider_error_type == "scraping_login_required"
    assert result.health.credential_state == CredentialState.INVALID


def test_scraping_drift_warning_preserves_selector_context() -> None:
    warning = scraping_drift_warning(
        provider_name="apnews-public",
        source_url="https://apnews.com/hub/financial-markets",
        selector="article[data-key]",
        occurred_at=FETCHED_AT,
        raw_snapshot_id="raw-apnews-fixture",
        metadata={"section": "financial-markets"},
    )

    assert warning.code == WarningCode.SCRAPING_DRIFT
    assert warning.severity == WarningSeverity.ERROR
    assert warning.provider_error_type == "scraping_drift"
    assert warning.raw_snapshot_id == "raw-apnews-fixture"
    assert warning.metadata["selector"] == "article[data-key]"
    assert warning.metadata["required"] is True
    assert warning.metadata["section"] == "financial-markets"
