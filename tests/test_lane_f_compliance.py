from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from nlp_stock_prediction.compliance import (
    ComplianceError,
    build_v1_disclaimer,
    classify_retrieval_source,
    validate_disclaimer_guardrails,
    validate_no_auto_trading_metadata,
    validate_report_guardrails,
)
from nlp_stock_prediction.contracts import (
    CredentialState,
    DailyReport,
    DataFreshnessSummary,
    Disclaimer,
    FreshnessStatus,
    ProviderHealth,
    ProviderStatus,
    RetrievalMethod,
    RiskProfile,
    SourceKind,
    SourceProvenance,
    TickerCandidate,
    TickerDiscoveryResult,
    TickerDiscoveryStatus,
    TickerReportSection,
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
