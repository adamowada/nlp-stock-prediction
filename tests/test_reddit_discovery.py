from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import (
    FreshnessStatus,
    RetrievalMethod,
    TickerDiscoveryRequest,
    TickerDiscoveryStatus,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.reddit.discovery import discover_tickers_from_devvit_html

RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 16, 0, tzinfo=UTC)
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "reddit"


def _request() -> TickerDiscoveryRequest:
    return TickerDiscoveryRequest(
        request_id="reddit-devvit-card-2026-05-11",
        run_date=RUN_DATE,
        source_url="https://old.reddit.com/r/wallstreetbets/",
        query="r/wallstreetbets Devvit daily ticker card",
    )


@pytest.mark.unit
def test_devvit_card_discovery_preserves_raw_order_identifiers_and_text() -> None:
    html = (FIXTURE_DIR / "devvit_card_normal.html").read_text(encoding="utf-8")

    result = discover_tickers_from_devvit_html(
        html,
        request=_request(),
        fetched_at=FETCHED_AT,
        raw_snapshot_id="raw-reddit-devvit-card-normal",
        retrieval_method=RetrievalMethod.FIXTURE,
    )

    assert result.status == TickerDiscoveryStatus.VALID
    assert result.tickers == ("TSLA", "NVDA", "AMD", "AI", "MU", "ON")
    assert [candidate.raw_identifier for candidate in result.candidates] == [
        "ticker-container-tsla",
        "ticker-container-nvda",
        "ticker-container-amd",
        "ticker-container-ai",
        "ticker-container-mu",
        "ticker-container-on",
    ]
    assert [candidate.raw_text for candidate in result.candidates] == [
        "TSLA",
        "NVDA",
        "AMD",
        "AI",
        "MU",
        "ON",
    ]
    assert [candidate.first_seen_rank for candidate in result.candidates] == list(range(6))
    assert result.raw_snapshot_id == "raw-reddit-devvit-card-normal"
    assert all(
        candidate.provenance.freshness_status == FreshnessStatus.FRESH
        for candidate in result.candidates
    )
    assert result.warnings == ()


@pytest.mark.unit
def test_duplicate_candidates_emit_warning_and_keep_first_seen_unique_order() -> None:
    html = """
    <div id="ticker-container-tsla">TSLA</div>
    <div id="ticker-container-nvda">NVDA</div>
    <div id="ticker-container-tsla">Tesla again</div>
    <div id="ticker-container-amd">AMD</div>
    <div id="ticker-container-ai">AI</div>
    <div id="ticker-container-mu">MU</div>
    <div id="ticker-container-on">ON</div>
    """

    result = discover_tickers_from_devvit_html(
        html,
        request=_request(),
        fetched_at=FETCHED_AT,
        raw_snapshot_id="raw-reddit-devvit-card-duplicates",
        retrieval_method=RetrievalMethod.FIXTURE,
    )

    assert result.status == TickerDiscoveryStatus.VALID
    assert result.tickers == ("TSLA", "NVDA", "AMD", "AI", "MU", "ON")
    assert [candidate.symbol for candidate in result.candidates] == [
        "TSLA",
        "NVDA",
        "TSLA",
        "AMD",
        "AI",
        "MU",
        "ON",
    ]
    assert len(result.warnings) == 1
    assert result.warnings[0].severity == WarningSeverity.WARNING
    assert result.warnings[0].metadata["duplicate_symbols"] == ["TSLA"]


@pytest.mark.unit
def test_duplicate_only_shortage_is_invalid_with_errors() -> None:
    html = """
    <div id="ticker-container-tsla">TSLA</div>
    <div id="ticker-container-nvda">NVDA</div>
    <div id="ticker-container-tsla">TSLA duplicate</div>
    <div id="ticker-container-amd">AMD</div>
    <div id="ticker-container-ai">AI</div>
    <div id="ticker-container-mu">MU</div>
    """

    result = discover_tickers_from_devvit_html(
        html,
        request=_request(),
        fetched_at=FETCHED_AT,
        raw_snapshot_id="raw-reddit-devvit-card-shortage",
        retrieval_method=RetrievalMethod.FIXTURE,
    )

    assert result.status == TickerDiscoveryStatus.TOO_FEW_UNIQUE
    assert result.tickers == ("TSLA", "NVDA", "AMD", "AI", "MU")
    assert any(warning.severity == WarningSeverity.ERROR for warning in result.warnings)
    assert {warning.metadata["validation"] for warning in result.warnings} == {
        "duplicate_tickers",
        "too_few_unique_tickers",
    }


@pytest.mark.unit
def test_extra_unique_candidates_are_invalid_and_preserved_for_audit() -> None:
    html = """
    <div id="ticker-container-tsla">TSLA</div>
    <div id="ticker-container-nvda">NVDA</div>
    <div id="ticker-container-amd">AMD</div>
    <div id="ticker-container-ai">AI</div>
    <div id="ticker-container-mu">MU</div>
    <div id="ticker-container-on">ON</div>
    <div id="ticker-container-it">IT</div>
    """

    result = discover_tickers_from_devvit_html(
        html,
        request=_request(),
        fetched_at=FETCHED_AT,
        raw_snapshot_id="raw-reddit-devvit-card-extra",
        retrieval_method=RetrievalMethod.FIXTURE,
    )

    assert result.status == TickerDiscoveryStatus.TOO_MANY_UNIQUE
    assert result.tickers == ("TSLA", "NVDA", "AMD", "AI", "MU", "ON", "IT")
    assert result.warnings[0].severity == WarningSeverity.ERROR
    assert result.warnings[0].metadata["unique_ticker_count"] == 7


@pytest.mark.unit
def test_invalid_ticker_container_identifier_returns_malformed_warning() -> None:
    html = """
    <div id="ticker-container-$bad">$BAD</div>
    <div id="ticker-container-tsla">TSLA</div>
    <div id="ticker-container-nvda">NVDA</div>
    <div id="ticker-container-amd">AMD</div>
    <div id="ticker-container-ai">AI</div>
    <div id="ticker-container-mu">MU</div>
    <div id="ticker-container-on">ON</div>
    """

    result = discover_tickers_from_devvit_html(
        html,
        request=_request(),
        fetched_at=FETCHED_AT,
        raw_snapshot_id="raw-reddit-devvit-card-invalid-identifier",
        retrieval_method=RetrievalMethod.FIXTURE,
    )

    assert result.status == TickerDiscoveryStatus.MALFORMED_SOURCE
    assert result.tickers == ("TSLA", "NVDA", "AMD", "AI", "MU", "ON")
    assert [candidate.symbol for candidate in result.candidates] == [
        "TSLA",
        "NVDA",
        "AMD",
        "AI",
        "MU",
        "ON",
    ]
    assert result.warnings[0].code == WarningCode.SCHEMA_MISMATCH
    assert result.warnings[0].severity == WarningSeverity.ERROR
    assert result.warnings[0].metadata["validation"] == ("invalid_ticker_container_identifiers")
    assert result.warnings[0].metadata["invalid_identifiers"] == ["ticker-container-$bad"]


@pytest.mark.unit
def test_malformed_card_without_ticker_containers_returns_validation_error() -> None:
    result = discover_tickers_from_devvit_html(
        "<article><p>Daily thread markup drifted.</p></article>",
        request=_request(),
        fetched_at=FETCHED_AT,
        raw_snapshot_id="raw-reddit-devvit-card-malformed",
        retrieval_method=RetrievalMethod.FIXTURE,
    )

    assert result.status == TickerDiscoveryStatus.MALFORMED_SOURCE
    assert result.tickers == ()
    assert result.candidates == ()
    assert len(result.warnings) == 1
    assert result.warnings[0].severity == WarningSeverity.ERROR
    assert result.warnings[0].metadata["validation"] == "missing_ticker_containers"
