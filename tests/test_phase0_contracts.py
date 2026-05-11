from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.cli import PHASE_0_NOT_IMPLEMENTED_EXIT_CODE, build_parser, main
from nlp_stock_prediction.contracts import (
    CredentialState,
    ProviderHealth,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    RetrievalMethod,
    SourceKind,
    SourceProvenance,
    TickerCandidate,
    TickerDiscoveryResult,
    TickerDiscoveryStatus,
)


def _fetched_at() -> datetime:
    return datetime(2026, 5, 11, 12, 0, tzinfo=UTC)


@pytest.mark.schema
def test_ticker_discovery_contract_requires_six_unique_tickers_for_valid_status() -> None:
    provenance = SourceProvenance(
        provider_name="fixture-reddit",
        source_kind=SourceKind.REDDIT_TICKER_CARD,
        retrieval_method=RetrievalMethod.FIXTURE,
        fetched_at=_fetched_at(),
    )
    tickers = ("TSLA", "NVDA", "AMD", "AAPL", "MU", "SPY")
    candidates = tuple(
        TickerCandidate(
            symbol=ticker,
            raw_identifier=f"ticker-container-{ticker.lower()}",
            first_seen_rank=rank,
            provenance=provenance,
        )
        for rank, ticker in enumerate(tickers)
    )

    result = TickerDiscoveryResult(
        run_date=date(2026, 5, 11),
        status=TickerDiscoveryStatus.VALID,
        candidates=candidates,
        tickers=tickers,
        raw_snapshot_id="raw-reddit-card-2026-05-11",
    )

    assert result.tickers == tickers


@pytest.mark.schema
def test_valid_ticker_discovery_rejects_duplicate_or_short_results() -> None:
    with pytest.raises(ValidationError):
        TickerDiscoveryResult(
            run_date=date(2026, 5, 11),
            status=TickerDiscoveryStatus.VALID,
            tickers=("TSLA", "NVDA", "AMD", "AAPL", "MU"),
        )

    with pytest.raises(ValidationError):
        TickerDiscoveryResult(
            run_date=date(2026, 5, 11),
            status=TickerDiscoveryStatus.VALID,
            tickers=("TSLA", "TSLA", "AMD", "AAPL", "MU", "SPY"),
        )


@pytest.mark.schema
def test_provider_result_envelope_serializes_health_and_request() -> None:
    request = ProviderRequest(
        request_id="provider-request-1",
        run_date=date(2026, 5, 11),
        tickers=("TSLA",),
        query="$TSLA lang:en",
    )
    health = ProviderHealth(
        provider_name="fixture-x",
        status=ProviderStatus.OK,
        checked_at=_fetched_at(),
        credential_state=CredentialState.NOT_REQUIRED,
    )
    result = ProviderResult[tuple[str, ...]](
        provider_name="fixture-x",
        status=ProviderStatus.OK,
        request=request,
        fetched_at=_fetched_at(),
        data=("ok",),
        health=health,
        raw_snapshot_id="raw-x-tsla",
    )

    dumped = result.model_dump(mode="json")

    assert dumped["provider_name"] == "fixture-x"
    assert dumped["health"]["status"] == "ok"
    assert dumped["request"]["query"] == "$TSLA lang:en"


@pytest.mark.unit
def test_cli_run_contract_parses_canonical_options_without_generating_report() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--date",
            "2026-05-11",
            "--output",
            "reports/",
            "--capital",
            "1000",
            "--risk-profile",
            "exploratory",
        ]
    )

    assert args.run_date == date(2026, 5, 11)
    assert (
        main(
            [
                "run",
                "--date",
                "2026-05-11",
                "--output",
                "reports/",
                "--capital",
                "1000",
            ]
        )
        == PHASE_0_NOT_IMPLEMENTED_EXIT_CODE
    )
