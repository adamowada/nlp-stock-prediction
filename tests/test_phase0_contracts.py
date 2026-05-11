from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.cli import build_parser, main
from nlp_stock_prediction.contracts import (
    CredentialState,
    FreshnessStatus,
    ProviderHealth,
    ProviderMetric,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    RetrievalMethod,
    SourceKind,
    SourceProvenance,
    TickerCandidate,
    TickerDiscoveryResult,
    TickerDiscoveryStatus,
    WarningCode,
    WarningSeverity,
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
        source_url="https://reddit.test/r/wallstreetbets",
        raw_identifier="ticker-card-2026-05-11",
        raw_snapshot_id="raw-reddit-card-2026-05-11",
        freshness_status=FreshnessStatus.FRESH,
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
def test_valid_ticker_discovery_requires_candidates_and_raw_snapshot() -> None:
    provenance = SourceProvenance(
        provider_name="fixture-reddit",
        source_kind=SourceKind.REDDIT_TICKER_CARD,
        retrieval_method=RetrievalMethod.FIXTURE,
        fetched_at=_fetched_at(),
        source_url="https://reddit.test/r/wallstreetbets",
        raw_identifier="ticker-card-2026-05-11",
        raw_snapshot_id="raw-reddit-card-2026-05-11",
        freshness_status=FreshnessStatus.FRESH,
    )
    candidates = tuple(
        TickerCandidate(
            symbol=ticker,
            raw_identifier=f"ticker-container-{ticker.lower()}",
            first_seen_rank=rank,
            provenance=provenance,
        )
        for rank, ticker in enumerate(("TSLA", "NVDA", "AMD", "AAPL", "MU", "SPY"))
    )

    with pytest.raises(ValidationError):
        TickerDiscoveryResult(
            run_date=date(2026, 5, 11),
            status=TickerDiscoveryStatus.VALID,
            tickers=("TSLA", "NVDA", "AMD", "AAPL", "MU", "SPY"),
        )

    with pytest.raises(ValidationError):
        TickerDiscoveryResult(
            run_date=date(2026, 5, 11),
            status=TickerDiscoveryStatus.VALID,
            candidates=candidates,
            tickers=("TSLA", "NVDA", "AMD", "AAPL", "MU", "QQQ"),
            raw_snapshot_id="raw-reddit-card-2026-05-11",
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


@pytest.mark.schema
def test_provider_result_rejects_inconsistent_failure_shapes() -> None:
    request = ProviderRequest(request_id="provider-request-1", run_date=date(2026, 5, 11))
    ok_health = ProviderHealth(
        provider_name="fixture-x",
        status=ProviderStatus.OK,
        checked_at=_fetched_at(),
        credential_state=CredentialState.NOT_REQUIRED,
    )
    failed_health = ProviderHealth(
        provider_name="fixture-x",
        status=ProviderStatus.FAILED,
        checked_at=_fetched_at(),
        credential_state=CredentialState.NOT_REQUIRED,
    )
    warning = ProviderWarning(
        code=WarningCode.UPSTREAM_UNAVAILABLE,
        severity=WarningSeverity.ERROR,
        message="fixture provider unavailable",
        provider_name="fixture-x",
        occurred_at=_fetched_at(),
    )

    with pytest.raises(ValidationError):
        ProviderResult[str](
            provider_name="fixture-x",
            status=ProviderStatus.OK,
            request=request,
            fetched_at=_fetched_at(),
            data=None,
            health=ok_health,
        )

    with pytest.raises(ValidationError):
        ProviderResult[str](
            provider_name="fixture-x",
            status=ProviderStatus.FAILED,
            request=request,
            fetched_at=_fetched_at(),
            data=None,
            health=ok_health,
            warnings=(warning,),
        )

    with pytest.raises(ValidationError):
        ProviderResult[str](
            provider_name="fixture-x",
            status=ProviderStatus.FAILED,
            request=request,
            fetched_at=_fetched_at(),
            data=None,
            health=failed_health,
        )


@pytest.mark.schema
def test_json_contracts_reject_non_serializable_metadata() -> None:
    with pytest.raises(ValidationError):
        ProviderMetric(name="bad-metadata", value=1, metadata={"object": object()})


@pytest.mark.unit
def test_cli_run_contract_parses_canonical_options_and_generates_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "reports"
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--date",
            "2026-05-11",
            "--output",
            str(output_dir),
            "--capital",
            "1000",
            "--risk-profile",
            "exploratory",
            "--offline",
        ]
    )

    assert args.run_date == date(2026, 5, 11)
    exit_code = main(
        [
            "run",
            "--date",
            "2026-05-11",
            "--output",
            str(output_dir),
            "--capital",
            "1000",
            "--offline",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "report.md" in captured.out
    assert captured.err == ""
    assert (output_dir / "2026-05-11" / "report.md").exists()
    assert (output_dir / "2026-05-11" / "report.json").exists()
