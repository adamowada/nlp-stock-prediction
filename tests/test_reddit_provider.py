from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    EvidenceRequest,
    ProviderStatus,
    TickerDiscoveryRequest,
    TickerDiscoveryStatus,
)
from nlp_stock_prediction.reddit.provider import FixtureRedditProvider

RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 16, 0, tzinfo=UTC)
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "reddit"


def _html() -> str:
    return (FIXTURE_DIR / "devvit_card_normal.html").read_text(encoding="utf-8")


def _records() -> list[dict[str, object]]:
    return cast(
        list[dict[str, object]],
        json.loads((FIXTURE_DIR / "discussion_records.json").read_text(encoding="utf-8")),
    )


@pytest.mark.integration
def test_fixture_reddit_provider_satisfies_discovery_and_discussion_contracts() -> None:
    provider = FixtureRedditProvider(
        ticker_card_html=_html(),
        discussion_records=_records(),
        fetched_at=FETCHED_AT,
        raw_ticker_snapshot_id="raw-reddit-devvit-card-normal",
        raw_discussion_snapshot_id="raw-reddit-discussion-normal",
    )

    discovery = provider.discover_tickers(
        TickerDiscoveryRequest(
            request_id="discover-reddit-2026-05-11",
            run_date=RUN_DATE,
            source_url="https://old.reddit.com/r/wallstreetbets/",
        )
    )
    discussion = provider.fetch_discussion(
        EvidenceRequest(
            request_id="reddit-discussion-2026-05-11",
            run_date=RUN_DATE,
            tickers=("TSLA", "MU", "AI", "ON"),
            include_posts=True,
            include_comments=True,
        )
    )

    assert discovery.provider_name == "reddit"
    assert discovery.status == ProviderStatus.OK
    assert discovery.health.ok is True
    assert discovery.data is not None
    assert discovery.data.status == TickerDiscoveryStatus.VALID
    assert discovery.data.tickers == ("TSLA", "NVDA", "AMD", "AI", "MU", "ON")
    assert discovery.raw_snapshot_id == "raw-reddit-devvit-card-normal"

    assert discussion.status == ProviderStatus.OK
    assert discussion.health.ok is True
    assert discussion.data is not None
    assert [record.evidence_id for record in discussion.data] == [
        "reddit-post-t3_lanea001",
        "reddit-comment-t1_lanea002",
    ]
    assert discussion.raw_snapshot_id == "raw-reddit-discussion-normal"


@pytest.mark.integration
def test_fixture_reddit_provider_surfaces_invalid_discovery_as_partial_result() -> None:
    provider = FixtureRedditProvider(
        ticker_card_html="<main>markup drift</main>",
        discussion_records=[],
        fetched_at=FETCHED_AT,
        raw_ticker_snapshot_id="raw-reddit-devvit-card-malformed",
        raw_discussion_snapshot_id="raw-reddit-discussion-empty",
    )

    discovery = provider.discover_tickers(
        TickerDiscoveryRequest(
            request_id="discover-reddit-malformed",
            run_date=RUN_DATE,
            source_url="https://old.reddit.com/r/wallstreetbets/",
        )
    )

    assert discovery.status == ProviderStatus.PARTIAL
    assert discovery.health.status == ProviderStatus.PARTIAL
    assert discovery.data is not None
    assert discovery.data.status == TickerDiscoveryStatus.MALFORMED_SOURCE
    assert discovery.warnings == discovery.data.warnings
