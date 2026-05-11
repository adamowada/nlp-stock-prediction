from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    EvidenceRequest,
    FreshnessStatus,
    RetrievalMethod,
    SourceKind,
)
from nlp_stock_prediction.reddit.evidence import normalize_reddit_evidence

RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 16, 0, tzinfo=UTC)
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "reddit"


def _records() -> list[dict[str, object]]:
    return cast(
        list[dict[str, object]],
        json.loads((FIXTURE_DIR / "discussion_records.json").read_text(encoding="utf-8")),
    )


def _request() -> EvidenceRequest:
    return EvidenceRequest(
        request_id="reddit-discussion-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA", "MU", "AI", "ON"),
        query="TSLA OR MU OR AI OR ON",
        limit=25,
        include_posts=True,
        include_comments=True,
    )


@pytest.mark.unit
def test_reddit_posts_and_comments_normalize_to_source_evidence_with_provenance() -> None:
    evidence = normalize_reddit_evidence(
        _records(),
        request=_request(),
        fetched_at=FETCHED_AT,
        raw_snapshot_id="raw-reddit-discussion-normal",
        retrieval_method=RetrievalMethod.FIXTURE,
    )

    assert [record.evidence_id for record in evidence] == [
        "reddit-post-t3_lanea001",
        "reddit-comment-t1_lanea002",
    ]

    post = evidence[0]
    assert post.source_kind == SourceKind.REDDIT_POST
    assert post.ticker == "MU"
    assert post.title == "Daily Watch: MU calls and TSLA momentum"
    assert post.text == (
        "Daily Watch: MU calls and TSLA momentum\n\n"
        "I like $AI after earnings; ON is only if volume confirms."
    )
    assert post.matched_tickers == ("MU", "TSLA", "AI", "ON")
    assert [(span.text, span.start_char, span.end_char) for span in post.match_spans] == [
        ("MU", 13, 15),
        ("TSLA", 26, 30),
        ("$AI", 48, 51),
        ("ON", 68, 70),
    ]
    assert post.author_hash is not None
    assert post.author_hash.startswith("sha256:")
    assert "RetailTraderOne" not in json.dumps(post.model_dump(mode="json"))
    assert post.created_at == datetime(2026, 5, 11, 15, 30, tzinfo=UTC)
    assert post.score == 184
    assert post.permalink is not None
    assert post.permalink.endswith("/daily_watch/")
    assert post.provenance.provider_name == "reddit"
    assert post.provenance.source_kind == SourceKind.REDDIT_POST
    assert post.provenance.raw_identifier == "t3_lanea001"
    assert post.provenance.raw_snapshot_id == "raw-reddit-discussion-normal"
    assert post.provenance.freshness_status == FreshnessStatus.FRESH
    assert post.provenance.freshness_seconds == 1800
    assert post.provenance.provider_metadata["reddit_id"] == "t3_lanea001"
    assert post.metadata["num_comments"] == 47
    assert post.metadata["matched_ticker_count"] == 4

    comment = evidence[1]
    assert comment.source_kind == SourceKind.REDDIT_COMMENT
    assert comment.ticker == "AI"
    assert comment.title is None
    assert comment.text == "AI calls look expensive; I prefer $MU shares."
    assert comment.matched_tickers == ("AI", "MU")
    assert comment.metadata["parent_id"] == "t3_lanea001"
    assert comment.metadata["link_id"] == "t3_lanea001"
    assert comment.provenance.provider_metadata["reddit_kind"] == "comment"


@pytest.mark.unit
def test_reddit_evidence_filtering_respects_requested_source_types() -> None:
    request = EvidenceRequest(
        request_id="reddit-posts-only",
        run_date=RUN_DATE,
        tickers=("MU",),
        include_posts=True,
        include_comments=False,
    )

    evidence = normalize_reddit_evidence(
        _records(),
        request=request,
        fetched_at=FETCHED_AT,
        raw_snapshot_id="raw-reddit-discussion-posts-only",
        retrieval_method=RetrievalMethod.FIXTURE,
    )

    assert len(evidence) == 1
    assert evidence[0].source_kind == SourceKind.REDDIT_POST
    assert evidence[0].matched_tickers == ("MU",)


@pytest.mark.unit
def test_reddit_evidence_skips_records_without_high_precision_ticker_matches() -> None:
    request = EvidenceRequest(
        request_id="reddit-short-symbol-false-positive",
        run_date=RUN_DATE,
        tickers=("AI", "ON", "IT"),
        include_posts=False,
        include_comments=True,
    )

    evidence = normalize_reddit_evidence(
        [_records()[2]],
        request=request,
        fetched_at=FETCHED_AT,
        raw_snapshot_id="raw-reddit-discussion-false-positive",
        retrieval_method=RetrievalMethod.FIXTURE,
    )

    assert evidence == ()
