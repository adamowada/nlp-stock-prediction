from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import (
    EvidenceRequest,
    ProviderStatus,
    RetrievalMethod,
    TickerDiscoveryRequest,
    TickerDiscoveryStatus,
    WarningCode,
)
from nlp_stock_prediction.providers.reddit_scrape import (
    HtmlResponse,
    RedditPublicPageProvider,
    StaticHtmlTransport,
)

RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 16, 0, tzinfo=UTC)
SUBREDDIT_URL = "https://www.reddit.com/r/wallstreetbets/"
POST_URL = "https://www.reddit.com/r/wallstreetbets/comments/public001/daily_watch/"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "reddit"


class _CountingTransport:
    def __init__(self, html: str) -> None:
        self.html = html
        self.calls = 0

    def get_html(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
        max_bytes: int = 2_000_000,
    ) -> HtmlResponse:
        del url, headers, timeout, max_bytes
        self.calls += 1
        return HtmlResponse(html=self.html)


def _html(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _ticker_request(source_url: str = SUBREDDIT_URL) -> TickerDiscoveryRequest:
    return TickerDiscoveryRequest(
        request_id="reddit-public-page-card",
        run_date=RUN_DATE,
        source_url=source_url,
        query="r/wallstreetbets Devvit daily ticker card",
    )


def _discussion_request() -> EvidenceRequest:
    return EvidenceRequest(
        request_id="reddit-public-discussion",
        run_date=RUN_DATE,
        tickers=("TSLA", "MU", "AI", "ON"),
        query="TSLA OR MU OR AI OR ON",
        include_posts=True,
        include_comments=True,
    )


def _provider(
    pages: Mapping[str, str | HtmlResponse],
    *,
    discussion_urls: tuple[str, ...] = (),
) -> RedditPublicPageProvider:
    return RedditPublicPageProvider(
        transport=StaticHtmlTransport(pages),
        discussion_urls=discussion_urls,
        now=lambda: FETCHED_AT,
    )


@pytest.mark.contract
def test_public_page_provider_discovers_valid_six_ticker_card_from_fixture_html() -> None:
    provider = _provider({SUBREDDIT_URL: _html("public_page_devvit_card.html")})

    result = provider.discover_tickers(_ticker_request())

    assert result.status == ProviderStatus.OK
    assert result.data is not None
    assert result.data.status == TickerDiscoveryStatus.VALID
    assert result.data.tickers == ("TSLA", "NVDA", "AMD", "AI", "MU", "ON")
    assert result.raw_snapshot_id is not None
    assert result.raw_snapshot_id.startswith("raw-reddit-ticker-card-")
    assert result.data.candidates[0].provenance.provider_name == "reddit-public-page"
    assert result.data.candidates[0].provenance.retrieval_method == RetrievalMethod.PUBLIC_SCRAPE
    assert result.data.candidates[0].provenance.source_url == SUBREDDIT_URL
    assert result.warnings == ()


@pytest.mark.contract
def test_public_page_provider_reports_duplicate_shortage_as_partial_result() -> None:
    html = """
    <main data-snapshot-observed-at="2026-05-11T15:45:00Z">
      <div id="ticker-container-tsla">TSLA</div>
      <div id="ticker-container-nvda">NVDA</div>
      <div id="ticker-container-tsla">TSLA duplicate</div>
      <div id="ticker-container-amd">AMD</div>
      <div id="ticker-container-ai">AI</div>
      <div id="ticker-container-mu">MU</div>
    </main>
    """
    provider = _provider({SUBREDDIT_URL: html})

    result = provider.discover_tickers(_ticker_request())

    assert result.status == ProviderStatus.PARTIAL
    assert result.data is not None
    assert result.data.status == TickerDiscoveryStatus.TOO_FEW_UNIQUE
    assert result.data.tickers == ("TSLA", "NVDA", "AMD", "AI", "MU")
    assert {warning.metadata["validation"] for warning in result.warnings} == {
        "duplicate_tickers",
        "too_few_unique_tickers",
    }


@pytest.mark.contract
def test_public_page_provider_reports_malformed_ticker_markup_without_raising() -> None:
    html = """
    <main data-snapshot-observed-at="2026-05-11T15:45:00Z">
      <article>Daily thread markup drifted away from ticker containers.</article>
    </main>
    """
    provider = _provider({SUBREDDIT_URL: html})

    result = provider.discover_tickers(_ticker_request())

    assert result.status == ProviderStatus.PARTIAL
    assert result.data is not None
    assert result.data.status == TickerDiscoveryStatus.MALFORMED_SOURCE
    assert result.data.tickers == ()
    assert result.warnings[0].code == WarningCode.SCRAPING_DRIFT
    assert result.warnings[0].metadata["validation"] == "missing_ticker_containers"


@pytest.mark.contract
def test_public_page_provider_marks_stale_ticker_snapshot() -> None:
    provider = _provider({SUBREDDIT_URL: _html("public_page_stale_devvit_card.html")})

    result = provider.discover_tickers(_ticker_request())

    assert result.status == ProviderStatus.STALE
    assert result.data is not None
    assert result.data.status == TickerDiscoveryStatus.VALID
    assert any(warning.code == WarningCode.STALE_DATA for warning in result.warnings)
    assert result.data.candidates[0].provenance.freshness_status.value == "stale"
    assert result.data.candidates[0].provenance.freshness_seconds == 173700


@pytest.mark.contract
def test_public_page_policy_blocks_reddit_json_endpoints_before_fetch() -> None:
    transport = _CountingTransport(_html("public_page_devvit_card.html"))
    provider = RedditPublicPageProvider(transport=transport, now=lambda: FETCHED_AT)

    result = provider.discover_tickers(
        _ticker_request("https://www.reddit.com/r/wallstreetbets/.json")
    )

    assert result.status == ProviderStatus.FAILED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.UPSTREAM_UNAVAILABLE
    assert result.warnings[0].provider_error_type == "scraping_blocked_by_policy"
    assert result.warnings[0].metadata["policy_reason"] == "reddit_json_endpoint_disallowed"
    assert transport.calls == 0


@pytest.mark.contract
def test_public_page_discussion_html_normalizes_to_source_evidence() -> None:
    provider = _provider(
        {POST_URL: _html("public_post_discussion.html")},
        discussion_urls=(POST_URL,),
    )

    result = provider.fetch_discussion(_discussion_request())

    assert result.status == ProviderStatus.OK
    assert result.data is not None
    assert [record.evidence_id for record in result.data] == [
        "reddit-post-t3_public001",
        "reddit-comment-t1_public002",
    ]
    post = result.data[0]
    assert post.ticker == "MU"
    assert post.title == "Daily Watch: MU calls and TSLA momentum"
    assert post.text == (
        "Daily Watch: MU calls and TSLA momentum\n\n"
        "I like $AI after earnings; ON is only if volume confirms."
    )
    assert post.matched_tickers == ("MU", "TSLA", "AI", "ON")
    assert post.created_at == datetime(2026, 5, 11, 15, 30, tzinfo=UTC)
    assert post.score == 184
    assert post.metadata["num_comments"] == 47
    assert post.provenance.retrieval_method == RetrievalMethod.PUBLIC_SCRAPE
    assert post.provenance.raw_snapshot_id is not None
    assert post.provenance.raw_snapshot_id.startswith("raw-reddit-discussion-page-")
    assert "RetailTraderOne" not in json.dumps(post.model_dump(mode="json"))

    comment = result.data[1]
    assert comment.ticker == "AI"
    assert comment.matched_tickers == ("AI", "MU")
    assert comment.metadata["parent_id"] == "t3_public001"
    assert comment.metadata["link_id"] == "t3_public001"


@pytest.mark.contract
def test_public_page_discussion_without_matches_returns_empty_result() -> None:
    no_discussion_url = "https://www.reddit.com/r/wallstreetbets/comments/public002/no_discussion/"
    provider = _provider(
        {no_discussion_url: "<main><p>No ticker discussion is visible.</p></main>"},
        discussion_urls=(no_discussion_url,),
    )

    result = provider.fetch_discussion(_discussion_request())

    assert result.status == ProviderStatus.EMPTY
    assert result.data is None
    assert result.warnings[-1].code == WarningCode.NO_DATA
    assert result.warnings[-1].metadata["validation"] == "no_public_discussion_evidence"


@pytest.mark.contract
def test_public_page_provider_does_not_scrape_live_by_default() -> None:
    provider = RedditPublicPageProvider(now=lambda: FETCHED_AT)

    result = provider.discover_tickers(_ticker_request())

    assert result.status == ProviderStatus.UNCONFIGURED
    assert result.data is None
    assert result.warnings[0].metadata == {}
    assert result.warnings[0].provider_error_type == "live_scraping_disabled"
