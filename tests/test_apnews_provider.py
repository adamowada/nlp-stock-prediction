from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import (
    CredentialState,
    EvidenceRequest,
    FreshnessStatus,
    ProviderStatus,
    RetrievalMethod,
    SourceKind,
    WarningCode,
)
from nlp_stock_prediction.providers.apnews import APNewsProvider
from nlp_stock_prediction.providers.scraping import HtmlResponse

RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "raw" / "apnews"


def _html(name: str) -> str:
    return FIXTURE_ROOT.joinpath(name).read_text(encoding="utf-8")


@dataclass
class _FakeHtmlTransport:
    responses: dict[str, str]
    calls: list[str] = field(default_factory=list)
    headers: list[Mapping[str, str] | None] = field(default_factory=list)

    def get_html(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
        max_bytes: int = 2_000_000,
    ) -> HtmlResponse:
        del timeout, max_bytes
        self.calls.append(url)
        self.headers.append(headers)
        for url_fragment, body in self.responses.items():
            if url_fragment in url:
                return HtmlResponse(html=body)
        raise AssertionError(f"Unexpected URL: {url}")


@pytest.mark.contract
def test_apnews_provider_extracts_hub_article_evidence_with_provenance() -> None:
    transport = _FakeHtmlTransport(
        {
            "hub/financial-markets": _html("hub_financial_markets.html"),
            "tesla-nvidia-markets": _html("article_tsla_nvidia.html"),
            "oil-prices-economy": _html("article_unrelated.html"),
        }
    )
    provider = APNewsProvider(transport=transport, now=lambda: FETCHED_AT)
    request = EvidenceRequest(
        request_id="apnews-tsla-nvda-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA", "NVDA"),
        limit=1,
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.OK
    assert result.health.credential_state == CredentialState.NOT_REQUIRED
    assert result.data is not None
    assert len(result.data) == 1
    article = result.data[0]
    assert article.source_kind == SourceKind.NEWS_ARTICLE
    assert article.ticker == "TSLA"
    assert article.title == "Stocks advance as Tesla and Nvidia lead markets"
    assert article.text.startswith("A broad rally lifted megacap technology shares")
    assert article.permalink == "https://apnews.com/article/tesla-nvidia-markets-2026-05-11"
    assert article.created_at == datetime(2026, 5, 11, 16, 30, tzinfo=UTC)
    assert article.matched_tickers == ("TSLA", "NVDA")
    assert article.provenance.provider_name == "ap-news"
    assert article.provenance.retrieval_method == RetrievalMethod.PUBLIC_SCRAPE
    assert article.provenance.freshness_status == FreshnessStatus.FRESH
    assert article.provenance.raw_snapshot_id == result.data[0].provenance.raw_snapshot_id
    assert article.provenance.provider_metadata["source_label"] == "AP Business"
    assert article.provenance.provider_metadata["author_label"] == "Alex Veiga"
    assert article.metadata["source_label"] == "AP Business"
    assert transport.calls == [
        "https://apnews.com/hub/financial-markets",
        "https://apnews.com/article/tesla-nvidia-markets-2026-05-11",
    ]
    assert transport.headers[0] is not None
    assert "User-Agent" in transport.headers[0]


@pytest.mark.contract
def test_apnews_provider_marks_articles_with_missing_timestamp_partial() -> None:
    transport = _FakeHtmlTransport(
        {
            "hub/financial-markets": _html("hub_missing_timestamp.html"),
            "nvidia-ai-markets": _html("article_missing_timestamp.html"),
        }
    )
    provider = APNewsProvider(transport=transport, now=lambda: FETCHED_AT)
    request = EvidenceRequest(
        request_id="apnews-missing-timestamp-2026-05-11",
        run_date=RUN_DATE,
        tickers=("NVDA",),
        limit=1,
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.PARTIAL
    assert result.data is not None
    assert result.data[0].created_at is None
    assert result.data[0].provenance.freshness_status == FreshnessStatus.MISSING
    assert result.warnings[0].code == WarningCode.PARTIAL_DATA
    assert (
        result.warnings[0].source_url == "https://apnews.com/article/nvidia-ai-markets-2026-05-11"
    )


@pytest.mark.contract
def test_apnews_provider_filters_unrelated_articles_to_no_data() -> None:
    transport = _FakeHtmlTransport(
        {
            "hub/financial-markets": _html("hub_unrelated.html"),
            "oil-prices-economy": _html("article_unrelated.html"),
        }
    )
    provider = APNewsProvider(transport=transport, now=lambda: FETCHED_AT)
    request = EvidenceRequest(
        request_id="apnews-unrelated-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        limit=3,
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.EMPTY
    assert result.data is None
    assert result.warnings[0].code == WarningCode.NO_DATA
    assert "no ticker-matched" in result.warnings[0].message
    assert len(transport.calls) == 2


@pytest.mark.contract
def test_apnews_provider_marks_stale_news_articles() -> None:
    transport = _FakeHtmlTransport(
        {
            "hub/financial-markets": _html("hub_stale.html"),
            "tesla-stale-markets": _html("article_stale.html"),
        }
    )
    provider = APNewsProvider(
        transport=transport,
        now=lambda: FETCHED_AT,
        stale_after_seconds=3 * 24 * 60 * 60,
    )
    request = EvidenceRequest(
        request_id="apnews-stale-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        limit=1,
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.STALE
    assert result.data is not None
    assert result.data[0].provenance.freshness_status == FreshnessStatus.STALE
    assert result.warnings[0].code == WarningCode.STALE_DATA
    assert result.warnings[0].metadata["published_at"] == "2026-04-30T14:00:00+00:00"


@pytest.mark.contract
def test_apnews_provider_reports_drift_when_hub_has_no_article_links() -> None:
    transport = _FakeHtmlTransport({"hub/financial-markets": _html("hub_drift.html")})
    provider = APNewsProvider(transport=transport, now=lambda: FETCHED_AT)
    request = EvidenceRequest(
        request_id="apnews-drift-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        limit=3,
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.MALFORMED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.SCRAPING_DRIFT
    assert result.warnings[0].metadata["validation"] == "missing_ap_article_links"
    assert transport.calls == ["https://apnews.com/hub/financial-markets"]


@pytest.mark.contract
def test_apnews_provider_reports_malformed_when_hub_html_has_no_readable_text() -> None:
    transport = _FakeHtmlTransport({"hub/financial-markets": "   "})
    provider = APNewsProvider(transport=transport, now=lambda: FETCHED_AT)
    request = EvidenceRequest(
        request_id="apnews-empty-hub-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        limit=3,
    )

    result = provider.fetch_articles(request)

    assert result.status == ProviderStatus.MALFORMED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.SCRAPING_DRIFT
    assert result.warnings[0].metadata["validation"] == "malformed_ap_hub"
    assert transport.calls == ["https://apnews.com/hub/financial-markets"]
