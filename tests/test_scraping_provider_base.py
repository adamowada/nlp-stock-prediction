from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request as UrlRequest

import pytest

from nlp_stock_prediction.contracts import WarningCode, WarningSeverity
from nlp_stock_prediction.providers import scraping
from nlp_stock_prediction.providers._base import MalformedProviderResponse, ProviderTransportError
from nlp_stock_prediction.providers.scraping import (
    DEFAULT_HTML_MAX_BYTES,
    DEFAULT_SCRAPE_USER_AGENT,
    HtmlCache,
    HtmlResponse,
    HtmlTextRequirement,
    UrllibHtmlTransport,
    build_scraping_headers,
    configured_scrape_min_delay_seconds,
    configured_scrape_user_agent,
    drift_warnings_for_missing_text,
    fetch_html,
    parse_html_document,
)

pytestmark = pytest.mark.unit

RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 12, 0, tzinfo=UTC)
HTML = """<!doctype html>
<html>
  <head>
    <title>Financial Markets</title>
    <link rel="canonical" href="/hub/financial-markets">
    <script>window.secretTicker = "IGNORE";</script>
  </head>
  <body>
    <main>
      <h1>Market snapshot</h1>
      <article><a href="/article/tesla-update">Tesla shares rise</a></article>
    </main>
  </body>
</html>
"""


@dataclass
class _FakeHtmlTransport:
    response: HtmlResponse
    calls: list[str] = field(default_factory=list)
    headers: list[Mapping[str, str] | None] = field(default_factory=list)
    max_bytes: list[int] = field(default_factory=list)

    def get_html(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
        max_bytes: int = DEFAULT_HTML_MAX_BYTES,
    ) -> HtmlResponse:
        del timeout
        self.calls.append(url)
        self.headers.append(headers)
        self.max_bytes.append(max_bytes)
        return self.response


def test_scraping_configuration_uses_polite_defaults_and_env_overrides() -> None:
    env = {
        "NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT": "fixture-agent/1.0",
        "NLP_STOCK_PREDICTION_SCRAPE_MIN_DELAY_SECONDS": "2.5",
    }

    assert configured_scrape_user_agent(env) == "fixture-agent/1.0"
    assert configured_scrape_min_delay_seconds(env) == 2.5
    assert configured_scrape_user_agent({}) == DEFAULT_SCRAPE_USER_AGENT
    assert (
        configured_scrape_min_delay_seconds({"NLP_STOCK_PREDICTION_SCRAPE_MIN_DELAY_SECONDS": "-1"})
        == 1.0
    )
    assert build_scraping_headers(
        user_agent="custom-agent",
        extra_headers={"Accept": "text/html"},
    ) == {"User-Agent": "custom-agent", "Accept": "text/html"}


def test_fetch_html_reuses_cache_without_second_transport_call(tmp_path: Path) -> None:
    transport = _FakeHtmlTransport(
        HtmlResponse(
            html=HTML,
            status_code=200,
            final_url="https://apnews.com/hub/financial-markets",
        )
    )
    cache = HtmlCache(tmp_path)

    first = fetch_html(
        transport=transport,
        url="https://apnews.com/hub/financial-markets",
        run_date=RUN_DATE,
        ticker=None,
        source="apnews-hub",
        cache_key="apnews:hub:2026-05-11",
        fetched_at=FETCHED_AT,
        cache=cache,
        user_agent="fixture-agent/1.0",
    )
    second = fetch_html(
        transport=transport,
        url="https://apnews.com/hub/financial-markets",
        run_date=RUN_DATE,
        ticker=None,
        source="apnews-hub",
        cache_key="apnews:hub:2026-05-11",
        fetched_at=FETCHED_AT,
        cache=cache,
        user_agent="fixture-agent/1.0",
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.raw_snapshot_id == second.raw_snapshot_id
    assert first.content_sha256 == second.content_sha256
    assert second.html == HTML
    assert len(transport.calls) == 1
    assert transport.headers[0] == {"User-Agent": "fixture-agent/1.0"}
    assert tmp_path.joinpath("2026-05-11", "all", "apnews-hub").exists()


def test_html_cache_treats_corrupt_cache_entry_as_miss(tmp_path: Path) -> None:
    cache = HtmlCache(tmp_path)
    path = cache.path_for(RUN_DATE, None, "apnews-hub", "corrupt")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe")

    assert (
        cache.load_html(
            run_date=RUN_DATE,
            ticker=None,
            source="apnews-hub",
            cache_key="corrupt",
        )
        is None
    )


def test_urllib_html_transport_classifies_wrapped_socket_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout_urlopen(*_args: object, **_kwargs: object) -> object:
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr("nlp_stock_prediction.providers.scraping.urlopen", timeout_urlopen)

    with pytest.raises(ProviderTransportError) as exc:
        UrllibHtmlTransport().get_html("https://example.com/")

    assert exc.value.error_type == "timeout"


def test_urllib_html_transport_percent_encodes_unicode_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}

    class _Response:
        status = 200
        url = "https://old.reddit.com/r/wallstreetbets/comments/test/title_%F0%9F%9A%80/"

        def __init__(self) -> None:
            self.headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _max_bytes: int) -> bytes:
            return b"<html><body>ok</body></html>"

    def fake_urlopen(request: UrlRequest, *, timeout: float) -> _Response:
        del timeout
        seen["url"] = request.full_url
        return _Response()

    monkeypatch.setattr(scraping, "urlopen", fake_urlopen)

    response = UrllibHtmlTransport().get_html(
        "https://old.reddit.com/r/wallstreetbets/comments/test/title_🚀/"
    )

    assert response.html == "<html><body>ok</body></html>"
    assert seen["url"].endswith("/title_%F0%9F%9A%80/")


def test_parse_html_document_extracts_visible_text_canonical_url_and_links() -> None:
    document = parse_html_document(
        HTML,
        source_url="https://apnews.com/hub/financial-markets",
    )

    assert document.title == "Financial Markets"
    assert document.canonical_url == "https://apnews.com/hub/financial-markets"
    assert "Market snapshot" in document.text
    assert "Tesla shares rise" in document.text
    assert "IGNORE" not in document.text
    assert document.links[0].url == "https://apnews.com/article/tesla-update"
    assert document.links[0].text == "Tesla shares rise"


def test_parse_html_document_reports_empty_or_unreadable_html_as_malformed() -> None:
    with pytest.raises(MalformedProviderResponse, match="empty HTML"):
        parse_html_document("", source_url="https://example.invalid/")

    with pytest.raises(MalformedProviderResponse, match="without readable text"):
        parse_html_document(
            "<html><script>window.onlyScript = true;</script></html>",
            source_url="https://example.invalid/",
        )


def test_drift_warnings_for_missing_text_preserve_required_and_optional_context() -> None:
    document = parse_html_document(
        HTML,
        source_url="https://apnews.com/hub/financial-markets",
    )

    warnings = drift_warnings_for_missing_text(
        document=document,
        requirements=(
            HtmlTextRequirement(selector="main h1", expected_text="Market snapshot"),
            HtmlTextRequirement(
                selector="article[data-key]",
                expected_text="Breaking",
                required=True,
            ),
            HtmlTextRequirement(
                selector=".optional",
                expected_text="Analyst table",
                required=False,
            ),
        ),
        provider_name="apnews-public",
        occurred_at=FETCHED_AT,
        raw_snapshot_id="raw-apnews-html",
    )

    assert [warning.code for warning in warnings] == [
        WarningCode.SCRAPING_DRIFT,
        WarningCode.SCRAPING_DRIFT,
    ]
    assert warnings[0].severity == WarningSeverity.ERROR
    assert warnings[0].metadata["selector"] == "article[data-key]"
    assert warnings[0].metadata["expected_text"] == "Breaking"
    assert warnings[1].severity == WarningSeverity.WARNING
    assert warnings[1].metadata["required"] is False
