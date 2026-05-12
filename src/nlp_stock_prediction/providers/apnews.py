"""AP News public financial-markets HTML provider."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from nlp_stock_prediction.contracts import (
    CredentialState,
    EvidenceRequest,
    FreshnessStatus,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.providers._base import (
    JsonPayload,
    MalformedProviderResponse,
    ProviderCache,
    ProviderTransportError,
    build_cache_key,
    find_ticker_matches,
    first_ticker,
    freshness_status,
    no_data_result,
    provider_health,
    provider_result,
    provider_warning,
    query_from_tickers,
    raw_snapshot_id_for_payload,
    source_provenance,
    stable_hash,
    transport_error_result,
    utc_now,
)

_DEFAULT_HUB_URL = "https://apnews.com/hub/financial-markets"
_DEFAULT_USER_AGENT = "nlp-stock-prediction/0.1 public-html-scraper"
_AP_HOSTS = frozenset({"apnews.com", "www.apnews.com"})
_DISALLOWED_PATH_PREFIXES = (
    "/api/",
    "/api",
    "/feed/",
    "/feed",
    "/rss/",
    "/rss",
    "/search/",
    "/search",
)


@dataclass(frozen=True)
class HtmlResponse:
    """Fetched HTML response plus minimal HTTP metadata."""

    text: str
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)


class HtmlTransport(Protocol):
    """Small HTML transport interface for fixture-backed provider tests."""

    def get_text(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> HtmlResponse: ...


class UrllibHtmlTransport:
    """Stdlib urllib-backed HTML transport used only by explicit live callers."""

    def get_text(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> HtmlResponse:
        request = Request(url, headers=dict(headers or {}))
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                status_code = int(getattr(response, "status", 200))
                response_headers = dict(response.headers.items())
                return HtmlResponse(text=body, status_code=status_code, headers=response_headers)
        except HTTPError as exc:
            raise ProviderTransportError(
                str(exc),
                status_code=exc.code,
                retryable=exc.code in {408, 425, 429, 500, 502, 503, 504},
                error_type="http_error",
            ) from exc
        except URLError as exc:
            raise ProviderTransportError(str(exc), retryable=True, error_type="url_error") from exc


@dataclass(frozen=True)
class APNewsProviderConfig:
    """Configuration for AP News public financial-markets HTML scraping."""

    provider_name: str = "ap-news"
    hub_url: str = _DEFAULT_HUB_URL
    user_agent: str = _DEFAULT_USER_AGENT
    max_article_candidates: int = 12
    default_limit: int = 10


@dataclass(frozen=True)
class _HtmlFetch:
    text: str
    raw_snapshot_id: str
    cache_key: str
    cache_hit: bool
    status_code: int
    source_url: str


@dataclass(frozen=True)
class _HubArticleLink:
    url: str
    headline: str | None
    rank: int


@dataclass(frozen=True)
class _ParsedArticle:
    canonical_url: str
    headline: str
    text: str
    published_at: datetime | None
    author_label: str | None
    source_label: str | None


class APNewsProvider:
    """Scrape AP News financial-markets hub and linked public article HTML."""

    def __init__(
        self,
        *,
        config: APNewsProviderConfig | None = None,
        transport: HtmlTransport | None = None,
        cache: ProviderCache | None = None,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 10.0,
        stale_after_seconds: int = 3 * 24 * 60 * 60,
    ) -> None:
        resolved_config = config or APNewsProviderConfig()
        self.provider_name = resolved_config.provider_name
        self._hub_url = _validate_apnews_public_url(resolved_config.hub_url)
        self._user_agent = resolved_config.user_agent
        self._max_article_candidates = _validate_positive_int(
            resolved_config.max_article_candidates,
            name="max_article_candidates",
        )
        self._default_limit = _validate_positive_int(
            resolved_config.default_limit,
            name="default_limit",
        )
        self._transport = transport or UrllibHtmlTransport()
        self._cache = cache
        self._now = now
        self._timeout = timeout
        self._stale_after_seconds = stale_after_seconds

    def fetch_articles(
        self,
        request: EvidenceRequest,
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        fetched_at = self._now()
        query = query_from_tickers(request)
        hub_cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="ap-news-hub",
            run_date=request.run_date,
            tickers=request.tickers,
            query=query,
            url=self._hub_url,
        )
        try:
            hub_fetch = _fetch_html(
                transport=self._transport,
                url=self._hub_url,
                run_date=request.run_date,
                ticker=first_ticker(request),
                source="ap-news-hub",
                cache_key=hub_cache_key,
                fetched_at=fetched_at,
                cache=self._cache,
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
            )
        except ProviderTransportError as exc:
            return transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=CredentialState.NOT_REQUIRED,
            )

        article_links = _extract_hub_article_links(hub_fetch.text, base_url=self._hub_url)
        if not article_links:
            warning = provider_warning(
                provider_name=self.provider_name,
                code=WarningCode.SCRAPING_DRIFT,
                severity=WarningSeverity.ERROR,
                message="AP News financial-markets hub did not expose public article links.",
                occurred_at=fetched_at,
                raw_snapshot_id=hub_fetch.raw_snapshot_id,
                source_url=self._hub_url,
                metadata={"validation": "missing_ap_article_links"},
            )
            return provider_result(
                provider_name=self.provider_name,
                status=ProviderStatus.MALFORMED,
                request=request,
                fetched_at=fetched_at,
                credential_state=CredentialState.NOT_REQUIRED,
                warnings=(warning,),
                raw_snapshot_id=hub_fetch.raw_snapshot_id,
                cache_key=hub_fetch.cache_key,
            )

        evidence: list[SourceEvidence] = []
        warnings: list[ProviderWarning] = []
        unmatched_article_count = 0
        result_limit = request.limit or self._default_limit
        for link in article_links[: self._max_article_candidates]:
            if len(evidence) >= result_limit:
                break
            article_cache_key = build_cache_key(
                provider_name=self.provider_name,
                source="ap-news-article",
                run_date=request.run_date,
                tickers=request.tickers,
                query=query,
                url=link.url,
                options={"hub_url": self._hub_url},
            )
            try:
                article_fetch = _fetch_html(
                    transport=self._transport,
                    url=link.url,
                    run_date=request.run_date,
                    ticker=first_ticker(request),
                    source="ap-news-article",
                    cache_key=article_cache_key,
                    fetched_at=fetched_at,
                    cache=self._cache,
                    headers={"User-Agent": self._user_agent},
                    timeout=self._timeout,
                )
                article = _extract_article(
                    article_fetch.text,
                    source_url=link.url,
                    fallback_headline=link.headline,
                )
            except ProviderTransportError as exc:
                warnings.append(
                    _transport_warning(
                        provider_name=self.provider_name,
                        fetched_at=fetched_at,
                        source_url=link.url,
                        error=exc,
                    )
                )
                continue
            except MalformedProviderResponse as exc:
                warnings.append(
                    provider_warning(
                        provider_name=self.provider_name,
                        code=WarningCode.SCRAPING_DRIFT,
                        severity=WarningSeverity.ERROR,
                        message=str(exc),
                        occurred_at=fetched_at,
                        source_url=link.url,
                        metadata={"validation": "malformed_ap_article", "article_rank": link.rank},
                    )
                )
                continue

            combined_text = f"{article.headline} {article.text}"
            matched_tickers, spans = find_ticker_matches(combined_text, request.tickers)
            if request.tickers and not matched_tickers:
                unmatched_article_count += 1
                continue

            freshness, freshness_seconds = _article_freshness(
                published_at=article.published_at,
                fetched_at=fetched_at,
                stale_after_seconds=self._stale_after_seconds,
            )
            if freshness == FreshnessStatus.MISSING:
                warnings.append(
                    provider_warning(
                        provider_name=self.provider_name,
                        code=WarningCode.PARTIAL_DATA,
                        severity=WarningSeverity.WARNING,
                        message="AP News article was missing a published timestamp.",
                        occurred_at=fetched_at,
                        raw_snapshot_id=article_fetch.raw_snapshot_id,
                        source_url=article.canonical_url,
                        metadata={"article_rank": link.rank},
                    )
                )
            elif freshness == FreshnessStatus.STALE:
                warnings.append(
                    provider_warning(
                        provider_name=self.provider_name,
                        code=WarningCode.STALE_DATA,
                        severity=WarningSeverity.WARNING,
                        message="AP News returned a stale financial-markets article.",
                        occurred_at=fetched_at,
                        raw_snapshot_id=article_fetch.raw_snapshot_id,
                        source_url=article.canonical_url,
                        metadata={
                            "article_rank": link.rank,
                            "published_at": article.published_at.isoformat()
                            if article.published_at
                            else None,
                        },
                    )
                )

            raw_identifier = article.canonical_url
            evidence.append(
                SourceEvidence(
                    evidence_id=f"ap-news:{stable_hash(raw_identifier, length=24)}",
                    source_kind=SourceKind.NEWS_ARTICLE,
                    ticker=matched_tickers[0] if matched_tickers else first_ticker(request),
                    title=article.headline,
                    text=article.text,
                    author_hash=_label_hash(article.author_label),
                    created_at=article.published_at,
                    score=None,
                    permalink=article.canonical_url,
                    matched_tickers=matched_tickers,
                    match_spans=spans,
                    provenance=source_provenance(
                        provider_name=self.provider_name,
                        source_kind=SourceKind.NEWS_ARTICLE,
                        retrieval_method=RetrievalMethod.PUBLIC_SCRAPE,
                        fetched_at=fetched_at,
                        observed_at=article.published_at,
                        source_url=article.canonical_url,
                        permalink=article.canonical_url,
                        raw_identifier=raw_identifier,
                        raw_snapshot_id=article_fetch.raw_snapshot_id,
                        query=query,
                        cache_key=article_fetch.cache_key,
                        freshness=freshness,
                        freshness_seconds=freshness_seconds,
                        provider_metadata={
                            "hub_url": self._hub_url,
                            "hub_cache_key": hub_fetch.cache_key,
                            "article_rank": link.rank,
                            "source_label": article.source_label,
                            "author_label": article.author_label,
                            "cache_hit": article_fetch.cache_hit,
                        },
                    ),
                    metadata={
                        "provider": self.provider_name,
                        "source_label": article.source_label,
                        "author_label": article.author_label,
                    },
                )
            )

        if not evidence:
            if warnings:
                return provider_result(
                    provider_name=self.provider_name,
                    status=_empty_failure_status(warnings),
                    request=request,
                    fetched_at=fetched_at,
                    credential_state=CredentialState.NOT_REQUIRED,
                    warnings=tuple(warnings),
                    raw_snapshot_id=hub_fetch.raw_snapshot_id,
                    cache_key=hub_fetch.cache_key,
                )
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message="AP News returned no ticker-matched financial-markets articles",
                raw_snapshot_id=hub_fetch.raw_snapshot_id,
                cache_key=hub_fetch.cache_key,
            )

        if unmatched_article_count:
            warnings.append(
                provider_warning(
                    provider_name=self.provider_name,
                    code=WarningCode.NO_DATA,
                    severity=WarningSeverity.INFO,
                    message="AP News skipped articles that did not mention requested tickers.",
                    occurred_at=fetched_at,
                    raw_snapshot_id=hub_fetch.raw_snapshot_id,
                    source_url=self._hub_url,
                    metadata={"unmatched_article_count": unmatched_article_count},
                )
            )
        status = _result_status(warnings)
        return provider_result(
            provider_name=self.provider_name,
            status=status,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            data=tuple(evidence),
            warnings=tuple(warnings),
            raw_snapshot_id=hub_fetch.raw_snapshot_id,
            cache_key=hub_fetch.cache_key,
        )

    def health(self) -> ProviderHealth:
        return provider_health(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            checked_at=self._now(),
            credential_state=CredentialState.NOT_REQUIRED,
        )


def _fetch_html(
    *,
    transport: HtmlTransport,
    url: str,
    run_date: date,
    ticker: str | None,
    source: str,
    cache_key: str,
    fetched_at: datetime,
    cache: ProviderCache | None,
    headers: Mapping[str, str] | None,
    timeout: float,
) -> _HtmlFetch:
    if cache is not None:
        cached = cache.load_json(
            run_date=run_date,
            ticker=ticker,
            source=source,
            cache_key=cache_key,
        )
        if cached is not None:
            html = cached.payload.get("html")
            cached_url = cached.payload.get("url")
            status_code = cached.payload.get("status_code", 200)
            if isinstance(html, str) and isinstance(cached_url, str):
                return _HtmlFetch(
                    text=html,
                    raw_snapshot_id=cached.raw_snapshot_id,
                    cache_key=cached.cache_key,
                    cache_hit=True,
                    status_code=status_code if isinstance(status_code, int) else 200,
                    source_url=cached_url,
                )

    response = transport.get_text(url, headers=headers, timeout=timeout)
    payload: JsonPayload = {
        "url": url,
        "html": response.text,
        "status_code": response.status_code,
    }
    if cache is None:
        return _HtmlFetch(
            text=response.text,
            raw_snapshot_id=raw_snapshot_id_for_payload(source, payload),
            cache_key=cache_key,
            cache_hit=False,
            status_code=response.status_code,
            source_url=url,
        )
    record = cache.save_json(
        run_date=run_date,
        ticker=ticker,
        source=source,
        cache_key=cache_key,
        payload=payload,
        fetched_at=fetched_at,
    )
    return _HtmlFetch(
        text=response.text,
        raw_snapshot_id=record.raw_snapshot_id,
        cache_key=record.cache_key,
        cache_hit=False,
        status_code=response.status_code,
        source_url=url,
    )


@dataclass(slots=True)
class _TextCapture:
    tag: str
    attrs: Mapping[str, str]
    parts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ParsedTime:
    datetime_value: str | None
    text: str | None


@dataclass(frozen=True)
class _ParsedLink:
    href: str
    text: str | None


class _APHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[_ParsedLink] = []
        self.meta: dict[str, str] = {}
        self.canonical_url: str | None = None
        self.paragraphs: list[str] = []
        self.headings: list[str] = []
        self.title: str | None = None
        self.times: list[_ParsedTime] = []
        self.jsonld_scripts: list[str] = []
        self._captures: list[_TextCapture] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        attr_map = _attrs_dict(attrs)
        if tag_name == "meta":
            _store_meta(self.meta, attr_map)
            return
        if tag_name == "link":
            rel = attr_map.get("rel", "").lower()
            href = attr_map.get("href")
            if href and "canonical" in rel.split():
                self.canonical_url = href
            return
        if tag_name in {"a", "p", "h1", "title", "time"}:
            capture = _TextCapture(tag=tag_name, attrs=attr_map)
            if tag_name == "a":
                fallback = attr_map.get("aria-label") or attr_map.get("title")
                if fallback:
                    capture.parts.append(fallback)
            self._captures.append(capture)
            return
        if tag_name == "script" and attr_map.get("type", "").lower() == "application/ld+json":
            self._captures.append(_TextCapture(tag=tag_name, attrs=attr_map))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not self._captures:
            return
        for capture in self._captures:
            capture.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        while self._captures:
            capture = self._captures.pop()
            self._finish_capture(capture)
            if capture.tag == tag_name:
                return

    def close(self) -> None:
        super().close()
        while self._captures:
            self._finish_capture(self._captures.pop())

    def _finish_capture(self, capture: _TextCapture) -> None:
        text = _normalize_text("".join(capture.parts))
        if capture.tag == "a":
            href = capture.attrs.get("href")
            if href:
                self.links.append(_ParsedLink(href=href, text=text))
        elif capture.tag == "p":
            if text:
                self.paragraphs.append(text)
        elif capture.tag == "h1":
            if text:
                self.headings.append(text)
        elif capture.tag == "title":
            if text and self.title is None:
                self.title = text
        elif capture.tag == "time":
            self.times.append(
                _ParsedTime(datetime_value=capture.attrs.get("datetime"), text=text or None)
            )
        elif capture.tag == "script" and text:
            self.jsonld_scripts.append(text)


def _extract_hub_article_links(html: str, *, base_url: str) -> tuple[_HubArticleLink, ...]:
    parser = _parse_html(html)
    links: list[_HubArticleLink] = []
    seen: set[str] = set()
    for parsed_link in parser.links:
        article_url = _normalize_apnews_article_url(parsed_link.href, base_url=base_url)
        if article_url is None or article_url in seen:
            continue
        seen.add(article_url)
        links.append(
            _HubArticleLink(
                url=article_url,
                headline=parsed_link.text,
                rank=len(links),
            )
        )
    return tuple(links)


def _extract_article(
    html: str,
    *,
    source_url: str,
    fallback_headline: str | None,
) -> _ParsedArticle:
    parser = _parse_html(html)
    jsonld = _extract_jsonld_article_metadata(parser.jsonld_scripts)
    canonical_url = (
        _normalize_apnews_article_url(_first_text(jsonld.get("url")), base_url=source_url)
        or _normalize_apnews_article_url(parser.canonical_url, base_url=source_url)
        or source_url
    )
    headline = _first_text(
        jsonld.get("headline"),
        parser.meta.get("og:title"),
        parser.meta.get("twitter:title"),
        parser.headings[0] if parser.headings else None,
        fallback_headline,
        parser.title,
    )
    description = _first_text(
        jsonld.get("description"),
        parser.meta.get("description"),
        parser.meta.get("og:description"),
    )
    article_body = _first_text(jsonld.get("articleBody"))
    paragraphs = _dedupe_text_blocks(parser.paragraphs)
    body = article_body or "\n\n".join(paragraphs)
    text = _join_article_text(description, body)
    if not headline or not text:
        raise MalformedProviderResponse("AP News article missing headline or body text")

    published_at = _first_datetime(
        jsonld.get("datePublished"),
        parser.meta.get("article:published_time"),
        parser.meta.get("date"),
        *(item.datetime_value for item in parser.times),
        *(item.text for item in parser.times),
    )
    return _ParsedArticle(
        canonical_url=canonical_url,
        headline=headline,
        text=text,
        published_at=published_at,
        author_label=_first_text(
            jsonld.get("author"),
            parser.meta.get("author"),
            parser.meta.get("byl"),
        ),
        source_label=_first_text(
            jsonld.get("publisher"),
            parser.meta.get("og:site_name"),
            "AP News",
        ),
    )


def _parse_html(html: str) -> _APHtmlParser:
    parser = _APHtmlParser()
    parser.feed(html)
    parser.close()
    return parser


def _extract_jsonld_article_metadata(scripts: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for script in scripts:
        try:
            root = json.loads(script)
        except json.JSONDecodeError:
            continue
        for node in _iter_jsonld_nodes(root):
            if not _is_article_jsonld_node(node):
                continue
            _copy_jsonld_text(metadata, node, "headline")
            _copy_jsonld_text(metadata, node, "description")
            _copy_jsonld_text(metadata, node, "articleBody")
            _copy_jsonld_text(metadata, node, "datePublished")
            _copy_jsonld_text(metadata, node, "url")
            author = _jsonld_name_value(node.get("author"))
            publisher = _jsonld_name_value(node.get("publisher"))
            if author and "author" not in metadata:
                metadata["author"] = author
            if publisher and "publisher" not in metadata:
                metadata["publisher"] = publisher
            if "headline" in metadata and ("articleBody" in metadata or "description" in metadata):
                return metadata
    return metadata


def _iter_jsonld_nodes(value: object) -> Iterator[Mapping[str, object]]:
    if isinstance(value, dict):
        node = cast(Mapping[str, object], value)
        yield node
        graph = node.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                yield from _iter_jsonld_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_jsonld_nodes(item)


def _is_article_jsonld_node(node: Mapping[str, object]) -> bool:
    type_value = node.get("@type")
    types: tuple[str, ...]
    if isinstance(type_value, str):
        types = (type_value,)
    elif isinstance(type_value, list):
        types = tuple(item for item in type_value if isinstance(item, str))
    else:
        types = ()
    normalized = {item.lower() for item in types}
    return bool({"newsarticle", "article"} & normalized)


def _copy_jsonld_text(metadata: dict[str, str], node: Mapping[str, object], key: str) -> None:
    value = _jsonld_text_value(node.get(key))
    if value and key not in metadata:
        metadata[key] = value


def _jsonld_text_value(value: object) -> str | None:
    if isinstance(value, str):
        return _normalize_text(value)
    return None


def _jsonld_name_value(value: object) -> str | None:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, Mapping):
        name = value.get("name")
        return _jsonld_text_value(name)
    if isinstance(value, list):
        names = tuple(name for item in value if (name := _jsonld_name_value(item)) is not None)
        return ", ".join(names) or None
    return None


def _article_freshness(
    *,
    published_at: datetime | None,
    fetched_at: datetime,
    stale_after_seconds: int,
) -> tuple[FreshnessStatus, int | None]:
    if published_at is None:
        return FreshnessStatus.MISSING, None
    return freshness_status(
        observed_at=published_at,
        fetched_at=fetched_at,
        stale_after_seconds=stale_after_seconds,
    )


def _first_datetime(*values: object) -> datetime | None:
    for value in values:
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _ensure_aware_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        if len(normalized) == 10:
            parsed_date = date.fromisoformat(normalized)
            return datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=UTC)
        return _ensure_aware_utc(datetime.fromisoformat(normalized))
    except ValueError:
        return None


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _transport_warning(
    *,
    provider_name: str,
    fetched_at: datetime,
    source_url: str,
    error: ProviderTransportError,
) -> ProviderWarning:
    code = (
        WarningCode.RATE_LIMITED if error.status_code == 429 else WarningCode.UPSTREAM_UNAVAILABLE
    )
    return provider_warning(
        provider_name=provider_name,
        code=code,
        severity=WarningSeverity.ERROR,
        message=str(error),
        occurred_at=fetched_at,
        retryable=error.retryable,
        provider_status_code=error.status_code,
        provider_error_type=error.error_type,
        source_url=source_url,
    )


def _result_status(warnings: Sequence[ProviderWarning]) -> ProviderStatus:
    if any(warning.code == WarningCode.STALE_DATA for warning in warnings):
        return ProviderStatus.STALE
    if warnings:
        return ProviderStatus.PARTIAL
    return ProviderStatus.OK


def _empty_failure_status(warnings: Sequence[ProviderWarning]) -> ProviderStatus:
    if any(warning.code == WarningCode.RATE_LIMITED for warning in warnings):
        return ProviderStatus.RATE_LIMITED
    if any(warning.code == WarningCode.SCRAPING_DRIFT for warning in warnings):
        return ProviderStatus.MALFORMED
    return ProviderStatus.FAILED


def _attrs_dict(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
    return {name.lower(): value for name, value in attrs if value is not None}


def _store_meta(meta: dict[str, str], attrs: Mapping[str, str]) -> None:
    content = attrs.get("content")
    if not content:
        return
    for key_name in ("property", "name", "itemprop"):
        key = attrs.get(key_name)
        if key:
            meta.setdefault(key.lower(), _normalize_text(content))


def _normalize_apnews_article_url(raw_url: str | None, *, base_url: str) -> str | None:
    normalized = _normalize_apnews_url(raw_url, base_url=base_url)
    if normalized is None:
        return None
    if not urlsplit(normalized).path.startswith("/article/"):
        return None
    return normalized


def _validate_apnews_public_url(raw_url: str) -> str:
    normalized = _normalize_apnews_url(raw_url, base_url=_DEFAULT_HUB_URL)
    if normalized is None:
        raise ValueError("AP News provider requires a public apnews.com HTML URL")
    return normalized


def _normalize_apnews_url(raw_url: str | None, *, base_url: str) -> str | None:
    if raw_url is None:
        return None
    raw_text = raw_url.strip()
    if not raw_text:
        return None
    split = urlsplit(urljoin(base_url, raw_text))
    if split.scheme not in {"http", "https"}:
        return None
    if split.netloc.lower() not in _AP_HOSTS:
        return None
    path = split.path or "/"
    if _is_disallowed_ap_path(path):
        return None
    return urlunsplit((split.scheme, split.netloc, path, "", ""))


def _is_disallowed_ap_path(path: str) -> bool:
    normalized = path.lower()
    return any(normalized.startswith(prefix) for prefix in _DISALLOWED_PATH_PREFIXES)


def _validate_positive_int(value: int, *, name: str) -> int:
    if value < 1:
        raise ValueError(f"AP News {name} must be positive")
    return value


def _join_article_text(description: str | None, body: str | None) -> str | None:
    blocks = _dedupe_text_blocks(tuple(item for item in (description, body) if item))
    if not blocks:
        return None
    return "\n\n".join(blocks)


def _dedupe_text_blocks(blocks: Sequence[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        normalized = _normalize_text(block)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return tuple(deduped)


def _first_text(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str):
            normalized = _normalize_text(value)
            if normalized:
                return normalized
    return None


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _label_hash(label: str | None) -> str | None:
    if label is None:
        return None
    normalized = label.strip().lower()
    if not normalized:
        return None
    return f"sha256:{stable_hash(normalized, length=24)}"


__all__ = [
    "APNewsProvider",
    "APNewsProviderConfig",
    "HtmlResponse",
    "HtmlTransport",
    "UrllibHtmlTransport",
]
