"""Shared helpers for public HTML provider adapters."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from nlp_stock_prediction.contracts import ProviderWarning, WarningCode, WarningSeverity
from nlp_stock_prediction.providers._base import (
    MalformedProviderResponse,
    ProviderTransportError,
    ensure_aware_utc,
    provider_warning,
    safe_path_component,
    stable_hash,
)
from nlp_stock_prediction.reliability import retry_call

SCRAPE_USER_AGENT_ENV = "NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT"
SCRAPE_MIN_DELAY_SECONDS_ENV = "NLP_STOCK_PREDICTION_SCRAPE_MIN_DELAY_SECONDS"
DEFAULT_SCRAPE_USER_AGENT = "nlp-stock-prediction/0.1 public-html-adapter"
DEFAULT_SCRAPE_MIN_DELAY_SECONDS = 1.0
DEFAULT_HTML_MAX_BYTES = 2_000_000


@dataclass(frozen=True)
class HtmlResponse:
    """Raw HTML response plus HTTP metadata from a provider transport."""

    html: str
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    final_url: str | None = None


class HtmlTransport(Protocol):
    """Small transport interface so HTML provider tests can avoid live network calls."""

    def get_html(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
        max_bytes: int = DEFAULT_HTML_MAX_BYTES,
    ) -> HtmlResponse: ...


class UrllibHtmlTransport:
    """Stdlib urllib-backed HTML transport used only by opt-in live callers."""

    def get_html(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
        max_bytes: int = DEFAULT_HTML_MAX_BYTES,
    ) -> HtmlResponse:
        request = Request(url, headers=dict(headers or {}))
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read(max_bytes + 1)
                status_code = int(getattr(response, "status", 200))
                if len(body) > max_bytes:
                    raise ProviderTransportError(
                        f"provider HTML response exceeded {max_bytes} bytes",
                        status_code=status_code,
                        retryable=False,
                        error_type="response_too_large",
                    )
                response_headers = dict(response.headers.items())
                return HtmlResponse(
                    html=_decode_html(body, response_headers),
                    status_code=status_code,
                    headers=response_headers,
                    final_url=str(getattr(response, "url", url)),
                )
        except HTTPError as exc:
            raise ProviderTransportError(
                str(exc),
                status_code=exc.code,
                retryable=exc.code in {408, 425, 429, 500, 502, 503, 504},
                error_type="http_error",
            ) from exc
        except URLError as exc:
            raise ProviderTransportError(str(exc), retryable=True, error_type="url_error") from exc
        except TimeoutError as exc:
            raise ProviderTransportError(str(exc), retryable=True, error_type="timeout") from exc
        except (OSError, UnicodeDecodeError, LookupError) as exc:
            raise ProviderTransportError(
                str(exc),
                retryable=True,
                error_type="read_decode_error",
            ) from exc


@dataclass(frozen=True)
class HtmlCacheRecord:
    html: str
    raw_snapshot_id: str
    cache_key: str
    source_url: str
    canonical_url: str
    content_sha256: str


class HtmlCache:
    """File cache for raw HTML snapshots and a small metadata sidecar."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, run_date: date, ticker: str | None, source: str, cache_key: str) -> Path:
        ticker_component = safe_path_component(ticker or "all")
        source_component = safe_path_component(source)
        cache_filename = f"{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:16]}.html"
        return (
            self.root / run_date.isoformat() / ticker_component / source_component / cache_filename
        )

    def metadata_path_for(
        self, run_date: date, ticker: str | None, source: str, cache_key: str
    ) -> Path:
        return self.path_for(run_date, ticker, source, cache_key).with_suffix(".json")

    def load_html(
        self,
        *,
        run_date: date,
        ticker: str | None,
        source: str,
        cache_key: str,
    ) -> HtmlCacheRecord | None:
        path = self.path_for(run_date, ticker, source, cache_key)
        if not path.exists():
            return None
        html = path.read_text(encoding="utf-8")
        metadata = _load_cache_metadata(self.metadata_path_for(run_date, ticker, source, cache_key))
        content_hash = content_sha256_for_html(html)
        raw_snapshot_id = _metadata_text(metadata, "raw_snapshot_id") or raw_snapshot_id_for_html(
            source, html
        )
        source_url = _metadata_text(metadata, "source_url") or ""
        canonical_url = _metadata_text(metadata, "canonical_url") or source_url
        return HtmlCacheRecord(
            html=html,
            raw_snapshot_id=raw_snapshot_id,
            cache_key=cache_key,
            source_url=source_url,
            canonical_url=canonical_url,
            content_sha256=content_hash,
        )

    def save_html(
        self,
        *,
        run_date: date,
        ticker: str | None,
        source: str,
        cache_key: str,
        html: str,
        source_url: str,
        canonical_url: str,
        fetched_at: datetime,
    ) -> HtmlCacheRecord:
        path = self.path_for(run_date, ticker, source, cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        raw_snapshot_id = raw_snapshot_id_for_html(source, html)
        content_hash = content_sha256_for_html(html)
        metadata_path = self.metadata_path_for(run_date, ticker, source, cache_key)
        metadata_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "cached_at": ensure_aware_utc(fetched_at).isoformat().replace("+00:00", "Z"),
                    "raw_snapshot_id": raw_snapshot_id,
                    "source_url": source_url,
                    "canonical_url": canonical_url,
                    "content_sha256": content_hash,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return HtmlCacheRecord(
            html=html,
            raw_snapshot_id=raw_snapshot_id,
            cache_key=cache_key,
            source_url=source_url,
            canonical_url=canonical_url,
            content_sha256=content_hash,
        )


@dataclass(frozen=True)
class HtmlFetch:
    """Fetched or cached HTML plus snapshot and cache metadata."""

    html: str
    raw_snapshot_id: str
    cache_key: str
    cache_hit: bool
    status_code: int
    source_url: str
    canonical_url: str
    content_sha256: str


@dataclass(frozen=True)
class HtmlLink:
    """A link discovered while parsing a public HTML document."""

    url: str
    text: str
    rel: str | None = None


@dataclass(frozen=True)
class HtmlDocument:
    """Small parsed HTML document representation for fixture-backed adapters."""

    source_url: str
    canonical_url: str
    title: str | None
    text: str
    links: tuple[HtmlLink, ...] = ()

    def contains_text(self, expected_text: str, *, case_sensitive: bool = False) -> bool:
        haystack = self.text if case_sensitive else self.text.lower()
        needle = expected_text if case_sensitive else expected_text.lower()
        return needle in haystack


@dataclass(frozen=True)
class HtmlTextRequirement:
    """Expected visible text or section label used to detect markup drift."""

    selector: str
    expected_text: str
    required: bool = True


def configured_scrape_user_agent(env: Mapping[str, str] | None = None) -> str:
    """Return the configured polite scraping User-Agent or the deterministic default."""

    value = (env or os.environ).get(SCRAPE_USER_AGENT_ENV)
    if value is None or not value.strip():
        return DEFAULT_SCRAPE_USER_AGENT
    return value.strip()


def configured_scrape_min_delay_seconds(
    env: Mapping[str, str] | None = None,
    *,
    default: float = DEFAULT_SCRAPE_MIN_DELAY_SECONDS,
) -> float:
    """Return a non-negative configured scrape delay for future adapter throttling."""

    value = (env or os.environ).get(SCRAPE_MIN_DELAY_SECONDS_ENV)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value.strip())
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def scraping_drift_warning(
    *,
    provider_name: str,
    source_url: str,
    selector: str,
    occurred_at: datetime,
    raw_snapshot_id: str | None = None,
    required: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> ProviderWarning:
    """Build a provider warning when a public HTML probe stops matching."""

    warning_message = (
        f"Required public HTML selector missing or changed: {selector}"
        if required
        else f"Optional public HTML selector missing or changed: {selector}"
    )
    warning_metadata = {
        "selector": selector,
        "required": required,
        "drift_type": "missing_selector",
    }
    warning_metadata.update(dict(metadata or {}))
    return provider_warning(
        provider_name=provider_name,
        code=WarningCode.SCRAPING_DRIFT,
        severity=WarningSeverity.ERROR if required else WarningSeverity.WARNING,
        message=warning_message,
        occurred_at=occurred_at,
        provider_error_type="scraping_drift",
        raw_snapshot_id=raw_snapshot_id,
        source_url=source_url,
        metadata=warning_metadata,
    )


def build_scraping_headers(
    *,
    user_agent: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build consistent public HTML request headers."""

    headers = {"User-Agent": user_agent or configured_scrape_user_agent()}
    headers.update(dict(extra_headers or {}))
    return headers


def fetch_html(
    *,
    transport: HtmlTransport,
    url: str,
    run_date: date,
    ticker: str | None,
    source: str,
    cache_key: str,
    fetched_at: datetime,
    cache: HtmlCache | None = None,
    headers: Mapping[str, str] | None = None,
    user_agent: str | None = None,
    timeout: float = 10.0,
    max_bytes: int = DEFAULT_HTML_MAX_BYTES,
) -> HtmlFetch:
    """Fetch public HTML with deterministic cache semantics for provider adapters."""

    if cache is not None:
        cached = cache.load_html(
            run_date=run_date,
            ticker=ticker,
            source=source,
            cache_key=cache_key,
        )
        if cached is not None:
            return HtmlFetch(
                html=cached.html,
                raw_snapshot_id=cached.raw_snapshot_id,
                cache_key=cached.cache_key,
                cache_hit=True,
                status_code=200,
                source_url=cached.source_url or url,
                canonical_url=cached.canonical_url or url,
                content_sha256=cached.content_sha256,
            )

    response = retry_call(
        lambda: transport.get_html(
            url,
            headers=build_scraping_headers(user_agent=user_agent, extra_headers=headers),
            timeout=timeout,
            max_bytes=max_bytes,
        ),
        should_retry=lambda exc: isinstance(exc, ProviderTransportError) and exc.retryable,
    )
    if not response.html.strip():
        raise MalformedProviderResponse("provider returned empty HTML")
    canonical_url = response.final_url or url
    if cache is None:
        return HtmlFetch(
            html=response.html,
            raw_snapshot_id=raw_snapshot_id_for_html(source, response.html),
            cache_key=cache_key,
            cache_hit=False,
            status_code=response.status_code,
            source_url=url,
            canonical_url=canonical_url,
            content_sha256=content_sha256_for_html(response.html),
        )
    record = cache.save_html(
        run_date=run_date,
        ticker=ticker,
        source=source,
        cache_key=cache_key,
        html=response.html,
        source_url=url,
        canonical_url=canonical_url,
        fetched_at=fetched_at,
    )
    return HtmlFetch(
        html=record.html,
        raw_snapshot_id=record.raw_snapshot_id,
        cache_key=record.cache_key,
        cache_hit=False,
        status_code=response.status_code,
        source_url=record.source_url,
        canonical_url=record.canonical_url,
        content_sha256=record.content_sha256,
    )


def parse_html_document(
    html: str,
    *,
    source_url: str,
    canonical_url: str | None = None,
) -> HtmlDocument:
    """Parse visible text, title, canonical URL, and links from public HTML."""

    if not html.strip():
        raise MalformedProviderResponse("provider returned empty HTML")
    parser = _VisibleTextParser(source_url)
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        raise MalformedProviderResponse("provider returned malformed HTML") from exc
    text = " ".join(parser.text_parts).strip()
    if not text:
        raise MalformedProviderResponse("provider returned HTML without readable text")
    parsed_title = " ".join(parser.title_parts).strip() or None
    parsed_canonical_url = parser.canonical_url or canonical_url or source_url
    return HtmlDocument(
        source_url=source_url,
        canonical_url=parsed_canonical_url,
        title=parsed_title,
        text=text,
        links=tuple(parser.links),
    )


def drift_warnings_for_missing_text(
    *,
    document: HtmlDocument,
    requirements: Sequence[HtmlTextRequirement],
    provider_name: str,
    occurred_at: datetime,
    raw_snapshot_id: str | None = None,
) -> tuple[ProviderWarning, ...]:
    """Build drift warnings for required visible-text probes that no longer match."""

    warnings: list[ProviderWarning] = []
    for requirement in requirements:
        if not document.contains_text(requirement.expected_text):
            warnings.append(
                scraping_drift_warning(
                    provider_name=provider_name,
                    source_url=document.source_url,
                    selector=requirement.selector,
                    occurred_at=occurred_at,
                    raw_snapshot_id=raw_snapshot_id,
                    required=requirement.required,
                    metadata={"expected_text": requirement.expected_text},
                )
            )
    return tuple(warnings)


def content_sha256_for_html(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def raw_snapshot_id_for_html(source: str, html: str) -> str:
    return f"raw-{safe_path_component(source)}-{stable_hash(html, length=20)}"


def _decode_html(body: bytes, headers: Mapping[str, str]) -> str:
    content_type = headers.get("Content-Type") or headers.get("content-type") or ""
    charset = "utf-8"
    for part in content_type.split(";"):
        stripped = part.strip()
        if stripped.lower().startswith("charset="):
            charset = stripped.split("=", 1)[1].strip() or "utf-8"
    return body.decode(charset, errors="replace")


def _load_cache_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _metadata_text(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


class _VisibleTextParser(HTMLParser):
    def __init__(self, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.text_parts: list[str] = []
        self.title_parts: list[str] = []
        self.links: list[HtmlLink] = []
        self.canonical_url: str | None = None
        self._skip_depth = 0
        self._title_depth = 0
        self._active_link_url: str | None = None
        self._active_link_rel: str | None = None
        self._active_link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        attr_map = _attrs_to_map(attrs)
        if tag_lower in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag_lower == "title":
            self._title_depth += 1
            return
        if tag_lower == "link":
            rel = attr_map.get("rel", "")
            href = attr_map.get("href")
            if href and "canonical" in rel.lower().split():
                self.canonical_url = urljoin(self.source_url, href)
            return
        if tag_lower == "a":
            href = attr_map.get("href")
            if href:
                self._active_link_url = urljoin(self.source_url, href)
                self._active_link_rel = attr_map.get("rel")
                self._active_link_text = []

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag_lower == "title" and self._title_depth:
            self._title_depth -= 1
            return
        if tag_lower == "a" and self._active_link_url:
            text = " ".join(self._active_link_text).strip()
            self.links.append(
                HtmlLink(
                    url=self._active_link_url,
                    text=text,
                    rel=self._active_link_rel,
                )
            )
            self._active_link_url = None
            self._active_link_rel = None
            self._active_link_text = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._title_depth:
            self.title_parts.append(text)
        self.text_parts.append(text)
        if self._active_link_url:
            self._active_link_text.append(text)


def _attrs_to_map(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
    return {key.lower(): value for key, value in attrs if value is not None}


__all__ = [
    "DEFAULT_HTML_MAX_BYTES",
    "DEFAULT_SCRAPE_MIN_DELAY_SECONDS",
    "DEFAULT_SCRAPE_USER_AGENT",
    "SCRAPE_MIN_DELAY_SECONDS_ENV",
    "SCRAPE_USER_AGENT_ENV",
    "HtmlCache",
    "HtmlCacheRecord",
    "HtmlDocument",
    "HtmlFetch",
    "HtmlLink",
    "HtmlResponse",
    "HtmlTextRequirement",
    "HtmlTransport",
    "UrllibHtmlTransport",
    "build_scraping_headers",
    "configured_scrape_min_delay_seconds",
    "configured_scrape_user_agent",
    "content_sha256_for_html",
    "drift_warnings_for_missing_text",
    "fetch_html",
    "parse_html_document",
    "raw_snapshot_id_for_html",
    "scraping_drift_warning",
]
