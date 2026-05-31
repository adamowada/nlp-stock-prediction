"""Policy-aware public Reddit search and discussion scraper."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TypeVar
from urllib.parse import urlencode, urlsplit

from nlp_stock_prediction.contracts import (
    CredentialState,
    EvidenceRequest,
    FreshnessStatus,
    ProviderHealth,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    RetrievalMethod,
    SourceEvidence,
    TickerDiscoveryRequest,
    TickerDiscoveryResult,
    TickerDiscoveryStatus,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.providers._base import (
    ProviderTransportError,
    build_cache_key,
    provider_health,
    provider_result,
    provider_warning,
    transport_error_result,
    utc_now,
)
from nlp_stock_prediction.providers.scraping import (
    DEFAULT_HTML_MAX_BYTES,
    HtmlCache,
    HtmlFetch,
    HtmlResponse,
    HtmlTransport,
    UrllibHtmlTransport,
    build_scraping_headers,
    fetch_html,
    raw_snapshot_id_for_html,
)
from nlp_stock_prediction.reddit.discovery import discover_tickers_from_devvit_html
from nlp_stock_prediction.reddit.evidence import normalize_reddit_evidence
from nlp_stock_prediction.reddit.public_html import (
    extract_reddit_discussion_records_from_public_html,
    extract_reddit_search_results_from_public_html,
    extract_snapshot_observed_at,
)

T = TypeVar("T")

_DEFAULT_REDDIT_SEARCH_URL = "https://www.reddit.com/search/"
_DEFAULT_USER_AGENT = (
    "nlp-stock-prediction/0.1 (educational fixture-backed stock report; public Reddit pages only)"
)
_DEFAULT_FRESHNESS_WINDOW_SECONDS = 86_400
_DEFAULT_SEARCH_RESULT_LIMIT = 8
_DEFAULT_DISCUSSION_PAGE_LIMIT = 6


@dataclass(frozen=True, slots=True)
class StaticHtmlTransport:
    """Deterministic HTML transport for tests and curated fixtures."""

    pages: Mapping[str, str | HtmlResponse]

    def get_html(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
        max_bytes: int = DEFAULT_HTML_MAX_BYTES,
    ) -> HtmlResponse:
        del headers, timeout, max_bytes
        page = self.pages.get(url)
        if page is None:
            for url_fragment, candidate in self.pages.items():
                if url_fragment in url:
                    page = candidate
                    break
        if page is None:
            raise ProviderTransportError(
                f"No fixture HTML is registered for {url}",
                status_code=404,
                error_type="fixture_not_found",
            )
        if isinstance(page, HtmlResponse):
            return page
        return HtmlResponse(html=page, final_url=url)


@dataclass(frozen=True, slots=True)
class RedditPublicPagePolicyDecision:
    """Decision from the local W2 policy shim."""

    allowed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class _RedditSearchSelection:
    discussion_urls: tuple[str, ...]
    metadata_by_url: Mapping[str, Mapping[str, object]]
    warnings: tuple[ProviderWarning, ...]
    raw_snapshot_ids: tuple[str, ...]
    search_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RedditPublicPagePolicy:
    """Allow only public Reddit HTML pages, never API/private/login paths."""

    allowed_hosts: tuple[str, ...] = ("www.reddit.com", "reddit.com")

    def evaluate(self, url: str) -> RedditPublicPagePolicyDecision:
        parsed = urlsplit(url)
        host = parsed.netloc.lower()
        path = parsed.path.lower().rstrip("/")
        path_with_slash = f"{path}/"
        if parsed.scheme != "https":
            return RedditPublicPagePolicyDecision(False, "reddit_public_pages_require_https")
        if host not in self.allowed_hosts:
            return RedditPublicPagePolicyDecision(False, "non_reddit_host")
        if path.endswith(".json") or "/.json" in path or parsed.query.lower().endswith(".json"):
            return RedditPublicPagePolicyDecision(False, "reddit_json_endpoint_disallowed")
        if any(
            disallowed in path_with_slash
            for disallowed in (
                "/api/",
                "/dev/",
                "/login/",
                "/register/",
                "/settings/",
                "/user/",
                "/users/",
                "/message/",
            )
        ):
            return RedditPublicPagePolicyDecision(False, "reddit_private_or_endpoint_path")
        if path in {"/search", "/search/"} or path.startswith("/r/"):
            return RedditPublicPagePolicyDecision(True, "allowed_public_reddit_html")
        return RedditPublicPagePolicyDecision(False, "outside_public_reddit_html_path")


class RedditPublicPageProvider:
    """Scrape public Reddit search/discussion HTML into existing contracts."""

    provider_name = "reddit-public-search"

    def __init__(
        self,
        *,
        source_url: str = _DEFAULT_REDDIT_SEARCH_URL,
        discussion_urls: Sequence[str] = (),
        transport: HtmlTransport | None = None,
        allow_live_scraping: bool = False,
        policy: RedditPublicPagePolicy | None = None,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 10.0,
        stale_after_seconds: int = _DEFAULT_FRESHNESS_WINDOW_SECONDS,
        search_result_limit: int = _DEFAULT_SEARCH_RESULT_LIMIT,
        discussion_page_limit: int = _DEFAULT_DISCUSSION_PAGE_LIMIT,
        search_sort: str = "new",
        user_agent: str = _DEFAULT_USER_AGENT,
        latency_ms: int | None = None,
        cache: HtmlCache | None = None,
    ) -> None:
        self._source_url = source_url
        self._discussion_urls = tuple(discussion_urls)
        self._transport = transport or (UrllibHtmlTransport() if allow_live_scraping else None)
        self._policy = policy or RedditPublicPagePolicy()
        self._now = now
        self._timeout = timeout
        self._stale_after_seconds = stale_after_seconds
        self._search_result_limit = max(1, search_result_limit)
        self._discussion_page_limit = max(1, discussion_page_limit)
        self._search_sort = search_sort
        self._user_agent = user_agent
        self._latency_ms = latency_ms
        self._cache = cache

    def discover_tickers(
        self,
        request: TickerDiscoveryRequest,
    ) -> ProviderResult[TickerDiscoveryResult]:
        fetched_at = self._now()
        source_url = request.source_url or self._source_url
        cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="reddit-ticker-card",
            run_date=request.run_date,
            query=request.query,
            url=source_url,
        )
        policy_warning = self._policy_warning(source_url, fetched_at)
        if policy_warning is not None:
            return self._blocked_result(
                request=request,
                fetched_at=fetched_at,
                warning=policy_warning,
                cache_key=cache_key,
            )
        disabled: ProviderResult[TickerDiscoveryResult] | None = self._disabled_result(
            request=request,
            fetched_at=fetched_at,
            cache_key=cache_key,
        )
        if disabled is not None:
            return disabled

        try:
            response = self._fetch(
                source_url,
                run_date=request.run_date,
                ticker=None,
                source="reddit-ticker-card",
                cache_key=cache_key,
            )
        except ProviderTransportError as exc:
            return transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=CredentialState.NOT_REQUIRED,
            )
        raw_snapshot_id = response.raw_snapshot_id
        observed_at = extract_snapshot_observed_at(response.html) or fetched_at
        effective_request = request.model_copy(update={"source_url": source_url})
        discovery = discover_tickers_from_devvit_html(
            response.html,
            request=effective_request,
            fetched_at=fetched_at,
            raw_snapshot_id=raw_snapshot_id,
            retrieval_method=RetrievalMethod.PUBLIC_SCRAPE,
            provider_name=self.provider_name,
            observed_at=observed_at,
            freshness_window_seconds=self._stale_after_seconds,
        )
        warnings = tuple(discovery.warnings)
        login_warning = self._login_wall_warning(
            response.html,
            source_url,
            fetched_at,
            has_public_content=bool(discovery.candidates),
        )
        if login_warning is not None:
            return self._blocked_result(
                request=request,
                fetched_at=fetched_at,
                warning=login_warning,
                cache_key=cache_key,
            )
        if _is_stale_observation(discovery):
            warnings += (
                provider_warning(
                    provider_name=self.provider_name,
                    code=WarningCode.STALE_DATA,
                    severity=WarningSeverity.WARNING,
                    message="Reddit ticker-card public page snapshot is stale.",
                    occurred_at=fetched_at,
                    raw_snapshot_id=raw_snapshot_id,
                    source_url=source_url,
                    stale_after=fetched_at,
                    metadata={"validation": "stale_reddit_ticker_card"},
                ),
            )
        status = _discovery_provider_status(discovery, warnings)
        return provider_result(
            provider_name=self.provider_name,
            status=status,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            data=discovery,
            warnings=warnings,
            raw_snapshot_id=raw_snapshot_id,
            cache_key=cache_key,
            latency_ms=self._latency_ms,
        )

    def fetch_discussion(
        self,
        request: EvidenceRequest,
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        fetched_at = self._now()
        explicit_urls = self._request_discussion_urls(request)
        search_selection = (
            _RedditSearchSelection(
                discussion_urls=explicit_urls,
                metadata_by_url={},
                warnings=(),
                raw_snapshot_ids=(),
                search_urls=(),
            )
            if explicit_urls
            else self._search_discussion_urls(request, fetched_at)
        )
        source_urls = search_selection.discussion_urls
        cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="reddit-discussion-pages",
            run_date=request.run_date,
            tickers=request.tickers,
            query=request.query,
            url="|".join((*search_selection.search_urls, *source_urls)),
        )
        disabled: ProviderResult[tuple[SourceEvidence, ...]] | None = self._disabled_result(
            request=request,
            fetched_at=fetched_at,
            cache_key=cache_key,
        )
        if disabled is not None:
            return disabled

        evidence: list[SourceEvidence] = []
        warnings: list[ProviderWarning] = list(search_selection.warnings)
        raw_snapshot_ids: list[str] = list(search_selection.raw_snapshot_ids)
        for source_url in source_urls:
            policy_warning = self._policy_warning(source_url, fetched_at)
            if policy_warning is not None:
                warnings.append(policy_warning)
                continue
            try:
                response = self._fetch(
                    source_url,
                    run_date=request.run_date,
                    ticker=None,
                    source="reddit-discussion-page",
                    cache_key=build_cache_key(
                        provider_name=self.provider_name,
                        source="reddit-discussion-page",
                        run_date=request.run_date,
                        tickers=request.tickers,
                        query=request.query,
                        url=source_url,
                    ),
                )
            except ProviderTransportError as exc:
                warnings.append(
                    provider_warning(
                        provider_name=self.provider_name,
                        code=_warning_code_from_transport(exc),
                        severity=WarningSeverity.ERROR,
                        message=str(exc),
                        occurred_at=fetched_at,
                        retryable=exc.retryable,
                        provider_status_code=exc.status_code,
                        provider_error_type=exc.error_type,
                        source_url=source_url,
                    )
                )
                continue
            discussion_raw_snapshot_id = response.raw_snapshot_id
            raw_snapshot_ids.append(discussion_raw_snapshot_id)
            records = extract_reddit_discussion_records_from_public_html(
                response.html,
                source_url=source_url,
            )
            metadata = search_selection.metadata_by_url.get(_canonical_url(source_url), {})
            records = tuple(
                _record_with_retrieval_metadata(
                    record,
                    discussion_url=source_url,
                    metadata=metadata,
                )
                for record in records
            )
            login_warning = self._login_wall_warning(
                response.html,
                source_url,
                fetched_at,
                has_public_content=bool(records),
            )
            if login_warning is not None:
                warnings.append(login_warning)
                continue
            evidence.extend(
                normalize_reddit_evidence(
                    records,
                    request=request,
                    fetched_at=fetched_at,
                    raw_snapshot_id=discussion_raw_snapshot_id,
                    retrieval_method=RetrievalMethod.PUBLIC_SCRAPE,
                    provider_name=self.provider_name,
                    freshness_window_seconds=self._stale_after_seconds,
                )
            )

        combined_raw_snapshot_id = "|".join(raw_snapshot_ids) or None
        if not evidence:
            status = _empty_discussion_status(warnings)
            if status != ProviderStatus.RATE_LIMITED:
                warnings.append(
                    provider_warning(
                        provider_name=self.provider_name,
                        code=WarningCode.NO_DATA,
                        severity=WarningSeverity.INFO,
                        message="Reddit public pages produced no high-confidence ticker evidence.",
                        occurred_at=fetched_at,
                        raw_snapshot_id=combined_raw_snapshot_id,
                        metadata={
                            "validation": "no_public_discussion_evidence",
                            "source_urls": list(source_urls),
                            "search_urls": list(search_selection.search_urls),
                            "search_terms": list(_reddit_search_terms(request)),
                        },
                    )
                )
            return provider_result(
                provider_name=self.provider_name,
                status=status,
                request=request,
                fetched_at=fetched_at,
                credential_state=CredentialState.NOT_REQUIRED,
                warnings=tuple(warnings),
                raw_snapshot_id=combined_raw_snapshot_id,
                cache_key=cache_key,
                latency_ms=self._latency_ms,
            )

        if any(item.provenance.freshness_status == FreshnessStatus.STALE for item in evidence):
            warnings.append(
                provider_warning(
                    provider_name=self.provider_name,
                    code=WarningCode.STALE_DATA,
                    severity=WarningSeverity.WARNING,
                    message="Reddit public pages returned stale discussion evidence.",
                    occurred_at=fetched_at,
                    raw_snapshot_id=combined_raw_snapshot_id,
                    metadata={"validation": "stale_reddit_discussion"},
                )
            )
        status = _discussion_provider_status(tuple(warnings))
        return provider_result(
            provider_name=self.provider_name,
            status=status,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            data=tuple(evidence),
            warnings=tuple(warnings),
            raw_snapshot_id=combined_raw_snapshot_id,
            cache_key=cache_key,
            latency_ms=self._latency_ms,
        )

    def health(self) -> ProviderHealth:
        checked_at = self._now()
        if self._transport is None:
            warning = self._disabled_warning(checked_at)
            return provider_health(
                provider_name=self.provider_name,
                status=ProviderStatus.UNCONFIGURED,
                checked_at=checked_at,
                credential_state=CredentialState.NOT_REQUIRED,
                warnings=(warning,),
                latency_ms=self._latency_ms,
            )
        return provider_health(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            checked_at=checked_at,
            credential_state=CredentialState.NOT_REQUIRED,
            latency_ms=self._latency_ms,
        )

    def _fetch(
        self,
        source_url: str,
        *,
        run_date: date,
        ticker: str | None,
        source: str,
        cache_key: str,
    ) -> HtmlFetch:
        if self._transport is None:
            raise ProviderTransportError(
                "Reddit live public-page scraping is disabled.",
                error_type="live_scraping_disabled",
            )
        return fetch_html(
            transport=self._transport,
            url=source_url,
            run_date=run_date,
            ticker=ticker,
            source=source,
            cache_key=cache_key,
            fetched_at=self._now(),
            cache=self._cache,
            headers=build_scraping_headers(
                user_agent=self._user_agent,
                extra_headers={"Accept": "text/html"},
            ),
            timeout=self._timeout,
        )

    def _request_discussion_urls(self, request: EvidenceRequest) -> tuple[str, ...]:
        option_urls = request.options.get("discussion_urls")
        if isinstance(option_urls, Sequence) and not isinstance(option_urls, str):
            urls = tuple(item for item in option_urls if isinstance(item, str) and item.strip())
            if urls:
                return urls
        if self._discussion_urls:
            return self._discussion_urls
        return ()

    def _search_discussion_urls(
        self,
        request: EvidenceRequest,
        fetched_at: datetime,
    ) -> _RedditSearchSelection:
        search_terms = _reddit_search_terms(request)
        search_urls = tuple(
            build_reddit_public_search_url(term, sort=self._search_sort) for term in search_terms
        )
        warnings: list[ProviderWarning] = []
        raw_snapshot_ids: list[str] = []
        discussion_urls: list[str] = []
        metadata_by_url: dict[str, Mapping[str, object]] = {}
        seen: set[str] = set()

        for search_url, search_term in zip(search_urls, search_terms, strict=True):
            policy_warning = self._policy_warning(search_url, fetched_at)
            if policy_warning is not None:
                warnings.append(policy_warning)
                continue
            try:
                response = self._fetch(
                    search_url,
                    run_date=request.run_date,
                    ticker=None,
                    source="reddit-search-page",
                    cache_key=build_cache_key(
                        provider_name=self.provider_name,
                        source="reddit-search-page",
                        run_date=request.run_date,
                        tickers=request.tickers,
                        query=search_term,
                        url=search_url,
                    ),
                )
            except ProviderTransportError as exc:
                warnings.append(
                    provider_warning(
                        provider_name=self.provider_name,
                        code=_warning_code_from_transport(exc),
                        severity=WarningSeverity.ERROR,
                        message=str(exc),
                        occurred_at=fetched_at,
                        retryable=exc.retryable,
                        provider_status_code=exc.status_code,
                        provider_error_type=exc.error_type,
                        source_url=search_url,
                        metadata={"search_query": search_term},
                    )
                )
                continue
            raw_snapshot_ids.append(response.raw_snapshot_id)
            results = extract_reddit_search_results_from_public_html(
                response.html,
                source_url=search_url,
            )
            login_warning = self._login_wall_warning(
                response.html,
                search_url,
                fetched_at,
                has_public_content=bool(results),
            )
            if login_warning is not None:
                warnings.append(login_warning)
                continue
            for rank, result in enumerate(results[: self._search_result_limit]):
                permalink = _string(result.get("permalink")) or _string(result.get("url"))
                if permalink is None:
                    continue
                key = _canonical_url(permalink)
                if key in seen:
                    continue
                seen.add(key)
                discussion_urls.append(permalink)
                metadata_by_url[key] = {
                    "search_query": search_term,
                    "search_url": search_url,
                    "result_rank": rank,
                    "result_title": result.get("title"),
                    "result_subreddit": result.get("subreddit"),
                }
                if len(discussion_urls) >= self._discussion_page_limit:
                    break
            if len(discussion_urls) >= self._discussion_page_limit:
                break

        if not discussion_urls:
            warnings.append(
                provider_warning(
                    provider_name=self.provider_name,
                    code=WarningCode.NO_DATA,
                    severity=WarningSeverity.INFO,
                    message="Reddit public search returned no discussion result links.",
                    occurred_at=fetched_at,
                    raw_snapshot_id="|".join(raw_snapshot_ids) or None,
                    metadata={
                        "validation": "no_public_search_results",
                        "search_terms": list(search_terms),
                        "search_urls": list(search_urls),
                    },
                )
            )

        return _RedditSearchSelection(
            discussion_urls=tuple(discussion_urls[: self._discussion_page_limit]),
            metadata_by_url=metadata_by_url,
            warnings=tuple(warnings),
            raw_snapshot_ids=tuple(raw_snapshot_ids),
            search_urls=search_urls,
        )

    def _policy_warning(self, source_url: str, fetched_at: datetime) -> ProviderWarning | None:
        decision = self._policy.evaluate(source_url)
        if decision.allowed:
            return None
        return provider_warning(
            provider_name=self.provider_name,
            code=WarningCode.UPSTREAM_UNAVAILABLE,
            severity=WarningSeverity.ERROR,
            message="Reddit scraper policy blocked a non-public or unsupported URL.",
            occurred_at=fetched_at,
            provider_error_type="scraping_blocked_by_policy",
            source_url=source_url,
            metadata={"policy_reason": decision.reason},
        )

    def _login_wall_warning(
        self,
        html: str,
        source_url: str,
        fetched_at: datetime,
        *,
        has_public_content: bool = False,
    ) -> ProviderWarning | None:
        if has_public_content:
            return None
        lowered = html.lower()
        if "login" not in lowered or "reddit" not in lowered:
            return None
        login_wall_markers = (
            'data-testid="login"',
            "log in to reddit",
            "log in to view",
            "login required",
            "sign up to continue",
        )
        if not any(marker in lowered for marker in login_wall_markers):
            return None
        return provider_warning(
            provider_name=self.provider_name,
            code=WarningCode.AUTH_FAILED,
            severity=WarningSeverity.ERROR,
            message="Reddit returned a login-only page instead of public discussion HTML.",
            occurred_at=fetched_at,
            provider_error_type="login_wall",
            source_url=source_url,
        )

    def _disabled_result(
        self,
        *,
        request: ProviderRequest,
        fetched_at: datetime,
        cache_key: str,
    ) -> ProviderResult[T] | None:
        if self._transport is not None:
            return None
        warning = self._disabled_warning(fetched_at)
        return provider_result(
            provider_name=self.provider_name,
            status=ProviderStatus.UNCONFIGURED,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            warnings=(warning,),
            cache_key=cache_key,
            latency_ms=self._latency_ms,
        )

    def _disabled_warning(self, fetched_at: datetime) -> ProviderWarning:
        return provider_warning(
            provider_name=self.provider_name,
            code=WarningCode.UPSTREAM_UNAVAILABLE,
            severity=WarningSeverity.INFO,
            message=(
                "Reddit live public-page scraping is disabled; provide fixture HTML transport "
                "or explicitly enable live scraping."
            ),
            occurred_at=fetched_at,
            provider_error_type="live_scraping_disabled",
        )

    def _blocked_result(
        self,
        *,
        request: ProviderRequest,
        fetched_at: datetime,
        warning: ProviderWarning,
        cache_key: str,
    ) -> ProviderResult[T]:
        return provider_result(
            provider_name=self.provider_name,
            status=ProviderStatus.FAILED,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            warnings=(warning,),
            cache_key=cache_key,
            latency_ms=self._latency_ms,
        )


def _raw_snapshot_id(source: str, html: str) -> str:
    return raw_snapshot_id_for_html(source, html)


def build_reddit_public_search_url(term: str, *, sort: str = "new") -> str:
    """Return the public Reddit HTML search URL for a bounded scrape query."""

    query = urlencode({"q": " ".join(term.split()), "type": "link", "sort": sort})
    return f"{_DEFAULT_REDDIT_SEARCH_URL}?{query}"


def _reddit_search_terms(request: EvidenceRequest) -> tuple[str, ...]:
    option_terms = request.options.get("reddit_search_terms")
    if isinstance(option_terms, Sequence) and not isinstance(option_terms, str):
        option_term_values = tuple(
            dict.fromkeys(item.strip() for item in option_terms if isinstance(item, str) and item)
        )
        if option_term_values:
            return option_term_values
    terms: list[str] = []
    for ticker in request.tickers:
        normalized = ticker.strip().upper()
        if not normalized:
            continue
        terms.append(f"${normalized}")
        terms.append(f"{normalized} stock")
    company_name = request.options.get("company_name")
    if isinstance(company_name, str) and company_name.strip():
        terms.append(f"{company_name.strip()} stock")
    aliases = request.options.get("aliases")
    if isinstance(aliases, Sequence) and not isinstance(aliases, str):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                terms.append(f"{alias.strip()} stock")
    if request.query and not terms:
        terms.append(request.query)
    return tuple(dict.fromkeys(terms or ["stock market"]))


def _record_with_retrieval_metadata(
    record: Mapping[str, object],
    *,
    discussion_url: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    copied = dict(record)
    copied["source_url"] = discussion_url
    copied["discussion_url"] = discussion_url
    for key in ("search_query", "search_url", "result_rank"):
        value = metadata.get(key)
        if _is_json_scalar(value):
            copied[key] = value
    return copied


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}/"


def _string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _is_json_scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool) or value is None


def _is_stale_observation(discovery: TickerDiscoveryResult) -> bool:
    return any(
        candidate.provenance.freshness_status == FreshnessStatus.STALE
        for candidate in discovery.candidates
    )


def _discovery_provider_status(
    discovery: TickerDiscoveryResult,
    warnings: tuple[ProviderWarning, ...],
) -> ProviderStatus:
    if discovery.status != TickerDiscoveryStatus.VALID:
        return ProviderStatus.PARTIAL
    if any(warning.code == WarningCode.STALE_DATA for warning in warnings):
        return ProviderStatus.STALE
    return ProviderStatus.OK


def _discussion_provider_status(warnings: tuple[ProviderWarning, ...]) -> ProviderStatus:
    if any(warning.code == WarningCode.STALE_DATA for warning in warnings):
        return ProviderStatus.STALE
    if warnings:
        return ProviderStatus.PARTIAL
    return ProviderStatus.OK


def _only_hard_failures(warnings: Sequence[ProviderWarning]) -> bool:
    return bool(warnings) and all(
        warning.code
        in {
            WarningCode.AUTH_FAILED,
            WarningCode.UPSTREAM_UNAVAILABLE,
        }
        for warning in warnings
    )


def _empty_discussion_status(warnings: Sequence[ProviderWarning]) -> ProviderStatus:
    if warnings and all(warning.code == WarningCode.RATE_LIMITED for warning in warnings):
        return ProviderStatus.RATE_LIMITED
    return ProviderStatus.FAILED if _only_hard_failures(warnings) else ProviderStatus.EMPTY


def _warning_code_from_transport(error: ProviderTransportError) -> WarningCode:
    if error.status_code in {401, 403}:
        return WarningCode.AUTH_FAILED
    if error.status_code == 429:
        return WarningCode.RATE_LIMITED
    return WarningCode.UPSTREAM_UNAVAILABLE


__all__ = [
    "HtmlResponse",
    "HtmlTransport",
    "RedditPublicPagePolicy",
    "RedditPublicPagePolicyDecision",
    "RedditPublicPageProvider",
    "StaticHtmlTransport",
    "UrllibHtmlTransport",
    "build_reddit_public_search_url",
]
