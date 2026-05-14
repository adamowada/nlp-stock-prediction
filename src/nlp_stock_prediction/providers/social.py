"""X/social provider adapter."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from nlp_stock_prediction.contracts.enums import (
    CredentialState,
    FreshnessStatus,
    ProviderStatus,
    RetrievalMethod,
    SourceKind,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.provenance import ProviderHealth, ProviderWarning
from nlp_stock_prediction.contracts.providers import EvidenceRequest, ProviderResult
from nlp_stock_prediction.providers._base import (
    JsonFetch,
    JsonTransport,
    MalformedProviderResponse,
    ProviderCache,
    ProviderTransportError,
    UrllibJsonTransport,
    append_query_params,
    build_cache_key,
    fetch_json,
    find_ticker_matches,
    first_ticker,
    freshness_status,
    malformed_result,
    missing_credentials_result,
    parse_optional_provider_datetime,
    provider_health,
    provider_warning,
    source_provenance,
    stable_hash,
    transport_error_result,
    utc_now,
)
from nlp_stock_prediction.providers.execution import (
    evidence_result_from_records,
    partial_item_warning,
)


def build_x_recent_search_query(
    ticker: str,
    *,
    lang: str = "en",
    exclude_retweets: bool = True,
) -> str:
    """Build the recent-search query used for a ticker cashtag."""

    symbol = ticker.strip().upper().removeprefix("$")
    parts = [f"${symbol}", f"lang:{lang}"]
    if exclude_retweets:
        parts.append("-is:retweet")
    return " ".join(parts)


class XRecentSearchProvider:
    """X recent-search adapter backed by the official API shape."""

    provider_name = "x-recent-search"
    default_limit = 50
    default_sort_order = "relevancy"

    def __init__(
        self,
        *,
        bearer_token: str | None = None,
        endpoint: str = "https://api.x.com/2/tweets/search/recent",
        sort_order: str = default_sort_order,
        default_limit: int = default_limit,
        transport: JsonTransport | None = None,
        cache: ProviderCache | None = None,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 10.0,
        stale_after_seconds: int = 7 * 24 * 60 * 60,
    ) -> None:
        self._bearer_token = bearer_token
        self._endpoint = endpoint
        self._sort_order = _validate_sort_order(sort_order)
        self._default_limit = _validate_limit(default_limit)
        self._transport = transport or UrllibJsonTransport()
        self._cache = cache
        self._now = now
        self._timeout = timeout
        self._stale_after_seconds = stale_after_seconds

    def fetch_social_posts(
        self, request: EvidenceRequest
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        fetched_at = self._now()
        if not self._bearer_token:
            return missing_credentials_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                credential_name="X bearer token",
            )
        ticker = first_ticker(request)
        query = request.query or build_x_recent_search_query(ticker or "")
        url = append_query_params(
            self._endpoint,
            {
                "query": query,
                "sort_order": self._sort_order,
                "max_results": request.limit or self._default_limit,
                "tweet.fields": "created_at,public_metrics,lang,author_id",
            },
        )
        cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="x-recent-search",
            run_date=request.run_date,
            tickers=request.tickers,
            query=query,
            url=url,
        )
        fetched: JsonFetch | None = None
        try:
            fetched = fetch_json(
                transport=self._transport,
                url=url,
                run_date=request.run_date,
                ticker=ticker,
                source="x-recent-search",
                cache_key=cache_key,
                fetched_at=fetched_at,
                cache=self._cache,
                headers={"Authorization": f"Bearer {self._bearer_token}"},
                timeout=self._timeout,
            )
            evidence, partial_warnings = self._map_payload(
                request,
                fetched.payload,
                fetched,
                query,
                url,
                fetched_at,
                self._sort_order,
            )
        except ProviderTransportError as exc:
            return transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=CredentialState.CONFIGURED,
            )
        except MalformedProviderResponse as exc:
            return malformed_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                credential_state=CredentialState.CONFIGURED,
                raw_snapshot_id=fetched.raw_snapshot_id if fetched is not None else None,
                cache_key=fetched.cache_key if fetched is not None else cache_key,
            )
        assert fetched is not None
        return evidence_result_from_records(
            provider_name=self.provider_name,
            request=request,
            fetched_at=fetched_at,
            evidence=evidence,
            warnings=partial_warnings,
            no_data_message="X recent search returned no posts",
            stale_message="X recent search returned stale social posts",
            credential_state=CredentialState.CONFIGURED,
            raw_snapshot_id=fetched.raw_snapshot_id,
            cache_key=fetched.cache_key,
        )

    def health(self) -> ProviderHealth:
        checked_at = self._now()
        if not self._bearer_token:
            warning = provider_warning(
                provider_name=self.provider_name,
                code=WarningCode.MISSING_CREDENTIALS,
                severity=WarningSeverity.ERROR,
                message="X bearer token is not configured",
                occurred_at=checked_at,
            )
            return provider_health(
                provider_name=self.provider_name,
                status=ProviderStatus.UNCONFIGURED,
                checked_at=checked_at,
                credential_state=CredentialState.MISSING,
                warnings=(warning,),
            )
        return provider_health(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            checked_at=checked_at,
            credential_state=CredentialState.CONFIGURED,
        )

    def _map_payload(
        self,
        request: EvidenceRequest,
        payload: dict[str, object],
        fetched: JsonFetch,
        query: str,
        source_url: str,
        fetched_at: datetime,
        sort_order: str,
    ) -> tuple[tuple[SourceEvidence, ...], tuple[ProviderWarning, ...]]:
        if "data" not in payload:
            raise MalformedProviderResponse("X response missing data")
        items = payload.get("data")
        if not isinstance(items, list):
            raise MalformedProviderResponse("X response data must be a list")
        evidence: list[SourceEvidence] = []
        warnings: list[ProviderWarning] = []
        for index, raw_item in enumerate(items):
            if not isinstance(raw_item, dict):
                warnings.append(
                    partial_item_warning(
                        provider_name=self.provider_name,
                        fetched_at=fetched_at,
                        raw_snapshot_id=fetched.raw_snapshot_id,
                        index=index,
                        message="X response item must be an object",
                    )
                )
                continue
            post_id = str(raw_item.get("id") or "").strip()
            text = str(raw_item.get("text") or "").strip()
            if not post_id or not text:
                warnings.append(
                    partial_item_warning(
                        provider_name=self.provider_name,
                        fetched_at=fetched_at,
                        raw_snapshot_id=fetched.raw_snapshot_id,
                        index=index,
                        message="X response item missing id or text",
                    )
                )
                continue
            created_at = parse_optional_provider_datetime(raw_item.get("created_at"))
            if created_at is None:
                freshness = FreshnessStatus.MISSING
                freshness_seconds = None
                warnings.append(
                    partial_item_warning(
                        provider_name=self.provider_name,
                        fetched_at=fetched_at,
                        raw_snapshot_id=fetched.raw_snapshot_id,
                        index=index,
                        message="X response item missing created_at timestamp",
                    )
                )
            else:
                freshness, freshness_seconds = freshness_status(
                    observed_at=created_at,
                    fetched_at=fetched_at,
                    stale_after_seconds=self._stale_after_seconds,
                )
            matched_tickers, spans = find_ticker_matches(text, request.tickers)
            if request.tickers and not matched_tickers:
                continue
            permalink = f"https://x.com/i/web/status/{post_id}"
            metrics = raw_item.get("public_metrics")
            metric_map = metrics if isinstance(metrics, dict) else {}
            evidence.append(
                SourceEvidence(
                    evidence_id=f"x:{post_id}",
                    source_kind=SourceKind.X_POST,
                    ticker=matched_tickers[0] if matched_tickers else None,
                    title=None,
                    text=text,
                    author_hash=_author_hash(raw_item.get("author_id")),
                    created_at=created_at,
                    score=_social_score(metric_map),
                    permalink=permalink,
                    matched_tickers=matched_tickers,
                    match_spans=spans,
                    provenance=source_provenance(
                        provider_name=self.provider_name,
                        source_kind=SourceKind.X_POST,
                        retrieval_method=RetrievalMethod.OFFICIAL_API,
                        fetched_at=fetched_at,
                        observed_at=created_at,
                        source_url=source_url,
                        permalink=permalink,
                        raw_identifier=post_id,
                        raw_snapshot_id=fetched.raw_snapshot_id,
                        query=query,
                        cache_key=fetched.cache_key,
                        freshness=freshness,
                        freshness_seconds=freshness_seconds,
                        provider_metadata={
                            "lang": raw_item.get("lang"),
                            "sort_order": sort_order,
                            "cache_hit": fetched.cache_hit,
                            "public_metrics": {
                                key: value
                                for key, value in metric_map.items()
                                if isinstance(key, str)
                            },
                        },
                    ),
                    metadata={"provider": self.provider_name},
                )
            )
        return tuple(evidence), tuple(warnings)


def _author_hash(author_id: object) -> str | None:
    if author_id is None:
        return None
    text = str(author_id).strip()
    if not text:
        return None
    return f"sha256:{stable_hash(text, length=24)}"


def _social_score(metrics: dict[object, object]) -> int | None:
    score = 0
    found = False
    for key in ("like_count", "retweet_count", "reply_count", "quote_count"):
        value = metrics.get(key)
        if isinstance(value, int):
            score += value
            found = True
    return score if found else None


def _validate_sort_order(sort_order: str) -> str:
    normalized = sort_order.strip().lower()
    if normalized not in {"relevancy", "recency"}:
        raise ValueError("X recent-search sort_order must be relevancy or recency")
    return normalized


def _validate_limit(limit: int) -> int:
    if limit < 10 or limit > 100:
        raise ValueError("X recent-search default_limit must be between 10 and 100")
    return limit


__all__ = ["XRecentSearchProvider", "build_x_recent_search_query"]
