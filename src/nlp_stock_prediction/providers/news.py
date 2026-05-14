"""Public news provider adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
    missing_credentials_result,
    parse_optional_provider_datetime,
    provider_health,
    provider_warning,
    query_from_tickers,
    source_provenance,
    stable_hash,
    transport_error_result,
    utc_now,
)
from nlp_stock_prediction.providers.execution import (
    ProviderExecutionContext,
    partial_item_warning,
)


@dataclass(frozen=True)
class PublicNewsProviderConfig:
    """Configurable mapping for public headline/article APIs such as NewsAPI."""

    provider_name: str = "newsapi"
    endpoint: str = "https://newsapi.org/v2/everything"
    api_key_param: str = "apiKey"
    query_param: str = "q"
    articles_key: str = "articles"
    requires_api_key: bool = True
    language: str = "en"


class PublicNewsProvider:
    """News article adapter using configured public provider field mappings."""

    def __init__(
        self,
        *,
        config: PublicNewsProviderConfig,
        api_key: str | None = None,
        transport: JsonTransport | None = None,
        cache: ProviderCache | None = None,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 10.0,
        stale_after_seconds: int = 3 * 24 * 60 * 60,
    ) -> None:
        self.provider_name = config.provider_name
        self._config = config
        self._api_key = api_key
        self._transport = transport or UrllibJsonTransport()
        self._cache = cache
        self._now = now
        self._timeout = timeout
        self._stale_after_seconds = stale_after_seconds

    def fetch_articles(
        self,
        request: EvidenceRequest,
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        fetched_at = self._now()
        if self._config.requires_api_key and not self._api_key:
            return missing_credentials_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                credential_name=f"{self.provider_name} API key",
            )
        query = query_from_tickers(request)
        ticker = first_ticker(request)
        params: dict[str, object | None] = {
            self._config.query_param: query,
            "pageSize": request.limit,
            "language": self._config.language,
            "sortBy": "publishedAt",
        }
        if self._api_key:
            params[self._config.api_key_param] = self._api_key
        url = append_query_params(self._config.endpoint, params)
        cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="news-articles",
            run_date=request.run_date,
            tickers=request.tickers,
            query=query,
            url=url,
        )
        credential_state = (
            CredentialState.CONFIGURED
            if self._config.requires_api_key
            else CredentialState.NOT_REQUIRED
        )
        execution = ProviderExecutionContext(
            provider_name=self.provider_name,
            request=request,
            fetched_at=fetched_at,
            credential_state=credential_state,
            cache_key=cache_key,
        )
        fetched: JsonFetch | None = None
        try:
            fetched = fetch_json(
                transport=self._transport,
                url=url,
                run_date=request.run_date,
                ticker=ticker,
                source="news-articles",
                cache_key=cache_key,
                fetched_at=fetched_at,
                cache=self._cache,
                timeout=self._timeout,
                cacheable_payload=lambda payload: isinstance(
                    payload.get(self._config.articles_key),
                    list,
                ),
            )
            evidence, partial_warnings = self._map_payload(request, fetched, query, url, fetched_at)
        except ProviderTransportError as exc:
            return transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=credential_state,
            )
        except MalformedProviderResponse as exc:
            if fetched is not None:
                execution = execution.with_fetch(
                    raw_snapshot_id=fetched.raw_snapshot_id,
                    cache_key=fetched.cache_key,
                )
            return execution.malformed(str(exc))
        assert fetched is not None
        return execution.with_fetch(
            raw_snapshot_id=fetched.raw_snapshot_id,
            cache_key=fetched.cache_key,
        ).evidence_result(
            evidence=evidence,
            warnings=partial_warnings,
            no_data_message=f"{self.provider_name} returned no articles",
            stale_message=f"{self.provider_name} returned stale news articles",
        )

    def health(self) -> ProviderHealth:
        checked_at = self._now()
        if self._config.requires_api_key and not self._api_key:
            warning = provider_warning(
                provider_name=self.provider_name,
                code=WarningCode.MISSING_CREDENTIALS,
                severity=WarningSeverity.ERROR,
                message=f"{self.provider_name} API key is not configured",
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
            credential_state=(
                CredentialState.CONFIGURED
                if self._config.requires_api_key
                else CredentialState.NOT_REQUIRED
            ),
        )

    def _map_payload(
        self,
        request: EvidenceRequest,
        fetched: JsonFetch,
        query: str,
        source_url: str,
        fetched_at: datetime,
    ) -> tuple[tuple[SourceEvidence, ...], tuple[ProviderWarning, ...]]:
        if self._config.articles_key not in fetched.payload:
            raise MalformedProviderResponse(
                f"{self.provider_name} response missing {self._config.articles_key}"
            )
        articles = fetched.payload.get(self._config.articles_key)
        if not isinstance(articles, list):
            raise MalformedProviderResponse(
                f"{self.provider_name} response {self._config.articles_key} must be a list"
            )
        evidence: list[SourceEvidence] = []
        warnings: list[ProviderWarning] = []
        for index, raw_article in enumerate(articles):
            if not isinstance(raw_article, dict):
                warnings.append(
                    partial_item_warning(
                        provider_name=self.provider_name,
                        fetched_at=fetched_at,
                        raw_snapshot_id=fetched.raw_snapshot_id,
                        index=index,
                        message=f"{self.provider_name} article must be an object",
                    )
                )
                continue
            title = _optional_text(raw_article.get("title"))
            text = (
                _optional_text(raw_article.get("content"))
                or _optional_text(raw_article.get("description"))
                or title
            )
            article_url = _optional_text(raw_article.get("url"))
            if not text or not article_url:
                warnings.append(
                    partial_item_warning(
                        provider_name=self.provider_name,
                        fetched_at=fetched_at,
                        raw_snapshot_id=fetched.raw_snapshot_id,
                        index=index,
                        message=f"{self.provider_name} article missing text or url",
                    )
                )
                continue
            published_at = parse_optional_provider_datetime(raw_article.get("publishedAt"))
            if published_at is None:
                freshness = FreshnessStatus.MISSING
                freshness_seconds = None
                warnings.append(
                    partial_item_warning(
                        provider_name=self.provider_name,
                        fetched_at=fetched_at,
                        raw_snapshot_id=fetched.raw_snapshot_id,
                        index=index,
                        message=f"{self.provider_name} article missing published timestamp",
                    )
                )
            else:
                freshness, freshness_seconds = freshness_status(
                    observed_at=published_at,
                    fetched_at=fetched_at,
                    stale_after_seconds=self._stale_after_seconds,
                )
            matched_tickers, spans = find_ticker_matches(text, request.tickers)
            if request.tickers and not matched_tickers:
                continue
            source = raw_article.get("source")
            source_name = source.get("name") if isinstance(source, dict) else None
            raw_identifier = article_url or f"{self.provider_name}:{index}:{stable_hash(text)}"
            evidence.append(
                SourceEvidence(
                    evidence_id=f"news:{stable_hash(raw_identifier, length=24)}",
                    source_kind=SourceKind.NEWS_ARTICLE,
                    ticker=matched_tickers[0] if matched_tickers else None,
                    title=title,
                    text=text,
                    author_hash=None,
                    created_at=published_at,
                    score=None,
                    permalink=article_url,
                    matched_tickers=matched_tickers,
                    match_spans=spans,
                    provenance=source_provenance(
                        provider_name=self.provider_name,
                        source_kind=SourceKind.NEWS_ARTICLE,
                        retrieval_method=RetrievalMethod.OFFICIAL_API,
                        fetched_at=fetched_at,
                        observed_at=published_at,
                        source_url=article_url,
                        permalink=article_url,
                        raw_identifier=raw_identifier,
                        raw_snapshot_id=fetched.raw_snapshot_id,
                        query=query,
                        cache_key=fetched.cache_key,
                        freshness=freshness,
                        freshness_seconds=freshness_seconds,
                        provider_metadata={
                            "source_name": source_name,
                            "author": raw_article.get("author"),
                            "cache_hit": fetched.cache_hit,
                            "source_query_url": _redact_sensitive_url(source_url),
                        },
                    ),
                    metadata={"source_name": source_name},
                )
            )
        return tuple(evidence), tuple(warnings)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _redact_sensitive_url(url: str) -> str:
    sensitive = {"api_key", "apikey", "token", "access_token", "key", "bearer"}
    split = urlsplit(url)
    query = urlencode(
        [
            (key, "REDACTED" if key.lower() in sensitive else value)
            for key, value in parse_qsl(split.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((split.scheme, split.netloc, split.path, query, split.fragment))


__all__ = ["PublicNewsProvider", "PublicNewsProviderConfig"]
