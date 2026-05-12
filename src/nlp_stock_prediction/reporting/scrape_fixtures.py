"""Fixture-backed scrape-mode orchestration for experimental source wiring."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast
from urllib.parse import parse_qs, unquote_plus, urlsplit

from nlp_stock_prediction.agents import FixtureFundamentalAgentProvider
from nlp_stock_prediction.analysis.fundamental_agent import apply_fundamental_agent_result
from nlp_stock_prediction.analysis.ml_signal import apply_technical_ml_signal
from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    AuditArtifact,
    AuditManifest,
    DailyReport,
    DataFreshnessSummary,
    EvidenceReference,
    EvidenceRequest,
    FreshnessStatus,
    FundamentalNlpAnalysisRequest,
    FundamentalNlpAnalysisResponse,
    JsonObject,
    JsonValue,
    MarketDataRequest,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    SourceEvidence,
    TechnicalMlSignal,
    TickerDiscoveryRequest,
    WarningCode,
)
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.providers._base import (
    JsonResponse,
    JsonTransport,
    ProviderCache,
    ProviderTransportError,
    utc_now,
)
from nlp_stock_prediction.providers.apnews import APNewsProvider
from nlp_stock_prediction.providers.candlecharts import CandlechartsMarketDataProvider
from nlp_stock_prediction.providers.reddit_scrape import (
    RedditPublicPageProvider,
)
from nlp_stock_prediction.providers.reddit_scrape import (
    StaticHtmlTransport as StaticRedditHtmlTransport,
)
from nlp_stock_prediction.providers.scraping import HtmlCache, HtmlResponse
from nlp_stock_prediction.providers.social import XRecentSearchProvider
from nlp_stock_prediction.reporting.audit import json_payload_sha256
from nlp_stock_prediction.reporting.fixtures import (
    TICKERS,
    OfflineFixtureBundle,
    build_offline_fixture_bundle,
)

_REDDIT_URL = "https://www.reddit.com/r/wallstreetbets/"
_REDDIT_POST_URL = "https://www.reddit.com/r/wallstreetbets/comments/scrape001/daily_watch/"
_AP_HUB_URL = "https://apnews.com/hub/financial-markets"
_AP_ARTICLE_URL = "https://apnews.com/article/markets-megacap-stocks-2026-05-11"
_X_BEARER_TOKEN_ENV = "NLP_STOCK_PREDICTION_X_BEARER_TOKEN"


@dataclass(frozen=True)
class ScrapeFixtureBundle:
    """Provider result payloads plus a contract-valid scrape-mode report."""

    report: DailyReport
    audit_payloads: dict[str, JsonObject]


class _StaticHtmlTransport:
    def __init__(self, pages: Mapping[str, str]) -> None:
        self._pages = pages

    def get_html(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
        max_bytes: int = 2_000_000,
    ) -> HtmlResponse:
        del headers, timeout, max_bytes
        for key, html in self._pages.items():
            if key in url:
                return HtmlResponse(html=html, final_url=url)
        raise ProviderTransportError(
            f"No scrape fixture HTML is registered for {url}",
            status_code=404,
            error_type="fixture_not_found",
        )


class _FixtureXTransport(JsonTransport):
    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> JsonResponse:
        del headers, timeout
        ticker = _ticker_from_x_query(url)
        return JsonResponse(
            payload={
                "data": [
                    {
                        "id": f"1900000000000000{index}{ticker.lower()}",
                        "text": (
                            f"${ticker} relevant fixture post discusses catalyst watch, "
                            "defined risk, and liquidity."
                        ),
                        "created_at": "2026-05-11T17:30:00Z",
                        "author_id": f"fixture-author-{ticker.lower()}-{index}",
                        "lang": "en",
                        "public_metrics": {
                            "like_count": 42 + index,
                            "retweet_count": 3,
                            "reply_count": 5,
                            "quote_count": 1,
                        },
                    }
                    for index in range(1, 3)
                ]
            }
        )


class _QuotaLimitedXTransport(JsonTransport):
    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> JsonResponse:
        del url, headers, timeout
        raise ProviderTransportError(
            "X recent-search quota exhausted in scrape-mode fixture probe",
            status_code=429,
            retryable=True,
            error_type="rate_limit",
        )


def build_scrape_fixture_bundle(config: RunConfig) -> ScrapeFixtureBundle:
    """Run the scrape-source providers against deterministic fixtures and render a report."""

    offline_bundle = build_offline_fixture_bundle(config)
    fetched_at = offline_bundle.report.generated_at
    provider_results = _provider_results(config, fetched_at)
    return _build_scrape_bundle_from_results(
        config=config,
        offline_bundle=offline_bundle,
        provider_results=provider_results,
        run_id=f"run-{config.run_date.isoformat()}-scrape-fixture",
        source_profile="fixture",
        generated_at=fetched_at,
        apply_fixture_sidecars=True,
    )


def build_live_scrape_bundle(config: RunConfig) -> ScrapeFixtureBundle:
    """Run scrape-source providers against live public/API sources and render a report."""

    offline_bundle = build_offline_fixture_bundle(config)
    fetched_at = utc_now()
    provider_results = _live_provider_results(config, fetched_at)
    return _build_scrape_bundle_from_results(
        config=config,
        offline_bundle=offline_bundle,
        provider_results=provider_results,
        run_id=f"run-{config.run_date.isoformat()}-scrape-live",
        source_profile="live",
        generated_at=fetched_at,
        apply_fixture_sidecars=False,
    )


def _build_scrape_bundle_from_results(
    *,
    config: RunConfig,
    offline_bundle: OfflineFixtureBundle,
    provider_results: tuple[ProviderResult[object], ...],
    run_id: str,
    source_profile: str,
    generated_at: datetime,
    apply_fixture_sidecars: bool,
) -> ScrapeFixtureBundle:
    provider_evidence = _provider_evidence(provider_results)
    provider_health = tuple(result.health for result in provider_results)
    report = _scrape_report(
        offline_bundle=offline_bundle,
        config=config,
        run_id=run_id,
        source_profile=source_profile,
        generated_at=generated_at,
        provider_results=provider_results,
        provider_health=provider_health,
        provider_evidence=provider_evidence,
        apply_fixture_sidecars=apply_fixture_sidecars,
    )
    audit_payloads = _scrape_audit_payloads(
        offline_bundle=offline_bundle,
        report=report,
        provider_results=provider_results,
        provider_evidence=provider_evidence,
        source_profile=source_profile,
    )
    audit_manifest = _scrape_audit_manifest(
        offline_manifest=cast(AuditManifest, offline_bundle.report.audit_manifest),
        config=config,
        run_id=run_id,
        source_profile=source_profile,
        generated_at=generated_at,
        provider_health=provider_health,
        provider_results=provider_results,
        audit_payloads=audit_payloads,
    )
    report = report.model_copy(update={"audit_manifest": audit_manifest})
    return ScrapeFixtureBundle(report=report, audit_payloads=audit_payloads)


def _live_provider_results(
    config: RunConfig,
    fetched_at: datetime,
) -> tuple[ProviderResult[object], ...]:
    html_cache = _html_cache(config)
    json_cache = _json_cache(config)
    reddit_provider = RedditPublicPageProvider(
        allow_live_scraping=True,
        now=lambda: fetched_at,
    )
    reddit_discovery = reddit_provider.discover_tickers(
        TickerDiscoveryRequest(
            request_id=f"live-scrape-reddit-discovery-{config.run_date.isoformat()}",
            run_date=config.run_date,
            source_url=_REDDIT_URL,
            query="r/wallstreetbets Devvit daily ticker card",
        )
    )
    reddit_discussion = reddit_provider.fetch_discussion(
        EvidenceRequest(
            request_id=f"live-scrape-reddit-discussion-{config.run_date.isoformat()}",
            run_date=config.run_date,
            tickers=TICKERS,
            query=" OR ".join(TICKERS),
            include_posts=True,
            include_comments=True,
        )
    )

    ap_articles = APNewsProvider(
        cache=html_cache,
        now=lambda: fetched_at,
    ).fetch_articles(
        EvidenceRequest(
            request_id=f"live-scrape-apnews-articles-{config.run_date.isoformat()}",
            run_date=config.run_date,
            tickers=TICKERS,
            limit=6,
        )
    )

    candlecharts = CandlechartsMarketDataProvider(
        cache=html_cache,
        allow_live=True,
        now=lambda: fetched_at,
    ).fetch_daily_candles(
        MarketDataRequest(
            request_id=f"live-scrape-candlecharts-aapl-{config.run_date.isoformat()}",
            run_date=config.run_date,
            tickers=("AAPL",),
        )
    )

    x_provider = XRecentSearchProvider(
        bearer_token=_env_value(_X_BEARER_TOKEN_ENV),
        cache=json_cache,
        now=lambda: fetched_at,
    )
    x_results = tuple(
        x_provider.fetch_social_posts(
            EvidenceRequest(
                request_id=f"live-scrape-x-{ticker.lower()}-{config.run_date.isoformat()}",
                run_date=config.run_date,
                tickers=(ticker,),
            )
        )
        for ticker in TICKERS
    )

    return cast(
        tuple[ProviderResult[object], ...],
        (
            reddit_discovery,
            reddit_discussion,
            ap_articles,
            candlecharts,
            *x_results,
        ),
    )


def _provider_results(
    config: RunConfig,
    fetched_at: datetime,
) -> tuple[ProviderResult[object], ...]:
    reddit_provider = RedditPublicPageProvider(
        transport=StaticRedditHtmlTransport(
            {
                _REDDIT_URL: _REDDIT_TICKER_CARD_HTML,
                _REDDIT_POST_URL: _REDDIT_DISCUSSION_HTML,
                "https://www.reddit.com/r/wallstreetbets/stale/": _REDDIT_STALE_TICKER_CARD_HTML,
            }
        ),
        discussion_urls=(_REDDIT_POST_URL,),
        now=lambda: fetched_at,
    )
    reddit_discovery = reddit_provider.discover_tickers(
        TickerDiscoveryRequest(
            request_id=f"scrape-reddit-discovery-{config.run_date.isoformat()}",
            run_date=config.run_date,
            source_url=_REDDIT_URL,
            query="r/wallstreetbets Devvit daily ticker card",
        )
    )
    reddit_discussion = reddit_provider.fetch_discussion(
        EvidenceRequest(
            request_id=f"scrape-reddit-discussion-{config.run_date.isoformat()}",
            run_date=config.run_date,
            tickers=TICKERS,
            query=" OR ".join(TICKERS),
            include_posts=True,
            include_comments=True,
        )
    )
    reddit_blocked = reddit_provider.discover_tickers(
        TickerDiscoveryRequest(
            request_id=f"scrape-reddit-blocked-{config.run_date.isoformat()}",
            run_date=config.run_date,
            source_url="https://www.reddit.com/r/wallstreetbets/.json",
        )
    )
    reddit_stale = reddit_provider.discover_tickers(
        TickerDiscoveryRequest(
            request_id=f"scrape-reddit-stale-{config.run_date.isoformat()}",
            run_date=config.run_date,
            source_url="https://www.reddit.com/r/wallstreetbets/stale/",
        )
    )

    ap_provider = APNewsProvider(
        transport=_StaticHtmlTransport(
            {
                "hub/financial-markets": _AP_HUB_HTML,
                "markets-megacap-stocks": _AP_ARTICLE_HTML,
            }
        ),
        now=lambda: fetched_at,
    )
    ap_articles = ap_provider.fetch_articles(
        EvidenceRequest(
            request_id=f"scrape-apnews-articles-{config.run_date.isoformat()}",
            run_date=config.run_date,
            tickers=TICKERS,
            limit=6,
        )
    )
    ap_drift = APNewsProvider(
        transport=_StaticHtmlTransport({"hub/financial-markets": _AP_HUB_DRIFT_HTML}),
        now=lambda: fetched_at,
    ).fetch_articles(
        EvidenceRequest(
            request_id=f"scrape-apnews-drift-{config.run_date.isoformat()}",
            run_date=config.run_date,
            tickers=("TSLA",),
        )
    )

    candlecharts = CandlechartsMarketDataProvider(
        html=_CANDLECHARTS_WIDGET_ONLY_HTML,
        now=lambda: fetched_at,
    ).fetch_daily_candles(
        MarketDataRequest(
            request_id=f"scrape-candlecharts-{config.run_date.isoformat()}",
            run_date=config.run_date,
            tickers=("TSLA",),
        )
    )

    x_results = tuple(
        XRecentSearchProvider(
            bearer_token="fixture-token",
            transport=_FixtureXTransport(),
            now=lambda: fetched_at,
        ).fetch_social_posts(
            EvidenceRequest(
                request_id=f"scrape-x-{ticker.lower()}-{config.run_date.isoformat()}",
                run_date=config.run_date,
                tickers=(ticker,),
            )
        )
        for ticker in TICKERS
    )
    x_missing_credentials = XRecentSearchProvider(
        transport=_FixtureXTransport(),
        now=lambda: fetched_at,
    ).fetch_social_posts(
        EvidenceRequest(
            request_id=f"scrape-x-missing-credentials-{config.run_date.isoformat()}",
            run_date=config.run_date,
            tickers=("AAPL",),
        )
    )
    x_rate_limited = XRecentSearchProvider(
        bearer_token="fixture-token",
        transport=_QuotaLimitedXTransport(),
        now=lambda: fetched_at,
    ).fetch_social_posts(
        EvidenceRequest(
            request_id=f"scrape-x-quota-{config.run_date.isoformat()}",
            run_date=config.run_date,
            tickers=("AAPL",),
        )
    )

    base_results = cast(
        tuple[ProviderResult[object], ...],
        (
            reddit_discovery,
            reddit_discussion,
            reddit_blocked,
            reddit_stale,
            ap_articles,
            ap_drift,
            candlecharts,
            *x_results,
            x_missing_credentials,
            x_rate_limited,
        ),
    )
    provider_evidence = _provider_evidence(base_results)
    fundamental_agent = _fundamental_agent_result(
        config=config,
        fetched_at=fetched_at,
        evidence=provider_evidence,
    )
    return (*base_results, cast(ProviderResult[object], fundamental_agent))


def _fundamental_agent_result(
    *,
    config: RunConfig,
    fetched_at: datetime,
    evidence: tuple[SourceEvidence, ...],
) -> ProviderResult[FundamentalNlpAnalysisResponse]:
    request_id = f"scrape-fundamental-agent-tsla-{config.run_date.isoformat()}"
    tsla_evidence = _first_evidence_for_ticker(evidence, "TSLA")
    if tsla_evidence is None:
        return FixtureFundamentalAgentProvider(
            payload=None, fetched_at=fetched_at
        ).analyze_fundamentals(
            FundamentalNlpAnalysisRequest(
                request_id=request_id,
                ticker="TSLA",
                run_date=config.run_date,
                as_of=fetched_at,
                evidence=(),
                prompt_version="scrape-fixture-fundamental-agent-v1",
                schema_version="fundamental-agent-response-v1",
            )
        )
    quote = _agent_quote(tsla_evidence)
    payload = {
        "request_id": request_id,
        "ticker": "TSLA",
        "as_of": fetched_at,
        "summary": (
            "Fixture fundamental agent treats TSLA evidence as catalyst-aware but still "
            "speculative."
        ),
        "signal": "mixed",
        "confidence": 0.62,
        "source_evidence_ids": [tsla_evidence.evidence_id],
        "citations": [
            {
                "evidence_id": tsla_evidence.evidence_id,
                "quote": quote,
                "relevance": 0.84,
            }
        ],
        "claims": [
            {
                "claim_id": "claim-tsla-discussion-catalyst",
                "claim_type": "observed",
                "text": "TSLA discussion mentions catalyst watch.",
                "citations": [
                    {
                        "evidence_id": tsla_evidence.evidence_id,
                        "quote": quote,
                        "relevance": 0.84,
                    }
                ],
                "confidence": 0.72,
            }
        ],
        "risks": [
            {
                "risk_id": "risk-tsla-speculative-social-evidence",
                "text": "Social evidence is speculative and can reverse quickly.",
                "severity": "medium",
                "citations": [
                    {
                        "evidence_id": tsla_evidence.evidence_id,
                        "quote": quote,
                        "relevance": 0.70,
                    }
                ],
            }
        ],
        "assumptions": ("Fixture agent output is deterministic and citation-bound.",),
        "confidence_inputs": {
            "claim_count": 1,
            "cited_source_count": 1,
            "stale_source_count": 0,
        },
    }
    provider = FixtureFundamentalAgentProvider(
        payload=payload,
        provider_name="fundamental-agent",
        fetched_at=fetched_at,
    )
    request = FundamentalNlpAnalysisRequest(
        request_id=request_id,
        ticker="TSLA",
        run_date=config.run_date,
        as_of=fetched_at,
        evidence=(tsla_evidence,),
        prompt_version="scrape-fixture-fundamental-agent-v1",
        schema_version="fundamental-agent-response-v1",
        focus_areas=("valuation", "profitability", "growth", "risk"),
    )
    return provider.analyze_fundamentals(request)


def _first_evidence_for_ticker(
    evidence: tuple[SourceEvidence, ...],
    ticker: str,
) -> SourceEvidence | None:
    normalized = ticker.upper()
    for record in evidence:
        if record.ticker == normalized or normalized in record.matched_tickers:
            return record
    return None


def _agent_quote(evidence: SourceEvidence) -> str:
    if "catalyst watch" in evidence.text:
        return "catalyst watch"
    return evidence.text[:80]


def _provider_evidence(results: tuple[ProviderResult[object], ...]) -> tuple[SourceEvidence, ...]:
    evidence: list[SourceEvidence] = []
    seen_ids: set[str] = set()
    for result in results:
        data = result.data
        if not isinstance(data, tuple):
            continue
        for item in data:
            if isinstance(item, SourceEvidence) and item.evidence_id not in seen_ids:
                seen_ids.add(item.evidence_id)
                evidence.append(item)
    return tuple(evidence)


def _provider_evidence_refs(
    evidence: tuple[SourceEvidence, ...],
    ticker: str,
    *,
    limit: int = 5,
) -> tuple[EvidenceReference, ...]:
    matching_records = tuple(
        record
        for record in evidence
        if record.ticker == ticker.upper() or ticker.upper() in record.matched_tickers
    )
    display_records = tuple(
        record for record in matching_records if not _looks_promotional_social_noise(record)
    )
    refs: list[EvidenceReference] = []
    for record in display_records:
        refs.append(
            EvidenceReference(
                evidence_id=record.evidence_id,
                quote=_quote(record.text),
                relevance=0.70,
            )
        )
        if len(refs) >= limit:
            break
    return tuple(refs)


def _looks_promotional_social_noise(evidence: SourceEvidence) -> bool:
    if evidence.provenance.provider_name != "x-recent-search":
        return False
    lowered = evidence.text.lower()
    promotional_phrases = (
        "recommend stock blogger",
        "excellent stock expert",
        "financial mentor",
        "profit easily",
        "profitable stock picks",
        "profitable results",
        "good returns",
        "make substantial profits",
        "want to make money in the stock market",
        "following his advice",
        "following her advice",
        "highly recommend checking",
        "every trader should follow",
        "daily bullish signals",
        "bullish signals on x",
        "proven us stock trader",
        "high-probability long calls",
        "top market analyst",
        "experienced market analyst",
        "long plays daily",
        "stock recommendations",
        "steady green candles",
        "solid gains",
        "real-time trades",
        "consistent profits",
        "easy strategies",
        "most consistent accounts",
        "account that actually delivers",
        "go check out",
        "live calls",
        "follow him",
        "follow her",
    )
    if any(phrase in lowered for phrase in promotional_phrases):
        return True
    return "@" in lowered and any(
        marker in lowered
        for marker in ("recommend", "follow", "profit", "gains", "mentor", "trader")
    )


def _live_observed_summary(ticker: str, evidence: tuple[SourceEvidence, ...]) -> str:
    evidence_count = sum(
        1
        for record in evidence
        if record.ticker == ticker or ticker.upper() in record.matched_tickers
    )
    if evidence_count:
        return (
            f"Live scrape mode collected {evidence_count} source evidence records for {ticker}; "
            "see the evidence references and provider-results audit artifact for provenance."
        )
    return (
        f"Live scrape mode did not collect ticker-matched evidence for {ticker}; provider "
        "warnings explain missing, blocked, stale, or unavailable sources."
    )


def _quote(text: str) -> str:
    stripped = " ".join(text.split())
    return stripped[:160] if stripped else ""


def _scrape_report(
    *,
    offline_bundle: OfflineFixtureBundle,
    config: RunConfig,
    run_id: str,
    source_profile: str,
    generated_at: datetime,
    provider_results: tuple[ProviderResult[object], ...],
    provider_health: tuple[ProviderHealth, ...],
    provider_evidence: tuple[SourceEvidence, ...],
    apply_fixture_sidecars: bool,
) -> DailyReport:
    live_providers = source_profile == "live"
    source_health_names = tuple(health.provider_name for health in provider_health)
    stale_names = _provider_names_with_status_or_warning(
        provider_health,
        statuses={ProviderStatus.STALE},
        warning_code=WarningCode.STALE_DATA,
    )
    missing_names = _provider_names_with_status_or_warning(
        provider_health,
        statuses={ProviderStatus.UNCONFIGURED, ProviderStatus.UNAUTHORIZED},
        warning_code=WarningCode.MISSING_CREDENTIALS,
    )
    command_args = dict(offline_bundle.report.command_args)
    command_args["source_mode"] = "scrape"
    command_args["offline"] = False
    command_args["live_providers"] = live_providers
    report = offline_bundle.report
    fundamental_agent_result = _fundamental_agent_result_from_results(provider_results)
    ml_signal = (
        _fixture_ml_signal(generated_at)
        if apply_fixture_sidecars and config.ml_artifact is None
        else None
    )
    summary_label = (
        "live Reddit/AP public HTML providers, the Candlecharts feasibility probe, and "
        "the X recent-search API"
        if live_providers
        else "Reddit/AP/X fixtures and Candlecharts feasibility probes"
    )
    freshness_summary = (
        "Experimental scrape source mode called live public/API providers and recorded "
        "provider-level degraded results for unavailable, stale, blocked, empty, or "
        "unconfigured sources."
        if live_providers
        else (
            "Experimental scrape source mode used deterministic Reddit/AP/X fixtures, "
            "a Candlecharts widget-only probe, and degraded-provider probes for missing "
            "credentials, quota, stale data, blocked scraping, and markup drift."
        )
    )
    ticker_sections = tuple(
        section.model_copy(
            update={
                "technical_analysis": (
                    apply_technical_ml_signal(section.technical_analysis, ml_signal)
                    if ml_signal is not None
                    and section.ticker == "TSLA"
                    and section.technical_analysis is not None
                    else section.technical_analysis
                ),
                "fundamental_analysis": (
                    apply_fundamental_agent_result(
                        section.fundamental_analysis,
                        fundamental_agent_result,
                    )
                    if section.ticker == "TSLA"
                    and section.fundamental_analysis is not None
                    and fundamental_agent_result is not None
                    else section.fundamental_analysis
                ),
                "social_news_summary": (
                    f"Experimental scrape source mode wired {summary_label} for "
                    f"{section.ticker}. Provider warnings are reported separately."
                ),
                "data_quality": {
                    **dict(section.data_quality),
                    "source_mode": "scrape",
                    "live_providers": live_providers,
                    "provider_names": list(source_health_names),
                    "provider_result_artifact": "provider-results",
                    "ml_signal": (
                        "fixture_sidecar"
                        if ml_signal is not None and section.ticker == "TSLA"
                        else None
                    ),
                    "fundamental_agent": (
                        "fixture_sidecar"
                        if apply_fixture_sidecars and section.ticker == "TSLA"
                        else None
                    ),
                },
            }
        )
        for section in report.ticker_sections
    )
    if live_providers:
        ticker_sections = tuple(
            section.model_copy(
                update={
                    "observed_discussion_summary": _live_observed_summary(
                        section.ticker,
                        provider_evidence,
                    ),
                    "strategy_clusters": (),
                    "technical_analysis": None,
                    "fundamental_analysis": None,
                    "sector_context": None,
                    "macro_context": None,
                    "opportunity_notes": (
                        "Live provider orchestration collected source evidence only; live "
                        "strategy extraction, analysis, and scoring remain disabled for this "
                        "mode.",
                    ),
                    "recommendation_ids": (),
                    "evidence": _provider_evidence_refs(provider_evidence, section.ticker),
                }
            )
            for section in ticker_sections
        )
    evidence_sources = (
        provider_evidence if live_providers else report.evidence_sources + provider_evidence
    )
    trade_candidates = () if live_providers else report.trade_candidates
    no_trade_summary = (
        "Live scrape mode collected provider evidence and audit metadata only; no qualified "
        "trades are produced from this run."
        if live_providers
        else report.no_trade_summary
    )
    return report.model_copy(
        update={
            "run_id": run_id,
            "generated_at": generated_at,
            "config_hash": f"scrape-{source_profile}-{config.run_date.isoformat()}",
            "command_args": command_args,
            "data_freshness": DataFreshnessSummary(
                as_of=generated_at,
                summary=freshness_summary,
                stale_provider_names=stale_names,
                missing_provider_names=missing_names,
            ),
            "provider_health": provider_health,
            "evidence_sources": evidence_sources,
            "ticker_sections": ticker_sections,
            "trade_candidates": trade_candidates,
            "no_trade_summary": no_trade_summary,
            "audit_manifest": None,
        }
    )


def _fundamental_agent_result_from_results(
    provider_results: tuple[ProviderResult[object], ...],
) -> ProviderResult[FundamentalNlpAnalysisResponse] | None:
    for result in provider_results:
        if result.provider_name == "fundamental-agent":
            return cast(ProviderResult[FundamentalNlpAnalysisResponse], result)
    return None


def _fixture_ml_signal(generated_at: datetime) -> TechnicalMlSignal:
    return TechnicalMlSignal(
        model_hash="fixture-technical-model-tsla-v1",
        dataset_hash="fixture-technical-dataset-tsla-v1",
        as_of=generated_at,
        feature_end=generated_at,
        prediction_horizon_sessions=1,
        probability_positive=0.61,
        calibrated_confidence=0.31,
        signal=AnalysisSignal.SUPPORTS,
        status="usable",
        freshness_status=FreshnessStatus.FRESH,
        validation_accuracy=0.58,
        validation_brier_score=0.21,
        limitations=(
            "Fixture ML signal is local research metadata and cannot qualify a trade alone.",
        ),
        metadata={"source_mode": "scrape", "fixture": True},
    )


def _scrape_audit_payloads(
    *,
    offline_bundle: OfflineFixtureBundle,
    report: DailyReport,
    provider_results: tuple[ProviderResult[object], ...],
    provider_evidence: tuple[SourceEvidence, ...],
    source_profile: str,
) -> dict[str, JsonObject]:
    payloads: dict[str, JsonObject] = {
        filename: cast(JsonObject, dict(payload))
        for filename, payload in offline_bundle.audit_payloads.items()
    }
    for payload in payloads.values():
        if "run_id" in payload:
            payload["run_id"] = report.run_id
    normalized = payloads.get("normalized-evidence.json")
    if normalized is not None:
        records_value = normalized.get("records")
        records: list[JsonValue] = (
            []
            if source_profile == "live"
            else list(records_value)
            if isinstance(records_value, list)
            else []
        )
        records.extend(
            cast(JsonValue, evidence.model_dump(mode="json")) for evidence in provider_evidence
        )
        normalized["records"] = records
    raw_snapshots = payloads.get("raw-snapshots.json")
    if raw_snapshots is not None:
        records_value = raw_snapshots.get("records")
        raw_records: list[JsonValue] = (
            []
            if source_profile == "live"
            else list(records_value)
            if isinstance(records_value, list)
            else []
        )
        raw_records.extend(
            cast(JsonValue, record)
            for record in _raw_snapshot_records(provider_results, source_profile=source_profile)
        )
        raw_snapshots["records"] = raw_records
    if source_profile == "live":
        for filename in (
            "extracted-strategies.json",
            "analysis-contexts.json",
            "scoring-inputs.json",
        ):
            live_payload = payloads.get(filename)
            if live_payload is not None:
                live_payload["records"] = []
    _refresh_analysis_context_payload(payloads, report)
    payloads["provider-results.json"] = _provider_results_payload(
        report.run_id,
        provider_results,
        source_profile=source_profile,
    )
    return payloads


def _refresh_analysis_context_payload(
    payloads: dict[str, JsonObject],
    report: DailyReport,
) -> None:
    analysis_contexts = payloads.get("analysis-contexts.json")
    if analysis_contexts is None:
        return
    records_value = analysis_contexts.get("records")
    if not isinstance(records_value, list):
        return
    sections_by_ticker = {section.ticker: section for section in report.ticker_sections}
    refreshed_records: list[JsonValue] = []
    for record in records_value:
        if not isinstance(record, dict):
            refreshed_records.append(record)
            continue
        ticker = record.get("ticker")
        if not isinstance(ticker, str) or ticker not in sections_by_ticker:
            refreshed_records.append(cast(JsonValue, record))
            continue
        section = sections_by_ticker[ticker]
        refreshed_record: JsonObject = cast(JsonObject, dict(record))
        if section.technical_analysis is not None:
            refreshed_record["technical"] = cast(
                JsonValue,
                section.technical_analysis.model_dump(mode="json"),
            )
        if section.fundamental_analysis is not None:
            refreshed_record["fundamental"] = cast(
                JsonValue,
                section.fundamental_analysis.model_dump(mode="json"),
            )
        if section.sector_context is not None:
            refreshed_record["sector"] = cast(
                JsonValue,
                section.sector_context.model_dump(mode="json"),
            )
        if section.macro_context is not None:
            refreshed_record["macro"] = cast(
                JsonValue,
                section.macro_context.model_dump(mode="json"),
            )
        refreshed_records.append(cast(JsonValue, refreshed_record))
    analysis_contexts["records"] = refreshed_records


def _scrape_audit_manifest(
    *,
    offline_manifest: AuditManifest,
    config: RunConfig,
    run_id: str,
    source_profile: str,
    generated_at: datetime,
    provider_health: tuple[ProviderHealth, ...],
    provider_results: tuple[ProviderResult[object], ...],
    audit_payloads: dict[str, JsonObject],
) -> AuditManifest:
    audit_dir = config.output_dir / config.run_date.isoformat() / "audit"
    live_providers = source_profile == "live"
    provider_payload = _provider_results_payload(
        run_id,
        provider_results,
        source_profile=source_profile,
    )
    provider_artifact = AuditArtifact(
        artifact_id="provider-results",
        artifact_type="provider_result",
        path=(audit_dir / "provider-results.json").as_posix(),
        created_at=generated_at,
        produced_by="scrape-source-orchestration",
        sha256=json_payload_sha256(provider_payload),
        record_count=_record_count(provider_payload),
        metadata={
            "source_mode": "scrape",
            "source_profile": source_profile,
            "fixture": source_profile == "fixture",
            "live_providers": live_providers,
        },
    )
    command_args = dict(offline_manifest.command_args)
    command_args["source_mode"] = "scrape"
    command_args["offline"] = False
    command_args["live_providers"] = live_providers
    artifacts = tuple(
        _with_final_payload_hash(artifact, audit_payloads)
        for artifact in offline_manifest.artifacts
    )
    return offline_manifest.model_copy(
        update={
            "run_id": run_id,
            "created_at": generated_at,
            "artifacts": (*artifacts, provider_artifact),
            "provider_run_ids": tuple(
                f"{health.provider_name}:{config.run_date.isoformat()}"
                for health in provider_health
            ),
            "config_hash": f"scrape-{source_profile}-{config.run_date.isoformat()}",
            "command_args": command_args,
        }
    )


def _with_final_payload_hash(
    artifact: AuditArtifact,
    audit_payloads: dict[str, JsonObject],
) -> AuditArtifact:
    filename_by_artifact_id = {
        "raw-snapshots": "raw-snapshots.json",
        "normalized-evidence": "normalized-evidence.json",
        "extracted-strategies": "extracted-strategies.json",
        "analysis-contexts": "analysis-contexts.json",
        "scoring-inputs": "scoring-inputs.json",
    }
    filename = filename_by_artifact_id.get(artifact.artifact_id)
    if filename is None:
        return artifact
    payload = audit_payloads[filename]
    return artifact.model_copy(
        update={
            "sha256": json_payload_sha256(payload),
            "record_count": _record_count(payload),
        }
    )


def _record_count(payload: JsonObject) -> int | None:
    records = payload.get("records")
    return len(records) if isinstance(records, list) else None


def _provider_results_payload(
    run_id: str,
    provider_results: tuple[ProviderResult[object], ...],
    *,
    source_profile: str,
) -> JsonObject:
    return {
        "schema_version": "audit.provider_results.v1",
        "run_id": run_id,
        "source_mode": "scrape",
        "source_profile": source_profile,
        "live_providers": source_profile == "live",
        "records": [
            cast(JsonObject, result.model_dump(mode="json")) for result in provider_results
        ],
    }


def _raw_snapshot_records(
    results: tuple[ProviderResult[object], ...],
    *,
    source_profile: str,
) -> list[JsonObject]:
    records: list[JsonObject] = []
    for result in results:
        if not result.raw_snapshot_id:
            continue
        records.append(
            {
                "raw_snapshot_id": result.raw_snapshot_id,
                "provider_name": result.provider_name,
                "source_kind": "provider_result",
                "content_type": "application/json",
                "payload": {
                    "status": result.status.value,
                    "request_id": result.request.request_id,
                    "cache_key": result.cache_key,
                    "warning_codes": [warning.code.value for warning in result.warnings],
                },
                "provider_metadata": {
                    "fixture": source_profile == "fixture",
                    "live_providers": source_profile == "live",
                    "source_mode": "scrape",
                    "source_profile": source_profile,
                },
            }
        )
    return records


def _provider_names_with_status_or_warning(
    provider_health: tuple[ProviderHealth, ...],
    *,
    statuses: set[ProviderStatus],
    warning_code: WarningCode,
) -> tuple[str, ...]:
    names: list[str] = []
    for health in provider_health:
        has_status_or_warning = health.status in statuses or any(
            warning.code == warning_code for warning in health.warnings
        )
        if has_status_or_warning and health.provider_name not in names:
            names.append(health.provider_name)
    return tuple(names)


def _ticker_from_x_query(url: str) -> str:
    query_values = parse_qs(urlsplit(url).query).get("query", ["$TSLA"])
    query = unquote_plus(query_values[0])
    for part in query.split():
        if part.startswith("$") and len(part) > 1:
            return part.removeprefix("$").upper()
    return "TSLA"


def _html_cache(config: RunConfig) -> HtmlCache | None:
    if config.cache_dir is None:
        return None
    return HtmlCache(config.cache_dir / "html")


def _json_cache(config: RunConfig) -> ProviderCache | None:
    if config.cache_dir is None:
        return None
    return ProviderCache(config.cache_dir / "json")


def _env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


_REDDIT_TICKER_CARD_HTML = """
<main data-snapshot-observed-at="2026-05-11T17:45:00Z">
  <div id="ticker-container-tsla">TSLA</div>
  <div id="ticker-container-nvda">NVDA</div>
  <div id="ticker-container-amd">AMD</div>
  <div id="ticker-container-aapl">AAPL</div>
  <div id="ticker-container-mu">MU</div>
  <div id="ticker-container-spy">SPY</div>
</main>
"""

_REDDIT_STALE_TICKER_CARD_HTML = """
<main data-snapshot-observed-at="2026-05-01T17:45:00Z">
  <div id="ticker-container-tsla">TSLA</div>
  <div id="ticker-container-nvda">NVDA</div>
  <div id="ticker-container-amd">AMD</div>
  <div id="ticker-container-aapl">AAPL</div>
  <div id="ticker-container-mu">MU</div>
  <div id="ticker-container-spy">SPY</div>
</main>
"""

_REDDIT_DISCUSSION_HTML = """
<main>
  <shreddit-post data-reddit-id="t3_scrape001" author="FixtureTrader"
      created-timestamp="2026-05-11T17:00:00Z" score="188"
      permalink="/r/wallstreetbets/comments/scrape001/daily_watch/" data-num-comments="44">
    <h1 slot="title">Daily watch: TSLA NVDA AMD AAPL MU and SPY setups</h1>
    <div slot="text-body">$TSLA and $NVDA lead discussion; $AAPL and $MU need confirmation.</div>
  </shreddit-post>
  <shreddit-comment data-reddit-id="t1_scrape002" author="FixtureCommenter"
      created-timestamp="2026-05-11T17:15:00Z" score="51"
      data-link-id="t3_scrape001" data-parent-id="t3_scrape001"
      permalink="/r/wallstreetbets/comments/scrape001/daily_watch/comment/scrape002/">
    $AMD and $SPY are watch-only unless volume expands.
  </shreddit-comment>
</main>
"""

_AP_HUB_HTML = f"""
<!doctype html>
<html>
  <body>
    <main>
      <a href="{_AP_ARTICLE_URL}">Stocks advance as megacap technology leads markets</a>
    </main>
  </body>
</html>
"""

_AP_ARTICLE_HTML = """
<!doctype html>
<html>
  <head>
    <title>Stocks advance as megacap technology leads markets</title>
    <link rel="canonical" href="https://apnews.com/article/markets-megacap-stocks-2026-05-11">
    <script type="application/ld+json">
    {
      "@type": "NewsArticle",
      "headline": "Stocks advance as megacap technology leads markets",
      "description": "Tesla, Nvidia, AMD, Apple and Micron were active Monday.",
      "articleBody": "Tesla, Nvidia, AMD, Apple and Micron moved while SPY tracked risk appetite.",
      "datePublished": "2026-05-11T16:30:00Z",
      "author": {"name": "AP Markets Fixture"},
      "publisher": {"name": "AP Business"},
      "url": "https://apnews.com/article/markets-megacap-stocks-2026-05-11"
    }
    </script>
  </head>
  <body><h1>Stocks advance as megacap technology leads markets</h1></body>
</html>
"""

_AP_HUB_DRIFT_HTML = """
<!doctype html>
<html>
  <body>
    <main><p>Financial markets hub shell without public article links.</p></main>
  </body>
</html>
"""

_CANDLECHARTS_WIDGET_ONLY_HTML = """
<!doctype html>
<html>
  <body>
    <iframe src="https://www.tradingview-widget.com/embed-widget/advanced-chart/"></iframe>
  </body>
</html>
"""


__all__ = ["ScrapeFixtureBundle", "build_live_scrape_bundle", "build_scrape_fixture_bundle"]
