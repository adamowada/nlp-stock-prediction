"""Discover WSB-mentioned symbols and feed them into batch research."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from nlp_stock_prediction.contracts import (
    BatchRunConfig,
    RunConfig,
    WsbBatchRunConfig,
    WsbTrendingDiscoveryReport,
    WsbTrendingStock,
)
from nlp_stock_prediction.orchestration.context import deterministic_generated_at
from nlp_stock_prediction.orchestration.report_bundle import ReportBundle, WsbBatchReportBundle
from nlp_stock_prediction.orchestration.research_batch import generate_batch_research_reports
from nlp_stock_prediction.providers._base import ProviderTransportError, build_cache_key
from nlp_stock_prediction.providers.reddit_scrape import (
    RedditPublicPagePolicy,
    StaticHtmlTransport,
)
from nlp_stock_prediction.providers.scraping import (
    HtmlCache,
    HtmlFetch,
    HtmlTransport,
    UrllibHtmlTransport,
    configured_scrape_user_agent,
    fetch_html,
)
from nlp_stock_prediction.reddit.matching import TickerMatch, find_ticker_matches
from nlp_stock_prediction.reddit.public_html import (
    extract_reddit_discussion_records_from_public_html,
    extract_reddit_search_results_from_public_html,
)

WSB_TRENDING_PROVIDER = "reddit-public-wsb-trending"
DEFAULT_WSB_SOURCE_URL = "https://www.reddit.com/r/wallstreetbets/"

ReportGenerator = Callable[[RunConfig], ReportBundle]

_CASHTAG_SYMBOL_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])\$([A-Za-z][A-Za-z0-9.\-]{0,9})(?![A-Za-z0-9_.-])"
)
_BARE_SYMBOL_RE = re.compile(r"(?<![A-Za-z0-9_.-])([A-Z][A-Z0-9.]{1,5})(?![A-Za-z0-9_.-])")
_NON_TICKER_WORDS = frozenset(
    {
        "ATH",
        "CEO",
        "CFO",
        "DD",
        "ETF",
        "FOMC",
        "GDP",
        "IMO",
        "IPO",
        "LOL",
        "SEC",
        "TLDR",
        "USA",
        "USD",
        "WSB",
        "YOLO",
    }
)


@dataclass(frozen=True)
class _FetchedPage:
    url: str
    html: str
    raw_snapshot_id: str


@dataclass(frozen=True)
class _Mention:
    symbol: str
    match: TickerMatch
    record_id: str
    source_url: str
    text: str
    first_seen_index: int


def generate_wsb_batch_research_reports(
    config: WsbBatchRunConfig,
    *,
    report_generator: ReportGenerator,
    transport: HtmlTransport | None = None,
    now: Callable[[], datetime] | None = None,
) -> WsbBatchReportBundle:
    """Discover WSB trending symbols, then run batch research over the discovered universe."""

    discovery = discover_wsb_trending_stocks(config, transport=transport, now=now)
    discovery_json_path, discovery_markdown_path = _write_discovery_artifacts(
        output_dir=config.output_dir,
        report=discovery,
    )
    symbols = tuple(stock.symbol for stock in discovery.trending_stocks)
    if not symbols:
        raise ValueError("WSB trending discovery did not produce symbols to batch analyze")
    batch_bundle = generate_batch_research_reports(
        BatchRunConfig(
            run_date=config.run_date,
            output_dir=config.output_dir,
            symbols=symbols,
            fixture_dir=config.fixture_dir,
            cache_dir=config.cache_dir,
            offline=config.offline,
            source_mode=config.source_mode,
            live_providers=config.live_providers,
            max_workers=config.max_workers,
        ),
        report_generator=report_generator,
    )
    return WsbBatchReportBundle(
        output_dir=config.output_dir.resolve(),
        discovery_markdown_path=discovery_markdown_path,
        discovery_json_path=discovery_json_path,
        discovery_report=discovery,
        batch_bundle=batch_bundle,
    )


def discover_wsb_trending_stocks(
    config: WsbBatchRunConfig,
    *,
    transport: HtmlTransport | None = None,
    now: Callable[[], datetime] | None = None,
) -> WsbTrendingDiscoveryReport:
    """Discover the most-mentioned WSB symbols from public Reddit HTML."""

    live_requested = config.source_mode == "live" or config.live_providers
    if not config.offline and not live_requested:
        raise ValueError("WSB discovery requires --offline or --live.")
    if config.offline and live_requested:
        raise ValueError(
            "WsbBatchRunConfig cannot request both offline fixtures and live providers."
        )

    fetched_at = (
        deterministic_generated_at(config.run_date)
        if config.offline
        else (now or (lambda: datetime.now(UTC)))()
    )
    active_transport = transport or _default_transport(config)
    cache = _html_cache(config)
    source_url = str(config.source_url)
    warnings: list[str] = []
    fetched_pages: list[_FetchedPage] = []

    source_page = _fetch_reddit_page(
        transport=active_transport,
        url=source_url,
        config=config,
        fetched_at=fetched_at,
        cache=cache,
        source="wsb-source-page",
    )
    fetched_pages.append(source_page)

    source_records = _records_from_page(source_page)
    discussion_urls = _discussion_urls_from_source_page(
        source_page,
        source_url=source_url,
        limit=config.max_discussion_pages,
    )
    for discussion_url in discussion_urls:
        try:
            fetched_pages.append(
                _fetch_reddit_page(
                    transport=active_transport,
                    url=discussion_url,
                    config=config,
                    fetched_at=fetched_at,
                    cache=cache,
                    source="wsb-discussion-page",
                )
            )
        except ProviderTransportError as exc:
            warnings.append(f"Failed to fetch WSB discussion page {discussion_url}: {exc}")

    records = [*source_records]
    for page in fetched_pages[1:]:
        records.extend(_records_from_page(page))

    mentions = _mentions_from_records(records)
    ranked = _rank_mentions(mentions, limit=config.limit)
    if len(ranked) < config.limit:
        warnings.append(
            f"WSB trending discovery produced {len(ranked)} symbols for requested limit "
            f"{config.limit}."
        )
    if not mentions:
        warnings.append("No high-confidence WSB ticker mentions were discovered.")

    return WsbTrendingDiscoveryReport(
        generated_at=fetched_at,
        run_date=config.run_date,
        source_url=source_url,
        provider_name=WSB_TRENDING_PROVIDER,
        mode="offline_fixture" if config.offline else "live",
        limit=config.limit,
        max_discussion_pages=config.max_discussion_pages,
        trending_stocks=ranked,
        raw_snapshot_ids=tuple(dict.fromkeys(page.raw_snapshot_id for page in fetched_pages)),
        source_urls=tuple(dict.fromkeys(page.url for page in fetched_pages)),
        warnings=tuple(dict.fromkeys(warnings)),
        metadata={
            "source_record_count": len(records),
            "mention_count": len(mentions),
            "discussion_page_count": len(fetched_pages) - 1,
            "ranking_policy": "wallstreetbets_public_mentions_not_trade_advice",
        },
    )


def _default_transport(config: WsbBatchRunConfig) -> HtmlTransport:
    if config.offline:
        fixture_root = _fixture_root(config.fixture_dir)
        source = fixture_root / "tests" / "fixtures" / "reddit" / "public_wsb_trending.html"
        discussion_one = (
            fixture_root / "tests" / "fixtures" / "reddit" / "public_wsb_trending_post_1.html"
        )
        discussion_two = (
            fixture_root / "tests" / "fixtures" / "reddit" / "public_wsb_trending_post_2.html"
        )
        source_html = source.read_text(encoding="utf-8")
        return StaticHtmlTransport(
            {
                DEFAULT_WSB_SOURCE_URL: source_html,
                str(config.source_url): source_html,
                "wsbtrending001/daily_moves": discussion_one.read_text(encoding="utf-8"),
                "wsbtrending002/earnings_watch": discussion_two.read_text(encoding="utf-8"),
            }
        )
    return UrllibHtmlTransport()


def _fixture_root(path: Path | None) -> Path:
    if path is None:
        return Path(__file__).resolve().parents[3]
    resolved = path.resolve()
    if (resolved / "tests" / "fixtures").exists():
        return resolved
    if resolved.name == "fixtures" and resolved.parent.name == "tests":
        return resolved.parent.parent
    raise ValueError("--fixture-dir must point to the repository root or tests/fixtures")


def _html_cache(config: WsbBatchRunConfig) -> HtmlCache | None:
    if config.cache_dir is None:
        return None
    return HtmlCache(config.cache_dir.resolve() / "html")


def _fetch_reddit_page(
    *,
    transport: HtmlTransport,
    url: str,
    config: WsbBatchRunConfig,
    fetched_at: datetime,
    cache: HtmlCache | None,
    source: str,
) -> _FetchedPage:
    decision = RedditPublicPagePolicy().evaluate(url)
    if not decision.allowed:
        raise ValueError(
            f"WSB discovery source is not allowed by Reddit page policy: {decision.reason}"
        )
    fetch = fetch_html(
        transport=transport,
        url=url,
        run_date=config.run_date,
        ticker=None,
        source=source,
        cache_key=build_cache_key(
            provider_name=WSB_TRENDING_PROVIDER,
            source=source,
            run_date=config.run_date,
            url=url,
        ),
        fetched_at=fetched_at,
        cache=cache,
        user_agent=configured_scrape_user_agent(),
    )
    return _FetchedPage(
        url=_canonical_url(fetch),
        html=fetch.html,
        raw_snapshot_id=fetch.raw_snapshot_id,
    )


def _canonical_url(fetch: HtmlFetch) -> str:
    return fetch.canonical_url or fetch.source_url


def _records_from_page(page: _FetchedPage) -> tuple[Mapping[str, object], ...]:
    records = extract_reddit_discussion_records_from_public_html(page.html, source_url=page.url)
    return tuple(record for record in records if _is_wsb_record(record, page.url))


def _discussion_urls_from_source_page(
    page: _FetchedPage,
    *,
    source_url: str,
    limit: int,
) -> tuple[str, ...]:
    if limit <= 0:
        return ()
    urls: list[str] = []
    for record in extract_reddit_search_results_from_public_html(page.html, source_url=source_url):
        permalink = _string(record.get("permalink")) or _string(record.get("url"))
        if permalink is None or _subreddit_from_url(permalink) != "wallstreetbets":
            continue
        urls.append(permalink)
        if len(urls) >= limit:
            break
    return tuple(dict.fromkeys(urls))


def _mentions_from_records(records: list[Mapping[str, object]]) -> tuple[_Mention, ...]:
    mentions: list[_Mention] = []
    for record_index, record in enumerate(records):
        text = _record_text(record)
        if not text:
            continue
        source_url = (
            _string(record.get("permalink"))
            or _string(record.get("url"))
            or _string(record.get("source_url"))
            or DEFAULT_WSB_SOURCE_URL
        )
        record_id = _string(record.get("id")) or f"record-{record_index}"
        for match in _ticker_matches(text):
            mentions.append(
                _Mention(
                    symbol=match.ticker,
                    match=match,
                    record_id=record_id,
                    source_url=source_url,
                    text=text,
                    first_seen_index=len(mentions),
                )
            )
    return tuple(mentions)


def _ticker_matches(text: str) -> tuple[TickerMatch, ...]:
    candidates: list[str] = []
    for match in _CASHTAG_SYMBOL_RE.finditer(text):
        candidates.append(match.group(1))
    for match in _BARE_SYMBOL_RE.finditer(text):
        candidates.append(match.group(1))
    candidate_symbols = tuple(
        dict.fromkeys(_normalize_symbol(symbol) for symbol in candidates if _is_ticker_like(symbol))
    )
    return find_ticker_matches(text, candidate_symbols)


def _rank_mentions(mentions: tuple[_Mention, ...], *, limit: int) -> tuple[WsbTrendingStock, ...]:
    by_symbol: dict[str, list[_Mention]] = defaultdict(list)
    for mention in mentions:
        by_symbol[mention.symbol].append(mention)

    sorted_symbols = sorted(
        by_symbol,
        key=lambda symbol: (
            -len(by_symbol[symbol]),
            -len({mention.record_id for mention in by_symbol[symbol]}),
            -sum(1 for mention in by_symbol[symbol] if mention.match.is_cashtag),
            min(mention.first_seen_index for mention in by_symbol[symbol]),
            symbol,
        ),
    )
    stocks: list[WsbTrendingStock] = []
    for rank, symbol in enumerate(sorted_symbols[:limit], start=1):
        symbol_mentions = by_symbol[symbol]
        source_urls = tuple(dict.fromkeys(mention.source_url for mention in symbol_mentions))
        snippets = tuple(
            dict.fromkeys(_snippet(mention.text, mention.match) for mention in symbol_mentions)
        )[:3]
        stocks.append(
            WsbTrendingStock(
                symbol=symbol,
                rank=rank,
                mention_count=len(symbol_mentions),
                cashtag_count=sum(1 for mention in symbol_mentions if mention.match.is_cashtag),
                source_record_count=len({mention.record_id for mention in symbol_mentions}),
                source_urls=source_urls,
                snippets=snippets,
                metadata={"first_seen_index": min(m.first_seen_index for m in symbol_mentions)},
            )
        )
    return tuple(stocks)


def _record_text(record: Mapping[str, object]) -> str:
    parts = (
        _string(record.get("title")),
        _string(record.get("selftext")),
        _string(record.get("body")),
    )
    return "\n\n".join(part for part in parts if part)


def _is_wsb_record(record: Mapping[str, object], source_url: str) -> bool:
    subreddit = _string(record.get("subreddit")) or _subreddit_from_url(source_url)
    return subreddit == "wallstreetbets"


def _is_ticker_like(symbol: str) -> bool:
    normalized = _normalize_symbol(symbol)
    if normalized in _NON_TICKER_WORDS:
        return False
    if len(normalized.replace(".", "")) > 5:
        return False
    return bool(normalized)


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().removeprefix("$").upper()


def _subreddit_from_url(value: str | None) -> str | None:
    if value is None:
        return None
    parts = [part for part in urlsplit(value).path.split("/") if part]
    for index, part in enumerate(parts):
        if part.lower() == "r" and index + 1 < len(parts):
            return parts[index + 1].lower()
    return None


def _string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _snippet(text: str, match: TickerMatch) -> str:
    start = max(0, match.start_char - 48)
    end = min(len(text), match.end_char + 96)
    return " ".join(text[start:end].split())


def _write_discovery_artifacts(
    *,
    output_dir: Path,
    report: WsbTrendingDiscoveryReport,
) -> tuple[Path, Path]:
    discovery_dir = output_dir.resolve() / report.run_date.isoformat() / "wsb-trending"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    json_path = discovery_dir / "discovery.json"
    markdown_path = discovery_dir / "discovery.md"
    payload = report.model_dump(mode="json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_discovery_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _render_discovery_markdown(report: WsbTrendingDiscoveryReport) -> str:
    lines = [
        "# WSB Trending Stock Discovery",
        "",
        (
            "This ranks public r/wallstreetbets ticker mentions for follow-up research. "
            "It is not a trading instruction, position-sizing output, or recommendation."
        ),
        "",
        f"- Report date: {report.run_date.isoformat()}",
        f"- Mode: {report.mode}",
        f"- Source: {report.source_url}",
        f"- Generated at: {report.generated_at.isoformat()}",
        "",
        "| Rank | Symbol | Mentions | Cashtags | Source Records |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for stock in report.trending_stocks:
        lines.append(
            "| "
            f"{stock.rank} | "
            f"{stock.symbol} | "
            f"{stock.mention_count} | "
            f"{stock.cashtag_count} | "
            f"{stock.source_record_count} |"
        )
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DEFAULT_WSB_SOURCE_URL",
    "WSB_TRENDING_PROVIDER",
    "discover_wsb_trending_stocks",
    "generate_wsb_batch_research_reports",
]
