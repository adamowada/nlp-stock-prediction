from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TypeVar, cast

import pytest

from nlp_stock_prediction.contracts import (
    EvidenceRequest,
    FundamentalsProvider,
    FundamentalsRequest,
    FundamentalsSnapshot,
    MacroProvider,
    MacroRequest,
    MacroSnapshot,
    MarketDataProvider,
    MarketDataRequest,
    MarketSnapshot,
    NewsProvider,
    PriceBar,
    ProviderHealth,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    RedditProvider,
    SourceEvidence,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.orchestration import ResearchService
from nlp_stock_prediction.orchestration.research_universe_discovery import (
    ResearchLiveSymbolUniverseProvider,
)
from nlp_stock_prediction.providers.reddit_scrape import RedditPublicPageProvider
from nlp_stock_prediction.providers.scraping import HtmlCache, configured_scrape_user_agent

ALLOW_LIVE_ENV = "NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS"
LIVE_REDDIT_DISCUSSION_URL_ENV = "NLP_STOCK_PREDICTION_LIVE_REDDIT_DISCUSSION_URL"
LIVE_REDDIT_SYMBOL_ENV = "NLP_STOCK_PREDICTION_LIVE_REDDIT_SYMBOL"
LIVE_USER_AGENT_ENV = "NLP_STOCK_PREDICTION_LIVE_USER_AGENT"

REPO_ROOT = Path(__file__).resolve().parents[1]
T = TypeVar("T")


@pytest.mark.e2e
@pytest.mark.live_scraping
def test_live_reddit_discussion_data_survives_full_research_loop(tmp_path: Path) -> None:
    """Run the live ResearchService loop and prove Reddit social evidence reaches the report."""

    _require_live_tests_enabled()
    discussion_url = _require_live_reddit_discussion_url()
    symbol = _env_value(LIVE_REDDIT_SYMBOL_ENV) or "TSLA"
    run_date = _recent_weekday()
    provider_factory = _LiveRedditE2EProviderFactory(
        reddit_discussion_url=discussion_url,
        cache_root=tmp_path / "data" / "provider-cache",
        env=os.environ,
    )
    service = ResearchService(
        repo_root=tmp_path,
        fixture_root=REPO_ROOT,
        database_path=Path("data") / "prediction-research.sqlite3",
        provider_cache_root=tmp_path / "data" / "provider-cache",
        live_provider_factory=provider_factory,
    )

    result = service.run_live_research_flow(
        run_date=run_date.isoformat(),
        output_dir="reports/live-reddit-e2e",
        symbol=symbol,
    )

    run_id = str(result["run_id"])
    tool_runs = service.store.list_tool_runs_for_run(run_id)
    evidence = service.store.list_evidence_for_run(run_id)
    reddit_evidence = tuple(
        record for record in evidence if record.provider == "reddit-public-search"
    )
    report = result["report"]
    assert isinstance(report, dict)
    report_payload = json.loads(Path(str(report["json_path"])).read_text(encoding="utf-8"))
    run = service.store.get_research_run(run_id)

    assert run is not None
    assert run.status == "completed"
    assert {
        "research_universe_discovery",
        "research_market_data",
        "research_technical_package",
        "research_social_evidence",
        "research_news_catalyst",
        "research_fundamentals",
        "research_sector_macro",
        "research_prediction_candidate_synthesis",
        "research_prediction_evaluation",
        "render_prediction_report",
    }.issubset({record.tool_name for record in tool_runs})
    assert reddit_evidence, (
        "Live Reddit E2E smoke did not persist Reddit evidence. Set "
        f"{LIVE_REDDIT_DISCUSSION_URL_ENV} to a public old Reddit discussion page containing "
        f"{symbol.upper()} text and reachable without an interactive verification wall."
    )
    assert all(
        record.source_type in {"reddit_post", "reddit_comment"} for record in reddit_evidence
    )
    assert all(record.url and "reddit.com/r/" in record.url for record in reddit_evidence)
    assert any(discussion_url.rstrip("/") in (record.url or "") for record in reddit_evidence)
    assert any(
        _reddit_discussion_url_from_provenance(record.provenance_json) for record in reddit_evidence
    )
    assert "reddit-public-search" in {
        source["provider_name"] for source in report_payload["evidence_sources"]
    }
    assert report_payload["audit_manifest"]["report_data_mode"] == "live"


@dataclass(frozen=True)
class _LiveRedditE2EProviderFactory:
    reddit_discussion_url: str
    cache_root: Path
    env: Mapping[str, str]

    def universe_provider(self) -> ResearchLiveSymbolUniverseProvider:
        return ResearchLiveSymbolUniverseProvider()

    def market_data_provider(self, symbol: str) -> MarketDataProvider:
        return cast(MarketDataProvider, _LiveSupportMarketProvider(symbol=symbol))

    def market_data_source_query_url(self, symbol: str) -> str:
        return f"https://live-data.invalid/market/{symbol.strip().upper()}"

    def reddit_provider(self) -> RedditProvider:
        return cast(
            RedditProvider,
            RedditPublicPageProvider(
                discussion_urls=(self.reddit_discussion_url,),
                allow_live_scraping=True,
                user_agent=_scrape_user_agent(self.env),
                cache=HtmlCache(self.cache_root / "html"),
                search_result_limit=2,
                discussion_page_limit=1,
            ),
        )

    def news_providers(self, symbol: str) -> tuple[NewsProvider, ...]:
        del symbol
        return (cast(NewsProvider, _LiveSupportNewsProvider()),)

    def fundamentals_providers(self, symbol: str) -> tuple[FundamentalsProvider, ...]:
        del symbol
        return (cast(FundamentalsProvider, _LiveSupportFundamentalsProvider()),)

    def macro_providers(self, symbol: str) -> tuple[MacroProvider, ...]:
        del symbol
        return (cast(MacroProvider, _LiveSupportMacroProvider()),)


@dataclass(frozen=True)
class _LiveSupportMarketProvider:
    symbol: str
    provider_name: str = "live-reddit-e2e-market"

    def fetch_daily_candles(
        self,
        request: MarketDataRequest,
    ) -> ProviderResult[MarketSnapshot]:
        fetched_at = datetime.now(UTC)
        ticker = request.tickers[0] if request.tickers else self.symbol.upper()
        bars = tuple(
            _price_bar(ticker, request.run_date - timedelta(days=offset), offset)
            for offset in range(24, -1, -1)
        )
        return ProviderResult[MarketSnapshot](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=fetched_at,
            data=MarketSnapshot(ticker=ticker, bars=bars),
            health=_health(self.provider_name, ProviderStatus.OK, fetched_at),
            raw_snapshot_id=f"raw-live-reddit-e2e-market-{ticker.lower()}",
            cache_key=f"live-reddit-e2e-market:{ticker}:{request.run_date.isoformat()}",
        )

    def health(self) -> ProviderHealth:
        return _health(self.provider_name, ProviderStatus.OK, datetime.now(UTC))


@dataclass(frozen=True)
class _LiveSupportNewsProvider:
    provider_name: str = "live-reddit-e2e-news"

    def fetch_articles(
        self,
        request: EvidenceRequest,
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        return _empty_result(self.provider_name, request, "news")

    def health(self) -> ProviderHealth:
        return _health(self.provider_name, ProviderStatus.EMPTY, datetime.now(UTC))


@dataclass(frozen=True)
class _LiveSupportFundamentalsProvider:
    provider_name: str = "live-reddit-e2e-fundamentals"

    def fetch_fundamentals(
        self,
        request: FundamentalsRequest,
    ) -> ProviderResult[FundamentalsSnapshot]:
        return _empty_result(self.provider_name, request, "fundamentals")

    def health(self) -> ProviderHealth:
        return _health(self.provider_name, ProviderStatus.EMPTY, datetime.now(UTC))


@dataclass(frozen=True)
class _LiveSupportMacroProvider:
    provider_name: str = "live-reddit-e2e-macro"

    def fetch_macro(self, request: MacroRequest) -> ProviderResult[MacroSnapshot]:
        return _empty_result(self.provider_name, request, "macro")

    def health(self) -> ProviderHealth:
        return _health(self.provider_name, ProviderStatus.EMPTY, datetime.now(UTC))


def _empty_result(
    provider_name: str,
    request: ProviderRequest,
    lane: str,
) -> ProviderResult[T]:
    fetched_at = datetime.now(UTC)
    warning = ProviderWarning(
        provider_name=provider_name,
        code=WarningCode.NO_DATA,
        severity=WarningSeverity.INFO,
        message=f"No supplemental {lane} records were requested for the Reddit-focused E2E run.",
        occurred_at=fetched_at,
        metadata={"lane": lane, "reason": "reddit_focused_e2e"},
    )
    return cast(
        ProviderResult[T],
        ProviderResult[object](
            provider_name=provider_name,
            status=ProviderStatus.EMPTY,
            request=request,
            fetched_at=fetched_at,
            health=_health(provider_name, ProviderStatus.EMPTY, fetched_at),
            warnings=(warning,),
            raw_snapshot_id=f"raw-{provider_name}-{lane}",
            cache_key=f"{provider_name}:{lane}:{request.run_date.isoformat()}",
        ),
    )


def _price_bar(ticker: str, timestamp: date, offset: int) -> PriceBar:
    close = Decimal("180.00") + Decimal(offset) / Decimal("10")
    return PriceBar(
        ticker=ticker,
        timestamp=timestamp,
        open=close - Decimal("0.50"),
        high=close + Decimal("1.00"),
        low=close - Decimal("1.25"),
        close=close,
        volume=40_000_000 + offset,
        adjusted_close=close,
    )


def _health(
    provider_name: str,
    status: ProviderStatus,
    checked_at: datetime,
) -> ProviderHealth:
    return ProviderHealth(
        provider_name=provider_name,
        status=status,
        checked_at=checked_at,
    )


def _reddit_discussion_url_from_provenance(provenance: Mapping[str, object]) -> object | None:
    provider_metadata = provenance.get("provider_metadata")
    if not isinstance(provider_metadata, dict):
        return None
    return provider_metadata.get("discussion_url")


def _require_live_tests_enabled() -> None:
    if _env_value(ALLOW_LIVE_ENV) != "1":
        pytest.skip(f"Set {ALLOW_LIVE_ENV}=1 to run live Reddit E2E checks.")


def _require_live_reddit_discussion_url() -> str:
    value = _env_value(LIVE_REDDIT_DISCUSSION_URL_ENV)
    if value is None:
        pytest.skip(
            f"Set {LIVE_REDDIT_DISCUSSION_URL_ENV} to a public old Reddit discussion URL "
            "containing the target ticker text to run the live Reddit E2E smoke."
        )
    if not value.startswith("https://old.reddit.com/r/") or "/comments/" not in value:
        pytest.fail(
            f"{LIVE_REDDIT_DISCUSSION_URL_ENV} must be an "
            "https://old.reddit.com/r/.../comments/... URL."
        )
    return value


def _recent_weekday() -> date:
    current = datetime.now(UTC).date()
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _scrape_user_agent(env: Mapping[str, str]) -> str:
    configured = env.get(LIVE_USER_AGENT_ENV)
    if configured is not None and configured.strip():
        return configured.strip()
    return configured_scrape_user_agent(env)
