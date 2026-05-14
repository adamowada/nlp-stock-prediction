from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.contracts import (
    FreshnessStatus,
    FundamentalsSnapshot,
    ProviderMetric,
    RedditProvider,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TimeHorizon,
    WarningCode,
)
from nlp_stock_prediction.orchestration.phase2_evidence import evidence_stance_from_record
from nlp_stock_prediction.orchestration.phase4_common import evidence_reference
from nlp_stock_prediction.orchestration.phase4_fundamentals import (
    run_phase4_fundamentals_tool,
)
from nlp_stock_prediction.orchestration.phase4_news import run_phase4_news_catalyst_tool
from nlp_stock_prediction.orchestration.phase4_sector_macro import (
    run_phase4_sector_macro_tool,
)
from nlp_stock_prediction.orchestration.phase4_social import run_phase4_social_evidence_tool
from nlp_stock_prediction.providers._base import JsonResponse
from nlp_stock_prediction.providers.apnews import APNewsProvider
from nlp_stock_prediction.providers.fred import FredMacroProvider
from nlp_stock_prediction.providers.news import PublicNewsProvider, PublicNewsProviderConfig
from nlp_stock_prediction.providers.scraping import HtmlResponse
from nlp_stock_prediction.providers.sec_edgar import SecEdgarFundamentalsProvider
from nlp_stock_prediction.providers.social import XRecentSearchProvider
from nlp_stock_prediction.reddit.provider import FixtureRedditProvider
from nlp_stock_prediction.storage import ResearchRunRecord, SQLiteStore

pytestmark = pytest.mark.integration

RUN_ID = "run-phase4-evidence-suite"
RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
FIXTURE_ROOT = Path(__file__).parent / "fixtures"
RAW_FIXTURE_ROOT = FIXTURE_ROOT / "raw"


@dataclass
class _FakeJsonTransport:
    responses: dict[str, JsonResponse]
    calls: list[str] = field(default_factory=list)
    headers: list[Mapping[str, str] | None] = field(default_factory=list)

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> JsonResponse:
        del timeout
        self.calls.append(url)
        self.headers.append(headers)
        for url_fragment, response in self.responses.items():
            if url_fragment in url:
                return response
        raise AssertionError(f"Unexpected URL: {url}")


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


def _json_fixture(*parts: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(RAW_FIXTURE_ROOT.joinpath(*parts).read_text(encoding="utf-8")),
    )


def _reddit_records() -> list[dict[str, object]]:
    return cast(
        list[dict[str, object]],
        json.loads((FIXTURE_ROOT / "reddit" / "discussion_records.json").read_text("utf-8")),
    )


def _reddit_html() -> str:
    return (FIXTURE_ROOT / "reddit" / "devvit_card_normal.html").read_text("utf-8")


def _ap_html(name: str) -> str:
    return (RAW_FIXTURE_ROOT / "apnews" / name).read_text("utf-8")


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "prediction-research.sqlite3")
    store.initialize()
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=RUN_ID,
            run_kind="phase4_evidence_suite",
            objective="fixture-backed phase4 evidence tool tests",
            status="running",
            started_at=FETCHED_AT,
        )
    )
    return store


def _artifact_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports" / RUN_DATE.isoformat() / "audit"


def _artifact_payload(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def store_evidence_text_with_newlines() -> SourceEvidence:
    text = "\n\nTSLA revenue beat expectations\nwhile margins stayed under pressure."
    return SourceEvidence(
        evidence_id="evidence-newline-span",
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="TSLA",
        text=text,
        permalink="https://example.test/newline",
        matched_tickers=("TSLA",),
        provenance=SourceProvenance(
            provider_name="fixture-news",
            source_kind=SourceKind.NEWS_ARTICLE,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=FETCHED_AT,
            source_url="https://example.test/newline",
            permalink="https://example.test/newline",
            raw_identifier="newline",
            raw_snapshot_id="raw-newline",
            query="TSLA",
            freshness_status=FreshnessStatus.FRESH,
        ),
    )


def test_phase4_social_tool_indexes_social_evidence_and_derived_labels(tmp_path: Path) -> None:
    store = _store(tmp_path)
    reddit_provider = FixtureRedditProvider(
        ticker_card_html=_reddit_html(),
        discussion_records=_reddit_records(),
        fetched_at=FETCHED_AT,
        raw_ticker_snapshot_id="raw-reddit-ticker-card",
        raw_discussion_snapshot_id="raw-reddit-discussion",
    )
    x_provider = XRecentSearchProvider(
        bearer_token="fixture-token",
        transport=_FakeJsonTransport(
            {"tweets/search/recent": JsonResponse(payload=_json_fixture("x", "recent_tsla.json"))}
        ),
        now=lambda: FETCHED_AT,
    )

    result = run_phase4_social_evidence_tool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_artifact_dir(tmp_path),
        run_id=RUN_ID,
        symbol="TSLA",
        run_date=RUN_DATE,
        generated_at=FETCHED_AT,
        reddit_provider=cast(RedditProvider, reddit_provider),
        x_provider=x_provider,
        instrument_id="instrument:equity:us:tsla",
        extra_tickers=("AI", "MU", "ON"),
        limit=25,
    )

    payload = _artifact_payload(result.artifact_path)
    labels = payload["derived_analysis"]["labels"]
    evidence_rows = store.list_evidence_for_run(RUN_ID)

    assert result.status == "successful"
    assert store.get_tool_run(result.tool_run_id) is not None
    assert store.get_artifact(result.artifact_id) is not None
    assert len(store.list_source_queries_for_run(RUN_ID)) == 2
    assert len(evidence_rows) == 3
    assert any(label["stance"] == "contradicts" for label in labels)
    assert any(label["catalysts"] == ["robotaxi"] for label in labels)
    assert all(row.metadata["source_evidence"] for row in evidence_rows)
    assert any("AI calls look expensive" in row.claim for row in evidence_rows)
    assert all("derived_analysis" in row.metadata for row in evidence_rows)
    assert any(evidence_stance_from_record(row) == "contradicts" for row in evidence_rows)
    x_query = next(
        query
        for query in store.list_source_queries_for_run(RUN_ID)
        if query.provider == "x-recent-search"
    )
    assert x_query.url is not None
    assert "tweets/search/recent" in x_query.url
    assert "x.com/i/web/status" not in x_query.url


def test_phase4_news_tool_preserves_articles_and_catalyst_labels(tmp_path: Path) -> None:
    store = _store(tmp_path)
    news_provider = PublicNewsProvider(
        config=PublicNewsProviderConfig(
            provider_name="fixture-news",
            endpoint="https://news.example.invalid/v1/search",
            api_key_param="token",
            query_param="search",
        ),
        api_key="fixture-key",
        transport=_FakeJsonTransport(
            {
                "news.example.invalid/v1/search": JsonResponse(
                    payload=_json_fixture("news", "tsla.json")
                )
            }
        ),
        now=lambda: FETCHED_AT,
    )
    ap_provider = APNewsProvider(
        transport=_FakeHtmlTransport(
            {
                "hub/financial-markets": _ap_html("hub_financial_markets.html"),
                "tesla-nvidia-markets": _ap_html("article_tsla_nvidia.html"),
                "oil-prices-economy": _ap_html("article_unrelated.html"),
            }
        ),
        now=lambda: FETCHED_AT,
    )

    result = run_phase4_news_catalyst_tool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_artifact_dir(tmp_path),
        run_id=RUN_ID,
        symbol="TSLA",
        run_date=RUN_DATE,
        generated_at=FETCHED_AT,
        providers=(news_provider, ap_provider),
        instrument_id="instrument:equity:us:tsla",
        extra_tickers=("NVDA",),
        limit=1,
    )

    payload = _artifact_payload(result.artifact_path)
    labels = payload["derived_analysis"]["labels"]
    evidence_rows = store.list_evidence_for_run(RUN_ID)

    assert result.status == "successful"
    assert len(store.list_source_queries_for_run(RUN_ID)) == 2
    assert len(evidence_rows) >= 2
    assert any(label["catalysts"] for label in labels)
    assert any("robotaxi product update" in row.claim for row in evidence_rows)
    assert all(row.source_type == "news_article" for row in evidence_rows)
    assert payload["derived_analysis"]["analysis_type"] == "news_catalyst_labels"
    assert any(evidence_stance_from_record(row) == "supports" for row in evidence_rows)
    source_queries = {query.provider: query for query in store.list_source_queries_for_run(RUN_ID)}
    assert source_queries["ap-news"].url == "https://apnews.com/hub/financial-markets"
    assert source_queries["fixture-news"].url is not None
    assert "token=REDACTED" in source_queries["fixture-news"].url
    assert all(row.url != source_queries["ap-news"].url for row in evidence_rows)


def test_phase4_fundamentals_tool_indexes_sec_metrics_and_analysis(tmp_path: Path) -> None:
    store = _store(tmp_path)
    provider = SecEdgarFundamentalsProvider(
        ticker_cik_map={"TSLA": "1318605"},
        user_agent="nlp-stock-prediction-test contact@example.test",
        transport=_FakeJsonTransport(
            {
                "companyfacts": JsonResponse(
                    payload=_json_fixture("sec_edgar", "companyfacts_tsla.json")
                ),
                "submissions": JsonResponse(
                    payload=_json_fixture("sec_edgar", "submissions_tsla.json")
                ),
            }
        ),
        now=lambda: FETCHED_AT,
    )

    result = run_phase4_fundamentals_tool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_artifact_dir(tmp_path),
        run_id=RUN_ID,
        symbol="TSLA",
        run_date=RUN_DATE,
        generated_at=FETCHED_AT,
        providers=(provider,),
        instrument_id="instrument:equity:us:tsla",
    )

    payload = _artifact_payload(result.artifact_path)
    evidence_rows = store.list_evidence_for_run(RUN_ID)

    assert result.status == "successful"
    assert len(store.list_source_queries_for_run(RUN_ID)) == 1
    assert evidence_rows
    assert any(row.source_type == "sec_filing" for row in evidence_rows)
    assert any(row.source_type == "fundamental_data" for row in evidence_rows)
    assert payload["fundamentals_snapshot"]["company_name"] == "Tesla, Inc."
    assert payload["derived_analysis"]["component"]["evidence"]
    assert all(row.metadata["phase4_metric_record"] is True for row in evidence_rows)
    source_query = store.list_source_queries_for_run(RUN_ID)[0]
    assert source_query.url == "https://data.sec.gov/api/xbrl/companyfacts/CIK0001318605.json"
    assert source_query.metadata["source_query_urls"] == [
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0001318605.json",
        "https://data.sec.gov/submissions/CIK0001318605.json",
    ]


def test_phase4_sector_macro_tool_preserves_stale_macro_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = FundamentalsSnapshot(
        ticker="TSLA",
        company_name="Tesla, Inc.",
        metrics=(
            ProviderMetric(
                name="pe_ratio",
                value=Decimal("48.5"),
                as_of=RUN_DATE,
                metadata={
                    "provider_name": "sec-edgar",
                    "raw_snapshot_id": "raw-sec-target",
                    "source_query_url": (
                        "https://data.sec.gov/api/xbrl/companyfacts/CIK0001318605.json"
                    ),
                },
            ),
            ProviderMetric(name="net_margin", value=Decimal("0.141"), as_of=RUN_DATE),
        ),
    )
    peers = (
        FundamentalsSnapshot(
            ticker="F",
            metrics=(
                ProviderMetric(name="pe_ratio", value=Decimal("12.0"), as_of=RUN_DATE),
                ProviderMetric(name="net_margin", value=Decimal("0.07"), as_of=RUN_DATE),
            ),
        ),
    )
    fred_provider = FredMacroProvider(
        api_key="fixture-key",
        transport=_FakeJsonTransport(
            {
                "series/observations": JsonResponse(
                    payload=_json_fixture("fred", "fedfunds_stale.json")
                )
            }
        ),
        now=lambda: FETCHED_AT,
        stale_after_days=30,
    )

    result = run_phase4_sector_macro_tool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_artifact_dir(tmp_path),
        run_id=RUN_ID,
        symbol="TSLA",
        run_date=RUN_DATE,
        generated_at=FETCHED_AT,
        target_snapshot=target,
        macro_providers=(fred_provider,),
        instrument_id="instrument:equity:us:tsla",
        peers=peers,
        sector="Consumer Discretionary",
        benchmark_symbol="XLY",
        benchmark_metrics=(
            ProviderMetric(
                name="sector_etf_return_20d",
                value=Decimal("-0.025"),
                unit="pct",
                as_of=RUN_DATE,
            ),
        ),
        macro_series_ids=("FEDFUNDS",),
        horizon=TimeHorizon.MONTHLY,
    )

    payload = _artifact_payload(result.artifact_path)
    evidence_rows = store.list_evidence_for_run(RUN_ID)

    assert result.status == "partial"
    assert any(WarningCode.STALE_DATA.value in warning for warning in result.warnings)
    assert len(store.list_source_queries_for_run(RUN_ID)) == 2
    assert any(
        row.source_type == "macro_series" and row.freshness_status == "stale"
        for row in evidence_rows
    )
    assert payload["derived_analysis"]["sector_context"]["evidence"]
    assert payload["derived_analysis"]["macro_context"]["warnings"]
    sec_sector_row = next(
        row for row in evidence_rows if row.claim.startswith("TSLA provider metric pe_ratio")
    )
    assert sec_sector_row.provider == "sec-edgar"
    assert sec_sector_row.provenance_json["provider_name"] == "sec-edgar"
    assert sec_sector_row.provenance_json["raw_snapshot_id"] == "raw-sec-target"
    assert sec_sector_row.url == "https://data.sec.gov/api/xbrl/companyfacts/CIK0001318605.json"


def test_phase4_evidence_reference_offsets_match_original_whitespace() -> None:
    evidence = store_evidence_text_with_newlines()
    reference = evidence_reference(evidence)

    assert reference.quote is not None
    assert reference.start_char is not None
    assert reference.end_char is not None
    assert evidence.text[reference.start_char : reference.end_char] == reference.quote


def test_phase4_news_tool_writes_warning_artifact_for_provider_failures(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    missing_key = PublicNewsProvider(
        config=PublicNewsProviderConfig(provider_name="missing-key-news"),
        transport=_FakeJsonTransport({}),
        now=lambda: FETCHED_AT,
    )
    malformed = PublicNewsProvider(
        config=PublicNewsProviderConfig(
            provider_name="malformed-news",
            endpoint="https://news.example.invalid/v1/search",
            api_key_param="token",
            query_param="search",
        ),
        api_key="fixture-key",
        transport=_FakeJsonTransport(
            {"news.example.invalid/v1/search": JsonResponse(payload={"not_articles": []})}
        ),
        now=lambda: FETCHED_AT,
    )

    result = run_phase4_news_catalyst_tool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_artifact_dir(tmp_path),
        run_id=RUN_ID,
        symbol="TSLA",
        run_date=RUN_DATE,
        generated_at=FETCHED_AT,
        providers=(missing_key, malformed),
    )

    payload = _artifact_payload(result.artifact_path)

    assert result.status == "partial"
    assert result.evidence_ids == ()
    assert store.get_artifact(result.artifact_id) is not None
    assert len(store.list_source_queries_for_run(RUN_ID)) == 2
    assert any(WarningCode.MISSING_CREDENTIALS.value in warning for warning in result.warnings)
    assert any(WarningCode.MALFORMED_RESPONSE.value in warning for warning in result.warnings)
    assert payload["source_evidence"] == []
    assert payload["warnings"]
