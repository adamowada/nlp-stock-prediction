from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    CredentialState,
    DateWindow,
    Direction,
    EvidenceReference,
    EvidenceRequest,
    ExtractionRequest,
    FreshnessStatus,
    FundamentalsProvider,
    FundamentalsRequest,
    FundamentalsSnapshot,
    InstrumentType,
    LLMExtractor,
    MacroProvider,
    MacroRequest,
    MacroSeries,
    MacroSnapshot,
    MarketDataProvider,
    MarketDataRequest,
    MarketSnapshot,
    NewsProvider,
    PositionType,
    PriceBar,
    ProviderHealth,
    ProviderMetric,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    RedditProvider,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    StrategyExtraction,
    TickerCandidate,
    TickerDiscoveryRequest,
    TickerDiscoveryResult,
    TickerDiscoveryStatus,
    TimeHorizon,
    WarningCode,
    WarningSeverity,
    XProvider,
)

RUN_DATE = date(2026, 5, 11)
FETCHED_AT = datetime(2026, 5, 11, 12, 0, tzinfo=UTC)
TICKERS = ("TSLA", "NVDA", "AMD", "AAPL", "MU", "SPY")


def _health(provider_name: str, status: ProviderStatus = ProviderStatus.OK) -> ProviderHealth:
    return ProviderHealth(
        provider_name=provider_name,
        status=status,
        checked_at=FETCHED_AT,
        credential_state=CredentialState.NOT_REQUIRED,
        latency_ms=7,
        last_success_at=FETCHED_AT if status == ProviderStatus.OK else None,
    )


def _warning(
    provider_name: str,
    *,
    code: WarningCode = WarningCode.PARTIAL_DATA,
    severity: WarningSeverity = WarningSeverity.WARNING,
) -> ProviderWarning:
    return ProviderWarning(
        code=code,
        severity=severity,
        message=f"{provider_name} fixture warning",
        provider_name=provider_name,
        occurred_at=FETCHED_AT,
        raw_snapshot_id=f"raw-{provider_name}-warning",
    )


def _provenance(
    provider_name: str,
    source_kind: SourceKind,
    retrieval_method: RetrievalMethod = RetrievalMethod.FIXTURE,
) -> SourceProvenance:
    return SourceProvenance(
        provider_name=provider_name,
        source_kind=source_kind,
        retrieval_method=retrieval_method,
        fetched_at=FETCHED_AT,
        observed_at=FETCHED_AT,
        source_url=f"https://example.invalid/{provider_name}",
        permalink=f"https://example.invalid/{provider_name}/permalink",
        raw_identifier=f"raw-{provider_name}-1",
        raw_snapshot_id=f"raw-{provider_name}-snapshot",
        query="TSLA OR NVDA",
        cache_key=f"{provider_name}:2026-05-11",
        freshness_status=FreshnessStatus.FRESH,
        freshness_seconds=60,
        provider_metadata={"fixture": True},
    )


def _evidence(
    provider_name: str,
    source_kind: SourceKind,
    ticker: str = "TSLA",
) -> SourceEvidence:
    return SourceEvidence(
        evidence_id=f"evidence-{provider_name}-{ticker}",
        source_kind=source_kind,
        ticker=ticker,
        title=f"{ticker} fixture discussion",
        text=f"{ticker} bulls mention a defined-risk call spread into the next catalyst.",
        author_hash="author-fixture",
        created_at=FETCHED_AT,
        score=42,
        permalink=f"https://example.invalid/{provider_name}/{ticker}",
        matched_tickers=(ticker,),
        provenance=_provenance(provider_name, source_kind),
        metadata={"source_rank": 1},
    )


def _discovery_result(provider_name: str) -> TickerDiscoveryResult:
    provenance = _provenance(provider_name, SourceKind.REDDIT_TICKER_CARD)
    candidates = tuple(
        TickerCandidate(
            symbol=ticker,
            raw_identifier=f"ticker-container-{ticker.lower()}",
            raw_text=ticker,
            first_seen_rank=index,
            source_url="https://reddit.example.invalid/r/wallstreetbets",
            provenance=provenance,
        )
        for index, ticker in enumerate(TICKERS)
    )
    return TickerDiscoveryResult(
        run_date=RUN_DATE,
        status=TickerDiscoveryStatus.VALID,
        candidates=candidates,
        tickers=TICKERS,
        raw_snapshot_id="raw-reddit-devvit-card-2026-05-11",
    )


class _FakeRedditProvider:
    provider_name = "fixture-reddit"

    def discover_tickers(
        self, request: TickerDiscoveryRequest
    ) -> ProviderResult[TickerDiscoveryResult]:
        return ProviderResult[TickerDiscoveryResult](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=FETCHED_AT,
            data=_discovery_result(self.provider_name),
            health=self.health(),
            raw_snapshot_id="raw-reddit-devvit-card-2026-05-11",
            cache_key="reddit:ticker-card:2026-05-11",
        )

    def fetch_discussion(
        self, request: EvidenceRequest
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        return ProviderResult[tuple[SourceEvidence, ...]](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=FETCHED_AT,
            data=(_evidence(self.provider_name, SourceKind.REDDIT_POST),),
            health=self.health(),
            raw_snapshot_id="raw-reddit-discussion-2026-05-11",
            cache_key="reddit:discussion:TSLA:2026-05-11",
        )

    def health(self) -> ProviderHealth:
        return _health(self.provider_name)


class _FakeXProvider:
    provider_name = "fixture-x"

    def fetch_social_posts(
        self, request: EvidenceRequest
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        return ProviderResult[tuple[SourceEvidence, ...]](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=FETCHED_AT,
            data=(_evidence(self.provider_name, SourceKind.X_POST),),
            health=self.health(),
            raw_snapshot_id="raw-x-posts-2026-05-11",
            cache_key="x:posts:TSLA:2026-05-11",
        )

    def health(self) -> ProviderHealth:
        return _health(self.provider_name)


class _FakeNewsProvider:
    provider_name = "fixture-news"

    def fetch_articles(
        self, request: EvidenceRequest
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        return ProviderResult[tuple[SourceEvidence, ...]](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=FETCHED_AT,
            data=(_evidence(self.provider_name, SourceKind.NEWS_ARTICLE),),
            health=self.health(),
            raw_snapshot_id="raw-news-articles-2026-05-11",
            cache_key="news:articles:TSLA:2026-05-11",
        )

    def health(self) -> ProviderHealth:
        return _health(self.provider_name)


class _FakeMarketDataProvider:
    provider_name = "fixture-market-data"

    def fetch_daily_candles(self, request: MarketDataRequest) -> ProviderResult[MarketSnapshot]:
        snapshot = MarketSnapshot(
            ticker="TSLA",
            bars=(
                PriceBar(
                    ticker="TSLA",
                    timestamp=RUN_DATE,
                    open=Decimal("180.00"),
                    high=Decimal("185.00"),
                    low=Decimal("178.50"),
                    close=Decimal("184.25"),
                    adjusted_close=Decimal("184.25"),
                    volume=123_456_789,
                ),
            ),
            liquidity_metrics=(
                ProviderMetric(
                    name="average_dollar_volume",
                    value=Decimal("22745234444.25"),
                    unit="USD",
                    as_of=RUN_DATE,
                    metadata={"window": "30d"},
                ),
            ),
        )
        return ProviderResult[MarketSnapshot](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=FETCHED_AT,
            data=snapshot,
            health=self.health(),
            raw_snapshot_id="raw-market-data-tsla-2026-05-11",
            cache_key="market:daily:TSLA:2026-05-11",
        )

    def health(self) -> ProviderHealth:
        return _health(self.provider_name)


class _FakeFundamentalsProvider:
    provider_name = "fixture-fundamentals"

    def fetch_fundamentals(
        self, request: FundamentalsRequest
    ) -> ProviderResult[FundamentalsSnapshot]:
        snapshot = FundamentalsSnapshot(
            ticker="TSLA",
            company_name="Tesla, Inc.",
            metrics=(
                ProviderMetric(
                    name="market_cap",
                    value=Decimal("575000000000"),
                    unit="USD",
                    as_of=RUN_DATE,
                    metadata={"statement": "fixture"},
                ),
            ),
        )
        return ProviderResult[FundamentalsSnapshot](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=FETCHED_AT,
            data=snapshot,
            health=self.health(),
            raw_snapshot_id="raw-fundamentals-tsla-2026-05-11",
            cache_key="fundamentals:TSLA:2026-05-11",
        )

    def health(self) -> ProviderHealth:
        return _health(self.provider_name)


class _FakeMacroProvider:
    provider_name = "fixture-macro"

    def fetch_macro(self, request: MacroRequest) -> ProviderResult[MacroSnapshot]:
        snapshot = MacroSnapshot(
            series=(
                MacroSeries(
                    series_id="FEDFUNDS",
                    name="Effective Federal Funds Rate",
                    values=(
                        ProviderMetric(
                            name="rate",
                            value=Decimal("4.25"),
                            unit="percent",
                            as_of=RUN_DATE,
                            metadata={"series_id": "FEDFUNDS"},
                        ),
                    ),
                ),
            )
        )
        return ProviderResult[MacroSnapshot](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=FETCHED_AT,
            data=snapshot,
            health=self.health(),
            raw_snapshot_id="raw-macro-2026-05-11",
            cache_key="macro:FEDFUNDS:2026-05-11",
        )

    def health(self) -> ProviderHealth:
        return _health(self.provider_name)


class _FakeLLMExtractor:
    provider_name = "fixture-llm-extractor"

    def extract_strategies(
        self, request: ExtractionRequest
    ) -> ProviderResult[tuple[StrategyExtraction, ...]]:
        extraction = StrategyExtraction(
            strategy_id="strategy-tsla-call-spread",
            ticker="TSLA",
            label="Defined-risk call spread",
            direction=Direction.BULLISH,
            instrument=InstrumentType.OPTION_SPREAD,
            position_type=PositionType.DEFINED_RISK,
            time_horizon=TimeHorizon.WEEKLY,
            catalyst="Fixture discussion mentions a near-term catalyst.",
            risk_or_hedge="Defined-risk spread limits premium at risk.",
            evidence=(
                EvidenceReference(
                    evidence_id=request.evidence[0].evidence_id,
                    quote="defined-risk call spread",
                    relevance=0.9,
                ),
            ),
            confidence=0.75,
            sarcasm_joke_risk=0.1,
        )
        return ProviderResult[tuple[StrategyExtraction, ...]](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=FETCHED_AT,
            data=(extraction,),
            health=self.health(),
            raw_snapshot_id="raw-llm-extraction-2026-05-11",
            cache_key="llm:extract:TSLA:2026-05-11",
        )

    def health(self) -> ProviderHealth:
        return _health(self.provider_name)


@pytest.mark.integration
def test_deterministic_fake_providers_return_protocol_result_shapes() -> None:
    reddit: RedditProvider = _FakeRedditProvider()
    x_provider: XProvider = _FakeXProvider()
    news: NewsProvider = _FakeNewsProvider()
    market: MarketDataProvider = _FakeMarketDataProvider()
    fundamentals: FundamentalsProvider = _FakeFundamentalsProvider()
    macro: MacroProvider = _FakeMacroProvider()
    extractor: LLMExtractor = _FakeLLMExtractor()

    ticker_request = TickerDiscoveryRequest(
        request_id="discover-2026-05-11",
        run_date=RUN_DATE,
        source_url="https://reddit.example.invalid/r/wallstreetbets",
    )
    evidence_request = EvidenceRequest(
        request_id="evidence-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        query="TSLA",
        limit=10,
    )
    market_request = MarketDataRequest(
        request_id="market-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        interval="1d",
    )
    fundamentals_request = FundamentalsRequest(
        request_id="fundamentals-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        fiscal_period="ttm",
    )
    macro_request = MacroRequest(
        request_id="macro-2026-05-11",
        run_date=RUN_DATE,
        series_ids=("FEDFUNDS",),
        horizon=TimeHorizon.MONTHLY,
    )

    discovery = reddit.discover_tickers(ticker_request)
    reddit_discussion = reddit.fetch_discussion(evidence_request)
    x_posts = x_provider.fetch_social_posts(evidence_request)
    articles = news.fetch_articles(evidence_request)
    candles = market.fetch_daily_candles(market_request)
    fundamental_snapshot = fundamentals.fetch_fundamentals(fundamentals_request)
    macro_snapshot = macro.fetch_macro(macro_request)
    extraction_request = ExtractionRequest(
        request_id="extract-tsla-2026-05-11",
        run_date=RUN_DATE,
        tickers=("TSLA",),
        evidence=reddit_discussion.data or (),
        prompt_version="phase1-fixture-v1",
        schema_version="strategy-extraction-v1",
    )
    extractions = extractor.extract_strategies(extraction_request)

    assert discovery.data is not None
    assert discovery.data.tickers == TICKERS
    assert reddit_discussion.data is not None
    assert reddit_discussion.data[0].provenance.provider_name == "fixture-reddit"
    assert x_posts.data is not None
    assert x_posts.data[0].source_kind == SourceKind.X_POST
    assert articles.data is not None
    assert articles.data[0].source_kind == SourceKind.NEWS_ARTICLE
    assert candles.data is not None
    assert candles.data.bars[0].close == Decimal("184.25")
    assert fundamental_snapshot.data is not None
    assert fundamental_snapshot.data.metrics[0].name == "market_cap"
    assert macro_snapshot.data is not None
    assert macro_snapshot.data.series[0].series_id == "FEDFUNDS"
    assert extractions.data is not None
    assert extractions.data[0].evidence[0].evidence_id == "evidence-fixture-reddit-TSLA"

    first_dump = reddit.fetch_discussion(evidence_request).model_dump(mode="json")
    second_dump = reddit.fetch_discussion(evidence_request).model_dump(mode="json")
    assert first_dump == second_dump


@pytest.mark.schema
@pytest.mark.parametrize(
    ("status", "data"),
    [
        (ProviderStatus.FAILED, None),
        (ProviderStatus.UNCONFIGURED, None),
        (ProviderStatus.UNAUTHORIZED, None),
        (ProviderStatus.RATE_LIMITED, None),
        (ProviderStatus.MALFORMED, None),
    ],
)
def test_provider_result_hard_failures_have_warning_and_no_data(
    status: ProviderStatus, data: tuple[SourceEvidence, ...] | None
) -> None:
    request = EvidenceRequest(request_id=f"hard-failure-{status}", run_date=RUN_DATE)
    result = ProviderResult[tuple[SourceEvidence, ...]](
        provider_name="fixture-provider",
        status=status,
        request=request,
        fetched_at=FETCHED_AT,
        data=data,
        warnings=(
            _warning(
                "fixture-provider",
                code=WarningCode.UPSTREAM_UNAVAILABLE,
                severity=WarningSeverity.ERROR,
            ),
        ),
        health=_health("fixture-provider", status),
    )

    assert result.data is None
    assert result.warnings[0].severity == WarningSeverity.ERROR
    assert result.health.status == status


@pytest.mark.schema
def test_provider_result_success_and_partial_envelopes_preserve_data() -> None:
    request = EvidenceRequest(request_id="partial-success", run_date=RUN_DATE, tickers=("TSLA",))
    data = (_evidence("fixture-provider", SourceKind.REDDIT_POST),)

    success = ProviderResult[tuple[SourceEvidence, ...]](
        provider_name="fixture-provider",
        status=ProviderStatus.OK,
        request=request,
        fetched_at=FETCHED_AT,
        data=data,
        health=_health("fixture-provider"),
    )
    partial = ProviderResult[tuple[SourceEvidence, ...]](
        provider_name="fixture-provider",
        status=ProviderStatus.PARTIAL,
        request=request,
        fetched_at=FETCHED_AT,
        data=data,
        warnings=(_warning("fixture-provider"),),
        health=_health("fixture-provider", ProviderStatus.PARTIAL),
    )

    assert success.data == data
    assert not success.warnings
    assert partial.data == data
    assert partial.warnings[0].code == WarningCode.PARTIAL_DATA


@pytest.mark.schema
def test_provider_result_empty_envelope_requires_no_data_warning_and_no_data() -> None:
    request = EvidenceRequest(request_id="empty-provider", run_date=RUN_DATE, tickers=("TSLA",))

    result = ProviderResult[tuple[SourceEvidence, ...]](
        provider_name="fixture-provider",
        status=ProviderStatus.EMPTY,
        request=request,
        fetched_at=FETCHED_AT,
        data=None,
        warnings=(
            _warning(
                "fixture-provider",
                code=WarningCode.NO_DATA,
                severity=WarningSeverity.INFO,
            ),
        ),
        health=_health("fixture-provider", ProviderStatus.EMPTY),
    )

    assert result.data is None
    assert result.warnings[0].code == WarningCode.NO_DATA


@pytest.mark.schema
def test_provider_result_rejects_invalid_envelope_semantics() -> None:
    request = ProviderRequest(request_id="invalid-envelope", run_date=RUN_DATE)

    with pytest.raises(ValidationError, match="usable provider results must include data"):
        ProviderResult[str](
            provider_name="fixture-provider",
            status=ProviderStatus.OK,
            request=request,
            fetched_at=FETCHED_AT,
            health=_health("fixture-provider"),
        )

    with pytest.raises(ValidationError, match="non-ok provider results must include"):
        ProviderResult[str](
            provider_name="fixture-provider",
            status=ProviderStatus.PARTIAL,
            request=request,
            fetched_at=FETCHED_AT,
            data="some data",
            health=_health("fixture-provider", ProviderStatus.PARTIAL),
        )

    with pytest.raises(ValidationError, match="hard provider failures must not include data"):
        ProviderResult[str](
            provider_name="fixture-provider",
            status=ProviderStatus.FAILED,
            request=request,
            fetched_at=FETCHED_AT,
            data="should not be present",
            warnings=(
                _warning(
                    "fixture-provider",
                    code=WarningCode.UPSTREAM_UNAVAILABLE,
                    severity=WarningSeverity.ERROR,
                ),
            ),
            health=_health("fixture-provider", ProviderStatus.FAILED),
        )

    with pytest.raises(ValidationError, match="empty provider results must not include data"):
        ProviderResult[str](
            provider_name="fixture-provider",
            status=ProviderStatus.EMPTY,
            request=request,
            fetched_at=FETCHED_AT,
            data="unexpected",
            warnings=(
                _warning(
                    "fixture-provider",
                    code=WarningCode.NO_DATA,
                    severity=WarningSeverity.INFO,
                ),
            ),
            health=_health("fixture-provider", ProviderStatus.EMPTY),
        )

    with pytest.raises(ValidationError, match="no_data warning"):
        ProviderResult[str](
            provider_name="fixture-provider",
            status=ProviderStatus.EMPTY,
            request=request,
            fetched_at=FETCHED_AT,
            data=None,
            warnings=(_warning("fixture-provider"),),
            health=_health("fixture-provider", ProviderStatus.EMPTY),
        )

    with pytest.raises(ValidationError, match="warnings must match provider_name"):
        ProviderResult[str](
            provider_name="fixture-provider",
            status=ProviderStatus.PARTIAL,
            request=request,
            fetched_at=FETCHED_AT,
            data="partial data",
            warnings=(_warning("other-provider"),),
            health=_health("fixture-provider", ProviderStatus.PARTIAL),
        )


@pytest.mark.schema
def test_provider_request_round_trips_through_json_serialization() -> None:
    request = EvidenceRequest(
        request_id="serialize-evidence-request",
        run_date=RUN_DATE,
        tickers=("tsla", "nvda"),
        window=DateWindow(
            start=datetime(2026, 5, 10, 9, 30, tzinfo=UTC),
            end=datetime(2026, 5, 11, 16, 0, tzinfo=UTC),
        ),
        limit=25,
        query="$TSLA OR $NVDA",
        options={
            "include_removed": False,
            "minimum_score": 10,
            "source_weights": {"reddit": 0.7, "x": 0.3},
        },
        include_comments=True,
        include_posts=False,
    )

    dumped = request.model_dump(mode="json")
    encoded = request.model_dump_json()
    decoded = EvidenceRequest.model_validate_json(encoded)

    assert dumped["request_id"] == "serialize-evidence-request"
    assert dumped["run_date"] == "2026-05-11"
    assert dumped["tickers"] == ["TSLA", "NVDA"]
    assert dumped["window"]["start"].startswith("2026-05-10T09:30:00")
    assert dumped["limit"] == 25
    assert dumped["options"]["source_weights"] == {"reddit": 0.7, "x": 0.3}
    assert dumped["include_comments"] is True
    assert dumped["include_posts"] is False
    assert decoded == request


@pytest.mark.schema
def test_date_window_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="date window end"):
        DateWindow(start=date(2026, 5, 12), end=date(2026, 5, 11))

    with pytest.raises(ValidationError, match="date window end"):
        DateWindow(
            start=datetime(2026, 5, 11, 16, 0, tzinfo=UTC),
            end=datetime(2026, 5, 11, 9, 30, tzinfo=UTC),
        )

    with pytest.raises(ValidationError, match="timezone-aware"):
        DateWindow(
            start=datetime(2026, 5, 11, 9, 30),
            end=datetime(2026, 5, 11, 16, 0, tzinfo=UTC),
        )
