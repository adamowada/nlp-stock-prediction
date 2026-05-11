from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    DEFAULT_MARKDOWN_REPORT_OUTLINE,
    AuditArtifact,
    AuditManifest,
    CredentialState,
    DailyReport,
    DataFreshnessSummary,
    DataReference,
    Direction,
    Disclaimer,
    EvidenceReference,
    FreshnessStatus,
    InstrumentType,
    PositionType,
    ProviderHealth,
    ProviderStatus,
    ProviderWarning,
    RecommendationAction,
    RetrievalMethod,
    RiskAssessment,
    RiskProfile,
    ScoreBreakdown,
    ScoreComponent,
    SourceKind,
    SourceProvenance,
    StrategyCluster,
    TickerCandidate,
    TickerDiscoveryResult,
    TickerDiscoveryStatus,
    TickerReportSection,
    TimeHorizon,
    TradeCandidate,
    WarningCode,
    WarningSeverity,
)

TICKERS = ("TSLA", "NVDA", "AMD", "AAPL", "MU", "SPY")


def _timestamp() -> datetime:
    return datetime(2026, 5, 11, 20, 0, tzinfo=UTC)


def _disclaimer() -> Disclaimer:
    return Disclaimer(
        disclaimer_id="educational-report-v1",
        version="2026-05-11",
        text="Educational research only; not financial advice; no automatic trading.",
    )


def _source_provenance(ticker: str) -> SourceProvenance:
    return SourceProvenance(
        provider_name="fixture-reddit",
        source_kind=SourceKind.REDDIT_TICKER_CARD,
        retrieval_method=RetrievalMethod.FIXTURE,
        fetched_at=_timestamp(),
        source_url="https://reddit.example/wsb/daily-card",
        raw_identifier=f"card-{ticker.lower()}",
        raw_snapshot_id="raw-reddit-card-2026-05-11",
        query="r/wallstreetbets daily ticker card",
        cache_key="fixture:reddit-card:2026-05-11",
        freshness_status=FreshnessStatus.FRESH,
        freshness_seconds=300,
        provider_metadata={"fixture": True},
    )


def _ticker_discovery(tickers: tuple[str, ...] = TICKERS) -> TickerDiscoveryResult:
    candidates = tuple(
        TickerCandidate(
            symbol=ticker,
            raw_identifier=f"ticker-container-{ticker.lower()}",
            raw_text=ticker,
            first_seen_rank=rank,
            source_url="https://reddit.example/wsb/daily-card",
            provenance=_source_provenance(ticker),
        )
        for rank, ticker in enumerate(tickers)
    )
    return TickerDiscoveryResult(
        run_date=date(2026, 5, 11),
        status=TickerDiscoveryStatus.VALID,
        candidates=candidates,
        tickers=tickers,
        raw_snapshot_id="raw-reddit-card-2026-05-11",
    )


def _evidence_ref(ticker: str) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=f"reddit-{ticker.lower()}-comment-1",
        quote=f"${ticker} momentum setup discussed in fixture comments",
        start_char=12,
        end_char=48,
        relevance=0.82,
    )


def _strategy_cluster(ticker: str) -> StrategyCluster:
    evidence = _evidence_ref(ticker)
    return StrategyCluster(
        cluster_id=f"cluster-{ticker.lower()}-bullish-shares",
        ticker=ticker,
        direction=Direction.BULLISH,
        instrument=InstrumentType.SHARES,
        time_horizon=TimeHorizon.SWING,
        catalyst_summary="Retail discussion cites momentum and upcoming event risk.",
        member_strategy_ids=(f"strategy-{ticker.lower()}-1",),
        evidence=(evidence,),
        confidence=0.72,
    )


def _ticker_sections(
    tickers: tuple[str, ...] = TICKERS, *, include_candidate_links: bool = False
) -> tuple[TickerReportSection, ...]:
    sections: list[TickerReportSection] = []
    for ticker in tickers:
        evidence = _evidence_ref(ticker)
        is_candidate_ticker = include_candidate_links and ticker == "TSLA"
        sections.append(
            TickerReportSection(
                ticker=ticker,
                company_name=f"{ticker} Fixture Co.",
                discovery_refs=(f"ticker-container-{ticker.lower()}",),
                observed_discussion_summary=f"Fixture discussion for {ticker}.",
                social_news_summary=f"Fixture news summary for {ticker}.",
                strategy_clusters=(_strategy_cluster(ticker),) if is_candidate_ticker else (),
                opportunity_notes=(f"Watch {ticker} only with fresh evidence.",),
                recommendation_ids=("candidate-tsla-shares-swing",) if is_candidate_ticker else (),
                evidence=(evidence,),
                warning_ids=("stale-market-data",) if ticker == "SPY" else (),
                data_quality={"freshness": "fresh", "evidence_count": 1},
            )
        )
    return tuple(sections)


def _provider_warning() -> ProviderWarning:
    return ProviderWarning(
        code=WarningCode.STALE_DATA,
        severity=WarningSeverity.WARNING,
        message="Market data fixture is older than the freshness target.",
        provider_name="fixture-market-data",
        occurred_at=_timestamp(),
        stale_after=datetime(2026, 5, 11, 19, 45, tzinfo=UTC),
        raw_snapshot_id="raw-market-data-2026-05-11",
        source_url="https://market.example/snapshot",
        metadata={"stale_seconds": 900},
    )


def _provider_health() -> tuple[ProviderHealth, ...]:
    return (
        ProviderHealth(
            provider_name="fixture-reddit",
            status=ProviderStatus.OK,
            checked_at=_timestamp(),
            credential_state=CredentialState.NOT_REQUIRED,
            latency_ms=42,
            rate_limit_remaining=98,
            last_success_at=_timestamp(),
        ),
        ProviderHealth(
            provider_name="fixture-market-data",
            status=ProviderStatus.STALE,
            checked_at=_timestamp(),
            credential_state=CredentialState.NOT_REQUIRED,
            latency_ms=55,
            last_success_at=datetime(2026, 5, 11, 19, 30, tzinfo=UTC),
            warnings=(_provider_warning(),),
        ),
    )


def _data_freshness() -> DataFreshnessSummary:
    return DataFreshnessSummary(
        as_of=_timestamp(),
        summary="Reddit evidence is fresh; market data is stale but still traceable.",
        stale_provider_names=("fixture-market-data",),
        missing_provider_names=("fixture-x",),
    )


def _audit_manifest() -> AuditManifest:
    return AuditManifest(
        run_id="run-2026-05-11",
        schema_version="audit.v1",
        created_at=_timestamp(),
        artifacts=(
            AuditArtifact(
                artifact_id="normalized-evidence-tsla",
                artifact_type="normalized_evidence",
                path="reports/2026-05-11/audit/normalized-evidence-tsla.json",
                created_at=_timestamp(),
                produced_by="fixture-reddit-normalizer",
                sha256="0" * 64,
                record_count=1,
                metadata={
                    "ticker": "TSLA",
                    "data_reference_ids": ["reddit-normalized-tsla"],
                },
            ),
            AuditArtifact(
                artifact_id="scoring-input-tsla",
                artifact_type="scoring_input",
                path="reports/2026-05-11/audit/scoring-input-tsla.json",
                created_at=_timestamp(),
                produced_by="fixture-scorer",
                sha256="1" * 64,
                record_count=1,
                metadata={"candidate_ids": ["candidate-tsla-shares-swing"]},
            ),
        ),
        provider_run_ids=("provider-run-reddit-1", "provider-run-market-1"),
        model_versions={"strategy_extractor": "fixture-llm-schema-v1"},
        prompt_versions={"strategy_extraction": "prompt-v1"},
        config_hash="config-hash-2026-05-11",
        command_args={"date": "2026-05-11", "output": "reports/", "offline": True},
        recommendation_trace_ids=("candidate-tsla-shares-swing",),
    )


def _trade_candidate() -> TradeCandidate:
    evidence = _evidence_ref("TSLA")
    score_component = ScoreComponent(
        name="evidence_alignment",
        raw_value="3 corroborating fixture comments",
        normalized_score=0.8,
        weight=0.4,
        contribution=0.32,
        rationale="Fixture discussion and price context point in the same direction.",
        evidence=(evidence,),
        data_reference_ids=("reddit-normalized-tsla", "analysis-context-tsla"),
    )
    return TradeCandidate(
        candidate_id="candidate-tsla-shares-swing",
        ticker="TSLA",
        action=RecommendationAction.QUALIFIED,
        strategy_cluster_id="cluster-tsla-bullish-shares",
        instrument=InstrumentType.SHARES,
        direction=Direction.BULLISH,
        position_type=PositionType.LONG,
        time_horizon=TimeHorizon.SWING,
        thesis="Fixture evidence supports a small, defined-risk TSLA watchlist idea.",
        entry_logic="Only consider entry after price confirms above the fixture trigger.",
        invalidation_criteria="Invalidate if the price loses the fixture support level.",
        risk_plan=RiskAssessment(
            risk_profile=RiskProfile.EXPLORATORY,
            defined_risk=True,
            margin_required=False,
            max_account_risk_pct=Decimal("0.02"),
            account_capital=Decimal("1000"),
            max_loss_estimate=Decimal("20"),
            position_size_pct=Decimal("0.02"),
            passed=True,
            sizing_basis="Fixture risk budget caps loss at 2% of account capital.",
        ),
        catalysts=("Fixture event risk",),
        score=ScoreBreakdown(
            score_version="score.v1",
            overall_score=0.76,
            confidence=0.68,
            threshold=0.7,
            components=(score_component,),
            penalties=(),
            failed_gates=(),
        ),
        assumptions=("Fixture market data remains directionally usable.",),
        risks=("Retail discussion may be sarcastic or crowded.",),
        contradictions=("Market data freshness warning reduces confidence.",),
        evidence=(evidence,),
        score_input_ids=("scoring-input-tsla",),
        disclaimer_id="educational-report-v1",
        metadata={
            "recommendation_source": "fixture-scorer",
            "confidence_inputs": {
                "evidence_alignment": 0.8,
                "freshness_penalty": 0.1,
            },
        },
    )


def _daily_report(
    *,
    ticker_sections: tuple[TickerReportSection, ...] | None = None,
    trade_candidates: tuple[TradeCandidate, ...] = (),
    no_trade_summary: str | None = "No qualified trades passed the fixture evidence gates.",
    audit_manifest: AuditManifest | DataReference | None = None,
) -> DailyReport:
    return DailyReport(
        schema_version="report.v1",
        run_id="run-2026-05-11",
        report_date=date(2026, 5, 11),
        generated_at=_timestamp(),
        timezone="America/Los_Angeles",
        app_version="0.1.0",
        git_sha="abc1234",
        config_hash="config-hash-2026-05-11",
        command_args={"date": "2026-05-11", "output": "reports/", "offline": True},
        risk_profile=RiskProfile.EXPLORATORY,
        account_capital="1000.00",
        disclaimer=_disclaimer(),
        ticker_discovery=_ticker_discovery(),
        data_freshness=_data_freshness(),
        provider_health=_provider_health(),
        ticker_sections=ticker_sections or _ticker_sections(),
        trade_candidates=trade_candidates,
        no_trade_summary=no_trade_summary,
        audit_manifest=audit_manifest if audit_manifest is not None else _audit_manifest(),
    )


@pytest.mark.schema
def test_daily_report_requires_six_ticker_sections_in_discovery_order() -> None:
    report = _daily_report()

    assert tuple(section.ticker for section in report.ticker_sections) == TICKERS

    with pytest.raises(ValidationError, match="exactly six ticker sections"):
        _daily_report(ticker_sections=_ticker_sections(TICKERS[:5]))

    out_of_order = ("NVDA", "TSLA", "AMD", "AAPL", "MU", "SPY")
    with pytest.raises(ValidationError, match="match discovered tickers in order"):
        _daily_report(ticker_sections=_ticker_sections(out_of_order))


@pytest.mark.schema
def test_daily_report_without_candidates_requires_no_trade_summary() -> None:
    report = _daily_report(trade_candidates=(), no_trade_summary="No setup passed risk gates.")

    assert report.no_trade_summary == "No setup passed risk gates."

    with pytest.raises(ValidationError, match="no_trade_summary"):
        _daily_report(trade_candidates=(), no_trade_summary=None)


@pytest.mark.schema
def test_daily_report_validates_candidate_section_and_disclaimer_links() -> None:
    candidate = _trade_candidate()

    with pytest.raises(ValidationError, match="referenced by a ticker section"):
        _daily_report(trade_candidates=(candidate,), no_trade_summary=None)

    with pytest.raises(ValidationError, match="recommendation_ids must reference candidates"):
        _daily_report(
            ticker_sections=(
                TickerReportSection(
                    ticker="TSLA",
                    recommendation_ids=("missing-candidate",),
                ),
                *_ticker_sections(TICKERS[1:]),
            ),
            no_trade_summary="No setup passed risk gates.",
        )

    with pytest.raises(ValidationError, match="match candidate ticker"):
        _daily_report(
            ticker_sections=(
                TickerReportSection(
                    ticker="TSLA",
                ),
                TickerReportSection(ticker="NVDA", recommendation_ids=(candidate.candidate_id,)),
                *_ticker_sections(TICKERS[2:]),
            ),
            trade_candidates=(candidate,),
            no_trade_summary=None,
        )

    with pytest.raises(ValidationError, match="disclaimer_id must match"):
        _daily_report(
            ticker_sections=_ticker_sections(include_candidate_links=True),
            trade_candidates=(
                TradeCandidate.model_validate(
                    {
                        **candidate.model_dump(),
                        "disclaimer_id": "other-disclaimer",
                    }
                ),
            ),
            no_trade_summary=None,
        )

    with pytest.raises(ValidationError, match="discovered tickers"):
        _daily_report(
            ticker_sections=(
                TickerReportSection(
                    ticker="TSLA",
                    recommendation_ids=(candidate.candidate_id,),
                ),
                *_ticker_sections(TICKERS[1:]),
            ),
            trade_candidates=(
                TradeCandidate.model_validate(
                    {
                        **candidate.model_dump(),
                        "ticker": "QQQ",
                    }
                ),
            ),
            no_trade_summary=None,
        )

    with pytest.raises(ValidationError, match="ids must be unique"):
        _daily_report(
            ticker_sections=(
                TickerReportSection(
                    ticker="TSLA",
                    recommendation_ids=(candidate.candidate_id,),
                ),
                TickerReportSection(
                    ticker="NVDA",
                    recommendation_ids=(candidate.candidate_id,),
                ),
                *_ticker_sections(TICKERS[2:]),
            ),
            trade_candidates=(candidate, candidate),
            no_trade_summary=None,
        )


@pytest.mark.schema
def test_report_header_shape_serializes_disclaimer_health_and_freshness() -> None:
    report = _daily_report()
    dumped = report.model_dump(mode="json")

    assert dumped["disclaimer"] == {
        "disclaimer_id": "educational-report-v1",
        "version": "2026-05-11",
        "text": "Educational research only; not financial advice; no automatic trading.",
        "educational_only": True,
        "not_financial_advice": True,
        "no_auto_trading": True,
        "applies_to": ["report", "trade_candidates"],
    }
    assert dumped["data_freshness"] == {
        "as_of": "2026-05-11T20:00:00Z",
        "summary": "Reddit evidence is fresh; market data is stale but still traceable.",
        "stale_provider_names": ["fixture-market-data"],
        "missing_provider_names": ["fixture-x"],
    }
    assert dumped["provider_health"][0]["provider_name"] == "fixture-reddit"
    assert dumped["provider_health"][0]["status"] == "ok"
    assert dumped["provider_health"][1]["status"] == "stale"
    assert dumped["provider_health"][1]["warnings"][0]["code"] == "stale_data"

    with pytest.raises(ValidationError, match="prohibit auto-trading"):
        Disclaimer(
            disclaimer_id="bad-disclaimer",
            version="2026-05-11",
            text="Missing the v1 no-auto-trading guardrail.",
            no_auto_trading=False,
        )


@pytest.mark.schema
def test_default_markdown_report_outline_freezes_required_section_shape() -> None:
    outline = DEFAULT_MARKDOWN_REPORT_OUTLINE
    dumped = outline.model_dump(mode="json")

    assert dumped["schema_version"] == "markdown-report.v1"
    assert dumped["heading_order"] == [
        "Daily Stock Opportunity Report",
        "Data Freshness",
        "Provider Warnings",
        "Ticker Sections",
        "Qualified Trading Strategies Or No-Trade Summary",
        "Disclaimer",
        "Audit Artifacts",
    ]
    assert outline.ticker_section_heading_template == "{ticker}"
    assert "Evidence References" in outline.required_ticker_subsections
    assert outline.final_section_headings == (
        "Qualified Trading Strategies",
        "No-Trade Summary",
    )
    assert outline.require_disclaimer is True
    assert outline.require_evidence_references is True
    assert outline.require_audit_artifacts is True


@pytest.mark.schema
def test_audit_manifest_serializes_artifacts_and_report_trace_ids() -> None:
    report = _daily_report(audit_manifest=_audit_manifest())
    dumped = report.model_dump(mode="json")

    manifest = dumped["audit_manifest"]
    assert manifest["run_id"] == "run-2026-05-11"
    assert manifest["provider_run_ids"] == ["provider-run-reddit-1", "provider-run-market-1"]
    assert manifest["recommendation_trace_ids"] == ["candidate-tsla-shares-swing"]
    assert manifest["artifacts"][0]["artifact_type"] == "normalized_evidence"
    assert manifest["artifacts"][0]["metadata"]["data_reference_ids"] == ["reddit-normalized-tsla"]
    assert manifest["artifacts"][1]["artifact_type"] == "scoring_input"


@pytest.mark.schema
def test_report_can_point_to_external_audit_manifest_data_reference() -> None:
    audit_reference = DataReference(
        reference_id="audit-manifest-json",
        reference_type="audit_artifact",
        path="reports/2026-05-11/audit/manifest.json",
        sha256="2" * 64,
        metadata={"run_id": "run-2026-05-11"},
    )

    dumped = _daily_report(audit_manifest=audit_reference).model_dump(mode="json")

    assert dumped["audit_manifest"] == {
        "reference_id": "audit-manifest-json",
        "reference_type": "audit_artifact",
        "path": "reports/2026-05-11/audit/manifest.json",
        "sha256": "2" * 64,
        "metadata": {"run_id": "run-2026-05-11"},
    }


@pytest.mark.schema
def test_json_round_trip_preserves_evidence_and_recommendation_inputs() -> None:
    report = _daily_report(
        ticker_sections=_ticker_sections(include_candidate_links=True),
        trade_candidates=(_trade_candidate(),),
        no_trade_summary=None,
    )

    dumped = report.model_dump(mode="json")
    round_tripped = DailyReport.model_validate_json(report.model_dump_json())

    assert dumped["ticker_sections"][0]["ticker"] == "TSLA"
    assert dumped["ticker_sections"][0]["recommendation_ids"] == ["candidate-tsla-shares-swing"]
    assert dumped["ticker_sections"][0]["strategy_clusters"][0]["evidence"][0] == {
        "evidence_id": "reddit-tsla-comment-1",
        "quote": "$TSLA momentum setup discussed in fixture comments",
        "start_char": 12,
        "end_char": 48,
        "relevance": 0.82,
    }

    candidate = dumped["trade_candidates"][0]
    assert candidate["action"] == "qualified"
    assert candidate["disclaimer_id"] == dumped["disclaimer"]["disclaimer_id"]
    assert candidate["evidence"][0]["evidence_id"] == "reddit-tsla-comment-1"
    assert candidate["score"]["components"][0]["data_reference_ids"] == [
        "reddit-normalized-tsla",
        "analysis-context-tsla",
    ]
    assert candidate["score"]["components"][0]["evidence"][0]["quote"] == (
        "$TSLA momentum setup discussed in fixture comments"
    )
    assert candidate["score_input_ids"] == ["scoring-input-tsla"]
    assert candidate["metadata"]["confidence_inputs"]["freshness_penalty"] == 0.1

    assert round_tripped.trade_candidates[0].candidate_id == "candidate-tsla-shares-swing"
    assert round_tripped.trade_candidates[0].evidence == (_evidence_ref("TSLA"),)
    assert round_tripped.ticker_sections[0].strategy_clusters[0].evidence == (
        _evidence_ref("TSLA"),
    )
