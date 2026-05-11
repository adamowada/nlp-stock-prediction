"""Deterministic offline report fixtures for Lane E orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

from nlp_stock_prediction import __version__
from nlp_stock_prediction.contracts import (
    AnalysisBundle,
    AnalysisSignal,
    AuditArtifact,
    AuditManifest,
    CredentialState,
    DailyReport,
    DataFreshnessSummary,
    Direction,
    Disclaimer,
    EvidenceReference,
    FreshnessStatus,
    FundamentalAnalysis,
    InstrumentType,
    JsonObject,
    MacroContext,
    PositionType,
    ProviderHealth,
    ProviderStatus,
    ProviderWarning,
    RecommendationAction,
    RetrievalMethod,
    RiskAssessment,
    ScoreBreakdown,
    ScoreComponent,
    SectorContext,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    StrategyCluster,
    StrategyExtraction,
    TechnicalAnalysis,
    TickerCandidate,
    TickerDiscoveryResult,
    TickerDiscoveryStatus,
    TickerReportSection,
    TimeHorizon,
    TradeCandidate,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.reporting.audit import json_payload_sha256

TICKERS: tuple[str, ...] = ("TSLA", "NVDA", "AMD", "AAPL", "MU", "SPY")

COMPANIES: dict[str, str] = {
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "AMD": "Advanced Micro Devices Inc.",
    "AAPL": "Apple Inc.",
    "MU": "Micron Technology Inc.",
    "SPY": "SPDR S&P 500 ETF Trust",
}

SECTORS: dict[str, str] = {
    "TSLA": "Consumer Discretionary",
    "NVDA": "Information Technology",
    "AMD": "Information Technology",
    "AAPL": "Information Technology",
    "MU": "Information Technology",
    "SPY": "Broad Market ETF",
}

BASE_PRICES: dict[str, Decimal] = {
    "TSLA": Decimal("184.50"),
    "NVDA": Decimal("119.30"),
    "AMD": Decimal("156.20"),
    "AAPL": Decimal("213.80"),
    "MU": Decimal("124.10"),
    "SPY": Decimal("531.40"),
}


@dataclass(frozen=True)
class OfflineFixtureBundle:
    """Pure fixture payloads plus a contract-valid report."""

    report: DailyReport
    audit_payloads: dict[str, JsonObject]


def build_offline_fixture_report(config: RunConfig) -> DailyReport:
    return build_offline_fixture_bundle(config).report


def build_offline_fixture_bundle(config: RunConfig) -> OfflineFixtureBundle:
    generated_at = _timestamp(config.run_date)
    report_dir = config.output_dir / config.run_date.isoformat()
    audit_dir = report_dir / "audit"
    run_id = f"run-{config.run_date.isoformat()}-offline-fixture"

    evidence_records = _source_evidence_records(config.run_date, generated_at)
    evidence_by_ticker = {record.ticker or "": record for record in evidence_records}
    evidence_refs = {
        ticker: _evidence_ref(ticker, evidence_by_ticker[ticker]) for ticker in TICKERS
    }
    discovery = _ticker_discovery(config.run_date, generated_at)
    strategy_extractions = _strategy_extractions(evidence_refs)
    strategy_clusters = _strategy_clusters(evidence_refs)
    analysis_bundles = _analysis_bundles(config.run_date, evidence_refs)
    provider_health = _provider_health(config.run_date, generated_at)

    trade_candidate = _trade_candidate(config, evidence_refs["TSLA"])
    ticker_sections = _ticker_sections(
        run_date=config.run_date,
        strategy_clusters=strategy_clusters,
        analysis_bundles=analysis_bundles,
        evidence_refs=evidence_refs,
    )

    audit_payloads = _audit_payloads(
        run_id=run_id,
        config=config,
        evidence_records=evidence_records,
        strategy_extractions=strategy_extractions,
        analysis_bundles=analysis_bundles,
        trade_candidate=trade_candidate,
        report_dir=report_dir,
        generated_at=generated_at,
    )
    audit_manifest = _audit_manifest(
        run_id=run_id,
        config=config,
        audit_dir=audit_dir,
        report_dir=report_dir,
        generated_at=generated_at,
        audit_payloads=audit_payloads,
        recommendation_trace_ids=(trade_candidate.candidate_id,),
    )

    report = DailyReport(
        schema_version="report.v1",
        run_id=run_id,
        report_date=config.run_date,
        generated_at=generated_at,
        timezone="America/Los_Angeles",
        app_version=__version__,
        git_sha="phase-1-contract-gate",
        config_hash=f"offline-fixture-{config.run_date.isoformat()}",
        command_args=_command_args(config),
        risk_profile=config.risk_profile,
        account_capital=_format_capital(config.capital),
        disclaimer=_disclaimer(config.run_date),
        ticker_discovery=discovery,
        data_freshness=DataFreshnessSummary(
            as_of=generated_at,
            summary=(
                "Deterministic offline fixture data is fresh for Reddit/news context; "
                "market data includes one stale fixture warning."
            ),
            stale_provider_names=("fixture-market-data",),
            missing_provider_names=(),
        ),
        provider_health=provider_health,
        ticker_sections=ticker_sections,
        trade_candidates=(trade_candidate,),
        no_trade_summary=None,
        audit_manifest=audit_manifest,
    )
    return OfflineFixtureBundle(report=report, audit_payloads=audit_payloads)


def _timestamp(run_date: date) -> datetime:
    return datetime.combine(run_date, time(20, 0), tzinfo=UTC)


def _command_args(config: RunConfig) -> JsonObject:
    return {
        "date": config.run_date.isoformat(),
        "output": _posix(config.output_dir),
        "capital": _format_capital(config.capital),
        "risk_profile": config.risk_profile.value,
        "fixture_dir": _posix(config.fixture_dir) if config.fixture_dir else None,
        "cache_dir": _posix(config.cache_dir) if config.cache_dir else None,
        "offline": config.offline,
    }


def _disclaimer(run_date: date) -> Disclaimer:
    return Disclaimer(
        disclaimer_id="educational-report-v1",
        version=run_date.isoformat(),
        text=(
            "Educational research only. This is not financial advice. "
            "No automatic trading or brokerage execution is performed."
        ),
    )


def _ticker_discovery(run_date: date, fetched_at: datetime) -> TickerDiscoveryResult:
    candidates = tuple(
        TickerCandidate(
            symbol=ticker,
            raw_identifier=f"ticker-container-{ticker.lower()}",
            raw_text=ticker,
            first_seen_rank=rank,
            source_url="https://reddit.example/r/wallstreetbets/daily-ticker-card",
            provenance=_provenance(
                provider_name="fixture-reddit",
                source_kind=SourceKind.REDDIT_TICKER_CARD,
                fetched_at=fetched_at,
                source_url="https://reddit.example/r/wallstreetbets/daily-ticker-card",
                raw_identifier=f"ticker-container-{ticker.lower()}",
                raw_snapshot_id=f"raw-reddit-card-{run_date.isoformat()}",
                query="r/wallstreetbets Devvit daily ticker card",
            ),
        )
        for rank, ticker in enumerate(TICKERS)
    )
    return TickerDiscoveryResult(
        run_date=run_date,
        status=TickerDiscoveryStatus.VALID,
        candidates=candidates,
        tickers=TICKERS,
        raw_snapshot_id=f"raw-reddit-card-{run_date.isoformat()}",
    )


def _source_evidence_records(run_date: date, fetched_at: datetime) -> tuple[SourceEvidence, ...]:
    records: list[SourceEvidence] = []
    for ticker in TICKERS:
        text = (
            f"${ticker} bullish swing setup discussed with a defined-risk entry trigger "
            f"and fresh catalyst watch."
        )
        records.append(
            SourceEvidence(
                evidence_id=f"evidence-{ticker.lower()}-reddit-1",
                source_kind=SourceKind.REDDIT_COMMENT,
                ticker=ticker,
                title=f"{ticker} daily fixture discussion",
                text=text,
                author_hash=f"fixture-author-{ticker.lower()}",
                created_at=fetched_at,
                score=42,
                permalink=(
                    "https://reddit.example/r/wallstreetbets/comments/"
                    f"{run_date.isoformat()}/{ticker.lower()}"
                ),
                matched_tickers=(ticker,),
                provenance=_provenance(
                    provider_name="fixture-reddit",
                    source_kind=SourceKind.REDDIT_COMMENT,
                    fetched_at=fetched_at,
                    permalink=(
                        "https://reddit.example/r/wallstreetbets/comments/"
                        f"{run_date.isoformat()}/{ticker.lower()}"
                    ),
                    raw_identifier=f"reddit-comment-{ticker.lower()}-1",
                    raw_snapshot_id=f"raw-reddit-discussion-{run_date.isoformat()}",
                    query=f"{ticker} r/wallstreetbets daily discussion",
                ),
                metadata={
                    "fixture": True,
                    "tone": "speculative",
                    "provider_record_rank": TICKERS.index(ticker) + 1,
                },
            )
        )
    return tuple(records)


def _evidence_ref(ticker: str, evidence: SourceEvidence) -> EvidenceReference:
    quote = f"${ticker} bullish swing setup discussed with a defined-risk entry trigger"
    return EvidenceReference(
        evidence_id=evidence.evidence_id,
        quote=quote,
        start_char=0,
        end_char=len(quote),
        relevance=0.84 if ticker == "TSLA" else 0.72,
    )


def _strategy_extractions(
    evidence_refs: dict[str, EvidenceReference],
) -> tuple[StrategyExtraction, ...]:
    return tuple(
        StrategyExtraction(
            strategy_id=f"strategy-{ticker.lower()}-shares-swing",
            ticker=ticker,
            label=f"{ticker} bullish shares swing",
            direction=Direction.BULLISH,
            instrument=InstrumentType.SHARES,
            position_type=PositionType.LONG,
            time_horizon=TimeHorizon.SWING,
            catalyst="Retail discussion cites momentum plus near-term catalyst watch.",
            risk_or_hedge="Use defined risk, no margin, and invalidate on loss of support.",
            slang_terms=("shares", "swing", "watchlist"),
            evidence=(evidence_refs[ticker],),
            confidence=0.78 if ticker == "TSLA" else 0.64,
            sarcasm_joke_risk=0.08 if ticker == "TSLA" else 0.18,
        )
        for ticker in TICKERS
    )


def _strategy_clusters(evidence_refs: dict[str, EvidenceReference]) -> dict[str, StrategyCluster]:
    return {
        ticker: StrategyCluster(
            cluster_id=f"cluster-{ticker.lower()}-shares-swing",
            ticker=ticker,
            direction=Direction.BULLISH,
            instrument=InstrumentType.SHARES,
            time_horizon=TimeHorizon.SWING,
            catalyst_summary="Momentum and catalyst-watch discussion cluster.",
            member_strategy_ids=(f"strategy-{ticker.lower()}-shares-swing",),
            evidence=(evidence_refs[ticker],),
            confidence=0.76 if ticker == "TSLA" else 0.61,
        )
        for ticker in TICKERS
    }


def _analysis_bundles(
    run_date: date,
    evidence_refs: dict[str, EvidenceReference],
) -> dict[str, AnalysisBundle]:
    bundles: dict[str, AnalysisBundle] = {}
    for ticker in TICKERS:
        technical = TechnicalAnalysis(
            ticker=ticker,
            summary=(
                f"{ticker} fixture price action is above short-term support with "
                "volume firm enough for a watchlist setup."
            ),
            signal=AnalysisSignal.SUPPORTS if ticker == "TSLA" else AnalysisSignal.MIXED,
            confidence=0.72 if ticker == "TSLA" else 0.55,
            evidence=(evidence_refs[ticker],),
            trend="short-term uptrend" if ticker == "TSLA" else "range with mixed momentum",
            support_levels=(BASE_PRICES[ticker] * Decimal("0.96"),),
            resistance_levels=(BASE_PRICES[ticker] * Decimal("1.05"),),
            volume_summary="Fixture relative volume is above the 20-day baseline.",
            volatility_summary="Volatility is elevated; position sizing should stay small.",
            gap_summary="No unfilled gap is used as a recommendation input.",
            candlestick_summary="Latest fixture candle closed near the upper half of its range.",
        )
        fundamental = FundamentalAnalysis(
            ticker=ticker,
            summary=(
                f"{COMPANIES[ticker]} fundamentals are represented by deterministic "
                "fixture metrics until Lane B/D providers are integrated."
            ),
            signal=AnalysisSignal.MIXED,
            confidence=0.52,
            evidence=(evidence_refs[ticker],),
            valuation_summary="Valuation is treated as a risk, not a standalone buy signal.",
            profitability_summary="Profitability context is positive but not decisive.",
            growth_summary="Growth context is watchlist-supportive in the fixture.",
            balance_sheet_risk="No balance-sheet emergency is present in fixture data.",
            earnings_timing="No imminent earnings entry is assumed by this report.",
            notable_filings=("Fixture filing context only; replace with SEC data later.",),
        )
        sector = SectorContext(
            ticker=ticker,
            summary=f"{SECTORS[ticker]} context is mixed but not disqualifying.",
            signal=AnalysisSignal.MIXED,
            confidence=0.5,
            evidence=(evidence_refs[ticker],),
            sector=SECTORS[ticker],
            peers=_peers_for(ticker),
            benchmark_symbol="XLK" if ticker in {"NVDA", "AMD", "AAPL", "MU"} else "SPY",
        )
        macro = MacroContext(
            as_of=run_date,
            summary="Macro backdrop is supportive of defined-risk watchlist ideas only.",
            signal=AnalysisSignal.MIXED,
            confidence=0.58,
            evidence=(evidence_refs[ticker],),
            horizon=TimeHorizon.SWING,
            supportive_factors=("Liquidity conditions are neutral in fixture data.",),
            conflicting_factors=("Rates and inflation surprises can pressure risk appetite.",),
        )
        bundles[ticker] = AnalysisBundle(
            analysis_id=f"analysis-{ticker.lower()}-offline",
            ticker=ticker,
            as_of=run_date,
            strategy_cluster_ids=(f"cluster-{ticker.lower()}-shares-swing",),
            technical=technical,
            fundamental=fundamental,
            sector=sector,
            macro=macro,
            signals=(technical.signal, fundamental.signal, sector.signal, macro.signal),
            contradictions=("Fixture fundamentals are not strong enough alone.",),
            assumptions=("Offline fixture data stands in for providers owned by other lanes.",),
            confidence_inputs={
                "technical": technical.confidence,
                "fundamental": fundamental.confidence,
                "sector": sector.confidence,
                "macro": macro.confidence,
            },
            evidence=(evidence_refs[ticker],),
        )
    return bundles


def _trade_candidate(config: RunConfig, evidence: EvidenceReference) -> TradeCandidate:
    account_capital = config.capital
    max_loss = (
        (account_capital * Decimal("0.01")).quantize(Decimal("0.01"))
        if account_capital is not None
        else None
    )
    risk_plan = RiskAssessment(
        risk_profile=config.risk_profile,
        defined_risk=True,
        margin_required=False,
        max_account_risk_pct=Decimal("0.01"),
        account_capital=account_capital,
        max_loss_estimate=max_loss,
        position_size_pct=Decimal("0.01"),
        passed=True,
        sizing_basis=(
            f"Maximum fixture loss is capped near 1% of ${_format_capital(account_capital)}."
            if account_capital is not None
            else "No account capital provided; show sizing as a maximum 1% risk budget."
        ),
    )
    score_component = ScoreComponent(
        name="evidence_alignment",
        raw_value="Reddit strategy, technical context, and catalyst watch align.",
        normalized_score=0.8,
        weight=0.45,
        contribution=0.36,
        rationale="Observed discussion and fixture analysis point in the same direction.",
        evidence=(evidence,),
        data_reference_ids=("normalized-evidence", "analysis-tsla-offline"),
    )
    technical_component = ScoreComponent(
        name="technical_alignment",
        raw_value="Fixture trend holds above support.",
        normalized_score=0.72,
        weight=0.35,
        contribution=0.252,
        rationale="Price context supports a watchlist idea but still needs confirmation.",
        evidence=(evidence,),
        data_reference_ids=("analysis-tsla-offline",),
    )
    freshness_penalty = ScoreComponent(
        name="freshness_penalty",
        raw_value="Market fixture includes a stale-data warning.",
        normalized_score=0.05,
        weight=0.2,
        contribution=-0.01,
        rationale="A stale fixture warning trims confidence until live providers land.",
        evidence=(evidence,),
        warning_ids=("fixture-market-data:stale_data",),
    )
    return TradeCandidate(
        candidate_id="candidate-tsla-shares-swing",
        ticker="TSLA",
        action=RecommendationAction.QUALIFIED,
        strategy_cluster_id="cluster-tsla-shares-swing",
        instrument=InstrumentType.SHARES,
        direction=Direction.BULLISH,
        position_type=PositionType.LONG,
        time_horizon=TimeHorizon.SWING,
        thesis="TSLA has the strongest fixture-backed discussion and analysis alignment.",
        entry_logic="Only consider after price confirms above the fixture trigger level.",
        invalidation_criteria="Invalidate if TSLA loses fixture support or catalyst tone reverses.",
        risk_plan=risk_plan,
        catalysts=("Retail momentum discussion", "near-term catalyst watch"),
        score=ScoreBreakdown(
            score_version="offline-score.v1",
            overall_score=0.74,
            confidence=0.67,
            threshold=0.7,
            components=(score_component, technical_component),
            penalties=(freshness_penalty,),
        ),
        assumptions=("Offline fixtures represent provider outputs until lanes A-D land.",),
        risks=("Retail discussion can be crowded, stale, sarcastic, or wrong.",),
        contradictions=("Market fixture freshness warning reduces confidence.",),
        evidence=(evidence,),
        score_input_ids=("scoring-input-tsla",),
        disclaimer_id="educational-report-v1",
        metadata={
            "recommendation_source": "offline-fixture-scorer",
            "confidence_inputs": {
                "reddit_strategy_confidence": 0.78,
                "technical_alignment": 0.72,
                "freshness_penalty": 0.05,
            },
        },
    )


def _ticker_sections(
    *,
    run_date: date,
    strategy_clusters: dict[str, StrategyCluster],
    analysis_bundles: dict[str, AnalysisBundle],
    evidence_refs: dict[str, EvidenceReference],
) -> tuple[TickerReportSection, ...]:
    sections: list[TickerReportSection] = []
    for ticker in TICKERS:
        bundle = analysis_bundles[ticker]
        sections.append(
            TickerReportSection(
                ticker=ticker,
                company_name=COMPANIES[ticker],
                discovery_refs=(f"ticker-container-{ticker.lower()}",),
                observed_discussion_summary=(
                    f"Reddit fixture evidence describes a speculative {ticker} "
                    "swing setup; this is reported as observed discussion, not fact."
                ),
                social_news_summary=(
                    f"Offline social/news fixture for {ticker} is neutral-to-watchlist; "
                    "live provider confirmation is a later integration point."
                ),
                strategy_clusters=(strategy_clusters[ticker],),
                technical_analysis=bundle.technical,
                fundamental_analysis=bundle.fundamental,
                sector_context=bundle.sector,
                macro_context=bundle.macro,
                opportunity_notes=_opportunity_notes(ticker),
                recommendation_ids=("candidate-tsla-shares-swing",) if ticker == "TSLA" else (),
                evidence=(evidence_refs[ticker],),
                warning_ids=("fixture-market-data:stale_data",),
                data_quality={
                    "freshness": "fresh_with_stale_market_warning",
                    "evidence_count": 1,
                    "provider_names": [
                        "fixture-reddit",
                        "fixture-news",
                        "fixture-market-data",
                        "fixture-macro",
                    ],
                    "raw_snapshot_ids": [f"raw-reddit-card-{run_date.isoformat()}"],
                    "confidence": 0.74 if ticker == "TSLA" else 0.58,
                },
            )
        )
    return tuple(sections)


def _provider_health(run_date: date, generated_at: datetime) -> tuple[ProviderHealth, ...]:
    stale_warning = ProviderWarning(
        code=WarningCode.STALE_DATA,
        severity=WarningSeverity.WARNING,
        message="Offline market fixture is older than the freshness target.",
        provider_name="fixture-market-data",
        retryable=False,
        occurred_at=generated_at,
        stale_after=generated_at,
        raw_snapshot_id=f"raw-market-data-{run_date.isoformat()}",
        source_url="https://market.example/fixture",
        metadata={"fallback": "offline_fixture", "stale_seconds": 900},
    )
    return (
        ProviderHealth(
            provider_name="fixture-reddit",
            status=ProviderStatus.OK,
            checked_at=generated_at,
            credential_state=CredentialState.NOT_REQUIRED,
            latency_ms=0,
            rate_limit_remaining=999,
            last_success_at=generated_at,
        ),
        ProviderHealth(
            provider_name="fixture-market-data",
            status=ProviderStatus.STALE,
            checked_at=generated_at,
            credential_state=CredentialState.NOT_REQUIRED,
            latency_ms=0,
            last_success_at=generated_at,
            warnings=(stale_warning,),
        ),
        ProviderHealth(
            provider_name="fixture-news",
            status=ProviderStatus.OK,
            checked_at=generated_at,
            credential_state=CredentialState.NOT_REQUIRED,
            latency_ms=0,
            last_success_at=generated_at,
        ),
        ProviderHealth(
            provider_name="fixture-macro",
            status=ProviderStatus.OK,
            checked_at=generated_at,
            credential_state=CredentialState.NOT_REQUIRED,
            latency_ms=0,
            last_success_at=generated_at,
        ),
    )


def _audit_payloads(
    *,
    run_id: str,
    config: RunConfig,
    evidence_records: tuple[SourceEvidence, ...],
    strategy_extractions: tuple[StrategyExtraction, ...],
    analysis_bundles: dict[str, AnalysisBundle],
    trade_candidate: TradeCandidate,
    report_dir: Path,
    generated_at: datetime,
) -> dict[str, JsonObject]:
    raw_snapshots: JsonObject = {
        "schema_version": "audit.raw_snapshots.v1",
        "run_id": run_id,
        "created_at": generated_at.isoformat(),
        "records": [
            {
                "raw_snapshot_id": f"raw-reddit-card-{config.run_date.isoformat()}",
                "provider_name": "fixture-reddit",
                "source_kind": SourceKind.REDDIT_TICKER_CARD.value,
                "content_type": "application/json",
                "payload": {"tickers": list(TICKERS), "source": "offline_fixture"},
                "provider_metadata": {"fixture": True, "redacted": True},
            },
            {
                "raw_snapshot_id": f"raw-reddit-discussion-{config.run_date.isoformat()}",
                "provider_name": "fixture-reddit",
                "source_kind": SourceKind.REDDIT_COMMENT.value,
                "content_type": "application/json",
                "payload": {"evidence_ids": [record.evidence_id for record in evidence_records]},
                "provider_metadata": {"fixture": True, "redacted": True},
            },
            {
                "raw_snapshot_id": f"raw-market-data-{config.run_date.isoformat()}",
                "provider_name": "fixture-market-data",
                "source_kind": SourceKind.MARKET_DATA.value,
                "content_type": "application/json",
                "payload": {"stale": True, "tickers": list(TICKERS)},
                "provider_metadata": {"fixture": True, "freshness_warning": "stale_data"},
            },
        ],
    }
    normalized_evidence: JsonObject = {
        "schema_version": "audit.normalized_evidence.v1",
        "run_id": run_id,
        "records": [
            cast(JsonObject, record.model_dump(mode="json")) for record in evidence_records
        ],
    }
    extracted_strategies: JsonObject = {
        "schema_version": "audit.extracted_strategies.v1",
        "run_id": run_id,
        "records": [
            cast(JsonObject, extraction.model_dump(mode="json"))
            for extraction in strategy_extractions
        ],
    }
    analysis_contexts: JsonObject = {
        "schema_version": "audit.analysis_contexts.v1",
        "run_id": run_id,
        "records": [
            cast(JsonObject, analysis_bundles[ticker].model_dump(mode="json")) for ticker in TICKERS
        ],
    }
    scoring_inputs: JsonObject = {
        "schema_version": "audit.scoring_inputs.v1",
        "run_id": run_id,
        "records": [
            {
                "scoring_input_id": "scoring-input-tsla",
                "candidate_id": trade_candidate.candidate_id,
                "ticker": trade_candidate.ticker,
                "score": cast(JsonObject, trade_candidate.score.model_dump(mode="json")),
                "risk_plan": cast(JsonObject, trade_candidate.risk_plan.model_dump(mode="json")),
                "evidence_ids": [reference.evidence_id for reference in trade_candidate.evidence],
                "confidence_inputs": trade_candidate.metadata["confidence_inputs"],
            }
        ],
    }
    final_reports: JsonObject = {
        "schema_version": "audit.final_reports.v1",
        "run_id": run_id,
        "records": [
            {
                "artifact_id": "markdown-report",
                "path": _posix(report_dir / "report.md"),
                "content_type": "text/markdown",
            },
            {
                "artifact_id": "json-report",
                "path": _posix(report_dir / "report.json"),
                "content_type": "application/json",
            },
        ],
    }
    return {
        "raw-snapshots.json": raw_snapshots,
        "normalized-evidence.json": normalized_evidence,
        "extracted-strategies.json": extracted_strategies,
        "analysis-contexts.json": analysis_contexts,
        "scoring-inputs.json": scoring_inputs,
        "final-reports.json": final_reports,
    }


def _audit_manifest(
    *,
    run_id: str,
    config: RunConfig,
    audit_dir: Path,
    report_dir: Path,
    generated_at: datetime,
    audit_payloads: dict[str, JsonObject],
    recommendation_trace_ids: tuple[str, ...],
) -> AuditManifest:
    artifact_specs: tuple[
        tuple[
            str,
            Literal[
                "raw_snapshot",
                "normalized_evidence",
                "extraction_output",
                "analysis_context",
                "scoring_input",
            ],
            str,
            str,
        ],
        ...,
    ] = (
        (
            "raw-snapshots",
            "raw_snapshot",
            "raw-snapshots.json",
            "offline-fixture-provider",
        ),
        (
            "normalized-evidence",
            "normalized_evidence",
            "normalized-evidence.json",
            "offline-fixture-normalizer",
        ),
        (
            "extracted-strategies",
            "extraction_output",
            "extracted-strategies.json",
            "offline-fixture-extractor",
        ),
        (
            "analysis-contexts",
            "analysis_context",
            "analysis-contexts.json",
            "offline-fixture-analysis",
        ),
        (
            "scoring-inputs",
            "scoring_input",
            "scoring-inputs.json",
            "offline-fixture-scorer",
        ),
    )
    artifacts = [
        _artifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            path=audit_dir / filename,
            created_at=generated_at,
            produced_by=produced_by,
            sha256=json_payload_sha256(audit_payloads[filename]),
            record_count=_record_count(audit_payloads[filename]),
        )
        for artifact_id, artifact_type, filename, produced_by in artifact_specs
    ]
    artifacts.extend(
        [
            _artifact(
                artifact_id="markdown-report",
                artifact_type="markdown_report",
                path=report_dir / "report.md",
                created_at=generated_at,
                produced_by="lane-e-markdown-renderer",
                record_count=1,
            ),
            _artifact(
                artifact_id="json-report",
                artifact_type="json_report",
                path=report_dir / "report.json",
                created_at=generated_at,
                produced_by="lane-e-json-renderer",
                record_count=1,
            ),
        ]
    )
    return AuditManifest(
        run_id=run_id,
        schema_version="audit.v1",
        created_at=generated_at,
        artifacts=tuple(artifacts),
        provider_run_ids=(
            f"fixture-reddit-{config.run_date.isoformat()}",
            f"fixture-market-data-{config.run_date.isoformat()}",
            f"fixture-news-{config.run_date.isoformat()}",
            f"fixture-macro-{config.run_date.isoformat()}",
        ),
        model_versions={"strategy_extractor": "offline-fixture-v1"},
        prompt_versions={"strategy_extraction": "offline-fixture-prompt-v1"},
        config_hash=f"offline-fixture-{config.run_date.isoformat()}",
        command_args=_command_args(config),
        recommendation_trace_ids=recommendation_trace_ids,
    )


def _artifact(
    *,
    artifact_id: str,
    artifact_type: Literal[
        "raw_snapshot",
        "normalized_evidence",
        "extraction_output",
        "analysis_context",
        "scoring_input",
        "markdown_report",
        "json_report",
    ],
    path: Path,
    created_at: datetime,
    produced_by: str,
    sha256: str | None = None,
    record_count: int | None = None,
) -> AuditArtifact:
    return AuditArtifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        path=_posix(path),
        created_at=created_at,
        produced_by=produced_by,
        sha256=sha256,
        record_count=record_count,
        metadata={"lane": "lane-e", "fixture": True},
    )


def _record_count(payload: JsonObject) -> int | None:
    records = payload.get("records")
    if isinstance(records, list):
        return len(records)
    return None


def _provenance(
    *,
    provider_name: str,
    source_kind: SourceKind,
    fetched_at: datetime,
    raw_identifier: str,
    raw_snapshot_id: str,
    source_url: str | None = None,
    permalink: str | None = None,
    query: str,
) -> SourceProvenance:
    return SourceProvenance(
        provider_name=provider_name,
        source_kind=source_kind,
        retrieval_method=RetrievalMethod.FIXTURE,
        fetched_at=fetched_at,
        observed_at=fetched_at,
        source_url=source_url,
        permalink=permalink,
        raw_identifier=raw_identifier,
        raw_snapshot_id=raw_snapshot_id,
        query=query,
        cache_key=f"{provider_name}:{raw_snapshot_id}:{raw_identifier}",
        freshness_status=FreshnessStatus.FRESH,
        freshness_seconds=300,
        provider_metadata={"fixture": True, "offline": True},
    )


def _opportunity_notes(ticker: str) -> tuple[str, ...]:
    if ticker == "TSLA":
        return (
            "Qualified only as an educational, defined-risk watchlist idea.",
            "Wait for confirmation; do not treat Reddit discussion as fact.",
        )
    return (
        "No standalone qualified candidate from this offline fixture.",
        "Use later provider/scoring lanes before upgrading from watchlist status.",
    )


def _peers_for(ticker: str) -> tuple[str, ...]:
    if ticker == "TSLA":
        return ("GM", "F")
    if ticker == "SPY":
        return ("QQQ", "DIA")
    return tuple(peer for peer in ("NVDA", "AMD", "AAPL", "MU") if peer != ticker)[:3]


def _format_capital(capital: Decimal | None) -> str | None:
    if capital is None:
        return None
    return str(capital.quantize(Decimal("0.01")))


def _posix(path: Path) -> str:
    return path.as_posix()


__all__ = [
    "OfflineFixtureBundle",
    "build_offline_fixture_bundle",
    "build_offline_fixture_report",
]
