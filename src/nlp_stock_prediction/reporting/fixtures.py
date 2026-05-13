"""Deterministic offline report fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    AssetClass,
    AuditManifest,
    CredentialState,
    DailyReport,
    DataFreshnessSummary,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    Instrument,
    InstrumentReportSection,
    JsonObject,
    PredictionCandidate,
    PredictionStatus,
    ProviderHealth,
    RetrievalMethod,
    RunConfig,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TechnicalAnalysis,
    TimeHorizon,
    TradabilityStatus,
)
from nlp_stock_prediction.contracts.instruments import (
    InstrumentDataAvailability,
    ProviderInstrumentId,
    TradabilityEvidence,
)


@dataclass(frozen=True)
class OfflineFixtureBundle:
    report: DailyReport
    audit_payloads: dict[str, JsonObject]


def build_offline_fixture_bundle(config: RunConfig) -> OfflineFixtureBundle:
    """Build a deterministic report bundle without network or provider calls."""

    generated_at = datetime(
        config.run_date.year,
        config.run_date.month,
        config.run_date.day,
        21,
        0,
        tzinfo=UTC,
    )
    evidence = _source_evidence(generated_at)
    instruments = _instruments(generated_at)
    evidence_refs = {
        record.evidence_id: EvidenceReference(
            evidence_id=record.evidence_id,
            quote=record.text[:140],
            relevance=0.84,
        )
        for record in evidence
    }
    sections = (
        InstrumentReportSection(
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            display_name="Tesla Inc.",
            observed_discussion_summary=(
                "Fixture evidence notes delivery-cycle uncertainty, margin pressure, and "
                "continued retail attention around autonomy narratives."
            ),
            social_news_summary=(
                "Provider fixtures include conflicting sentiment rather than a one-way thesis."
            ),
            technical_analysis=TechnicalAnalysis(
                ticker="TSLA",
                summary=(
                    "Fixture technical context is mixed and should be treated as baseline "
                    "context, not a standalone signal."
                ),
                signal=AnalysisSignal.MIXED,
                confidence=0.42,
                evidence=(evidence_refs["fixture-news-tsla-001"],),
                assumptions=("Offline fixture data is synthetic and deterministic.",),
            ),
            analysis_summary=(
                "Evidence supports a monitored volatility scenario with explicit uncertainty."
            ),
            prediction_candidate_ids=("prediction-tsla-volatility-context",),
            evidence=(evidence_refs["fixture-news-tsla-001"],),
            data_quality={"mode": "offline_fixture", "evidence_count": 1},
        ),
        InstrumentReportSection(
            instrument_id="instrument:etf:us:spy",
            symbol="SPY",
            display_name="SPDR S&P 500 ETF Trust",
            observed_discussion_summary=(
                "Fixture evidence frames SPY as market-regime context for individual equities."
            ),
            social_news_summary="No direct fixture candidate is produced for SPY.",
            analysis_summary="SPY remains baseline context only in this fixture report.",
            evidence=(evidence_refs["fixture-market-spy-001"],),
            data_quality={"mode": "offline_fixture", "evidence_count": 1},
        ),
        InstrumentReportSection(
            instrument_id="instrument:crypto:btc-usd",
            symbol="BTC/USD",
            display_name="Bitcoin versus U.S. dollar",
            observed_discussion_summary=(
                "Fixture evidence shows data availability for a crypto instrument without "
                "forcing an equity-only report shape."
            ),
            social_news_summary="No direct fixture candidate is produced for BTC/USD.",
            analysis_summary="Crypto coverage is represented as availability context only.",
            evidence=(evidence_refs["fixture-crypto-btc-001"],),
            data_quality={"mode": "offline_fixture", "evidence_count": 1},
        ),
    )
    candidate = PredictionCandidate(
        candidate_id="prediction-tsla-volatility-context",
        instrument_id="instrument:equity:us:tsla",
        symbol="TSLA",
        horizon=TimeHorizon.SWING,
        direction=Direction.MIXED,
        status=PredictionStatus.EVIDENCE_SUPPORTED,
        thesis=(
            "TSLA may remain unusually sensitive to delivery, margin, and autonomy headlines "
            "over the short swing horizon."
        ),
        baseline=(
            "The baseline is no directional edge: evidence is mixed and the report should be "
            "read as scenario context."
        ),
        confidence=0.41,
        evidence_for=(evidence_refs["fixture-news-tsla-001"],),
        evidence_against=(evidence_refs["fixture-market-spy-001"],),
        assumptions=("Offline fixtures are a deterministic contract exercise.",),
        uncertainties=(
            "Synthetic fixture evidence cannot substitute for live provider freshness.",
            "Market-wide regime changes could dominate issuer-specific evidence.",
        ),
    )
    report = DailyReport(
        schema_version="daily-report.v2",
        run_id=f"research-{config.run_date.isoformat()}",
        report_date=config.run_date,
        generated_at=generated_at,
        timezone="UTC",
        objective="Generate an evidence-backed prediction research report.",
        universe="Mixed offline fixture universe: one equity, one ETF, one crypto pair.",
        command_args=_command_args(config),
        instruments=instruments,
        data_freshness=DataFreshnessSummary(
            as_of=generated_at,
            summary="Offline fixture evidence is deterministic and current to the run date.",
        ),
        provider_health=(
            ProviderHealth(
                provider_name="offline-fixture",
                status="ok",
                checked_at=generated_at,
                credential_state=CredentialState.NOT_REQUIRED,
            ),
        ),
        evidence_sources=evidence,
        instrument_sections=sections,
        prediction_candidates=(candidate,),
        audit_manifest=AuditManifest(
            run_id=f"research-{config.run_date.isoformat()}",
            schema_version="audit-manifest.v2",
            created_at=generated_at,
            command_args=_command_args(config),
            prediction_trace_ids=(candidate.candidate_id,),
        ),
    )
    return OfflineFixtureBundle(
        report=report,
        audit_payloads={
            "normalized-evidence.json": _evidence_payload(report),
            "analysis-contexts.json": _analysis_payload(report),
            "prediction-inputs.json": _prediction_payload(report),
        },
    )


def _source_evidence(generated_at: datetime) -> tuple[SourceEvidence, ...]:
    return (
        _evidence(
            evidence_id="fixture-news-tsla-001",
            source_kind=SourceKind.NEWS_ARTICLE,
            ticker="TSLA",
            title="Fixture TSLA delivery and margin context",
            text=(
                "Fixture source describes TSLA delivery uncertainty, margin pressure, and "
                "headline sensitivity without asserting a direction."
            ),
            raw_identifier="fixture-news-tsla-001",
            generated_at=generated_at,
            source_url="https://example.com/fixtures/tsla-delivery-context",
        ),
        _evidence(
            evidence_id="fixture-market-spy-001",
            source_kind=SourceKind.MARKET_DATA,
            ticker="SPY",
            title="Fixture SPY regime context",
            text=(
                "Fixture source describes broad-index regime risk as a possible driver of "
                "single-name volatility."
            ),
            raw_identifier="fixture-market-spy-001",
            generated_at=generated_at,
            source_url="https://example.com/fixtures/spy-regime-context",
        ),
        _evidence(
            evidence_id="fixture-crypto-btc-001",
            source_kind=SourceKind.MARKET_DATA,
            ticker=None,
            title="Fixture BTC/USD data availability",
            text=(
                "Fixture source records BTC/USD as a researchable crypto pair when provider "
                "data is available."
            ),
            raw_identifier="fixture-crypto-btc-001",
            generated_at=generated_at,
            source_url="https://example.com/fixtures/btc-usd-availability",
            matched_tickers=(),
        ),
    )


def _evidence(
    *,
    evidence_id: str,
    source_kind: SourceKind,
    ticker: str | None,
    title: str,
    text: str,
    raw_identifier: str,
    generated_at: datetime,
    source_url: str,
    matched_tickers: tuple[str, ...] | None = None,
) -> SourceEvidence:
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=source_kind,
        ticker=ticker,
        title=title,
        text=text,
        created_at=generated_at,
        permalink=source_url,
        matched_tickers=matched_tickers
        if matched_tickers is not None
        else ((ticker,) if ticker else ()),
        provenance=SourceProvenance(
            provider_name="offline-fixture",
            source_kind=source_kind,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=generated_at,
            observed_at=generated_at,
            source_url=source_url,
            permalink=source_url,
            raw_identifier=raw_identifier,
            raw_snapshot_id=f"raw-{raw_identifier}",
            freshness_status=FreshnessStatus.FRESH,
            provider_metadata={"fixture": True},
        ),
        metadata={"fixture": True},
    )


def _instruments(generated_at: datetime) -> tuple[Instrument, ...]:
    return (
        _instrument(
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            display_name="Tesla Inc.",
            asset_class=AssetClass.STOCK,
            provider_identifier="TSLA",
            generated_at=generated_at,
        ),
        _instrument(
            instrument_id="instrument:etf:us:spy",
            symbol="SPY",
            display_name="SPDR S&P 500 ETF Trust",
            asset_class=AssetClass.ETF,
            provider_identifier="SPY",
            generated_at=generated_at,
        ),
        _instrument(
            instrument_id="instrument:crypto:btc-usd",
            symbol="BTC/USD",
            display_name="Bitcoin versus U.S. dollar",
            asset_class=AssetClass.CRYPTO,
            provider_identifier="BTC/USD",
            generated_at=generated_at,
        ),
    )


def _instrument(
    *,
    instrument_id: str,
    symbol: str,
    display_name: str,
    asset_class: AssetClass,
    provider_identifier: str,
    generated_at: datetime,
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        display_name=display_name,
        asset_class=asset_class,
        provider_ids=(
            ProviderInstrumentId(
                provider="offline-fixture",
                identifier=provider_identifier,
                namespace="fixture-symbol",
            ),
        ),
        tradability_evidence=(
            TradabilityEvidence(
                provider="offline-fixture",
                status=TradabilityStatus.UNKNOWN,
                retrieved_at=generated_at,
                raw_identifier=f"{provider_identifier}:fixture-tradability",
                notes="Fixture records availability context only.",
            ),
        ),
        data_availability=(
            InstrumentDataAvailability(
                provider="offline-fixture",
                data_type="research-fixture",
                status=TradabilityStatus.AVAILABLE,
                checked_at=generated_at,
                provider_identifier=provider_identifier,
            ),
        ),
        metadata={"fixture": True},
    )


def _command_args(config: RunConfig) -> JsonObject:
    return {
        "run_date": config.run_date.isoformat(),
        "output_dir": str(config.output_dir),
        "fixture_dir": str(config.fixture_dir) if config.fixture_dir else None,
        "cache_dir": str(config.cache_dir) if config.cache_dir else None,
        "offline": config.offline,
        "source_mode": config.source_mode,
        "live_providers": config.live_providers,
    }


def _evidence_payload(report: DailyReport) -> JsonObject:
    return {
        "schema_version": "audit.normalized-evidence.v2",
        "run_id": report.run_id,
        "records": [record.model_dump(mode="json") for record in report.evidence_sources],
    }


def _analysis_payload(report: DailyReport) -> JsonObject:
    return {
        "schema_version": "audit.analysis-contexts.v2",
        "run_id": report.run_id,
        "records": [section.model_dump(mode="json") for section in report.instrument_sections],
    }


def _prediction_payload(report: DailyReport) -> JsonObject:
    return {
        "schema_version": "audit.prediction-inputs.v2",
        "run_id": report.run_id,
        "records": [
            candidate.model_dump(mode="json") for candidate in report.prediction_candidates
        ],
    }


__all__ = ["OfflineFixtureBundle", "build_offline_fixture_bundle"]
