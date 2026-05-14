from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    AssetClass,
    DailyReport,
    DataFreshnessSummary,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    Instrument,
    InstrumentReportSection,
    InstrumentResolution,
    InstrumentResolutionStatus,
    PredictionCandidate,
    PredictionStatus,
    RelatedInstrument,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TimeHorizon,
)


def _now() -> datetime:
    return datetime(2026, 5, 11, 18, 0, tzinfo=UTC)


def _evidence() -> SourceEvidence:
    return SourceEvidence(
        evidence_id="evidence-tsla-1",
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="TSLA",
        text="Tesla fixture evidence includes both upside narrative and margin uncertainty.",
        created_at=_now(),
        permalink="https://example.com/tsla",
        matched_tickers=("TSLA",),
        provenance=SourceProvenance(
            provider_name="fixture-news",
            source_kind=SourceKind.NEWS_ARTICLE,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=_now(),
            observed_at=_now(),
            source_url="https://example.com/tsla",
            permalink="https://example.com/tsla",
            raw_identifier="fixture-tsla-1",
            raw_snapshot_id="raw-fixture-tsla-1",
            freshness_status=FreshnessStatus.FRESH,
        ),
    )


def _instrument() -> Instrument:
    return Instrument(
        instrument_id="instrument:equity:us:tsla",
        symbol="TSLA",
        display_name="Tesla Inc.",
        asset_class=AssetClass.STOCK,
    )


def _candidate() -> PredictionCandidate:
    return PredictionCandidate(
        candidate_id="prediction-tsla-1",
        instrument_id="instrument:equity:us:tsla",
        symbol="TSLA",
        horizon=TimeHorizon.SWING,
        direction=Direction.MIXED,
        status=PredictionStatus.EVIDENCE_SUPPORTED,
        thesis="TSLA may remain headline-sensitive over the swing horizon.",
        baseline="No directional edge is assumed.",
        confidence=0.4,
        evidence_for=(EvidenceReference(evidence_id="evidence-tsla-1"),),
        uncertainties=("Fixture evidence is not live evidence.",),
    )


def _report(*, include_sources: bool = True) -> DailyReport:
    evidence = _evidence()
    candidate = _candidate()
    return DailyReport(
        schema_version="daily-report.v2",
        run_id="research-2026-05-11",
        report_date=date(2026, 5, 11),
        generated_at=_now(),
        timezone="UTC",
        objective="Generate an evidence-backed prediction research report.",
        universe="Unit-test fixture universe.",
        instruments=(_instrument(),),
        data_freshness=DataFreshnessSummary(as_of=_now(), summary="fresh fixture"),
        evidence_sources=(evidence,) if include_sources else (),
        instrument_resolutions=(
            InstrumentResolution(
                query="TSLA",
                status=InstrumentResolutionStatus.RESOLVED,
                matches=(_instrument(),),
                selected_instrument_id="instrument:equity:us:tsla",
            ),
        ),
        instrument_sections=(
            InstrumentReportSection(
                instrument_id="instrument:equity:us:tsla",
                symbol="TSLA",
                prediction_candidate_ids=(candidate.candidate_id,),
                evidence=(EvidenceReference(evidence_id="evidence-tsla-1"),),
            ),
        ),
        prediction_candidates=(candidate,),
    )


@pytest.mark.schema
def test_prediction_candidate_requires_evidence_for_supported_status() -> None:
    with pytest.raises(ValidationError, match="evidence_for"):
        PredictionCandidate(
            candidate_id="prediction-tsla-1",
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            status=PredictionStatus.EVIDENCE_SUPPORTED,
            thesis="TSLA may remain headline-sensitive.",
            baseline="No directional edge is assumed.",
            confidence=0.4,
        )


@pytest.mark.schema
def test_daily_report_is_instrument_based_not_fixed_to_six_tickers() -> None:
    report = _report()

    assert len(report.instrument_sections) == 1
    assert report.prediction_candidates[0].status == PredictionStatus.EVIDENCE_SUPPORTED


@pytest.mark.schema
def test_daily_report_requires_every_cited_evidence_source_even_when_sources_empty() -> None:
    with pytest.raises(ValidationError, match="evidence_sources"):
        _report(include_sources=False)


@pytest.mark.schema
def test_daily_report_without_candidates_requires_insufficient_evidence_summary() -> None:
    candidate_free = _report().model_copy(
        update={
            "prediction_candidates": (),
            "instrument_sections": (
                InstrumentReportSection(
                    instrument_id="instrument:equity:us:tsla",
                    symbol="TSLA",
                    evidence=(EvidenceReference(evidence_id="evidence-tsla-1"),),
                ),
            ),
        }
    )

    assert candidate_free.insufficient_evidence_summary is None
    with pytest.raises(ValidationError, match="insufficient_evidence_summary"):
        DailyReport.model_validate(candidate_free.model_dump(mode="python"))


@pytest.mark.schema
def test_daily_report_validates_related_instrument_evidence_ids() -> None:
    report = _report().model_copy(
        update={
            "instruments": (
                _instrument().model_copy(
                    update={
                        "related_instruments": (
                            RelatedInstrument(
                                instrument_id="instrument:sector:consumer-discretionary",
                                relationship="sector_proxy",
                                rationale="Sector proxy used for context.",
                                evidence_ids=("missing-sector-evidence",),
                            ),
                        )
                    }
                ),
            )
        }
    )

    with pytest.raises(ValidationError, match="related instrument evidence_ids"):
        DailyReport.model_validate(report.model_dump(mode="python"))


@pytest.mark.schema
def test_daily_report_validates_resolution_selected_ids_against_report_instruments() -> None:
    report = _report().model_copy(
        update={
            "instrument_resolutions": (
                InstrumentResolution(
                    query="NVDA",
                    status=InstrumentResolutionStatus.RESOLVED,
                    matches=(
                        Instrument(
                            instrument_id="instrument:equity:us:nvda",
                            symbol="NVDA",
                            display_name="NVIDIA Corp.",
                            asset_class=AssetClass.STOCK,
                        ),
                    ),
                    selected_instrument_id="instrument:equity:us:nvda",
                ),
            )
        }
    )

    with pytest.raises(ValidationError, match="instrument_resolutions"):
        DailyReport.model_validate(report.model_dump(mode="python"))


@pytest.mark.schema
def test_daily_report_validates_evidence_reference_quotes_against_source_text() -> None:
    payload = _report().model_dump(mode="python")
    payload["prediction_candidates"][0]["evidence_for"][0]["quote"] = "not in source text"

    with pytest.raises(ValidationError, match="quote must appear"):
        DailyReport.model_validate(payload)


@pytest.mark.schema
def test_daily_report_validates_evidence_reference_spans_against_source_text() -> None:
    payload = _report().model_dump(mode="python")
    reference = payload["prediction_candidates"][0]["evidence_for"][0]
    reference["quote"] = "Tesla"
    reference["start_char"] = 1
    reference["end_char"] = 6

    with pytest.raises(ValidationError, match="span must match"):
        DailyReport.model_validate(payload)
