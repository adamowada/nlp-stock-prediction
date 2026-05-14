from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    AssetClass,
    AuditArtifact,
    AuditManifest,
    DailyReport,
    DataFreshnessSummary,
    Direction,
    DissentingEvidence,
    EvidenceReference,
    FreshnessStatus,
    Instrument,
    InstrumentReportSection,
    InstrumentResolution,
    InstrumentResolutionStatus,
    InsufficientEvidenceReport,
    MaterialClaimTrace,
    PredictionCandidate,
    PredictionChangeTrigger,
    PredictionStatus,
    RelatedInstrument,
    ReportSourceReference,
    RetrievalMethod,
    SignalArtifactFamily,
    SignalArtifactReference,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TechnicalAnalysis,
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
        change_triggers=(
            PredictionChangeTrigger(
                trigger_id="change-tsla-fresh-sources",
                summary="Fresh source evidence would change the confidence context.",
                trigger_type="provider_refresh",
                evidence=(EvidenceReference(evidence_id="evidence-tsla-1"),),
                rationale="The candidate is based on one fixture source.",
            ),
        ),
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
        source_references=(
            ReportSourceReference(
                reference_id="source-ref-tsla-1",
                label="TSLA unit-test evidence",
                reference_type="source_evidence",
                evidence_ids=("evidence-tsla-1",),
                candidate_ids=(candidate.candidate_id,),
            ),
        ),
        material_claim_traces=(
            MaterialClaimTrace(
                claim_id="claim-tsla-headline-sensitive",
                claim="TSLA may remain headline-sensitive over the swing horizon.",
                claim_type="analysis",
                evidence=(EvidenceReference(evidence_id="evidence-tsla-1"),),
                source_reference_ids=("source-ref-tsla-1",),
                candidate_ids=(candidate.candidate_id,),
            ),
        ),
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
            change_trigger_limitations=(
                "No evidence exists, so change triggers cannot be defined.",
            ),
        )


@pytest.mark.schema
def test_contradicted_candidate_requires_opposing_evidence() -> None:
    payload = _candidate().model_dump(mode="python")
    payload["status"] = PredictionStatus.CONTRADICTED
    payload["evidence_for"] = [EvidenceReference(evidence_id="evidence-tsla-1").model_dump()]
    payload["evidence_against"] = []
    payload["dissenting_evidence"] = []

    with pytest.raises(ValidationError, match="opposing evidence"):
        PredictionCandidate.model_validate(payload)


@pytest.mark.schema
def test_nested_analysis_fields_reject_trading_instructions() -> None:
    payload = _report().model_dump(mode="python")
    payload["instrument_sections"][0]["technical_analysis"] = TechnicalAnalysis(
        ticker="TSLA",
        summary="Technical context is included for evidence review.",
        trend="Buy TSLA now",
        evidence=(EvidenceReference(evidence_id="evidence-tsla-1"),),
    ).model_dump(mode="python")

    with pytest.raises(ValidationError, match="trading instructions"):
        DailyReport.model_validate(payload)


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
def test_daily_report_without_candidates_requires_structured_insufficient_evidence() -> None:
    candidate_free = _report().model_copy(
        update={
            "prediction_candidates": (),
            "material_claim_traces": (),
            "instrument_sections": (
                InstrumentReportSection(
                    instrument_id="instrument:equity:us:tsla",
                    symbol="TSLA",
                    evidence=(EvidenceReference(evidence_id="evidence-tsla-1"),),
                ),
            ),
        }
    )

    assert candidate_free.insufficient_evidence is None
    with pytest.raises(ValidationError, match="structured insufficient_evidence"):
        DailyReport.model_validate(candidate_free.model_dump(mode="python"))


@pytest.mark.schema
def test_daily_report_accepts_structured_insufficient_evidence_without_candidates() -> None:
    candidate_free = _report().model_copy(
        update={
            "prediction_candidates": (),
            "material_claim_traces": (),
            "instrument_sections": (
                InstrumentReportSection(
                    instrument_id="instrument:equity:us:tsla",
                    symbol="TSLA",
                    evidence=(EvidenceReference(evidence_id="evidence-tsla-1"),),
                ),
            ),
            "source_references": (
                ReportSourceReference(
                    reference_id="source-ref-tsla-1",
                    label="TSLA unit-test evidence",
                    reference_type="source_evidence",
                    evidence_ids=("evidence-tsla-1",),
                ),
            ),
            "insufficient_evidence": InsufficientEvidenceReport(
                summary="No supported candidate is available.",
                blocking_reasons=("Only one fixture evidence source is present.",),
                evidence=(EvidenceReference(evidence_id="evidence-tsla-1"),),
            ),
        }
    )

    assert DailyReport.model_validate(candidate_free.model_dump(mode="python"))


@pytest.mark.schema
def test_prediction_candidate_requires_change_trigger_context() -> None:
    payload = _candidate().model_dump(mode="python")
    payload["change_triggers"] = []

    with pytest.raises(ValidationError, match="change triggers"):
        PredictionCandidate.model_validate(payload)


@pytest.mark.schema
def test_prediction_candidate_rejects_trading_instruction_synonyms() -> None:
    payload = _candidate().model_dump(mode="python")
    payload["thesis"] = "Investors should accumulate TSLA."

    with pytest.raises(ValidationError, match="trading language"):
        PredictionCandidate.model_validate(payload)


@pytest.mark.schema
def test_daily_report_requires_material_claim_traces_for_candidates() -> None:
    payload = _report().model_dump(mode="python")
    payload["material_claim_traces"] = []

    with pytest.raises(ValidationError, match="material_claim_traces"):
        DailyReport.model_validate(payload)


@pytest.mark.schema
def test_daily_report_validates_material_claim_trace_source_reference_ids() -> None:
    payload = _report().model_dump(mode="python")
    payload["material_claim_traces"][0]["source_reference_ids"] = ["missing-source-ref"]

    with pytest.raises(ValidationError, match="source_reference_ids"):
        DailyReport.model_validate(payload)


@pytest.mark.schema
def test_daily_report_requires_inline_audit_manifest_for_cited_artifacts() -> None:
    signal_ref = SignalArtifactReference(
        artifact_id="artifact-technical-tsla",
        family=SignalArtifactFamily.TECHNICALS,
        artifact_type="technical_package",
    )
    payload = _report().model_dump(mode="python")
    payload["prediction_candidates"][0]["signal_artifact_ids"] = [signal_ref.artifact_id]
    payload["prediction_candidates"][0]["signal_artifacts"] = [signal_ref.model_dump(mode="python")]

    with pytest.raises(ValidationError, match="audit manifest"):
        DailyReport.model_validate(payload)

    payload["audit_manifest"] = AuditManifest(
        run_id="research-2026-05-11",
        schema_version="audit-manifest.v1",
        created_at=_now(),
        artifacts=(
            AuditArtifact(
                artifact_id=signal_ref.artifact_id,
                artifact_type="technical_package",
                path="reports/audit/technical.json",
                created_at=_now(),
                produced_by="phase4_technical_package",
            ),
        ),
    ).model_dump(mode="python")

    assert DailyReport.model_validate(payload)


@pytest.mark.schema
def test_daily_report_requires_audit_manifest_run_id_to_match() -> None:
    payload = _report().model_dump(mode="python")
    payload["audit_manifest"] = AuditManifest(
        run_id="different-run",
        schema_version="audit-manifest.v2",
        created_at=_now(),
    ).model_dump(mode="python")

    with pytest.raises(ValidationError, match="run_id"):
        DailyReport.model_validate(payload)


@pytest.mark.schema
def test_audit_manifest_rejects_duplicate_artifact_ids() -> None:
    artifact = AuditArtifact(
        artifact_id="artifact-duplicate",
        artifact_type="json_report",
        path="reports/report.json",
        created_at=_now(),
        produced_by="unit-test",
    )

    with pytest.raises(ValidationError, match="artifact ids"):
        AuditManifest(
            run_id="research-2026-05-11",
            schema_version="audit-manifest.v2",
            created_at=_now(),
            artifacts=(artifact, artifact),
        )


@pytest.mark.schema
def test_daily_report_requires_each_candidate_to_have_material_claim_trace() -> None:
    payload = _report().model_dump(mode="python")
    payload["material_claim_traces"][0]["candidate_ids"] = []

    with pytest.raises(ValidationError, match="every prediction candidate"):
        DailyReport.model_validate(payload)


@pytest.mark.schema
def test_report_source_reference_requires_source_target() -> None:
    candidate = _candidate()

    with pytest.raises(ValidationError, match="source target"):
        ReportSourceReference(
            reference_id="source-ref-candidate-only",
            label="Candidate-only reference is not source provenance.",
            reference_type="source_evidence",
            candidate_ids=(candidate.candidate_id,),
        )


@pytest.mark.schema
def test_dissenting_evidence_requires_source_or_artifact_reference() -> None:
    with pytest.raises(ValidationError, match="dissenting evidence"):
        DissentingEvidence(summary="Dissent without a trace is not reportable.")


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
