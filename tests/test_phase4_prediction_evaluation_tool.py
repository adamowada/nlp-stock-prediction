from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    AssetClass,
    AuditArtifact,
    AuditManifest,
    DailyReport,
    DataFreshnessSummary,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    Instrument,
    InstrumentReportSection,
    MaterialClaimTrace,
    PredictionCandidate,
    PredictionChangeTrigger,
    PredictionStatus,
    ReportSourceReference,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TimeHorizon,
)
from nlp_stock_prediction.evaluation import (
    attach_evaluation_metadata,
    write_prediction_evaluation_artifact,
)
from nlp_stock_prediction.orchestration.report_data_modes import (
    OFFLINE_FIXTURE_REPORT_DATA_MODE,
    report_data_mode_metadata,
)
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.storage import (
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    SQLiteStore,
)

NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
RUN_ID = "run-phase4-evaluation"
INSTRUMENT_ID = "instrument:equity:us:tsla"


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "data" / "prediction-research.sqlite3")
    store.initialize()
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id=INSTRUMENT_ID,
            symbol="TSLA",
            asset_class="stock",
            name="Tesla Inc.",
        )
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=RUN_ID,
            run_kind="phase4_prediction_evaluation",
            objective="Evaluate prediction scenario quality.",
            status="running",
            started_at=NOW,
            metadata=report_data_mode_metadata(OFFLINE_FIXTURE_REPORT_DATA_MODE),
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-support",
            source_type=SourceKind.NEWS_ARTICLE.value,
            provider="fixture-news",
            retrieved_at=NOW,
            published_at=NOW,
            instruments=(INSTRUMENT_ID,),
            claim="Fixture catalyst supports scenario quality.",
            freshness_status=FreshnessStatus.FRESH.value,
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-tsla-quality",
            run_id=RUN_ID,
            instrument_id=INSTRUMENT_ID,
            prediction_horizon=TimeHorizon.SWING.value,
            prediction_type="directional",
            scenario="TSLA fixture scenario quality depends on attributed evidence.",
            direction=Direction.MIXED.value,
            confidence=0.62,
            status=PredictionStatus.EVIDENCE_SUPPORTED.value,
            evidence_for=("evidence-support",),
            baseline={"comparison": "no_directional_edge"},
            uncertainty="Fixture sources are deterministic test inputs.",
        )
    )
    return store


def _source() -> SourceEvidence:
    return SourceEvidence(
        evidence_id="evidence-support",
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="TSLA",
        text="Fixture catalyst supports scenario quality.",
        created_at=NOW,
        permalink="https://example.test/evidence-support",
        matched_tickers=("TSLA",),
        matched_instrument_ids=(INSTRUMENT_ID,),
        instrument_id=INSTRUMENT_ID,
        provenance=SourceProvenance(
            provider_name="fixture-news",
            source_kind=SourceKind.NEWS_ARTICLE,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=NOW,
            observed_at=NOW,
            source_url="https://example.test/evidence-support",
            permalink="https://example.test/evidence-support",
            raw_identifier="evidence-support",
            raw_snapshot_id="raw-evidence-support",
            freshness_status=FreshnessStatus.FRESH,
        ),
    )


def _candidate() -> PredictionCandidate:
    return PredictionCandidate(
        candidate_id="candidate-tsla-quality",
        instrument_id=INSTRUMENT_ID,
        symbol="TSLA",
        horizon=TimeHorizon.SWING,
        direction=Direction.MIXED,
        status=PredictionStatus.EVIDENCE_SUPPORTED,
        thesis="TSLA fixture scenario quality depends on attributed evidence.",
        baseline="No directional edge is assumed without source-backed evidence.",
        confidence=0.62,
        evidence_for=(EvidenceReference(evidence_id="evidence-support"),),
        uncertainties=("Fixture sources are deterministic test inputs.",),
        change_triggers=(
            PredictionChangeTrigger(
                trigger_id="change-tsla-quality-fresh-evidence",
                summary="Fresh contradictory or confirming evidence would change support.",
                trigger_type="new_source_evidence",
                evidence=(EvidenceReference(evidence_id="evidence-support"),),
                rationale="The test candidate has one attributed source.",
            ),
        ),
        metadata={
            "baseline": {
                "baseline_id": "no_directional_edge",
                "summary": "No directional edge is assumed without source-backed evidence.",
                "provenance": {
                    "provider": "fixture-baseline",
                    "retrieved_at": NOW.isoformat(),
                },
            }
        },
    )


def _report(
    *,
    candidate: PredictionCandidate,
    source: SourceEvidence,
    artifact: AuditArtifact,
) -> DailyReport:
    instrument = Instrument(
        instrument_id=INSTRUMENT_ID,
        symbol="TSLA",
        display_name="Tesla Inc.",
        asset_class=AssetClass.STOCK,
    )
    return DailyReport(
        schema_version="daily-report.v2",
        run_id=RUN_ID,
        report_date=date(2026, 5, 13),
        generated_at=NOW,
        timezone="UTC",
        objective="Evaluate prediction scenario quality.",
        universe="Phase 4 fixture universe.",
        instruments=(instrument,),
        data_freshness=DataFreshnessSummary(as_of=NOW, summary="fresh fixture"),
        evidence_sources=(source,),
        instrument_sections=(
            InstrumentReportSection(
                instrument_id=INSTRUMENT_ID,
                symbol="TSLA",
                prediction_candidate_ids=(candidate.candidate_id,),
                evidence=(EvidenceReference(evidence_id=source.evidence_id),),
            ),
        ),
        prediction_candidates=(candidate,),
        source_references=(
            ReportSourceReference(
                reference_id="source-ref-evidence-support",
                label="Evaluation test source evidence",
                reference_type="source_evidence",
                evidence_ids=(source.evidence_id,),
                candidate_ids=(candidate.candidate_id,),
            ),
            ReportSourceReference(
                reference_id=f"source-ref-{artifact.artifact_id}",
                label="Prediction evaluation artifact",
                reference_type="prediction_evaluation",
                artifact_ids=(artifact.artifact_id,),
                candidate_ids=(candidate.candidate_id,),
            ),
        ),
        material_claim_traces=(
            MaterialClaimTrace(
                claim_id="claim-candidate-tsla-quality",
                claim=candidate.thesis,
                claim_type="prediction_evaluation",
                evidence=candidate.evidence_for,
                artifact_ids=(artifact.artifact_id,),
                source_reference_ids=(
                    "source-ref-evidence-support",
                    f"source-ref-{artifact.artifact_id}",
                ),
                candidate_ids=(candidate.candidate_id,),
            ),
        ),
        audit_manifest=AuditManifest(
            run_id=RUN_ID,
            schema_version="audit-manifest.v2",
            created_at=NOW,
            artifacts=(artifact,),
            prediction_trace_ids=(candidate.candidate_id,),
        ),
    )


@pytest.mark.unit
def test_prediction_evaluation_tool_writes_indexed_artifact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    candidate = _candidate()

    evaluation, artifact = write_prediction_evaluation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        candidate=candidate,
        evidence_sources=(_source(),),
        created_at=NOW,
    )

    artifact_path = Path(artifact.path)
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    indexed_artifact = store.get_artifact(artifact.artifact_id)
    candidate_artifact_links = store.list_candidate_artifact_links(candidate.candidate_id)
    candidate_evidence_links = store.list_candidate_evidence_links(candidate.candidate_id)

    assert evaluation.status == PredictionStatus.EVIDENCE_SUPPORTED
    assert artifact.artifact_type == "prediction_evaluation"
    assert indexed_artifact is not None
    assert artifact.sha256 == indexed_artifact.sha256
    assert indexed_artifact.path == Path("reports") / RUN_ID / "audit" / artifact_path.name
    assert tuple(item.artifact_id for item in store.list_artifacts_for_run(RUN_ID)) == (
        artifact.artifact_id,
    )
    assert candidate_artifact_links[0].relationship == "prediction_evaluation"
    assert candidate_evidence_links[0].relationship == "evaluation_supports"
    assert artifact_payload["evaluation"]["baseline_comparison"]["baseline_id"] == (
        "no_directional_edge"
    )
    assert artifact_payload["evaluation"]["uncertainty"]
    assert artifact_payload["evaluation"]["evidence_counts"]["supporting_source_evidence"] == 1
    assert artifact_payload["evaluation"]["evidence_for"][0]["evidence_id"] == "evidence-support"
    assert artifact_payload["evaluation"]["quality_language"]["purpose"] == "prediction_quality"
    quality_language = artifact.metadata["quality_language"]
    assert isinstance(quality_language, dict)
    assert quality_language["trading_action_language"] == "excluded"


@pytest.mark.unit
def test_prediction_evaluation_tool_rejects_in_memory_candidate_mismatch(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = _candidate().model_copy(
        update={
            "instrument_id": "instrument:equity:us:nvda",
            "symbol": "NVDA",
        }
    )

    with pytest.raises(ValueError, match="stored prediction candidate mismatch"):
        write_prediction_evaluation_artifact(
            store=store,
            repo_root=tmp_path,
            artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
            run_id=RUN_ID,
            candidate=candidate,
            evidence_sources=(_source(),),
            created_at=NOW,
        )


@pytest.mark.unit
def test_reports_preserve_prediction_quality_language_without_advice_terms(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = _candidate()
    source = _source()
    evaluation, artifact = write_prediction_evaluation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        candidate=candidate,
        evidence_sources=(source,),
        created_at=NOW,
    )
    evaluated_candidate = attach_evaluation_metadata(
        candidate,
        evaluation,
        artifact=artifact,
    )
    report = _report(candidate=evaluated_candidate, source=source, artifact=artifact)

    markdown = render_markdown_report(report)
    json_payload = json.loads(render_json_report(report))

    assert "Evaluation quality: Prediction quality evaluation" in markdown
    assert artifact.artifact_id in markdown
    assert (
        json_payload["prediction_candidates"][0]["metadata"]["prediction_evaluation"][
            "quality_label"
        ]
        == "Prediction quality evaluation"
    )
    rendered = f"{markdown}\n{render_json_report(report)}".lower()
    assert "recommendation" not in rendered
    assert "buy" not in rendered
    assert "sell" not in rendered


@pytest.mark.unit
def test_report_rejects_mismatched_evaluation_status_metadata(tmp_path: Path) -> None:
    store = _store(tmp_path)
    candidate = _candidate()
    source = _source()
    evaluation, artifact = write_prediction_evaluation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        candidate=candidate,
        evidence_sources=(source,),
        created_at=NOW,
    )
    evaluated_candidate = attach_evaluation_metadata(candidate, evaluation, artifact=artifact)
    mismatched_candidate = evaluated_candidate.model_copy(
        update={"status": PredictionStatus.INSUFFICIENT_EVIDENCE}
    )

    with pytest.raises(ValidationError, match="prediction evaluation status"):
        _report(candidate=mismatched_candidate, source=source, artifact=artifact)


@pytest.mark.unit
def test_attach_evaluation_metadata_updates_rendered_candidate_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    candidate = _candidate()
    source = _source().model_copy(
        update={
            "provenance": _source().provenance.model_copy(
                update={"freshness_status": FreshnessStatus.STALE}
            )
        }
    )
    evaluation, artifact = write_prediction_evaluation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        candidate=candidate,
        evidence_sources=(source,),
        created_at=NOW,
    )
    evaluated_candidate = attach_evaluation_metadata(candidate, evaluation, artifact=artifact)
    report = _report(candidate=evaluated_candidate, source=source, artifact=artifact)

    markdown = render_markdown_report(report)
    json_payload = json.loads(render_json_report(report))

    assert evaluated_candidate.status == PredictionStatus.INSUFFICIENT_EVIDENCE
    assert "Status: insufficient_evidence" in markdown
    assert json_payload["prediction_candidates"][0]["status"] == "insufficient_evidence"


@pytest.mark.unit
def test_report_authored_fields_reject_trading_instructions_but_source_text_is_evidence(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = _candidate()
    source = _source().model_copy(update={"text": "Buy TSLA now, the article claims."})
    evaluation, artifact = write_prediction_evaluation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        candidate=candidate,
        evidence_sources=(source,),
        created_at=NOW,
    )
    evaluated_candidate = attach_evaluation_metadata(candidate, evaluation, artifact=artifact)
    safe_report = _report(candidate=evaluated_candidate, source=source, artifact=artifact)

    assert safe_report.evidence_sources[0].text == "Buy TSLA now, the article claims."

    with pytest.raises(ValidationError, match="imperative trading language"):
        evaluated_candidate.model_copy(
            update={"thesis": "Buy TSLA now because the fixture catalyst is strong."}
        )
