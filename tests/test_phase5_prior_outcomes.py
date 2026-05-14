from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.contracts import (
    FreshnessStatus,
    JsonObject,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
)
from nlp_stock_prediction.orchestration import OFFLINE_FIXTURE_REPORT_DATA_MODE, Phase4Service
from nlp_stock_prediction.orchestration.prior_outcomes import PRIOR_REPORT_STALE_AFTER_DAYS
from nlp_stock_prediction.orchestration.report_data_modes import report_data_mode_metadata
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ReportArtifactRecord,
    SourceQueryRecord,
    ToolRunRecord,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


@pytest.mark.integration
def test_first_indexed_report_records_unavailable_prior_outcome(tmp_path: Path) -> None:
    service = _service(tmp_path)

    result = service.run_offline_phase4_flow(
        run_date="2026-05-13",
        output_dir="reports/phase5-prior-first-run",
        symbol="TSLA",
    )

    payload = _payload(result)
    candidate = cast(dict[str, Any], payload["prediction_candidates"][0])
    reviews = cast(list[dict[str, Any]], payload["prior_outcome_reviews"])

    assert reviews[0]["status"] == "not_available"
    assert "first indexed report" in reviews[0]["limitations"][0]
    assert candidate["prior_outcome_review_ids"] == [reviews[0]["review_id"]]
    assert any(
        trigger["trigger_type"] == "new_source_evidence"
        for trigger in cast(list[dict[str, Any]], candidate["change_triggers"])
    )


@pytest.mark.integration
def test_supported_follow_up_evidence_reviews_prior_report_artifact(tmp_path: Path) -> None:
    service = _service(tmp_path)
    prior = service.run_offline_phase4_flow(
        run_date="2026-05-12",
        output_dir="reports/phase5-prior-supported",
        symbol="TSLA",
    )

    current = service.run_offline_phase4_flow(
        run_date="2026-05-13",
        output_dir="reports/phase5-prior-supported",
        symbol="TSLA",
    )

    prior_json_artifact_id = _json_report_artifact_id(prior)
    payload = _payload(current)
    candidate = cast(dict[str, Any], payload["prediction_candidates"][0])
    reviews = cast(list[dict[str, Any]], payload["prior_outcome_reviews"])
    audit_artifacts = {
        artifact["artifact_id"]: artifact
        for artifact in cast(list[dict[str, Any]], payload["audit_manifest"]["artifacts"])
    }

    assert reviews[0]["status"] == "available"
    assert "supports" in reviews[0]["summary"]
    assert reviews[0]["artifact_ids"] == [prior_json_artifact_id]
    assert reviews[0]["metadata"]["prior_report_artifact_id"] == prior_json_artifact_id
    outcome = cast(dict[str, Any], reviews[0]["metadata"]["prediction_outcome"])
    outcome_evaluation = cast(
        dict[str, Any],
        reviews[0]["metadata"]["prediction_outcome_evaluation"],
    )
    assert outcome["status"] == "observed"
    assert outcome["observed_result"] == "supported"
    assert outcome["candidate_id"] == candidate["candidate_id"]
    assert outcome_evaluation["status"] == "confirmed"
    assert outcome_evaluation["outcome_id"] == outcome["outcome_id"]
    assert reviews[0]["outcome_evidence"]
    assert audit_artifacts[prior_json_artifact_id]["metadata"]["prior_outcome_source"] is True
    assert any(
        trigger["trigger_type"] == "outcome_data"
        for trigger in cast(list[dict[str, Any]], candidate["change_triggers"])
    )


@pytest.mark.integration
def test_stale_prior_report_becomes_explicit_limitation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    prior = service.run_offline_phase4_flow(
        run_date="2026-05-12",
        output_dir="reports/phase5-prior-stale",
        symbol="TSLA",
    )
    _mark_prior_json_report_stale(service, prior, stale_report_date=date(2026, 3, 1))

    current = service.run_offline_phase4_flow(
        run_date="2026-05-13",
        output_dir="reports/phase5-prior-stale",
        symbol="TSLA",
    )

    payload = _payload(current)
    review = cast(dict[str, Any], payload["prior_outcome_reviews"][0])

    assert review["status"] == "stale"
    assert f"older than {PRIOR_REPORT_STALE_AFTER_DAYS} days" in review["limitations"][0]


@pytest.mark.integration
def test_contradictory_follow_up_evidence_reviews_prior_report(tmp_path: Path) -> None:
    service = _service(tmp_path)
    prior = service.run_offline_phase4_flow(
        run_date="2026-05-12",
        output_dir="reports/phase5-prior-contradicted",
        symbol="TSLA",
    )
    started = service.start_research_run(
        run_date="2026-05-13",
        output_dir="reports/phase5-prior-contradicted",
        symbol="TSLA",
        objective="Contradictory prior outcome review test.",
    )
    run_id = str(started["run_id"])
    _insert_contradictory_candidate(service, run_id=run_id)

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    payload = _read_report_payload(rendered)
    prior_json_artifact_id = _json_report_artifact_id(prior)
    candidate = cast(dict[str, Any], payload["prediction_candidates"][0])
    review = cast(dict[str, Any], payload["prior_outcome_reviews"][0])

    assert candidate["status"] == "contradicted"
    assert review["status"] == "available"
    assert "contradicts" in review["summary"]
    assert review["artifact_ids"] == [prior_json_artifact_id]
    assert review["outcome_evidence"][0]["evidence_id"] == "evidence-prior-contradiction"
    assert any(
        trigger["trigger_type"] == "new_source_evidence"
        and "contradictory" in trigger["trigger_id"]
        for trigger in cast(list[dict[str, Any]], candidate["change_triggers"])
    )


def _service(tmp_path: Path) -> Phase4Service:
    return Phase4Service(
        repo_root=tmp_path,
        fixture_root=REPO_ROOT,
        database_path=Path("data") / "prediction-research.sqlite3",
    )


def _payload(result: Mapping[str, object]) -> dict[str, Any]:
    report = cast(dict[str, object], result["report"])
    return _read_report_payload(report)


def _read_report_payload(report: Mapping[str, object]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path(str(report["json_path"])).read_text(encoding="utf-8")),
    )


def _json_report_artifact_id(result: Mapping[str, object]) -> str:
    report = cast(dict[str, object], result["report"])
    audit_payload = cast(
        dict[str, Any],
        json.loads(Path(str(report["audit_manifest_path"])).read_text(encoding="utf-8")),
    )
    artifacts = cast(list[dict[str, Any]], audit_payload["artifacts"])
    return next(
        cast(str, artifact["artifact_id"])
        for artifact in artifacts
        if artifact["artifact_type"] == "json_report"
    )


def _mark_prior_json_report_stale(
    service: Phase4Service,
    result: Mapping[str, object],
    *,
    stale_report_date: date,
) -> None:
    report = cast(dict[str, object], result["report"])
    json_path = Path(str(report["json_path"]))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["report_date"] = stale_report_date.isoformat()
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    artifact_id = _json_report_artifact_id(result)
    artifact = service.store.get_artifact(artifact_id)
    report_artifact = service.store.get_report_artifact(artifact_id)
    assert artifact is not None
    assert report_artifact is not None
    service.store.record_artifact(
        ArtifactRecord(
            artifact_id=artifact.artifact_id,
            tool_run_id=artifact.tool_run_id,
            artifact_type=artifact.artifact_type,
            path=artifact.path,
            sha256=digest,
            schema_version=artifact.schema_version,
            produced_by=artifact.produced_by,
            record_count=artifact.record_count,
            metadata=artifact.metadata,
            created_at=artifact.created_at,
        )
    )
    service.store.record_report_artifact(
        ReportArtifactRecord(
            artifact_id=report_artifact.artifact_id,
            run_id=report_artifact.run_id,
            tool_run_id=report_artifact.tool_run_id,
            artifact_type=report_artifact.artifact_type,
            path=report_artifact.path,
            sha256=digest,
            schema_version=report_artifact.schema_version,
            report_schema_version=report_artifact.report_schema_version,
            report_date=stale_report_date,
            report_data_mode=report_artifact.report_data_mode,
            source_run_started_at=report_artifact.source_run_started_at,
            source_run_completed_at=report_artifact.source_run_completed_at,
            instrument_id=report_artifact.instrument_id,
            symbol=report_artifact.symbol,
            metadata=report_artifact.metadata,
            created_at=report_artifact.created_at,
        )
    )


def _insert_contradictory_candidate(service: Phase4Service, *, run_id: str) -> None:
    service.store.upsert_instrument(
        InstrumentRecord(
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            asset_class="stock",
            name="Tesla Inc.",
        )
    )
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-prior-contradiction",
            run_id=run_id,
            tool_name="phase5_prior_outcome_test_evidence",
            tool_version="phase5.prior-outcome.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={
                "symbol": "TSLA",
                **report_data_mode_metadata(OFFLINE_FIXTURE_REPORT_DATA_MODE),
            },
        )
    )
    service.store.record_source_query(
        SourceQueryRecord(
            source_query_id="query-prior-contradiction",
            tool_run_id="tool-prior-contradiction",
            provider="fixture-news",
            query="TSLA contradictory follow-up",
            url="https://example.test/tsla-prior-contradiction",
            retrieved_at=NOW,
            metadata=report_data_mode_metadata(OFFLINE_FIXTURE_REPORT_DATA_MODE),
        )
    )
    provenance = SourceProvenance(
        provider_name="fixture-news",
        source_kind=SourceKind.NEWS_ARTICLE,
        retrieval_method=RetrievalMethod.FIXTURE,
        fetched_at=NOW,
        observed_at=NOW,
        source_url="https://example.test/tsla-prior-contradiction",
        permalink="https://example.test/tsla-prior-contradiction",
        raw_identifier="fixture-prior-contradiction",
        raw_snapshot_id="raw-prior-contradiction",
        query="TSLA contradictory follow-up",
        freshness_status=FreshnessStatus.FRESH,
    )
    source_evidence = SourceEvidence(
        evidence_id="evidence-prior-contradiction",
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="TSLA",
        title="TSLA contradictory follow-up",
        text="Follow-up fixture evidence contradicts the prior TSLA scenario.",
        created_at=NOW,
        permalink="https://example.test/tsla-prior-contradiction",
        instrument_id="instrument:equity:us:tsla",
        matched_tickers=("TSLA",),
        matched_instrument_ids=("instrument:equity:us:tsla",),
        provenance=provenance,
        metadata={
            "stance": "contradicts",
            "extraction_confidence": 0.8,
            "source_reliability": "fixture",
        },
    )
    service.store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-prior-contradiction",
            tool_run_id="tool-prior-contradiction",
            source_query_id="query-prior-contradiction",
            source_type=SourceKind.NEWS_ARTICLE.value,
            provider="fixture-news",
            url="https://example.test/tsla-prior-contradiction",
            query="TSLA contradictory follow-up",
            retrieved_at=NOW,
            published_at=NOW,
            instruments=("instrument:equity:us:tsla",),
            claim=source_evidence.text,
            extraction_confidence=0.8,
            source_reliability="fixture",
            freshness_status=FreshnessStatus.FRESH.value,
            raw_excerpt=source_evidence.text,
            provenance_json=cast(JsonObject, provenance.model_dump(mode="json")),
            metadata={
                "stance": "contradicts",
                "source_evidence": cast(JsonObject, source_evidence.model_dump(mode="json")),
                **report_data_mode_metadata(OFFLINE_FIXTURE_REPORT_DATA_MODE),
            },
        )
    )
    service.store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-prior-contradiction",
            run_id=run_id,
            instrument_id="instrument:equity:us:tsla",
            prediction_horizon="swing",
            prediction_type="directional",
            scenario="Follow-up evidence contradicts the prior TSLA scenario.",
            status="contradicted",
            confidence=0.32,
            direction="bearish",
            evidence_against=("evidence-prior-contradiction",),
            baseline={"summary": "No directional edge is assumed without source-backed evidence."},
            uncertainty="Single follow-up source limits confidence.",
            metadata={"symbol": "TSLA"},
        )
    )
