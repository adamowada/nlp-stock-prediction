from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.contracts import (
    EvidenceReference,
    FreshnessStatus,
    JsonObject,
    PredictionOutcomeResult,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
)
from nlp_stock_prediction.evaluation.outcomes import (
    PointInTimeOutcomeEvaluationArtifacts,
    build_prediction_evaluation_target,
    write_point_in_time_outcome_evaluation_artifacts,
)
from nlp_stock_prediction.orchestration import Phase4Service
from nlp_stock_prediction.orchestration.report_data_modes import OFFLINE_FIXTURE_REPORT_DATA_MODE
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    EvidenceRecord,
    SourceQueryRecord,
    ToolRunRecord,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTCOME_EVIDENCE_ID = "evidence-tsla-prior-window-close"
NOW = datetime(2026, 5, 13, 22, 0, tzinfo=UTC)


@pytest.mark.integration
def test_cross_run_prior_review_uses_persisted_outcome_payload(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    prior = service.run_offline_phase4_flow(
        run_date="2026-05-12",
        output_dir="reports/phase7-outcome-summaries",
        symbol="TSLA",
    )
    persisted = _write_persisted_prior_outcome(service, prior=prior)

    current = service.run_offline_phase4_flow(
        run_date="2026-05-13",
        output_dir="reports/phase7-outcome-summaries",
        symbol="TSLA",
    )

    payload = _payload(current)
    markdown = Path(str(cast(dict[str, object], current["report"])["markdown_path"])).read_text(
        encoding="utf-8"
    )
    review = cast(dict[str, Any], payload["prior_outcome_reviews"][0])
    source_refs = cast(list[dict[str, Any]], payload["source_references"])
    traces = cast(list[dict[str, Any]], payload["material_claim_traces"])
    evidence_ids = {
        str(evidence["evidence_id"])
        for evidence in cast(list[dict[str, Any]], payload["evidence_sources"])
    }
    audit_artifacts = {
        str(artifact["artifact_id"]): artifact
        for artifact in cast(list[dict[str, Any]], payload["audit_manifest"]["artifacts"])
    }

    assert review["status"] == "available"
    assert review["review_id"] == persisted.outcome_evaluation.outcome_evaluation_id
    assert review["metadata"]["source"] == "persisted_prediction_outcome_evaluation"
    assert review["metadata"]["prior_candidate_id"] == _candidate_id(prior)
    assert review["metadata"]["prediction_outcome"]["outcome_id"] == persisted.outcome.outcome_id
    assert (
        review["metadata"]["prediction_outcome_evaluation"]["outcome_evaluation_id"]
        == persisted.outcome_evaluation.outcome_evaluation_id
    )
    assert review["metadata"]["phase7_freshness"]["evidence_aging_records"][0]["age_status"] == (
        "stale"
    )
    assert review["outcome_evidence"][0]["evidence_id"] == OUTCOME_EVIDENCE_ID
    assert OUTCOME_EVIDENCE_ID in evidence_ids
    assert persisted.outcome_evaluation_artifact.artifact_id in audit_artifacts
    assert any(review["review_id"] in ref["prior_outcome_review_ids"] for ref in source_refs)
    assert any(review["review_id"] in trace["prior_outcome_review_ids"] for trace in traces)
    assert "Aged evidence" in markdown
    assert OUTCOME_EVIDENCE_ID in markdown


@pytest.mark.integration
@pytest.mark.parametrize("failure_mode", ("missing", "hash_mismatch", "malformed"))
def test_invalid_persisted_outcome_artifact_blocks_synthetic_prior_history(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    service = _service(tmp_path)
    prior = service.run_offline_phase4_flow(
        run_date="2026-05-12",
        output_dir=f"reports/phase7-outcome-summaries-{failure_mode}",
        symbol="TSLA",
    )
    persisted = _write_persisted_prior_outcome(service, prior=prior)
    artifact_id = persisted.outcome_evaluation_artifact.artifact_id
    artifact = service.store.get_artifact(artifact_id)
    assert artifact is not None
    artifact_path = tmp_path / artifact.path
    if failure_mode == "missing":
        artifact_path.unlink()
    elif failure_mode == "hash_mismatch":
        artifact_path.write_text('{"schema_version":"corrupted"}\n', encoding="utf-8")
    else:
        artifact_path.write_text('{"schema_version":"corrupted"}\n', encoding="utf-8")
        service.store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact.artifact_id,
                tool_run_id=artifact.tool_run_id,
                artifact_type=artifact.artifact_type,
                path=artifact.path,
                sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                schema_version=artifact.schema_version,
                produced_by=artifact.produced_by,
                record_count=artifact.record_count,
                metadata=artifact.metadata,
                created_at=artifact.created_at,
            )
        )

    current = service.run_offline_phase4_flow(
        run_date="2026-05-13",
        output_dir=f"reports/phase7-outcome-summaries-{failure_mode}",
        symbol="TSLA",
    )

    review = cast(dict[str, Any], _payload(current)["prior_outcome_reviews"][0])

    assert review["status"] == "unavailable"
    assert review["metadata"]["source"] == "persisted_prediction_outcome_evaluation"
    assert "prediction_outcome" not in review["metadata"]
    limitation_text = " ".join(cast(list[str], review["limitations"])).lower()
    assert (
        "missing" in limitation_text
        if failure_mode == "missing"
        else "sha256 mismatch" in limitation_text
        if failure_mode == "hash_mismatch"
        else "malformed" in limitation_text
    )


def _service(tmp_path: Path) -> Phase4Service:
    return Phase4Service(
        repo_root=tmp_path,
        fixture_root=REPO_ROOT,
        database_path=Path("data") / "prediction-research.sqlite3",
    )


def _payload(result: Mapping[str, object]) -> dict[str, Any]:
    report = cast(dict[str, object], result["report"])
    return cast(dict[str, Any], json.loads(Path(str(report["json_path"])).read_text()))


def _candidate_id(result: Mapping[str, object]) -> str:
    candidates = cast(list[dict[str, Any]], _payload(result)["prediction_candidates"])
    return str(candidates[0]["candidate_id"])


def _write_persisted_prior_outcome(
    service: Phase4Service,
    *,
    prior: Mapping[str, object],
) -> PointInTimeOutcomeEvaluationArtifacts:
    prior_payload = _payload(prior)
    run_id = str(prior["run_id"])
    prior_candidate = cast(list[dict[str, Any]], prior_payload["prediction_candidates"])[0]
    candidate_id = str(prior_candidate["candidate_id"])
    _record_outcome_evidence(service, run_id=run_id)
    target = build_prediction_evaluation_target(
        store=service.store,
        run_id=run_id,
        candidate_id=candidate_id,
        point_in_time_cutoff=datetime(2026, 5, 12, 20, 0, tzinfo=UTC),
        evaluation_window_start=datetime(2026, 5, 13, 13, 30, tzinfo=UTC),
        evaluation_window_end=datetime(2026, 5, 13, 20, 0, tzinfo=UTC),
    )
    phase7_freshness = {
        "reviewed_at": "2026-05-13T20:00:00+00:00",
        "evidence_aging_records": [
            {
                "evidence_id": _aged_evidence_id(prior_candidate, target.evidence_ids),
                "reviewed_at": "2026-05-13T20:00:00+00:00",
                "retrieved_at": "2026-05-12T12:00:00+00:00",
                "published_at": "2026-05-12T12:00:00+00:00",
                "age_seconds": 115200.0,
                "age_status": "stale",
                "freshness_status": "stale",
                "source_type": "news_article",
                "provider": "fixture-news",
                "metadata": {},
            }
        ],
        "artifact_freshness_reviews": [],
    }
    target = target.model_copy(
        update={
            "metadata": {**target.metadata, "phase7_freshness": phase7_freshness},
            "candidate_snapshot": {
                **target.candidate_snapshot,
                "phase7_freshness": phase7_freshness,
            },
        }
    )
    return write_point_in_time_outcome_evaluation_artifacts(
        store=service.store,
        repo_root=service.repo_root,
        artifact_dir=service.repo_root / "reports" / "phase7-outcome-summaries" / "audit",
        run_id=run_id,
        target=target,
        observed_result=PredictionOutcomeResult.SUPPORTED,
        observed_at=datetime(2026, 5, 13, 20, 15, tzinfo=UTC),
        result_summary="Persisted market evidence supported the prior TSLA scenario.",
        outcome_evidence=(EvidenceReference(evidence_id=OUTCOME_EVIDENCE_ID),),
        created_at=datetime(2026, 5, 13, 20, 20, tzinfo=UTC),
        evaluated_at=NOW,
    )


def _aged_evidence_id(
    prior_candidate: dict[str, Any],
    target_evidence_ids: tuple[str, ...],
) -> str:
    for field_name in ("evidence_for", "evidence_against"):
        references = prior_candidate.get(field_name)
        if not isinstance(references, list):
            continue
        for reference in references:
            if isinstance(reference, dict) and isinstance(reference.get("evidence_id"), str):
                return str(reference["evidence_id"])
    return str(target_evidence_ids[0]) if target_evidence_ids else OUTCOME_EVIDENCE_ID


def _record_outcome_evidence(service: Phase4Service, *, run_id: str) -> None:
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-tsla-prior-window-close",
            run_id=run_id,
            tool_name="phase7_prior_outcome_summary_test",
            tool_version="phase7.test.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "TSLA"},
        )
    )
    service.store.record_source_query(
        SourceQueryRecord(
            source_query_id="query-tsla-prior-window-close",
            tool_run_id="tool-tsla-prior-window-close",
            provider="verified-market-data",
            query="TSLA prior outcome close",
            retrieved_at=NOW,
            url="https://example.test/tsla-prior-outcome",
        )
    )
    provenance = SourceProvenance(
        provider_name="verified-market-data",
        source_kind=SourceKind.MARKET_DATA,
        retrieval_method=RetrievalMethod.DERIVED,
        fetched_at=NOW,
        observed_at=NOW,
        source_url="https://example.test/tsla-prior-outcome",
        permalink="https://example.test/tsla-prior-outcome",
        raw_identifier="tsla-prior-close",
        raw_snapshot_id="artifact-tsla-prior-close",
        query="TSLA prior outcome close",
        freshness_status=FreshnessStatus.FRESH,
    )
    source_evidence = SourceEvidence(
        evidence_id=OUTCOME_EVIDENCE_ID,
        source_kind=SourceKind.MARKET_DATA,
        ticker="TSLA",
        text="TSLA prior window close supported the stored prediction scenario.",
        created_at=NOW,
        permalink="https://example.test/tsla-prior-outcome",
        instrument_id="instrument:equity:us:tsla",
        matched_tickers=("TSLA",),
        matched_instrument_ids=("instrument:equity:us:tsla",),
        provenance=provenance,
        metadata={"stance": "supports", "extraction_confidence": 0.9},
    )
    service.store.record_evidence(
        EvidenceRecord(
            evidence_id=OUTCOME_EVIDENCE_ID,
            tool_run_id="tool-tsla-prior-window-close",
            source_query_id="query-tsla-prior-window-close",
            source_type=SourceKind.MARKET_DATA.value,
            provider="verified-market-data",
            url="https://example.test/tsla-prior-outcome",
            query="TSLA prior outcome close",
            retrieved_at=NOW,
            published_at=NOW,
            instruments=("instrument:equity:us:tsla",),
            claim=source_evidence.text,
            extraction_confidence=0.9,
            source_reliability="verified_market_data",
            freshness_status=FreshnessStatus.FRESH.value,
            raw_excerpt=source_evidence.text,
            provenance_json=cast(JsonObject, provenance.model_dump(mode="json")),
            metadata={
                "stance": "supports",
                "source_evidence": cast(JsonObject, source_evidence.model_dump(mode="json")),
                "report_data_mode": OFFLINE_FIXTURE_REPORT_DATA_MODE,
                "provider_mode": OFFLINE_FIXTURE_REPORT_DATA_MODE,
            },
        )
    )
