from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    Direction,
    EvidenceReference,
    FreshnessStatus,
    PredictionOutcomeResult,
    PredictionType,
    RetrievalMethod,
    SourceKind,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.evaluation import (
    build_prediction_evaluation_target,
    write_point_in_time_outcome_evaluation_artifacts,
)
from nlp_stock_prediction.orchestration.phase2_report import _primary_instrument
from nlp_stock_prediction.orchestration.phase4_service import Phase4Service
from nlp_stock_prediction.orchestration.phase6_service import Phase6Service
from nlp_stock_prediction.orchestration.report_trace import build_report_trace
from nlp_stock_prediction.storage import (
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ToolRunRecord,
)

RUN_DATE = date(2026, 5, 13)
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 5, 13, 20, 0, tzinfo=UTC)
WINDOW_START = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 18, 20, 0, tzinfo=UTC)
OBSERVED_AT = datetime(2026, 5, 18, 20, 5, tzinfo=UTC)
EVALUATED_AT = datetime(2026, 5, 18, 21, 0, tzinfo=UTC)
AS_OF = datetime(2026, 5, 22, 0, 0, tzinfo=UTC)
INSTRUMENT_ID = "instrument:equity:us:msft"


def _started_service(tmp_path: Path) -> tuple[Phase4Service, str, Path]:
    service = Phase4Service(repo_root=tmp_path)
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/phase6-report-integration",
        symbol="MSFT",
        objective="Phase 6 report integration test.",
    )
    run_id = str(started["run_id"])
    audit_dir = Path(str(started["audit_dir"]))
    run = service.store.get_research_run(run_id)
    assert run is not None
    service.store.upsert_research_run(replace(run, started_at=NOW))
    return service, run_id, audit_dir


def _seed_candidate(service: Phase4Service, run_id: str) -> None:
    service.store.upsert_instrument(
        InstrumentRecord(
            instrument_id=INSTRUMENT_ID,
            symbol="MSFT",
            asset_class="stock",
            name="Microsoft Corporation",
        )
    )
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-phase6-report-evidence",
            run_id=run_id,
            tool_name="phase6_report_integration_evidence",
            tool_version="test.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "MSFT"},
        )
    )
    service.store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-msft-support",
            tool_run_id="tool-phase6-report-evidence",
            source_type=SourceKind.NEWS_ARTICLE.value,
            provider="verified-news",
            retrieved_at=NOW,
            published_at=NOW,
            instruments=(INSTRUMENT_ID,),
            claim="Verified evidence supported the monitored MSFT scenario before cutoff.",
            freshness_status=FreshnessStatus.FRESH.value,
            provenance_json={
                "provider_name": "verified-news",
                "source_kind": SourceKind.NEWS_ARTICLE.value,
                "retrieval_method": RetrievalMethod.OFFICIAL_API.value,
                "fetched_at": NOW.isoformat(),
                "observed_at": NOW.isoformat(),
                "source_url": "https://example.test/msft/support",
                "permalink": "https://example.test/msft/support",
                "raw_identifier": "evidence-msft-support",
                "raw_snapshot_id": "raw-evidence-msft-support",
                "freshness_status": FreshnessStatus.FRESH.value,
            },
        )
    )
    service.store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-msft-outcome",
            tool_run_id="tool-phase6-report-evidence",
            source_type=SourceKind.MARKET_DATA.value,
            provider="verified-market-data",
            retrieved_at=OBSERVED_AT,
            published_at=OBSERVED_AT,
            instruments=(INSTRUMENT_ID,),
            claim="MSFT closed above the baseline comparison at the evaluation window.",
            freshness_status=FreshnessStatus.FRESH.value,
            provenance_json={
                "provider_name": "verified-market-data",
                "source_kind": SourceKind.MARKET_DATA.value,
                "retrieval_method": RetrievalMethod.OFFICIAL_API.value,
                "fetched_at": OBSERVED_AT.isoformat(),
                "observed_at": OBSERVED_AT.isoformat(),
                "source_url": "https://example.test/msft/outcome",
                "permalink": "https://example.test/msft/outcome",
                "raw_identifier": "evidence-msft-outcome",
                "raw_snapshot_id": "raw-evidence-msft-outcome",
                "freshness_status": FreshnessStatus.FRESH.value,
            },
        )
    )
    service.store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-msft-phase6-report",
            run_id=run_id,
            instrument_id=INSTRUMENT_ID,
            prediction_horizon=TimeHorizon.SWING.value,
            prediction_type=PredictionType.DIRECTIONAL.value,
            scenario="MSFT evidence supported a monitored upside scenario.",
            direction=Direction.BULLISH.value,
            confidence=0.64,
            status="evidence_supported",
            evidence_for=("evidence-msft-support",),
            baseline={
                "summary": "No directional edge is assumed without source-backed evidence.",
                "baseline_score": 0.5,
            },
            uncertainty="Outcome review depends on attributed post-window evidence.",
            metadata={"symbol": "MSFT"},
        )
    )


@pytest.mark.integration
def test_rendered_report_surfaces_phase6_outcomes_and_calibration_artifacts(
    tmp_path: Path,
) -> None:
    service, run_id, audit_dir = _started_service(tmp_path)
    _seed_candidate(service, run_id)
    target = build_prediction_evaluation_target(
        store=service.store,
        run_id=run_id,
        candidate_id="candidate-msft-phase6-report",
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
    )
    outcome_artifacts = write_point_in_time_outcome_evaluation_artifacts(
        store=service.store,
        repo_root=tmp_path,
        artifact_dir=audit_dir,
        run_id=run_id,
        target=target,
        observed_result=PredictionOutcomeResult.SUPPORTED,
        observed_at=OBSERVED_AT,
        result_summary="MSFT closed above the comparison baseline.",
        outcome_evidence=(EvidenceReference(evidence_id="evidence-msft-outcome"),),
        created_at=OBSERVED_AT,
        evaluated_at=EVALUATED_AT,
    )
    calibration = Phase6Service(repo_root=tmp_path).phase6_calibration_summary(
        run_id=run_id,
        cohort_id="phase6-evalcal-stage8-msft-swing",
        as_of=AS_OF.isoformat(),
        artifact_dir=audit_dir.as_posix(),
        bin_edges=(0.0, 0.5, 1.0),
    )

    rendered = service.render_prediction_report(run_id=run_id, symbol="MSFT")
    rerendered = service.render_prediction_report(run_id=run_id, symbol="MSFT")

    payload = json.loads(Path(str(rendered["json_path"])).read_text(encoding="utf-8"))
    rerendered_payload = json.loads(Path(str(rerendered["json_path"])).read_text(encoding="utf-8"))
    candidate = payload["prediction_candidates"][0]
    review = payload["prior_outcome_reviews"][0]
    source_references = cast(list[dict[str, object]], payload["source_references"])
    artifact_types = {
        artifact["artifact_type"] for artifact in payload["audit_manifest"]["artifacts"]
    }
    markdown = Path(str(rendered["markdown_path"])).read_text(encoding="utf-8")

    assert (
        outcome_artifacts.outcome_evaluation.outcome_evaluation_id
        in candidate["prior_outcome_review_ids"]
    )
    assert review["status"] == "available"
    assert review["candidate_id"] == "candidate-msft-phase6-report"
    assert review["artifact_ids"]
    assert any(
        _references_prior_review(reference, str(review["review_id"]))
        for reference in source_references
    )
    assert any(
        _references_artifact(reference, str(calibration["artifact_id"]))
        for reference in source_references
    )
    assert {
        "prediction_outcome",
        "prediction_outcome_evaluation",
        "calibration_summary",
    }.issubset(artifact_types)
    assert "(calibration_summary)" in markdown
    assert "(prediction_outcome_evaluation)" in markdown
    assert "Stored outcome evaluation is confirmed" in markdown
    assert [
        artifact["artifact_id"]
        for artifact in rerendered_payload["audit_manifest"]["artifacts"]
        if artifact["artifact_type"] in {"markdown_report", "json_report", "audit_manifest"}
    ] == [
        artifact["artifact_id"]
        for artifact in payload["audit_manifest"]["artifacts"]
        if artifact["artifact_type"] in {"markdown_report", "json_report", "audit_manifest"}
    ]
    assert (
        sum(
            artifact["artifact_type"] in {"markdown_report", "json_report", "audit_manifest"}
            for artifact in rerendered_payload["audit_manifest"]["artifacts"]
        )
        == 0
    )


def _references_prior_review(reference: dict[str, object], review_id: str) -> bool:
    prior_review_ids = reference.get("prior_outcome_review_ids")
    return (
        reference.get("reference_type") == "prior_outcome"
        and isinstance(prior_review_ids, list)
        and review_id in prior_review_ids
    )


def _references_artifact(reference: dict[str, object], artifact_id: str) -> bool:
    artifact_ids = reference.get("artifact_ids")
    return (
        reference.get("reference_type") == "tool_artifact"
        and isinstance(artifact_ids, list)
        and artifact_id in artifact_ids
    )


@pytest.mark.unit
def test_report_trace_limits_cohort_artifacts_to_source_candidates() -> None:
    trace = build_report_trace(
        evidence_sources=(),
        prediction_candidates=(
            SimpleNamespace(
                candidate_id="candidate-in-cohort",
                thesis="Candidate in calibration cohort.",
                evidence_for=(),
                evidence_against=(),
                signal_artifact_ids=(),
                signal_artifacts=(),
                prior_outcome_review_ids=(),
            ),
            SimpleNamespace(
                candidate_id="candidate-outside-cohort",
                thesis="Candidate outside calibration cohort.",
                evidence_for=(),
                evidence_against=(),
                signal_artifact_ids=(),
                signal_artifacts=(),
                prior_outcome_review_ids=(),
            ),
        ),
        audit_artifacts=(
            AuditArtifact(
                artifact_id="artifact-calibration-cohort",
                artifact_type="calibration_summary",
                path="reports/run/audit/calibration.json",
                created_at=NOW,
                produced_by="phase6_calibration_summary",
                sha256="a" * 64,
                metadata={"source_candidate_ids": ["candidate-in-cohort"]},
            ),
        ),
        prior_outcome_reviews=(),
    )

    references_by_candidate = {
        trace.candidate_ids[0]: trace.source_reference_ids for trace in trace.material_claim_traces
    }
    assert references_by_candidate["candidate-in-cohort"] == (
        "source-ref-artifact-calibration-cohort",
    )
    assert references_by_candidate["candidate-outside-cohort"] == ()


@pytest.mark.unit
def test_report_primary_instrument_rejects_ambiguous_symbol(tmp_path: Path) -> None:
    service = Phase4Service(repo_root=tmp_path)
    service.store.upsert_instrument(
        InstrumentRecord(
            instrument_id="instrument:equity:us:ai",
            symbol="AI",
            asset_class="stock",
            name="C3.ai",
        )
    )
    service.store.upsert_instrument(
        InstrumentRecord(
            instrument_id="instrument:crypto:ai-usd",
            symbol="AI",
            asset_class="crypto",
            name="AI token",
        )
    )

    with pytest.raises(ValueError, match="ambiguous instrument symbol"):
        _primary_instrument(
            service.store,
            symbol="AI",
            fallback_generated_at=NOW,
        )
