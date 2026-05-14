from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.orchestration import Phase4Service
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ToolRunRecord,
)

RUN_DATE = date(2026, 5, 13)
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_report_assembly_traces_candidates_to_stored_evidence_and_artifacts(
    tmp_path: Path,
) -> None:
    service = Phase4Service(
        repo_root=tmp_path,
        fixture_root=REPO_ROOT,
        database_path=Path("data") / "prediction-research.sqlite3",
    )

    result = service.run_offline_phase4_flow(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/phase5-report-assembly",
        symbol="TSLA",
    )

    report = cast(dict[str, object], result["report"])
    payload = _read_report_payload(report)
    candidates = cast(list[dict[str, Any]], payload["prediction_candidates"])
    candidate = candidates[0]
    candidate_id = cast(str, candidate["candidate_id"])
    evidence_for = cast(list[dict[str, Any]], candidate["evidence_for"])
    candidate_metadata = cast(dict[str, Any], candidate["metadata"])
    evaluation_metadata = cast(dict[str, Any], candidate_metadata["prediction_evaluation"])
    evaluation_artifact_id = cast(str, evaluation_metadata["artifact_id"])

    audit_manifest = cast(dict[str, Any], payload["audit_manifest"])
    audit_artifacts = cast(list[dict[str, Any]], audit_manifest["artifacts"])
    artifacts = {artifact["artifact_id"]: artifact for artifact in audit_artifacts}
    assert all(artifact["path"] for artifact in artifacts.values())
    assert all(artifact["sha256"] for artifact in artifacts.values())
    assert artifacts[evaluation_artifact_id]["artifact_type"] == "prediction_evaluation"

    source_references = cast(list[dict[str, Any]], payload["source_references"])
    assert any(
        reference["reference_type"] == "prediction_evaluation"
        and evaluation_artifact_id in reference["artifact_ids"]
        and candidate_id in reference["candidate_ids"]
        for reference in source_references
    )
    assert any(
        reference["reference_type"] == "source_evidence"
        and evidence_for[0]["evidence_id"] in reference["evidence_ids"]
        and reference["candidate_ids"] == [candidate_id]
        for reference in source_references
    )

    traces = {
        trace["claim_id"]: trace
        for trace in cast(list[dict[str, Any]], payload["material_claim_traces"])
    }
    prediction_input_ids = [
        artifact_id
        for artifact_id, artifact in artifacts.items()
        if artifact["artifact_type"] == "prediction_input"
    ]
    assert traces[f"claim-{candidate_id}-baseline"]["claim_type"] == "baseline"
    assert evaluation_artifact_id in traces[f"claim-{candidate_id}-evaluation"]["artifact_ids"]
    assert any(
        artifact_id in traces[f"claim-{candidate_id}"]["artifact_ids"]
        for artifact_id in prediction_input_ids
    )


@pytest.mark.integration
def test_report_assembly_missing_candidate_artifact_becomes_insufficient_evidence(
    tmp_path: Path,
) -> None:
    service, run_id = _service_with_started_run(tmp_path)
    service.store.upsert_prediction_candidate(
        _candidate(signal_artifacts=("artifact-missing-technical",))
    )

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    assert rendered["status"] == "empty"
    assert rendered["candidate_count"] == 0
    assert rendered["stored_candidate_count"] == 1
    assert rendered["excluded_candidate_ids"] == ["candidate-stage3-assembly"]
    rendered_warnings = cast(list[str], rendered["warnings"])
    assert any("Missing required artifact" in warning for warning in rendered_warnings)

    payload = _read_report_payload(rendered)
    insufficient = cast(dict[str, Any], payload["insufficient_evidence"])
    assert payload["prediction_candidates"] == []
    assert any(
        "Missing required artifact artifact-missing-technical" in reason
        for reason in insufficient["blocking_reasons"]
    )
    assert "report-assembly" in insufficient["provider_names"]
    provider_health = cast(list[dict[str, Any]], payload["provider_health"])
    assert any(
        health["provider_name"] == "report-assembly" and health["status"] == "malformed"
        for health in provider_health
    )


@pytest.mark.integration
def test_report_assembly_malformed_required_artifact_becomes_insufficient_evidence(
    tmp_path: Path,
) -> None:
    service, run_id = _service_with_started_run(tmp_path)
    artifact_id = "artifact-bad-technical"
    artifact_path = _write_bad_technical_artifact(service=service, run_id=run_id, tmp_path=tmp_path)
    service.store.upsert_prediction_candidate(_candidate(signal_artifacts=(artifact_id,)))
    service.store.record_artifact(
        ArtifactRecord(
            artifact_id=artifact_id,
            artifact_type="technical_package",
            path=artifact_path.relative_to(tmp_path),
            sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            schema_version="phase4.technical-package.v1",
            tool_run_id="tool-bad-technical",
            produced_by="phase4_technical_package",
            record_count=1,
            metadata={"symbol": "TSLA"},
            created_at=NOW,
        )
    )

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    assert rendered["status"] == "empty"
    assert rendered["candidate_count"] == 0
    assert rendered["excluded_candidate_ids"] == ["candidate-stage3-assembly"]
    rendered_warnings = cast(list[str], rendered["warnings"])
    assert any("Artifact payload is malformed" in warning for warning in rendered_warnings)

    payload = _read_report_payload(rendered)
    audit_manifest = cast(dict[str, Any], payload["audit_manifest"])
    audit_artifacts = cast(list[dict[str, Any]], audit_manifest["artifacts"])
    artifact = next(
        artifact for artifact in audit_artifacts if artifact["artifact_id"] == artifact_id
    )
    assert artifact["metadata"]["assembly_required"] is True
    assert artifact["metadata"]["assembly_status"] == "warning"
    insufficient = cast(dict[str, Any], payload["insufficient_evidence"])
    assert any(
        "Artifact payload is malformed" in reason for reason in insufficient["blocking_reasons"]
    )


def _service_with_started_run(tmp_path: Path) -> tuple[Phase4Service, str]:
    service = Phase4Service(repo_root=tmp_path)
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/phase5-report-assembly",
        symbol="TSLA",
        objective="Stage 3 report assembly test.",
    )
    service.store.upsert_instrument(
        InstrumentRecord(
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            asset_class="stock",
            name="Tesla Inc.",
        )
    )
    return service, str(started["run_id"])


def _candidate(*, signal_artifacts: tuple[str, ...]) -> PredictionCandidateRecord:
    return PredictionCandidateRecord(
        candidate_id="candidate-stage3-assembly",
        run_id=f"phase4-{RUN_DATE.isoformat()}-tsla",
        instrument_id="instrument:equity:us:tsla",
        prediction_horizon="swing",
        prediction_type="directional",
        scenario="TSLA report assembly candidate depends on stored artifact integrity.",
        status="insufficient_evidence",
        confidence=0.18,
        direction="mixed",
        signal_artifacts=signal_artifacts,
        baseline={"summary": "No directional edge is assumed without source-backed evidence."},
        uncertainty="Artifact availability and schema validity limit confidence.",
        metadata={"symbol": "TSLA"},
    )


def _write_bad_technical_artifact(
    *,
    service: Phase4Service,
    run_id: str,
    tmp_path: Path,
) -> Path:
    run = service.store.get_research_run(run_id)
    assert run is not None
    audit_dir = Path(str(run.metadata["output_dir"]))
    if not audit_dir.is_absolute():
        audit_dir = tmp_path / audit_dir
    artifact_path = (
        audit_dir / RUN_DATE.isoformat() / "tsla" / "audit" / "technical-package" / "bad.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("{not-json", encoding="utf-8")
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-bad-technical",
            run_id=run_id,
            tool_name="phase4_technical_package",
            tool_version="phase4.technical-package.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "TSLA"},
        )
    )
    return artifact_path


def _read_report_payload(report: Mapping[str, object]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path(str(report["json_path"])).read_text(encoding="utf-8")),
    )
