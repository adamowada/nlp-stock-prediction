from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.orchestration import (
    OFFLINE_FIXTURE_REPORT_DATA_MODE,
    Phase4Service,
)
from nlp_stock_prediction.reporting.json import load_json_report

RUN_DATE = date(2026, 5, 13)
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_phase5_json_report_round_trips_and_final_artifacts_are_indexed(
    tmp_path: Path,
) -> None:
    service = Phase4Service(
        repo_root=tmp_path,
        fixture_root=REPO_ROOT,
        database_path=Path("data") / "prediction-research.sqlite3",
    )

    result = service.run_offline_phase4_flow(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/phase5-json-report-index",
        symbol="TSLA",
    )

    run_id = str(result["run_id"])
    report_result = cast(dict[str, object], result["report"])
    json_path = Path(str(report_result["json_path"]))
    report = load_json_report(json_path.read_text(encoding="utf-8"))
    report_artifacts = service.store.list_report_artifacts_for_run(run_id)
    artifacts_by_id = {
        artifact.artifact_id: artifact for artifact in service.store.list_artifacts_for_run(run_id)
    }

    assert report.run_id == run_id
    assert tuple(artifact.artifact_type for artifact in report_artifacts) == (
        "markdown_report",
        "json_report",
        "audit_manifest",
    )
    assert len(report_artifacts) == 3
    for report_artifact in report_artifacts:
        indexed_artifact = artifacts_by_id[report_artifact.artifact_id]
        artifact_path = tmp_path / report_artifact.path
        assert artifact_path.exists()
        assert report_artifact.sha256 == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert report_artifact.sha256 == indexed_artifact.sha256
        assert report_artifact.path == indexed_artifact.path
        assert report_artifact.schema_version == indexed_artifact.schema_version
        assert report_artifact.report_schema_version == report.schema_version
        assert report_artifact.report_date == RUN_DATE
        assert report_artifact.instrument_id == report.instruments[0].instrument_id
        assert report_artifact.symbol == "TSLA"
        assert report_artifact.report_data_mode == OFFLINE_FIXTURE_REPORT_DATA_MODE
        assert report_artifact.created_at is not None
        assert report_artifact.source_run_started_at <= report_artifact.created_at
        assert report_artifact.source_run_completed_at is not None
        assert report_artifact.metadata["candidate_count"] == len(report.prediction_candidates)
        assert "prediction_candidates" not in report_artifact.metadata

    latest_json = service.store.get_latest_report_artifact(
        artifact_type="json_report",
        report_date=RUN_DATE,
        instrument_id=report.instruments[0].instrument_id,
    )
    assert latest_json is not None
    assert latest_json.artifact_type == "json_report"
    assert latest_json.path == json_path.relative_to(tmp_path)
    assert service.store.list_latest_report_artifact_bundle(symbol="TSLA") == report_artifacts
