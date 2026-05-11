from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.cli import CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
from nlp_stock_prediction.contracts import (
    AuditManifest,
    DailyReport,
    JsonObject,
    RiskProfile,
    RunConfig,
)
from nlp_stock_prediction.pipeline import generate_daily_report
from nlp_stock_prediction.reporting.audit import json_payload_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
EXPECTED_AUDIT_FILES = {
    "raw-snapshots.json",
    "normalized-evidence.json",
    "extracted-strategies.json",
    "analysis-contexts.json",
    "scoring-inputs.json",
    "final-reports.json",
    "audit-manifest.json",
}


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_DIR) if not existing_pythonpath else f"{SRC_DIR}{os.pathsep}{existing_pythonpath}"
    )
    return env


def _read_json_object(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(JsonObject, payload)


def _json_records(payload: JsonObject) -> list[JsonObject]:
    records = payload["records"]
    assert isinstance(records, list)
    assert all(isinstance(record, dict) for record in records)
    return cast(list[JsonObject], records)


@pytest.mark.e2e
def test_generate_daily_report_writes_markdown_json_and_audit_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"

    bundle = generate_daily_report(
        RunConfig(
            run_date=date(2026, 5, 11),
            output_dir=output_dir,
            risk_profile=RiskProfile.EXPLORATORY,
            offline=True,
        )
    )

    report_dir = output_dir / "2026-05-11"
    audit_dir = report_dir / "audit"

    assert bundle.report_dir == report_dir
    assert bundle.markdown_path == report_dir / "report.md"
    assert bundle.json_path == report_dir / "report.json"
    assert bundle.audit_dir == audit_dir
    assert bundle.markdown_path.exists()
    assert bundle.json_path.exists()
    assert bundle.audit_manifest_path == audit_dir / "audit-manifest.json"
    assert bundle.audit_manifest_path.exists()

    assert {path.name for path in audit_dir.iterdir()} == EXPECTED_AUDIT_FILES

    report = DailyReport.model_validate_json(bundle.json_path.read_text(encoding="utf-8"))
    assert report.report_date == date(2026, 5, 11)
    assert tuple(section.ticker for section in report.ticker_sections) == (
        "TSLA",
        "NVDA",
        "AMD",
        "AAPL",
        "MU",
        "SPY",
    )
    assert report.trade_candidates[0].candidate_id == "candidate-tsla-shares-swing"
    assert isinstance(report.audit_manifest, AuditManifest)

    manifest = AuditManifest.model_validate(_read_json_object(bundle.audit_manifest_path))
    assert manifest == report.audit_manifest
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in manifest.artifacts}
    assert set(artifacts_by_id) == {
        "raw-snapshots",
        "normalized-evidence",
        "extracted-strategies",
        "analysis-contexts",
        "scoring-inputs",
        "markdown-report",
        "json-report",
    }
    assert {artifact.artifact_type for artifact in manifest.artifacts} == {
        "raw_snapshot",
        "normalized_evidence",
        "extraction_output",
        "analysis_context",
        "scoring_input",
        "markdown_report",
        "json_report",
    }

    audit_payload_ids = {
        "raw-snapshots": "raw-snapshots.json",
        "normalized-evidence": "normalized-evidence.json",
        "extracted-strategies": "extracted-strategies.json",
        "analysis-contexts": "analysis-contexts.json",
        "scoring-inputs": "scoring-inputs.json",
    }
    for artifact_id, filename in audit_payload_ids.items():
        artifact = artifacts_by_id[artifact_id]
        artifact_path = audit_dir / filename
        payload = _read_json_object(artifact_path)
        records = _json_records(payload)

        assert Path(artifact.path) == artifact_path
        assert artifact_path.exists()
        assert artifact.record_count == len(records)
        assert artifact.sha256 == json_payload_sha256(payload)

    assert Path(artifacts_by_id["markdown-report"].path) == bundle.markdown_path
    assert Path(artifacts_by_id["json-report"].path) == bundle.json_path
    assert artifacts_by_id["markdown-report"].record_count == 1
    assert artifacts_by_id["json-report"].record_count == 1

    final_report_records = _json_records(_read_json_object(audit_dir / "final-reports.json"))
    assert final_report_records == [
        {
            "artifact_id": "markdown-report",
            "path": bundle.markdown_path.as_posix(),
            "content_type": "text/markdown",
        },
        {
            "artifact_id": "json-report",
            "path": bundle.json_path.as_posix(),
            "content_type": "application/json",
        },
    ]

    raw_records = _json_records(_read_json_object(audit_dir / "raw-snapshots.json"))
    assert [record["source_kind"] for record in raw_records] == [
        "reddit_ticker_card",
        "reddit_comment",
        "market_data",
    ]
    assert all(
        cast(JsonObject, record["provider_metadata"])["fixture"] is True for record in raw_records
    )

    normalized_records = _json_records(_read_json_object(audit_dir / "normalized-evidence.json"))
    first_normalized = normalized_records[0]
    first_provenance = cast(JsonObject, first_normalized["provenance"])
    first_provider_metadata = cast(JsonObject, first_provenance["provider_metadata"])
    assert first_normalized["evidence_id"] == "evidence-tsla-reddit-1"
    assert first_provenance["raw_identifier"] == "reddit-comment-tsla-1"
    assert first_provider_metadata["fixture"] is True

    extracted_records = _json_records(_read_json_object(audit_dir / "extracted-strategies.json"))
    assert len(extracted_records) == 6
    assert {record["ticker"] for record in extracted_records} == set(
        report.ticker_discovery.tickers
    )
    for record in extracted_records:
        evidence = record["evidence"]
        assert isinstance(evidence, list)
        assert evidence
        assert all(isinstance(reference, dict) for reference in evidence)

    analysis_records = _json_records(_read_json_object(audit_dir / "analysis-contexts.json"))
    assert len(analysis_records) == 6
    for record in analysis_records:
        assert record["ticker"] in report.ticker_discovery.tickers
        assert isinstance(record["technical"], dict)
        assert isinstance(record["fundamental"], dict)
        assert isinstance(record["sector"], dict)
        assert isinstance(record["macro"], dict)

    scoring_records = _json_records(_read_json_object(audit_dir / "scoring-inputs.json"))
    assert scoring_records == [
        {
            "scoring_input_id": "scoring-input-tsla",
            "candidate_id": "candidate-tsla-shares-swing",
            "ticker": "TSLA",
            "score": cast(JsonObject, report.trade_candidates[0].score.model_dump(mode="json")),
            "risk_plan": cast(
                JsonObject,
                report.trade_candidates[0].risk_plan.model_dump(mode="json"),
            ),
            "evidence_ids": ["evidence-tsla-reddit-1"],
            "confidence_inputs": {
                "reddit_strategy_confidence": 0.78,
                "technical_alignment": 0.72,
                "freshness_penalty": 0.05,
            },
        }
    ]


@pytest.mark.e2e
def test_cli_offline_run_writes_report_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nlp_stock_prediction",
            "run",
            "--date",
            "2026-05-11",
            "--output",
            str(output_dir),
            "--offline",
        ],
        cwd=PROJECT_ROOT,
        env=_module_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "report.md" in result.stdout
    assert "Wrote audit artifacts:" in result.stdout
    assert result.stderr == ""

    report_dir = output_dir / "2026-05-11"
    audit_dir = report_dir / "audit"
    markdown_path = report_dir / "report.md"
    json_path = report_dir / "report.json"
    manifest_path = audit_dir / "audit-manifest.json"

    assert markdown_path.exists()
    assert json_path.exists()
    assert audit_dir.is_dir()
    assert {path.name for path in audit_dir.iterdir()} == EXPECTED_AUDIT_FILES

    report = DailyReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert isinstance(report.audit_manifest, AuditManifest)
    assert AuditManifest.model_validate(_read_json_object(manifest_path)) == report.audit_manifest


@pytest.mark.e2e
def test_cli_run_requires_offline_until_live_orchestration_is_enabled(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nlp_stock_prediction",
            "run",
            "--date",
            "2026-05-11",
            "--output",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=_module_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
    assert "Live-provider report orchestration is not enabled yet" in result.stderr
    assert result.stdout == ""
    assert not output_dir.exists()
