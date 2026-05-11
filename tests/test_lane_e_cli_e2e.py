from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from nlp_stock_prediction.cli import CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
from nlp_stock_prediction.contracts import DailyReport, RiskProfile, RunConfig
from nlp_stock_prediction.pipeline import generate_daily_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_DIR) if not existing_pythonpath else f"{SRC_DIR}{os.pathsep}{existing_pythonpath}"
    )
    return env


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

    expected_audit_files = {
        "raw-snapshots.json",
        "normalized-evidence.json",
        "extracted-strategies.json",
        "analysis-contexts.json",
        "scoring-inputs.json",
        "final-reports.json",
        "audit-manifest.json",
    }
    assert expected_audit_files.issubset({path.name for path in audit_dir.iterdir()})

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
    assert report.audit_manifest is not None

    audit_manifest = json.loads((audit_dir / "audit-manifest.json").read_text(encoding="utf-8"))
    artifact_types = {artifact["artifact_type"] for artifact in audit_manifest["artifacts"]}
    assert {
        "raw_snapshot",
        "normalized_evidence",
        "extraction_output",
        "analysis_context",
        "scoring_input",
        "markdown_report",
        "json_report",
    }.issubset(artifact_types)

    normalized = json.loads((audit_dir / "normalized-evidence.json").read_text(encoding="utf-8"))
    assert normalized["records"][0]["evidence_id"] == "evidence-tsla-reddit-1"
    assert normalized["records"][0]["provenance"]["raw_identifier"] == "reddit-comment-tsla-1"
    assert normalized["records"][0]["provenance"]["provider_metadata"]["fixture"] is True


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
    assert result.stderr == ""
    assert (output_dir / "2026-05-11" / "report.md").exists()
    assert (output_dir / "2026-05-11" / "report.json").exists()


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
