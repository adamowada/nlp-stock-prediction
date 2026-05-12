from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.analysis.ml_signal import (
    build_timesfm_ml_signal,
    load_timesfm_ml_signal_attachment,
)
from nlp_stock_prediction.contracts import (
    AuditManifest,
    DailyReport,
    FreshnessStatus,
    JsonObject,
    RiskProfile,
    RunConfig,
)
from nlp_stock_prediction.ml.timesfm.evaluate import (
    TimesFmBaselineEvaluation,
    TimesFmEvaluationArtifact,
    TimesFmEvaluationConfig,
    TimesFmEvaluationMetrics,
    TimesFmEvaluationRecord,
    TimesFmEvaluationStatus,
    write_timesfm_evaluation_artifact,
)
from nlp_stock_prediction.pipeline import generate_daily_report
from nlp_stock_prediction.reporting.audit import json_payload_sha256

RUN_DATE = date(2026, 5, 11)
GENERATED_AT = datetime(2026, 5, 11, 20, 0, tzinfo=UTC)


@pytest.mark.unit
def test_timesfm_evaluation_artifact_maps_to_report_safe_ml_signal(tmp_path: Path) -> None:
    artifact_path = tmp_path / "evaluation.json"
    write_timesfm_evaluation_artifact(_evaluation_artifact(), artifact_path)

    attachment = load_timesfm_ml_signal_attachment(artifact_path)

    assert attachment.ticker == "TSLA"
    assert attachment.signal.status == "usable"
    assert attachment.signal.signal.value == "supports"
    assert attachment.signal.prediction_horizon_sessions == 2
    assert attachment.signal.expected_return == 0.08
    assert attachment.signal.forecast_interval_width == 0.08
    assert attachment.signal.source_artifact_sha256 == attachment.artifact_sha256
    assert attachment.signal.metadata["model_kind"] == "timesfm_2_5_lora_evaluation"
    assert attachment.artifact_payload["schema_version"] == "ml.timesfm.evaluation.v1"


@pytest.mark.e2e
def test_offline_report_attaches_explicit_timesfm_artifact_to_markdown_json_and_audit(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "evaluation.json"
    write_timesfm_evaluation_artifact(_evaluation_artifact(), artifact_path)
    output_dir = tmp_path / "reports"

    bundle = generate_daily_report(
        RunConfig(
            run_date=RUN_DATE,
            output_dir=output_dir,
            risk_profile=RiskProfile.EXPLORATORY,
            offline=True,
            ml_artifact=artifact_path,
        )
    )

    report = DailyReport.model_validate_json(bundle.json_path.read_text(encoding="utf-8"))
    markdown = bundle.markdown_path.read_text(encoding="utf-8")
    tsla_section = next(section for section in report.ticker_sections if section.ticker == "TSLA")
    assert tsla_section.technical_analysis is not None
    assert tsla_section.technical_analysis.ml_signal is not None
    ml_signal = tsla_section.technical_analysis.ml_signal
    assert ml_signal.metadata["model_kind"] == "timesfm_2_5_lora_evaluation"
    assert ml_signal.model_hash == "a" * 64
    assert ml_signal.source_artifact_id == "ml-timesfm-evaluation"
    assert tsla_section.data_quality["ml_signal"] == "timesfm_evaluation_artifact"
    assert "TimesFM signal: supports" in markdown
    assert f"model hash `{'a' * 64}`" in markdown
    assert "TimesFM limitations:" in markdown

    manifest = AuditManifest.model_validate(
        json.loads(bundle.audit_manifest_path.read_text(encoding="utf-8"))
    )
    assert manifest == report.audit_manifest
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in manifest.artifacts}
    assert artifacts_by_id["ml-artifacts"].artifact_type == "ml_artifact"
    ml_payload = cast(JsonObject, json.loads((bundle.audit_dir / "ml-artifacts.json").read_text()))
    assert artifacts_by_id["ml-artifacts"].sha256 == json_payload_sha256(ml_payload)
    assert artifacts_by_id["ml-artifacts"].record_count == 1
    ml_records = cast(list[JsonObject], ml_payload["records"])
    assert ml_records[0]["sha256"] == ml_signal.source_artifact_sha256
    assert cast(JsonObject, ml_records[0]["payload"])["model_hash"] == "a" * 64

    analysis_payload = cast(
        JsonObject,
        json.loads((bundle.audit_dir / "analysis-contexts.json").read_text(encoding="utf-8")),
    )
    analysis_records = cast(list[JsonObject], analysis_payload["records"])
    tsla_context = next(record for record in analysis_records if record["ticker"] == "TSLA")
    technical_context = cast(JsonObject, tsla_context["technical"])
    audit_ml_signal = cast(JsonObject, technical_context["ml_signal"])
    assert audit_ml_signal["source_artifact_id"] == "ml-timesfm-evaluation"

    scoring_payload = cast(
        JsonObject,
        json.loads((bundle.audit_dir / "scoring-inputs.json").read_text(encoding="utf-8")),
    )
    scoring_record = cast(list[JsonObject], scoring_payload["records"])[0]
    assert "timesfm" not in cast(JsonObject, scoring_record["confidence_inputs"])


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("status", "reasons", "expected_signal_status", "expected_freshness"),
    [
        ("weak", ("underperforms_best_rmse_baseline",), "weak", FreshnessStatus.FRESH),
        ("weak", ("stale_evaluation_data",), "stale", FreshnessStatus.STALE),
        (
            "unavailable",
            ("timesfm_evaluation_unavailable",),
            "unavailable",
            FreshnessStatus.UNKNOWN,
        ),
    ],
)
def test_reports_render_cleanly_for_non_usable_timesfm_artifacts(
    tmp_path: Path,
    status: TimesFmEvaluationStatus,
    reasons: tuple[str, ...],
    expected_signal_status: str,
    expected_freshness: FreshnessStatus,
) -> None:
    artifact_path = tmp_path / "evaluation.json"
    write_timesfm_evaluation_artifact(
        _evaluation_artifact(
            status=status,
            suitable_for_scoring=False,
            suitability_reasons=reasons,
            include_record=status != "unavailable",
        ),
        artifact_path,
    )

    signal = build_timesfm_ml_signal(
        TimesFmEvaluationArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    )
    assert signal.status == expected_signal_status
    assert signal.freshness_status == expected_freshness

    bundle = generate_daily_report(
        RunConfig(
            run_date=RUN_DATE,
            output_dir=tmp_path / "reports",
            risk_profile=RiskProfile.EXPLORATORY,
            offline=True,
            ml_artifact=artifact_path,
        )
    )
    report = DailyReport.model_validate_json(bundle.json_path.read_text(encoding="utf-8"))
    tsla_section = next(section for section in report.ticker_sections if section.ticker == "TSLA")
    assert tsla_section.technical_analysis is not None
    assert tsla_section.technical_analysis.ml_signal is not None
    assert tsla_section.technical_analysis.ml_signal.status == expected_signal_status
    markdown = bundle.markdown_path.read_text(encoding="utf-8")
    assert "TimesFM signal:" in markdown
    assert f"status {expected_signal_status}" in markdown


@pytest.mark.e2e
def test_report_attachment_rejects_unmatched_timesfm_ticker(tmp_path: Path) -> None:
    artifact_path = tmp_path / "evaluation.json"
    write_timesfm_evaluation_artifact(_evaluation_artifact(ticker="ZZZZ"), artifact_path)

    with pytest.raises(ValueError, match="did not match a report ticker"):
        generate_daily_report(
            RunConfig(
                run_date=RUN_DATE,
                output_dir=tmp_path / "reports",
                risk_profile=RiskProfile.EXPLORATORY,
                offline=True,
                ml_artifact=artifact_path,
            )
        )


def _evaluation_artifact(
    *,
    ticker: str = "TSLA",
    status: TimesFmEvaluationStatus = "suitable",
    suitable_for_scoring: bool = True,
    suitability_reasons: tuple[str, ...] = (),
    include_record: bool = True,
) -> TimesFmEvaluationArtifact:
    records = (
        (
            TimesFmEvaluationRecord(
                context_end=RUN_DATE - timedelta(days=2),
                horizon_end=RUN_DATE,
                actual_final_value=106.0,
                timesfm_final_value=108.0,
                persistence_final_value=100.0,
                recent_mean_return_final_value=104.0,
                actual_return=0.06,
                timesfm_return=0.08,
                persistence_return=0.0,
                recent_mean_return=0.04,
                interval_lower=102.0,
                interval_upper=110.0,
                interval_covered=True,
            ),
        )
        if include_record
        else ()
    )
    sample_count = len(records)
    metrics = TimesFmEvaluationMetrics(
        sample_count=sample_count,
        mae=2.0 if sample_count else 0.0,
        rmse=2.0 if sample_count else 0.0,
        directional_accuracy=1.0 if sample_count else 0.0,
        mean_return_error=0.02 if sample_count else 0.0,
        interval_coverage=1.0 if sample_count else None,
        mean_interval_width=0.08 if sample_count else None,
        calibration_proxy=0.8 if sample_count else None,
    )
    baseline_metrics = TimesFmEvaluationMetrics(
        sample_count=sample_count,
        mae=6.0 if sample_count else 0.0,
        rmse=6.0 if sample_count else 0.0,
        directional_accuracy=0.0,
        mean_return_error=0.06 if sample_count else 0.0,
    )
    return TimesFmEvaluationArtifact(
        status=status,
        suitable_for_scoring=suitable_for_scoring,
        suitability_reasons=suitability_reasons,
        ticker=ticker,
        model_id="google/timesfm-2.5-200m-transformers",
        model_revision="fake-revision",
        model_hash="a" * 64,
        adapter_sha256="b" * 64,
        training_metadata_sha256="c" * 64,
        dataset_hash="d" * 64,
        evaluation_source_kind="synthetic",
        evaluation_source_sha256="e" * 64,
        evaluated_at=GENERATED_AT,
        latest_bar_timestamp=RUN_DATE,
        as_of=RUN_DATE,
        config=TimesFmEvaluationConfig(min_evaluation_windows=1, min_directional_accuracy=0.0),
        metrics=metrics,
        baselines=(
            TimesFmBaselineEvaluation(
                name="last_close_persistence",
                metrics=baseline_metrics,
                mae_delta_vs_timesfm=4.0 if sample_count else 0.0,
                rmse_delta_vs_timesfm=4.0 if sample_count else 0.0,
                directional_accuracy_delta_vs_timesfm=-1.0 if sample_count else 0.0,
            ),
        ),
        records=records,
        training_metadata={
            "schema_version": "ml.timesfm.training_metadata.v1",
            "split": {"horizon_length": 2},
            "usage_limitations": "fixture",
        },
    )
