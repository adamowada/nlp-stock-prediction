from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import JsonObject, PriceBar
from nlp_stock_prediction.ml.timesfm import evaluate as timesfm_evaluate
from nlp_stock_prediction.ml.timesfm.artifacts import directory_sha256
from nlp_stock_prediction.ml.timesfm.dataset import (
    TimesFmDataset,
    TimesFmDatasetConfig,
    TimesFmWindow,
    build_timesfm_dataset,
)
from nlp_stock_prediction.ml.timesfm.evaluate import (
    TimesFmEvaluateError,
    TimesFmEvaluationConfig,
    TimesFmEvaluationDataSource,
    TimesFmEvaluationModelSource,
    TimesFmWindowPrediction,
    evaluate_timesfm_dataset,
    load_timesfm_evaluation_model_source,
    write_timesfm_evaluation_artifact,
)

RUN_DATE = date(2026, 5, 11)


def _bars(count: int = 48) -> tuple[PriceBar, ...]:
    bars: list[PriceBar] = []
    previous_close = Decimal("100")
    for index in range(count):
        close = Decimal("100") + Decimal(index) * Decimal("0.30")
        open_price = previous_close * Decimal("1.001")
        high = max(open_price, close) * Decimal("1.006")
        low = min(open_price, close) * Decimal("0.994")
        bars.append(
            PriceBar(
                ticker="TSLA",
                timestamp=RUN_DATE - timedelta(days=count - index),
                open=open_price.quantize(Decimal("0.0001")),
                high=high.quantize(Decimal("0.0001")),
                low=low.quantize(Decimal("0.0001")),
                close=close.quantize(Decimal("0.0001")),
                volume=900_000 + (index % 5) * 50_000,
            )
        )
        previous_close = close
    return tuple(bars)


def _dataset() -> TimesFmDataset:
    return build_timesfm_dataset(
        "TSLA",
        _bars(),
        config=TimesFmDatasetConfig(
            context_length=4,
            horizon_length=2,
            train_fraction=0.5,
            validation_fraction=0.25,
        ),
    )


def _data_source() -> TimesFmEvaluationDataSource:
    return TimesFmEvaluationDataSource(
        kind="synthetic",
        sha256="c" * 64,
    )


def _model_source(tmp_path: Path) -> TimesFmEvaluationModelSource:
    model_dir = tmp_path / "timesfm"
    adapter_dir = model_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 4}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-adapter")
    metadata = {
        "schema_version": "ml.timesfm.training_metadata.v1",
        "model_id": "google/timesfm-2.5-200m-transformers",
        "model_revision": "fake-revision",
        "dataset_hash": "b" * 64,
        "adapter": {
            "adapter_dir_name": "adapter",
            "adapter_artifact_sha256": directory_sha256(adapter_dir),
        },
        "usage_limitations": "not investment advice",
    }
    metadata_path = model_dir / "training-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return load_timesfm_evaluation_model_source(model_dir)


def _perfect_predictor(window: TimesFmWindow) -> TimesFmWindowPrediction:
    full_predictions = tuple((value * 0.99, value, value * 1.01) for value in window.future_values)
    return TimesFmWindowPrediction(
        point_forecast=window.future_values,
        full_predictions=full_predictions,
    )


def _bad_predictor(window: TimesFmWindow) -> TimesFmWindowPrediction:
    context_value = window.context_values[-1]
    point = tuple(context_value * 0.80 for _ in window.future_values)
    full_predictions = tuple((value * 0.99, value, value * 1.01) for value in point)
    return TimesFmWindowPrediction(
        point_forecast=point,
        full_predictions=full_predictions,
    )


@pytest.mark.unit
def test_timesfm_evaluation_marks_model_suitable_when_it_beats_baselines(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    artifact = evaluate_timesfm_dataset(
        dataset,
        _model_source(tmp_path),
        _data_source(),
        TimesFmEvaluationConfig(
            min_evaluation_windows=1,
            min_directional_accuracy=0.0,
            max_rmse_ratio_vs_best_baseline=1.0,
        ),
        predictor=_perfect_predictor,
    )

    assert artifact.status == "suitable"
    assert artifact.suitable_for_scoring is True
    assert artifact.metrics.rmse == 0.0
    assert artifact.metrics.directional_accuracy == 1.0
    assert artifact.metrics.interval_coverage == 1.0
    assert len(artifact.model_hash) == 64
    assert artifact.baselines
    assert all(baseline.metrics.rmse > artifact.metrics.rmse for baseline in artifact.baselines)


@pytest.mark.unit
def test_timesfm_evaluation_marks_underperforming_model_weak(tmp_path: Path) -> None:
    dataset = _dataset()
    artifact = evaluate_timesfm_dataset(
        dataset,
        _model_source(tmp_path),
        _data_source(),
        TimesFmEvaluationConfig(
            min_evaluation_windows=1,
            min_directional_accuracy=0.0,
            max_rmse_ratio_vs_best_baseline=1.0,
        ),
        predictor=_bad_predictor,
    )

    assert artifact.status == "weak"
    assert artifact.suitable_for_scoring is False
    assert "underperforms_best_rmse_baseline" in artifact.suitability_reasons


@pytest.mark.unit
def test_timesfm_evaluation_marks_stale_evaluation_data_weak(tmp_path: Path) -> None:
    dataset = _dataset()
    artifact = evaluate_timesfm_dataset(
        dataset,
        _model_source(tmp_path),
        _data_source(),
        TimesFmEvaluationConfig(
            min_evaluation_windows=1,
            min_directional_accuracy=0.0,
            max_rmse_ratio_vs_best_baseline=1.0,
            as_of=RUN_DATE + timedelta(days=10),
            suitability_max_latest_bar_age_days=1,
        ),
        predictor=_perfect_predictor,
    )

    assert artifact.status == "weak"
    assert artifact.suitable_for_scoring is False
    assert "stale_evaluation_data" in artifact.suitability_reasons


@pytest.mark.unit
def test_timesfm_evaluation_writes_artifact_json(tmp_path: Path) -> None:
    dataset = _dataset()
    artifact = evaluate_timesfm_dataset(
        dataset,
        _model_source(tmp_path),
        _data_source(),
        TimesFmEvaluationConfig(min_evaluation_windows=1, min_directional_accuracy=0.0),
        predictor=_perfect_predictor,
    )
    output_path = tmp_path / "evaluation.json"

    write_timesfm_evaluation_artifact(artifact, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "ml.timesfm.evaluation.v1"
    assert payload["dataset_hash"] == dataset.dataset_hash
    assert payload["status"] == "suitable"
    assert payload["suitable_for_scoring"] is True


@pytest.mark.unit
def test_timesfm_evaluation_command_writes_artifact_with_fake_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "tsla.csv"
    output_path = tmp_path / "evaluation.json"
    _write_csv(csv_path, _bars())
    model_source = _model_source(tmp_path)

    def fake_runner(
        dataset: TimesFmDataset,
        loaded_model_source: TimesFmEvaluationModelSource,
        data_source: TimesFmEvaluationDataSource,
        config: TimesFmEvaluationConfig,
    ) -> timesfm_evaluate.TimesFmEvaluationArtifact:
        assert loaded_model_source.adapter_sha256 == model_source.adapter_sha256
        assert data_source.kind == "csv"
        assert config.requested_device == "cpu"
        return evaluate_timesfm_dataset(
            dataset,
            loaded_model_source,
            data_source,
            config,
            predictor=_perfect_predictor,
        )

    exit_code = timesfm_evaluate.main(
        [
            "--csv",
            str(csv_path),
            "--ticker",
            "TSLA",
            "--model-dir",
            str(model_source.model_dir),
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--context-length",
            "4",
            "--horizon-length",
            "2",
            "--min-evaluation-windows",
            "1",
            "--min-directional-accuracy",
            "0",
        ],
        runner=fake_runner,
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0, captured.err
    assert payload["status"] == "suitable"
    assert payload["suitable_for_scoring"] is True
    assert artifact_payload["evaluation_source_kind"] == "csv"
    assert artifact_payload["adapter_sha256"] == model_source.adapter_sha256


@pytest.mark.unit
def test_timesfm_evaluation_model_source_warns_on_adapter_hash_mismatch(tmp_path: Path) -> None:
    model_source = _model_source(tmp_path)
    metadata_path = model_source.model_dir / "training-metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    adapter_section = cast(JsonObject, payload["adapter"])
    adapter_section["adapter_artifact_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    mismatched = load_timesfm_evaluation_model_source(model_source.model_dir)

    assert mismatched.warning_ids == ("timesfm_adapter_hash_mismatch",)
    artifact = evaluate_timesfm_dataset(
        _dataset(),
        mismatched,
        _data_source(),
        TimesFmEvaluationConfig(
            min_evaluation_windows=1,
            min_directional_accuracy=0.0,
            max_rmse_ratio_vs_best_baseline=1.0,
        ),
        predictor=_perfect_predictor,
    )
    assert artifact.status == "weak"
    assert artifact.suitable_for_scoring is False
    assert "timesfm_adapter_hash_mismatch" in artifact.suitability_reasons


@pytest.mark.unit
def test_timesfm_evaluation_missing_metadata_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(TimesFmEvaluateError, match="missing TimesFM training metadata"):
        load_timesfm_evaluation_model_source(tmp_path / "missing")


def _write_csv(path: Path, bars: tuple[PriceBar, ...]) -> None:
    rows = ["timestamp,open,high,low,close,volume"]
    rows.extend(
        f"{bar.timestamp},{bar.open},{bar.high},{bar.low},{bar.close},{bar.volume}" for bar in bars
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
