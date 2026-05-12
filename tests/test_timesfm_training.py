from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import JsonObject, PriceBar
from nlp_stock_prediction.ml.timesfm import train as timesfm_train
from nlp_stock_prediction.ml.timesfm.artifacts import (
    TimesFmLossMetrics,
    TimesFmTrainingConfig,
    TimesFmTrainingDeviceMetadata,
    TimesFmTrainingResult,
    directory_sha256,
    file_sha256,
)
from nlp_stock_prediction.ml.timesfm.dataset import TimesFmDataset
from nlp_stock_prediction.ml.timesfm.train import (
    TimesFmTrainError,
    TimesFmTrainingRun,
    TimesFmTrainingSource,
)

RUN_DATE = date(2026, 5, 11)


class _FakeAdapter:
    def save_pretrained(self, path: str) -> None:
        adapter_dir = Path(path)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"r": 4, "target_modules": "all-linear"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-timesfm-adapter")


def _bars(count: int = 48) -> tuple[PriceBar, ...]:
    bars: list[PriceBar] = []
    previous_close = Decimal("100")
    for index in range(count):
        trend = Decimal(index) * Decimal("0.11")
        cycle = Decimal((index % 6) - 2) * Decimal("0.33")
        close = Decimal("100") + trend + cycle
        open_price = previous_close * Decimal("1.001")
        high = max(open_price, close) * Decimal("1.007")
        low = min(open_price, close) * Decimal("0.993")
        bars.append(
            PriceBar(
                ticker="TSLA",
                timestamp=RUN_DATE - timedelta(days=count - index),
                open=open_price.quantize(Decimal("0.0001")),
                high=high.quantize(Decimal("0.0001")),
                low=low.quantize(Decimal("0.0001")),
                close=close.quantize(Decimal("0.0001")),
                volume=900_000 + (index % 5) * 40_000,
            )
        )
        previous_close = close
    return tuple(bars)


def _write_csv(path: Path, bars: tuple[PriceBar, ...]) -> None:
    rows = ["timestamp,open,high,low,close,volume"]
    rows.extend(
        f"{bar.timestamp},{bar.open},{bar.high},{bar.low},{bar.close},{bar.volume}" for bar in bars
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fake_result(
    dataset: TimesFmDataset,
    config: TimesFmTrainingConfig,
    source: TimesFmTrainingSource,
) -> TimesFmTrainingResult:
    dataset_metadata = cast(JsonObject, dataset.model_dump(mode="json")["metadata"])
    return TimesFmTrainingResult(
        ticker=dataset.ticker,
        model_id=config.model_id,
        model_revision="fake-revision",
        source_kind=source.kind,
        source_path=source.path,
        source_sha256=source.sha256,
        csv_sha256=source.sha256 if source.kind == "csv" else None,
        dataset_hash=dataset.dataset_hash,
        trained_at=datetime(2026, 5, 12, tzinfo=UTC),
        config=config,
        device=TimesFmTrainingDeviceMetadata(
            requested_device=config.requested_device,
            selected_device="cpu",
            cuda_available=False,
            backend="fake-timesfm-lora",
            notes=("fake trainer",),
        ),
        dataset_metadata=dataset_metadata,
        split_metadata={
            "train_windows": len(dataset.train_windows),
            "validation_windows": len(dataset.validation_windows),
            "test_windows": len(dataset.test_windows),
            "context_length": dataset.context_length,
            "horizon_length": dataset.horizon_length,
        },
        train_metrics=TimesFmLossMetrics(
            steps=2,
            samples=4,
            losses=(0.42, 0.31),
            final_loss=0.31,
            mean_loss=0.365,
        ),
        validation_metrics=TimesFmLossMetrics(
            steps=1,
            samples=2,
            losses=(0.29,),
            final_loss=0.29,
            mean_loss=0.29,
        ),
        trainable_parameters=1_382_912,
        total_parameters=232_672_192,
        trainable_param_percent=0.5944,
        runtime_metadata={
            "torch_version": "fake-torch",
            "transformers_version": "fake-transformers",
            "peft_version": "fake-peft",
        },
    )


@pytest.mark.unit
def test_timesfm_training_command_writes_deterministic_artifacts_with_fake_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "tsla.csv"
    output_dir = tmp_path / "timesfm"
    _write_csv(csv_path, _bars())

    def fake_runner(
        dataset: TimesFmDataset,
        config: TimesFmTrainingConfig,
        source: TimesFmTrainingSource,
    ) -> TimesFmTrainingRun:
        assert dataset.ticker == "TSLA"
        assert config.requested_device == "cpu"
        assert config.max_steps == 2
        assert config.lora.r == 4
        assert source.kind == "csv"
        return TimesFmTrainingRun(
            result=_fake_result(dataset, config, source),
            adapter_model=_FakeAdapter(),
        )

    exit_code = timesfm_train.main(
        [
            "--csv",
            str(csv_path),
            "--ticker",
            "TSLA",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--context-length",
            "4",
            "--horizon-length",
            "2",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--as-of",
            RUN_DATE.isoformat(),
            "--max-latest-bar-age-days",
            "5",
        ],
        runner=fake_runner,
    )

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)
    metadata_path = Path(payload["metadata_path"])
    metrics_path = Path(payload["metrics_path"])
    adapter_dir = Path(payload["adapter_dir"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metadata["schema_version"] == "ml.timesfm.training_metadata.v1"
    assert metrics["schema_version"] == "ml.timesfm.training_metrics.v1"
    assert metadata["model_id"] == "google/timesfm-2.5-200m-transformers"
    assert metadata["model_revision"] == "fake-revision"
    assert metadata["source_kind"] == "csv"
    assert metadata["csv_sha256"] == payload["csv_sha256"]
    assert metadata["source_sha256"] == payload["source_sha256"]
    assert metadata["config"]["max_steps"] == 2
    assert metadata["config"]["lora"]["target_modules"] == "all-linear"
    assert metadata["split"]["train_windows"] > 0
    assert metadata["metrics"]["validation"]["final_loss"] == 0.29
    assert metadata["adapter"]["adapter_artifact_sha256"] == payload["artifact_sha256"]["adapter"]
    assert metadata["metrics_artifact_sha256"] == payload["artifact_sha256"]["metrics"]
    assert payload["artifact_sha256"]["adapter"] == directory_sha256(adapter_dir)
    assert payload["artifact_sha256"]["metrics"] == file_sha256(metrics_path)
    assert payload["artifact_sha256"]["metadata"] == file_sha256(metadata_path)
    assert (adapter_dir / "adapter_config.json").exists()
    assert (adapter_dir / "adapter_model.safetensors").exists()
    assert "not live trading" in payload["usage_limitations"]


@pytest.mark.unit
def test_timesfm_training_command_surfaces_dataset_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unreachable_runner(
        _dataset: TimesFmDataset,
        _config: TimesFmTrainingConfig,
        _source: TimesFmTrainingSource,
    ) -> TimesFmTrainingRun:
        raise AssertionError("runner should not be called")

    exit_code = timesfm_train.main(
        [
            "--synthetic",
            "--ticker",
            "TSLA",
            "--synthetic-bars",
            "20",
            "--output-dir",
            str(tmp_path / "timesfm"),
        ],
        runner=unreachable_runner,
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "TimesFM training failed:" in captured.err
    assert "at least 40 bars" in captured.err


@pytest.mark.unit
def test_timesfm_training_stack_missing_dependency_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTransformers:
        TimesFm2_5ModelForPrediction = object

    class _FakeTorch:
        pass

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "torch":
            return _FakeTorch()
        if name == "transformers":
            return _FakeTransformers()
        if name == "peft":
            raise ModuleNotFoundError(name="peft")
        raise AssertionError(f"unexpected import: {name} {package}")

    monkeypatch.setattr(
        "nlp_stock_prediction.ml.timesfm.train.importlib.import_module", fake_import
    )

    with pytest.raises(TimesFmTrainError, match=r"optional TimesFM dependency.*peft"):
        timesfm_train._load_timesfm_training_stack()


@pytest.mark.unit
def test_timesfm_loss_metrics_reject_inconsistent_step_counts() -> None:
    with pytest.raises(ValueError, match="steps must match recorded losses"):
        TimesFmLossMetrics(
            steps=2,
            samples=2,
            losses=(0.5,),
            final_loss=0.5,
            mean_loss=0.5,
        )
