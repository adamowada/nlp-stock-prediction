from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_stock_prediction.ml.timesfm import smoke


def test_timesfm_package_imports_without_optional_dependencies() -> None:
    assert smoke.DEFAULT_MODEL_ID == "google/timesfm-2.5-200m-transformers"


def test_missing_optional_dependency_has_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str) -> object:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr("importlib.import_module", fake_import)

    with pytest.raises(smoke.TimesFmSmokeError, match="optional TimesFM dependency"):
        smoke._load_timesfm_stack()


def test_cuda_requested_without_available_gpu_has_actionable_message() -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    with pytest.raises(smoke.TimesFmSmokeError, match="CUDA was requested"):
        smoke._select_device(_FakeTorch(), "cuda")


def test_smoke_cli_writes_json_artifact_with_injected_runner(tmp_path: Path) -> None:
    output_path = tmp_path / "smoke-result.json"

    def fake_runner(config: smoke.TimesFmSmokeConfig) -> smoke.TimesFmSmokeResult:
        assert config.requested_device == "cuda"
        assert config.steps == 2
        assert config.output_path == output_path
        return smoke.TimesFmSmokeResult(
            status="passed",
            model_id=config.model_id,
            platform="Windows-11",
            python="3.12.13",
            torch="2.11.0+cu128",
            transformers="5.8.0",
            peft="0.19.1",
            cuda_available=True,
            cuda_runtime="12.8",
            requested_device="cuda",
            selected_device="cuda",
            device_name="NVIDIA GeForce RTX 3090",
            device_capability=[8, 6],
            total_vram_gb=24.0,
            context_len=config.context_len,
            horizon_len=config.horizon_len,
            batch_size=config.batch_size,
            steps=config.steps,
            forecast_mean_shape=[1, 128],
            forecast_full_shape=[1, 128, 10],
            losses=[0.2, 0.1],
            elapsed_seconds=0.5,
            trainable_params=1_382_912,
            total_params=232_672_192,
            trainable_param_percent=0.5944,
            memory_allocated_gb=0.8,
            max_memory_allocated_gb=1.0,
            limitation="synthetic smoke only",
        )

    exit_code = smoke.main(
        ["--device", "cuda", "--steps", "2", "--output", str(output_path)],
        runner=fake_runner,
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["device_name"] == "NVIDIA GeForce RTX 3090"
    assert payload["forecast_mean_shape"] == [1, 128]
    assert payload["max_memory_allocated_gb"] == 1.0
