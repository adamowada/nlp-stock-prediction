from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from nlp_stock_prediction.ml.timesfm.focused_hpo import (
    HpoTrial,
    TickerPolicy,
    _evaluation_artifact_mismatch,
    build_hpo_trials,
    choice_from_training_metadata,
    choose_best_training_run,
    ensure_symbol_data,
    minimum_rows_for_trial,
    select_trainable_trials,
    write_manifest,
    write_ohlcv_csv,
)


@pytest.mark.unit
def test_sndk_policy_clips_history_start_and_records_lineage_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "timestamp": "2025-02-24",
            "open": "10.000000",
            "high": "11.000000",
            "low": "9.000000",
            "close": "10.500000",
            "volume": "1000",
            "adjusted_close": "10.500000",
        }
        for _ in range(180)
    ]
    captured: dict[str, date] = {}

    def fake_fetch(symbol: str, start: date, end: date) -> list[dict[str, str]]:
        captured["start"] = start
        captured["end"] = end
        assert symbol == "SNDK"
        return rows

    monkeypatch.setattr(
        "nlp_stock_prediction.ml.timesfm.focused_hpo.fetch_adjusted_ohlcv",
        fake_fetch,
    )

    record = ensure_symbol_data(
        "SNDK",
        TickerPolicy(
            symbol="SNDK",
            history_start_override=date(2025, 2, 24),
            minimum_rows=180,
            notes=("current standalone history only",),
        ),
        data_dir=tmp_path,
        start=date(2016, 5, 11),
        as_of=date(2026, 5, 11),
        refresh=False,
        sleep_seconds=0.0,
    )

    assert captured["start"] == date(2025, 2, 24)
    assert record["rows"] == 180
    metadata = json.loads((tmp_path / "SNDK.metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_start"] == "2025-02-24"
    assert metadata["lineage_notes"] == ["current standalone history only"]


@pytest.mark.unit
def test_cached_data_policy_mismatch_is_refetched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_rows = [
        {
            "timestamp": "2015-01-01",
            "open": "1.000000",
            "high": "1.000000",
            "low": "1.000000",
            "close": "1.000000",
            "volume": "1000",
            "adjusted_close": "1.000000",
        }
    ]
    fresh_rows = [
        {
            "timestamp": "2025-02-24",
            "open": "10.000000",
            "high": "11.000000",
            "low": "9.000000",
            "close": "10.500000",
            "volume": "1000",
            "adjusted_close": "10.500000",
        }
        for _ in range(180)
    ]
    write_ohlcv_csv(tmp_path / "SNDK.csv", stale_rows)
    (tmp_path / "SNDK.metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "ml.timesfm.focused_ohlcv.v1",
                "symbol": "SNDK",
                "asset_type": "operating_company",
                "requested_start": "2016-05-13",
                "effective_start": "2016-05-13",
                "as_of": "2026-05-11",
                "source": "yahoo-chart",
                "source_url_template": "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
                "price_policy": "split_dividend_adjusted_ohlcv_from_yahoo_adjclose_ratio",
                "lineage_notes": ["stale policy"],
                "rows": 1,
                "first": "2015-01-01",
                "last": "2015-01-01",
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, date] = {}

    def fake_fetch(symbol: str, start: date, end: date) -> list[dict[str, str]]:
        captured["start"] = start
        captured["end"] = end
        assert symbol == "SNDK"
        return fresh_rows

    monkeypatch.setattr(
        "nlp_stock_prediction.ml.timesfm.focused_hpo.fetch_adjusted_ohlcv",
        fake_fetch,
    )

    record = ensure_symbol_data(
        "SNDK",
        TickerPolicy(
            symbol="SNDK",
            history_start_override=date(2025, 2, 24),
            minimum_rows=180,
            notes=("current standalone history only",),
        ),
        data_dir=tmp_path,
        start=date(2016, 5, 13),
        as_of=date(2026, 5, 11),
        refresh=False,
        sleep_seconds=0.0,
    )

    assert captured["start"] == date(2025, 2, 24)
    assert record["first"] == "2025-02-24"
    assert record["csv_sha256"]
    metadata = json.loads((tmp_path / "SNDK.metadata.json").read_text(encoding="utf-8"))
    assert metadata["effective_start"] == "2025-02-24"
    assert metadata["csv_sha256"] == record["csv_sha256"]


@pytest.mark.unit
def test_hpo_trials_start_with_strong_first_pass_recipe() -> None:
    trials = build_hpo_trials(
        context_lengths=(64, 128),
        horizon_lengths=(10, 16),
        max_steps_values=(500, 1000),
        batch_sizes=(4, 8),
        learning_rates=(1e-5, 3e-5),
        lora_ranks=(4, 8),
        lora_dropouts=(0.05, 0.10),
        lora_alpha_multiplier=2,
    )

    assert trials[0] == HpoTrial(
        context_length=128,
        horizon_length=16,
        max_steps=1000,
        batch_size=8,
        learning_rate=3e-5,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.10,
    )


@pytest.mark.unit
def test_capped_trial_selection_diversifies_core_hyperparameters() -> None:
    trials = build_hpo_trials(
        context_lengths=(64, 128, 256),
        horizon_lengths=(5, 10, 16, 20),
        max_steps_values=(500, 1000, 2000),
        batch_sizes=(4, 8, 16),
        learning_rates=(1e-5, 3e-5, 1e-4),
        lora_ranks=(4, 8, 16),
        lora_dropouts=(0.05, 0.10, 0.15),
        lora_alpha_multiplier=2,
    )

    selected = select_trainable_trials(trials, row_count=500, max_trials=24)

    assert len(selected) == 24
    assert selected[0] == HpoTrial(
        context_length=128,
        horizon_length=16,
        max_steps=1000,
        batch_size=8,
        learning_rate=3e-5,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.10,
    )
    assert len({trial.context_length for trial in selected}) > 1
    assert len({trial.horizon_length for trial in selected}) > 1
    assert len({trial.max_steps for trial in selected}) > 1
    assert len({trial.batch_size for trial in selected}) > 1


@pytest.mark.unit
def test_select_trainable_trials_filters_contexts_that_do_not_fit_history() -> None:
    short = HpoTrial(
        context_length=64,
        horizon_length=5,
        max_steps=500,
        batch_size=4,
        learning_rate=1e-5,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
    )
    long = HpoTrial(
        context_length=256,
        horizon_length=20,
        max_steps=500,
        batch_size=4,
        learning_rate=1e-5,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
    )

    selected = select_trainable_trials(
        (long, short),
        row_count=minimum_rows_for_trial(short),
        max_trials=0,
    )

    assert selected == (short,)


@pytest.mark.unit
def test_choose_best_training_run_uses_validation_loss(tmp_path: Path) -> None:
    trial = HpoTrial(
        context_length=128,
        horizon_length=16,
        max_steps=1000,
        batch_size=8,
        learning_rate=3e-5,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.10,
    )
    better_trial = HpoTrial(
        context_length=64,
        horizon_length=10,
        max_steps=500,
        batch_size=4,
        learning_rate=1e-5,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
    )
    args = _args()
    data_record = {"csv_sha256": "abc123"}
    weak_path = _write_training_metadata(
        tmp_path / "weak" / "training-metadata.json",
        trial=trial,
        csv_sha256="abc123",
        validation_mean_loss=0.40,
        validation_final_loss=0.39,
    )
    better_path = _write_training_metadata(
        tmp_path / "better" / "training-metadata.json",
        trial=better_trial,
        csv_sha256="abc123",
        validation_mean_loss=0.30,
        validation_final_loss=0.35,
    )

    weak = choice_from_training_metadata(
        "MU",
        trial,
        weak_path,
        weak_path.parent,
        data_record=data_record,
        args=args,
    )
    better = choice_from_training_metadata(
        "MU",
        better_trial,
        better_path,
        better_path.parent,
        data_record=data_record,
        args=args,
    )

    assert choose_best_training_run((weak, better)) == better


@pytest.mark.unit
def test_training_metadata_reuse_rejects_stale_csv_hash(tmp_path: Path) -> None:
    trial = HpoTrial(
        context_length=128,
        horizon_length=16,
        max_steps=1000,
        batch_size=8,
        learning_rate=3e-5,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.10,
    )
    metadata_path = _write_training_metadata(
        tmp_path / "training-metadata.json",
        trial=trial,
        csv_sha256="old-hash",
        validation_mean_loss=0.30,
        validation_final_loss=0.35,
    )

    with pytest.raises(ValueError, match="csv_sha256 expected"):
        choice_from_training_metadata(
            "MU",
            trial,
            metadata_path,
            tmp_path,
            data_record={"csv_sha256": "new-hash"},
            args=_args(),
        )


@pytest.mark.unit
def test_evaluation_reuse_rejects_stale_csv_hash(tmp_path: Path) -> None:
    trial = HpoTrial(
        context_length=128,
        horizon_length=16,
        max_steps=1000,
        batch_size=8,
        learning_rate=3e-5,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.10,
    )
    metadata_path = _write_training_metadata(
        tmp_path / "training-metadata.json",
        trial=trial,
        csv_sha256="new-hash",
        validation_mean_loss=0.30,
        validation_final_loss=0.35,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "schema_version": "ml.timesfm.evaluation.v1",
                "ticker": "MU",
                "model_id": "google/timesfm-2.5-200m-transformers",
                "evaluation_source_kind": "csv",
                "evaluation_source_sha256": "old-hash",
                "training_metadata_sha256": _sha256(metadata_path),
                "as_of": "2026-05-11",
                "config": {
                    "requested_device": "cuda",
                    "min_evaluation_windows": 3,
                    "suitability_max_latest_bar_age_days": 5,
                    "as_of": "2026-05-11",
                },
                "training_metadata": metadata,
            }
        ),
        encoding="utf-8",
    )

    mismatch = _evaluation_artifact_mismatch(
        "MU",
        trial,
        evaluation_path,
        tmp_path,
        data_record={"csv_sha256": "new-hash"},
        args=_args(),
    )

    assert mismatch == "evaluation_source_sha256 expected 'new-hash', found 'old-hash'"


@pytest.mark.unit
def test_write_manifest_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"

    write_manifest(path, {"records": []})

    assert json.loads(path.read_text(encoding="utf-8")) == {"records": []}


def _write_training_metadata(
    path: Path,
    *,
    trial: HpoTrial,
    csv_sha256: str,
    validation_mean_loss: float,
    validation_final_loss: float,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "ml.timesfm.training_metadata.v1",
                "ticker": "MU",
                "model_id": "google/timesfm-2.5-200m-transformers",
                "source_kind": "csv",
                "csv_sha256": csv_sha256,
                "dataset": {
                    "target_field": "adjusted_close",
                    "as_of": "2026-05-11",
                    "max_latest_bar_age_days": 5,
                },
                "split": {
                    "context_length": trial.context_length,
                    "horizon_length": trial.horizon_length,
                    "target_field": "adjusted_close",
                },
                "config": {
                    "requested_device": "cuda",
                    "epochs": 20,
                    "max_steps": trial.max_steps,
                    "batch_size": trial.batch_size,
                    "learning_rate": trial.learning_rate,
                    "validation_batches": 9999,
                    "seed": 42,
                    "gradient_clip_norm": 1.0,
                    "lora": {
                        "r": trial.lora_r,
                        "lora_alpha": trial.lora_alpha,
                        "lora_dropout": trial.lora_dropout,
                    },
                },
                "metrics": {
                    "validation": {
                        "mean_loss": validation_mean_loss,
                        "final_loss": validation_final_loss,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model_id="google/timesfm-2.5-200m-transformers",
        device="cuda",
        epochs=20,
        validation_batches=9999,
        max_latest_bar_age_days=5,
        min_evaluation_windows=3,
        suitability_max_latest_bar_age_days=5,
        as_of="2026-05-11",
        seed=42,
        gradient_clip_norm=1.0,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
