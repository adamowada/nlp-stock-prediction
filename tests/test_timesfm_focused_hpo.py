from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from nlp_stock_prediction.ml.timesfm.focused_hpo import (
    HpoTrial,
    TickerPolicy,
    build_hpo_trials,
    choose_best_evaluation,
    ensure_symbol_data,
    minimum_rows_for_trial,
    select_trainable_trials,
    write_manifest,
)
from nlp_stock_prediction.ml.timesfm.focused_hpo import (
    choice_from_evaluation as load_choice_from_evaluation,
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
def test_choose_best_evaluation_prefers_suitable_then_rmse_ratio(tmp_path: Path) -> None:
    weak_path = _write_evaluation(
        tmp_path / "weak.json",
        status="weak",
        suitable=False,
        accuracy=0.70,
        rmse=8.0,
        baseline_rmse=10.0,
    )
    suitable_path = _write_evaluation(
        tmp_path / "suitable.json",
        status="suitable",
        suitable=True,
        accuracy=0.55,
        rmse=9.0,
        baseline_rmse=10.0,
    )
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

    weak = load_choice_from_evaluation("MU", trial, weak_path, tmp_path / "weak")
    suitable = load_choice_from_evaluation("MU", trial, suitable_path, tmp_path / "suitable")

    assert choose_best_evaluation((weak, suitable)) == suitable


@pytest.mark.unit
def test_write_manifest_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"

    write_manifest(path, {"records": []})

    assert json.loads(path.read_text(encoding="utf-8")) == {"records": []}


def _write_evaluation(
    path: Path,
    *,
    status: str,
    suitable: bool,
    accuracy: float,
    rmse: float,
    baseline_rmse: float,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "suitable_for_scoring": suitable,
                "metrics": {
                    "directional_accuracy": accuracy,
                    "rmse": rmse,
                },
                "baselines": [
                    {
                        "name": "last_close_persistence",
                        "metrics": {"rmse": baseline_rmse},
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path
