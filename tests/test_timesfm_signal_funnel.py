from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from nlp_stock_prediction.ml.timesfm import signal_funnel
from nlp_stock_prediction.ml.timesfm.focused_hpo import write_ohlcv_csv


@pytest.mark.unit
def test_signal_funnel_stage0_dry_run_writes_manifest_and_leaderboard(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    data_dir = tmp_path / "data"

    exit_code = signal_funnel.main(
        [
            "--dry-run",
            "--symbols",
            "MU",
            "--as-of",
            "2026-05-11",
            "--data-dir",
            str(data_dir),
            "--output-root",
            str(output_root),
            "--run-id",
            "stage0-dry-run",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert not data_dir.exists()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    leaderboard = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))
    csv_rows = _read_csv_rows(output_root / "leaderboard.csv")

    assert manifest["schema_version"] == "ml.timesfm.signal_funnel_manifest.v1"
    assert manifest["run_id"] == "stage0-dry-run"
    assert manifest["implemented_stages"] == [
        "data_check",
        "baseline_screen",
        "raw_timesfm_screen",
    ]
    assert manifest["skipped_stages"][0]["reason"] == "stage_not_implemented"
    assert len(leaderboard["rows"]) == 3
    assert leaderboard["rows"][0]["symbol"] == "MU"
    assert leaderboard["rows"][0]["status"] == "passed"
    assert leaderboard["rows"][0]["decision"] == "continue"
    assert leaderboard["rows"][0]["selected_for_next_stage"] is True
    assert leaderboard["rows"][0]["dataset_hash"] is None
    assert leaderboard["rows"][1]["stage"] == "baseline_screen"
    assert leaderboard["rows"][1]["status"] == "skipped"
    assert leaderboard["rows"][1]["kill_reason"] == "dry_run_no_csv_loaded"
    assert leaderboard["rows"][2]["stage"] == "raw_timesfm_screen"
    assert leaderboard["rows"][2]["status"] == "skipped"
    assert leaderboard["rows"][2]["kill_reason"] == "dry_run_no_model_loaded"
    assert csv_rows[0]["symbol"] == "MU"
    assert csv_rows[0]["status"] == "passed"


@pytest.mark.unit
def test_signal_funnel_stage0_marks_limited_test_windows_research_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "SNDK.csv"
    rows = _ohlcv_rows("2026-05-11", count=315)
    write_ohlcv_csv(csv_path, rows)

    monkeypatch.setattr(
        signal_funnel,
        "ensure_symbol_data",
        lambda *_args, **_kwargs: _data_record(csv_path, rows),
    )

    output_root = tmp_path / "out"
    exit_code = signal_funnel.main(
        [
            "--symbols",
            "SNDK",
            "--as-of",
            "2026-05-11",
            "--data-dir",
            str(tmp_path / "data"),
            "--output-root",
            str(output_root),
            "--run-id",
            "stage0-limited",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--stop-after",
            "data_check",
        ]
    )

    leaderboard = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))
    row = leaderboard["rows"][0]

    assert exit_code == 0
    assert row["symbol"] == "SNDK"
    assert row["status"] == "research_only"
    assert row["decision"] == "audit_only"
    assert row["test_windows"] < 3
    assert row["selected_for_next_stage"] is False
    assert row["kill_reason"].startswith("limited_test_windows:")
    assert row["dataset_hash"]


@pytest.mark.unit
def test_signal_funnel_stage0_records_dataset_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "MU.csv"
    rows = _ohlcv_rows("2026-05-11", count=40)
    write_ohlcv_csv(csv_path, rows)

    monkeypatch.setattr(
        signal_funnel,
        "ensure_symbol_data",
        lambda *_args, **_kwargs: _data_record(csv_path, rows),
    )

    output_root = tmp_path / "out"
    exit_code = signal_funnel.main(
        [
            "--symbols",
            "MU",
            "--as-of",
            "2026-05-11",
            "--data-dir",
            str(tmp_path / "data"),
            "--output-root",
            str(output_root),
            "--run-id",
            "stage0-failure",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--stop-after",
            "data_check",
        ]
    )

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    row = manifest["records"][0]

    assert exit_code == 1
    assert row["status"] == "failed"
    assert row["decision"] == "stop"
    assert row["bar_count"] == 40
    assert row["latest_bar"] == "2026-05-11"
    assert "insufficient history" in row["kill_reason"]
    assert row["selected_for_next_stage"] is False


@pytest.mark.unit
def test_signal_funnel_stage1_writes_baseline_and_technical_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "MU.csv"
    rows = _ohlcv_rows("2026-05-11", count=420)
    write_ohlcv_csv(csv_path, rows)

    monkeypatch.setattr(
        signal_funnel,
        "ensure_symbol_data",
        lambda *_args, **_kwargs: _data_record(csv_path, rows),
    )

    output_root = tmp_path / "out"
    exit_code = signal_funnel.main(
        [
            "--symbols",
            "MU",
            "--as-of",
            "2026-05-11",
            "--data-dir",
            str(data_dir),
            "--output-root",
            str(output_root),
            "--run-id",
            "stage1-baselines",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--stop-after",
            "baseline_screen",
        ]
    )

    assert exit_code == 0
    leaderboard = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))
    records = leaderboard["rows"]
    methods = {row["method"]: row for row in records}

    assert len(records) == 4
    assert records[0]["stage"] == "data_check"
    assert {
        "last_close_persistence",
        "recent_mean_return",
        "deterministic_technical_analysis",
    } <= set(methods)
    last_close = methods["last_close_persistence"]
    recent_mean = methods["recent_mean_return"]
    technical = methods["deterministic_technical_analysis"]

    assert last_close["stage"] == "baseline_screen"
    assert last_close["status"] == "passed"
    assert last_close["decision"] == "continue"
    assert last_close["max_windows"] == 16
    assert last_close["rmse"] >= 0
    assert last_close["best_baseline_rmse"] >= 0
    assert last_close["directional_accuracy"] >= 0
    assert last_close["best_baseline_directional_accuracy"] >= 0
    assert "sample_count=16" in last_close["notes"]
    assert recent_mean["rmse_ratio_vs_best_baseline"] is not None
    assert technical["rmse"] is None
    assert "signal=" in technical["notes"]
    assert "trend=" in technical["notes"]
    assert technical["selected_for_next_stage"] is True


@pytest.mark.unit
def test_signal_funnel_stage2_promotes_raw_timesfm_when_it_beats_baselines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "MU.csv"
    rows = _ohlcv_rows("2026-05-11", count=420)
    write_ohlcv_csv(csv_path, rows)

    monkeypatch.setattr(
        signal_funnel,
        "ensure_symbol_data",
        lambda *_args, **_kwargs: _data_record(csv_path, rows),
    )

    def raw_predictor(
        _dataset: Any,
        windows: Any,
        args: Any,
    ) -> signal_funnel._RawTimesFmPredictionBatch:
        return signal_funnel._RawTimesFmPredictionBatch(
            model_id="fake-raw-timesfm",
            model_revision="fake-revision",
            point_forecasts=tuple(tuple(window.future_values) for window in windows),
            runtime_metadata={"backend": "fake_raw_predictor", "device": args.device},
        )

    output_root = tmp_path / "out"
    exit_code = signal_funnel.main(
        [
            "--symbols",
            "MU",
            "--as-of",
            "2026-05-11",
            "--data-dir",
            str(data_dir),
            "--output-root",
            str(output_root),
            "--run-id",
            "stage2-promote",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
        ],
        raw_predictor=raw_predictor,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    methods = {row["method"]: row for row in records}
    raw_row = methods["raw_timesfm_base"]

    assert len(records) == 5
    assert raw_row["stage"] == "raw_timesfm_screen"
    assert raw_row["status"] == "passed"
    assert raw_row["decision"] == "run_adapter_smoke"
    assert raw_row["selected_for_next_stage"] is True
    assert raw_row["model_id"] == "fake-raw-timesfm"
    assert raw_row["model_revision"] == "fake-revision"
    assert raw_row["rmse"] == 0.0
    assert raw_row["raw_timesfm_rmse"] == 0.0
    assert raw_row["rmse_ratio_vs_best_baseline"] == 0.0
    assert "sample_count=16" in raw_row["notes"]

    artifact_path = Path(raw_row["evaluation_artifact"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "ml.timesfm.raw_evaluation.v1"
    assert artifact["status"] == "passed"
    assert artifact["decision"] == "run_adapter_smoke"
    assert artifact["metrics"]["sample_count"] == 16
    assert len(artifact["records"]) == 16


@pytest.mark.unit
def test_signal_funnel_stage2_kills_raw_timesfm_when_materially_worse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "MU.csv"
    rows = _ohlcv_rows("2026-05-11", count=420)
    write_ohlcv_csv(csv_path, rows)

    monkeypatch.setattr(
        signal_funnel,
        "ensure_symbol_data",
        lambda *_args, **_kwargs: _data_record(csv_path, rows),
    )

    def raw_predictor(
        _dataset: Any,
        windows: Any,
        _args: Any,
    ) -> signal_funnel._RawTimesFmPredictionBatch:
        return signal_funnel._RawTimesFmPredictionBatch(
            model_id="fake-raw-timesfm",
            point_forecasts=tuple(
                tuple(window.context_values[-1] - 10.0 for _ in window.future_values)
                for window in windows
            ),
            runtime_metadata={"backend": "fake_raw_predictor"},
        )

    output_root = tmp_path / "out"
    exit_code = signal_funnel.main(
        [
            "--symbols",
            "MU",
            "--as-of",
            "2026-05-11",
            "--data-dir",
            str(data_dir),
            "--output-root",
            str(output_root),
            "--run-id",
            "stage2-kill",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
        ],
        raw_predictor=raw_predictor,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    raw_row = next(row for row in records if row["stage"] == "raw_timesfm_screen")

    assert raw_row["status"] == "killed"
    assert raw_row["decision"] == "stop"
    assert raw_row["selected_for_next_stage"] is False
    assert raw_row["rmse_ratio_vs_best_baseline"] > 1.15
    assert raw_row["directional_delta_vs_best_baseline"] < -0.05
    assert raw_row["kill_reason"].startswith("raw_timesfm_underperformed_baselines:")


@pytest.mark.unit
def test_signal_funnel_stage1_skips_after_failed_data_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "MU.csv"
    rows = _ohlcv_rows("2026-05-11", count=40)
    write_ohlcv_csv(csv_path, rows)

    monkeypatch.setattr(
        signal_funnel,
        "ensure_symbol_data",
        lambda *_args, **_kwargs: _data_record(csv_path, rows),
    )

    output_root = tmp_path / "out"
    exit_code = signal_funnel.main(
        [
            "--symbols",
            "MU",
            "--as-of",
            "2026-05-11",
            "--data-dir",
            str(tmp_path / "data"),
            "--output-root",
            str(output_root),
            "--run-id",
            "stage1-skip",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 1
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    assert records[0]["stage"] == "data_check"
    assert records[0]["status"] == "failed"
    assert records[1]["stage"] == "baseline_screen"
    assert records[1]["status"] == "skipped"
    assert records[1]["kill_reason"] == "data_check_failed"


def _ohlcv_rows(last_day: str, *, count: int) -> list[dict[str, str]]:
    latest = date.fromisoformat(last_day)
    start = latest - timedelta(days=count - 1)
    rows: list[dict[str, str]] = []
    for index in range(count):
        current = start + timedelta(days=index)
        close = 100 + (index * 0.01)
        rows.append(
            {
                "timestamp": current.isoformat(),
                "open": f"{close - 0.01:.6f}",
                "high": f"{close + 0.02:.6f}",
                "low": f"{close - 0.02:.6f}",
                "close": f"{close:.6f}",
                "volume": "1000",
                "adjusted_close": f"{close:.6f}",
            }
        )
    return rows


def _data_record(csv_path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "csv_path": str(csv_path),
        "metadata_path": str(csv_path.with_suffix(".metadata.json")),
        "csv_sha256": "test-sha",
        "rows": len(rows),
        "first": rows[0]["timestamp"],
        "last": rows[-1]["timestamp"],
        "metadata": {
            "source": "test",
            "asset_type": "operating_company",
            "lineage_notes": [],
        },
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
