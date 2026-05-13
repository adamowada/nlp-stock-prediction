from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from nlp_stock_prediction.analysis.ml_signal import load_timesfm_ml_signal_attachment
from nlp_stock_prediction.ml.timesfm import signal_funnel
from nlp_stock_prediction.ml.timesfm.evaluate import TimesFmEvaluationArtifact
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
        "adapter_smoke",
        "survivor_hpo",
        "final_eval",
        "report_ready",
    ]
    assert manifest["skipped_stages"] == []
    assert len(leaderboard["rows"]) == 7
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
    assert leaderboard["rows"][3]["stage"] == "adapter_smoke"
    assert leaderboard["rows"][3]["status"] == "skipped"
    assert leaderboard["rows"][3]["kill_reason"] == "dry_run_no_adapter_training"
    assert leaderboard["rows"][4]["stage"] == "survivor_hpo"
    assert leaderboard["rows"][4]["status"] == "skipped"
    assert leaderboard["rows"][4]["kill_reason"] == "dry_run_no_hpo"
    assert leaderboard["rows"][5]["stage"] == "final_eval"
    assert leaderboard["rows"][5]["status"] == "skipped"
    assert leaderboard["rows"][5]["kill_reason"] == "dry_run_no_final_eval"
    assert leaderboard["rows"][6]["stage"] == "report_ready"
    assert leaderboard["rows"][6]["status"] == "skipped"
    assert leaderboard["rows"][6]["kill_reason"] == "dry_run_no_report_ready_artifact"
    assert csv_rows[0]["symbol"] == "MU"
    assert csv_rows[0]["status"] == "passed"


@pytest.mark.unit
def test_signal_funnel_quick_profile_defaults_to_raw_screen(tmp_path: Path) -> None:
    output_root = tmp_path / "out"

    exit_code = signal_funnel.main(
        [
            "--dry-run",
            "--symbols",
            "MU",
            "--as-of",
            "2026-05-11",
            "--output-root",
            str(output_root),
            "--run-id",
            "quick-profile",
            "--device",
            "cpu",
            "--profile",
            "quick",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    leaderboard = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))

    assert manifest["requested_stop_after"] == "raw_timesfm_screen"
    assert [row["stage"] for row in leaderboard["rows"]] == [
        "data_check",
        "baseline_screen",
        "raw_timesfm_screen",
    ]


@pytest.mark.unit
def test_signal_funnel_defaults_to_small_hpo_budget() -> None:
    args = signal_funnel.build_parser().parse_args([])

    assert args.max_hpo_trials_per_ticker == 8


@pytest.mark.unit
def test_signal_funnel_can_load_sp500_universe_from_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "universes"
    cache_dir.mkdir()
    cached_symbols = [f"A{index}" for index in range(401)]
    (cache_dir / "sp500-symbols.csv").write_text(
        "symbol\n" + "\n".join(cached_symbols) + "\n",
        encoding="utf-8",
    )

    args = signal_funnel.build_parser().parse_args(
        ["--universe", "sp500", "--universe-cache-dir", str(cache_dir)]
    )
    symbols, source = signal_funnel._resolve_symbols(args)

    assert symbols == tuple(cached_symbols)
    assert source["universe"] == "sp500"
    assert source["source"] == "cache"


@pytest.mark.unit
def test_signal_funnel_rejects_truncated_sp500_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "universes"
    cache_dir.mkdir()
    (cache_dir / "sp500-symbols.csv").write_text("symbol\nMSFT\nGOOG\n", encoding="utf-8")

    with pytest.raises(ValueError, match="returned only 2 symbols"):
        signal_funnel._load_sp500_symbols(cache_dir=cache_dir, refresh=False)


@pytest.mark.unit
def test_signal_funnel_rejects_empty_explicit_symbols() -> None:
    args = signal_funnel.build_parser().parse_args(["--symbols", ""])

    with pytest.raises(ValueError, match="at least one symbol is required"):
        signal_funnel._resolve_symbols(args)


@pytest.mark.unit
def test_signal_funnel_can_load_symbols_file(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text(" $mu, msft\n# comment\nGOOG\nMSFT\n", encoding="utf-8")

    exit_code = signal_funnel.main(
        [
            "--dry-run",
            "--symbols-file",
            str(symbols_file),
            "--as-of",
            "2026-05-11",
            "--output-root",
            str(output_root),
            "--run-id",
            "symbols-file",
            "--device",
            "cpu",
            "--stop-after",
            "data_check",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    leaderboard = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))

    assert manifest["symbols"] == ["MU", "MSFT", "GOOG"]
    assert manifest["symbol_source"] == {"kind": "symbols_file", "path": str(symbols_file)}
    assert [row["symbol"] for row in leaderboard["rows"]] == ["MU", "MSFT", "GOOG"]


@pytest.mark.unit
def test_signal_funnel_parses_sp500_wikitext_templates() -> None:
    templates = "\n".join(f"|{{{{NyseSymbol|A{index}}}}}" for index in range(401))
    wikitext = templates + "\n|{{NyseSymbol|BRK.B}}\n|{{NasdaqSymbol|GOOG}}\n"

    symbols = signal_funnel._parse_sp500_wikitext(wikitext)

    assert symbols[:3] == ("A0", "A1", "A2")
    assert "BRK.B" in symbols
    assert "GOOG" in symbols


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
            "--stop-after",
            "raw_timesfm_screen",
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

    candidates = json.loads((output_root / "raw_candidates.json").read_text(encoding="utf-8"))
    candidate_rows = candidates["rows"]
    assert candidate_rows[0]["symbol"] == "MU"
    assert candidate_rows[0]["rank"] == 1
    assert candidate_rows[0]["status"] == "passed"
    assert candidate_rows[0]["candidate_score"] == pytest.approx(
        round(
            1.0
            - raw_row["rmse_ratio_vs_best_baseline"]
            + raw_row["directional_delta_vs_best_baseline"],
            8,
        )
    )
    candidate_csv_rows = _read_csv_rows(output_root / "raw_candidates.csv")
    assert candidate_csv_rows[0]["symbol"] == "MU"

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
            "--stop-after",
            "raw_timesfm_screen",
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

    candidates = json.loads((output_root / "raw_candidates.json").read_text(encoding="utf-8"))
    assert candidates["rows"] == []
    assert _read_csv_rows(output_root / "raw_candidates.csv") == []


@pytest.mark.unit
def test_signal_funnel_stage3_promotes_adapter_smoke_with_positive_lift(
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
            "stage3-promote",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--stop-after",
            "adapter_smoke",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    adapter_row = next(row for row in records if row["stage"] == "adapter_smoke")

    assert len(records) == 6
    assert adapter_row["method"] == "lora_adapter_smoke"
    assert adapter_row["status"] == "passed"
    assert adapter_row["decision"] == "run_hpo"
    assert adapter_row["selected_for_next_stage"] is True
    assert adapter_row["adapter_sha256"] == "fake-adapter-sha"
    assert adapter_row["training_metadata"].endswith("training-metadata.json")
    assert adapter_row["validation_mean_loss"] == 0.0123
    assert adapter_row["adapter_rmse_ratio_vs_raw"] == 0.0
    assert adapter_row["adapter_directional_delta_vs_raw"] == 0.0
    assert adapter_row["rmse_ratio_vs_best_baseline"] == 0.0

    artifact = json.loads(Path(adapter_row["evaluation_artifact"]).read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "ml.timesfm.adapter_smoke_evaluation.v1"
    assert artifact["decision"] == "run_hpo"
    assert artifact["adapter_sha256"] == "fake-adapter-sha"
    assert artifact["metrics"]["sample_count"] == 16
    assert artifact["adapter_rmse_ratio_vs_raw"] == 0.0


@pytest.mark.unit
def test_signal_funnel_stage3_kills_adapter_smoke_without_raw_lift(
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
            "stage3-kill",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--stop-after",
            "adapter_smoke",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_bad_adapter_smoke_runner,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    adapter_row = next(row for row in records if row["stage"] == "adapter_smoke")

    assert adapter_row["status"] == "killed"
    assert adapter_row["decision"] == "stop"
    assert adapter_row["selected_for_next_stage"] is False
    assert adapter_row["adapter_rmse_ratio_vs_raw"] > 1.0
    assert adapter_row["adapter_directional_delta_vs_raw"] < 0.0
    assert adapter_row["kill_reason"].startswith("adapter_smoke_no_lift_vs_raw:")


@pytest.mark.unit
def test_signal_funnel_stage3_records_failed_adapter_training(
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

    def failing_adapter_runner(
        *_args: Any, **_kwargs: Any
    ) -> signal_funnel._AdapterSmokePredictionBatch:
        raise RuntimeError("synthetic adapter failure")

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
            "stage3-failure",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--stop-after",
            "adapter_smoke",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=failing_adapter_runner,
    )

    assert exit_code == 1
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    adapter_row = next(row for row in records if row["stage"] == "adapter_smoke")
    assert adapter_row["status"] == "failed"
    assert adapter_row["decision"] == "stop"
    assert adapter_row["kill_reason"] == "synthetic adapter failure"


@pytest.mark.unit
def test_signal_funnel_stage3_skips_when_raw_screen_kills_ticker(
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

    def forbidden_adapter_runner(
        *_args: Any, **_kwargs: Any
    ) -> signal_funnel._AdapterSmokePredictionBatch:
        raise AssertionError("adapter smoke should not run after raw kill")

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
            "stage3-raw-kill-skip",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--stop-after",
            "adapter_smoke",
        ],
        raw_predictor=raw_predictor,
        adapter_smoke_runner=forbidden_adapter_runner,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    raw_row = next(row for row in records if row["stage"] == "raw_timesfm_screen")
    adapter_row = next(row for row in records if row["stage"] == "adapter_smoke")

    assert raw_row["status"] == "killed"
    assert adapter_row["status"] == "skipped"
    assert adapter_row["decision"] == "stop"
    assert adapter_row["kill_reason"] == "raw_timesfm_not_selected"


@pytest.mark.unit
def test_signal_funnel_stage3_reuses_existing_adapter_smoke_artifact(
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

    calls = 0

    def counted_adapter_runner(
        dataset: Any,
        windows: Any,
        csv_path_arg: Path,
        output_dir: Path,
        args: Any,
    ) -> signal_funnel._AdapterSmokePredictionBatch:
        nonlocal calls
        calls += 1
        return _exact_adapter_smoke_runner(dataset, windows, csv_path_arg, output_dir, args)

    output_root = tmp_path / "out"
    common_args = [
        "--symbols",
        "MU",
        "--as-of",
        "2026-05-11",
        "--data-dir",
        str(data_dir),
        "--output-root",
        str(output_root),
        "--sleep-seconds",
        "0",
        "--device",
        "cpu",
        "--screen-max-windows",
        "16",
        "--stop-after",
        "adapter_smoke",
    ]
    assert (
        signal_funnel.main(
            [*common_args, "--run-id", "stage3-reuse-first"],
            raw_predictor=_halfway_positive_raw_predictor,
            adapter_smoke_runner=counted_adapter_runner,
        )
        == 0
    )
    assert (
        signal_funnel.main(
            [*common_args, "--run-id", "stage3-reuse-second"],
            raw_predictor=_halfway_positive_raw_predictor,
            adapter_smoke_runner=counted_adapter_runner,
        )
        == 0
    )

    assert calls == 1
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    adapter_row = next(row for row in records if row["stage"] == "adapter_smoke")
    assert adapter_row["status"] == "passed"
    assert adapter_row["decision"] == "run_hpo"
    assert "reused=true" in adapter_row["notes"]


@pytest.mark.unit
def test_signal_funnel_stage4_selects_hpo_by_baseline_lift_before_loss(
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
            "stage4-select",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--stop-after",
            "survivor_hpo",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
        survivor_hpo_runner=_baseline_lift_hpo_runner,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    hpo_rows = [row for row in records if row["stage"] == "survivor_hpo"]
    selected = next(row for row in hpo_rows if row["selected_for_next_stage"])
    low_loss = next(row for row in hpo_rows if row["method"] == "trial_low_loss_weak")

    assert len(hpo_rows) == 2
    assert selected["method"] == "trial_high_loss_exact"
    assert selected["status"] == "passed"
    assert selected["decision"] == "continue"
    assert selected["validation_mean_loss"] == 0.2
    assert selected["rmse_ratio_vs_best_baseline"] == 0.0
    assert low_loss["validation_mean_loss"] == 0.001
    assert low_loss["selected_for_next_stage"] is False

    summary = json.loads(
        (output_root / "survivor_hpo" / "MU.summary.json").read_text(encoding="utf-8")
    )
    assert summary["selected_trial_id"] == "trial_high_loss_exact"


@pytest.mark.unit
def test_signal_funnel_stage4_skips_when_adapter_smoke_does_not_promote(
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

    def forbidden_hpo_runner(
        *_args: Any, **_kwargs: Any
    ) -> tuple[signal_funnel._SurvivorHpoPredictionBatch, ...]:
        raise AssertionError("survivor HPO should not run after adapter smoke kill")

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
            "stage4-skip",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--stop-after",
            "survivor_hpo",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_bad_adapter_smoke_runner,
        survivor_hpo_runner=forbidden_hpo_runner,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    adapter_row = next(row for row in records if row["stage"] == "adapter_smoke")
    hpo_row = next(row for row in records if row["stage"] == "survivor_hpo")

    assert adapter_row["status"] == "killed"
    assert hpo_row["status"] == "skipped"
    assert hpo_row["decision"] == "stop"
    assert hpo_row["kill_reason"] == "adapter_smoke_not_selected"


@pytest.mark.unit
def test_signal_funnel_stage4_records_hpo_failure(
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

    def failing_hpo_runner(
        *_args: Any, **_kwargs: Any
    ) -> tuple[signal_funnel._SurvivorHpoPredictionBatch, ...]:
        raise RuntimeError("synthetic survivor hpo failure")

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
            "stage4-failure",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--stop-after",
            "survivor_hpo",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
        survivor_hpo_runner=failing_hpo_runner,
    )

    assert exit_code == 1
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    hpo_row = next(row for row in records if row["stage"] == "survivor_hpo")
    assert hpo_row["status"] == "failed"
    assert hpo_row["decision"] == "stop"
    assert hpo_row["kill_reason"] == "synthetic survivor hpo failure"


@pytest.mark.unit
def test_signal_funnel_stage5_promotes_suitable_held_out_final_eval(
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
            "stage5-promote",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--final-max-windows",
            "8",
            "--stop-after",
            "final_eval",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
        survivor_hpo_runner=_baseline_lift_hpo_runner,
        final_eval_runner=_exact_final_eval_runner,
    )

    assert exit_code == 0
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    final_row = next(row for row in records if row["stage"] == "final_eval")

    assert manifest["requested_stop_after"] == "final_eval"
    assert final_row["method"] == "trial_high_loss_exact"
    assert final_row["status"] == "suitable"
    assert final_row["decision"] == "promote_for_scoring"
    assert final_row["selected_for_next_stage"] is True
    assert final_row["promoted_for_scoring"] is True
    assert final_row["rmse"] == 0.0
    assert final_row["rmse_ratio_vs_best_baseline"] == 0.0
    assert final_row["max_windows"] == 8
    assert "split=test" in final_row["notes"]

    artifact = json.loads(Path(final_row["evaluation_artifact"]).read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "ml.timesfm.final_evaluation.v1"
    assert artifact["status"] == "suitable"
    assert artifact["suitable_for_scoring"] is True
    assert artifact["selected_hpo"]["method"] == "trial_high_loss_exact"
    assert artifact["metrics"]["sample_count"] == 8
    assert len(artifact["records"]) == 8
    assert artifact["warning_ids"] == []
    assert artifact["model_source"]["training_ticker"] == "MU"
    assert artifact["forward_forecast"]["expected_return"] == pytest.approx(0.02)


@pytest.mark.unit
def test_signal_funnel_stage5_keeps_weak_final_eval_audit_only(
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
            "stage5-weak",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--final-max-windows",
            "12",
            "--stop-after",
            "final_eval",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
        survivor_hpo_runner=_baseline_lift_hpo_runner,
        final_eval_runner=_bad_final_eval_runner,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    final_row = next(row for row in records if row["stage"] == "final_eval")

    assert final_row["status"] == "weak"
    assert final_row["decision"] == "audit_only"
    assert final_row["selected_for_next_stage"] is False
    assert final_row["promoted_for_scoring"] is False
    assert "low_directional_accuracy" in final_row["kill_reason"]
    assert "underperforms_best_rmse_baseline" in final_row["kill_reason"]

    artifact = json.loads(Path(final_row["evaluation_artifact"]).read_text(encoding="utf-8"))
    assert artifact["suitable_for_scoring"] is False
    assert "low_directional_accuracy" in artifact["suitability_reasons"]


@pytest.mark.unit
def test_signal_funnel_stage5_blocks_missing_forward_forecast(
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
            "stage5-missing-forward",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--final-max-windows",
            "8",
            "--stop-after",
            "final_eval",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
        survivor_hpo_runner=_baseline_lift_hpo_runner,
        final_eval_runner=_missing_forward_forecast_final_eval_runner,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    final_row = next(row for row in records if row["stage"] == "final_eval")

    assert final_row["status"] == "weak"
    assert final_row["decision"] == "audit_only"
    assert final_row["selected_for_next_stage"] is False
    assert final_row["promoted_for_scoring"] is False
    assert "missing_forward_forecast" in final_row["kill_reason"]

    artifact = json.loads(Path(final_row["evaluation_artifact"]).read_text(encoding="utf-8"))
    assert artifact["suitable_for_scoring"] is False
    assert artifact["forward_forecast"] is None
    assert "missing_forward_forecast" in artifact["suitability_reasons"]


@pytest.mark.unit
def test_signal_funnel_stage5_blocks_model_source_warning(
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
            "stage5-warning",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--final-max-windows",
            "8",
            "--stop-after",
            "final_eval",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
        survivor_hpo_runner=_baseline_lift_hpo_runner,
        final_eval_runner=_warning_final_eval_runner,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    final_row = next(row for row in records if row["stage"] == "final_eval")

    assert final_row["status"] == "weak"
    assert final_row["decision"] == "audit_only"
    assert final_row["selected_for_next_stage"] is False
    assert final_row["promoted_for_scoring"] is False
    assert "timesfm_adapter_hash_mismatch" in final_row["kill_reason"]

    artifact = json.loads(Path(final_row["evaluation_artifact"]).read_text(encoding="utf-8"))
    assert artifact["suitable_for_scoring"] is False
    assert artifact["warning_ids"] == ["timesfm_adapter_hash_mismatch"]
    assert "timesfm_adapter_hash_mismatch" in artifact["suitability_reasons"]


@pytest.mark.unit
def test_signal_funnel_stage5_blocks_training_ticker_mismatch(
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
            "stage5-ticker-mismatch",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--final-max-windows",
            "8",
            "--stop-after",
            "final_eval",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
        survivor_hpo_runner=_baseline_lift_hpo_runner,
        final_eval_runner=_mismatched_ticker_final_eval_runner,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    final_row = next(row for row in records if row["stage"] == "final_eval")

    assert final_row["status"] == "weak"
    assert final_row["decision"] == "audit_only"
    assert final_row["selected_for_next_stage"] is False
    assert final_row["promoted_for_scoring"] is False
    assert "training_ticker_mismatch" in final_row["kill_reason"]

    artifact = json.loads(Path(final_row["evaluation_artifact"]).read_text(encoding="utf-8"))
    assert artifact["suitable_for_scoring"] is False
    assert artifact["model_source"]["training_ticker"] == "SPY"
    assert "training_ticker_mismatch" in artifact["suitability_reasons"]


@pytest.mark.unit
def test_signal_funnel_stage5_skips_when_survivor_hpo_has_no_selection(
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

    def forbidden_hpo_runner(
        *_args: Any, **_kwargs: Any
    ) -> tuple[signal_funnel._SurvivorHpoPredictionBatch, ...]:
        raise AssertionError("survivor HPO should not run after adapter smoke kill")

    def forbidden_final_runner(
        *_args: Any, **_kwargs: Any
    ) -> signal_funnel._FinalEvalPredictionBatch:
        raise AssertionError("final eval should not run without a selected HPO survivor")

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
            "stage5-skip",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--stop-after",
            "final_eval",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_bad_adapter_smoke_runner,
        survivor_hpo_runner=forbidden_hpo_runner,
        final_eval_runner=forbidden_final_runner,
    )

    assert exit_code == 0
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    final_row = next(row for row in records if row["stage"] == "final_eval")

    assert final_row["status"] == "skipped"
    assert final_row["decision"] == "stop"
    assert final_row["kill_reason"] == "survivor_hpo_not_selected"


@pytest.mark.unit
def test_signal_funnel_stage5_records_final_eval_failure(
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

    def failing_final_eval_runner(
        *_args: Any, **_kwargs: Any
    ) -> signal_funnel._FinalEvalPredictionBatch:
        raise RuntimeError("synthetic final eval failure")

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
            "stage5-failure",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--final-max-windows",
            "12",
            "--stop-after",
            "final_eval",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
        survivor_hpo_runner=_baseline_lift_hpo_runner,
        final_eval_runner=failing_final_eval_runner,
    )

    assert exit_code == 1
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    final_row = next(row for row in records if row["stage"] == "final_eval")
    assert final_row["status"] == "failed"
    assert final_row["decision"] == "stop"
    assert final_row["kill_reason"] == "synthetic final eval failure"


@pytest.mark.unit
def test_signal_funnel_stage6_promotes_report_ready_artifact(
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
            "stage6-promote",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--final-max-windows",
            "8",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
        survivor_hpo_runner=_baseline_lift_hpo_runner,
        final_eval_runner=_exact_final_eval_runner,
    )

    assert exit_code == 0
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    report_row = next(row for row in records if row["stage"] == "report_ready")

    assert manifest["requested_stop_after"] == "report_ready"
    assert report_row["method"] == "trial_high_loss_exact"
    assert report_row["status"] == "suitable"
    assert report_row["decision"] == "promote_for_scoring"
    assert report_row["promoted_for_scoring"] is True
    assert Path(report_row["evaluation_artifact"]).parts[-4:] == (
        "report_ready",
        "MU",
        "best",
        "evaluation.json",
    )

    promoted_path = Path(report_row["evaluation_artifact"])
    promoted = TimesFmEvaluationArtifact.model_validate_json(
        promoted_path.read_text(encoding="utf-8")
    )
    attachment = load_timesfm_ml_signal_attachment(promoted_path)
    report_manifest = json.loads(
        (output_root / "report_ready" / "MU" / "manifest.json").read_text(encoding="utf-8")
    )

    assert promoted.schema_version == "ml.timesfm.evaluation.v1"
    assert promoted.status == "suitable"
    assert promoted.suitable_for_scoring is True
    assert promoted.ticker == "MU"
    assert promoted.metrics.sample_count == 8
    assert len(promoted.records) == 8
    assert promoted.forward_forecast is not None
    assert promoted.forward_forecast.expected_return == pytest.approx(0.02)
    assert attachment.signal.status == "usable"
    assert attachment.signal.feature_end == promoted.forward_forecast.context_end
    assert attachment.signal.expected_return == pytest.approx(
        promoted.forward_forecast.expected_return
    )
    forward_metadata = attachment.signal.metadata["forward_forecast"]
    assert isinstance(forward_metadata, dict)
    assert forward_metadata["expected_return"] == pytest.approx(0.02)
    assert manifest["promoted_artifacts"] == [
        {
            "symbol": "MU",
            "path": str(promoted_path),
            "source_final_eval_artifact": str(output_root / "final_eval" / "MU.evaluation.json"),
        }
    ]
    assert manifest["final_states"][0]["promoted_artifact"] == str(promoted_path)
    assert report_manifest["promoted_evaluation_artifact"] == str(promoted_path)
    assert "--ml-artifact" in report_manifest["report_command_hint"]


@pytest.mark.unit
def test_signal_funnel_stage6_skips_weak_final_eval_without_promotion(
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
            "stage6-weak",
            "--sleep-seconds",
            "0",
            "--device",
            "cpu",
            "--screen-max-windows",
            "16",
            "--final-max-windows",
            "12",
        ],
        raw_predictor=_halfway_positive_raw_predictor,
        adapter_smoke_runner=_exact_adapter_smoke_runner,
        survivor_hpo_runner=_baseline_lift_hpo_runner,
        final_eval_runner=_bad_final_eval_runner,
    )

    assert exit_code == 0
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    records = json.loads((output_root / "leaderboard.json").read_text(encoding="utf-8"))["rows"]
    report_row = next(row for row in records if row["stage"] == "report_ready")
    report_manifest = json.loads(
        Path(report_row["evaluation_artifact"]).read_text(encoding="utf-8")
    )

    assert report_row["status"] == "skipped"
    assert report_row["decision"] == "audit_only"
    assert report_row["promoted_for_scoring"] is False
    assert report_row["kill_reason"] == "final_eval_not_suitable_for_scoring"
    assert manifest["promoted_artifacts"] == []
    assert manifest["final_states"][0]["promoted_artifact"] is None
    assert report_manifest["promoted_evaluation_artifact"] is None
    assert not (output_root / "report_ready" / "MU" / "best" / "evaluation.json").exists()


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


def _halfway_positive_raw_predictor(
    _dataset: Any,
    windows: Any,
    args: Any,
) -> signal_funnel._RawTimesFmPredictionBatch:
    return signal_funnel._RawTimesFmPredictionBatch(
        model_id="fake-raw-timesfm",
        model_revision="fake-revision",
        point_forecasts=tuple(
            tuple(window.context_values[-1] + 0.08 for _ in window.future_values)
            for window in windows
        ),
        runtime_metadata={"backend": "fake_raw_predictor", "device": args.device},
    )


def _exact_adapter_smoke_runner(
    _dataset: Any,
    windows: Any,
    _csv_path: Path,
    output_dir: Path,
    args: Any,
) -> signal_funnel._AdapterSmokePredictionBatch:
    metadata_path = output_dir / "training-metadata.json"
    return signal_funnel._AdapterSmokePredictionBatch(
        model_id="fake-adapter-timesfm",
        model_revision="fake-adapter-revision",
        adapter_sha256="fake-adapter-sha",
        training_metadata_path=str(metadata_path),
        validation_mean_loss=0.0123,
        point_forecasts=tuple(tuple(window.future_values) for window in windows),
        runtime_metadata={"backend": "fake_adapter_runner", "device": args.device},
    )


def _bad_adapter_smoke_runner(
    _dataset: Any,
    windows: Any,
    _csv_path: Path,
    output_dir: Path,
    _args: Any,
) -> signal_funnel._AdapterSmokePredictionBatch:
    metadata_path = output_dir / "training-metadata.json"
    return signal_funnel._AdapterSmokePredictionBatch(
        model_id="fake-adapter-timesfm",
        adapter_sha256="fake-adapter-sha",
        training_metadata_path=str(metadata_path),
        validation_mean_loss=0.4567,
        point_forecasts=tuple(
            tuple(window.context_values[-1] - 10.0 for _ in window.future_values)
            for window in windows
        ),
        runtime_metadata={"backend": "fake_adapter_runner"},
    )


def _baseline_lift_hpo_runner(
    _dataset: Any,
    windows: Any,
    _csv_path: Path,
    output_dir: Path,
    _args: Any,
) -> tuple[signal_funnel._SurvivorHpoPredictionBatch, ...]:
    return (
        signal_funnel._SurvivorHpoPredictionBatch(
            trial_id="trial_low_loss_weak",
            model_id="fake-hpo-timesfm",
            adapter_sha256="fake-hpo-weak-sha",
            training_metadata_path=str(output_dir / "weak" / "training-metadata.json"),
            validation_mean_loss=0.001,
            point_forecasts=tuple(
                tuple(window.context_values[-1] + 0.08 for _ in window.future_values)
                for window in windows
            ),
            runtime_metadata={"backend": "fake_hpo_runner"},
        ),
        signal_funnel._SurvivorHpoPredictionBatch(
            trial_id="trial_high_loss_exact",
            model_id="fake-hpo-timesfm",
            adapter_sha256="fake-hpo-exact-sha",
            training_metadata_path=str(output_dir / "exact" / "training-metadata.json"),
            validation_mean_loss=0.2,
            point_forecasts=tuple(tuple(window.future_values) for window in windows),
            runtime_metadata={"backend": "fake_hpo_runner"},
        ),
    )


def _exact_final_eval_runner(
    dataset: Any,
    windows: Any,
    selected_hpo_row: signal_funnel.SignalFunnelLeaderboardRow,
    _output_dir: Path,
    args: Any,
) -> signal_funnel._FinalEvalPredictionBatch:
    assert {window.split for window in windows} == {"test"}
    return signal_funnel._FinalEvalPredictionBatch(
        source_trial_id=selected_hpo_row.method,
        model_id=selected_hpo_row.model_id,
        model_revision=selected_hpo_row.model_revision,
        adapter_sha256=selected_hpo_row.adapter_sha256,
        training_ticker=dataset.ticker,
        training_metadata_path=selected_hpo_row.training_metadata,
        validation_mean_loss=selected_hpo_row.validation_mean_loss,
        point_forecasts=tuple(tuple(window.future_values) for window in windows),
        forward_forecast=_forward_forecast_payload(dataset, expected_return=0.02),
        runtime_metadata={"backend": "fake_final_eval_runner", "device": args.device},
    )


def _bad_final_eval_runner(
    dataset: Any,
    windows: Any,
    selected_hpo_row: signal_funnel.SignalFunnelLeaderboardRow,
    _output_dir: Path,
    _args: Any,
) -> signal_funnel._FinalEvalPredictionBatch:
    assert {window.split for window in windows} == {"test"}
    return signal_funnel._FinalEvalPredictionBatch(
        source_trial_id=selected_hpo_row.method,
        model_id=selected_hpo_row.model_id,
        adapter_sha256=selected_hpo_row.adapter_sha256,
        training_ticker=dataset.ticker,
        training_metadata_path=selected_hpo_row.training_metadata,
        validation_mean_loss=selected_hpo_row.validation_mean_loss,
        point_forecasts=tuple(
            tuple(window.context_values[-1] - 10.0 for _ in window.future_values)
            for window in windows
        ),
        forward_forecast=_forward_forecast_payload(dataset, expected_return=-0.02),
        runtime_metadata={"backend": "fake_final_eval_runner"},
    )


def _warning_final_eval_runner(
    dataset: Any,
    windows: Any,
    selected_hpo_row: signal_funnel.SignalFunnelLeaderboardRow,
    output_dir: Path,
    args: Any,
) -> signal_funnel._FinalEvalPredictionBatch:
    batch = _exact_final_eval_runner(dataset, windows, selected_hpo_row, output_dir, args)
    return batch.model_copy(update={"warning_ids": ("timesfm_adapter_hash_mismatch",)})


def _missing_forward_forecast_final_eval_runner(
    dataset: Any,
    windows: Any,
    selected_hpo_row: signal_funnel.SignalFunnelLeaderboardRow,
    output_dir: Path,
    args: Any,
) -> signal_funnel._FinalEvalPredictionBatch:
    batch = _exact_final_eval_runner(dataset, windows, selected_hpo_row, output_dir, args)
    return batch.model_copy(update={"forward_forecast": None})


def _mismatched_ticker_final_eval_runner(
    dataset: Any,
    windows: Any,
    selected_hpo_row: signal_funnel.SignalFunnelLeaderboardRow,
    output_dir: Path,
    args: Any,
) -> signal_funnel._FinalEvalPredictionBatch:
    batch = _exact_final_eval_runner(dataset, windows, selected_hpo_row, output_dir, args)
    return batch.model_copy(update={"training_ticker": "SPY"})


def _forward_forecast_payload(dataset: Any, *, expected_return: float) -> dict[str, Any]:
    latest_window = max(dataset.windows, key=lambda window: window.horizon_end_index)
    context_final = latest_window.future_values[-1]
    final_value = context_final * (1.0 + expected_return)
    point_forecast = [
        context_final + ((final_value - context_final) * ((index + 1) / dataset.horizon_length))
        for index in range(dataset.horizon_length)
    ]
    return {
        "context_start": str(latest_window.context_start),
        "context_end": str(latest_window.horizon_end),
        "forecast_horizon_sessions": dataset.horizon_length,
        "point_forecast": point_forecast,
        "expected_return": expected_return,
        "interval_lower": min(context_final, final_value) * 0.99,
        "interval_upper": max(context_final, final_value) * 1.01,
        "interval_width": 0.04,
        "directional_probability_proxy": 0.8 if expected_return > 0 else 0.2,
    }


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
