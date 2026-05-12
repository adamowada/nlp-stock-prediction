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
    assert manifest["implemented_stages"] == ["data_check"]
    assert manifest["skipped_stages"][0]["reason"] == "stage_not_implemented"
    assert leaderboard["rows"][0]["symbol"] == "MU"
    assert leaderboard["rows"][0]["status"] == "passed"
    assert leaderboard["rows"][0]["decision"] == "continue"
    assert leaderboard["rows"][0]["selected_for_next_stage"] is True
    assert leaderboard["rows"][0]["dataset_hash"] is None
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
