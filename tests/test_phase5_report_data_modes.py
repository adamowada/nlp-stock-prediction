from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.contracts import RunConfig
from nlp_stock_prediction.orchestration import (
    LIVE_REPORT_DATA_MODE,
    OFFLINE_FIXTURE_REPORT_DATA_MODE,
    Phase4Service,
    Phase4ToolExecutionError,
)
from nlp_stock_prediction.pipeline import (
    LIVE_ORCHESTRATION_DISABLED_MESSAGE,
    generate_daily_report,
)
from nlp_stock_prediction.storage import ToolRunRecord

RUN_DATE = date(2026, 5, 13)
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


@pytest.mark.unit
def test_pipeline_refuses_live_config_without_fixture_or_dummy_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="refusing to use fixture or dummy fallback"):
        generate_daily_report(
            RunConfig(
                run_date=RUN_DATE,
                output_dir=tmp_path / "reports",
                offline=False,
            )
        )

    assert "fixture or dummy fallback" in LIVE_ORCHESTRATION_DISABLED_MESSAGE


@pytest.mark.integration
def test_offline_phase4_report_inputs_are_machine_marked(tmp_path: Path) -> None:
    bundle = generate_daily_report(
        RunConfig(
            run_date="2026-05-11",
            output_dir=tmp_path / "reports",
            offline=True,
        )
    )

    payload = json.loads(bundle.json_path.read_text(encoding="utf-8"))
    audit_payload = json.loads(bundle.audit_manifest_path.read_text(encoding="utf-8"))

    assert payload["command_args"]["report_data_mode"] == OFFLINE_FIXTURE_REPORT_DATA_MODE
    assert payload["audit_manifest"]["command_args"]["report_data_mode"] == (
        OFFLINE_FIXTURE_REPORT_DATA_MODE
    )
    assert audit_payload["command_args"]["report_data_mode"] == OFFLINE_FIXTURE_REPORT_DATA_MODE
    tool_records = cast(Any, bundle.tool_records)
    assert any(
        record.inputs.get("report_data_mode") == OFFLINE_FIXTURE_REPORT_DATA_MODE
        for record in tool_records
    )


@pytest.mark.integration
def test_live_phase4_report_without_inputs_renders_insufficient_evidence(
    tmp_path: Path,
) -> None:
    service = Phase4Service(repo_root=tmp_path)
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/live-empty",
        symbol="TSLA",
        report_data_mode=LIVE_REPORT_DATA_MODE,
    )

    rendered = service.render_prediction_report(run_id=str(started["run_id"]), symbol="TSLA")

    payload = json.loads(Path(str(rendered["json_path"])).read_text(encoding="utf-8"))
    assert rendered["report_data_mode"] == LIVE_REPORT_DATA_MODE
    assert payload["command_args"]["report_data_mode"] == LIVE_REPORT_DATA_MODE
    assert payload["prediction_candidates"] == []
    assert payload["insufficient_evidence"]["provider_names"] == ["live-providers"]


@pytest.mark.integration
def test_live_phase4_report_rejects_fixture_or_dummy_leakage(tmp_path: Path) -> None:
    service = Phase4Service(repo_root=tmp_path)
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/live-leakage",
        symbol="TSLA",
        report_data_mode=LIVE_REPORT_DATA_MODE,
    )
    run_id = str(started["run_id"])
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-fixture-leakage",
            run_id=run_id,
            tool_name="phase4_universe_discovery",
            tool_version="phase4.fixture.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "TSLA", "report_data_mode": OFFLINE_FIXTURE_REPORT_DATA_MODE},
        )
    )

    with pytest.raises(Phase4ToolExecutionError) as exc_info:
        service.render_prediction_report(run_id=run_id, symbol="TSLA")

    assert isinstance(exc_info.value.original_error, ValueError)
    assert "live report assembly cannot use non-live report inputs" in str(
        exc_info.value.original_error
    )
