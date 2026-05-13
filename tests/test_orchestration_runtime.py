from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import (
    AssetClass,
    AuditManifest,
    DailyReport,
    InstrumentUniverse,
    JsonObject,
    RunConfig,
)
from nlp_stock_prediction.orchestration import (
    DEFAULT_STAGE_ORDER,
    ArtifactWriter,
    OrchestrationExecutionError,
    OrchestrationState,
    RunContext,
    StagedExecutor,
    ToolRegistry,
    ToolRunResult,
    ToolSpec,
    build_dummy_tool_registry,
    deterministic_generated_at,
    generate_dummy_report_bundle,
)
from nlp_stock_prediction.reporting.audit import stable_json_bytes

RUN_DATE = date(2026, 5, 12)


def _config(tmp_path: Path, *, offline: bool = True) -> RunConfig:
    return RunConfig(
        run_date=RUN_DATE,
        output_dir=tmp_path,
        offline=offline,
        source_mode="offline" if offline else "disabled",
    )


@pytest.mark.unit
def test_artifact_writer_writes_stable_json_with_audit_metadata(tmp_path: Path) -> None:
    created_at = deterministic_generated_at(RUN_DATE)
    writer = ArtifactWriter(
        base_dir=tmp_path / "audit",
        created_at=created_at,
        produced_by="unit-test",
    )
    payload: JsonObject = {
        "z": 1,
        "a": {"nested": True},
        "records": [{"id": "first"}, {"id": "second"}],
    }

    artifact = writer.write_json(
        artifact_id="stable-json",
        artifact_type="provider_result",
        filename="stable-json.json",
        payload=payload,
        record_count=2,
    )

    content = Path(artifact.path).read_bytes()
    assert content == stable_json_bytes(payload)
    assert artifact.sha256 == hashlib.sha256(content).hexdigest()
    assert artifact.created_at == created_at
    assert artifact.produced_by == "unit-test"
    assert artifact.record_count == 2


@pytest.mark.unit
def test_tool_registry_orders_by_stage_then_tool_id_and_rejects_duplicates() -> None:
    registry = build_dummy_tool_registry()

    ordered_ids = [tool.spec.tool_id for tool in registry.ordered_for_stages(DEFAULT_STAGE_ORDER)]

    assert ordered_ids == [
        "dummy.instrument-discovery",
        "dummy.evidence-collection",
        "dummy.analysis",
        "dummy.prediction-scoring",
        "dummy.report-assembly",
    ]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.get("dummy.analysis"))


@pytest.mark.unit
def test_staged_executor_runs_dummy_tools_and_records_artifacts(tmp_path: Path) -> None:
    context = RunContext.from_config(_config(tmp_path))

    result = StagedExecutor(
        registry=build_dummy_tool_registry(),
        stage_order=DEFAULT_STAGE_ORDER,
    ).run(context)

    report = result.state.require("daily_report", DailyReport)
    assert report.run_id == "research-2026-05-12"
    assert [record.tool_id for record in result.tool_records] == [
        "dummy.instrument-discovery",
        "dummy.evidence-collection",
        "dummy.analysis",
        "dummy.prediction-scoring",
        "dummy.report-assembly",
    ]
    assert [artifact.artifact_id for artifact in result.artifacts] == [
        "instrument-universe",
        "normalized-evidence",
        "analysis-contexts",
        "prediction-inputs",
    ]
    assert all(Path(artifact.path).exists() for artifact in result.artifacts)
    universe = result.state.require("instrument_universe", InstrumentUniverse)
    assert set(universe.instrument_ids) >= {
        "instrument:equity:us:tsla",
        "instrument:etf:us:spy",
        "instrument:crypto:btc-usd",
        "instrument:futures:cme:esm6",
    }
    assert [instrument.asset_class for instrument in universe.instruments] == [
        AssetClass.STOCK,
        AssetClass.ETF,
        AssetClass.CRYPTO,
        AssetClass.FUTURES,
    ]
    assert result.state.require("instrument_resolutions", tuple) == universe.resolutions
    assert universe.warnings
    assert isinstance(report.audit_manifest, AuditManifest)
    assert report.instrument_resolutions == universe.resolutions
    assert report.audit_manifest.prediction_trace_ids == ("prediction-tsla-dummy-volatility",)


@pytest.mark.unit
def test_staged_executor_records_failed_tool_before_raising(tmp_path: Path) -> None:
    class FailingTool:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                tool_id="dummy.failing-tool",
                stage="fail",
                description="Raise a deterministic failure.",
            )

        def run(self, _context: RunContext, _state: OrchestrationState) -> ToolRunResult:
            raise ValueError("deterministic failure")

    context = RunContext.from_config(_config(tmp_path))

    with pytest.raises(OrchestrationExecutionError) as exc_info:
        StagedExecutor(
            registry=ToolRegistry((FailingTool(),)),
            stage_order=("fail",),
        ).run(context)

    assert isinstance(exc_info.value.original_error, ValueError)
    assert exc_info.value.tool_id == "dummy.failing-tool"
    assert exc_info.value.partial_result.tool_records[-1].status == "failed"
    assert exc_info.value.partial_result.tool_records[-1].error_message == "deterministic failure"


@pytest.mark.unit
def test_generate_dummy_report_bundle_writes_markdown_json_and_manifest(tmp_path: Path) -> None:
    bundle = generate_dummy_report_bundle(_config(tmp_path))

    assert bundle.markdown_path.exists()
    assert bundle.json_path.exists()
    assert bundle.audit_manifest_path.exists()
    assert bundle.audit_dir == tmp_path / RUN_DATE.isoformat() / "audit"

    report_payload = json.loads(bundle.json_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(bundle.audit_manifest_path.read_text(encoding="utf-8"))
    markdown = bundle.markdown_path.read_text(encoding="utf-8")

    assert report_payload["run_id"] == "research-2026-05-12"
    assert report_payload["prediction_candidates"][0]["status"] == "evidence_supported"
    assert [instrument["asset_class"] for instrument in report_payload["instruments"]] == [
        "stock",
        "etf",
        "crypto",
        "futures",
    ]
    assert any(
        resolution["status"] == "ambiguous"
        for resolution in report_payload["instrument_resolutions"]
    )
    assert [artifact["artifact_id"] for artifact in manifest_payload["artifacts"]] == [
        "instrument-universe",
        "normalized-evidence",
        "analysis-contexts",
        "prediction-inputs",
    ]
    assert "not a buy or sell instruction" in markdown
    assert bundle.tool_records[-1].updated_keys == ("daily_report",)

    first_json = bundle.json_path.read_text(encoding="utf-8")
    second_bundle = generate_dummy_report_bundle(_config(tmp_path))
    assert second_bundle.json_path.read_text(encoding="utf-8") == first_json


@pytest.mark.unit
def test_generate_dummy_report_bundle_requires_offline_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="offline RunConfig"):
        generate_dummy_report_bundle(_config(tmp_path, offline=False))
