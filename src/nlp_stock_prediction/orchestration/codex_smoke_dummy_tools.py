"""Dummy Codex Smoke Stage MCP tools used by the real-agent smoke path."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.instruments.registry import instrument_to_record
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.instrument_universe import (
    FixtureUniverseTool,
)
from nlp_stock_prediction.orchestration.orchestration_common import (
    ReportRunPaths,
    codex_smoke_instrument,
    stable_digest,
    utc_now,
)
from nlp_stock_prediction.orchestration.report_data_modes import (
    DUMMY_SMOKE_REPORT_DATA_MODE,
    report_data_mode_metadata,
)
from nlp_stock_prediction.storage.records import ToolRunRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore


def run_codex_smoke_dummy_universe_tool(
    *,
    store: SQLiteStore,
    repo_root: Path,
    paths: ReportRunPaths,
    run_id: str,
    symbol: str,
    tool_run_id: str | None = None,
    record_tool_run: bool = True,
    tool_name: str = "run_dummy_universe_tool",
    tool_version: str = "instrument_universe.fixture.v1",
    tool_status: str = "ok",
) -> JsonObject:
    now = utc_now()
    normalized_symbol = symbol.strip().upper()
    universe_result = FixtureUniverseTool().run(
        request_id=f"instrument_universe-fixture-universe-{stable_digest(f'{run_id}:{normalized_symbol}')}",
        generated_at=now,
        run_id=run_id,
        primary_symbol=normalized_symbol,
        primary_instrument_id=f"instrument:codex:{normalized_symbol}",
    )
    universe = universe_result.universe
    artifact_id = f"artifact-instrument-universe-{stable_digest(run_id)}"
    resolved_tool_run_id = tool_run_id or f"tool-dummy-universe-{run_id}"
    if record_tool_run:
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=resolved_tool_run_id,
                run_id=run_id,
                tool_name=tool_name,
                tool_version=tool_version,
                status=tool_status,
                started_at=now,
                completed_at=now,
                inputs={
                    "symbol": symbol,
                    "universe_id": universe.request_id,
                    **report_data_mode_metadata(DUMMY_SMOKE_REPORT_DATA_MODE),
                },
                warnings=universe.warnings,
            )
        )
    ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=paths.audit_dir,
        created_at=now,
        produced_by=tool_name,
        tool_run_id=resolved_tool_run_id,
        schema_version="instrument_universe.instrument-universe.v1",
        default_metadata=report_data_mode_metadata(DUMMY_SMOKE_REPORT_DATA_MODE),
    ).write_json(
        artifact_id=artifact_id,
        artifact_type="instrument_universe",
        filename="instrument-universe.json",
        payload=universe_result.artifact_payload,
        metadata={
            "universe_id": universe.request_id,
            "instrument_ids": list(universe.instrument_ids),
            "resolution_status_counts": universe.metadata.get(
                "resolution_status_counts",
                {},
            ),
            "warnings": list(universe.warnings),
        },
    )
    for instrument_record in universe_result.instrument_records:
        store.upsert_instrument(instrument_record)
    return {
        "run_id": run_id,
        "tool_run_id": resolved_tool_run_id,
        "universe_id": universe.request_id,
        "instrument_ids": list(universe.instrument_ids),
        "artifact_id": artifact_id,
        "warnings": list(universe.warnings),
    }


def run_codex_smoke_dummy_analysis_tool(
    *,
    store: SQLiteStore,
    repo_root: Path,
    paths: ReportRunPaths,
    run_id: str,
    symbol: str,
    tool_run_id: str | None = None,
    record_tool_run: bool = True,
    tool_name: str = "run_dummy_analysis_tool",
    tool_version: str = "codex_smoke.v1",
    tool_status: str = "ok",
) -> JsonObject:
    now = utc_now()
    artifact_id = f"artifact-dummy-analysis-{stable_digest(run_id)}"
    resolved_tool_run_id = tool_run_id or f"tool-dummy-analysis-{run_id}"
    payload: JsonObject = {
        "schema_version": "dummy-analysis.v1",
        "run_id": run_id,
        "records": [
            {
                "symbol": symbol.upper(),
                "signal": "mixed",
                "summary": "Dummy analysis marks live-search evidence as the only real signal.",
                "confidence": 0.3,
            }
        ],
    }
    if record_tool_run:
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=resolved_tool_run_id,
                run_id=run_id,
                tool_name=tool_name,
                tool_version=tool_version,
                status=tool_status,
                started_at=now,
                completed_at=now,
                inputs={
                    "symbol": symbol,
                    **report_data_mode_metadata(DUMMY_SMOKE_REPORT_DATA_MODE),
                },
            )
        )
    ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=paths.audit_dir,
        created_at=now,
        produced_by=tool_name,
        tool_run_id=resolved_tool_run_id,
        schema_version="dummy-analysis.v1",
        default_metadata=report_data_mode_metadata(DUMMY_SMOKE_REPORT_DATA_MODE),
    ).write_json(
        artifact_id=artifact_id,
        artifact_type="analysis_context",
        filename="dummy-analysis.json",
        payload=payload,
        metadata={"symbol": symbol.upper()},
    )
    return {"run_id": run_id, "tool_run_id": resolved_tool_run_id, "artifact_id": artifact_id}


def upsert_codex_smoke_instrument(
    store: SQLiteStore,
    *,
    symbol: str,
    retrieved_at: datetime,
) -> None:
    instrument = codex_smoke_instrument(symbol.strip().upper(), retrieved_at)
    store.upsert_instrument(instrument_to_record(instrument, metadata={"codex_smoke_mcp": True}))


__all__ = [
    "run_codex_smoke_dummy_analysis_tool",
    "run_codex_smoke_dummy_universe_tool",
    "upsert_codex_smoke_instrument",
]
