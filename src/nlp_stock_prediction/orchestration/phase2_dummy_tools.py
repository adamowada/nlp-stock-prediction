"""Dummy Phase 2 MCP tools used by the real-agent smoke path."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.instruments.registry import instrument_to_record
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase2_common import (
    Phase2RunPaths,
    phase2_instrument,
    stable_digest,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase3_universe import (
    Phase3FixtureUniverseTool,
)
from nlp_stock_prediction.storage.records import ToolRunRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore


def run_phase2_dummy_universe_tool(
    *,
    store: SQLiteStore,
    repo_root: Path,
    paths: Phase2RunPaths,
    run_id: str,
    symbol: str,
) -> JsonObject:
    now = utc_now()
    normalized_symbol = symbol.strip().upper()
    universe_result = Phase3FixtureUniverseTool().run(
        request_id=f"phase3-fixture-universe-{stable_digest(f'{run_id}:{normalized_symbol}')}",
        generated_at=now,
        run_id=run_id,
        primary_symbol=normalized_symbol,
        primary_instrument_id=f"instrument:codex:{normalized_symbol}",
    )
    universe = universe_result.universe
    artifact_id = f"artifact-instrument-universe-{stable_digest(run_id)}"
    tool_run_id = f"tool-dummy-universe-{run_id}"
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name="run_dummy_universe_tool",
            tool_version="phase3.fixture.v1",
            status="ok",
            started_at=now,
            completed_at=now,
            inputs={"symbol": symbol, "universe_id": universe.request_id},
            warnings=universe.warnings,
        )
    )
    ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=paths.audit_dir,
        created_at=now,
        produced_by="run_dummy_universe_tool",
        tool_run_id=tool_run_id,
        schema_version="phase3.instrument-universe.v1",
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
        "tool_run_id": tool_run_id,
        "universe_id": universe.request_id,
        "instrument_ids": list(universe.instrument_ids),
        "artifact_id": artifact_id,
        "warnings": list(universe.warnings),
    }


def run_phase2_dummy_analysis_tool(
    *,
    store: SQLiteStore,
    repo_root: Path,
    paths: Phase2RunPaths,
    run_id: str,
    symbol: str,
) -> JsonObject:
    now = utc_now()
    artifact_id = f"artifact-dummy-analysis-{stable_digest(run_id)}"
    tool_run_id = f"tool-dummy-analysis-{run_id}"
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
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name="run_dummy_analysis_tool",
            tool_version="phase2.v1",
            status="ok",
            started_at=now,
            completed_at=now,
            inputs={"symbol": symbol},
        )
    )
    ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=paths.audit_dir,
        created_at=now,
        produced_by="run_dummy_analysis_tool",
        tool_run_id=tool_run_id,
        schema_version="dummy-analysis.v1",
    ).write_json(
        artifact_id=artifact_id,
        artifact_type="analysis_context",
        filename="dummy-analysis.json",
        payload=payload,
        metadata={"symbol": symbol.upper()},
    )
    return {"run_id": run_id, "artifact_id": artifact_id}


def upsert_phase2_instrument(store: SQLiteStore, *, symbol: str, retrieved_at: datetime) -> None:
    instrument = phase2_instrument(symbol.strip().upper(), retrieved_at)
    store.upsert_instrument(instrument_to_record(instrument, metadata={"phase2_mcp": True}))


__all__ = [
    "run_phase2_dummy_analysis_tool",
    "run_phase2_dummy_universe_tool",
    "upsert_phase2_instrument",
]
