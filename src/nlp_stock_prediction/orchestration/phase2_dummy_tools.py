"""Dummy Phase 2 MCP tools used by the real-agent smoke path."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts import JsonObject
from nlp_stock_prediction.orchestration.phase2_common import (
    Phase2RunPaths,
    phase2_instrument,
    stable_digest,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase3_universe import (
    build_phase3_fixture_universe,
    instrument_record_from_contract,
    phase3_universe_artifact_payload,
)
from nlp_stock_prediction.reporting.audit import write_json_artifact
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    InstrumentRecord,
    SQLiteStore,
    ToolRunRecord,
)


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
    universe = build_phase3_fixture_universe(
        request_id=f"phase3-fixture-universe-{stable_digest(f'{run_id}:{normalized_symbol}')}",
        generated_at=now,
        primary_symbol=normalized_symbol,
        primary_instrument_id=f"instrument:codex:{normalized_symbol}",
    )
    payload = phase3_universe_artifact_payload(run_id=run_id, universe=universe)
    artifact_id = f"artifact-instrument-universe-{stable_digest(run_id)}"
    path = paths.audit_dir / "instrument-universe.json"
    sha256 = write_json_artifact(path, payload)
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
    for instrument in universe.instruments:
        store.upsert_instrument(instrument_record_from_contract(instrument))
    store.record_artifact(
        ArtifactRecord(
            artifact_id=artifact_id,
            tool_run_id=tool_run_id,
            artifact_type="instrument_universe",
            path=path.relative_to(repo_root),
            sha256=sha256,
            schema_version="phase3.instrument-universe.v1",
            metadata={
                "universe_id": universe.request_id,
                "instrument_ids": list(universe.instrument_ids),
                "resolution_status_counts": universe.metadata.get(
                    "resolution_status_counts",
                    {},
                ),
                "warnings": list(universe.warnings),
            },
            created_at=now,
        )
    )
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
    path = paths.audit_dir / "dummy-analysis.json"
    sha256 = write_json_artifact(path, payload)
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
    store.record_artifact(
        ArtifactRecord(
            artifact_id=artifact_id,
            tool_run_id=tool_run_id,
            artifact_type="analysis_context",
            path=path.relative_to(repo_root),
            sha256=sha256,
            schema_version="dummy-analysis.v1",
            metadata={"symbol": symbol.upper()},
            created_at=now,
        )
    )
    return {"run_id": run_id, "artifact_id": artifact_id}


def upsert_phase2_instrument(store: SQLiteStore, *, symbol: str, retrieved_at: datetime) -> None:
    instrument = phase2_instrument(symbol.strip().upper(), retrieved_at)
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id=instrument.instrument_id,
            symbol=instrument.symbol,
            asset_class=instrument.asset_class.value,
            name=instrument.display_name,
            venue=instrument.venue,
            provider_ids=tuple(
                cast(JsonObject, item.model_dump(mode="json")) for item in instrument.provider_ids
            ),
            tradability_evidence=tuple(
                cast(JsonObject, item.model_dump(mode="json"))
                for item in instrument.tradability_evidence
            ),
            data_availability=tuple(
                cast(JsonObject, item.model_dump(mode="json"))
                for item in instrument.data_availability
            ),
            metadata={"phase2_mcp": True},
        )
    )


__all__ = [
    "run_phase2_dummy_analysis_tool",
    "run_phase2_dummy_universe_tool",
    "upsert_phase2_instrument",
]
