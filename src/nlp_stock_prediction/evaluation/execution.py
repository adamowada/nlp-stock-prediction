"""Atomic execution wrapper for evaluation artifact writers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.orchestration.research_execution import safe_research_tool_execution
from nlp_stock_prediction.storage.sqlite import SQLiteStore


@contextmanager
def evaluation_artifact_execution(
    *,
    store: SQLiteStore,
    artifact_roots: Sequence[Path],
    tool_run_id: str,
    run_id: str,
    tool_name: str,
    tool_version: str,
    started_at: datetime,
    inputs: JsonObject,
) -> Iterator[None]:
    """Run an evaluation writer atomically across tool-run rows and artifact files."""

    with safe_research_tool_execution(
        store=store,
        artifact_roots=artifact_roots,
        tool_run_id=tool_run_id,
        run_id=run_id,
        tool_name=tool_name,
        tool_version=tool_version,
        started_at=started_at,
        inputs=inputs,
    ):
        yield


__all__ = ["evaluation_artifact_execution"]
