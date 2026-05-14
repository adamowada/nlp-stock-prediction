"""Durable execution helpers for first-class Phase 4 tools."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.orchestration.artifacts import (
    ArtifactFileTransaction,
    ArtifactIndex,
    ArtifactType,
)
from nlp_stock_prediction.storage.records import ToolRunRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE4_TOOL_VERSION = "phase4.evidence-suite.v1"
PHASE4_RUNNING_TOOL_RUN_STATUS = "running"
Phase4StoredToolRunStatus = Literal[
    "running",
    "successful",
    "partial",
    "empty",
    "skipped",
    "failed",
]


def record_tool_started(
    *,
    store: SQLiteStore,
    tool_run_id: str,
    run_id: str,
    tool_name: str,
    started_at: datetime,
    inputs: JsonObject,
    tool_version: str = PHASE4_TOOL_VERSION,
) -> None:
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=tool_name,
            tool_version=tool_version,
            status=PHASE4_RUNNING_TOOL_RUN_STATUS,
            started_at=started_at,
            inputs=inputs,
        )
    )


def record_tool_completed(
    *,
    store: SQLiteStore,
    tool_run_id: str,
    run_id: str,
    tool_name: str,
    started_at: datetime,
    completed_at: datetime,
    status: str,
    inputs: JsonObject,
    warnings: Sequence[str] = (),
    error_message: str | None = None,
    tool_version: str = PHASE4_TOOL_VERSION,
) -> None:
    normalized_status = standardize_phase4_tool_run_status(status, warnings=warnings)
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=tool_name,
            tool_version=tool_version,
            status=normalized_status,
            started_at=started_at,
            completed_at=completed_at,
            inputs=inputs,
            warnings=tuple(warnings),
            error_message=error_message,
        )
    )


@contextmanager
def safe_phase4_tool_execution(
    *,
    store: SQLiteStore,
    artifact_roots: Sequence[Path],
    tool_run_id: str,
    run_id: str,
    tool_name: str,
    started_at: datetime,
    inputs: JsonObject,
    tool_version: str = PHASE4_TOOL_VERSION,
) -> Iterator[None]:
    """Run a Phase 4 tool atomically across SQLite rows and artifact files."""

    file_transactions = tuple(
        ArtifactFileTransaction.begin(root) for root in _unique_resolved_paths(artifact_roots)
    )
    try:
        with store.transaction():
            record_tool_started(
                store=store,
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name=tool_name,
                started_at=started_at,
                inputs=inputs,
                tool_version=tool_version,
            )
            yield
    except Exception as exc:
        for file_transaction in reversed(file_transactions):
            file_transaction.rollback_new_files()
        store.delete_tool_run_outputs(tool_run_id)
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name=tool_name,
                tool_version=tool_version,
                status="failed",
                started_at=started_at,
                completed_at=_failure_completed_at(started_at),
                inputs=inputs,
                error_message=str(exc),
            )
        )
        raise


def standardize_phase4_tool_run_status(
    status: str,
    *,
    warnings: Sequence[str] = (),
) -> Phase4StoredToolRunStatus:
    """Map direct Phase 4 statuses into the run-graph vocabulary."""

    normalized = status.strip().lower()
    if normalized in {
        "running",
        "successful",
        "partial",
        "empty",
        "skipped",
        "failed",
    }:
        return cast(Phase4StoredToolRunStatus, normalized)
    if normalized == "ok":
        return "successful"
    if normalized == "warning":
        return "partial" if warnings else "empty"
    if normalized == "unavailable":
        return "empty"
    return "failed"


def write_phase4_json_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    created_at: datetime,
    produced_by: str,
    tool_run_id: str,
    schema_version: str,
    artifact_id: str,
    artifact_type: ArtifactType,
    filename: str,
    payload: JsonObject,
    record_count: int | None,
    metadata: JsonObject,
) -> Path:
    artifact = ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=artifact_dir,
        created_at=created_at,
        produced_by=produced_by,
        tool_run_id=tool_run_id,
        schema_version=schema_version,
    ).write_json(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        filename=filename,
        payload=payload,
        record_count=record_count,
        metadata=metadata,
    )
    return Path(artifact.path)


def _unique_resolved_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    unique: dict[Path, Path] = {}
    for path in paths:
        resolved = path.resolve()
        unique[resolved] = resolved
    return tuple(unique.values())


def _failure_completed_at(started_at: datetime) -> datetime:
    return datetime.now(UTC).astimezone(started_at.tzinfo or UTC)


__all__ = [
    "PHASE4_RUNNING_TOOL_RUN_STATUS",
    "PHASE4_TOOL_VERSION",
    "Phase4StoredToolRunStatus",
    "record_tool_completed",
    "record_tool_started",
    "safe_phase4_tool_execution",
    "standardize_phase4_tool_run_status",
    "write_phase4_json_artifact",
]
