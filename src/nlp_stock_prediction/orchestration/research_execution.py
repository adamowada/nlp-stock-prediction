"""Durable execution helpers for first-class Research Stage tools."""

from __future__ import annotations

import hashlib
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
from nlp_stock_prediction.orchestration.report_data_modes import (
    report_data_mode_metadata_for_run_id,
)
from nlp_stock_prediction.storage.records import ToolRunRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

RESEARCH_TOOL_VERSION = "research.evidence-suite.v1"
RESEARCH_RUNNING_TOOL_RUN_STATUS = "running"
StoredResearchToolRunStatus = Literal[
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
    tool_version: str = RESEARCH_TOOL_VERSION,
) -> None:
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=tool_name,
            tool_version=tool_version,
            status=RESEARCH_RUNNING_TOOL_RUN_STATUS,
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
    tool_version: str = RESEARCH_TOOL_VERSION,
) -> None:
    normalized_status = standardize_research_tool_run_status(status, warnings=warnings)
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=tool_name,
            tool_version=tool_version,
            status=normalized_status,
            started_at=started_at,
            completed_at=completed_at,
            inputs=_with_run_mode_metadata(store, run_id, inputs),
            warnings=tuple(warnings),
            error_message=error_message,
        )
    )


@contextmanager
def safe_research_tool_execution(
    *,
    store: SQLiteStore,
    artifact_roots: Sequence[Path],
    tool_run_id: str,
    run_id: str,
    tool_name: str,
    started_at: datetime,
    inputs: JsonObject,
    tool_version: str = RESEARCH_TOOL_VERSION,
) -> Iterator[None]:
    """Run a Research Stage tool atomically across SQLite rows and artifact files."""

    file_transactions = tuple(
        ArtifactFileTransaction.begin(root) for root in _unique_resolved_paths(artifact_roots)
    )
    previous_tool_run = store.get_tool_run(tool_run_id)
    try:
        with store.transaction():
            enriched_inputs = _with_run_mode_metadata(store, run_id, inputs)
            record_tool_started(
                store=store,
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name=tool_name,
                started_at=started_at,
                inputs=enriched_inputs,
                tool_version=tool_version,
            )
            yield
    except Exception as exc:
        rollback_errors: list[str] = []
        for file_transaction in reversed(file_transactions):
            try:
                file_transaction.rollback_new_files()
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        completed_at = _failure_completed_at(started_at)
        failed_tool_run_id = _failed_tool_run_id(
            tool_run_id=tool_run_id,
            previous_exists=previous_tool_run is not None
            and previous_tool_run.status != RESEARCH_RUNNING_TOOL_RUN_STATUS,
            completed_at=completed_at,
            error_message=str(exc),
        )
        error_message = str(exc)
        if rollback_errors:
            error_message = f"{error_message}; artifact rollback errors: " + "; ".join(
                dict.fromkeys(rollback_errors)
            )
        store.record_tool_run(
            ToolRunRecord(
                tool_run_id=failed_tool_run_id,
                run_id=run_id,
                tool_name=tool_name,
                tool_version=tool_version,
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                inputs=_with_run_mode_metadata(store, run_id, inputs),
                error_message=error_message,
            )
        )
        raise
    else:
        for file_transaction in file_transactions:
            file_transaction.cleanup()


def standardize_research_tool_run_status(
    status: str,
    *,
    warnings: Sequence[str] = (),
) -> StoredResearchToolRunStatus:
    """Map direct Research Stage statuses into the run-graph vocabulary."""

    normalized = status.strip().lower()
    if normalized in {
        "running",
        "successful",
        "partial",
        "empty",
        "skipped",
        "failed",
    }:
        return cast(StoredResearchToolRunStatus, normalized)
    if normalized == "ok":
        return "successful"
    if normalized == "warning":
        return "partial" if warnings else "empty"
    if normalized == "unavailable":
        return "empty"
    return "failed"


def write_research_json_artifact(
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
        default_metadata=_metadata_for_tool_run(store, tool_run_id),
    ).write_json(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        filename=filename,
        payload=payload,
        record_count=record_count,
        metadata=metadata,
    )
    return Path(artifact.path)


def _with_run_mode_metadata(store: SQLiteStore, run_id: str, inputs: JsonObject) -> JsonObject:
    return {**inputs, **report_data_mode_metadata_for_run_id(store, run_id)}


def _metadata_for_tool_run(store: SQLiteStore, tool_run_id: str) -> JsonObject:
    tool_run = store.get_tool_run(tool_run_id)
    if tool_run is None or tool_run.run_id is None:
        return {}
    return report_data_mode_metadata_for_run_id(store, tool_run.run_id)


def _unique_resolved_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    unique: dict[Path, Path] = {}
    for path in paths:
        resolved = path.resolve()
        unique[resolved] = resolved
    return tuple(unique.values())


def _failure_completed_at(started_at: datetime) -> datetime:
    return datetime.now(UTC).astimezone(started_at.tzinfo or UTC)


def _failed_tool_run_id(
    *,
    tool_run_id: str,
    previous_exists: bool,
    completed_at: datetime,
    error_message: str,
) -> str:
    if not previous_exists:
        return tool_run_id
    digest = hashlib.sha256(
        f"{tool_run_id}|{completed_at.isoformat()}|{error_message}".encode()
    ).hexdigest()[:12]
    return f"{tool_run_id}-failed-{digest}"


__all__ = [
    "RESEARCH_RUNNING_TOOL_RUN_STATUS",
    "RESEARCH_TOOL_VERSION",
    "StoredResearchToolRunStatus",
    "record_tool_completed",
    "record_tool_started",
    "safe_research_tool_execution",
    "standardize_research_tool_run_status",
    "write_research_json_artifact",
]
