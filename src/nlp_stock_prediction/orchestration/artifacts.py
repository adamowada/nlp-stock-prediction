"""Deterministic audit artifact writing for orchestration runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.reporting.audit import stable_json_bytes
from nlp_stock_prediction.storage.records import ArtifactRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

ArtifactType = Literal[
    "raw_snapshot",
    "normalized_evidence",
    "extraction_output",
    "analysis_context",
    "prediction_input",
    "markdown_report",
    "json_report",
    "provider_result",
    "ml_forecast",
    "instrument_universe",
    "prediction_evaluation",
    "audit_manifest",
]


@dataclass(frozen=True)
class ArtifactWriter:
    """Write stable files and return report-contract audit records."""

    base_dir: Path
    created_at: datetime
    produced_by: str

    def write_json(
        self,
        *,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        payload: JsonObject,
        record_count: int | None = None,
        metadata: JsonObject | None = None,
    ) -> AuditArtifact:
        content = stable_json_bytes(payload)
        return self._write_bytes(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
            content=content,
            record_count=record_count,
            metadata=metadata,
        )

    def write_text(
        self,
        *,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        content: str,
        record_count: int | None = None,
        metadata: JsonObject | None = None,
    ) -> AuditArtifact:
        return self._write_bytes(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
            content=content.encode("utf-8"),
            record_count=record_count,
            metadata=metadata,
        )

    def _write_bytes(
        self,
        *,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        content: bytes,
        record_count: int | None,
        metadata: JsonObject | None,
    ) -> AuditArtifact:
        path = self._resolve_artifact_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return AuditArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            path=path.as_posix(),
            created_at=self.created_at,
            produced_by=self.produced_by,
            sha256=hashlib.sha256(content).hexdigest(),
            record_count=record_count,
            metadata={} if metadata is None else metadata,
        )

    def _resolve_artifact_path(self, filename: str) -> Path:
        requested_path = Path(filename)
        if requested_path.is_absolute():
            raise ValueError("artifact filename must be relative to the artifact base directory")
        base_dir = self.base_dir.resolve()
        path = (base_dir / requested_path).resolve()
        try:
            path.relative_to(base_dir)
        except ValueError as exc:
            raise ValueError(
                "artifact filename must stay within the artifact base directory"
            ) from exc
        return path


@dataclass(frozen=True)
class ArtifactIndex:
    """Write stable artifacts and index them in research SQLite."""

    store: SQLiteStore
    repo_root: Path
    writer: ArtifactWriter
    tool_run_id: str | None
    schema_version: str

    @classmethod
    def for_directory(
        cls,
        *,
        store: SQLiteStore,
        repo_root: Path,
        base_dir: Path,
        created_at: datetime,
        produced_by: str,
        tool_run_id: str | None,
        schema_version: str,
    ) -> ArtifactIndex:
        resolved_repo_root = repo_root.resolve()
        resolved_base_dir = base_dir.resolve()
        try:
            resolved_base_dir.relative_to(resolved_repo_root)
        except ValueError as exc:
            raise ValueError(
                "artifact base directory must stay within the repository root"
            ) from exc
        return cls(
            store=store,
            repo_root=resolved_repo_root,
            writer=ArtifactWriter(
                base_dir=resolved_base_dir,
                created_at=created_at,
                produced_by=produced_by,
            ),
            tool_run_id=tool_run_id,
            schema_version=schema_version,
        )

    def write_json(
        self,
        *,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        payload: JsonObject,
        record_count: int | None = None,
        metadata: JsonObject | None = None,
    ) -> AuditArtifact:
        artifact = self.writer.write_json(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
            payload=payload,
            record_count=record_count,
            metadata=metadata,
        )
        self._record_artifact(artifact)
        return artifact

    def write_text(
        self,
        *,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        content: str,
        record_count: int | None = None,
        metadata: JsonObject | None = None,
    ) -> AuditArtifact:
        artifact = self.writer.write_text(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
            content=content,
            record_count=record_count,
            metadata=metadata,
        )
        self._record_artifact(artifact)
        return artifact

    def _record_artifact(self, artifact: AuditArtifact) -> None:
        if artifact.sha256 is None:
            raise ValueError("indexed artifacts require sha256")
        repo_root = self.repo_root.resolve()
        artifact_path = Path(artifact.path).resolve()
        self.store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact.artifact_id,
                tool_run_id=self.tool_run_id,
                artifact_type=artifact.artifact_type,
                path=artifact_path.relative_to(repo_root),
                sha256=artifact.sha256,
                schema_version=self.schema_version,
                metadata=artifact.metadata,
                created_at=artifact.created_at,
            )
        )


__all__ = ["ArtifactIndex", "ArtifactType", "ArtifactWriter"]
