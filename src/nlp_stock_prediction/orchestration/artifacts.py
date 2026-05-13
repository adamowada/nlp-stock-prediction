"""Deterministic audit artifact writing for orchestration runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from nlp_stock_prediction.contracts import AuditArtifact, JsonObject
from nlp_stock_prediction.reporting.audit import stable_json_bytes

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


__all__ = ["ArtifactType", "ArtifactWriter"]
