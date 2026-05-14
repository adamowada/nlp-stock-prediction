"""Signal artifact reference assembly for report and evaluation inputs."""

from __future__ import annotations

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import SignalArtifactFamily
from nlp_stock_prediction.contracts.evaluation import SignalArtifactReference
from nlp_stock_prediction.contracts.signal_artifact_references import (
    metadata_signal_artifact_references,
    signal_artifact_reference_from_metadata,
)
from nlp_stock_prediction.contracts.signal_artifacts import (
    SignalArtifactType,
)
from nlp_stock_prediction.storage.records import ArtifactRecord


def signal_artifact_references_for_records(
    artifacts: tuple[ArtifactRecord, ...],
) -> tuple[SignalArtifactReference, ...]:
    references: list[SignalArtifactReference] = []
    seen: set[str] = set()
    for artifact in artifacts:
        reference = signal_artifact_reference_for_record(artifact)
        if reference is None or reference.artifact_id in seen:
            continue
        seen.add(reference.artifact_id)
        references.append(reference)
    return tuple(references)


def signal_artifact_reference_for_record(
    artifact: ArtifactRecord,
) -> SignalArtifactReference | None:
    family_type = signal_artifact_family_type_for_record(
        artifact_type=artifact.artifact_type,
        produced_by=artifact.produced_by,
    )
    if family_type is None:
        return None
    family, artifact_type = family_type
    source_evidence_ids = artifact.metadata.get("evidence_ids")
    return SignalArtifactReference(
        artifact_id=artifact.artifact_id,
        family=family,
        artifact_type=artifact_type,
        schema_version=artifact.schema_version,
        tool_run_id=artifact.tool_run_id,
        produced_by=artifact.produced_by,
        created_at=artifact.created_at,
        sha256=artifact.sha256,
        source_evidence_ids=(
            tuple(item for item in source_evidence_ids if isinstance(item, str) and item)
            if isinstance(source_evidence_ids, list | tuple)
            else ()
        ),
        metadata={"derived_from_run_artifact_index": True},
    )


def signal_artifact_family_type_for_record(
    *,
    artifact_type: str,
    produced_by: str | None,
) -> tuple[SignalArtifactFamily, SignalArtifactType] | None:
    normalized_producer = (produced_by or "").lower()
    if artifact_type == "market_data":
        return SignalArtifactFamily.TECHNICALS, "market_data"
    if artifact_type == "technical_package":
        return SignalArtifactFamily.TECHNICALS, "technical_package"
    if artifact_type == "ml_forecast":
        return SignalArtifactFamily.TIMESFM, "ml_forecast"
    if artifact_type == "normalized_evidence" and "social" in normalized_producer:
        return SignalArtifactFamily.SOCIAL, "normalized_evidence"
    if artifact_type == "normalized_evidence" and "news" in normalized_producer:
        return SignalArtifactFamily.NEWS, "normalized_evidence"
    if artifact_type == "analysis_context" and "fundamental" in normalized_producer:
        return SignalArtifactFamily.FUNDAMENTALS, "analysis_context"
    if artifact_type == "analysis_context" and (
        "sector" in normalized_producer or "macro" in normalized_producer
    ):
        return SignalArtifactFamily.SECTOR_MACRO, "analysis_context"
    return None


def signal_artifact_metadata(
    references: tuple[SignalArtifactReference, ...],
) -> list[JsonObject]:
    return [reference.model_dump(mode="json") for reference in references]


__all__ = [
    "metadata_signal_artifact_references",
    "signal_artifact_family_type_for_record",
    "signal_artifact_metadata",
    "signal_artifact_reference_for_record",
    "signal_artifact_reference_from_metadata",
    "signal_artifact_references_for_records",
]
