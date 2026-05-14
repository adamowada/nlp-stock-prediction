"""Signal artifact reference helpers shared by contracts and orchestration."""

from __future__ import annotations

from pydantic import ValidationError

from nlp_stock_prediction.contracts.evaluation import SignalArtifactReference
from nlp_stock_prediction.contracts.signal_artifacts import (
    legacy_signal_artifact_family_type,
)


def legacy_signal_artifact_reference(artifact_id: str) -> SignalArtifactReference:
    family, artifact_type = legacy_signal_artifact_family_type(artifact_id)
    return SignalArtifactReference(
        artifact_id=artifact_id,
        family=family,
        artifact_type=artifact_type,
        metadata={"legacy_flat_reference": True},
    )


def signal_artifact_reference_from_metadata(
    value: object,
) -> SignalArtifactReference | None:
    if not isinstance(value, dict):
        return None
    try:
        return SignalArtifactReference.model_validate(value)
    except ValidationError:
        return None


def metadata_signal_artifact_references(
    value: object,
) -> tuple[SignalArtifactReference, ...]:
    if not isinstance(value, list | tuple):
        return ()
    references: list[SignalArtifactReference] = []
    for item in value:
        reference = signal_artifact_reference_from_metadata(item)
        if reference is not None:
            references.append(reference)
    return tuple(references)


__all__ = [
    "legacy_signal_artifact_reference",
    "metadata_signal_artifact_references",
    "signal_artifact_reference_from_metadata",
]
