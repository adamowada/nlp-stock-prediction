"""Shared signal artifact policy for prediction contracts."""

from __future__ import annotations

from typing import Literal

from nlp_stock_prediction.contracts.enums import SignalArtifactFamily

SignalArtifactType = Literal[
    "market_data",
    "technical_package",
    "ml_forecast",
    "normalized_evidence",
    "analysis_context",
]

SIGNAL_ARTIFACT_TYPES_BY_FAMILY: dict[SignalArtifactFamily, frozenset[SignalArtifactType]] = {
    SignalArtifactFamily.TECHNICALS: frozenset({"market_data", "technical_package"}),
    SignalArtifactFamily.TIMESFM: frozenset({"ml_forecast", "technical_package"}),
    SignalArtifactFamily.SOCIAL: frozenset({"normalized_evidence"}),
    SignalArtifactFamily.NEWS: frozenset({"normalized_evidence"}),
    SignalArtifactFamily.FUNDAMENTALS: frozenset({"analysis_context", "normalized_evidence"}),
    SignalArtifactFamily.SECTOR_MACRO: frozenset({"analysis_context"}),
}


def allowed_signal_artifact_types(
    family: SignalArtifactFamily,
) -> frozenset[SignalArtifactType]:
    return SIGNAL_ARTIFACT_TYPES_BY_FAMILY[family]


def validate_signal_artifact_family_type(
    *,
    family: SignalArtifactFamily,
    artifact_type: str,
) -> SignalArtifactType:
    if artifact_type not in SIGNAL_ARTIFACT_TYPES_BY_FAMILY[family]:
        raise ValueError("signal artifact family does not allow artifact_type")
    return artifact_type


def legacy_signal_artifact_family_type(
    artifact_id: str,
) -> tuple[SignalArtifactFamily, SignalArtifactType]:
    normalized = artifact_id.lower()
    if "timesfm" in normalized or normalized.startswith("artifact-ml"):
        return SignalArtifactFamily.TIMESFM, "ml_forecast"
    return SignalArtifactFamily.TECHNICALS, "technical_package"


__all__ = [
    "SIGNAL_ARTIFACT_TYPES_BY_FAMILY",
    "SignalArtifactType",
    "allowed_signal_artifact_types",
    "legacy_signal_artifact_family_type",
    "validate_signal_artifact_family_type",
]
