"""Artifact type policy shared by orchestration writers and report assembly."""

from __future__ import annotations

from typing import Literal, get_args

ArtifactType = Literal[
    "raw_snapshot",
    "normalized_evidence",
    "extraction_output",
    "analysis_context",
    "prediction_input",
    "markdown_report",
    "json_report",
    "provider_result",
    "market_data",
    "technical_package",
    "ml_forecast",
    "instrument_universe",
    "prediction_evaluation",
    "audit_manifest",
]

ALLOWED_ARTIFACT_TYPES = frozenset(get_args(ArtifactType))
JSON_ARTIFACT_TYPES = frozenset(
    {
        "raw_snapshot",
        "normalized_evidence",
        "extraction_output",
        "analysis_context",
        "prediction_input",
        "json_report",
        "provider_result",
        "ml_forecast",
        "instrument_universe",
        "prediction_evaluation",
        "audit_manifest",
        "market_data",
        "technical_package",
    }
)
FINAL_REPORT_ARTIFACT_TYPES = frozenset({"markdown_report", "json_report", "audit_manifest"})


def source_reference_type_for_artifact(artifact_type: str) -> str:
    if artifact_type == "prediction_evaluation":
        return "prediction_evaluation"
    if artifact_type == "instrument_universe":
        return "instrument_resolution"
    return "tool_artifact"


__all__ = [
    "ALLOWED_ARTIFACT_TYPES",
    "FINAL_REPORT_ARTIFACT_TYPES",
    "JSON_ARTIFACT_TYPES",
    "ArtifactType",
    "source_reference_type_for_artifact",
]
