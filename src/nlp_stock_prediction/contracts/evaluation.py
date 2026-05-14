"""Prediction-quality evaluation contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    Confidence,
    ContractModel,
    JsonObject,
    NonEmptyStr,
)
from nlp_stock_prediction.contracts.enums import Direction, PredictionStatus, TimeHorizon
from nlp_stock_prediction.contracts.provenance import EvidenceReference


class EvaluationEvidenceCounts(ContractModel):
    """Evidence depth used by prediction-quality scoring."""

    supporting_source_evidence: int = Field(ge=0)
    contradicting_source_evidence: int = Field(ge=0)
    missing_source_references: int = Field(ge=0)
    technical_signal_artifacts: int = Field(default=0, ge=0)
    ml_signal_count: int = Field(default=0, ge=0)
    supporting_reference_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    contradicting_reference_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    missing_reference_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)

    @property
    def cited_source_evidence(self) -> int:
        return self.supporting_source_evidence + self.contradicting_source_evidence

    @model_validator(mode="after")
    def validate_count_alignment(self) -> EvaluationEvidenceCounts:
        if self.supporting_source_evidence != len(self.supporting_reference_ids):
            raise ValueError("supporting_source_evidence must match supporting_reference_ids")
        if self.contradicting_source_evidence != len(self.contradicting_reference_ids):
            raise ValueError("contradicting_source_evidence must match contradicting_reference_ids")
        if self.missing_source_references != len(self.missing_reference_ids):
            raise ValueError("missing_source_references must match missing_reference_ids")
        return self


class BaselineComparison(ContractModel):
    """Candidate score compared with an explicit no-edge baseline."""

    baseline_id: NonEmptyStr
    baseline_summary: NonEmptyStr
    baseline_score: Confidence
    candidate_score: Confidence
    score_delta: float = Field(ge=-1.0, le=1.0)
    verdict: Literal[
        "above_baseline",
        "near_baseline",
        "below_baseline",
        "baseline_unavailable",
    ]


class PredictionQualityLanguage(ContractModel):
    """Report-safe language metadata for evaluation artifacts."""

    purpose: Literal["prediction_quality"] = "prediction_quality"
    report_label: NonEmptyStr = "Prediction quality evaluation"
    trading_action_language: Literal["excluded"] = "excluded"
    summary: NonEmptyStr = (
        "Scores evidence support, baseline context, and uncertainty; trading guidance is out of "
        "scope."
    )


class PredictionEvaluation(ContractModel):
    """Typed score for a prediction scenario, not an outcome or trading record."""

    schema_version: NonEmptyStr = "prediction-evaluation.v1"
    evaluation_id: NonEmptyStr
    candidate_id: NonEmptyStr
    instrument_id: NonEmptyStr
    symbol: NonEmptyStr
    created_at: AwareDatetime
    horizon: TimeHorizon = TimeHorizon.UNKNOWN
    direction: Direction = Direction.UNKNOWN
    status: PredictionStatus
    score: Confidence
    baseline_comparison: BaselineComparison
    uncertainty: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    evidence_counts: EvaluationEvidenceCounts
    evidence_for: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    evidence_against: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    signal_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    quality_language: PredictionQualityLanguage = Field(default_factory=PredictionQualityLanguage)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_supported_scores_have_source_evidence(self) -> PredictionEvaluation:
        if (
            self.status == PredictionStatus.EVIDENCE_SUPPORTED
            and self.evidence_counts.supporting_source_evidence == 0
        ):
            raise ValueError(
                "evidence-supported evaluations require attributable supporting source evidence"
            )
        if not self.uncertainty:
            raise ValueError("prediction evaluations require uncertainty context")
        return self


class PredictionEvaluationArtifactPayload(ContractModel):
    """Stable JSON payload written by the Phase 4 evaluation tool."""

    schema_version: NonEmptyStr = "prediction-evaluation-artifact.v1"
    run_id: NonEmptyStr
    created_at: AwareDatetime
    evaluation: PredictionEvaluation
    source_evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)


__all__ = [
    "BaselineComparison",
    "EvaluationEvidenceCounts",
    "PredictionEvaluation",
    "PredictionEvaluationArtifactPayload",
    "PredictionQualityLanguage",
]
