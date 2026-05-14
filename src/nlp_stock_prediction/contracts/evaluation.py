"""Prediction-quality evaluation contracts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    Confidence,
    ContractModel,
    JsonObject,
    NonEmptyStr,
)
from nlp_stock_prediction.contracts.enums import (
    Direction,
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
    PredictionStatus,
    PredictionType,
    SignalArtifactFamily,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.provenance import EvidenceReference


class SignalArtifactReference(ContractModel):
    """Typed reference to a signal artifact family used by a prediction."""

    artifact_id: NonEmptyStr
    family: SignalArtifactFamily
    artifact_type: Literal[
        "market_data",
        "technical_package",
        "ml_forecast",
        "normalized_evidence",
        "analysis_context",
    ]
    schema_version: str | None = None
    tool_run_id: str | None = None
    produced_by: str | None = None
    created_at: AwareDatetime | None = None
    as_of: AwareDatetime | None = None
    sha256: str | None = None
    source_evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_family_artifact_type(self) -> SignalArtifactReference:
        allowed_types = _SIGNAL_ARTIFACT_TYPES[self.family]
        if self.artifact_type not in allowed_types:
            raise ValueError("signal artifact family does not allow artifact_type")
        if self.created_at is not None and self.as_of is not None and self.as_of > self.created_at:
            raise ValueError("signal artifact as_of must be at or before created_at")
        return self


class SignalArtifactCounts(ContractModel):
    """Per-family counts for signal artifacts used during scoring."""

    technicals: int = Field(default=0, ge=0)
    timesfm: int = Field(default=0, ge=0)
    social: int = Field(default=0, ge=0)
    news: int = Field(default=0, ge=0)
    fundamentals: int = Field(default=0, ge=0)
    sector_macro: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return (
            self.technicals
            + self.timesfm
            + self.social
            + self.news
            + self.fundamentals
            + self.sector_macro
        )

    @classmethod
    def from_references(cls, references: Iterable[SignalArtifactReference]) -> SignalArtifactCounts:
        counts = {
            SignalArtifactFamily.TECHNICALS: 0,
            SignalArtifactFamily.TIMESFM: 0,
            SignalArtifactFamily.SOCIAL: 0,
            SignalArtifactFamily.NEWS: 0,
            SignalArtifactFamily.FUNDAMENTALS: 0,
            SignalArtifactFamily.SECTOR_MACRO: 0,
        }
        for reference in references:
            counts[reference.family] += 1
        return cls(
            technicals=counts[SignalArtifactFamily.TECHNICALS],
            timesfm=counts[SignalArtifactFamily.TIMESFM],
            social=counts[SignalArtifactFamily.SOCIAL],
            news=counts[SignalArtifactFamily.NEWS],
            fundamentals=counts[SignalArtifactFamily.FUNDAMENTALS],
            sector_macro=counts[SignalArtifactFamily.SECTOR_MACRO],
        )


class EvaluationEvidenceCounts(ContractModel):
    """Evidence depth used by prediction-quality scoring."""

    supporting_source_evidence: int = Field(ge=0)
    contradicting_source_evidence: int = Field(ge=0)
    missing_source_references: int = Field(ge=0)
    technical_signal_artifacts: int = Field(default=0, ge=0)
    ml_signal_count: int = Field(default=0, ge=0)
    signal_artifacts_by_family: SignalArtifactCounts = Field(default_factory=SignalArtifactCounts)
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

    @model_validator(mode="after")
    def validate_score_delta_and_verdict(self) -> BaselineComparison:
        if self.verdict == "baseline_unavailable":
            if self.score_delta != 0.0:
                raise ValueError("baseline_unavailable comparisons require zero score_delta")
            return self

        expected_delta = round(self.candidate_score - self.baseline_score, 6)
        if abs(self.score_delta - expected_delta) > 1e-6:
            raise ValueError("baseline comparison score_delta must match candidate-baseline score")
        expected_verdict = (
            "above_baseline"
            if expected_delta > 0.05
            else "below_baseline"
            if expected_delta < -0.05
            else "near_baseline"
        )
        if self.verdict != expected_verdict:
            raise ValueError("baseline comparison verdict must match score_delta")
        return self


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
    prediction_type: PredictionType = PredictionType.DIRECTIONAL
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
    signal_artifacts: tuple[SignalArtifactReference, ...] = Field(default_factory=tuple)
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
        if self.status == PredictionStatus.EVIDENCE_SUPPORTED and not self.evidence_for:
            raise ValueError("evidence-supported evaluations require evidence_for references")
        if not self.uncertainty:
            raise ValueError("prediction evaluations require uncertainty context")
        evidence_for_ids = tuple(reference.evidence_id for reference in self.evidence_for)
        evidence_against_ids = tuple(reference.evidence_id for reference in self.evidence_against)
        if self.evidence_counts.supporting_reference_ids != evidence_for_ids:
            raise ValueError(
                "prediction evaluation evidence_for must match supporting_reference_ids"
            )
        if self.evidence_counts.contradicting_reference_ids != evidence_against_ids:
            raise ValueError(
                "prediction evaluation evidence_against must match contradicting_reference_ids"
            )
        typed_ids = tuple(reference.artifact_id for reference in self.signal_artifacts)
        if len(set(typed_ids)) != len(typed_ids):
            raise ValueError("prediction evaluation signal artifact references must be unique")
        if self.signal_artifact_ids != typed_ids:
            raise ValueError(
                "prediction evaluation signal_artifact_ids must match typed signal_artifacts"
            )
        expected_signal_counts = SignalArtifactCounts.from_references(self.signal_artifacts)
        if (
            self.evidence_counts.signal_artifacts_by_family.model_dump()
            != expected_signal_counts.model_dump()
        ):
            raise ValueError(
                "prediction evaluation signal_artifacts_by_family must match signal_artifacts"
            )
        if self.evidence_counts.technical_signal_artifacts != len(typed_ids):
            raise ValueError(
                "prediction evaluation technical_signal_artifacts must match signal_artifacts"
            )
        return self


class PredictionOutcome(ContractModel):
    """Observed or unavailable result for a prediction over a fixed evaluation window."""

    schema_version: NonEmptyStr = "prediction-outcome.v1"
    outcome_id: NonEmptyStr
    candidate_id: NonEmptyStr
    instrument_id: NonEmptyStr
    symbol: NonEmptyStr
    prediction_type: PredictionType
    horizon: TimeHorizon = TimeHorizon.UNKNOWN
    evaluation_window_start: AwareDatetime
    evaluation_window_end: AwareDatetime
    status: PredictionOutcomeStatus
    observed_result: PredictionOutcomeResult | None = None
    observed_at: AwareDatetime | None = None
    result_summary: str | None = None
    result_value: float | None = None
    baseline_value: float | None = None
    outcome_evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome_shape(self) -> PredictionOutcome:
        if self.evaluation_window_end <= self.evaluation_window_start:
            raise ValueError("prediction outcome evaluation window end must be after start")
        if self.observed_at is not None and self.observed_at < self.evaluation_window_start:
            raise ValueError("prediction outcome observed_at must be inside or after the window")
        if self.status == PredictionOutcomeStatus.OBSERVED:
            if self.observed_result is None or self.observed_at is None:
                raise ValueError(
                    "observed prediction outcomes require observed_result and observed_at"
                )
            if not (self.outcome_evidence or self.artifact_ids):
                raise ValueError("observed prediction outcomes require evidence or artifacts")
        else:
            if self.observed_result is not None:
                raise ValueError(
                    "non-observed prediction outcomes must not include observed_result"
                )
            if self.observed_at is not None:
                raise ValueError("non-observed prediction outcomes must not include observed_at")
            if self.result_value is not None or self.baseline_value is not None:
                raise ValueError(
                    "non-observed prediction outcomes must not include observed values"
                )
            if self.outcome_evidence:
                raise ValueError(
                    "non-observed prediction outcomes must not include outcome_evidence"
                )
            if not self.limitations:
                raise ValueError("non-observed prediction outcomes require limitations")
        return self


class PredictionOutcomeEvaluation(ContractModel):
    """Evaluation of a prior prediction outcome against its original scenario."""

    schema_version: NonEmptyStr = "prediction-outcome-evaluation.v1"
    outcome_evaluation_id: NonEmptyStr
    outcome_id: NonEmptyStr
    candidate_id: NonEmptyStr
    instrument_id: NonEmptyStr
    symbol: NonEmptyStr
    evaluated_at: AwareDatetime
    status: PredictionOutcomeEvaluationStatus
    outcome: PredictionOutcome
    quality_score: Confidence | None = None
    baseline_comparison: BaselineComparison | None = None
    evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome_evaluation_shape(self) -> PredictionOutcomeEvaluation:
        if self.outcome_id != self.outcome.outcome_id:
            raise ValueError("outcome evaluation outcome_id must match outcome")
        if self.candidate_id != self.outcome.candidate_id:
            raise ValueError("outcome evaluation candidate_id must match outcome")
        if self.instrument_id != self.outcome.instrument_id:
            raise ValueError("outcome evaluation instrument_id must match outcome")
        if self.symbol.upper() != self.outcome.symbol.upper():
            raise ValueError("outcome evaluation symbol must match outcome")

        resolved_statuses = {
            PredictionOutcomeEvaluationStatus.CONFIRMED,
            PredictionOutcomeEvaluationStatus.MISSED,
            PredictionOutcomeEvaluationStatus.MIXED,
            PredictionOutcomeEvaluationStatus.INCONCLUSIVE,
        }
        if self.status in resolved_statuses:
            if self.outcome.status != PredictionOutcomeStatus.OBSERVED:
                raise ValueError("resolved outcome evaluations require an observed outcome")
            if (
                self.outcome.observed_at is not None
                and self.evaluated_at < self.outcome.observed_at
            ):
                raise ValueError("resolved outcome evaluations must occur after observed_at")
            if self.quality_score is None:
                raise ValueError("resolved outcome evaluations require quality_score")
            if not (
                self.evidence
                or self.artifact_ids
                or self.outcome.outcome_evidence
                or self.outcome.artifact_ids
            ):
                raise ValueError("resolved outcome evaluations require evidence or artifacts")
        else:
            if not self.limitations:
                raise ValueError("unresolved outcome evaluations require limitations")
        return self


class PredictionEvaluationArtifactPayload(ContractModel):
    """Stable JSON payload written by the Phase 4 evaluation tool."""

    schema_version: NonEmptyStr = "prediction-evaluation-artifact.v1"
    run_id: NonEmptyStr
    created_at: AwareDatetime
    evaluation: PredictionEvaluation
    source_evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)


_SIGNAL_ARTIFACT_TYPES: dict[SignalArtifactFamily, set[str]] = {
    SignalArtifactFamily.TECHNICALS: {"market_data", "technical_package"},
    SignalArtifactFamily.TIMESFM: {"ml_forecast", "technical_package"},
    SignalArtifactFamily.SOCIAL: {"normalized_evidence"},
    SignalArtifactFamily.NEWS: {"normalized_evidence"},
    SignalArtifactFamily.FUNDAMENTALS: {"analysis_context", "normalized_evidence"},
    SignalArtifactFamily.SECTOR_MACRO: {"analysis_context"},
}


__all__ = [
    "BaselineComparison",
    "EvaluationEvidenceCounts",
    "PredictionEvaluation",
    "PredictionEvaluationArtifactPayload",
    "PredictionOutcome",
    "PredictionOutcomeEvaluation",
    "PredictionQualityLanguage",
    "SignalArtifactCounts",
    "SignalArtifactReference",
]
