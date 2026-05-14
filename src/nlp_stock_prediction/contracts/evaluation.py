"""Prediction-quality evaluation contracts."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
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
from nlp_stock_prediction.contracts.signal_artifacts import (
    SignalArtifactType,
    validate_signal_artifact_family_type,
)


class SignalArtifactReference(ContractModel):
    """Typed reference to a signal artifact family used by a prediction."""

    artifact_id: NonEmptyStr
    family: SignalArtifactFamily
    artifact_type: SignalArtifactType
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
        validate_signal_artifact_family_type(
            family=self.family,
            artifact_type=self.artifact_type,
        )
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


class PredictionEvaluationTarget(ContractModel):
    """Frozen point-in-time target used by outcome evaluation and calibration."""

    schema_version: NonEmptyStr = "prediction-evaluation-target.v1"
    target_id: NonEmptyStr
    run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    instrument_id: NonEmptyStr
    symbol: NonEmptyStr
    prediction_type: PredictionType
    horizon: TimeHorizon = TimeHorizon.UNKNOWN
    direction: Direction = Direction.UNKNOWN
    report_date: date | None = None
    prediction_created_at: AwareDatetime
    point_in_time_cutoff: AwareDatetime
    evaluation_window_start: AwareDatetime
    evaluation_window_end: AwareDatetime
    candidate_snapshot: JsonObject
    baseline_comparison: BaselineComparison | None = None
    evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    signal_artifacts: tuple[SignalArtifactReference, ...] = Field(default_factory=tuple)
    report_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    source_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target_shape(self) -> PredictionEvaluationTarget:
        if self.evaluation_window_end <= self.evaluation_window_start:
            raise ValueError("evaluation target window end must be after start")
        if self.prediction_created_at > self.evaluation_window_start:
            raise ValueError("prediction target must be created before the evaluation window")
        if self.point_in_time_cutoff > self.evaluation_window_start:
            raise ValueError("point-in-time cutoff must not be after the evaluation window starts")
        if not self.candidate_snapshot:
            raise ValueError("prediction evaluation targets require a frozen candidate snapshot")
        if self.baseline_comparison is None and not self.limitations:
            raise ValueError("targets without a baseline comparison require limitations")
        _validate_unique("target evidence_ids", self.evidence_ids)
        _validate_unique("target report_artifact_ids", self.report_artifact_ids)
        _validate_unique("target source_artifact_ids", self.source_artifact_ids)
        signal_ids = tuple(reference.artifact_id for reference in self.signal_artifacts)
        _validate_unique("target signal_artifacts", signal_ids)
        for reference in self.signal_artifacts:
            if reference.as_of is not None and reference.as_of > self.point_in_time_cutoff:
                raise ValueError("signal artifact as_of must not be after point-in-time cutoff")
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
        if self.evidence_counts.technical_signal_artifacts != expected_signal_counts.technicals:
            raise ValueError(
                "prediction evaluation technical_signal_artifacts must match technical artifacts"
            )
        if (
            self.status == PredictionStatus.CONTRADICTED
            and self.evidence_counts.contradicting_source_evidence == 0
        ):
            raise ValueError("contradicted evaluations require contradicting evidence")
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
            if self.observed_at < self.evaluation_window_end:
                raise ValueError(
                    "fixed-window prediction outcomes cannot resolve before window end"
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
            _validate_outcome_evaluation_result_alignment(self.status, self.outcome)
        else:
            if (
                self.outcome.observed_at is not None
                and self.evaluated_at < self.outcome.observed_at
            ):
                raise ValueError("outcome evaluations must occur after observed_at")
            if not self.limitations:
                raise ValueError("unresolved outcome evaluations require limitations")
        return self


class PredictionOutcomeArtifactPayload(ContractModel):
    """Stable artifact payload for an observed, pending, or unavailable outcome."""

    schema_version: NonEmptyStr = "prediction-outcome-artifact.v1"
    run_id: NonEmptyStr
    created_at: AwareDatetime
    target: PredictionEvaluationTarget
    outcome: PredictionOutcome
    source_evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    market_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    outcome_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome_payload(self) -> PredictionOutcomeArtifactPayload:
        _validate_target_outcome_alignment(self.target, self.outcome)
        if (
            self.outcome.status == PredictionOutcomeStatus.OBSERVED
            and self.created_at < self.outcome.evaluation_window_end
        ):
            raise ValueError("outcome artifact cannot be created before the evaluation window ends")
        _validate_unique("outcome source_evidence_ids", self.source_evidence_ids)
        _validate_unique("outcome market_artifact_ids", self.market_artifact_ids)
        _validate_unique("outcome outcome_artifact_ids", self.outcome_artifact_ids)
        return self


class PredictionOutcomeEvaluationArtifactPayload(ContractModel):
    """Stable artifact payload for a prior prediction outcome review."""

    schema_version: NonEmptyStr = "prediction-outcome-evaluation-artifact.v1"
    run_id: NonEmptyStr
    created_at: AwareDatetime
    target: PredictionEvaluationTarget
    outcome_evaluation: PredictionOutcomeEvaluation
    baseline_comparison: BaselineComparison | None = None
    evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome_evaluation_payload(
        self,
    ) -> PredictionOutcomeEvaluationArtifactPayload:
        _validate_target_outcome_alignment(self.target, self.outcome_evaluation.outcome)
        if self.created_at < self.outcome_evaluation.evaluated_at:
            raise ValueError("outcome evaluation artifact cannot be created before evaluation")
        if self.baseline_comparison is None and not self.limitations:
            raise ValueError("outcome evaluation artifacts without baselines require limitations")
        _validate_unique("outcome evaluation evidence_ids", self.evidence_ids)
        _validate_unique("outcome evaluation artifact_ids", self.artifact_ids)
        return self


class CalibrationBin(ContractModel):
    """Reliability bin for prediction-quality calibration summaries."""

    bin_id: NonEmptyStr
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float = Field(ge=0.0, le=1.0)
    prediction_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    average_score: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_frequency: float | None = Field(default=None, ge=0.0, le=1.0)
    brier_score: float | None = Field(default=None, ge=0.0)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_bin(self) -> CalibrationBin:
        if self.upper_bound <= self.lower_bound:
            raise ValueError("calibration bin upper_bound must exceed lower_bound")
        if self.resolved_count > self.prediction_count:
            raise ValueError("calibration bin resolved_count cannot exceed prediction_count")
        if self.resolved_count == 0 and (
            self.average_score is not None
            or self.observed_frequency is not None
            or self.brier_score is not None
        ):
            raise ValueError("empty calibration bins must not report resolved metrics")
        if self.prediction_count > 0 and self.resolved_count == 0 and not self.limitations:
            raise ValueError("unresolved calibration bins require limitations")
        return self


class SignalFamilyCalibrationSummary(ContractModel):
    """Calibration summary for one signal family within a cohort."""

    family: SignalArtifactFamily
    prediction_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    signal_artifact_count: int = Field(ge=0)
    average_score: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_frequency: float | None = Field(default=None, ge=0.0, le=1.0)
    brier_score: float | None = Field(default=None, ge=0.0)
    score_delta_vs_baseline: float | None = Field(default=None, ge=-1.0, le=1.0)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_signal_family_summary(self) -> SignalFamilyCalibrationSummary:
        if self.resolved_count > self.prediction_count:
            raise ValueError("signal family resolved_count cannot exceed prediction_count")
        if self.signal_artifact_count == 0 and self.prediction_count > 0 and not self.limitations:
            raise ValueError("signal family summaries without artifacts require limitations")
        if self.resolved_count == 0 and (
            self.average_score is not None
            or self.observed_frequency is not None
            or self.brier_score is not None
            or self.score_delta_vs_baseline is not None
        ):
            raise ValueError("unresolved signal family summaries must not report metrics")
        return self


class SignalFamilyAblation(ContractModel):
    """Prediction-quality delta when one signal family is included or withheld."""

    schema_version: NonEmptyStr = "signal-family-ablation.v1"
    ablation_id: NonEmptyStr
    cohort_id: NonEmptyStr
    created_at: AwareDatetime
    family: SignalArtifactFamily
    prediction_type: PredictionType | None = None
    horizon: TimeHorizon | None = None
    included_prediction_count: int = Field(ge=0)
    excluded_prediction_count: int = Field(ge=0)
    resolved_included_count: int = Field(ge=0)
    resolved_excluded_count: int = Field(ge=0)
    included_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    excluded_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_score_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    baseline_comparison: BaselineComparison | None = None
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ablation(self) -> SignalFamilyAblation:
        if self.resolved_included_count > self.included_prediction_count:
            raise ValueError("resolved_included_count cannot exceed included_prediction_count")
        if self.resolved_excluded_count > self.excluded_prediction_count:
            raise ValueError("resolved_excluded_count cannot exceed excluded_prediction_count")
        resolved_total = self.resolved_included_count + self.resolved_excluded_count
        metrics = (
            self.included_quality_score,
            self.excluded_quality_score,
            self.quality_score_delta,
        )
        if resolved_total == 0 and any(value is not None for value in metrics):
            raise ValueError("unresolved ablations must not report metrics")
        if resolved_total == 0 and not self.limitations:
            raise ValueError("unresolved signal family ablations require limitations")
        return self


class CalibrationSummary(ContractModel):
    """Cohort-level prediction-quality calibration summary."""

    schema_version: NonEmptyStr = "calibration-summary.v1"
    calibration_id: NonEmptyStr
    cohort_id: NonEmptyStr
    created_at: AwareDatetime
    as_of: AwareDatetime
    prediction_type: PredictionType | None = None
    horizon: TimeHorizon | None = None
    sample_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    pending_count: int = Field(default=0, ge=0)
    stale_count: int = Field(default=0, ge=0)
    unavailable_count: int = Field(default=0, ge=0)
    not_evaluable_count: int = Field(default=0, ge=0)
    bins: tuple[CalibrationBin, ...] = Field(default_factory=tuple)
    brier_score: float | None = Field(default=None, ge=0.0)
    log_loss: float | None = Field(default=None, ge=0.0)
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_calibration_error: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_comparison: BaselineComparison | None = None
    signal_families: tuple[SignalFamilyCalibrationSummary, ...] = Field(default_factory=tuple)
    source_outcome_evaluation_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    source_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_calibration_summary(self) -> CalibrationSummary:
        status_total = (
            self.resolved_count
            + self.pending_count
            + self.stale_count
            + self.unavailable_count
            + self.not_evaluable_count
        )
        if status_total != self.sample_count:
            raise ValueError("calibration sample_count must equal status counts")
        if self.resolved_count == 0 and _has_calibration_metrics(self):
            raise ValueError(
                "calibration summaries without resolved outcomes must not report metrics"
            )
        if self.resolved_count == 0 and not self.limitations:
            raise ValueError("calibration summaries without resolved outcomes require limitations")
        if self.resolved_count > 0 and not self.bins:
            raise ValueError("resolved calibration summaries require reliability bins")
        _validate_unique("calibration bin ids", tuple(item.bin_id for item in self.bins))
        _validate_unique(
            "calibration source_outcome_evaluation_ids",
            self.source_outcome_evaluation_ids,
        )
        _validate_unique("calibration source_artifact_ids", self.source_artifact_ids)
        family_ids = tuple(item.family for item in self.signal_families)
        _validate_unique("calibration signal families", family_ids)
        if self.baseline_comparison is None and self.resolved_count > 0 and not self.limitations:
            raise ValueError("resolved calibration summaries without baselines require limitations")
        return self


class PredictionEvaluationArtifactPayload(ContractModel):
    """Stable JSON payload written by the Phase 4 evaluation tool."""

    schema_version: NonEmptyStr = "prediction-evaluation-artifact.v1"
    run_id: NonEmptyStr
    created_at: AwareDatetime
    evaluation: PredictionEvaluation
    source_evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)


def _validate_target_outcome_alignment(
    target: PredictionEvaluationTarget,
    outcome: PredictionOutcome,
) -> None:
    if outcome.candidate_id != target.candidate_id:
        raise ValueError("outcome candidate_id must match target")
    if outcome.instrument_id != target.instrument_id:
        raise ValueError("outcome instrument_id must match target")
    if outcome.symbol.upper() != target.symbol.upper():
        raise ValueError("outcome symbol must match target")
    if outcome.prediction_type != target.prediction_type:
        raise ValueError("outcome prediction_type must match target")
    if outcome.horizon != target.horizon:
        raise ValueError("outcome horizon must match target")
    if outcome.evaluation_window_start != target.evaluation_window_start:
        raise ValueError("outcome evaluation_window_start must match target")
    if outcome.evaluation_window_end != target.evaluation_window_end:
        raise ValueError("outcome evaluation_window_end must match target")


def _validate_unique(label: str, values: tuple[object, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _validate_outcome_evaluation_result_alignment(
    status: PredictionOutcomeEvaluationStatus,
    outcome: PredictionOutcome,
) -> None:
    result = outcome.observed_result
    if status == PredictionOutcomeEvaluationStatus.CONFIRMED and result not in {
        PredictionOutcomeResult.SUPPORTED,
        PredictionOutcomeResult.NEUTRAL,
    }:
        raise ValueError("confirmed outcome evaluations require a supported or neutral result")
    if status == PredictionOutcomeEvaluationStatus.MISSED and result not in {
        PredictionOutcomeResult.NOT_SUPPORTED,
        PredictionOutcomeResult.CONTRADICTED,
    }:
        raise ValueError(
            "missed outcome evaluations require a not-supported or contradicted result"
        )
    if status == PredictionOutcomeEvaluationStatus.MIXED and result not in {
        PredictionOutcomeResult.MIXED,
        PredictionOutcomeResult.NEUTRAL,
    }:
        raise ValueError("mixed outcome evaluations require a mixed or neutral result")


def _has_calibration_metrics(summary: CalibrationSummary) -> bool:
    return any(
        value is not None
        for value in (
            summary.brier_score,
            summary.log_loss,
            summary.accuracy,
            summary.expected_calibration_error,
        )
    )


__all__ = [
    "BaselineComparison",
    "CalibrationBin",
    "CalibrationSummary",
    "EvaluationEvidenceCounts",
    "PredictionEvaluation",
    "PredictionEvaluationArtifactPayload",
    "PredictionEvaluationTarget",
    "PredictionOutcome",
    "PredictionOutcomeArtifactPayload",
    "PredictionOutcomeEvaluation",
    "PredictionOutcomeEvaluationArtifactPayload",
    "PredictionQualityLanguage",
    "SignalArtifactCounts",
    "SignalArtifactReference",
    "SignalFamilyAblation",
    "SignalFamilyCalibrationSummary",
]
