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
    FreshnessStatus,
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
    PredictionStatus,
    PredictionType,
    RetrievalMethod,
    SignalArtifactFamily,
    SourceKind,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.provenance import EvidenceReference
from nlp_stock_prediction.contracts.signal_artifacts import (
    SignalArtifactType,
    validate_signal_artifact_family_type,
)

type Phase7LiveDataMode = Literal["live"]
type EvidenceAgeStatus = Literal[
    "fresh",
    "aged_out",
    "stale",
    "missing",
    "unknown",
    "superseded",
    "provider_replaced",
    "malformed",
]
type ArtifactFreshnessReviewStatus = Literal[
    "fresh",
    "stale",
    "missing",
    "unknown",
    "malformed",
    "hash_mismatch",
    "aged_out",
    "superseded",
    "provider_replaced",
]
type SourceReliabilityRating = Literal["high", "medium", "low", "unknown", "unavailable"]
type ProviderFamily = Literal[
    "market_data",
    "news",
    "social",
    "fundamentals",
    "macro",
    "scraping",
    "unknown",
]
type ProviderCompatibilityStatus = Literal[
    "compatible",
    "compatible_with_limitations",
    "incompatible",
    "not_evaluable",
    "unknown",
]
type CalibrationDriftStatus = Literal[
    "stable",
    "watch",
    "degraded",
    "improved",
    "inconclusive",
    "insufficient_history",
    "not_evaluable",
]

_RESOLVED_OUTCOME_EVALUATION_STATUSES = frozenset(
    {
        PredictionOutcomeEvaluationStatus.CONFIRMED,
        PredictionOutcomeEvaluationStatus.MISSED,
        PredictionOutcomeEvaluationStatus.MIXED,
        PredictionOutcomeEvaluationStatus.INCONCLUSIVE,
    }
)
_RESOLVED_DRIFT_STATUSES = frozenset({"stable", "watch", "degraded", "improved"})
_REPORT_COUPLING_MARKERS = frozenset(
    {
        "inline_report_calculation",
        "report_calculation",
        "candidate_score_adjustment",
        "trading_performance",
        "pnl",
    }
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


class SourceReliabilityNote(ContractModel):
    """Auditable reliability note for one source evidence item."""

    schema_version: NonEmptyStr = "source-reliability-note.v1"
    note_id: NonEmptyStr
    evidence_id: NonEmptyStr
    provider: NonEmptyStr
    source_type: SourceKind
    retrieval_method: RetrievalMethod
    retrieved_at: AwareDatetime
    observed_at: AwareDatetime | None = None
    source_url: str | None = None
    permalink: str | None = None
    raw_identifier: str | None = None
    raw_snapshot_id: str | None = None
    freshness_status: FreshnessStatus
    extraction_confidence: Confidence | None = None
    reliability: SourceReliabilityRating
    evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    related_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    report_data_mode: Phase7LiveDataMode = "live"
    provider_mode: Phase7LiveDataMode = "live"
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source_reliability_note(self) -> SourceReliabilityNote:
        if self.observed_at is not None and self.observed_at > self.retrieved_at:
            raise ValueError("source reliability observed_at must not be after retrieved_at")
        if _external_source_requires_trace(self.source_type, self.retrieval_method) and not (
            self.source_url or self.permalink or self.raw_identifier or self.raw_snapshot_id
        ):
            raise ValueError("external source reliability notes require traceability")
        if self.reliability in {"unknown", "unavailable", "low"} and not self.limitations:
            raise ValueError("limited source reliability notes require limitations")
        _validate_unique("source reliability evidence_ids", self.evidence_ids)
        _validate_unique("source reliability related_artifact_ids", self.related_artifact_ids)
        _validate_phase7_metadata(self.metadata)
        return self


class EvidenceAgingRecord(ContractModel):
    """Structured review of how one prior evidence item aged across report runs."""

    schema_version: NonEmptyStr = "evidence-aging-record.v1"
    aging_record_id: NonEmptyStr
    evidence_id: NonEmptyStr
    provider: NonEmptyStr
    source_type: SourceKind
    retrieved_at: AwareDatetime | None = None
    published_at: AwareDatetime | None = None
    reviewed_at: AwareDatetime
    age_status: EvidenceAgeStatus
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    source_reliability_note_id: str | None = None
    source_artifact_id: str | None = None
    replacement_provider: str | None = None
    replacement_evidence_id: str | None = None
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    report_data_mode: Phase7LiveDataMode = "live"
    provider_mode: Phase7LiveDataMode = "live"
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evidence_aging_record(self) -> EvidenceAgingRecord:
        if (
            self.published_at is not None
            and self.retrieved_at is not None
            and self.published_at > self.retrieved_at
        ):
            raise ValueError("evidence aging published_at must not be after retrieved_at")
        if self.retrieved_at is not None and self.retrieved_at > self.reviewed_at:
            raise ValueError("evidence aging retrieved_at must not be after reviewed_at")
        if self.age_status != "fresh" and not self.limitations:
            raise ValueError("non-fresh evidence aging records require limitations")
        if self.age_status == "provider_replaced" and not (
            self.replacement_provider or self.replacement_evidence_id
        ):
            raise ValueError("provider-replaced evidence requires a replacement reference")
        _validate_phase7_metadata(self.metadata)
        return self


class ArtifactFreshnessReview(ContractModel):
    """Structured freshness review for an artifact used by evaluation hardening."""

    schema_version: NonEmptyStr = "artifact-freshness-review.v1"
    freshness_review_id: NonEmptyStr
    artifact_id: NonEmptyStr
    artifact_type: NonEmptyStr
    provider: NonEmptyStr | None = None
    produced_by: str | None = None
    reviewed_at: AwareDatetime
    created_at: AwareDatetime | None = None
    as_of: AwareDatetime | None = None
    observed_at: AwareDatetime | None = None
    freshness_status: ArtifactFreshnessReviewStatus
    sha256: str | None = None
    expected_sha256: str | None = None
    source_evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    source_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    report_data_mode: Phase7LiveDataMode = "live"
    provider_mode: Phase7LiveDataMode = "live"
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_artifact_freshness_review(self) -> ArtifactFreshnessReview:
        if self.created_at is not None and self.created_at > self.reviewed_at:
            raise ValueError("artifact freshness created_at must not be after reviewed_at")
        if self.as_of is not None and self.created_at is not None and self.as_of > self.created_at:
            raise ValueError("artifact freshness as_of must not be after created_at")
        if self.observed_at is not None and self.observed_at > self.reviewed_at:
            raise ValueError("artifact freshness observed_at must not be after reviewed_at")
        if self.freshness_status == "fresh" and self.as_of is None and self.observed_at is None:
            raise ValueError("fresh artifact reviews require as_of or observed_at")
        if self.freshness_status != "fresh" and not self.limitations:
            raise ValueError("non-fresh artifact reviews require limitations")
        if self.freshness_status == "hash_mismatch" and not (self.sha256 and self.expected_sha256):
            raise ValueError("hash-mismatch artifact reviews require actual and expected hashes")
        _validate_unique("artifact freshness source_evidence_ids", self.source_evidence_ids)
        _validate_unique("artifact freshness source_artifact_ids", self.source_artifact_ids)
        _validate_phase7_metadata(self.metadata)
        return self


class ProviderCompatibilityNote(ContractModel):
    """Compatibility check for replacing one provider with another."""

    schema_version: NonEmptyStr = "provider-compatibility-note.v1"
    compatibility_note_id: NonEmptyStr
    provider_family: ProviderFamily
    source_provider: NonEmptyStr
    replacement_provider: NonEmptyStr
    checked_at: AwareDatetime
    compatibility_status: ProviderCompatibilityStatus
    required_fields: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    preserved_fields: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    missing_fields: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    expected_artifact_type: str | None = None
    replacement_artifact_type: str | None = None
    source_schema_version: str | None = None
    replacement_schema_version: str | None = None
    source_reliability_note_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    report_data_mode: Phase7LiveDataMode = "live"
    provider_mode: Phase7LiveDataMode = "live"
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provider_compatibility_note(self) -> ProviderCompatibilityNote:
        _validate_unique("provider compatibility required_fields", self.required_fields)
        _validate_unique("provider compatibility preserved_fields", self.preserved_fields)
        _validate_unique("provider compatibility missing_fields", self.missing_fields)
        _validate_unique(
            "provider compatibility source_reliability_note_ids",
            self.source_reliability_note_ids,
        )
        if self.compatibility_status == "compatible":
            if self.missing_fields:
                raise ValueError("compatible provider notes must not include missing_fields")
            missing_required = tuple(
                field for field in self.required_fields if field not in self.preserved_fields
            )
            if missing_required:
                raise ValueError("compatible provider notes must preserve required_fields")
        elif not self.limitations:
            raise ValueError("limited provider compatibility notes require limitations")
        if self.missing_fields and self.compatibility_status == "compatible":
            raise ValueError("compatible provider notes must not include missing_fields")
        _validate_phase7_metadata(self.metadata)
        return self


class ProviderReplacementPlaybook(ContractModel):
    """Operational playbook for provider replacement without losing provenance."""

    schema_version: NonEmptyStr = "provider-replacement-playbook.v1"
    playbook_id: NonEmptyStr
    provider_family: ProviderFamily
    source_provider: NonEmptyStr
    replacement_provider: NonEmptyStr
    created_at: AwareDatetime
    compatibility_notes: tuple[ProviderCompatibilityNote, ...] = Field(default_factory=tuple)
    required_provenance_fields: tuple[NonEmptyStr, ...]
    artifact_schema_versions: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    credential_requirements: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    unsupported_modes: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    report_data_mode: Phase7LiveDataMode = "live"
    provider_mode: Phase7LiveDataMode = "live"
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provider_replacement_playbook(self) -> ProviderReplacementPlaybook:
        if not self.compatibility_notes:
            raise ValueError("provider replacement playbooks require compatibility notes")
        _validate_unique(
            "provider replacement compatibility notes",
            tuple(note.compatibility_note_id for note in self.compatibility_notes),
        )
        _validate_unique(
            "provider replacement required_provenance_fields",
            self.required_provenance_fields,
        )
        _validate_unique(
            "provider replacement artifact_schema_versions",
            self.artifact_schema_versions,
        )
        _validate_unique(
            "provider replacement credential_requirements",
            self.credential_requirements,
        )
        for note in self.compatibility_notes:
            if note.provider_family != self.provider_family:
                raise ValueError("provider playbook note family must match playbook")
            if note.source_provider != self.source_provider:
                raise ValueError("provider playbook note source_provider must match playbook")
            if note.replacement_provider != self.replacement_provider:
                raise ValueError("provider playbook note replacement_provider must match playbook")
        if not any(
            note.compatibility_status in {"compatible", "compatible_with_limitations"}
            for note in self.compatibility_notes
        ):
            raise ValueError("compatible playbooks require at least one compatible provider note")
        if (
            any(
                note.compatibility_status == "compatible_with_limitations"
                for note in self.compatibility_notes
            )
            and not self.limitations
        ):
            raise ValueError("limited provider replacement playbooks require limitations")
        _validate_phase7_metadata(self.metadata)
        return self


class OutcomeReviewSummary(ContractModel):
    """Cross-run summary of a prior outcome review and its aging context."""

    schema_version: NonEmptyStr = "outcome-review-summary.v1"
    summary_id: NonEmptyStr
    run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    instrument_id: NonEmptyStr
    symbol: NonEmptyStr
    prediction_type: PredictionType
    horizon: TimeHorizon = TimeHorizon.UNKNOWN
    direction: Direction = Direction.UNKNOWN
    created_at: AwareDatetime
    prior_run_id: str | None = None
    outcome_id: NonEmptyStr
    outcome_evaluation_id: NonEmptyStr
    outcome_status: PredictionOutcomeStatus
    outcome_evaluation_status: PredictionOutcomeEvaluationStatus
    quality_score: Confidence | None = None
    outcome_evidence_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    evidence_aging_records: tuple[EvidenceAgingRecord, ...] = Field(default_factory=tuple)
    artifact_freshness_reviews: tuple[ArtifactFreshnessReview, ...] = Field(default_factory=tuple)
    source_reliability_notes: tuple[SourceReliabilityNote, ...] = Field(default_factory=tuple)
    source_calibration_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    report_data_mode: Phase7LiveDataMode = "live"
    provider_mode: Phase7LiveDataMode = "live"
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome_review_summary(self) -> OutcomeReviewSummary:
        resolved = self.outcome_evaluation_status in _RESOLVED_OUTCOME_EVALUATION_STATUSES
        if resolved:
            if self.outcome_status != PredictionOutcomeStatus.OBSERVED:
                raise ValueError("resolved outcome summaries require observed outcomes")
            if self.quality_score is None:
                raise ValueError("resolved outcome summaries require quality_score")
            if not (
                self.outcome_evidence_ids
                or self.artifact_ids
                or self.evidence_aging_records
                or self.artifact_freshness_reviews
            ):
                raise ValueError("resolved outcome summaries require evidence or artifacts")
        else:
            if self.quality_score is not None:
                raise ValueError("unresolved outcome summaries must not report quality_score")
            if not self.limitations:
                raise ValueError("unresolved outcome summaries require limitations")
        _validate_unique("outcome summary outcome_evidence_ids", self.outcome_evidence_ids)
        _validate_unique("outcome summary artifact_ids", self.artifact_ids)
        _validate_unique(
            "outcome summary source_calibration_artifact_ids",
            self.source_calibration_artifact_ids,
        )
        _validate_unique(
            "outcome summary evidence aging records",
            tuple(record.aging_record_id for record in self.evidence_aging_records),
        )
        _validate_unique(
            "outcome summary artifact freshness reviews",
            tuple(review.freshness_review_id for review in self.artifact_freshness_reviews),
        )
        _validate_unique(
            "outcome summary source reliability notes",
            tuple(note.note_id for note in self.source_reliability_notes),
        )
        _validate_phase7_metadata(self.metadata)
        return self


class CalibrationDriftCheck(ContractModel):
    """Separate audit artifact describing calibration movement between cohorts."""

    schema_version: NonEmptyStr = "calibration-drift-check.v1"
    drift_check_id: NonEmptyStr
    cohort_id: NonEmptyStr
    created_at: AwareDatetime
    as_of: AwareDatetime
    prior_calibration_id: str | None = None
    current_calibration_id: str | None = None
    prediction_type: PredictionType | None = None
    horizon: TimeHorizon | None = None
    signal_family: SignalArtifactFamily | None = None
    drift_status: CalibrationDriftStatus
    metric_deltas: JsonObject = Field(default_factory=dict)
    source_calibration_artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    source_outcome_evaluation_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    provider_compatibility_note_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    evidence_aging_record_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    artifact_freshness_review_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    limitations: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    report_data_mode: Phase7LiveDataMode = "live"
    provider_mode: Phase7LiveDataMode = "live"
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_calibration_drift_check(self) -> CalibrationDriftCheck:
        if self.created_at < self.as_of:
            raise ValueError("calibration drift created_at must not be before as_of")
        if self.drift_status in _RESOLVED_DRIFT_STATUSES:
            if not self.prior_calibration_id or not self.current_calibration_id:
                raise ValueError("resolved calibration drift requires prior and current summaries")
            if not self.metric_deltas:
                raise ValueError("resolved calibration drift requires metric_deltas")
            if len(self.source_calibration_artifact_ids) < 2:
                raise ValueError("resolved calibration drift requires source calibration artifacts")
        else:
            if self.metric_deltas:
                raise ValueError("unresolved calibration drift must not report metric_deltas")
            if not self.limitations:
                raise ValueError("unresolved calibration drift requires limitations")
        _validate_unique(
            "calibration drift source_calibration_artifact_ids",
            self.source_calibration_artifact_ids,
        )
        _validate_unique(
            "calibration drift source_outcome_evaluation_ids",
            self.source_outcome_evaluation_ids,
        )
        _validate_unique(
            "calibration drift provider_compatibility_note_ids",
            self.provider_compatibility_note_ids,
        )
        _validate_unique(
            "calibration drift evidence_aging_record_ids",
            self.evidence_aging_record_ids,
        )
        _validate_unique(
            "calibration drift artifact_freshness_review_ids",
            self.artifact_freshness_review_ids,
        )
        _validate_calibration_drift_metadata(self.metadata)
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


def _external_source_requires_trace(
    source_type: SourceKind,
    retrieval_method: RetrievalMethod,
) -> bool:
    return not (
        source_type == SourceKind.INTERNAL_ANALYSIS
        or retrieval_method in {RetrievalMethod.DERIVED, RetrievalMethod.LLM}
    )


def _validate_phase7_metadata(metadata: JsonObject) -> None:
    _validate_metadata_key_policy(metadata)


def _validate_calibration_drift_metadata(metadata: JsonObject) -> None:
    if _metadata_contains_report_coupling_marker(metadata):
        raise ValueError(
            "calibration drift artifacts must remain separate from report calculations"
        )
    _validate_phase7_metadata(metadata)


def _validate_metadata_key_policy(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _REPORT_COUPLING_MARKERS:
                raise ValueError("Phase 7 metadata must not include report-coupled fields")
            _validate_metadata_key_policy(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _validate_metadata_key_policy(item)


def _metadata_contains_report_coupling_marker(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key in _REPORT_COUPLING_MARKERS or _metadata_contains_report_coupling_marker(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_metadata_contains_report_coupling_marker(item) for item in value)
    return False


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
    "ArtifactFreshnessReview",
    "BaselineComparison",
    "CalibrationBin",
    "CalibrationDriftCheck",
    "CalibrationSummary",
    "EvaluationEvidenceCounts",
    "EvidenceAgingRecord",
    "OutcomeReviewSummary",
    "PredictionEvaluation",
    "PredictionEvaluationArtifactPayload",
    "PredictionEvaluationTarget",
    "PredictionOutcome",
    "PredictionOutcomeArtifactPayload",
    "PredictionOutcomeEvaluation",
    "PredictionOutcomeEvaluationArtifactPayload",
    "PredictionQualityLanguage",
    "ProviderCompatibilityNote",
    "ProviderReplacementPlaybook",
    "SignalArtifactCounts",
    "SignalArtifactReference",
    "SignalFamilyAblation",
    "SignalFamilyCalibrationSummary",
    "SourceReliabilityNote",
]
