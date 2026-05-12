"""Conservative ML sidecar integration for technical analysis."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from nlp_stock_prediction.analysis._metrics import analysis_metric, clamp
from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    FreshnessStatus,
    MetricValue,
    TechnicalAnalysis,
    TechnicalMlSignal,
)
from nlp_stock_prediction.ml.training import EvaluationResult, TechnicalLogisticModel

_DEFAULT_LIMITATION = (
    "Experimental local technical model output; not investment advice and not a standalone "
    "recommendation input."
)
_MlSignalStatus = Literal["usable", "weak", "stale", "conflicting", "unavailable"]


def build_technical_ml_signal(
    *,
    model: TechnicalLogisticModel,
    evaluation: EvaluationResult,
    as_of: date | datetime,
    min_validation_accuracy: float = 0.52,
    min_confidence: float = 0.12,
) -> TechnicalMlSignal:
    """Map a model/evaluation artifact into a report-safe technical sidecar."""

    if not evaluation.predictions:
        return TechnicalMlSignal(
            model_hash=model.model_hash,
            dataset_hash=model.dataset_hash,
            as_of=as_of,
            feature_end=as_of,
            prediction_horizon_sessions=model.label_horizon_sessions,
            probability_positive=0.5,
            calibrated_confidence=0.0,
            signal=AnalysisSignal.UNKNOWN,
            status="unavailable",
            freshness_status=FreshnessStatus.UNKNOWN,
            validation_accuracy=evaluation.metrics.accuracy,
            validation_brier_score=evaluation.metrics.brier_score,
            warning_ids=("ml-technical-signal:no_predictions",),
            limitations=(_DEFAULT_LIMITATION,),
            metadata={
                "model_kind": model.model_kind,
                "threshold": model.threshold,
                "validation_samples": evaluation.metrics.samples,
            },
        )
    latest = max(
        evaluation.predictions, key=lambda prediction: _timestamp_key(prediction.feature_end)
    )
    probability = latest.probability
    confidence = _calibrated_confidence(probability, evaluation.metrics.accuracy)
    signal = _signal_from_probability(probability)
    warning_ids: list[str] = []
    status: _MlSignalStatus = "usable"
    if evaluation.metrics.samples <= 0 or evaluation.metrics.accuracy < min_validation_accuracy:
        status = "weak"
        warning_ids.append("ml-technical-signal:weak_validation")
    if confidence < min_confidence:
        status = "weak"
        warning_ids.append("ml-technical-signal:weak_confidence")

    return TechnicalMlSignal(
        model_hash=model.model_hash,
        dataset_hash=model.dataset_hash,
        as_of=as_of,
        feature_end=latest.feature_end,
        prediction_horizon_sessions=model.label_horizon_sessions,
        probability_positive=round(probability, 6),
        calibrated_confidence=round(confidence, 6),
        signal=signal,
        status=status,
        freshness_status=FreshnessStatus.FRESH,
        validation_accuracy=evaluation.metrics.accuracy,
        validation_brier_score=evaluation.metrics.brier_score,
        warning_ids=tuple(dict.fromkeys(warning_ids)),
        limitations=(_DEFAULT_LIMITATION,),
        metadata={
            "model_kind": model.model_kind,
            "threshold": model.threshold,
            "validation_samples": evaluation.metrics.samples,
        },
    )


def apply_technical_ml_signal(
    analysis: TechnicalAnalysis,
    signal: TechnicalMlSignal | None,
) -> TechnicalAnalysis:
    """Attach a conservative ML sidecar without overriding deterministic indicators."""

    if signal is None:
        return analysis
    metrics = (*analysis.metrics, *_ml_metrics(signal))
    assumptions = (
        *analysis.assumptions,
        "ML signal is a sidecar and cannot qualify a recommendation on its own.",
    )
    summary = (
        f"{analysis.summary} ML sidecar: {signal.signal.value} with "
        f"{signal.probability_positive:.1%} positive-return probability and "
        f"{signal.calibrated_confidence:.2f} calibrated confidence."
    )
    return analysis.model_copy(
        update={
            "summary": summary,
            "metrics": metrics,
            "assumptions": tuple(dict.fromkeys(assumptions)),
            "ml_signal": signal,
        }
    )


def _ml_metrics(signal: TechnicalMlSignal) -> tuple[MetricValue, ...]:
    metrics = [
        analysis_metric(
            "ml-positive-return-probability",
            signal.probability_positive,
            unit="probability",
            as_of=signal.as_of,
            metadata={"model_hash": signal.model_hash, "status": signal.status},
        ),
        analysis_metric(
            "ml-calibrated-confidence",
            signal.calibrated_confidence,
            unit="score",
            as_of=signal.as_of,
            metadata={"model_hash": signal.model_hash, "status": signal.status},
        ),
    ]
    if signal.validation_accuracy is not None:
        metrics.append(
            analysis_metric(
                "ml-validation-accuracy",
                signal.validation_accuracy,
                unit="score",
                as_of=signal.as_of,
                metadata={"model_hash": signal.model_hash},
            )
        )
    if signal.validation_brier_score is not None:
        metrics.append(
            analysis_metric(
                "ml-validation-brier-score",
                signal.validation_brier_score,
                unit="score",
                as_of=signal.as_of,
                metadata={"model_hash": signal.model_hash},
            )
        )
    return tuple(metrics)


def _signal_from_probability(probability: float) -> AnalysisSignal:
    if probability >= 0.56:
        return AnalysisSignal.SUPPORTS
    if probability <= 0.44:
        return AnalysisSignal.CONFLICTS
    return AnalysisSignal.MIXED


def _calibrated_confidence(probability: float, validation_accuracy: float) -> float:
    probability_edge = abs(probability - 0.5) * 2.0
    accuracy_edge = max(0.0, validation_accuracy - 0.5) * 2.0
    return clamp((probability_edge * 0.70) + (accuracy_edge * 0.30))


def _timestamp_key(value: date | datetime) -> int:
    if isinstance(value, datetime):
        return int(value.timestamp())
    return value.toordinal()


__all__ = ["apply_technical_ml_signal", "build_technical_ml_signal"]
