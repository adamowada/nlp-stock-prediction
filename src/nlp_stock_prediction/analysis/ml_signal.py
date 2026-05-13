"""Conservative ML sidecar integration for technical analysis."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from nlp_stock_prediction.analysis._metrics import analysis_metric, clamp
from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    FreshnessStatus,
    MetricValue,
    TechnicalAnalysis,
    TechnicalMlSignal,
)
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.ml.timesfm.contracts import TimesFmForecastArtifact
from nlp_stock_prediction.ml.training import EvaluationResult, TechnicalLogisticModel

_RAW_TIMESFM_MODEL_KIND = "timesfm_2_5_raw_forecast"
_MlSignalStatus = Literal["usable", "weak", "stale", "conflicting", "unavailable"]


def build_technical_ml_signal(
    *,
    model: TechnicalLogisticModel,
    evaluation: EvaluationResult,
    as_of: date | datetime,
    min_validation_accuracy: float = 0.52,
    min_confidence: float = 0.12,
) -> TechnicalMlSignal:
    """Map a local model/evaluation result into a report-safe technical sidecar."""

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
        metadata={
            "model_kind": model.model_kind,
            "threshold": model.threshold,
            "validation_samples": evaluation.metrics.samples,
        },
    )


def load_timesfm_forecast_signal(path: Path) -> TechnicalMlSignal:
    """Load a raw TimesFM forecast artifact as a technical sidecar."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("TimesFM forecast artifact must be a JSON object")
    artifact = TimesFmForecastArtifact.model_validate(payload)
    artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return build_timesfm_forecast_signal(
        artifact,
        artifact_path=path,
        artifact_sha256=artifact_sha256,
    )


def build_timesfm_forecast_signal(
    artifact: TimesFmForecastArtifact,
    *,
    artifact_path: Path | None = None,
    artifact_sha256: str | None = None,
) -> TechnicalMlSignal:
    """Map a raw TimesFM forecast artifact into the generic technical ML sidecar."""

    probability = artifact.directional_probability_proxy
    if probability is None:
        probability = _timesfm_probability_proxy(
            artifact.expected_return,
            artifact.interval_width,
        )
    confidence = _timesfm_forecast_confidence(artifact)
    status: _MlSignalStatus = "usable"
    if artifact.status == "unavailable":
        status = "unavailable"
    elif artifact.status == "weak":
        status = "weak"
    metadata: JsonObject = {
        "model_kind": _RAW_TIMESFM_MODEL_KIND,
        "model_id": artifact.model_id,
        "model_revision": artifact.model_revision,
        "target_field": artifact.target_field,
        "forecast_timestamp": artifact.forecast_timestamp.isoformat(),
        "warning_ids": list(artifact.warning_ids),
    }
    if artifact_path is not None:
        metadata["artifact_path"] = str(artifact_path)
    if artifact_sha256 is not None:
        metadata["artifact_sha256"] = artifact_sha256
    if artifact.metadata:
        metadata["source_metadata"] = artifact.metadata

    return TechnicalMlSignal(
        model_hash=artifact.model_id,
        dataset_hash=artifact.dataset_hash,
        as_of=artifact.forecast_timestamp,
        feature_end=artifact.context_end or artifact.forecast_timestamp,
        prediction_horizon_sessions=artifact.forecast_horizon_sessions,
        probability_positive=round(probability, 6),
        calibrated_confidence=round(confidence, 6),
        signal=_signal_from_probability(probability),
        status=status,
        freshness_status=(
            FreshnessStatus.UNKNOWN if artifact.status == "unavailable" else FreshnessStatus.FRESH
        ),
        expected_return=artifact.expected_return,
        forecast_interval_width=artifact.interval_width,
        source_artifact_id="ml-timesfm-raw-forecast",
        source_artifact_sha256=artifact_sha256,
        warning_ids=artifact.warning_ids,
        metadata=metadata,
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
        "ML signal is a technical sidecar and cannot support a prediction without evidence.",
    )
    sidecar_label = _sidecar_label(signal)
    summary = (
        f"{analysis.summary} {sidecar_label}: {signal.signal.value} with "
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
    if signal.expected_return is not None:
        metrics.append(
            analysis_metric(
                "ml-expected-return",
                round(signal.expected_return, 8),
                unit="pct",
                as_of=signal.as_of,
                metadata={"model_hash": signal.model_hash, "status": signal.status},
            )
        )
    if signal.forecast_interval_width is not None:
        metrics.append(
            analysis_metric(
                "ml-forecast-interval-width",
                round(signal.forecast_interval_width, 8),
                unit="pct",
                as_of=signal.as_of,
                metadata={"model_hash": signal.model_hash, "status": signal.status},
            )
        )
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


def _timesfm_forecast_confidence(artifact: TimesFmForecastArtifact) -> float:
    if artifact.status == "unavailable":
        return 0.0
    probability = artifact.directional_probability_proxy
    probability_edge = abs((probability if probability is not None else 0.5) - 0.5) * 2.0
    uncertainty = 1.0 - min(artifact.uncertainty_score or artifact.interval_width or 1.0, 1.0)
    confidence = probability_edge * 0.55 + uncertainty * 0.45
    if artifact.status == "weak":
        confidence = min(confidence, 0.24)
    return clamp(confidence)


def _timesfm_probability_proxy(
    expected_return: float | None, interval_width: float | None
) -> float:
    if expected_return is None or expected_return == 0:
        return 0.5
    uncertainty = max(interval_width if interval_width is not None else abs(expected_return), 0.01)
    edge = min(0.49, abs(expected_return) / uncertainty * 0.25)
    probability = 0.5 + edge if expected_return > 0 else 0.5 - edge
    return round(clamp(probability), 6)


def _sidecar_label(signal: TechnicalMlSignal) -> str:
    if signal.metadata.get("model_kind") == _RAW_TIMESFM_MODEL_KIND:
        return "TimesFM raw forecast sidecar"
    return "ML sidecar"


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


__all__ = [
    "apply_technical_ml_signal",
    "build_technical_ml_signal",
    "build_timesfm_forecast_signal",
    "load_timesfm_forecast_signal",
]
