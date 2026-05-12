"""Conservative ML sidecar integration for technical analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal, cast

from nlp_stock_prediction.analysis._metrics import analysis_metric, clamp
from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    FreshnessStatus,
    MetricValue,
    TechnicalAnalysis,
    TechnicalMlSignal,
)
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.ml.timesfm.evaluate import (
    TimesFmEvaluationArtifact,
    TimesFmEvaluationRecord,
)
from nlp_stock_prediction.ml.training import EvaluationResult, TechnicalLogisticModel

_DEFAULT_LIMITATION = (
    "Experimental local technical model output; not investment advice and not a standalone "
    "recommendation input."
)
_TIMESFM_LIMITATION_FALLBACK = (
    "Experimental local TimesFM technical-analysis evaluation; not investment advice and not a "
    "standalone recommendation input."
)
_TIMESFM_MODEL_KIND = "timesfm_2_5_lora_evaluation"
_TIMESFM_AUDIT_ARTIFACT_ID = "ml-timesfm-evaluation"
_MlSignalStatus = Literal["usable", "weak", "stale", "conflicting", "unavailable"]


@dataclass(frozen=True)
class TimesFmMlSignalAttachment:
    """Validated TimesFM evaluation artifact prepared for report attachment."""

    ticker: str
    signal: TechnicalMlSignal
    artifact_payload: JsonObject
    artifact_path: Path
    artifact_sha256: str


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


def load_timesfm_ml_signal_attachment(path: Path) -> TimesFmMlSignalAttachment:
    """Load a TimesFM evaluation artifact and convert it to a report-safe sidecar."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("TimesFM ML artifact must be a JSON object")
    artifact = TimesFmEvaluationArtifact.model_validate(payload)
    artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    signal = build_timesfm_ml_signal(
        artifact,
        artifact_path=path,
        artifact_sha256=artifact_sha256,
    )
    return TimesFmMlSignalAttachment(
        ticker=artifact.ticker,
        signal=signal,
        artifact_payload=cast(JsonObject, artifact.model_dump(mode="json")),
        artifact_path=path,
        artifact_sha256=artifact_sha256,
    )


def build_timesfm_ml_signal(
    artifact: TimesFmEvaluationArtifact,
    *,
    artifact_path: Path | None = None,
    artifact_sha256: str | None = None,
) -> TechnicalMlSignal:
    """Map a TimesFM rolling evaluation artifact into the generic technical ML sidecar."""

    latest_record = _latest_timesfm_record(artifact)
    expected_return = latest_record.timesfm_return if latest_record is not None else None
    interval_width = _timesfm_record_interval_width(latest_record) if latest_record else None
    status = _timesfm_status(artifact)
    warning_ids = tuple(
        dict.fromkeys(
            (
                *(f"timesfm-evaluation:{reason}" for reason in artifact.suitability_reasons),
                *artifact.warning_ids,
            )
        )
    )
    metadata: JsonObject = {
        "model_kind": _TIMESFM_MODEL_KIND,
        "model_id": artifact.model_id,
        "model_revision": artifact.model_revision,
        "adapter_sha256": artifact.adapter_sha256,
        "training_metadata_sha256": artifact.training_metadata_sha256,
        "evaluation_status": artifact.status,
        "suitable_for_scoring": artifact.suitable_for_scoring,
        "suitability_reasons": list(artifact.suitability_reasons),
        "evaluation_source_kind": artifact.evaluation_source_kind,
        "evaluation_source_sha256": artifact.evaluation_source_sha256,
        "evaluated_at": artifact.evaluated_at.isoformat(),
        "latest_bar_timestamp": _timestamp_to_string(artifact.latest_bar_timestamp),
        "metrics": cast(JsonObject, artifact.metrics.model_dump(mode="json")),
        "baselines": [baseline.model_dump(mode="json") for baseline in artifact.baselines],
    }
    if latest_record is not None:
        metadata["latest_record"] = cast(JsonObject, latest_record.model_dump(mode="json"))
    if artifact_path is not None:
        metadata["artifact_path"] = str(artifact_path)
    if artifact_sha256 is not None:
        metadata["artifact_sha256"] = artifact_sha256

    return TechnicalMlSignal(
        model_hash=artifact.model_hash,
        dataset_hash=artifact.dataset_hash,
        as_of=artifact.as_of or artifact.evaluated_at,
        feature_end=latest_record.context_end
        if latest_record is not None
        else artifact.evaluated_at,
        prediction_horizon_sessions=_timesfm_horizon_sessions(artifact),
        probability_positive=_timesfm_probability_proxy(expected_return, interval_width),
        calibrated_confidence=_timesfm_confidence(artifact, interval_width, status),
        signal=_timesfm_signal(expected_return, status),
        status=status,
        freshness_status=(
            FreshnessStatus.STALE
            if status == "stale"
            else FreshnessStatus.UNKNOWN
            if status == "unavailable"
            else FreshnessStatus.FRESH
        ),
        expected_return=expected_return,
        forecast_interval_width=interval_width,
        source_artifact_id=_TIMESFM_AUDIT_ARTIFACT_ID,
        source_artifact_sha256=artifact_sha256,
        validation_accuracy=artifact.metrics.directional_accuracy
        if artifact.metrics.sample_count > 0
        else None,
        warning_ids=warning_ids,
        limitations=(artifact.usage_limitations or _TIMESFM_LIMITATION_FALLBACK,),
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
        "ML signal is a sidecar and cannot qualify a recommendation on its own.",
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


def _latest_timesfm_record(
    artifact: TimesFmEvaluationArtifact,
) -> TimesFmEvaluationRecord | None:
    if not artifact.records:
        return None
    return max(artifact.records, key=lambda record: _timestamp_key(record.horizon_end))


def _timesfm_status(artifact: TimesFmEvaluationArtifact) -> _MlSignalStatus:
    if artifact.status == "unavailable":
        return "unavailable"
    if "stale_evaluation_data" in artifact.suitability_reasons:
        return "stale"
    if artifact.status == "suitable" and artifact.suitable_for_scoring:
        return "usable"
    return "weak"


def _timesfm_signal(expected_return: float | None, status: _MlSignalStatus) -> AnalysisSignal:
    if expected_return is None or status == "unavailable":
        return AnalysisSignal.UNKNOWN
    if expected_return > 0:
        return AnalysisSignal.SUPPORTS
    if expected_return < 0:
        return AnalysisSignal.CONFLICTS
    return AnalysisSignal.MIXED


def _timesfm_confidence(
    artifact: TimesFmEvaluationArtifact,
    interval_width: float | None,
    status: _MlSignalStatus,
) -> float:
    if artifact.metrics.sample_count <= 0 or status == "unavailable":
        return 0.0
    calibration = artifact.metrics.calibration_proxy or 0.0
    uncertainty_score = 1.0 - min(interval_width if interval_width is not None else 1.0, 1.0)
    raw_confidence = (
        artifact.metrics.directional_accuracy * 0.55 + calibration * 0.25 + uncertainty_score * 0.20
    )
    if status in {"weak", "stale"}:
        raw_confidence = min(raw_confidence, 0.24)
    return round(clamp(raw_confidence), 6)


def _timesfm_probability_proxy(
    expected_return: float | None, interval_width: float | None
) -> float:
    if expected_return is None or expected_return == 0:
        return 0.5
    uncertainty = max(interval_width if interval_width is not None else abs(expected_return), 0.01)
    edge = min(0.49, abs(expected_return) / uncertainty * 0.25)
    probability = 0.5 + edge if expected_return > 0 else 0.5 - edge
    return round(clamp(probability), 6)


def _timesfm_record_interval_width(record: TimesFmEvaluationRecord | None) -> float | None:
    if record is None or record.interval_lower is None or record.interval_upper is None:
        return None
    denominator = abs(record.persistence_final_value)
    if denominator == 0:
        return None
    return round(abs(record.interval_upper - record.interval_lower) / denominator, 8)


def _timesfm_horizon_sessions(artifact: TimesFmEvaluationArtifact) -> int:
    split_metadata = artifact.training_metadata.get("split")
    if isinstance(split_metadata, dict):
        value = split_metadata.get("horizon_length")
        if isinstance(value, int) and value >= 1:
            return value
    config_metadata = artifact.training_metadata.get("config")
    if isinstance(config_metadata, dict):
        value = config_metadata.get("horizon_length")
        if isinstance(value, int) and value >= 1:
            return value
    return 1


def _timestamp_to_string(value: date | datetime) -> str:
    return value.isoformat()


def _sidecar_label(signal: TechnicalMlSignal) -> str:
    if signal.metadata.get("model_kind") == _TIMESFM_MODEL_KIND:
        return "TimesFM sidecar"
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
    "TimesFmMlSignalAttachment",
    "apply_technical_ml_signal",
    "build_technical_ml_signal",
    "build_timesfm_ml_signal",
    "load_timesfm_ml_signal_attachment",
]
