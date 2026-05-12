"""Contracts for local TimesFM 2.5 inference artifacts."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    Confidence,
    ContractModel,
    JsonObject,
    NonEmptyStr,
    TickerSymbol,
)
from nlp_stock_prediction.ml.timesfm.dataset import ResolvedTargetField

TimesFmForecastStatus = Literal["usable", "weak", "unavailable"]


class TimesFmQuantileForecast(ContractModel):
    """One indexed TimesFM full-prediction path.

    TimesFM exposes a fixed set of full prediction paths. The adapter preserves the index without
    claiming a quantile level that is not provided by the local Transformers API.
    """

    quantile_index: int = Field(ge=0)
    values: tuple[float, ...]

    @model_validator(mode="after")
    def validate_values(self) -> TimesFmQuantileForecast:
        if not self.values:
            raise ValueError("quantile forecast values cannot be empty")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("quantile forecast values must be finite")
        return self


class TimesFmForecastArtifact(ContractModel):
    """Serializable local TimesFM forecast artifact."""

    model_config = ConfigDict(extra="ignore")

    schema_version: NonEmptyStr = "ml.timesfm.forecast.v1"
    status: TimesFmForecastStatus
    ticker: TickerSymbol
    model_id: NonEmptyStr
    model_revision: str | None = None
    dataset_hash: NonEmptyStr
    input_hash: NonEmptyStr
    forecast_timestamp: AwareDatetime
    target_field: ResolvedTargetField
    context_start: datetime | None = None
    context_end: datetime | None = None
    forecast_horizon_sessions: int = Field(ge=1)
    point_forecast: tuple[float, ...] = Field(default_factory=tuple)
    quantile_forecasts: tuple[TimesFmQuantileForecast, ...] = Field(default_factory=tuple)
    expected_return: float | None = None
    interval_width: float | None = Field(default=None, ge=0.0)
    directional_probability_proxy: Confidence | None = None
    uncertainty_score: Confidence | None = None
    warning_ids: tuple[str, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_artifact(self) -> TimesFmForecastArtifact:
        if self.status in {"usable", "weak"}:
            if len(self.point_forecast) != self.forecast_horizon_sessions:
                raise ValueError("point forecast length must match forecast horizon")
            if any(not isfinite(value) for value in self.point_forecast):
                raise ValueError("point forecast values must be finite")
        for forecast in self.quantile_forecasts:
            if len(forecast.values) != self.forecast_horizon_sessions:
                raise ValueError("quantile forecast lengths must match forecast horizon")
        if self.status == "unavailable" and not self.warning_ids:
            raise ValueError("unavailable TimesFM artifacts must include a warning")
        return self


__all__ = [
    "TimesFmForecastArtifact",
    "TimesFmForecastStatus",
    "TimesFmQuantileForecast",
]
