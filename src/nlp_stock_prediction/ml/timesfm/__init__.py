"""Optional raw TimesFM 2.5 inference helpers.

This package stays importable without Torch, Transformers, or Hugging Face dependencies installed.
Heavy dependencies are imported inside command/runtime functions only.
"""

from nlp_stock_prediction.ml.timesfm.contracts import (
    TimesFmForecastArtifact,
    TimesFmForecastStatus,
    TimesFmQuantileForecast,
)
from nlp_stock_prediction.ml.timesfm.dataset import (
    ResolvedTargetField,
    TimesFmDataset,
    TimesFmDatasetConfig,
    TimesFmWindow,
    build_timesfm_dataset,
)

__all__ = [
    "ResolvedTargetField",
    "TimesFmDataset",
    "TimesFmDatasetConfig",
    "TimesFmForecastArtifact",
    "TimesFmForecastStatus",
    "TimesFmQuantileForecast",
    "TimesFmWindow",
    "build_timesfm_dataset",
]
