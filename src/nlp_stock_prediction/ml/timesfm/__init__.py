"""Optional Google TimesFM 2.5 helpers.

This package must stay importable without Torch, Transformers, PEFT, or Hugging Face dependencies
installed. Heavy dependencies are imported inside command/runtime functions only.
"""

from nlp_stock_prediction.ml.timesfm.dataset import (
    TimesFmDataset,
    TimesFmDatasetConfig,
    TimesFmWindow,
    build_timesfm_dataset,
)

__all__ = [
    "TimesFmDataset",
    "TimesFmDatasetConfig",
    "TimesFmWindow",
    "build_timesfm_dataset",
    "smoke",
]
