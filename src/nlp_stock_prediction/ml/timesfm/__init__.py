"""Optional Google TimesFM 2.5 helpers.

This package must stay importable without Torch, Transformers, PEFT, or Hugging Face dependencies
installed. Heavy dependencies are imported inside command/runtime functions only.

Raw TimesFM inference remains an optional technical signal. LoRA training, HPO, and multi-stage
promotion modules are legacy compatibility surfaces and should not be used for new product flow.
"""

from nlp_stock_prediction.ml.timesfm.dataset import (
    TimesFmDataset,
    TimesFmDatasetConfig,
    TimesFmWindow,
    build_timesfm_dataset,
)

LEGACY_TIMESFM_TRAINING_MODULES = (
    "artifacts",
    "focused_hpo",
    "signal_funnel",
    "train",
)
LEGACY_TIMESFM_TRAINING_NOTE = (
    "TimesFM LoRA training, HPO, and signal-funnel promotion are retained only for legacy "
    "experiment reproduction. Phase 2+ work should depend on raw TimesFM adapter artifacts at most."
)

__all__ = [
    "LEGACY_TIMESFM_TRAINING_MODULES",
    "LEGACY_TIMESFM_TRAINING_NOTE",
    "TimesFmDataset",
    "TimesFmDatasetConfig",
    "TimesFmWindow",
    "adapter",
    "artifacts",
    "build_timesfm_dataset",
    "contracts",
    "evaluate",
    "smoke",
    "train",
]
