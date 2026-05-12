"""Optional Google TimesFM 2.5 helpers.

This package must stay importable without Torch, Transformers, PEFT, or Hugging Face dependencies
installed. Heavy dependencies are imported inside command/runtime functions only.
"""

__all__ = ["smoke"]
