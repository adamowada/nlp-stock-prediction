"""Local environment configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DISABLE_DOTENV_ENV = "NLP_STOCK_PREDICTION_DISABLE_DOTENV"


def load_local_dotenv() -> Path | None:
    """Load the nearest local ``.env`` file without overriding shell variables."""

    if os.environ.get(DISABLE_DOTENV_ENV) == "1":
        return None
    dotenv_path = find_dotenv(filename=".env", usecwd=True)
    if not dotenv_path:
        return None
    resolved_path = Path(dotenv_path)
    load_dotenv(dotenv_path=resolved_path, override=False)
    return resolved_path


__all__ = ["DISABLE_DOTENV_ENV", "load_local_dotenv"]
