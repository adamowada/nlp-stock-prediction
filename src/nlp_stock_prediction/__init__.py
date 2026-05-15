"""Public package for the nlp-stock-prediction CLI."""

import tomllib
from importlib import metadata
from pathlib import Path

try:
    __version__ = metadata.version("nlp-stock-prediction")
except metadata.PackageNotFoundError:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject_path.exists():
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        __version__ = str(pyproject["project"]["version"])
    else:
        __version__ = "0+unknown"
