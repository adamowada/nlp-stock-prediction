"""Convenience launcher for the persistent terminal app."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src = repo_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _ensure_src_on_path()
    from nlp_stock_prediction.app import run_app

    return run_app(repo_root=Path(__file__).resolve().parent)


if __name__ == "__main__":
    raise SystemExit(main())
