from __future__ import annotations

import os
from pathlib import Path


def module_subprocess_env(repo_root: Path, *, preserve_os_path: bool = True) -> dict[str, str]:
    env: dict[str, str] = {}
    if preserve_os_path:
        for key in (
            "COMSPEC",
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "WINDIR",
        ):
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
    src_path = str(repo_root / "src")
    existing_pythonpath = os.environ.get("PYTHONPATH") if preserve_os_path else None
    env["PYTHONPATH"] = (
        src_path if existing_pythonpath is None else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    env["NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS"] = "0"
    env["NLP_STOCK_PREDICTION_DISABLE_DOTENV"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env
