from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path if existing_pythonpath is None else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    return env


def test_contract_namespace_exports_clean_report_surface() -> None:
    contracts = importlib.import_module("nlp_stock_prediction.contracts")
    exported_names = tuple(contracts.__all__)

    assert len(exported_names) == len(set(exported_names))
    assert "InstrumentReportSection" in exported_names
    assert "PredictionCandidate" in exported_names
    assert "PredictionStatus" in exported_names


def test_removed_recommendation_module_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nlp_stock_prediction.contracts.recommendation")


def test_contract_namespace_star_import_matches_public_all() -> None:
    contracts = importlib.import_module("nlp_stock_prediction.contracts")
    namespace: dict[str, object] = {}

    exec("from nlp_stock_prediction.contracts import *", namespace)

    imported_names = {
        name for name in namespace if not name.startswith("__") and name != "annotations"
    }
    assert imported_names == set(contracts.__all__)


@pytest.mark.parametrize(
    "module_name",
    (
        "analysis",
        "base",
        "discovery",
        "enums",
        "evidence",
        "extraction",
        "fixtures",
        "instruments",
        "provenance",
        "providers",
        "report",
    ),
)
def test_contract_submodules_define_unique_public_exports(module_name: str) -> None:
    module = importlib.import_module(f"nlp_stock_prediction.contracts.{module_name}")
    exported_names = tuple(module.__all__)

    assert len(exported_names) == len(set(exported_names))
    for export_name in exported_names:
        assert hasattr(module, export_name)


def test_top_level_package_and_module_entrypoint_import_cleanly() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import nlp_stock_prediction; "
                "import nlp_stock_prediction.__main__ as entrypoint; "
                "import nlp_stock_prediction.cli as cli; "
                "assert nlp_stock_prediction.__version__ == '0.1.0'; "
                "assert entrypoint.main is cli.main"
            ),
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_python_module_entrypoint_exposes_help_without_import_errors() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nlp_stock_prediction", "--help"],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Generate evidence-grounded prediction research artifacts." in completed.stdout
    assert completed.stderr == ""
