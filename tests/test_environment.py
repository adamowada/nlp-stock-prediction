from __future__ import annotations

import os
from pathlib import Path

import pytest

from nlp_stock_prediction.environment import DISABLE_DOTENV_ENV, load_local_dotenv


@pytest.mark.unit
def test_load_local_dotenv_reads_nearest_env_without_overriding_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "NLP_STOCK_PREDICTION_DOTENV_TEST_FROM_FILE=loaded",
                "NLP_STOCK_PREDICTION_DOTENV_TEST_EXISTING=from-file",
            ]
        ),
        encoding="utf-8",
    )
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()

    monkeypatch.chdir(nested_dir)
    monkeypatch.delenv(DISABLE_DOTENV_ENV, raising=False)
    monkeypatch.delenv("NLP_STOCK_PREDICTION_DOTENV_TEST_FROM_FILE", raising=False)
    monkeypatch.setenv("NLP_STOCK_PREDICTION_DOTENV_TEST_EXISTING", "from-shell")

    loaded_path = load_local_dotenv()

    assert loaded_path == dotenv_path
    assert os.environ["NLP_STOCK_PREDICTION_DOTENV_TEST_FROM_FILE"] == "loaded"
    assert os.environ["NLP_STOCK_PREDICTION_DOTENV_TEST_EXISTING"] == "from-shell"


@pytest.mark.unit
def test_load_local_dotenv_reads_from_explicit_start_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "NLP_STOCK_PREDICTION_DOTENV_TEST_EXPLICIT=loaded\n",
        encoding="utf-8",
    )
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()

    monkeypatch.delenv(DISABLE_DOTENV_ENV, raising=False)
    monkeypatch.delenv("NLP_STOCK_PREDICTION_DOTENV_TEST_EXPLICIT", raising=False)

    loaded_path = load_local_dotenv(nested_dir)

    assert loaded_path == dotenv_path
    assert os.environ["NLP_STOCK_PREDICTION_DOTENV_TEST_EXPLICIT"] == "loaded"


@pytest.mark.unit
def test_load_local_dotenv_can_be_disabled_for_deterministic_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "NLP_STOCK_PREDICTION_DOTENV_TEST_DISABLED=loaded\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(DISABLE_DOTENV_ENV, "1")
    monkeypatch.delenv("NLP_STOCK_PREDICTION_DOTENV_TEST_DISABLED", raising=False)

    loaded_path = load_local_dotenv()

    assert loaded_path is None
    assert "NLP_STOCK_PREDICTION_DOTENV_TEST_DISABLED" not in os.environ
