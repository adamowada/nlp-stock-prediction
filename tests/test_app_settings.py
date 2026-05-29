from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_stock_prediction.app.settings import AppSettings
from nlp_stock_prediction.app.state import AppState, load_app_state, save_app_state

pytestmark = pytest.mark.unit


def test_app_settings_default_to_live_research() -> None:
    settings = AppSettings()

    assert settings.default_symbol == ""
    assert settings.default_mode == "live"
    assert settings.output_dir == Path("reports")
    assert settings.cache_dir == Path("cache")
    assert settings.enable_web_search is True


def test_app_state_persists_only_allow_listed_settings(tmp_path: Path) -> None:
    loaded = AppSettings.from_json(
        {
            "default_symbol": "msft",
            "default_mode": "offline",
            "output_dir": "custom-reports",
            "OPENAI_API_KEY": "sk-not-a-real-key",
            "NLP_STOCK_PREDICTION_X_BEARER_TOKEN": "secret",
        }
    )
    state = AppState(settings=loaded)

    save_app_state(tmp_path, state)
    payload = json.loads((tmp_path / "data" / "app-state.json").read_text(encoding="utf-8"))

    assert payload["settings"]["default_symbol"] == "MSFT"
    assert payload["settings"]["default_mode"] == "offline"
    assert "OPENAI_API_KEY" not in payload["settings"]
    assert "NLP_STOCK_PREDICTION_X_BEARER_TOKEN" not in payload["settings"]
    assert load_app_state(tmp_path).settings.default_symbol == "MSFT"


def test_legacy_tsla_default_symbol_is_cleared() -> None:
    state = AppState.from_json(
        {
            "schema_version": "terminal-app-state.v1",
            "settings": {"default_symbol": "TSLA", "default_mode": "live"},
        }
    )

    assert state.settings.default_symbol == ""
