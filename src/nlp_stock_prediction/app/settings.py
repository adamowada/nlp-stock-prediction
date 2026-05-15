"""Local terminal-app settings stored in ignored app state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nlp_stock_prediction.contracts.base import JsonObject, JsonValue

AppMode = Literal["live", "offline"]


@dataclass(frozen=True)
class AppSettings:
    """User preferences for the persistent terminal app.

    The app state deliberately serializes only these known fields so copied environment mappings or
    accidental API-key-shaped values cannot be persisted as settings.
    """

    default_symbol: str = "TSLA"
    default_mode: AppMode = "live"
    output_dir: Path = Path("reports")
    cache_dir: Path = Path("cache")
    codex_executable: str = "codex"
    codex_model: str | None = None
    codex_profile: str | None = None
    enable_web_search: bool = True

    @classmethod
    def from_json(cls, payload: object) -> AppSettings:
        if not isinstance(payload, dict):
            return cls()
        return cls(
            default_symbol=_string(payload.get("default_symbol"), cls.default_symbol).upper(),
            default_mode=_mode(payload.get("default_mode")),
            output_dir=Path(_string(payload.get("output_dir"), "reports")),
            cache_dir=Path(_string(payload.get("cache_dir"), "cache")),
            codex_executable=_string(payload.get("codex_executable"), "codex"),
            codex_model=_optional_string(payload.get("codex_model")),
            codex_profile=_optional_string(payload.get("codex_profile")),
            enable_web_search=_bool(payload.get("enable_web_search"), True),
        ).normalized()

    def normalized(self) -> AppSettings:
        symbol = self.default_symbol.strip().upper() or "TSLA"
        executable = self.codex_executable.strip() or "codex"
        return AppSettings(
            default_symbol=symbol,
            default_mode=self.default_mode if self.default_mode in {"live", "offline"} else "live",
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            codex_executable=executable,
            codex_model=_optional_string(self.codex_model),
            codex_profile=_optional_string(self.codex_profile),
            enable_web_search=self.enable_web_search,
        )

    def with_updates(self, **updates: object) -> AppSettings:
        data: dict[str, object] = {
            "default_symbol": self.default_symbol,
            "default_mode": self.default_mode,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "codex_executable": self.codex_executable,
            "codex_model": self.codex_model,
            "codex_profile": self.codex_profile,
            "enable_web_search": self.enable_web_search,
        }
        data.update(updates)
        return AppSettings(
            default_symbol=str(data["default_symbol"]),
            default_mode=_mode(data["default_mode"]),
            output_dir=Path(str(data["output_dir"])),
            cache_dir=Path(str(data["cache_dir"])),
            codex_executable=str(data["codex_executable"]),
            codex_model=_optional_string(data["codex_model"]),
            codex_profile=_optional_string(data["codex_profile"]),
            enable_web_search=_bool(data["enable_web_search"], True),
        ).normalized()

    def to_json(self) -> JsonObject:
        payload: JsonObject = {
            "default_symbol": self.default_symbol,
            "default_mode": self.default_mode,
            "output_dir": self.output_dir.as_posix(),
            "cache_dir": self.cache_dir.as_posix(),
            "codex_executable": self.codex_executable,
            "enable_web_search": self.enable_web_search,
        }
        if self.codex_model is not None:
            payload["codex_model"] = self.codex_model
        if self.codex_profile is not None:
            payload["codex_profile"] = self.codex_profile
        return payload


def _string(value: object, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _mode(value: object) -> AppMode:
    if isinstance(value, str) and value.strip().lower() == "offline":
        return "offline"
    return "live"


def json_without_secrets(settings: AppSettings) -> dict[str, JsonValue]:
    """Return persisted settings using only the allow-listed app preference fields."""

    return settings.to_json()


__all__ = ["AppMode", "AppSettings", "json_without_secrets"]
