"""Audit artifact serialization helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nlp_stock_prediction.contracts import JsonObject


def stable_json_bytes(payload: JsonObject) -> bytes:
    """Serialize audit payloads deterministically for hashing and fixtures."""

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    return f"{rendered}\n".encode()


def json_payload_sha256(payload: JsonObject) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def write_json_artifact(path: Path, payload: JsonObject) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = stable_json_bytes(payload)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


__all__ = ["json_payload_sha256", "stable_json_bytes", "write_json_artifact"]
