from __future__ import annotations

import os
import socket
from collections.abc import Iterator

import pytest


def _network_guard(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError(
        "Network access is disabled for deterministic tests. "
        "Use a live_api or live_scraping marker with "
        "NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS=1 for opt-in live checks."
    )


@pytest.fixture(autouse=True)
def block_network_by_default(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    live_marker = request.node.get_closest_marker("live_api") or request.node.get_closest_marker(
        "live_scraping"
    )
    live_opt_in = os.environ.get("NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS") == "1"
    if live_marker and live_opt_in:
        yield
        return

    monkeypatch.setattr(socket, "create_connection", _network_guard)
    monkeypatch.setattr(socket.socket, "connect", _network_guard)
    monkeypatch.setattr(socket.socket, "connect_ex", _network_guard)
    yield
