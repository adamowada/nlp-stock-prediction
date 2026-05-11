from __future__ import annotations

import pytest


@pytest.mark.live_api
@pytest.mark.skip(reason="Phase 2/3 live API checks are opt-in and not implemented yet.")
def test_live_api_marker_placeholder() -> None:
    pass


@pytest.mark.live_scraping
@pytest.mark.skip(reason="Phase 2/3 live scraping checks are opt-in and not implemented yet.")
def test_live_scraping_marker_placeholder() -> None:
    pass


@pytest.mark.e2e
@pytest.mark.skip(reason="Phase 2/3 fixture-backed e2e report generation is not implemented yet.")
def test_e2e_marker_placeholder() -> None:
    pass
