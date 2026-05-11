from __future__ import annotations

import os
from urllib.request import Request, urlopen

import pytest

ALLOW_LIVE_ENV = "NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS"
LIVE_USER_AGENT_ENV = "NLP_STOCK_PREDICTION_LIVE_USER_AGENT"
LIVE_SCRAPE_URL_ENV = "NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL"


def _live_tests_enabled() -> bool:
    return os.environ.get(ALLOW_LIVE_ENV) == "1"


@pytest.mark.live_api
@pytest.mark.skipif(
    not _live_tests_enabled(),
    reason="Set NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS=1 to run live API smoke checks.",
)
def test_live_sec_company_tickers_official_api_smoke() -> None:
    user_agent = os.environ.get(LIVE_USER_AGENT_ENV)
    if not user_agent:
        pytest.skip(f"Set {LIVE_USER_AGENT_ENV} to identify the scheduled live API smoke check.")

    request = Request(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )

    with urlopen(request, timeout=10) as response:
        body = response.read(1024)

    assert response.status == 200
    assert b"ticker" in body.lower()


@pytest.mark.live_scraping
@pytest.mark.skipif(
    not _live_tests_enabled(),
    reason="Set NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS=1 to run live scraping smoke checks.",
)
def test_live_public_scraping_configured_url_smoke() -> None:
    scrape_url = os.environ.get(LIVE_SCRAPE_URL_ENV)
    if not scrape_url:
        pytest.skip(f"Set {LIVE_SCRAPE_URL_ENV} to the narrow public page to smoke-test.")

    request = Request(scrape_url, headers={"User-Agent": "nlp-stock-prediction-live-smoke/0.1"})

    with urlopen(request, timeout=10) as response:
        body = response.read(2048)

    assert response.status < 500
    assert body.strip()
