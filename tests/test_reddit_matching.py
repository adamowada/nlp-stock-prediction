from __future__ import annotations

import pytest

from nlp_stock_prediction.reddit.matching import find_ticker_matches, matched_tickers


@pytest.mark.unit
def test_cashtags_match_case_insensitively_and_preserve_text_order() -> None:
    text = "Buying $ai and $MU before TSLA and NVDA calls."

    matches = find_ticker_matches(text, ("AI", "MU", "TSLA", "NVDA"))

    assert [(match.ticker, match.text) for match in matches] == [
        ("AI", "$ai"),
        ("MU", "$MU"),
        ("TSLA", "TSLA"),
        ("NVDA", "NVDA"),
    ]
    assert matched_tickers(matches) == ("AI", "MU", "TSLA", "NVDA")


@pytest.mark.unit
def test_short_tickers_do_not_match_lowercase_words_or_common_plain_english() -> None:
    text = "I am on it with an AI summary after the museum trip, no position."

    matches = find_ticker_matches(text, ("MU", "AI", "ON", "IT"))

    assert matches == ()


@pytest.mark.unit
def test_short_bare_symbols_require_trading_context() -> None:
    text = "MU calls are active. AI puts into earnings. ON volume breakout. IT shares over 40."

    matches = find_ticker_matches(text, ("MU", "AI", "ON", "IT"))

    assert matched_tickers(matches) == ("MU", "AI", "ON", "IT")
    assert [(match.ticker, match.text) for match in matches] == [
        ("MU", "MU"),
        ("AI", "AI"),
        ("ON", "ON"),
        ("IT", "IT"),
    ]


@pytest.mark.unit
def test_tickers_do_not_match_inside_longer_tokens() -> None:
    text = "MUSK mentioned AIM and ONON while the item was split."

    matches = find_ticker_matches(text, ("MU", "AI", "ON", "IT"))

    assert matches == ()
