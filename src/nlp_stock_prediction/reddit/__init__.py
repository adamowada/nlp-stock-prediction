"""Reddit discovery and evidence helpers for Phase 2 Lane A."""

from nlp_stock_prediction.reddit.discovery import discover_tickers_from_devvit_html
from nlp_stock_prediction.reddit.evidence import normalize_reddit_evidence
from nlp_stock_prediction.reddit.matching import TickerMatch, find_ticker_matches, matched_tickers
from nlp_stock_prediction.reddit.provider import FixtureRedditProvider

__all__ = [
    "FixtureRedditProvider",
    "TickerMatch",
    "discover_tickers_from_devvit_html",
    "find_ticker_matches",
    "matched_tickers",
    "normalize_reddit_evidence",
]
