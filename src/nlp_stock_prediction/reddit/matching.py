"""High-precision ticker matching for Reddit text."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TickerMatch:
    """A ticker mention and its character span in source text."""

    ticker: str
    text: str
    start_char: int
    end_char: int
    is_cashtag: bool = False


_TOKEN_RE = re.compile(r"\$?[A-Za-z][A-Za-z0-9.\-']*|\d+(?:\.\d+)?")
_BOUNDARY_CHARS = r"A-Za-z0-9_.\-"
_SHORT_SYMBOLS_REQUIRING_CONTEXT = frozenset({"AI", "IT", "MU", "ON"})
_TRADING_CONTEXT_WORDS = frozenset(
    {
        "active",
        "bag",
        "bags",
        "breakout",
        "buy",
        "buying",
        "call",
        "calls",
        "catalyst",
        "chart",
        "contracts",
        "dip",
        "earnings",
        "entry",
        "gap",
        "hold",
        "holding",
        "holds",
        "iv",
        "leaps",
        "long",
        "moon",
        "options",
        "over",
        "puts",
        "resistance",
        "shares",
        "short",
        "spread",
        "stock",
        "strike",
        "support",
        "swing",
        "ticker",
        "volume",
        "weekly",
        "yolo",
    }
)


def find_ticker_matches(text: str, tickers: Iterable[str]) -> tuple[TickerMatch, ...]:
    """Find high-confidence ticker mentions in first-seen text order."""

    normalized_tickers = _normalize_tickers(tickers)
    found: list[TickerMatch] = []
    seen_spans: set[tuple[int, int]] = set()

    for ticker in normalized_tickers:
        for match in _find_cashtag_matches(text, ticker):
            span = (match.start_char, match.end_char)
            if span not in seen_spans:
                found.append(match)
                seen_spans.add(span)

        for match in _find_bare_symbol_matches(text, ticker):
            span = (match.start_char, match.end_char)
            if span in seen_spans:
                continue
            if match.start_char > 0 and text[match.start_char - 1] == "$":
                continue
            if _requires_context(ticker) and not _has_trading_context(
                text, match.start_char, match.end_char
            ):
                continue
            found.append(match)
            seen_spans.add(span)

    return tuple(
        sorted(
            found,
            key=lambda ticker_match: (ticker_match.start_char, ticker_match.end_char),
        )
    )


def matched_tickers(matches: Iterable[TickerMatch]) -> tuple[str, ...]:
    """Return unique matched tickers in first-seen match order."""

    ordered: list[str] = []
    seen: set[str] = set()
    for match in matches:
        if match.ticker not in seen:
            ordered.append(match.ticker)
            seen.add(match.ticker)
    return tuple(ordered)


def _normalize_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        normalized = ticker.strip().removeprefix("$").upper()
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return tuple(ordered)


def _find_cashtag_matches(text: str, ticker: str) -> Iterable[TickerMatch]:
    pattern = re.compile(
        rf"(?<![{_BOUNDARY_CHARS}])\${re.escape(ticker)}(?![{_BOUNDARY_CHARS}])",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        yield TickerMatch(
            ticker=ticker,
            text=match.group(0),
            start_char=match.start(),
            end_char=match.end(),
            is_cashtag=True,
        )


def _find_bare_symbol_matches(text: str, ticker: str) -> Iterable[TickerMatch]:
    pattern = re.compile(rf"(?<![{_BOUNDARY_CHARS}]){re.escape(ticker)}(?![{_BOUNDARY_CHARS}])")
    for match in pattern.finditer(text):
        yield TickerMatch(
            ticker=ticker,
            text=match.group(0),
            start_char=match.start(),
            end_char=match.end(),
        )


def _requires_context(ticker: str) -> bool:
    return len(ticker) <= 2 or ticker in _SHORT_SYMBOLS_REQUIRING_CONTEXT


def _has_trading_context(text: str, start_char: int, end_char: int) -> bool:
    tokens = tuple(_TOKEN_RE.finditer(text))
    matched_index: int | None = None
    for index, token in enumerate(tokens):
        if token.start() == start_char and token.end() == end_char:
            matched_index = index
            break
    if matched_index is None:
        return False

    lower_bound = max(0, matched_index - 5)
    upper_bound = min(len(tokens), matched_index + 6)
    for index in range(lower_bound, upper_bound):
        if index == matched_index:
            continue
        context = tokens[index].group(0).lstrip("$").strip("'").lower()
        if context in _TRADING_CONTEXT_WORDS:
            return True
    return False


__all__ = ["TickerMatch", "find_ticker_matches", "matched_tickers"]
