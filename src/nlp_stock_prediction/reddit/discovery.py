"""Devvit ticker-card discovery from Reddit HTML."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Any

from nlp_stock_prediction.contracts import (
    FreshnessStatus,
    ProviderWarning,
    RetrievalMethod,
    SourceKind,
    SourceProvenance,
    TickerCandidate,
    TickerDiscoveryRequest,
    TickerDiscoveryResult,
    TickerDiscoveryStatus,
    WarningCode,
    WarningSeverity,
)

_IDENTIFIER_PREFIX = "ticker-container-"
_DEFAULT_SOURCE_URL = "https://www.reddit.com/search/"
_TICKER_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_DEFAULT_FRESHNESS_WINDOW_SECONDS = 86_400


@dataclass(slots=True)
class _ParsedTickerContainer:
    raw_identifier: str
    symbol: str
    text_parts: list[str] = field(default_factory=list)

    @property
    def raw_text(self) -> str | None:
        normalized = " ".join("".join(self.text_parts).split())
        return normalized or None


@dataclass(frozen=True, slots=True)
class _ParsedTickerContainers:
    candidates: tuple[_ParsedTickerContainer, ...]
    invalid_identifiers: tuple[str, ...]


class _TickerContainerHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.candidates: list[_ParsedTickerContainer] = []
        self.invalid_identifiers: list[str] = []
        self._stack: list[tuple[str, tuple[int, ...]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        indices: list[int] = []
        for raw_identifier in _extract_identifiers(attrs):
            symbol = raw_identifier.removeprefix(_IDENTIFIER_PREFIX).strip().upper()
            if not _is_valid_symbol(symbol):
                self.invalid_identifiers.append(raw_identifier)
                continue
            self.candidates.append(
                _ParsedTickerContainer(raw_identifier=raw_identifier, symbol=symbol)
            )
            indices.append(len(self.candidates) - 1)
        self._stack.append((tag, tuple(indices)))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not data.strip():
            return
        active_indices = {
            candidate_index for _tag, indices in self._stack for candidate_index in indices
        }
        for candidate_index in active_indices:
            self.candidates[candidate_index].text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        while self._stack:
            open_tag, _indices = self._stack.pop()
            if open_tag == tag:
                return


def discover_tickers_from_devvit_html(
    html: str,
    *,
    request: TickerDiscoveryRequest,
    fetched_at: datetime,
    raw_snapshot_id: str,
    retrieval_method: RetrievalMethod = RetrievalMethod.PUBLIC_SCRAPE,
    provider_name: str = "reddit",
    observed_at: datetime | None = None,
    freshness_window_seconds: int = _DEFAULT_FRESHNESS_WINDOW_SECONDS,
) -> TickerDiscoveryResult:
    """Parse a Devvit ticker card into a frozen discovery contract."""

    parsed = _parse_ticker_containers(html)
    parsed_candidates = parsed.candidates
    source_url = request.source_url or _DEFAULT_SOURCE_URL
    observed = observed_at or fetched_at
    freshness_status, freshness_seconds = _freshness(
        observed_at=observed,
        fetched_at=fetched_at,
        freshness_window_seconds=freshness_window_seconds,
    )
    warnings: list[ProviderWarning] = []

    if parsed.invalid_identifiers:
        warnings.append(
            _warning(
                provider_name=provider_name,
                fetched_at=fetched_at,
                raw_snapshot_id=raw_snapshot_id,
                source_url=source_url,
                severity=WarningSeverity.ERROR,
                code=WarningCode.SCHEMA_MISMATCH,
                message="Reddit Devvit ticker card contained invalid ticker-container identifiers.",
                metadata={
                    "validation": "invalid_ticker_container_identifiers",
                    "invalid_identifiers": list(parsed.invalid_identifiers),
                },
            )
        )

    if not parsed_candidates:
        warnings.append(
            _warning(
                provider_name=provider_name,
                fetched_at=fetched_at,
                raw_snapshot_id=raw_snapshot_id,
                source_url=source_url,
                severity=WarningSeverity.ERROR,
                code=WarningCode.SCRAPING_DRIFT,
                message="Reddit Devvit ticker card did not contain ticker-container identifiers.",
                metadata={"validation": "missing_ticker_containers"},
            )
        )
        return TickerDiscoveryResult(
            run_date=request.run_date,
            status=TickerDiscoveryStatus.MALFORMED_SOURCE,
            warnings=tuple(warnings),
            raw_snapshot_id=raw_snapshot_id,
        )

    tickers = tuple(dict.fromkeys(candidate.symbol for candidate in parsed_candidates))
    duplicate_symbols = _duplicate_symbols(parsed_candidates)
    if duplicate_symbols:
        warnings.append(
            _warning(
                provider_name=provider_name,
                fetched_at=fetched_at,
                raw_snapshot_id=raw_snapshot_id,
                source_url=source_url,
                severity=WarningSeverity.WARNING,
                code=WarningCode.PARTIAL_DATA,
                message="Duplicate ticker-container candidates were removed from ticker order.",
                metadata={
                    "validation": "duplicate_tickers",
                    "duplicate_symbols": list(duplicate_symbols),
                },
            )
        )

    status = (
        TickerDiscoveryStatus.MALFORMED_SOURCE
        if parsed.invalid_identifiers
        else TickerDiscoveryStatus.VALID
    )
    if len(tickers) < 6:
        status = TickerDiscoveryStatus.TOO_FEW_UNIQUE
        warnings.append(
            _warning(
                provider_name=provider_name,
                fetched_at=fetched_at,
                raw_snapshot_id=raw_snapshot_id,
                source_url=source_url,
                severity=WarningSeverity.ERROR,
                code=WarningCode.PARTIAL_DATA,
                message="Reddit Devvit ticker card produced fewer than six unique tickers.",
                metadata={
                    "validation": "too_few_unique_tickers",
                    "unique_ticker_count": len(tickers),
                    "expected_unique_ticker_count": 6,
                },
            )
        )
    elif len(tickers) > 6:
        status = TickerDiscoveryStatus.TOO_MANY_UNIQUE
        warnings.append(
            _warning(
                provider_name=provider_name,
                fetched_at=fetched_at,
                raw_snapshot_id=raw_snapshot_id,
                source_url=source_url,
                severity=WarningSeverity.ERROR,
                code=WarningCode.SCHEMA_MISMATCH,
                message="Reddit Devvit ticker card produced more than six unique tickers.",
                metadata={
                    "validation": "too_many_unique_tickers",
                    "unique_ticker_count": len(tickers),
                    "expected_unique_ticker_count": 6,
                },
            )
        )

    candidates = tuple(
        _to_ticker_candidate(
            candidate=parsed_candidate,
            rank=rank,
            request=request,
            fetched_at=fetched_at,
            observed_at=observed,
            raw_snapshot_id=raw_snapshot_id,
            retrieval_method=retrieval_method,
            provider_name=provider_name,
            source_url=source_url,
            freshness_status=freshness_status,
            freshness_seconds=freshness_seconds,
        )
        for rank, parsed_candidate in enumerate(parsed_candidates)
    )
    return TickerDiscoveryResult(
        run_date=request.run_date,
        status=status,
        candidates=candidates,
        tickers=tickers,
        warnings=tuple(warnings),
        raw_snapshot_id=raw_snapshot_id,
    )


def _parse_ticker_containers(html: str) -> _ParsedTickerContainers:
    parser = _TickerContainerHtmlParser()
    parser.feed(html)
    parser.close()
    return _ParsedTickerContainers(
        candidates=tuple(parser.candidates),
        invalid_identifiers=tuple(parser.invalid_identifiers),
    )


def _is_valid_symbol(symbol: str) -> bool:
    return _TICKER_SYMBOL_RE.fullmatch(symbol) is not None


def _extract_identifiers(attrs: list[tuple[str, str | None]]) -> tuple[str, ...]:
    identifiers: list[str] = []
    for _name, value in attrs:
        if value is None:
            continue
        for token in value.split():
            if token.startswith(_IDENTIFIER_PREFIX):
                identifiers.append(token)
    return tuple(identifiers)


def _duplicate_symbols(
    candidates: tuple[_ParsedTickerContainer, ...],
) -> tuple[str, ...]:
    counts = Counter(candidate.symbol for candidate in candidates)
    ordered_duplicates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if counts[candidate.symbol] > 1 and candidate.symbol not in seen:
            ordered_duplicates.append(candidate.symbol)
            seen.add(candidate.symbol)
    return tuple(ordered_duplicates)


def _to_ticker_candidate(
    *,
    candidate: _ParsedTickerContainer,
    rank: int,
    request: TickerDiscoveryRequest,
    fetched_at: datetime,
    observed_at: datetime,
    raw_snapshot_id: str,
    retrieval_method: RetrievalMethod,
    provider_name: str,
    source_url: str,
    freshness_status: FreshnessStatus,
    freshness_seconds: int,
) -> TickerCandidate:
    provenance = SourceProvenance(
        provider_name=provider_name,
        source_kind=SourceKind.REDDIT_TICKER_CARD,
        retrieval_method=retrieval_method,
        fetched_at=fetched_at,
        observed_at=observed_at,
        source_url=source_url,
        raw_identifier=candidate.raw_identifier,
        raw_snapshot_id=raw_snapshot_id,
        query=request.query,
        cache_key=f"reddit:ticker-card:{request.run_date.isoformat()}",
        freshness_status=freshness_status,
        freshness_seconds=freshness_seconds,
        provider_metadata={
            "request_id": request.request_id,
            "candidate_rank": rank,
            "raw_identifier": candidate.raw_identifier,
        },
    )
    return TickerCandidate(
        symbol=candidate.symbol,
        raw_identifier=candidate.raw_identifier,
        raw_text=candidate.raw_text,
        first_seen_rank=rank,
        source_url=source_url,
        provenance=provenance,
    )


def _freshness(
    *,
    observed_at: datetime,
    fetched_at: datetime,
    freshness_window_seconds: int,
) -> tuple[FreshnessStatus, int]:
    raw_age_seconds = int((fetched_at - observed_at).total_seconds())
    if raw_age_seconds < -300:
        return FreshnessStatus.UNKNOWN, 0
    age_seconds = max(0, raw_age_seconds)
    status = (
        FreshnessStatus.FRESH if age_seconds <= freshness_window_seconds else FreshnessStatus.STALE
    )
    return status, age_seconds


def _warning(
    *,
    provider_name: str,
    fetched_at: datetime,
    raw_snapshot_id: str,
    source_url: str,
    severity: WarningSeverity,
    code: WarningCode,
    message: str,
    metadata: dict[str, Any],
) -> ProviderWarning:
    return ProviderWarning(
        code=code,
        severity=severity,
        message=message,
        provider_name=provider_name,
        occurred_at=fetched_at,
        raw_snapshot_id=raw_snapshot_id,
        source_url=source_url,
        metadata=metadata,
    )


__all__ = ["discover_tickers_from_devvit_html"]
