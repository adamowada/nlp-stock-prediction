"""Normalize Reddit discussion records into source evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

from nlp_stock_prediction.contracts import (
    EvidenceRequest,
    FreshnessStatus,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TextSpan,
)
from nlp_stock_prediction.reddit.matching import find_ticker_matches, matched_tickers

_DEFAULT_FRESHNESS_WINDOW_SECONDS = 86_400
_DEFAULT_SOURCE_URL = "https://www.reddit.com/search/"


def normalize_reddit_evidence(
    records: Iterable[Mapping[str, object]],
    *,
    request: EvidenceRequest,
    fetched_at: datetime,
    raw_snapshot_id: str,
    retrieval_method: RetrievalMethod = RetrievalMethod.PUBLIC_SCRAPE,
    provider_name: str = "reddit",
    freshness_window_seconds: int = _DEFAULT_FRESHNESS_WINDOW_SECONDS,
) -> tuple[SourceEvidence, ...]:
    """Normalize fixture/API-shaped Reddit posts and comments into evidence records."""

    evidence: list[SourceEvidence] = []
    for source_rank, record in enumerate(records):
        source_kind = _source_kind(record)
        if source_kind == SourceKind.REDDIT_POST and not request.include_posts:
            continue
        if source_kind == SourceKind.REDDIT_COMMENT and not request.include_comments:
            continue

        text = _record_text(record, source_kind)
        if not text:
            continue

        matches = find_ticker_matches(text, request.tickers)
        ticker_order = matched_tickers(matches)
        if not ticker_order:
            continue

        reddit_id = _string(record, "id") or f"record-{source_rank}"
        created_at = _created_at(record)
        permalink = _string(record, "permalink")
        source_url = (
            _string(record, "source_url")
            or _string(record, "url")
            or permalink
            or _DEFAULT_SOURCE_URL
        )
        freshness_status, freshness_seconds = _freshness(
            created_at=created_at,
            fetched_at=fetched_at,
            freshness_window_seconds=freshness_window_seconds,
        )
        reddit_kind = "post" if source_kind == SourceKind.REDDIT_POST else "comment"
        metadata = _metadata(record, source_rank=source_rank, matched=ticker_order)

        evidence.append(
            SourceEvidence(
                evidence_id=f"reddit-{reddit_kind}-{reddit_id}",
                source_kind=source_kind,
                ticker=ticker_order[0],
                title=_string(record, "title") if source_kind == SourceKind.REDDIT_POST else None,
                text=text,
                author_hash=_author_hash(record),
                created_at=created_at,
                score=_int(record, "score"),
                permalink=permalink,
                matched_tickers=ticker_order,
                match_spans=tuple(
                    TextSpan(
                        text=match.text,
                        start_char=match.start_char,
                        end_char=match.end_char,
                    )
                    for match in matches
                    if match.ticker in ticker_order
                ),
                provenance=SourceProvenance(
                    provider_name=provider_name,
                    source_kind=source_kind,
                    retrieval_method=retrieval_method,
                    fetched_at=fetched_at,
                    observed_at=created_at,
                    source_url=source_url,
                    permalink=permalink,
                    raw_identifier=reddit_id,
                    raw_snapshot_id=raw_snapshot_id,
                    query=request.query,
                    cache_key=_cache_key(request, reddit_kind),
                    freshness_status=freshness_status,
                    freshness_seconds=freshness_seconds,
                    provider_metadata={
                        "request_id": request.request_id,
                        "reddit_id": reddit_id,
                        "reddit_kind": reddit_kind,
                        "source_rank": source_rank,
                        **_provenance_metadata(record),
                    },
                ),
                metadata=metadata,
            )
        )

    return tuple(evidence)


def _source_kind(record: Mapping[str, object]) -> SourceKind:
    kind = (_string(record, "kind") or "post").lower()
    if kind in {"comment", "reddit_comment", "t1"}:
        return SourceKind.REDDIT_COMMENT
    return SourceKind.REDDIT_POST


def _record_text(record: Mapping[str, object], source_kind: SourceKind) -> str | None:
    if source_kind == SourceKind.REDDIT_COMMENT:
        return _string(record, "body")

    title = _string(record, "title")
    selftext = _string(record, "selftext")
    if title and selftext:
        return f"{title}\n\n{selftext}"
    return title or selftext


def _metadata(
    record: Mapping[str, object],
    *,
    source_rank: int,
    matched: tuple[str, ...],
) -> dict[str, object]:
    keys = (
        "subreddit",
        "source_url",
        "search_query",
        "search_url",
        "discussion_url",
        "result_rank",
        "parent_id",
        "link_id",
        "depth",
        "is_submitter",
        "distinguished",
        "stickied",
        "num_comments",
        "upvote_ratio",
    )
    metadata: dict[str, object] = {
        "reddit_id": _string(record, "id") or f"record-{source_rank}",
        "source_rank": source_rank,
        "matched_tickers": list(matched),
        "matched_ticker_count": len(matched),
    }
    for key in keys:
        value = record.get(key)
        if _is_json_scalar(value):
            metadata[key] = value
    return metadata


def _provenance_metadata(record: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "source_url",
        "search_query",
        "search_url",
        "discussion_url",
        "result_rank",
        "subreddit",
    )
    return {key: value for key in keys if _is_json_scalar(value := record.get(key))}


def _author_hash(record: Mapping[str, object]) -> str | None:
    author_hash = _string(record, "author_hash")
    if author_hash:
        return author_hash
    author = _string(record, "author")
    if not author:
        return None
    digest = hashlib.sha256(author.strip().lower().encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _created_at(record: Mapping[str, object]) -> datetime | None:
    created_utc = record.get("created_utc")
    if isinstance(created_utc, int | float):
        return datetime.fromtimestamp(created_utc, tz=UTC)

    created_at = _string(record, "created_at")
    if not created_at:
        return None
    normalized = created_at.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _freshness(
    *,
    created_at: datetime | None,
    fetched_at: datetime,
    freshness_window_seconds: int,
) -> tuple[FreshnessStatus, int | None]:
    if created_at is None:
        return FreshnessStatus.MISSING, None
    raw_age_seconds = int((fetched_at - created_at).total_seconds())
    if raw_age_seconds < -300:
        return FreshnessStatus.UNKNOWN, 0
    age_seconds = max(0, raw_age_seconds)
    status = (
        FreshnessStatus.FRESH if age_seconds <= freshness_window_seconds else FreshnessStatus.STALE
    )
    return status, age_seconds


def _cache_key(request: EvidenceRequest, reddit_kind: str) -> str:
    ticker_part = ",".join(request.tickers) if request.tickers else "all"
    return f"reddit:discussion:{reddit_kind}:{request.run_date.isoformat()}:{ticker_part}"


def _string(record: Mapping[str, object], key: str) -> str | None:
    value = record.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _int(record: Mapping[str, object], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _is_json_scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool) or value is None


__all__ = ["normalize_reddit_evidence"]
