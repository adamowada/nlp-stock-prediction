"""Small public Reddit HTML parsers used by the scrape adapter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

_DEFAULT_SOURCE_URL = "https://www.reddit.com/search/"


@dataclass(slots=True)
class _DiscussionRecordBuilder:
    kind: str
    raw_attributes: dict[str, str]
    source_rank: int
    title_parts: list[str] = field(default_factory=list)
    body_parts: list[str] = field(default_factory=list)

    def append(self, field_name: str, text: str) -> None:
        if field_name == "title":
            self.title_parts.append(text)
        elif field_name == "body":
            self.body_parts.append(text)

    def to_record(self, *, source_url: str) -> dict[str, object] | None:
        title = _clean_text(self.title_parts)
        body = _clean_text(self.body_parts)
        if self.kind == "post":
            if not (title or body):
                return None
        elif not body:
            return None

        raw_identifier = _first_attr(
            self.raw_attributes,
            "data-reddit-id",
            "thingid",
            "thing-id",
            "post-id",
            "comment-id",
            "id",
        )
        permalink = _normalize_url(
            _first_attr(self.raw_attributes, "permalink", "data-permalink", "href"),
            source_url,
        )
        if not raw_identifier:
            digest_source = f"{self.kind}:{permalink or source_url}:{title}:{body}"
            raw_identifier = f"{_thing_prefix(self.kind)}_scraped_{_stable_hash(digest_source)}"

        subreddit = (
            _first_attr(self.raw_attributes, "subreddit", "data-subreddit")
            or _subreddit_from_url(permalink)
            or _subreddit_from_url(source_url)
        )
        record: dict[str, object] = {
            "kind": self.kind,
            "id": raw_identifier,
        }
        if subreddit:
            record["subreddit"] = subreddit
        if title and self.kind == "post":
            record["title"] = title
        if body:
            record["selftext" if self.kind == "post" else "body"] = body

        author = _first_attr(self.raw_attributes, "author", "data-author")
        if author:
            record["author"] = author

        created_utc = _created_utc(self.raw_attributes)
        if created_utc is not None:
            record["created_utc"] = created_utc
        else:
            created_at = _created_at(self.raw_attributes)
            if created_at is not None:
                record["created_at"] = created_at.isoformat().replace("+00:00", "Z")

        score = _int_attr(self.raw_attributes, "score", "data-score")
        if score is not None:
            record["score"] = score

        if permalink:
            record["permalink"] = permalink
            if self.kind == "post":
                record["url"] = permalink

        link_id = _first_attr(self.raw_attributes, "linkid", "link-id", "data-link-id")
        if link_id:
            record["link_id"] = link_id
        parent_id = _first_attr(self.raw_attributes, "parentid", "parent-id", "data-parent-id")
        if parent_id:
            record["parent_id"] = parent_id
        depth = _int_attr(self.raw_attributes, "depth", "data-depth")
        if depth is not None:
            record["depth"] = depth
        num_comments = _int_attr(
            self.raw_attributes,
            "comment-count",
            "comments",
            "num-comments",
            "data-num-comments",
        )
        if num_comments is not None:
            record["num_comments"] = num_comments

        return record


class _RedditDiscussionHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[_DiscussionRecordBuilder] = []
        self._record_stack: list[tuple[int, str]] = []
        self._field_stack: list[tuple[int, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        attr_map = _attrs_to_dict(attrs)
        kind = _record_kind(normalized_tag, attr_map)
        if kind is not None:
            self.records.append(
                _DiscussionRecordBuilder(
                    kind=kind,
                    raw_attributes=attr_map,
                    source_rank=len(self.records),
                )
            )
            self._record_stack.append((len(self.records) - 1, normalized_tag))

        current_record = self._record_stack[-1][0] if self._record_stack else None
        if current_record is None:
            return
        field_name = _field_name(normalized_tag, attr_map, self.records[current_record].kind)
        if field_name is not None:
            self._field_stack.append((current_record, field_name, normalized_tag))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not data.strip() or not self._field_stack:
            return
        record_index, field_name, _tag = self._field_stack[-1]
        self.records[record_index].append(field_name, data)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if self._field_stack and self._field_stack[-1][2] == normalized_tag:
            self._field_stack.pop()
        if self._record_stack and self._record_stack[-1][1] == normalized_tag:
            self._record_stack.pop()


class _ObservedAtHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = _attrs_to_dict(attrs)
        for key in (
            "data-snapshot-observed-at",
            "data-observed-at",
            "created-timestamp",
            "datetime",
        ):
            value = attr_map.get(key)
            if value:
                self.values.append(value)
                return
        if tag.lower() == "meta" and attr_map.get("content"):
            marker = (attr_map.get("property") or attr_map.get("name") or "").lower()
            if marker in {"article:published_time", "og:updated_time"}:
                self.values.append(attr_map["content"])


@dataclass(slots=True)
class _SearchLinkBuilder:
    href: str
    source_rank: int
    text_parts: list[str] = field(default_factory=list)

    def append(self, text: str) -> None:
        self.text_parts.append(text)

    def to_record(self, *, source_url: str) -> dict[str, object] | None:
        permalink = _normalize_url(self.href, source_url)
        if not _is_public_discussion_url(permalink):
            return None
        title = _clean_text(self.text_parts)
        raw_identifier = _reddit_id_from_url(permalink)
        if raw_identifier is None:
            raw_identifier = f"t3_search_{_stable_hash(permalink or self.href)}"
        record: dict[str, object] = {
            "kind": "post",
            "id": raw_identifier,
            "permalink": permalink,
            "url": permalink,
            "source_url": source_url,
            "result_rank": self.source_rank,
        }
        subreddit = _subreddit_from_url(permalink)
        if subreddit:
            record["subreddit"] = subreddit
        if title:
            record["title"] = title
        return record


class _RedditSearchHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[_SearchLinkBuilder] = []
        self._link_stack: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        attr_map = _attrs_to_dict(attrs)
        href = attr_map.get("href")
        if normalized_tag != "a" or href is None:
            return
        if "/comments/" not in href:
            return
        self.links.append(_SearchLinkBuilder(href=href, source_rank=len(self.links)))
        self._link_stack.append((len(self.links) - 1, normalized_tag))

    def handle_data(self, data: str) -> None:
        if not data.strip() or not self._link_stack:
            return
        self.links[self._link_stack[-1][0]].append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if self._link_stack and self._link_stack[-1][1] == normalized_tag:
            self._link_stack.pop()


def extract_reddit_discussion_records_from_public_html(
    html: str,
    *,
    source_url: str = _DEFAULT_SOURCE_URL,
) -> tuple[dict[str, object], ...]:
    """Extract public post/comment records from static Reddit HTML.

    The parser intentionally understands only public page markup such as ``shreddit-post`` and
    ``shreddit-comment`` elements plus fixture-only ``data-reddit-kind`` tags. It does not consume
    API payloads or login-only resources.
    """

    parser = _RedditDiscussionHtmlParser()
    parser.feed(html)
    parser.close()
    records: list[dict[str, object]] = []
    for builder in parser.records:
        record = builder.to_record(source_url=source_url)
        if record is not None:
            records.append(record)
    return tuple(records)


def extract_reddit_search_results_from_public_html(
    html: str,
    *,
    source_url: str = _DEFAULT_SOURCE_URL,
) -> tuple[dict[str, object], ...]:
    """Extract public Reddit search result links without consuming API payloads."""

    records: list[dict[str, object]] = []
    seen: set[str] = set()

    for record in extract_reddit_discussion_records_from_public_html(
        html,
        source_url=source_url,
    ):
        permalink = _string(record.get("permalink")) or _string(record.get("url"))
        if not _is_public_discussion_url(permalink):
            continue
        key = _canonical_url(permalink)
        if key in seen:
            continue
        seen.add(key)
        result = dict(record)
        result["source_url"] = source_url
        result["result_rank"] = len(records)
        records.append(result)

    parser = _RedditSearchHtmlParser()
    parser.feed(html)
    parser.close()
    for builder in parser.links:
        link_record = builder.to_record(source_url=source_url)
        if link_record is None:
            continue
        permalink = _string(link_record.get("permalink"))
        key = _canonical_url(permalink)
        if key in seen:
            continue
        seen.add(key)
        link_record["result_rank"] = len(records)
        records.append(link_record)

    return tuple(records)


def extract_snapshot_observed_at(html: str) -> datetime | None:
    """Return the first public timestamp exposed by a Reddit page, if any."""

    parser = _ObservedAtHtmlParser()
    parser.feed(html)
    parser.close()
    for raw_value in parser.values:
        parsed = _parse_datetime(raw_value)
        if parsed is not None:
            return parsed
    return None


def _record_kind(tag: str, attrs: dict[str, str]) -> str | None:
    explicit_kind = attrs.get("data-reddit-kind", "").lower()
    test_id = attrs.get("data-testid", "").lower()
    if tag == "shreddit-post" or explicit_kind == "post" or test_id in {"post", "reddit-post"}:
        return "post"
    if (
        tag == "shreddit-comment"
        or explicit_kind == "comment"
        or test_id in {"comment", "reddit-comment"}
    ):
        return "comment"
    return None


def _field_name(tag: str, attrs: dict[str, str], record_kind: str) -> str | None:
    slot = attrs.get("slot", "").lower()
    data_field = attrs.get("data-field", "").lower()
    test_id = attrs.get("data-testid", "").lower()
    if slot == "title" or data_field == "title" or test_id in {"post-title", "title"}:
        return "title"
    if slot in {"text-body", "post-content", "comment", "body"}:
        return "body"
    if data_field in {"selftext", "body"} or test_id in {"post-content", "comment-content"}:
        return "body"
    if tag == "shreddit-comment" and record_kind == "comment":
        return "body"
    return None


def _attrs_to_dict(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in attrs:
        if value is not None:
            normalized[key.lower()] = value.strip()
    return normalized


def _first_attr(attrs: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = attrs.get(key)
        if value:
            return value
    return None


def _created_utc(attrs: dict[str, str]) -> int | None:
    value = _first_attr(attrs, "created-utc", "data-created-utc")
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _created_at(attrs: dict[str, str]) -> datetime | None:
    value = _first_attr(
        attrs,
        "created-timestamp",
        "data-created-at",
        "datetime",
    )
    if value is None:
        return None
    return _parse_datetime(value)


def _parse_datetime(value: str) -> datetime | None:
    normalized = value.strip().replace("Z", "+00:00")
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _int_attr(attrs: dict[str, str], *keys: str) -> int | None:
    value = _first_attr(attrs, *keys)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _normalize_url(value: str | None, source_url: str) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return urljoin(source_url, stripped)


def _is_public_discussion_url(value: str | None) -> bool:
    if value is None:
        return False
    parsed = urlsplit(value)
    path = parsed.path.lower()
    return path.startswith("/r/") and "/comments/" in path


def _subreddit_from_url(value: str | None) -> str | None:
    if value is None:
        return None
    parts = [part for part in urlsplit(value).path.split("/") if part]
    if len(parts) >= 2 and parts[0].lower() == "r":
        return parts[1]
    return None


def _reddit_id_from_url(value: str | None) -> str | None:
    if value is None:
        return None
    parts = [part for part in urlsplit(value).path.split("/") if part]
    if len(parts) >= 4 and parts[0].lower() == "r" and parts[2].lower() == "comments":
        return f"t3_{parts[3]}"
    return None


def _canonical_url(value: str | None) -> str:
    if value is None:
        return ""
    parsed = urlsplit(value)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}/"


def _string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _clean_text(parts: list[str]) -> str | None:
    text = " ".join("".join(parts).split())
    return text or None


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _thing_prefix(kind: str) -> str:
    return "t3" if kind == "post" else "t1"


__all__ = [
    "extract_reddit_discussion_records_from_public_html",
    "extract_reddit_search_results_from_public_html",
    "extract_snapshot_observed_at",
]
