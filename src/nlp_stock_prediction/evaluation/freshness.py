"""Structured freshness and evidence-aging reviews for evaluation hardening."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import cast

from pydantic import Field

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
)
from nlp_stock_prediction.contracts.enums import FreshnessStatus, SourceKind
from nlp_stock_prediction.contracts.evaluation import (
    ArtifactFreshnessReview,
    ArtifactFreshnessReviewStatus,
    EvidenceAgingRecord,
)
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.evaluation.common import aware_utc, digest, slug
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    EvidenceRecord,
    PredictionCandidateRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

DATE_ONLY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DEFAULT_ARTIFACT_MAX_AGE = timedelta(days=7)
DEFAULT_EVIDENCE_MAX_AGE = timedelta(days=14)


@dataclass(frozen=True)
class FreshnessPolicy:
    """Deterministic freshness windows for persisted records."""

    artifact_max_age: timedelta = DEFAULT_ARTIFACT_MAX_AGE
    evidence_max_age: timedelta = DEFAULT_EVIDENCE_MAX_AGE


DEFAULT_FRESHNESS_POLICY = FreshnessPolicy()


@dataclass(frozen=True)
class WrittenFreshnessArtifact:
    """One indexed Phase 7 freshness artifact and its typed payload."""

    artifact: AuditArtifact
    payload: ArtifactFreshnessReviewArtifactPayload | EvidenceAgingSummaryArtifactPayload


class ArtifactFreshnessReviewArtifactPayload(ContractModel):
    """Stable artifact payload for artifact freshness reviews."""

    schema_version: NonEmptyStr = "artifact-freshness-review-artifact.v1"
    run_id: NonEmptyStr
    created_at: AwareDatetime
    artifact_freshness_reviews: tuple[ArtifactFreshnessReview, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)


class EvidenceAgingSummaryArtifactPayload(ContractModel):
    """Stable artifact payload for evidence aging summaries."""

    schema_version: NonEmptyStr = "evidence-aging-summary-artifact.v1"
    run_id: NonEmptyStr
    created_at: AwareDatetime
    evidence_aging_records: tuple[EvidenceAgingRecord, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactTimestampReview:
    """Normalized artifact timestamps and parser limitations."""

    as_of: datetime | None
    observed_at: datetime | None
    metadata: JsonObject
    limitations: tuple[str, ...]
    malformed: bool = False

    @property
    def reference_time(self) -> datetime | None:
        return self.as_of or self.observed_at


def review_artifact_freshness(
    *,
    artifact: ArtifactRecord | None,
    reviewed_at: datetime,
    artifact_id: str | None = None,
    policy: FreshnessPolicy = DEFAULT_FRESHNESS_POLICY,
    expected_sha256: str | None = None,
    source_evidence_ids: tuple[str, ...] = (),
    source_artifact_ids: tuple[str, ...] = (),
) -> ArtifactFreshnessReview:
    """Classify one stored artifact without mutating it."""

    reviewed = aware_utc(reviewed_at, "reviewed_at")
    if artifact is None:
        missing_artifact_id = artifact_id or "missing-artifact"
        return ArtifactFreshnessReview(
            freshness_review_id=_freshness_review_id(missing_artifact_id, reviewed),
            artifact_id=missing_artifact_id,
            artifact_type="unknown",
            reviewed_at=reviewed,
            freshness_status="missing",
            source_evidence_ids=tuple(dict.fromkeys(source_evidence_ids)),
            source_artifact_ids=tuple(dict.fromkeys(source_artifact_ids)),
            limitations=("Artifact was not present in the research database.",),
            metadata={"review_reason": "missing_artifact"},
        )

    created_at = (
        None
        if artifact.created_at is None
        else aware_utc(artifact.created_at, "artifact.created_at")
    )
    timestamp_review = normalize_artifact_timestamps(artifact)
    as_of = timestamp_review.as_of
    observed_at = timestamp_review.observed_at
    limitations = list(timestamp_review.limitations)
    provider = _json_optional_text(artifact.metadata, "provider") or _json_optional_text(
        artifact.metadata, "provider_name"
    )

    timestamp_malformed = timestamp_review.malformed
    if as_of is not None and as_of > reviewed:
        timestamp_malformed = True
        limitations.append("Artifact as_of is after reviewed_at.")
        as_of = None
    if observed_at is not None and observed_at > reviewed:
        timestamp_malformed = True
        limitations.append("Artifact observed_at is after reviewed_at.")
        observed_at = None
    if created_at is not None and created_at > reviewed:
        timestamp_malformed = True
        limitations.append("Artifact created_at is after reviewed_at.")
        created_at = None

    reference_time = as_of or observed_at
    if expected_sha256 is not None and expected_sha256 != artifact.sha256:
        status = "hash_mismatch"
        limitations.append(
            f"Artifact hash mismatch: expected {expected_sha256}, observed {artifact.sha256}."
        )
    elif timestamp_malformed:
        status = "malformed"
    elif reference_time is None:
        status = "unknown"
        limitations.append("Artifact freshness could not be determined from typed timestamps.")
    elif reviewed - reference_time > policy.artifact_max_age:
        status = "stale"
        limitations.append(
            "Artifact freshness window exceeded for "
            f"{artifact.artifact_id}: reviewed_at={reviewed.isoformat()}, "
            f"reference_time={reference_time.isoformat()}."
        )
    else:
        status = "fresh"

    return ArtifactFreshnessReview(
        freshness_review_id=_freshness_review_id(artifact.artifact_id, reviewed),
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type,
        provider=provider,
        produced_by=artifact.produced_by,
        reviewed_at=reviewed,
        created_at=created_at,
        as_of=as_of,
        observed_at=observed_at,
        freshness_status=status,
        sha256=artifact.sha256,
        expected_sha256=expected_sha256,
        source_evidence_ids=tuple(dict.fromkeys(source_evidence_ids)),
        source_artifact_ids=tuple(dict.fromkeys(source_artifact_ids)),
        limitations=tuple(dict.fromkeys(limitations)),
        metadata={
            **timestamp_review.metadata,
            "artifact_path": artifact.path.as_posix(),
            "schema_version": artifact.schema_version,
            "record_count": artifact.record_count,
            "instrument_id": _json_optional_text(artifact.metadata, "instrument_id"),
        },
    )


def review_artifact_file_freshness(
    *,
    artifact: ArtifactRecord | None,
    reviewed_at: datetime,
    repo_root: Path,
    artifact_id: str | None = None,
    policy: FreshnessPolicy = DEFAULT_FRESHNESS_POLICY,
    source_evidence_ids: tuple[str, ...] = (),
    source_artifact_ids: tuple[str, ...] = (),
) -> ArtifactFreshnessReview:
    """Classify an artifact row after checking the current file on disk."""

    if artifact is None:
        return review_artifact_freshness(
            artifact=None,
            artifact_id=artifact_id,
            reviewed_at=reviewed_at,
            policy=policy,
            source_evidence_ids=source_evidence_ids,
            source_artifact_ids=source_artifact_ids,
        )
    reviewed = aware_utc(reviewed_at, "reviewed_at")
    path = artifact.path if artifact.path.is_absolute() else repo_root / artifact.path
    if not path.exists():
        return _file_state_freshness_review(
            artifact=artifact,
            reviewed_at=reviewed,
            status="missing",
            limitations=(f"Artifact file is missing: {path.as_posix()}.",),
            metadata={"review_reason": "missing_file", "artifact_path": path.as_posix()},
            source_evidence_ids=source_evidence_ids,
            source_artifact_ids=source_artifact_ids,
        )
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != artifact.sha256:
        return _file_state_freshness_review(
            artifact=artifact,
            reviewed_at=reviewed,
            status="hash_mismatch",
            limitations=(
                f"Artifact hash mismatch: expected {artifact.sha256}, observed {actual_sha256}.",
            ),
            metadata={"review_reason": "hash_mismatch", "artifact_path": path.as_posix()},
            sha256=actual_sha256,
            expected_sha256=artifact.sha256,
            source_evidence_ids=source_evidence_ids,
            source_artifact_ids=source_artifact_ids,
        )
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _file_state_freshness_review(
                artifact=artifact,
                reviewed_at=reviewed,
                status="malformed",
                limitations=(f"Artifact JSON payload is malformed: {exc}.",),
                metadata={
                    "review_reason": "malformed_file",
                    "artifact_path": path.as_posix(),
                    "schema_version": artifact.schema_version,
                },
                source_evidence_ids=source_evidence_ids,
                source_artifact_ids=source_artifact_ids,
            )
        if not isinstance(payload, dict):
            return _file_state_freshness_review(
                artifact=artifact,
                reviewed_at=reviewed,
                status="malformed",
                limitations=("Artifact JSON payload must be an object.",),
                metadata={
                    "review_reason": "malformed_file",
                    "artifact_path": path.as_posix(),
                    "schema_version": artifact.schema_version,
                },
                source_evidence_ids=source_evidence_ids,
                source_artifact_ids=source_artifact_ids,
            )
    return review_artifact_freshness(
        artifact=artifact,
        reviewed_at=reviewed,
        policy=policy,
        source_evidence_ids=source_evidence_ids,
        source_artifact_ids=source_artifact_ids,
    )


def review_evidence_aging(
    *,
    evidence: EvidenceRecord | None,
    reviewed_at: datetime,
    evidence_id: str | None = None,
    policy: FreshnessPolicy = DEFAULT_FRESHNESS_POLICY,
    replacement_provider: str | None = None,
    replacement_evidence_id: str | None = None,
    source_reliability_note_id: str | None = None,
) -> EvidenceAgingRecord:
    """Classify how one evidence record aged across report/evaluation runs."""

    reviewed = aware_utc(reviewed_at, "reviewed_at")
    if evidence is None:
        missing_evidence_id = evidence_id or "missing-evidence"
        return EvidenceAgingRecord(
            aging_record_id=_aging_record_id(missing_evidence_id, reviewed),
            evidence_id=missing_evidence_id,
            provider="unknown",
            source_type=SourceKind.INTERNAL_ANALYSIS,
            reviewed_at=reviewed,
            age_status="missing",
            freshness_status=FreshnessStatus.MISSING,
            limitations=("Evidence was not present in the research database.",),
            metadata={"review_reason": "missing_evidence"},
        )

    retrieved_at = aware_utc(evidence.retrieved_at, "evidence.retrieved_at")
    published_at = (
        None
        if evidence.published_at is None
        else aware_utc(evidence.published_at, "evidence.published_at")
    )
    source_type, malformed_source_type = _source_kind(evidence.source_type)
    source_freshness = _freshness_status(evidence.freshness_status)
    limitations: list[str] = []
    resolved_replacement_provider = replacement_provider or _json_optional_text(
        evidence.metadata, "replacement_provider"
    )
    resolved_replacement_evidence_id = replacement_evidence_id or _json_optional_text(
        evidence.metadata, "replacement_evidence_id"
    )
    superseded_by = _json_optional_text(evidence.metadata, "superseded_by_evidence_id")
    age_basis = published_at or retrieved_at

    if retrieved_at > reviewed or (published_at is not None and published_at > reviewed):
        age_status = "malformed"
        limitations.append("Evidence timestamp is after reviewed_at.")
        retrieved_at_for_record = None
        published_at_for_record = None
    elif malformed_source_type:
        age_status = "malformed"
        limitations.append(f"Evidence source_type is not recognized: {evidence.source_type}.")
        retrieved_at_for_record = retrieved_at
        published_at_for_record = published_at
    elif resolved_replacement_provider or resolved_replacement_evidence_id:
        age_status = "provider_replaced"
        limitations.append(
            "Evidence provider was replaced; use the replacement source for new reviews."
        )
        retrieved_at_for_record = retrieved_at
        published_at_for_record = published_at
    elif superseded_by is not None:
        age_status = "superseded"
        limitations.append(f"Evidence was superseded by {superseded_by}.")
        retrieved_at_for_record = retrieved_at
        published_at_for_record = published_at
    elif source_freshness == FreshnessStatus.STALE:
        age_status = "stale"
        limitations.append("Evidence source already reported stale freshness.")
        retrieved_at_for_record = retrieved_at
        published_at_for_record = published_at
    elif source_freshness == FreshnessStatus.MISSING:
        age_status = "missing"
        limitations.append("Evidence source reported missing freshness metadata.")
        retrieved_at_for_record = retrieved_at
        published_at_for_record = published_at
    elif reviewed - age_basis > policy.evidence_max_age:
        age_status = "aged_out"
        source_freshness = FreshnessStatus.STALE
        limitations.append(
            "Evidence freshness window exceeded for "
            f"{evidence.evidence_id}: reviewed_at={reviewed.isoformat()}, "
            f"age_basis={age_basis.isoformat()}."
        )
        retrieved_at_for_record = retrieved_at
        published_at_for_record = published_at
    else:
        age_status = "fresh"
        retrieved_at_for_record = retrieved_at
        published_at_for_record = published_at

    if source_freshness == FreshnessStatus.UNKNOWN and age_status == "fresh":
        limitations.append("Evidence source freshness was unknown at review time.")

    return EvidenceAgingRecord(
        aging_record_id=_aging_record_id(evidence.evidence_id, reviewed),
        evidence_id=evidence.evidence_id,
        provider=evidence.provider,
        source_type=source_type,
        retrieved_at=retrieved_at_for_record,
        published_at=published_at_for_record,
        reviewed_at=reviewed,
        age_status=age_status,
        freshness_status=source_freshness,
        source_reliability_note_id=source_reliability_note_id,
        source_artifact_id=evidence.artifact_id,
        replacement_provider=resolved_replacement_provider,
        replacement_evidence_id=resolved_replacement_evidence_id,
        limitations=tuple(dict.fromkeys(limitations)),
        metadata={
            "source_query_id": evidence.source_query_id,
            "url": evidence.url,
            "query": evidence.query,
            "instruments": list(evidence.instruments),
            "superseded_by_evidence_id": superseded_by,
        },
    )


def review_candidate_evidence_aging(
    *,
    store: SQLiteStore,
    candidate: PredictionCandidateRecord,
    reviewed_at: datetime,
    policy: FreshnessPolicy = DEFAULT_FRESHNESS_POLICY,
) -> tuple[EvidenceAgingRecord, ...]:
    records = [
        review_evidence_aging(
            evidence=store.get_evidence(evidence_id),
            evidence_id=evidence_id,
            reviewed_at=reviewed_at,
            policy=policy,
        )
        for evidence_id in dict.fromkeys((*candidate.evidence_for, *candidate.evidence_against))
    ]
    return tuple(records)


def review_candidate_artifact_freshness(
    *,
    store: SQLiteStore,
    candidate: PredictionCandidateRecord,
    reviewed_at: datetime,
    repo_root: Path | None = None,
    policy: FreshnessPolicy = DEFAULT_FRESHNESS_POLICY,
) -> tuple[ArtifactFreshnessReview, ...]:
    linked_artifacts = store.list_candidate_artifact_links(candidate.candidate_id)
    evidence_ids = tuple(dict.fromkeys((*candidate.evidence_for, *candidate.evidence_against)))
    evidence_artifact_ids_by_artifact: dict[str, list[str]] = {}
    for evidence_id in evidence_ids:
        evidence = store.get_evidence(evidence_id)
        if evidence is not None and evidence.artifact_id:
            evidence_artifact_ids_by_artifact.setdefault(evidence.artifact_id, []).append(
                evidence_id
            )
    artifact_ids = tuple(
        dict.fromkeys(
            (
                *candidate.signal_artifacts,
                *(link.artifact_id for link in linked_artifacts),
                *evidence_artifact_ids_by_artifact,
            )
        )
    )
    return tuple(
        (
            review_artifact_file_freshness(
                artifact=store.get_artifact(artifact_id),
                artifact_id=artifact_id,
                reviewed_at=reviewed_at,
                repo_root=repo_root,
                policy=policy,
                source_evidence_ids=tuple(evidence_artifact_ids_by_artifact.get(artifact_id, ())),
            )
            if repo_root is not None
            else review_artifact_freshness(
                artifact=store.get_artifact(artifact_id),
                artifact_id=artifact_id,
                reviewed_at=reviewed_at,
                policy=policy,
                source_evidence_ids=tuple(evidence_artifact_ids_by_artifact.get(artifact_id, ())),
            )
        )
        for artifact_id in artifact_ids
    )


def phase7_freshness_metadata(
    *,
    reviewed_at: datetime,
    evidence_aging_records: tuple[EvidenceAgingRecord, ...],
    artifact_freshness_reviews: tuple[ArtifactFreshnessReview, ...],
) -> JsonObject:
    return {
        "reviewed_at": aware_utc(reviewed_at, "reviewed_at").isoformat(),
        "evidence_aging_records": [
            cast(JsonObject, record.model_dump(mode="json")) for record in evidence_aging_records
        ],
        "artifact_freshness_reviews": [
            cast(JsonObject, review.model_dump(mode="json"))
            for review in artifact_freshness_reviews
        ],
    }


def freshness_limitations(
    *,
    evidence_aging_records: tuple[EvidenceAgingRecord, ...],
    artifact_freshness_reviews: tuple[ArtifactFreshnessReview, ...],
) -> tuple[str, ...]:
    limitations: list[str] = []
    for record in evidence_aging_records:
        if record.age_status == "fresh":
            continue
        limitations.append(f"Evidence aging review {record.evidence_id}: {record.age_status}.")
    for review in artifact_freshness_reviews:
        if review.freshness_status == "fresh":
            continue
        limitations.append(
            f"Artifact freshness review {review.artifact_id}: {review.freshness_status}."
        )
    return tuple(dict.fromkeys(limitations))


def normalize_artifact_timestamps(artifact: ArtifactRecord) -> ArtifactTimestampReview:
    """Normalize artifact as-of/observed timestamps with explicit precedence."""

    limitations: list[str] = []
    metadata: JsonObject = {}
    for key in ("as_of", "latest_usable_bar", "latest_bar"):
        value = artifact.metadata.get(key)
        if isinstance(value, str) and value.strip():
            parsed = _parse_artifact_timestamp(value)
            if parsed is None:
                return ArtifactTimestampReview(
                    as_of=None,
                    observed_at=None,
                    metadata={"as_of_source": key, "invalid_as_of": value},
                    limitations=(f"Artifact {key} timestamp is malformed.",),
                    malformed=True,
                )
            metadata["as_of_source"] = key
            if DATE_ONLY_PATTERN.fullmatch(value.strip()):
                metadata["date_only_as_of_normalized"] = True
                limitations.append("Date-only artifact as_of was normalized to UTC start-of-day.")
            if artifact.created_at is not None and parsed > aware_utc(
                artifact.created_at, "artifact.created_at"
            ):
                return ArtifactTimestampReview(
                    as_of=None,
                    observed_at=None,
                    metadata={**metadata, "invalid_as_of": value},
                    limitations=(f"Artifact {key} timestamp is after created_at.",),
                    malformed=True,
                )
            return ArtifactTimestampReview(
                as_of=parsed,
                observed_at=None,
                metadata=metadata,
                limitations=tuple(limitations),
            )
    for key in ("observed_at", "latest_observation_at"):
        value = artifact.metadata.get(key)
        if isinstance(value, str) and value.strip():
            parsed = _parse_artifact_timestamp(value)
            if parsed is None:
                return ArtifactTimestampReview(
                    as_of=None,
                    observed_at=None,
                    metadata={"observed_at_source": key, "invalid_observed_at": value},
                    limitations=(f"Artifact {key} timestamp is malformed.",),
                    malformed=True,
                )
            metadata["observed_at_source"] = key
            if DATE_ONLY_PATTERN.fullmatch(value.strip()):
                metadata["date_only_observed_at_normalized"] = True
                limitations.append(
                    "Date-only artifact observed_at was normalized to UTC start-of-day."
                )
            return ArtifactTimestampReview(
                as_of=None,
                observed_at=parsed,
                metadata=metadata,
                limitations=tuple(limitations),
            )
    if artifact.created_at is None:
        return ArtifactTimestampReview(
            as_of=None,
            observed_at=None,
            metadata={"timestamp_source": "missing"},
            limitations=("Artifact has no typed as_of, observed_at, or created_at timestamp.",),
        )
    return ArtifactTimestampReview(
        as_of=None,
        observed_at=aware_utc(artifact.created_at, "artifact.created_at"),
        metadata={"timestamp_source": "created_at_fallback"},
        limitations=("Artifact freshness used created_at because as_of/observed_at was absent.",),
    )


def artifact_reference_as_of(artifact: ArtifactRecord) -> tuple[datetime | None, str | None]:
    """Return the target-freezing as_of timestamp or a deterministic exclusion reason."""

    review = normalize_artifact_timestamps(artifact)
    if review.malformed:
        return None, next(iter(review.limitations), "malformed artifact timestamp")
    return review.reference_time, None


def write_artifact_freshness_review_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    reviews: tuple[ArtifactFreshnessReview, ...],
    created_at: datetime,
    tool_run_id: str | None = None,
    artifact_id: str | None = None,
    filename: str | None = None,
    metadata: JsonObject | None = None,
) -> WrittenFreshnessArtifact:
    created = aware_utc(created_at, "created_at")
    payload = ArtifactFreshnessReviewArtifactPayload(
        run_id=run_id,
        created_at=created,
        artifact_freshness_reviews=reviews,
        metadata={} if metadata is None else metadata,
    )
    resolved_artifact_id = artifact_id or _review_artifact_id(
        prefix="artifact-freshness-review",
        run_id=run_id,
        created_at=created,
        record_ids=tuple(review.freshness_review_id for review in reviews),
    )
    artifact = ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=artifact_dir,
        created_at=created,
        produced_by="phase7_freshness_review",
        tool_run_id=tool_run_id,
        schema_version=payload.schema_version,
    ).write_json(
        artifact_id=resolved_artifact_id,
        artifact_type="artifact_freshness_review",
        filename=filename
        or (
            "artifact-freshness/"
            f"{slug(run_id, allow_file_safe_punctuation=True)}-"
            f"{resolved_artifact_id[-8:]}.json"
        ),
        payload=cast(JsonObject, payload.model_dump(mode="json")),
        record_count=len(reviews),
        metadata={
            "run_id": run_id,
            "review_count": len(reviews),
            "freshness_review_ids": [review.freshness_review_id for review in reviews],
        },
    )
    return WrittenFreshnessArtifact(artifact=artifact, payload=payload)


def write_evidence_aging_summary_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    aging_records: tuple[EvidenceAgingRecord, ...],
    created_at: datetime,
    tool_run_id: str | None = None,
    artifact_id: str | None = None,
    filename: str | None = None,
    metadata: JsonObject | None = None,
) -> WrittenFreshnessArtifact:
    created = aware_utc(created_at, "created_at")
    payload = EvidenceAgingSummaryArtifactPayload(
        run_id=run_id,
        created_at=created,
        evidence_aging_records=aging_records,
        metadata={} if metadata is None else metadata,
    )
    resolved_artifact_id = artifact_id or _review_artifact_id(
        prefix="evidence-aging-summary",
        run_id=run_id,
        created_at=created,
        record_ids=tuple(record.aging_record_id for record in aging_records),
    )
    artifact = ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=artifact_dir,
        created_at=created,
        produced_by="phase7_evidence_aging",
        tool_run_id=tool_run_id,
        schema_version=payload.schema_version,
    ).write_json(
        artifact_id=resolved_artifact_id,
        artifact_type="evidence_aging_summary",
        filename=filename
        or (
            "evidence-aging/"
            f"{slug(run_id, allow_file_safe_punctuation=True)}-"
            f"{resolved_artifact_id[-8:]}.json"
        ),
        payload=cast(JsonObject, payload.model_dump(mode="json")),
        record_count=len(aging_records),
        metadata={
            "run_id": run_id,
            "aging_record_count": len(aging_records),
            "aging_record_ids": [record.aging_record_id for record in aging_records],
        },
    )
    return WrittenFreshnessArtifact(artifact=artifact, payload=payload)


def _parse_artifact_timestamp(value: str) -> datetime | None:
    normalized = value.strip()
    if DATE_ONLY_PATTERN.fullmatch(normalized):
        return datetime.combine(date.fromisoformat(normalized), time.min, tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _source_kind(value: str) -> tuple[SourceKind, bool]:
    try:
        return SourceKind(value), False
    except ValueError:
        return SourceKind.INTERNAL_ANALYSIS, True


def _freshness_status(value: str) -> FreshnessStatus:
    try:
        return FreshnessStatus(value)
    except ValueError:
        return FreshnessStatus.UNKNOWN


def _json_optional_text(data: JsonObject, key: str) -> str | None:
    value = data.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _file_state_freshness_review(
    *,
    artifact: ArtifactRecord,
    reviewed_at: datetime,
    status: str,
    limitations: tuple[str, ...],
    metadata: JsonObject,
    source_evidence_ids: tuple[str, ...],
    source_artifact_ids: tuple[str, ...],
    sha256: str | None = None,
    expected_sha256: str | None = None,
) -> ArtifactFreshnessReview:
    created_at = (
        None
        if artifact.created_at is None
        else aware_utc(artifact.created_at, "artifact.created_at")
    )
    return ArtifactFreshnessReview(
        freshness_review_id=_freshness_review_id(artifact.artifact_id, reviewed_at),
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type,
        provider=_json_optional_text(artifact.metadata, "provider")
        or _json_optional_text(artifact.metadata, "provider_name"),
        produced_by=artifact.produced_by,
        reviewed_at=reviewed_at,
        created_at=created_at if created_at is None or created_at <= reviewed_at else None,
        freshness_status=cast(ArtifactFreshnessReviewStatus, status),
        sha256=artifact.sha256 if sha256 is None else sha256,
        expected_sha256=expected_sha256,
        source_evidence_ids=tuple(dict.fromkeys(source_evidence_ids)),
        source_artifact_ids=tuple(dict.fromkeys(source_artifact_ids)),
        limitations=limitations,
        metadata={
            **metadata,
            "record_count": artifact.record_count,
            "instrument_id": _json_optional_text(artifact.metadata, "instrument_id"),
        },
    )


def _freshness_review_id(artifact_id: str, reviewed_at: datetime) -> str:
    return (
        f"freshness-{slug(artifact_id, allow_file_safe_punctuation=True)}-"
        f"{digest(artifact_id + '|' + reviewed_at.isoformat())[:8]}"
    )


def _aging_record_id(evidence_id: str, reviewed_at: datetime) -> str:
    return (
        f"aging-{slug(evidence_id, allow_file_safe_punctuation=True)}-"
        f"{digest(evidence_id + '|' + reviewed_at.isoformat())[:8]}"
    )


def _review_artifact_id(
    *,
    prefix: str,
    run_id: str,
    created_at: datetime,
    record_ids: tuple[str, ...],
) -> str:
    material = "|".join((run_id, created_at.isoformat(), *record_ids))
    run_slug = slug(run_id, allow_file_safe_punctuation=True)
    return f"artifact-{prefix}-{run_slug}-{digest(material)[:8]}"


__all__ = [
    "DEFAULT_FRESHNESS_POLICY",
    "ArtifactFreshnessReviewArtifactPayload",
    "ArtifactTimestampReview",
    "EvidenceAgingSummaryArtifactPayload",
    "FreshnessPolicy",
    "WrittenFreshnessArtifact",
    "artifact_reference_as_of",
    "freshness_limitations",
    "normalize_artifact_timestamps",
    "phase7_freshness_metadata",
    "review_artifact_file_freshness",
    "review_artifact_freshness",
    "review_candidate_artifact_freshness",
    "review_candidate_evidence_aging",
    "review_evidence_aging",
    "write_artifact_freshness_review_artifact",
    "write_evidence_aging_summary_artifact",
]
