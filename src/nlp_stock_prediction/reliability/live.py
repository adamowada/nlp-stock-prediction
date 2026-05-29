"""Reliability Stage source reliability notes and provider replacement playbooks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    FreshnessStatus,
    ProviderCompatibilityNote,
    ProviderReplacementPlaybook,
    RetrievalMethod,
    SourceProvenance,
    SourceReliabilityNote,
)
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.evaluation import ProviderFamily
from nlp_stock_prediction.contracts.live_validation import (
    require_live_metadata,
    require_live_retrieval_method,
    text_is_non_live,
)
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.storage.records import EvidenceRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

_TRACE_FIELDS = frozenset({"source_url", "permalink", "raw_identifier", "raw_snapshot_id"})

_REQUIRED_FIELDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "market_data": (
        "provider",
        "source_url",
        "raw_identifier",
        "raw_snapshot_id",
        "retrieved_at",
        "observed_at",
        "freshness_status",
        "artifact_type",
        "schema_version",
    ),
    "news": (
        "provider",
        "source_url",
        "raw_identifier",
        "raw_snapshot_id",
        "retrieved_at",
        "published_at",
        "freshness_status",
        "artifact_type",
        "schema_version",
    ),
    "social": (
        "provider",
        "source_url",
        "raw_identifier",
        "raw_snapshot_id",
        "retrieved_at",
        "published_at",
        "freshness_status",
        "artifact_type",
        "schema_version",
    ),
    "fundamentals": (
        "provider",
        "source_url",
        "raw_identifier",
        "raw_snapshot_id",
        "retrieved_at",
        "observed_at",
        "freshness_status",
        "artifact_type",
        "schema_version",
    ),
    "macro": (
        "provider",
        "source_url",
        "raw_identifier",
        "raw_snapshot_id",
        "retrieved_at",
        "observed_at",
        "freshness_status",
        "artifact_type",
        "schema_version",
    ),
    "scraping": (
        "provider",
        "source_url",
        "raw_identifier",
        "raw_snapshot_id",
        "retrieved_at",
        "freshness_status",
        "artifact_type",
        "schema_version",
    ),
    "unknown": ("provider", "retrieved_at"),
}


@dataclass(frozen=True, slots=True)
class ProviderReplacementSpec:
    """Inputs for checking one provider replacement path."""

    provider_family: ProviderFamily
    source_provider: str
    replacement_provider: str
    preserved_fields: tuple[str, ...]
    expected_artifact_type: str | None = None
    replacement_artifact_type: str | None = None
    source_schema_version: str | None = None
    replacement_schema_version: str | None = None
    artifact_schema_versions: tuple[str, ...] = ()
    credential_requirements: tuple[str, ...] = ()
    unsupported_modes: tuple[str, ...] = ()
    source_reliability_note_ids: tuple[str, ...] = ()
    metadata: JsonObject | None = None

    @property
    def required_fields(self) -> tuple[str, ...]:
        return _REQUIRED_FIELDS_BY_FAMILY.get(
            self.provider_family,
            _REQUIRED_FIELDS_BY_FAMILY["unknown"],
        )


@dataclass(frozen=True, slots=True)
class SourceReliabilityPolicy:
    """Scoring policy for live source reliability notes."""

    minimum_confidence: float = 0.6
    medium_confidence: float = 0.75
    high_confidence: float = 0.85

    def rating(
        self,
        *,
        record: EvidenceRecord,
        provenance: SourceProvenance,
        freshness_status: FreshnessStatus,
        observed_at: datetime | None,
        limitations: tuple[str, ...],
    ) -> str:
        confidence = record.extraction_confidence
        if freshness_status == FreshnessStatus.MISSING:
            return "unavailable"
        if confidence is None:
            return "unknown"
        if confidence < self.minimum_confidence:
            return "low"
        if freshness_status in {FreshnessStatus.STALE, FreshnessStatus.UNKNOWN}:
            return "low"
        if provenance.retrieval_method == RetrievalMethod.PUBLIC_SCRAPE:
            return (
                "medium"
                if confidence >= self.high_confidence and observed_at is not None
                else "low"
            )
        if provenance.retrieval_method == RetrievalMethod.OFFICIAL_API:
            return "high" if confidence >= self.high_confidence and not limitations else "medium"
        if (
            record.source_reliability in {"provider_metric", "verified_market_data"}
            and confidence >= self.high_confidence
        ):
            return "high"
        if confidence >= self.medium_confidence:
            return "medium"
        return "low"


DEFAULT_SOURCE_RELIABILITY_POLICY = SourceReliabilityPolicy()


def build_source_reliability_note(record: EvidenceRecord) -> SourceReliabilityNote:
    """Build a live Reliability Stage reliability note from one stored evidence row."""

    provenance = _source_provenance(record)
    if provenance.retrieval_method == RetrievalMethod.FIXTURE:
        raise ValueError("live source reliability notes cannot use fixture retrieval")
    _require_live_source_evidence(record, provenance)

    freshness_status, freshness_limitations = _freshness_status(record, provenance)
    observed_at = provenance.observed_at
    limitations: list[str] = [*freshness_limitations]
    if observed_at is not None and observed_at > record.retrieved_at:
        observed_at = None
        limitations.append("Source observed timestamp was after retrieved_at.")
    if observed_at is None and provenance.retrieval_method in {
        RetrievalMethod.OFFICIAL_API,
        RetrievalMethod.PUBLIC_SCRAPE,
    }:
        limitations.append("Source did not preserve an observed timestamp.")
    if provenance.retrieval_method == RetrievalMethod.PUBLIC_SCRAPE:
        limitations.append("public scraping source may drift and requires markup monitoring.")
    if record.extraction_confidence is None:
        limitations.append("Source extraction confidence was not recorded.")
    elif record.extraction_confidence < 0.6:
        limitations.append("Source extraction confidence is below the reliability threshold.")

    reliability = _source_reliability_rating(
        record=record,
        provenance=provenance,
        freshness_status=freshness_status,
        observed_at=observed_at,
        limitations=tuple(limitations),
    )
    try:
        return SourceReliabilityNote(
            note_id=_source_reliability_note_id(record.evidence_id),
            evidence_id=record.evidence_id,
            provider=provenance.provider_name,
            source_type=provenance.source_kind,
            retrieval_method=provenance.retrieval_method,
            retrieved_at=record.retrieved_at,
            observed_at=observed_at,
            source_url=provenance.source_url,
            permalink=provenance.permalink,
            raw_identifier=provenance.raw_identifier,
            raw_snapshot_id=provenance.raw_snapshot_id,
            freshness_status=freshness_status,
            extraction_confidence=record.extraction_confidence,
            reliability=reliability,
            evidence_ids=(record.evidence_id,),
            related_artifact_ids=((record.artifact_id,) if record.artifact_id else ()),
            limitations=tuple(dict.fromkeys(limitations)),
            metadata={
                "source_query_id": record.source_query_id,
                "source_reliability": record.source_reliability,
                "provider_mode": "live",
                "report_data_mode": "live",
            },
        )
    except ValidationError as exc:
        raise ValueError(
            f"could not build source reliability note for {record.evidence_id}: {exc}"
        ) from exc


def build_source_reliability_notes(
    records: tuple[EvidenceRecord, ...],
) -> tuple[SourceReliabilityNote, ...]:
    """Build reliability notes for all source evidence rows."""

    return tuple(build_source_reliability_note(record) for record in records)


def write_source_reliability_note_artifacts(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    tool_run_id: str | None,
    notes: tuple[SourceReliabilityNote, ...],
    created_at: datetime,
) -> tuple[AuditArtifact, ...]:
    """Write one source reliability note artifact per note."""

    from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex

    artifacts: list[AuditArtifact] = []
    for note in notes:
        artifact_id = (
            "artifact-source-reliability-"
            f"{_stable_digest('|'.join((run_id, tool_run_id or '', note.note_id)))}"
        )
        artifacts.append(
            ArtifactIndex.for_directory(
                store=store,
                repo_root=repo_root,
                base_dir=artifact_dir,
                created_at=created_at,
                produced_by="reliability_source_reliability",
                tool_run_id=tool_run_id,
                schema_version=note.schema_version,
            ).write_json(
                artifact_id=artifact_id,
                artifact_type="source_reliability_note",
                filename=f"source-reliability/{note.note_id}.json",
                payload=cast(JsonObject, note.model_dump(mode="json")),
                record_count=1,
                metadata={
                    "run_id": run_id,
                    "note_id": note.note_id,
                    "evidence_id": note.evidence_id,
                    "provider": note.provider,
                    "reliability": note.reliability,
                    "report_data_mode": "live",
                    "provider_mode": "live",
                },
            )
        )
    return tuple(artifacts)


def build_provider_compatibility_note(
    spec: ProviderReplacementSpec,
    *,
    checked_at: datetime,
) -> ProviderCompatibilityNote:
    """Check whether one replacement provider preserves required provenance."""

    required_fields = spec.required_fields
    preserved_fields = tuple(dict.fromkeys(spec.preserved_fields))
    missing_fields = tuple(field for field in required_fields if field not in preserved_fields)
    limitations: list[str] = []

    if missing_fields:
        limitations.append(
            "Replacement provider does not preserve required fields: "
            + ", ".join(missing_fields)
            + "."
        )
    if _TRACE_FIELDS.intersection(missing_fields):
        limitations.append(
            "Replacement provider is not evaluable because it does not preserve the "
            "required provenance trace."
        )
        status = "not_evaluable"
    elif (
        spec.expected_artifact_type
        and spec.replacement_artifact_type
        and spec.expected_artifact_type != spec.replacement_artifact_type
    ):
        limitations.append(
            "Replacement artifact type does not match the expected provider contract."
        )
        status = "incompatible"
    elif missing_fields:
        status = "compatible_with_limitations"
    else:
        status = "compatible"

    if "freshness_status" in missing_fields:
        limitations.append("Replacement provider has incompatible freshness semantics.")

    replacement_digest = _stable_digest(
        "|".join((spec.source_provider, spec.replacement_provider, spec.provider_family))
    )
    return ProviderCompatibilityNote(
        compatibility_note_id=f"provider-compatibility-{replacement_digest}",
        provider_family=spec.provider_family,
        source_provider=spec.source_provider,
        replacement_provider=spec.replacement_provider,
        checked_at=checked_at,
        compatibility_status=status,
        required_fields=required_fields,
        preserved_fields=preserved_fields,
        missing_fields=missing_fields,
        expected_artifact_type=spec.expected_artifact_type,
        replacement_artifact_type=spec.replacement_artifact_type,
        source_schema_version=spec.source_schema_version,
        replacement_schema_version=spec.replacement_schema_version,
        source_reliability_note_ids=spec.source_reliability_note_ids,
        limitations=tuple(dict.fromkeys(limitations)),
        metadata={
            **({} if spec.metadata is None else spec.metadata),
            "report_data_mode": "live",
            "provider_mode": "live",
        },
    )


def build_provider_replacement_playbook(
    spec: ProviderReplacementSpec,
    *,
    created_at: datetime,
) -> ProviderReplacementPlaybook:
    """Build one operational provider replacement playbook."""

    note = build_provider_compatibility_note(spec, checked_at=created_at)
    if note.compatibility_status not in {"compatible", "compatible_with_limitations"}:
        raise ValueError(
            "provider replacement playbooks require a compatible or limited-compatible note"
        )
    limitations = note.limitations
    replacement_digest = _stable_digest(
        "|".join((spec.source_provider, spec.replacement_provider, spec.provider_family))
    )
    return ProviderReplacementPlaybook(
        playbook_id=f"provider-playbook-{replacement_digest}",
        provider_family=spec.provider_family,
        source_provider=spec.source_provider,
        replacement_provider=spec.replacement_provider,
        created_at=created_at,
        compatibility_notes=(note,),
        required_provenance_fields=spec.required_fields,
        artifact_schema_versions=spec.artifact_schema_versions,
        credential_requirements=spec.credential_requirements,
        unsupported_modes=spec.unsupported_modes,
        limitations=limitations,
        metadata={
            **({} if spec.metadata is None else spec.metadata),
            "provider_id_mapping": {
                spec.source_provider: spec.replacement_provider,
            },
            "freshness_semantics": (
                "replacement must preserve observed/retrieved timestamps or state limitations"
            ),
            "report_data_mode": "live",
            "provider_mode": "live",
        },
    )


def default_provider_replacement_playbooks(
    *,
    created_at: datetime,
) -> tuple[ProviderReplacementPlaybook, ...]:
    """Return the implemented provider replacement playbook catalog."""

    return tuple(
        build_provider_replacement_playbook(spec, created_at=created_at)
        for spec in _DEFAULT_PROVIDER_REPLACEMENT_SPECS
    )


def write_provider_replacement_playbook_artifacts(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    tool_run_id: str | None,
    playbooks: tuple[ProviderReplacementPlaybook, ...],
    created_at: datetime,
) -> tuple[AuditArtifact, ...]:
    """Write provider replacement playbook audit artifacts."""

    from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex

    artifacts: list[AuditArtifact] = []
    for playbook in playbooks:
        artifact_id = (
            "artifact-provider-playbook-"
            f"{_stable_digest('|'.join((run_id, tool_run_id or '', playbook.playbook_id)))}"
        )
        artifacts.append(
            ArtifactIndex.for_directory(
                store=store,
                repo_root=repo_root,
                base_dir=artifact_dir,
                created_at=created_at,
                produced_by="reliability_provider_replacement_playbook",
                tool_run_id=tool_run_id,
                schema_version=playbook.schema_version,
            ).write_json(
                artifact_id=artifact_id,
                artifact_type="provider_replacement_playbook",
                filename=f"provider-playbooks/{playbook.playbook_id}.json",
                payload=cast(JsonObject, playbook.model_dump(mode="json")),
                record_count=len(playbook.compatibility_notes),
                metadata={
                    "run_id": run_id,
                    "playbook_id": playbook.playbook_id,
                    "provider_family": playbook.provider_family,
                    "source_provider": playbook.source_provider,
                    "replacement_provider": playbook.replacement_provider,
                    "report_data_mode": "live",
                    "provider_mode": "live",
                },
            )
        )
    return tuple(artifacts)


def write_reliability_audit_artifacts(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    tool_run_id: str | None,
    evidence_records: tuple[EvidenceRecord, ...],
    created_at: datetime,
) -> tuple[AuditArtifact, ...]:
    """Write reliability note and provider playbook artifacts for a live report."""

    notes = build_source_reliability_notes(evidence_records)
    note_artifacts = write_source_reliability_note_artifacts(
        store=store,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        run_id=run_id,
        tool_run_id=tool_run_id,
        notes=notes,
        created_at=created_at,
    )
    playbook_artifacts = write_provider_replacement_playbook_artifacts(
        store=store,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        run_id=run_id,
        tool_run_id=tool_run_id,
        playbooks=default_provider_replacement_playbooks(created_at=created_at),
        created_at=created_at,
    )
    return (*note_artifacts, *playbook_artifacts)


def _source_provenance(record: EvidenceRecord) -> SourceProvenance:
    try:
        return SourceProvenance.model_validate(record.provenance_json)
    except ValidationError as exc:
        raise ValueError(
            f"source evidence {record.evidence_id} lacks required traceability"
        ) from exc


def _require_live_source_evidence(
    record: EvidenceRecord,
    provenance: SourceProvenance,
) -> None:
    require_live_metadata("source reliability evidence", record.evidence_id, record.metadata)
    require_live_retrieval_method(
        record_type="source reliability evidence",
        record_id=record.evidence_id,
        retrieval_method=provenance.retrieval_method,
    )
    for value in (record.provider, provenance.provider_name, provenance.source_url or ""):
        if text_is_non_live(value):
            raise ValueError(
                f"live source reliability evidence {record.evidence_id} contains non-live "
                f"provider provenance: {value}"
            )


def _freshness_status(
    record: EvidenceRecord,
    provenance: SourceProvenance,
) -> tuple[FreshnessStatus, tuple[str, ...]]:
    limitations: list[str] = []
    try:
        freshness = FreshnessStatus(record.freshness_status)
    except ValueError:
        freshness = FreshnessStatus.UNKNOWN
        limitations.append(
            f"Evidence freshness_status is not recognized: {record.freshness_status}."
        )
    if freshness != provenance.freshness_status:
        limitations.append("Evidence row freshness differs from provenance freshness.")
        if freshness == FreshnessStatus.UNKNOWN:
            freshness = provenance.freshness_status
    if freshness in {FreshnessStatus.STALE, FreshnessStatus.MISSING, FreshnessStatus.UNKNOWN}:
        limitations.append(f"Evidence freshness is {freshness.value}.")
    return freshness, tuple(limitations)


def _source_reliability_rating(
    *,
    record: EvidenceRecord,
    provenance: SourceProvenance,
    freshness_status: FreshnessStatus,
    observed_at: datetime | None,
    limitations: tuple[str, ...],
) -> str:
    return DEFAULT_SOURCE_RELIABILITY_POLICY.rating(
        record=record,
        provenance=provenance,
        freshness_status=freshness_status,
        observed_at=observed_at,
        limitations=limitations,
    )


def _source_reliability_note_id(evidence_id: str) -> str:
    return f"source-reliability-{_stable_digest(evidence_id)}"


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


_DEFAULT_PROVIDER_REPLACEMENT_SPECS = (
    ProviderReplacementSpec(
        provider_family="market_data",
        source_provider="alpha-vantage-market-data",
        replacement_provider="candlecharts-market-data",
        preserved_fields=(
            "provider",
            "source_url",
            "raw_identifier",
            "raw_snapshot_id",
            "retrieved_at",
            "freshness_status",
            "artifact_type",
            "schema_version",
        ),
        expected_artifact_type="market_data",
        replacement_artifact_type="market_data",
        artifact_schema_versions=("research-market-data.v1",),
        credential_requirements=("NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT",),
        unsupported_modes=("widget_only_without_ohlcv",),
    ),
    ProviderReplacementSpec(
        provider_family="news",
        source_provider="ap-news",
        replacement_provider="newsapi",
        preserved_fields=_REQUIRED_FIELDS_BY_FAMILY["news"],
        expected_artifact_type="normalized_evidence",
        replacement_artifact_type="normalized_evidence",
        artifact_schema_versions=("research-news-evidence.v1",),
        credential_requirements=("NEWS_API_KEY",),
    ),
    ProviderReplacementSpec(
        provider_family="social",
        source_provider="x-recent-search",
        replacement_provider="reddit-public-page",
        preserved_fields=tuple(
            field for field in _REQUIRED_FIELDS_BY_FAMILY["social"] if field != "published_at"
        ),
        expected_artifact_type="normalized_evidence",
        replacement_artifact_type="normalized_evidence",
        artifact_schema_versions=("research-social-evidence.v1",),
        credential_requirements=("NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT",),
        unsupported_modes=("private_or_authenticated_pages",),
    ),
    ProviderReplacementSpec(
        provider_family="fundamentals",
        source_provider="alpha-vantage-fundamentals",
        replacement_provider="sec-edgar",
        preserved_fields=_REQUIRED_FIELDS_BY_FAMILY["fundamentals"],
        expected_artifact_type="provider_result",
        replacement_artifact_type="provider_result",
        artifact_schema_versions=("research-fundamentals.v1",),
        credential_requirements=("NLP_STOCK_PREDICTION_SEC_USER_AGENT",),
    ),
    ProviderReplacementSpec(
        provider_family="macro",
        source_provider="fred",
        replacement_provider="alpha-vantage-economic-indicators",
        preserved_fields=_REQUIRED_FIELDS_BY_FAMILY["macro"],
        expected_artifact_type="provider_result",
        replacement_artifact_type="provider_result",
        artifact_schema_versions=("research-macro-context.v1",),
        credential_requirements=("NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY",),
    ),
    ProviderReplacementSpec(
        provider_family="scraping",
        source_provider="reddit-public-page",
        replacement_provider="ap-news",
        preserved_fields=_REQUIRED_FIELDS_BY_FAMILY["scraping"],
        expected_artifact_type="normalized_evidence",
        replacement_artifact_type="normalized_evidence",
        artifact_schema_versions=("research-scrape-evidence.v1",),
        credential_requirements=("NLP_STOCK_PREDICTION_LIVE_USER_AGENT",),
        unsupported_modes=("blocked_by_robots_or_authentication",),
    ),
)


__all__ = [
    "DEFAULT_SOURCE_RELIABILITY_POLICY",
    "ProviderReplacementSpec",
    "SourceReliabilityPolicy",
    "build_provider_compatibility_note",
    "build_provider_replacement_playbook",
    "build_source_reliability_note",
    "build_source_reliability_notes",
    "default_provider_replacement_playbooks",
    "write_provider_replacement_playbook_artifacts",
    "write_reliability_audit_artifacts",
    "write_source_reliability_note_artifacts",
]
