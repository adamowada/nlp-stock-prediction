"""Typed row records for local SQLite persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from nlp_stock_prediction.contracts.base import JsonObject


@dataclass(frozen=True)
class InstrumentRecord:
    instrument_id: str
    symbol: str
    asset_class: str
    name: str | None = None
    venue: str | None = None
    aliases: tuple[str, ...] = ()
    provider_ids: tuple[JsonObject, ...] = ()
    related_instruments: tuple[JsonObject, ...] = ()
    tradability_evidence: tuple[JsonObject, ...] = ()
    data_availability: tuple[JsonObject, ...] = ()
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class InstrumentTradabilityEvidenceRecord:
    instrument_id: str
    provider: str
    status: str
    retrieved_at: datetime
    source_query_id: str | None = None
    url: str | None = None
    raw_identifier: str | None = None
    extraction_confidence: float | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class WatchlistRecord:
    watchlist_id: str
    name: str
    description: str | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class WatchlistItemRecord:
    watchlist_id: str
    instrument_id: str
    sort_order: int = 0
    notes: str | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchRunRecord:
    run_id: str
    run_kind: str
    objective: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class ToolRunRecord:
    tool_run_id: str
    tool_name: str
    tool_version: str
    status: str
    started_at: datetime
    run_id: str | None = None
    completed_at: datetime | None = None
    inputs: JsonObject = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error_message: str | None = None


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    artifact_type: str
    path: Path
    sha256: str
    schema_version: str
    tool_run_id: str | None = None
    produced_by: str | None = None
    record_count: int | None = None
    metadata: JsonObject = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class SourceQueryRecord:
    source_query_id: str
    provider: str
    query: str
    retrieved_at: datetime
    url: str | None = None
    tool_run_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source_type: str
    provider: str
    retrieved_at: datetime
    claim: str
    tool_run_id: str | None = None
    source_query_id: str | None = None
    url: str | None = None
    query: str | None = None
    published_at: datetime | None = None
    instruments: tuple[str, ...] = ()
    extraction_confidence: float | None = None
    source_reliability: str | None = None
    freshness_status: str = "unknown"
    artifact_id: str | None = None
    raw_excerpt: str | None = None
    provenance_json: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionCandidateRecord:
    candidate_id: str
    instrument_id: str
    prediction_horizon: str
    prediction_type: str
    scenario: str
    status: str
    run_id: str | None = None
    confidence: float | None = None
    direction: str | None = None
    evidence_for: tuple[str, ...] = ()
    evidence_against: tuple[str, ...] = ()
    signal_artifacts: tuple[str, ...] = ()
    baseline: JsonObject = field(default_factory=dict)
    uncertainty: str | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvidenceLinkRecord:
    candidate_id: str
    evidence_id: str
    relationship: str
    metadata: JsonObject = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class CandidateArtifactLinkRecord:
    candidate_id: str
    artifact_id: str
    relationship: str
    metadata: JsonObject = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class PlanRecord:
    plan_id: str
    slug: str
    title: str
    goal: str
    status: str
    priority: int = 0
    owner_agent: str | None = None
    non_goals: JsonObject = field(default_factory=dict)
    context: JsonObject = field(default_factory=dict)
    superseded_by_plan_id: str | None = None


@dataclass(frozen=True)
class PlanMilestoneRecord:
    milestone_id: str
    plan_id: str
    title: str
    status: str
    sort_order: int = 0
    details: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class PlanAcceptanceCriterionRecord:
    criterion_id: str
    plan_id: str
    description: str
    status: str
    verification_command: str | None = None


@dataclass(frozen=True)
class PlanDecisionRecord:
    decision_id: str
    plan_id: str
    decision: str
    rationale: str
    alternatives_considered: str | None = None
    consequences: str | None = None
    decided_at: datetime | None = None


@dataclass(frozen=True)
class PlanProgressRecord:
    progress_id: str
    plan_id: str
    event_type: str
    summary: str
    details: str | None = None
    linked_artifact_id: str | None = None
    occurred_at: datetime | None = None


@dataclass(frozen=True)
class PlanArtifactLinkRecord:
    plan_id: str
    artifact_id: str
    relationship: str


@dataclass(frozen=True)
class PlanCommitLinkRecord:
    plan_id: str
    commit_sha: str
    relationship: str


__all__ = [
    "ArtifactRecord",
    "CandidateArtifactLinkRecord",
    "CandidateEvidenceLinkRecord",
    "EvidenceRecord",
    "InstrumentRecord",
    "InstrumentTradabilityEvidenceRecord",
    "PlanAcceptanceCriterionRecord",
    "PlanArtifactLinkRecord",
    "PlanCommitLinkRecord",
    "PlanDecisionRecord",
    "PlanMilestoneRecord",
    "PlanProgressRecord",
    "PlanRecord",
    "PredictionCandidateRecord",
    "ResearchRunRecord",
    "SourceQueryRecord",
    "ToolRunRecord",
    "WatchlistItemRecord",
    "WatchlistRecord",
]
