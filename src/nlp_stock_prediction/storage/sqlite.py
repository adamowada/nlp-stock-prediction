"""SQLite-backed operational state for the prediction research assistant."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject

CURRENT_RESEARCH_SCHEMA_VERSION = 2
CURRENT_PLANNING_SCHEMA_VERSION = 1
CURRENT_SCHEMA_VERSION = CURRENT_RESEARCH_SCHEMA_VERSION
DEFAULT_RESEARCH_DATABASE_PATH = Path("data/prediction-research.sqlite3")
DEFAULT_PLANNING_DATABASE_PATH = Path("plans/planning.sqlite3")
DEFAULT_DATABASE_PATH = DEFAULT_RESEARCH_DATABASE_PATH


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
    url: str | None = None
    query: str | None = None
    published_at: datetime | None = None
    instruments: tuple[str, ...] = ()
    extraction_confidence: float | None = None
    source_reliability: str | None = None
    freshness_status: str = "unknown"
    artifact_id: str | None = None
    raw_excerpt: str | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionCandidateRecord:
    candidate_id: str
    instrument_id: str
    prediction_horizon: str
    prediction_type: str
    scenario: str
    status: str
    confidence: float | None = None
    direction: str | None = None
    evidence_for: tuple[str, ...] = ()
    evidence_against: tuple[str, ...] = ()
    signal_artifacts: tuple[str, ...] = ()
    baseline: JsonObject = field(default_factory=dict)
    uncertainty: str | None = None
    metadata: JsonObject = field(default_factory=dict)


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


class SQLiteStore:
    """Repository wrapper around the local ignored research SQLite database."""

    def __init__(self, path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            _initialize_connection(connection)

    def schema_version(self) -> int:
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_migrations"
            ).fetchone()
            if row is None or row["version"] is None:
                return 0
            return int(row["version"])

    def upsert_instrument(self, record: InstrumentRecord) -> None:
        _validate_required(record.instrument_id, "instrument_id")
        _validate_required(record.symbol, "symbol")
        _validate_required(record.asset_class, "asset_class")
        now = _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO instruments (
                    instrument_id, symbol, name, asset_class, venue, aliases_json,
                    provider_ids_json, related_instruments_json, tradability_evidence_json,
                    data_availability_json, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(instrument_id) DO UPDATE SET
                    symbol = excluded.symbol,
                    name = excluded.name,
                    asset_class = excluded.asset_class,
                    venue = excluded.venue,
                    aliases_json = excluded.aliases_json,
                    provider_ids_json = excluded.provider_ids_json,
                    related_instruments_json = excluded.related_instruments_json,
                    tradability_evidence_json = excluded.tradability_evidence_json,
                    data_availability_json = excluded.data_availability_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.instrument_id,
                    record.symbol,
                    record.name,
                    record.asset_class,
                    record.venue,
                    _dump_json_array(record.aliases),
                    _dump_json_object_array(record.provider_ids),
                    _dump_json_object_array(record.related_instruments),
                    _dump_json_object_array(record.tradability_evidence),
                    _dump_json_object_array(record.data_availability),
                    _dump_json(record.metadata),
                    _format_datetime(now),
                    _format_datetime(now),
                ),
            )

    def get_instrument(self, instrument_id: str) -> InstrumentRecord | None:
        _validate_required(instrument_id, "instrument_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM instruments WHERE instrument_id = ?", (instrument_id,)
            ).fetchone()
        if row is None:
            return None
        return _instrument_from_row(row)

    def upsert_research_run(self, record: ResearchRunRecord) -> None:
        _validate_required(record.run_id, "run_id")
        _validate_required(record.run_kind, "run_kind")
        _validate_required(record.objective, "objective")
        _validate_required(record.status, "status")
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO research_runs (
                    run_id, run_kind, objective, status, started_at, completed_at,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    run_kind = excluded.run_kind,
                    objective = excluded.objective,
                    status = excluded.status,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.run_id,
                    record.run_kind,
                    record.objective,
                    record.status,
                    _format_datetime(record.started_at),
                    _format_optional_datetime(record.completed_at),
                    _dump_json(record.metadata),
                    _format_datetime(_utc_now()),
                    _format_datetime(_utc_now()),
                ),
            )

    def get_research_run(self, run_id: str) -> ResearchRunRecord | None:
        _validate_required(run_id, "run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM research_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return _research_run_from_row(row)

    def record_tool_run(self, record: ToolRunRecord) -> None:
        _validate_required(record.tool_run_id, "tool_run_id")
        _validate_required(record.tool_name, "tool_name")
        _validate_required(record.tool_version, "tool_version")
        _validate_required(record.status, "status")
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO tool_runs (
                    tool_run_id, run_id, tool_name, tool_version, status, inputs_json,
                    started_at, completed_at, warnings_json, error_message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_run_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    tool_name = excluded.tool_name,
                    tool_version = excluded.tool_version,
                    status = excluded.status,
                    inputs_json = excluded.inputs_json,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at,
                    warnings_json = excluded.warnings_json,
                    error_message = excluded.error_message
                """,
                (
                    record.tool_run_id,
                    record.run_id,
                    record.tool_name,
                    record.tool_version,
                    record.status,
                    _dump_json(record.inputs),
                    _format_datetime(record.started_at),
                    _format_optional_datetime(record.completed_at),
                    _dump_json_array(record.warnings),
                    record.error_message,
                    _format_datetime(_utc_now()),
                ),
            )

    def get_tool_run(self, tool_run_id: str) -> ToolRunRecord | None:
        _validate_required(tool_run_id, "tool_run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM tool_runs WHERE tool_run_id = ?", (tool_run_id,)
            ).fetchone()
        if row is None:
            return None
        return _tool_run_from_row(row)

    def record_artifact(self, record: ArtifactRecord) -> None:
        _validate_required(record.artifact_id, "artifact_id")
        _validate_required(record.artifact_type, "artifact_type")
        _validate_required(record.sha256, "sha256")
        _validate_required(record.schema_version, "schema_version")
        created_at = record.created_at or _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, tool_run_id, artifact_type, path, sha256,
                    schema_version, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    tool_run_id = excluded.tool_run_id,
                    artifact_type = excluded.artifact_type,
                    path = excluded.path,
                    sha256 = excluded.sha256,
                    schema_version = excluded.schema_version,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.artifact_id,
                    record.tool_run_id,
                    record.artifact_type,
                    str(record.path),
                    record.sha256,
                    record.schema_version,
                    _dump_json(record.metadata),
                    _format_datetime(created_at),
                ),
            )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        _validate_required(artifact_id, "artifact_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            return None
        return _artifact_from_row(row)

    def record_source_query(self, record: SourceQueryRecord) -> None:
        _validate_required(record.source_query_id, "source_query_id")
        _validate_required(record.provider, "provider")
        _validate_required(record.query, "query")
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO source_queries (
                    source_query_id, tool_run_id, provider, query, url,
                    retrieved_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_query_id) DO UPDATE SET
                    tool_run_id = excluded.tool_run_id,
                    provider = excluded.provider,
                    query = excluded.query,
                    url = excluded.url,
                    retrieved_at = excluded.retrieved_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.source_query_id,
                    record.tool_run_id,
                    record.provider,
                    record.query,
                    record.url,
                    _format_datetime(record.retrieved_at),
                    _dump_json(record.metadata),
                ),
            )

    def get_source_query(self, source_query_id: str) -> SourceQueryRecord | None:
        _validate_required(source_query_id, "source_query_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM source_queries WHERE source_query_id = ?", (source_query_id,)
            ).fetchone()
        if row is None:
            return None
        return _source_query_from_row(row)

    def record_evidence(self, record: EvidenceRecord) -> None:
        _validate_required(record.evidence_id, "evidence_id")
        _validate_required(record.source_type, "source_type")
        _validate_required(record.provider, "provider")
        _validate_required(record.claim, "claim")
        _validate_confidence(record.extraction_confidence)
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO evidence_items (
                    evidence_id, source_type, provider, url, query, retrieved_at,
                    published_at, instruments_json, claim, extraction_confidence,
                    source_reliability, freshness_status, artifact_id, raw_excerpt,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    provider = excluded.provider,
                    url = excluded.url,
                    query = excluded.query,
                    retrieved_at = excluded.retrieved_at,
                    published_at = excluded.published_at,
                    instruments_json = excluded.instruments_json,
                    claim = excluded.claim,
                    extraction_confidence = excluded.extraction_confidence,
                    source_reliability = excluded.source_reliability,
                    freshness_status = excluded.freshness_status,
                    artifact_id = excluded.artifact_id,
                    raw_excerpt = excluded.raw_excerpt,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.evidence_id,
                    record.source_type,
                    record.provider,
                    record.url,
                    record.query,
                    _format_datetime(record.retrieved_at),
                    _format_optional_datetime(record.published_at),
                    _dump_json_array(record.instruments),
                    record.claim,
                    record.extraction_confidence,
                    record.source_reliability,
                    record.freshness_status,
                    record.artifact_id,
                    record.raw_excerpt,
                    _dump_json(record.metadata),
                ),
            )

    def get_evidence(self, evidence_id: str) -> EvidenceRecord | None:
        _validate_required(evidence_id, "evidence_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM evidence_items WHERE evidence_id = ?", (evidence_id,)
            ).fetchone()
        if row is None:
            return None
        return _evidence_from_row(row)

    def upsert_prediction_candidate(self, record: PredictionCandidateRecord) -> None:
        _validate_required(record.candidate_id, "candidate_id")
        _validate_required(record.instrument_id, "instrument_id")
        _validate_required(record.prediction_horizon, "prediction_horizon")
        _validate_required(record.prediction_type, "prediction_type")
        _validate_required(record.scenario, "scenario")
        _validate_required(record.status, "status")
        _validate_confidence(record.confidence)
        now = _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO prediction_candidates (
                    candidate_id, instrument_id, prediction_horizon, prediction_type,
                    scenario, direction, confidence, status, evidence_for_json,
                    evidence_against_json, signal_artifacts_json, baseline_json,
                    uncertainty, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    instrument_id = excluded.instrument_id,
                    prediction_horizon = excluded.prediction_horizon,
                    prediction_type = excluded.prediction_type,
                    scenario = excluded.scenario,
                    direction = excluded.direction,
                    confidence = excluded.confidence,
                    status = excluded.status,
                    evidence_for_json = excluded.evidence_for_json,
                    evidence_against_json = excluded.evidence_against_json,
                    signal_artifacts_json = excluded.signal_artifacts_json,
                    baseline_json = excluded.baseline_json,
                    uncertainty = excluded.uncertainty,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.candidate_id,
                    record.instrument_id,
                    record.prediction_horizon,
                    record.prediction_type,
                    record.scenario,
                    record.direction,
                    record.confidence,
                    record.status,
                    _dump_json_array(record.evidence_for),
                    _dump_json_array(record.evidence_against),
                    _dump_json_array(record.signal_artifacts),
                    _dump_json(record.baseline),
                    record.uncertainty,
                    _dump_json(record.metadata),
                    _format_datetime(now),
                    _format_datetime(now),
                ),
            )

    def get_prediction_candidate(self, candidate_id: str) -> PredictionCandidateRecord | None:
        _validate_required(candidate_id, "candidate_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM prediction_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            return None
        return _prediction_candidate_from_row(row)


ResearchSQLiteStore = SQLiteStore


class PlanningSQLiteStore:
    """Repository wrapper around the tracked planning SQLite database."""

    def __init__(self, path: Path = DEFAULT_PLANNING_DATABASE_PATH) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            _initialize_planning_connection(connection)

    def schema_version(self) -> int:
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_migrations"
            ).fetchone()
            if row is None or row["version"] is None:
                return 0
            return int(row["version"])

    def upsert_plan(self, record: PlanRecord) -> None:
        _validate_required(record.plan_id, "plan_id")
        _validate_required(record.slug, "slug")
        _validate_required(record.title, "title")
        _validate_required(record.goal, "goal")
        _validate_required(record.status, "status")
        now = _utc_now()
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            connection.execute(
                """
                INSERT INTO plans (
                    plan_id, slug, title, goal, non_goals_json, context_json, status,
                    priority, owner_agent, superseded_by_plan_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    slug = excluded.slug,
                    title = excluded.title,
                    goal = excluded.goal,
                    non_goals_json = excluded.non_goals_json,
                    context_json = excluded.context_json,
                    status = excluded.status,
                    priority = excluded.priority,
                    owner_agent = excluded.owner_agent,
                    superseded_by_plan_id = excluded.superseded_by_plan_id,
                    updated_at = excluded.updated_at
                """,
                (
                    record.plan_id,
                    record.slug,
                    record.title,
                    record.goal,
                    _dump_json(record.non_goals),
                    _dump_json(record.context),
                    record.status,
                    record.priority,
                    record.owner_agent,
                    record.superseded_by_plan_id,
                    _format_datetime(now),
                    _format_datetime(now),
                ),
            )

    def get_plan_by_slug(self, slug: str) -> PlanRecord | None:
        _validate_required(slug, "slug")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            row = connection.execute("SELECT * FROM plans WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            return None
        return _plan_from_row(row)

    def add_plan_decision(self, record: PlanDecisionRecord) -> None:
        _validate_required(record.decision_id, "decision_id")
        _validate_required(record.plan_id, "plan_id")
        _validate_required(record.decision, "decision")
        _validate_required(record.rationale, "rationale")
        decided_at = record.decided_at or _utc_now()
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            connection.execute(
                """
                INSERT INTO plan_decisions (
                    decision_id, plan_id, decision, rationale, alternatives_considered,
                    consequences, decided_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.decision_id,
                    record.plan_id,
                    record.decision,
                    record.rationale,
                    record.alternatives_considered,
                    record.consequences,
                    _format_datetime(decided_at),
                ),
            )

    def list_plan_decisions(self, plan_id: str) -> tuple[PlanDecisionRecord, ...]:
        _validate_required(plan_id, "plan_id")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM plan_decisions
                WHERE plan_id = ?
                ORDER BY decided_at, decision_id
                """,
                (plan_id,),
            ).fetchall()
        return tuple(_plan_decision_from_row(row) for row in rows)

    def add_plan_progress(self, record: PlanProgressRecord) -> None:
        _validate_required(record.progress_id, "progress_id")
        _validate_required(record.plan_id, "plan_id")
        _validate_required(record.event_type, "event_type")
        _validate_required(record.summary, "summary")
        occurred_at = record.occurred_at or _utc_now()
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            connection.execute(
                """
                INSERT INTO plan_progress_events (
                    progress_id, plan_id, event_type, summary, details,
                    linked_artifact_id, occurred_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.progress_id,
                    record.plan_id,
                    record.event_type,
                    record.summary,
                    record.details,
                    record.linked_artifact_id,
                    _format_datetime(occurred_at),
                ),
            )

    def list_plan_progress(self, plan_id: str) -> tuple[PlanProgressRecord, ...]:
        _validate_required(plan_id, "plan_id")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM plan_progress_events
                WHERE plan_id = ?
                ORDER BY occurred_at, progress_id
                """,
                (plan_id,),
            ).fetchall()
        return tuple(_plan_progress_from_row(row) for row in rows)

    def upsert_plan_milestone(self, record: PlanMilestoneRecord) -> None:
        _validate_required(record.milestone_id, "milestone_id")
        _validate_required(record.plan_id, "plan_id")
        _validate_required(record.title, "title")
        _validate_required(record.status, "status")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            connection.execute(
                """
                INSERT INTO plan_milestones (
                    milestone_id, plan_id, title, status, sort_order, details_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(milestone_id) DO UPDATE SET
                    plan_id = excluded.plan_id,
                    title = excluded.title,
                    status = excluded.status,
                    sort_order = excluded.sort_order,
                    details_json = excluded.details_json
                """,
                (
                    record.milestone_id,
                    record.plan_id,
                    record.title,
                    record.status,
                    record.sort_order,
                    _dump_json(record.details),
                ),
            )

    def list_plan_milestones(self, plan_id: str) -> tuple[PlanMilestoneRecord, ...]:
        _validate_required(plan_id, "plan_id")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM plan_milestones
                WHERE plan_id = ?
                ORDER BY sort_order, milestone_id
                """,
                (plan_id,),
            ).fetchall()
        return tuple(_plan_milestone_from_row(row) for row in rows)

    def upsert_plan_acceptance_criterion(self, record: PlanAcceptanceCriterionRecord) -> None:
        _validate_required(record.criterion_id, "criterion_id")
        _validate_required(record.plan_id, "plan_id")
        _validate_required(record.description, "description")
        _validate_required(record.status, "status")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            connection.execute(
                """
                INSERT INTO plan_acceptance_criteria (
                    criterion_id, plan_id, description, status, verification_command
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(criterion_id) DO UPDATE SET
                    plan_id = excluded.plan_id,
                    description = excluded.description,
                    status = excluded.status,
                    verification_command = excluded.verification_command
                """,
                (
                    record.criterion_id,
                    record.plan_id,
                    record.description,
                    record.status,
                    record.verification_command,
                ),
            )

    def list_plan_acceptance_criteria(
        self, plan_id: str
    ) -> tuple[PlanAcceptanceCriterionRecord, ...]:
        _validate_required(plan_id, "plan_id")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM plan_acceptance_criteria
                WHERE plan_id = ?
                ORDER BY criterion_id
                """,
                (plan_id,),
            ).fetchall()
        return tuple(_plan_acceptance_criterion_from_row(row) for row in rows)

    def link_plan_artifact(self, record: PlanArtifactLinkRecord) -> None:
        _validate_required(record.plan_id, "plan_id")
        _validate_required(record.artifact_id, "artifact_id")
        _validate_required(record.relationship, "relationship")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            connection.execute(
                """
                INSERT OR IGNORE INTO plan_artifact_links (
                    plan_id, artifact_id, relationship
                )
                VALUES (?, ?, ?)
                """,
                (record.plan_id, record.artifact_id, record.relationship),
            )

    def list_plan_artifact_links(self, plan_id: str) -> tuple[PlanArtifactLinkRecord, ...]:
        _validate_required(plan_id, "plan_id")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM plan_artifact_links
                WHERE plan_id = ?
                ORDER BY artifact_id, relationship
                """,
                (plan_id,),
            ).fetchall()
        return tuple(_plan_artifact_link_from_row(row) for row in rows)

    def link_plan_commit(self, record: PlanCommitLinkRecord) -> None:
        _validate_required(record.plan_id, "plan_id")
        _validate_required(record.commit_sha, "commit_sha")
        _validate_required(record.relationship, "relationship")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            connection.execute(
                """
                INSERT OR IGNORE INTO plan_commit_links (
                    plan_id, commit_sha, relationship
                )
                VALUES (?, ?, ?)
                """,
                (record.plan_id, record.commit_sha, record.relationship),
            )

    def list_plan_commit_links(self, plan_id: str) -> tuple[PlanCommitLinkRecord, ...]:
        _validate_required(plan_id, "plan_id")
        with self.connect() as connection:
            _ensure_planning_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM plan_commit_links
                WHERE plan_id = ?
                ORDER BY commit_sha, relationship
                """,
                (plan_id,),
            ).fetchall()
        return tuple(_plan_commit_link_from_row(row) for row in rows)


def initialize_database(path: Path = DEFAULT_RESEARCH_DATABASE_PATH) -> SQLiteStore:
    return initialize_research_database(path)


def initialize_research_database(path: Path = DEFAULT_RESEARCH_DATABASE_PATH) -> SQLiteStore:
    store = SQLiteStore(path)
    store.initialize()
    return store


def initialize_planning_database(
    path: Path = DEFAULT_PLANNING_DATABASE_PATH,
) -> PlanningSQLiteStore:
    store = PlanningSQLiteStore(path)
    store.initialize()
    return store


def _initialize_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_RESEARCH_SCHEMA_SQL)
    _migrate_research_schema(connection)
    connection.execute(
        """
        INSERT OR IGNORE INTO schema_migrations(version, name, applied_at)
        VALUES (?, ?, ?)
        """,
        (
            CURRENT_RESEARCH_SCHEMA_VERSION,
            "instrument_provenance_research_schema",
            _format_datetime(_utc_now()),
        ),
    )


def _migrate_research_schema(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(instruments)").fetchall()
    }
    migrations = {
        "provider_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "related_instruments_json": "TEXT NOT NULL DEFAULT '[]'",
        "tradability_evidence_json": "TEXT NOT NULL DEFAULT '[]'",
        "data_availability_json": "TEXT NOT NULL DEFAULT '[]'",
    }
    for column, definition in migrations.items():
        if column not in columns:
            connection.execute(f"ALTER TABLE instruments ADD COLUMN {column} {definition}")


def _initialize_planning_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_PLANNING_SCHEMA_SQL)
    connection.execute(
        """
        INSERT OR IGNORE INTO schema_migrations(version, name, applied_at)
        VALUES (?, ?, ?)
        """,
        (
            CURRENT_PLANNING_SCHEMA_VERSION,
            "initial_prediction_planning_schema",
            _format_datetime(_utc_now()),
        ),
    )


def _ensure_initialized(connection: sqlite3.Connection) -> None:
    try:
        row = connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
    except sqlite3.OperationalError as exc:
        raise RuntimeError("SQLite prediction research database is not initialized") from exc
    if row is None or row["version"] != CURRENT_RESEARCH_SCHEMA_VERSION:
        raise RuntimeError("SQLite prediction research database schema is not current")


def _ensure_planning_initialized(connection: sqlite3.Connection) -> None:
    try:
        row = connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
    except sqlite3.OperationalError as exc:
        raise RuntimeError("SQLite prediction planning database is not initialized") from exc
    if row is None or row["version"] != CURRENT_PLANNING_SCHEMA_VERSION:
        raise RuntimeError("SQLite prediction planning database schema is not current")


def _validate_required(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _validate_confidence(value: float | None) -> None:
    if value is None:
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence values must be between 0.0 and 1.0")


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetimes stored in SQLite must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _format_optional_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _format_datetime(value)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return _parse_datetime(value)


def _dump_json(value: JsonObject) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _dump_json_array(value: Iterable[str]) -> str:
    return json.dumps(tuple(value), sort_keys=True, separators=(",", ":"))


def _dump_json_object_array(value: Iterable[JsonObject]) -> str:
    return json.dumps(tuple(value), sort_keys=True, separators=(",", ":"))


def _load_json_object(value: str) -> JsonObject:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("stored JSON value must be an object")
    return cast(JsonObject, loaded)


def _load_string_tuple(value: str) -> tuple[str, ...]:
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError("stored JSON value must be an array")
    if not all(isinstance(item, str) for item in loaded):
        raise ValueError("stored JSON array must contain strings")
    return tuple(loaded)


def _load_json_object_tuple(value: str) -> tuple[JsonObject, ...]:
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError("stored JSON value must be an array")
    if not all(isinstance(item, dict) for item in loaded):
        raise ValueError("stored JSON array must contain objects")
    return tuple(cast(JsonObject, item) for item in loaded)


def _row_text(row: sqlite3.Row, column: str) -> str:
    return cast(str, row[column])


def _row_optional_text(row: sqlite3.Row, column: str) -> str | None:
    return cast(str | None, row[column])


def _row_optional_float(row: sqlite3.Row, column: str) -> float | None:
    return cast(float | None, row[column])


def _instrument_from_row(row: sqlite3.Row) -> InstrumentRecord:
    return InstrumentRecord(
        instrument_id=_row_text(row, "instrument_id"),
        symbol=_row_text(row, "symbol"),
        name=_row_optional_text(row, "name"),
        asset_class=_row_text(row, "asset_class"),
        venue=_row_optional_text(row, "venue"),
        aliases=_load_string_tuple(_row_text(row, "aliases_json")),
        provider_ids=_load_json_object_tuple(_row_text(row, "provider_ids_json")),
        related_instruments=_load_json_object_tuple(_row_text(row, "related_instruments_json")),
        tradability_evidence=_load_json_object_tuple(_row_text(row, "tradability_evidence_json")),
        data_availability=_load_json_object_tuple(_row_text(row, "data_availability_json")),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _artifact_from_row(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=_row_text(row, "artifact_id"),
        tool_run_id=_row_optional_text(row, "tool_run_id"),
        artifact_type=_row_text(row, "artifact_type"),
        path=Path(_row_text(row, "path")),
        sha256=_row_text(row, "sha256"),
        schema_version=_row_text(row, "schema_version"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
        created_at=_parse_datetime(_row_text(row, "created_at")),
    )


def _research_run_from_row(row: sqlite3.Row) -> ResearchRunRecord:
    return ResearchRunRecord(
        run_id=_row_text(row, "run_id"),
        run_kind=_row_text(row, "run_kind"),
        objective=_row_text(row, "objective"),
        status=_row_text(row, "status"),
        started_at=_parse_datetime(_row_text(row, "started_at")),
        completed_at=_parse_optional_datetime(_row_optional_text(row, "completed_at")),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _tool_run_from_row(row: sqlite3.Row) -> ToolRunRecord:
    return ToolRunRecord(
        tool_run_id=_row_text(row, "tool_run_id"),
        run_id=_row_optional_text(row, "run_id"),
        tool_name=_row_text(row, "tool_name"),
        tool_version=_row_text(row, "tool_version"),
        status=_row_text(row, "status"),
        started_at=_parse_datetime(_row_text(row, "started_at")),
        completed_at=_parse_optional_datetime(_row_optional_text(row, "completed_at")),
        inputs=_load_json_object(_row_text(row, "inputs_json")),
        warnings=_load_string_tuple(_row_text(row, "warnings_json")),
        error_message=_row_optional_text(row, "error_message"),
    )


def _source_query_from_row(row: sqlite3.Row) -> SourceQueryRecord:
    return SourceQueryRecord(
        source_query_id=_row_text(row, "source_query_id"),
        tool_run_id=_row_optional_text(row, "tool_run_id"),
        provider=_row_text(row, "provider"),
        query=_row_text(row, "query"),
        url=_row_optional_text(row, "url"),
        retrieved_at=_parse_datetime(_row_text(row, "retrieved_at")),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _evidence_from_row(row: sqlite3.Row) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=_row_text(row, "evidence_id"),
        source_type=_row_text(row, "source_type"),
        provider=_row_text(row, "provider"),
        url=_row_optional_text(row, "url"),
        query=_row_optional_text(row, "query"),
        retrieved_at=_parse_datetime(_row_text(row, "retrieved_at")),
        published_at=_parse_optional_datetime(_row_optional_text(row, "published_at")),
        instruments=_load_string_tuple(_row_text(row, "instruments_json")),
        claim=_row_text(row, "claim"),
        extraction_confidence=_row_optional_float(row, "extraction_confidence"),
        source_reliability=_row_optional_text(row, "source_reliability"),
        freshness_status=_row_text(row, "freshness_status"),
        artifact_id=_row_optional_text(row, "artifact_id"),
        raw_excerpt=_row_optional_text(row, "raw_excerpt"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _prediction_candidate_from_row(row: sqlite3.Row) -> PredictionCandidateRecord:
    return PredictionCandidateRecord(
        candidate_id=_row_text(row, "candidate_id"),
        instrument_id=_row_text(row, "instrument_id"),
        prediction_horizon=_row_text(row, "prediction_horizon"),
        prediction_type=_row_text(row, "prediction_type"),
        scenario=_row_text(row, "scenario"),
        direction=_row_optional_text(row, "direction"),
        confidence=_row_optional_float(row, "confidence"),
        status=_row_text(row, "status"),
        evidence_for=_load_string_tuple(_row_text(row, "evidence_for_json")),
        evidence_against=_load_string_tuple(_row_text(row, "evidence_against_json")),
        signal_artifacts=_load_string_tuple(_row_text(row, "signal_artifacts_json")),
        baseline=_load_json_object(_row_text(row, "baseline_json")),
        uncertainty=_row_optional_text(row, "uncertainty"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _plan_from_row(row: sqlite3.Row) -> PlanRecord:
    return PlanRecord(
        plan_id=_row_text(row, "plan_id"),
        slug=_row_text(row, "slug"),
        title=_row_text(row, "title"),
        goal=_row_text(row, "goal"),
        non_goals=_load_json_object(_row_text(row, "non_goals_json")),
        context=_load_json_object(_row_text(row, "context_json")),
        status=_row_text(row, "status"),
        priority=int(row["priority"]),
        owner_agent=_row_optional_text(row, "owner_agent"),
        superseded_by_plan_id=_row_optional_text(row, "superseded_by_plan_id"),
    )


def _plan_decision_from_row(row: sqlite3.Row) -> PlanDecisionRecord:
    return PlanDecisionRecord(
        decision_id=_row_text(row, "decision_id"),
        plan_id=_row_text(row, "plan_id"),
        decision=_row_text(row, "decision"),
        rationale=_row_text(row, "rationale"),
        alternatives_considered=_row_optional_text(row, "alternatives_considered"),
        consequences=_row_optional_text(row, "consequences"),
        decided_at=_parse_datetime(_row_text(row, "decided_at")),
    )


def _plan_progress_from_row(row: sqlite3.Row) -> PlanProgressRecord:
    return PlanProgressRecord(
        progress_id=_row_text(row, "progress_id"),
        plan_id=_row_text(row, "plan_id"),
        event_type=_row_text(row, "event_type"),
        summary=_row_text(row, "summary"),
        details=_row_optional_text(row, "details"),
        linked_artifact_id=_row_optional_text(row, "linked_artifact_id"),
        occurred_at=_parse_datetime(_row_text(row, "occurred_at")),
    )


def _plan_milestone_from_row(row: sqlite3.Row) -> PlanMilestoneRecord:
    return PlanMilestoneRecord(
        milestone_id=_row_text(row, "milestone_id"),
        plan_id=_row_text(row, "plan_id"),
        title=_row_text(row, "title"),
        status=_row_text(row, "status"),
        sort_order=int(row["sort_order"]),
        details=_load_json_object(_row_text(row, "details_json")),
    )


def _plan_acceptance_criterion_from_row(row: sqlite3.Row) -> PlanAcceptanceCriterionRecord:
    return PlanAcceptanceCriterionRecord(
        criterion_id=_row_text(row, "criterion_id"),
        plan_id=_row_text(row, "plan_id"),
        description=_row_text(row, "description"),
        status=_row_text(row, "status"),
        verification_command=_row_optional_text(row, "verification_command"),
    )


def _plan_artifact_link_from_row(row: sqlite3.Row) -> PlanArtifactLinkRecord:
    return PlanArtifactLinkRecord(
        plan_id=_row_text(row, "plan_id"),
        artifact_id=_row_text(row, "artifact_id"),
        relationship=_row_text(row, "relationship"),
    )


def _plan_commit_link_from_row(row: sqlite3.Row) -> PlanCommitLinkRecord:
    return PlanCommitLinkRecord(
        plan_id=_row_text(row, "plan_id"),
        commit_sha=_row_text(row, "commit_sha"),
        relationship=_row_text(row, "relationship"),
    )


_RESEARCH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL CHECK(length(name) > 0),
    applied_at TEXT NOT NULL CHECK(length(applied_at) > 0)
);

CREATE TABLE IF NOT EXISTS instruments (
    instrument_id TEXT PRIMARY KEY CHECK(length(instrument_id) > 0),
    symbol TEXT NOT NULL CHECK(length(symbol) > 0),
    name TEXT,
    asset_class TEXT NOT NULL CHECK(length(asset_class) > 0),
    venue TEXT,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    provider_ids_json TEXT NOT NULL DEFAULT '[]',
    related_instruments_json TEXT NOT NULL DEFAULT '[]',
    tradability_evidence_json TEXT NOT NULL DEFAULT '[]',
    data_availability_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_instruments_symbol ON instruments(symbol);
CREATE INDEX IF NOT EXISTS idx_instruments_asset_class ON instruments(asset_class);

CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY CHECK(length(run_id) > 0),
    run_kind TEXT NOT NULL CHECK(length(run_kind) > 0),
    objective TEXT NOT NULL CHECK(length(objective) > 0),
    status TEXT NOT NULL CHECK(length(status) > 0),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_research_runs_status ON research_runs(status);

CREATE TABLE IF NOT EXISTS tool_runs (
    tool_run_id TEXT PRIMARY KEY CHECK(length(tool_run_id) > 0),
    run_id TEXT REFERENCES research_runs(run_id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL CHECK(length(tool_name) > 0),
    tool_version TEXT NOT NULL CHECK(length(tool_version) > 0),
    status TEXT NOT NULL CHECK(length(status) > 0),
    inputs_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    warnings_json TEXT NOT NULL DEFAULT '[]',
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_runs_run_id ON tool_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_runs_tool_name ON tool_runs(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_runs_status ON tool_runs(status);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY CHECK(length(artifact_id) > 0),
    tool_run_id TEXT REFERENCES tool_runs(tool_run_id) ON DELETE SET NULL,
    artifact_type TEXT NOT NULL CHECK(length(artifact_type) > 0),
    path TEXT NOT NULL CHECK(length(path) > 0),
    sha256 TEXT NOT NULL CHECK(length(sha256) > 0),
    schema_version TEXT NOT NULL CHECK(length(schema_version) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_tool_run_id ON artifacts(tool_run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_path ON artifacts(path);
CREATE INDEX IF NOT EXISTS idx_artifacts_sha256 ON artifacts(sha256);

CREATE TABLE IF NOT EXISTS source_queries (
    source_query_id TEXT PRIMARY KEY CHECK(length(source_query_id) > 0),
    tool_run_id TEXT REFERENCES tool_runs(tool_run_id) ON DELETE SET NULL,
    provider TEXT NOT NULL CHECK(length(provider) > 0),
    query TEXT NOT NULL CHECK(length(query) > 0),
    url TEXT,
    retrieved_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_source_queries_provider ON source_queries(provider);
CREATE INDEX IF NOT EXISTS idx_source_queries_retrieved_at ON source_queries(retrieved_at);

CREATE TABLE IF NOT EXISTS evidence_items (
    evidence_id TEXT PRIMARY KEY CHECK(length(evidence_id) > 0),
    source_type TEXT NOT NULL CHECK(length(source_type) > 0),
    provider TEXT NOT NULL CHECK(length(provider) > 0),
    url TEXT,
    query TEXT,
    retrieved_at TEXT NOT NULL,
    published_at TEXT,
    instruments_json TEXT NOT NULL DEFAULT '[]',
    claim TEXT NOT NULL CHECK(length(claim) > 0),
    extraction_confidence REAL CHECK(
        extraction_confidence IS NULL
        OR (extraction_confidence >= 0.0 AND extraction_confidence <= 1.0)
    ),
    source_reliability TEXT,
    freshness_status TEXT NOT NULL DEFAULT 'unknown',
    artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
    raw_excerpt TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_evidence_provider ON evidence_items(provider);
CREATE INDEX IF NOT EXISTS idx_evidence_source_type ON evidence_items(source_type);
CREATE INDEX IF NOT EXISTS idx_evidence_retrieved_at ON evidence_items(retrieved_at);
CREATE INDEX IF NOT EXISTS idx_evidence_freshness ON evidence_items(freshness_status);

CREATE TABLE IF NOT EXISTS prediction_candidates (
    candidate_id TEXT PRIMARY KEY CHECK(length(candidate_id) > 0),
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id) ON DELETE RESTRICT,
    prediction_horizon TEXT NOT NULL CHECK(length(prediction_horizon) > 0),
    prediction_type TEXT NOT NULL CHECK(length(prediction_type) > 0),
    scenario TEXT NOT NULL CHECK(length(scenario) > 0),
    direction TEXT,
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    status TEXT NOT NULL CHECK(length(status) > 0),
    evidence_for_json TEXT NOT NULL DEFAULT '[]',
    evidence_against_json TEXT NOT NULL DEFAULT '[]',
    signal_artifacts_json TEXT NOT NULL DEFAULT '[]',
    baseline_json TEXT NOT NULL DEFAULT '{}',
    uncertainty TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prediction_candidates_instrument
ON prediction_candidates(instrument_id);
CREATE INDEX IF NOT EXISTS idx_prediction_candidates_status
ON prediction_candidates(status);
"""

_PLANNING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL CHECK(length(name) > 0),
    applied_at TEXT NOT NULL CHECK(length(applied_at) > 0)
);

CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY CHECK(length(plan_id) > 0),
    slug TEXT NOT NULL UNIQUE CHECK(length(slug) > 0),
    title TEXT NOT NULL CHECK(length(title) > 0),
    goal TEXT NOT NULL CHECK(length(goal) > 0),
    non_goals_json TEXT NOT NULL DEFAULT '{}',
    context_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK(length(status) > 0),
    priority INTEGER NOT NULL DEFAULT 0,
    owner_agent TEXT,
    superseded_by_plan_id TEXT REFERENCES plans(plan_id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status);
CREATE INDEX IF NOT EXISTS idx_plans_priority ON plans(priority);

CREATE TABLE IF NOT EXISTS plan_milestones (
    milestone_id TEXT PRIMARY KEY CHECK(length(milestone_id) > 0),
    plan_id TEXT NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
    title TEXT NOT NULL CHECK(length(title) > 0),
    status TEXT NOT NULL CHECK(length(status) > 0),
    sort_order INTEGER NOT NULL DEFAULT 0,
    details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_plan_milestones_plan_id ON plan_milestones(plan_id);

CREATE TABLE IF NOT EXISTS plan_acceptance_criteria (
    criterion_id TEXT PRIMARY KEY CHECK(length(criterion_id) > 0),
    plan_id TEXT NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
    description TEXT NOT NULL CHECK(length(description) > 0),
    status TEXT NOT NULL CHECK(length(status) > 0),
    verification_command TEXT
);

CREATE INDEX IF NOT EXISTS idx_plan_acceptance_plan_id
ON plan_acceptance_criteria(plan_id);

CREATE TABLE IF NOT EXISTS plan_decisions (
    decision_id TEXT PRIMARY KEY CHECK(length(decision_id) > 0),
    plan_id TEXT NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
    decision TEXT NOT NULL CHECK(length(decision) > 0),
    rationale TEXT NOT NULL CHECK(length(rationale) > 0),
    alternatives_considered TEXT,
    consequences TEXT,
    decided_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plan_decisions_plan_id ON plan_decisions(plan_id);

CREATE TABLE IF NOT EXISTS plan_progress_events (
    progress_id TEXT PRIMARY KEY CHECK(length(progress_id) > 0),
    plan_id TEXT NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL CHECK(length(event_type) > 0),
    summary TEXT NOT NULL CHECK(length(summary) > 0),
    details TEXT,
    linked_artifact_id TEXT,
    occurred_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plan_progress_plan_id ON plan_progress_events(plan_id);
CREATE INDEX IF NOT EXISTS idx_plan_progress_occurred_at ON plan_progress_events(occurred_at);

CREATE TABLE IF NOT EXISTS plan_artifact_links (
    plan_id TEXT NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL CHECK(length(artifact_id) > 0),
    relationship TEXT NOT NULL CHECK(length(relationship) > 0),
    PRIMARY KEY(plan_id, artifact_id, relationship)
);

CREATE TABLE IF NOT EXISTS plan_commit_links (
    plan_id TEXT NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
    commit_sha TEXT NOT NULL CHECK(length(commit_sha) > 0),
    relationship TEXT NOT NULL CHECK(length(relationship) > 0),
    PRIMARY KEY(plan_id, commit_sha, relationship)
);
"""


__all__ = [
    "CURRENT_PLANNING_SCHEMA_VERSION",
    "CURRENT_RESEARCH_SCHEMA_VERSION",
    "CURRENT_SCHEMA_VERSION",
    "DEFAULT_DATABASE_PATH",
    "DEFAULT_PLANNING_DATABASE_PATH",
    "DEFAULT_RESEARCH_DATABASE_PATH",
    "ArtifactRecord",
    "EvidenceRecord",
    "InstrumentRecord",
    "PlanAcceptanceCriterionRecord",
    "PlanArtifactLinkRecord",
    "PlanCommitLinkRecord",
    "PlanDecisionRecord",
    "PlanMilestoneRecord",
    "PlanProgressRecord",
    "PlanRecord",
    "PlanningSQLiteStore",
    "PredictionCandidateRecord",
    "ResearchRunRecord",
    "ResearchSQLiteStore",
    "SQLiteStore",
    "SourceQueryRecord",
    "ToolRunRecord",
    "initialize_database",
    "initialize_planning_database",
    "initialize_research_database",
]
