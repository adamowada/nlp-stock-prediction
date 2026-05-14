"""SQLite-backed operational state for the prediction research assistant."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    CalibrationRunRecord,
    CalibrationSliceRecord,
    CandidateArtifactLinkRecord,
    CandidateEvidenceLinkRecord,
    EvidenceRecord,
    InstrumentRecord,
    InstrumentTradabilityEvidenceRecord,
    OutcomeArtifactLinkRecord,
    OutcomeEvaluationArtifactLinkRecord,
    OutcomeEvaluationEvidenceLinkRecord,
    OutcomeEvidenceLinkRecord,
    PlanAcceptanceCriterionRecord,
    PlanArtifactLinkRecord,
    PlanCommitLinkRecord,
    PlanDecisionRecord,
    PlanMilestoneRecord,
    PlanProgressRecord,
    PlanRecord,
    PredictionCandidateRecord,
    PredictionEvaluationRecord,
    PredictionOutcomeEvaluationRecord,
    PredictionOutcomeRecord,
    ResearchRunRecord,
    SourceQueryRecord,
    ToolRunRecord,
    WatchlistItemRecord,
    WatchlistRecord,
)
from nlp_stock_prediction.storage.run_graph import (
    fetch_artifact_rows,
    fetch_evidence_rows,
    fetch_prediction_candidate_rows,
    fetch_source_query_rows,
    fetch_tool_run_rows,
)

CURRENT_RESEARCH_SCHEMA_VERSION = 6
CURRENT_PLANNING_SCHEMA_VERSION = 1
CURRENT_SCHEMA_VERSION = CURRENT_RESEARCH_SCHEMA_VERSION
DEFAULT_RESEARCH_DATABASE_PATH = Path("data/prediction-research.sqlite3")
DEFAULT_PLANNING_DATABASE_PATH = Path("plans/planning.sqlite3")
DEFAULT_DATABASE_PATH = DEFAULT_RESEARCH_DATABASE_PATH


class SQLiteStore:
    """Repository wrapper around the local ignored research SQLite database."""

    def __init__(self, path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.path = path
        self._active_connection: sqlite3.Connection | None = None

    @contextmanager
    def connect(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        if self._active_connection is not None:
            yield self._active_connection
            return
        connection = _open_sqlite_connection(self.path, create=create)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if self._active_connection is not None:
            yield
            return
        with self.connect() as connection:
            _ensure_initialized(connection)
            self._active_connection = connection
            try:
                yield
            finally:
                self._active_connection = None

    def initialize(self) -> None:
        with self.connect(create=True) as connection:
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
            _replace_instrument_children(connection, record, now)

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

    def list_instruments(self) -> tuple[InstrumentRecord, ...]:
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                "SELECT * FROM instruments ORDER BY asset_class, symbol, instrument_id"
            ).fetchall()
        return tuple(_instrument_from_row(row) for row in rows)

    def find_instruments_by_symbol_or_alias(self, value: str) -> tuple[InstrumentRecord, ...]:
        _validate_required(value, "value")
        normalized = _normalize_lookup(value)
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT DISTINCT i.* FROM instruments AS i
                LEFT JOIN instrument_aliases AS a
                    ON a.instrument_id = i.instrument_id
                WHERE lower(i.symbol) = ?
                   OR a.alias_lower = ?
                ORDER BY i.asset_class, i.symbol, i.instrument_id
                """,
                (normalized, normalized),
            ).fetchall()
        return tuple(_instrument_from_row(row) for row in rows)

    def find_instrument_by_provider_id(
        self, provider: str, namespace: str, identifier: str
    ) -> InstrumentRecord | None:
        _validate_required(provider, "provider")
        _validate_required(namespace, "namespace")
        _validate_required(identifier, "identifier")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                """
                SELECT i.* FROM instruments AS i
                INNER JOIN instrument_provider_ids AS p
                    ON p.instrument_id = i.instrument_id
                WHERE p.provider_lower = ?
                  AND p.namespace_lower = ?
                  AND lower(p.identifier) = ?
                ORDER BY i.instrument_id
                LIMIT 1
                """,
                (
                    _normalize_lookup(provider),
                    _normalize_lookup(namespace),
                    _normalize_lookup(identifier),
                ),
            ).fetchone()
        if row is None:
            return None
        return _instrument_from_row(row)

    def list_instruments_by_asset_class(self, asset_class: str) -> tuple[InstrumentRecord, ...]:
        _validate_required(asset_class, "asset_class")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM instruments
                WHERE lower(asset_class) = ?
                ORDER BY symbol, instrument_id
                """,
                (_normalize_lookup(asset_class),),
            ).fetchall()
        return tuple(_instrument_from_row(row) for row in rows)

    def append_tradability_evidence(self, record: InstrumentTradabilityEvidenceRecord) -> None:
        _validate_required(record.instrument_id, "instrument_id")
        _validate_required(record.provider, "provider")
        _validate_required(record.status, "status")
        _validate_confidence(record.extraction_confidence)
        with self.connect() as connection:
            _ensure_initialized(connection)
            _insert_tradability_evidence(connection, record)

    def get_latest_tradability_evidence(
        self, instrument_id: str, provider: str | None = None
    ) -> InstrumentTradabilityEvidenceRecord | None:
        _validate_required(instrument_id, "instrument_id")
        params: tuple[str, ...]
        provider_filter = ""
        if provider is None:
            params = (instrument_id,)
        else:
            _validate_required(provider, "provider")
            provider_filter = "AND provider_lower = ?"
            params = (instrument_id, _normalize_lookup(provider))
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                f"""
                SELECT * FROM instrument_tradability_evidence
                WHERE instrument_id = ?
                {provider_filter}
                ORDER BY retrieved_at DESC, evidence_rowid DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        return _tradability_evidence_from_row(row)

    def upsert_watchlist(self, record: WatchlistRecord) -> None:
        _validate_required(record.watchlist_id, "watchlist_id")
        _validate_required(record.name, "name")
        now = _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO watchlists (
                    watchlist_id, name, description, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(watchlist_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.watchlist_id,
                    record.name,
                    record.description,
                    _dump_json(record.metadata),
                    _format_datetime(now),
                    _format_datetime(now),
                ),
            )

    def get_watchlist(self, watchlist_id: str) -> WatchlistRecord | None:
        _validate_required(watchlist_id, "watchlist_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM watchlists WHERE watchlist_id = ?", (watchlist_id,)
            ).fetchone()
        if row is None:
            return None
        return _watchlist_from_row(row)

    def list_watchlists(self) -> tuple[WatchlistRecord, ...]:
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                "SELECT * FROM watchlists ORDER BY name, watchlist_id"
            ).fetchall()
        return tuple(_watchlist_from_row(row) for row in rows)

    def upsert_watchlist_item(self, record: WatchlistItemRecord) -> None:
        _validate_required(record.watchlist_id, "watchlist_id")
        _validate_required(record.instrument_id, "instrument_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO watchlist_items (
                    watchlist_id, instrument_id, sort_order, notes, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(watchlist_id, instrument_id) DO UPDATE SET
                    sort_order = excluded.sort_order,
                    notes = excluded.notes,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.watchlist_id,
                    record.instrument_id,
                    record.sort_order,
                    record.notes,
                    _dump_json(record.metadata),
                    _format_datetime(_utc_now()),
                ),
            )

    def list_watchlist_items(self, watchlist_id: str) -> tuple[WatchlistItemRecord, ...]:
        _validate_required(watchlist_id, "watchlist_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM watchlist_items
                WHERE watchlist_id = ?
                ORDER BY sort_order, instrument_id
                """,
                (watchlist_id,),
            ).fetchall()
        return tuple(_watchlist_item_from_row(row) for row in rows)

    def list_watchlist_instruments(self, watchlist_id: str) -> tuple[InstrumentRecord, ...]:
        _validate_required(watchlist_id, "watchlist_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT i.* FROM instruments AS i
                INNER JOIN watchlist_items AS wi
                    ON wi.instrument_id = i.instrument_id
                WHERE wi.watchlist_id = ?
                ORDER BY wi.sort_order, i.instrument_id
                """,
                (watchlist_id,),
            ).fetchall()
        return tuple(_instrument_from_row(row) for row in rows)

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

    def list_tool_runs_for_run(self, run_id: str) -> tuple[ToolRunRecord, ...]:
        _validate_required(run_id, "run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = fetch_tool_run_rows(connection, run_id)
        return tuple(_tool_run_from_row(row) for row in rows)

    def record_artifact(self, record: ArtifactRecord) -> None:
        _validate_required(record.artifact_id, "artifact_id")
        _validate_required(record.artifact_type, "artifact_type")
        _validate_required(record.sha256, "sha256")
        _validate_required(record.schema_version, "schema_version")
        _validate_relative_artifact_path(record.path)
        if record.produced_by is not None:
            _validate_required(record.produced_by, "produced_by")
        if record.record_count is not None and record.record_count < 0:
            raise ValueError("artifact record_count must be non-negative")
        created_at = record.created_at or _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, tool_run_id, artifact_type, path, sha256,
                    schema_version, produced_by, record_count, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    tool_run_id = excluded.tool_run_id,
                    artifact_type = excluded.artifact_type,
                    path = excluded.path,
                    sha256 = excluded.sha256,
                    schema_version = excluded.schema_version,
                    produced_by = excluded.produced_by,
                    record_count = excluded.record_count,
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (
                    record.artifact_id,
                    record.tool_run_id,
                    record.artifact_type,
                    str(record.path),
                    record.sha256,
                    record.schema_version,
                    record.produced_by,
                    record.record_count,
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

    def list_artifacts(self, run_id: str) -> tuple[ArtifactRecord, ...]:
        return self.list_artifacts_for_run(run_id)

    def list_artifacts_for_run(self, run_id: str) -> tuple[ArtifactRecord, ...]:
        _validate_required(run_id, "run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = fetch_artifact_rows(connection, run_id)
        return tuple(_artifact_from_row(row) for row in rows)

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

    def list_source_queries(self, run_id: str) -> tuple[SourceQueryRecord, ...]:
        return self.list_source_queries_for_run(run_id)

    def list_source_queries_for_run(self, run_id: str) -> tuple[SourceQueryRecord, ...]:
        _validate_required(run_id, "run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = fetch_source_query_rows(connection, run_id)
        return tuple(_source_query_from_row(row) for row in rows)

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
                    evidence_id, tool_run_id, source_query_id, source_type, provider,
                    url, query, retrieved_at, published_at, instruments_json, claim,
                    extraction_confidence, source_reliability, freshness_status,
                    artifact_id, raw_excerpt, provenance_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO UPDATE SET
                    tool_run_id = excluded.tool_run_id,
                    source_query_id = excluded.source_query_id,
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
                    provenance_json = excluded.provenance_json,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.evidence_id,
                    record.tool_run_id,
                    record.source_query_id,
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
                    _dump_json(record.provenance_json),
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

    def list_evidence(self, run_id: str) -> tuple[EvidenceRecord, ...]:
        return self.list_evidence_for_run(run_id)

    def list_evidence_for_run(self, run_id: str) -> tuple[EvidenceRecord, ...]:
        _validate_required(run_id, "run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = fetch_evidence_rows(connection, run_id)
        return tuple(_evidence_from_row(row) for row in rows)

    def delete_tool_run_outputs(self, tool_run_id: str) -> None:
        """Delete run-graph rows produced by one failed tool run."""

        _validate_required(tool_run_id, "tool_run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                DELETE FROM calibration_runs
                WHERE tool_run_id = ?
                   OR artifact_id IN (
                        SELECT artifact_id FROM artifacts WHERE tool_run_id = ?
                   )
                """,
                (tool_run_id, tool_run_id),
            )
            connection.execute(
                """
                DELETE FROM prediction_evaluations
                WHERE artifact_id IN (
                    SELECT artifact_id FROM artifacts WHERE tool_run_id = ?
                )
                """,
                (tool_run_id,),
            )
            connection.execute(
                """
                DELETE FROM prediction_outcome_evaluations
                WHERE artifact_id IN (
                    SELECT artifact_id FROM artifacts WHERE tool_run_id = ?
                )
                   OR outcome_evaluation_id IN (
                        SELECT outcome_evaluation_id
                        FROM outcome_evaluation_artifact_links
                        WHERE artifact_id IN (
                            SELECT artifact_id FROM artifacts WHERE tool_run_id = ?
                        )
                   )
                """,
                (tool_run_id, tool_run_id),
            )
            connection.execute(
                """
                DELETE FROM calibration_runs
                WHERE EXISTS (
                    SELECT 1
                    FROM json_each(calibration_runs.source_outcome_evaluation_ids_json) AS source
                    LEFT JOIN prediction_outcome_evaluations AS evaluation
                        ON evaluation.outcome_evaluation_id = source.value
                    WHERE evaluation.outcome_evaluation_id IS NULL
                )
                """
            )
            connection.execute(
                """
                DELETE FROM prediction_outcomes
                WHERE outcome_id IN (
                    SELECT outcome_id FROM outcome_artifact_links
                    WHERE artifact_id IN (
                        SELECT artifact_id FROM artifacts WHERE tool_run_id = ?
                    )
                )
                """,
                (tool_run_id,),
            )
            connection.execute(
                """
                DELETE FROM candidate_artifact_links
                WHERE artifact_id IN (
                    SELECT artifact_id FROM artifacts WHERE tool_run_id = ?
                )
                """,
                (tool_run_id,),
            )
            connection.execute(
                """
                DELETE FROM candidate_evidence_links
                WHERE evidence_id IN (
                    SELECT evidence_id FROM evidence_items
                    WHERE tool_run_id = ?
                       OR source_query_id IN (
                            SELECT source_query_id FROM source_queries
                            WHERE tool_run_id = ?
                       )
                       OR artifact_id IN (
                            SELECT artifact_id FROM artifacts WHERE tool_run_id = ?
                       )
                )
                """,
                (tool_run_id, tool_run_id, tool_run_id),
            )
            connection.execute(
                """
                DELETE FROM evidence_items
                WHERE tool_run_id = ?
                   OR source_query_id IN (
                        SELECT source_query_id FROM source_queries WHERE tool_run_id = ?
                   )
                   OR artifact_id IN (
                        SELECT artifact_id FROM artifacts WHERE tool_run_id = ?
                   )
                """,
                (tool_run_id, tool_run_id, tool_run_id),
            )
            connection.execute("DELETE FROM source_queries WHERE tool_run_id = ?", (tool_run_id,))
            connection.execute("DELETE FROM artifacts WHERE tool_run_id = ?", (tool_run_id,))
            connection.execute("DELETE FROM tool_runs WHERE tool_run_id = ?", (tool_run_id,))

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
                    candidate_id, run_id, instrument_id, prediction_horizon,
                    prediction_type, scenario, direction, confidence, status,
                    evidence_for_json, evidence_against_json, signal_artifacts_json,
                    baseline_json, uncertainty, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    run_id = excluded.run_id,
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
                    record.run_id,
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
            self._sync_candidate_links(connection, record)

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

    def list_prediction_candidates(self, run_id: str) -> tuple[PredictionCandidateRecord, ...]:
        return self.list_prediction_candidates_for_run(run_id)

    def list_candidates_for_run(self, run_id: str) -> tuple[PredictionCandidateRecord, ...]:
        return self.list_prediction_candidates_for_run(run_id)

    def list_prediction_candidates_for_run(
        self, run_id: str
    ) -> tuple[PredictionCandidateRecord, ...]:
        _validate_required(run_id, "run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = fetch_prediction_candidate_rows(connection, run_id)
        return tuple(_prediction_candidate_from_row(row) for row in rows)

    def link_candidate_evidence(self, record: CandidateEvidenceLinkRecord) -> None:
        _validate_required(record.candidate_id, "candidate_id")
        _validate_required(record.evidence_id, "evidence_id")
        _validate_required(record.relationship, "relationship")
        created_at = record.created_at or _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO candidate_evidence_links (
                    candidate_id, evidence_id, relationship, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id, evidence_id, relationship) DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (
                    record.candidate_id,
                    record.evidence_id,
                    record.relationship,
                    _dump_json(record.metadata),
                    _format_datetime(created_at),
                ),
            )

    def list_candidate_evidence_links(
        self, candidate_id: str
    ) -> tuple[CandidateEvidenceLinkRecord, ...]:
        _validate_required(candidate_id, "candidate_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM candidate_evidence_links
                WHERE candidate_id = ?
                ORDER BY relationship, evidence_id
                """,
                (candidate_id,),
            ).fetchall()
        return tuple(_candidate_evidence_link_from_row(row) for row in rows)

    def link_candidate_artifact(self, record: CandidateArtifactLinkRecord) -> None:
        _validate_required(record.candidate_id, "candidate_id")
        _validate_required(record.artifact_id, "artifact_id")
        _validate_required(record.relationship, "relationship")
        created_at = record.created_at or _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO candidate_artifact_links (
                    candidate_id, artifact_id, relationship, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id, artifact_id, relationship) DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (
                    record.candidate_id,
                    record.artifact_id,
                    record.relationship,
                    _dump_json(record.metadata),
                    _format_datetime(created_at),
                ),
            )

    def list_candidate_artifact_links(
        self, candidate_id: str
    ) -> tuple[CandidateArtifactLinkRecord, ...]:
        _validate_required(candidate_id, "candidate_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM candidate_artifact_links
                WHERE candidate_id = ?
                ORDER BY relationship, artifact_id
                """,
                (candidate_id,),
            ).fetchall()
        return tuple(_candidate_artifact_link_from_row(row) for row in rows)

    def record_prediction_evaluation(self, record: PredictionEvaluationRecord) -> None:
        _validate_required(record.evaluation_id, "evaluation_id")
        _validate_required(record.candidate_id, "candidate_id")
        _validate_required(record.instrument_id, "instrument_id")
        _validate_required(record.symbol, "symbol")
        _validate_required(record.prediction_type, "prediction_type")
        _validate_required(record.horizon, "horizon")
        _validate_required(record.status, "status")
        _validate_confidence(record.score)
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO prediction_evaluations (
                    evaluation_id, run_id, candidate_id, instrument_id, symbol, created_at,
                    prediction_type, horizon, direction, status, score,
                    baseline_comparison_json, evidence_counts_json, signal_counts_json,
                    artifact_id, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    candidate_id = excluded.candidate_id,
                    instrument_id = excluded.instrument_id,
                    symbol = excluded.symbol,
                    created_at = excluded.created_at,
                    prediction_type = excluded.prediction_type,
                    horizon = excluded.horizon,
                    direction = excluded.direction,
                    status = excluded.status,
                    score = excluded.score,
                    baseline_comparison_json = excluded.baseline_comparison_json,
                    evidence_counts_json = excluded.evidence_counts_json,
                    signal_counts_json = excluded.signal_counts_json,
                    artifact_id = excluded.artifact_id,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.evaluation_id,
                    record.run_id,
                    record.candidate_id,
                    record.instrument_id,
                    record.symbol,
                    _format_datetime(record.created_at),
                    record.prediction_type,
                    record.horizon,
                    record.direction,
                    record.status,
                    record.score,
                    _dump_json(record.baseline_comparison),
                    _dump_json(record.evidence_counts),
                    _dump_json(record.signal_counts),
                    record.artifact_id,
                    _dump_json(record.metadata),
                ),
            )

    def get_prediction_evaluation(self, evaluation_id: str) -> PredictionEvaluationRecord | None:
        _validate_required(evaluation_id, "evaluation_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM prediction_evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
        if row is None:
            return None
        return _prediction_evaluation_from_row(row)

    def list_prediction_evaluations_for_run(
        self, run_id: str
    ) -> tuple[PredictionEvaluationRecord, ...]:
        _validate_required(run_id, "run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM prediction_evaluations
                WHERE run_id = ?
                ORDER BY created_at, evaluation_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(_prediction_evaluation_from_row(row) for row in rows)

    def list_prediction_evaluations_for_candidate(
        self, candidate_id: str
    ) -> tuple[PredictionEvaluationRecord, ...]:
        _validate_required(candidate_id, "candidate_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM prediction_evaluations
                WHERE candidate_id = ?
                ORDER BY created_at, evaluation_id
                """,
                (candidate_id,),
            ).fetchall()
        return tuple(_prediction_evaluation_from_row(row) for row in rows)

    def upsert_prediction_outcome(self, record: PredictionOutcomeRecord) -> None:
        _validate_required(record.outcome_id, "outcome_id")
        _validate_required(record.candidate_id, "candidate_id")
        _validate_required(record.instrument_id, "instrument_id")
        _validate_required(record.symbol, "symbol")
        _validate_required(record.prediction_type, "prediction_type")
        _validate_required(record.horizon, "horizon")
        _validate_required(record.status, "status")
        if record.evaluation_window_end <= record.evaluation_window_start:
            raise ValueError("prediction outcome evaluation window end must be after start")
        now = _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            candidate = connection.execute(
                """
                SELECT instrument_id FROM prediction_candidates
                WHERE candidate_id = ?
                """,
                (record.candidate_id,),
            ).fetchone()
            if (
                candidate is not None
                and _row_text(candidate, "instrument_id") != record.instrument_id
            ):
                raise ValueError("prediction outcome candidate/instrument mismatch")
            connection.execute(
                """
                INSERT INTO prediction_outcomes (
                    outcome_id, candidate_id, instrument_id, symbol, prediction_type, horizon,
                    evaluation_window_start, evaluation_window_end, status, observed_result,
                    observed_at, result_summary, result_value, baseline_value, limitations_json,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(outcome_id) DO UPDATE SET
                    candidate_id = excluded.candidate_id,
                    instrument_id = excluded.instrument_id,
                    symbol = excluded.symbol,
                    prediction_type = excluded.prediction_type,
                    horizon = excluded.horizon,
                    evaluation_window_start = excluded.evaluation_window_start,
                    evaluation_window_end = excluded.evaluation_window_end,
                    status = excluded.status,
                    observed_result = excluded.observed_result,
                    observed_at = excluded.observed_at,
                    result_summary = excluded.result_summary,
                    result_value = excluded.result_value,
                    baseline_value = excluded.baseline_value,
                    limitations_json = excluded.limitations_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.outcome_id,
                    record.candidate_id,
                    record.instrument_id,
                    record.symbol,
                    record.prediction_type,
                    record.horizon,
                    _format_datetime(record.evaluation_window_start),
                    _format_datetime(record.evaluation_window_end),
                    record.status,
                    record.observed_result,
                    _format_optional_datetime(record.observed_at),
                    record.result_summary,
                    record.result_value,
                    record.baseline_value,
                    _dump_json_array(record.limitations),
                    _dump_json(record.metadata),
                    _format_datetime(now),
                    _format_datetime(now),
                ),
            )

    def get_prediction_outcome(self, outcome_id: str) -> PredictionOutcomeRecord | None:
        _validate_required(outcome_id, "outcome_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM prediction_outcomes WHERE outcome_id = ?",
                (outcome_id,),
            ).fetchone()
        if row is None:
            return None
        return _prediction_outcome_from_row(row)

    def list_prediction_outcomes_for_candidate(
        self, candidate_id: str
    ) -> tuple[PredictionOutcomeRecord, ...]:
        _validate_required(candidate_id, "candidate_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM prediction_outcomes
                WHERE candidate_id = ?
                ORDER BY evaluation_window_start, outcome_id
                """,
                (candidate_id,),
            ).fetchall()
        return tuple(_prediction_outcome_from_row(row) for row in rows)

    def link_outcome_evidence(self, record: OutcomeEvidenceLinkRecord) -> None:
        _validate_required(record.outcome_id, "outcome_id")
        _validate_required(record.evidence_id, "evidence_id")
        _validate_required(record.relationship, "relationship")
        created_at = record.created_at or _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO outcome_evidence_links (
                    outcome_id, evidence_id, relationship, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(outcome_id, evidence_id, relationship) DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (
                    record.outcome_id,
                    record.evidence_id,
                    record.relationship,
                    _dump_json(record.metadata),
                    _format_datetime(created_at),
                ),
            )

    def list_outcome_evidence_links(self, outcome_id: str) -> tuple[OutcomeEvidenceLinkRecord, ...]:
        _validate_required(outcome_id, "outcome_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM outcome_evidence_links
                WHERE outcome_id = ?
                ORDER BY relationship, evidence_id
                """,
                (outcome_id,),
            ).fetchall()
        return tuple(_outcome_evidence_link_from_row(row) for row in rows)

    def link_outcome_artifact(self, record: OutcomeArtifactLinkRecord) -> None:
        _validate_required(record.outcome_id, "outcome_id")
        _validate_required(record.artifact_id, "artifact_id")
        _validate_required(record.relationship, "relationship")
        created_at = record.created_at or _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO outcome_artifact_links (
                    outcome_id, artifact_id, relationship, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(outcome_id, artifact_id, relationship) DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (
                    record.outcome_id,
                    record.artifact_id,
                    record.relationship,
                    _dump_json(record.metadata),
                    _format_datetime(created_at),
                ),
            )

    def list_outcome_artifact_links(self, outcome_id: str) -> tuple[OutcomeArtifactLinkRecord, ...]:
        _validate_required(outcome_id, "outcome_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM outcome_artifact_links
                WHERE outcome_id = ?
                ORDER BY relationship, artifact_id
                """,
                (outcome_id,),
            ).fetchall()
        return tuple(_outcome_artifact_link_from_row(row) for row in rows)

    def upsert_prediction_outcome_evaluation(
        self, record: PredictionOutcomeEvaluationRecord
    ) -> None:
        _validate_required(record.outcome_evaluation_id, "outcome_evaluation_id")
        _validate_required(record.outcome_id, "outcome_id")
        _validate_required(record.candidate_id, "candidate_id")
        _validate_required(record.instrument_id, "instrument_id")
        _validate_required(record.symbol, "symbol")
        _validate_required(record.status, "status")
        _validate_confidence(record.quality_score)
        now = _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            outcome = connection.execute(
                """
                SELECT candidate_id, instrument_id, symbol FROM prediction_outcomes
                WHERE outcome_id = ?
                """,
                (record.outcome_id,),
            ).fetchone()
            if outcome is None:
                raise ValueError(f"prediction outcome does not exist: {record.outcome_id}")
            if _row_text(outcome, "candidate_id") != record.candidate_id:
                raise ValueError("outcome evaluation candidate_id must match outcome")
            if _row_text(outcome, "instrument_id") != record.instrument_id:
                raise ValueError("outcome evaluation instrument_id must match outcome")
            if _row_text(outcome, "symbol").upper() != record.symbol.upper():
                raise ValueError("outcome evaluation symbol must match outcome")
            candidate = connection.execute(
                """
                SELECT run_id, instrument_id FROM prediction_candidates
                WHERE candidate_id = ?
                """,
                (record.candidate_id,),
            ).fetchone()
            if candidate is not None:
                if _row_text(candidate, "instrument_id") != record.instrument_id:
                    raise ValueError("outcome evaluation candidate/instrument mismatch")
                candidate_run_id = candidate["run_id"]
                if (
                    record.run_id is not None
                    and candidate_run_id is not None
                    and str(candidate_run_id) != record.run_id
                ):
                    raise ValueError("outcome evaluation run_id must match candidate run_id")
            connection.execute(
                """
                INSERT INTO prediction_outcome_evaluations (
                    outcome_evaluation_id, run_id, outcome_id, candidate_id, instrument_id,
                    symbol, evaluated_at, status, quality_score, baseline_comparison_json,
                    artifact_id, limitations_json, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(outcome_evaluation_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    outcome_id = excluded.outcome_id,
                    candidate_id = excluded.candidate_id,
                    instrument_id = excluded.instrument_id,
                    symbol = excluded.symbol,
                    evaluated_at = excluded.evaluated_at,
                    status = excluded.status,
                    quality_score = excluded.quality_score,
                    baseline_comparison_json = excluded.baseline_comparison_json,
                    artifact_id = excluded.artifact_id,
                    limitations_json = excluded.limitations_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.outcome_evaluation_id,
                    record.run_id,
                    record.outcome_id,
                    record.candidate_id,
                    record.instrument_id,
                    record.symbol,
                    _format_datetime(record.evaluated_at),
                    record.status,
                    record.quality_score,
                    _dump_json(record.baseline_comparison),
                    record.artifact_id,
                    _dump_json_array(record.limitations),
                    _dump_json(record.metadata),
                    _format_datetime(now),
                    _format_datetime(now),
                ),
            )

    def get_prediction_outcome_evaluation(
        self, outcome_evaluation_id: str
    ) -> PredictionOutcomeEvaluationRecord | None:
        _validate_required(outcome_evaluation_id, "outcome_evaluation_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                """
                SELECT * FROM prediction_outcome_evaluations
                WHERE outcome_evaluation_id = ?
                """,
                (outcome_evaluation_id,),
            ).fetchone()
        if row is None:
            return None
        return _prediction_outcome_evaluation_from_row(row)

    def list_outcome_evaluations_for_run(
        self, run_id: str
    ) -> tuple[PredictionOutcomeEvaluationRecord, ...]:
        _validate_required(run_id, "run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM prediction_outcome_evaluations
                WHERE run_id = ?
                ORDER BY evaluated_at, outcome_evaluation_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(_prediction_outcome_evaluation_from_row(row) for row in rows)

    def list_outcome_evaluations_for_candidate(
        self, candidate_id: str
    ) -> tuple[PredictionOutcomeEvaluationRecord, ...]:
        _validate_required(candidate_id, "candidate_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM prediction_outcome_evaluations
                WHERE candidate_id = ?
                ORDER BY evaluated_at, outcome_evaluation_id
                """,
                (candidate_id,),
            ).fetchall()
        return tuple(_prediction_outcome_evaluation_from_row(row) for row in rows)

    def link_outcome_evaluation_evidence(self, record: OutcomeEvaluationEvidenceLinkRecord) -> None:
        _validate_required(record.outcome_evaluation_id, "outcome_evaluation_id")
        _validate_required(record.evidence_id, "evidence_id")
        _validate_required(record.relationship, "relationship")
        created_at = record.created_at or _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO outcome_evaluation_evidence_links (
                    outcome_evaluation_id, evidence_id, relationship,
                    metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(outcome_evaluation_id, evidence_id, relationship)
                DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (
                    record.outcome_evaluation_id,
                    record.evidence_id,
                    record.relationship,
                    _dump_json(record.metadata),
                    _format_datetime(created_at),
                ),
            )

    def list_outcome_evaluation_evidence_links(
        self, outcome_evaluation_id: str
    ) -> tuple[OutcomeEvaluationEvidenceLinkRecord, ...]:
        _validate_required(outcome_evaluation_id, "outcome_evaluation_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM outcome_evaluation_evidence_links
                WHERE outcome_evaluation_id = ?
                ORDER BY relationship, evidence_id
                """,
                (outcome_evaluation_id,),
            ).fetchall()
        return tuple(_outcome_evaluation_evidence_link_from_row(row) for row in rows)

    def link_outcome_evaluation_artifact(self, record: OutcomeEvaluationArtifactLinkRecord) -> None:
        _validate_required(record.outcome_evaluation_id, "outcome_evaluation_id")
        _validate_required(record.artifact_id, "artifact_id")
        _validate_required(record.relationship, "relationship")
        created_at = record.created_at or _utc_now()
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO outcome_evaluation_artifact_links (
                    outcome_evaluation_id, artifact_id, relationship,
                    metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(outcome_evaluation_id, artifact_id, relationship)
                DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (
                    record.outcome_evaluation_id,
                    record.artifact_id,
                    record.relationship,
                    _dump_json(record.metadata),
                    _format_datetime(created_at),
                ),
            )

    def list_outcome_evaluation_artifact_links(
        self, outcome_evaluation_id: str
    ) -> tuple[OutcomeEvaluationArtifactLinkRecord, ...]:
        _validate_required(outcome_evaluation_id, "outcome_evaluation_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM outcome_evaluation_artifact_links
                WHERE outcome_evaluation_id = ?
                ORDER BY relationship, artifact_id
                """,
                (outcome_evaluation_id,),
            ).fetchall()
        return tuple(_outcome_evaluation_artifact_link_from_row(row) for row in rows)

    def record_calibration_run(self, record: CalibrationRunRecord) -> None:
        _validate_required(record.calibration_id, "calibration_id")
        _validate_required(record.run_id, "run_id")
        _validate_required(record.method_version, "method_version")
        with self.connect() as connection:
            _ensure_initialized(connection)
            if record.source_outcome_evaluation_ids:
                placeholders = ",".join("?" for _ in record.source_outcome_evaluation_ids)
                rows = connection.execute(
                    f"""
                    SELECT outcome_evaluation_id, run_id
                    FROM prediction_outcome_evaluations
                    WHERE outcome_evaluation_id IN ({placeholders})
                    """,
                    record.source_outcome_evaluation_ids,
                ).fetchall()
                rows_by_id = {_row_text(row, "outcome_evaluation_id"): row for row in rows}
                missing = tuple(
                    outcome_evaluation_id
                    for outcome_evaluation_id in record.source_outcome_evaluation_ids
                    if outcome_evaluation_id not in rows_by_id
                )
                if missing:
                    raise ValueError(
                        "calibration source outcome evaluations do not exist: " + ", ".join(missing)
                    )
                mismatched = tuple(
                    outcome_evaluation_id
                    for outcome_evaluation_id, row in rows_by_id.items()
                    if row["run_id"] is not None and str(row["run_id"]) != record.run_id
                )
                if mismatched:
                    raise ValueError(
                        "calibration source outcome evaluations must match run_id: "
                        + ", ".join(sorted(mismatched))
                    )
            connection.execute(
                """
                INSERT INTO calibration_runs (
                    calibration_id, run_id, tool_run_id, method_version, created_at,
                    point_in_time_cutoff, cohort_query_json,
                    source_outcome_evaluation_ids_json, artifact_id, limitations_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(calibration_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    tool_run_id = excluded.tool_run_id,
                    method_version = excluded.method_version,
                    created_at = excluded.created_at,
                    point_in_time_cutoff = excluded.point_in_time_cutoff,
                    cohort_query_json = excluded.cohort_query_json,
                    source_outcome_evaluation_ids_json =
                        excluded.source_outcome_evaluation_ids_json,
                    artifact_id = excluded.artifact_id,
                    limitations_json = excluded.limitations_json,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.calibration_id,
                    record.run_id,
                    record.tool_run_id,
                    record.method_version,
                    _format_datetime(record.created_at),
                    _format_datetime(record.point_in_time_cutoff),
                    _dump_json(record.cohort_query),
                    _dump_json_array(record.source_outcome_evaluation_ids),
                    record.artifact_id,
                    _dump_json_array(record.limitations),
                    _dump_json(record.metadata),
                ),
            )

    def delete_calibration_slices(self, calibration_id: str) -> None:
        _validate_required(calibration_id, "calibration_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                "DELETE FROM calibration_slices WHERE calibration_id = ?",
                (calibration_id,),
            )

    def get_calibration_run(self, calibration_id: str) -> CalibrationRunRecord | None:
        _validate_required(calibration_id, "calibration_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            row = connection.execute(
                "SELECT * FROM calibration_runs WHERE calibration_id = ?",
                (calibration_id,),
            ).fetchone()
        if row is None:
            return None
        return _calibration_run_from_row(row)

    def list_calibration_runs_for_run(self, run_id: str) -> tuple[CalibrationRunRecord, ...]:
        _validate_required(run_id, "run_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM calibration_runs
                WHERE run_id = ?
                ORDER BY created_at, calibration_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(_calibration_run_from_row(row) for row in rows)

    def record_calibration_slice(self, record: CalibrationSliceRecord) -> None:
        _validate_required(record.slice_id, "slice_id")
        _validate_required(record.calibration_id, "calibration_id")
        _validate_required(record.cohort_label, "cohort_label")
        _validate_non_negative_count(record.sample_count, "sample_count")
        _validate_non_negative_count(record.resolved_count, "resolved_count")
        _validate_non_negative_count(record.pending_count, "pending_count")
        _validate_non_negative_count(record.stale_count, "stale_count")
        _validate_non_negative_count(record.unavailable_count, "unavailable_count")
        _validate_non_negative_count(record.not_evaluable_count, "not_evaluable_count")
        status_total = (
            record.resolved_count
            + record.pending_count
            + record.stale_count
            + record.unavailable_count
            + record.not_evaluable_count
        )
        if status_total != record.sample_count:
            raise ValueError("calibration slice sample_count must equal status counts")
        with self.connect() as connection:
            _ensure_initialized(connection)
            connection.execute(
                """
                INSERT INTO calibration_slices (
                    slice_id, calibration_id, signal_family, prediction_type, horizon,
                    cohort_label, sample_count, resolved_count, pending_count, stale_count,
                    unavailable_count, not_evaluable_count, metrics_json,
                    baseline_comparison_json, provenance_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slice_id) DO UPDATE SET
                    calibration_id = excluded.calibration_id,
                    signal_family = excluded.signal_family,
                    prediction_type = excluded.prediction_type,
                    horizon = excluded.horizon,
                    cohort_label = excluded.cohort_label,
                    sample_count = excluded.sample_count,
                    resolved_count = excluded.resolved_count,
                    pending_count = excluded.pending_count,
                    stale_count = excluded.stale_count,
                    unavailable_count = excluded.unavailable_count,
                    not_evaluable_count = excluded.not_evaluable_count,
                    metrics_json = excluded.metrics_json,
                    baseline_comparison_json = excluded.baseline_comparison_json,
                    provenance_json = excluded.provenance_json,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.slice_id,
                    record.calibration_id,
                    record.signal_family,
                    record.prediction_type,
                    record.horizon,
                    record.cohort_label,
                    record.sample_count,
                    record.resolved_count,
                    record.pending_count,
                    record.stale_count,
                    record.unavailable_count,
                    record.not_evaluable_count,
                    _dump_json(record.metrics),
                    _dump_json(record.baseline_comparison),
                    _dump_json(record.provenance),
                    _dump_json(record.metadata),
                ),
            )

    def list_calibration_slices(self, calibration_id: str) -> tuple[CalibrationSliceRecord, ...]:
        _validate_required(calibration_id, "calibration_id")
        with self.connect() as connection:
            _ensure_initialized(connection)
            rows = connection.execute(
                """
                SELECT * FROM calibration_slices
                WHERE calibration_id = ?
                ORDER BY cohort_label, signal_family, slice_id
                """,
                (calibration_id,),
            ).fetchall()
        return tuple(_calibration_slice_from_row(row) for row in rows)

    def _sync_candidate_links(
        self,
        connection: sqlite3.Connection,
        record: PredictionCandidateRecord,
    ) -> None:
        connection.execute(
            """
            DELETE FROM candidate_evidence_links
            WHERE candidate_id = ? AND relationship IN ('supports', 'contradicts')
            """,
            (record.candidate_id,),
        )
        connection.execute(
            """
            DELETE FROM candidate_artifact_links
            WHERE candidate_id = ? AND relationship = 'signal'
            """,
            (record.candidate_id,),
        )
        created_at = _format_datetime(_utc_now())
        for relationship, evidence_ids in (
            ("supports", record.evidence_for),
            ("contradicts", record.evidence_against),
        ):
            for evidence_id in dict.fromkeys(evidence_ids):
                connection.execute(
                    """
                    INSERT INTO candidate_evidence_links (
                        candidate_id, evidence_id, relationship, metadata_json, created_at
                    )
                    SELECT ?, ?, ?, ?, ?
                    WHERE EXISTS (
                        SELECT 1 FROM evidence_items WHERE evidence_id = ?
                    )
                    ON CONFLICT(candidate_id, evidence_id, relationship) DO UPDATE SET
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        record.candidate_id,
                        evidence_id,
                        relationship,
                        _dump_json({"source": "prediction_candidate_record"}),
                        created_at,
                        evidence_id,
                    ),
                )
        for artifact_id in dict.fromkeys(record.signal_artifacts):
            connection.execute(
                """
                INSERT INTO candidate_artifact_links (
                    candidate_id, artifact_id, relationship, metadata_json, created_at
                )
                SELECT ?, ?, ?, ?, ?
                WHERE EXISTS (
                    SELECT 1 FROM artifacts WHERE artifact_id = ?
                )
                ON CONFLICT(candidate_id, artifact_id, relationship) DO UPDATE SET
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.candidate_id,
                    artifact_id,
                    "signal",
                    _dump_json({"source": "prediction_candidate_record"}),
                    created_at,
                    artifact_id,
                ),
            )


ResearchSQLiteStore = SQLiteStore


class PlanningSQLiteStore:
    """Repository wrapper around the tracked planning SQLite database."""

    def __init__(self, path: Path = DEFAULT_PLANNING_DATABASE_PATH) -> None:
        self.path = path

    @contextmanager
    def connect(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        connection = _open_sqlite_connection(self.path, create=create)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect(create=True) as connection:
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
    _apply_research_migrations(connection)


@dataclass(frozen=True)
class _SchemaMigrationStep:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _apply_research_migrations(connection: sqlite3.Connection) -> None:
    applied_versions = {
        int(row["version"]) for row in connection.execute("SELECT version FROM schema_migrations")
    }
    for step in _RESEARCH_MIGRATION_STEPS:
        if step.version in applied_versions:
            continue
        step.apply(connection)
        connection.execute(
            """
            INSERT INTO schema_migrations(version, name, applied_at)
            VALUES (?, ?, ?)
            ON CONFLICT(version) DO UPDATE SET
                name = excluded.name,
                applied_at = excluded.applied_at
            """,
            (step.version, step.name, _format_datetime(_utc_now())),
        )


def _noop_migration(_connection: sqlite3.Connection) -> None:
    return None


def _migrate_research_schema_v4(connection: sqlite3.Connection) -> None:
    columns = _table_columns(connection, "instruments")
    migrations = {
        "provider_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "related_instruments_json": "TEXT NOT NULL DEFAULT '[]'",
        "tradability_evidence_json": "TEXT NOT NULL DEFAULT '[]'",
        "data_availability_json": "TEXT NOT NULL DEFAULT '[]'",
    }
    for column, definition in migrations.items():
        if column not in columns:
            connection.execute(f"ALTER TABLE instruments ADD COLUMN {column} {definition}")
    candidate_columns = _table_columns(connection, "prediction_candidates")
    if "run_id" not in candidate_columns:
        connection.execute(
            """
            ALTER TABLE prediction_candidates
            ADD COLUMN run_id TEXT REFERENCES research_runs(run_id) ON DELETE SET NULL
            """
        )
    evidence_columns = _table_columns(connection, "evidence_items")
    evidence_migrations = {
        "tool_run_id": "TEXT REFERENCES tool_runs(tool_run_id) ON DELETE SET NULL",
        "source_query_id": ("TEXT REFERENCES source_queries(source_query_id) ON DELETE SET NULL"),
        "provenance_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    for column, definition in evidence_migrations.items():
        if column not in evidence_columns:
            connection.execute(f"ALTER TABLE evidence_items ADD COLUMN {column} {definition}")
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_prediction_candidates_run_id
        ON prediction_candidates(run_id);
        CREATE INDEX IF NOT EXISTS idx_evidence_tool_run_id
        ON evidence_items(tool_run_id);
        CREATE INDEX IF NOT EXISTS idx_evidence_source_query_id
        ON evidence_items(source_query_id);
        """
    )
    connection.executescript(_RESEARCH_REGISTRY_SCHEMA_SQL)


def _migrate_research_schema_v5(connection: sqlite3.Connection) -> None:
    artifact_columns = _table_columns(connection, "artifacts")
    artifact_migrations = {
        "produced_by": "TEXT",
        "record_count": "INTEGER CHECK(record_count IS NULL OR record_count >= 0)",
    }
    for column, definition in artifact_migrations.items():
        if column not in artifact_columns:
            connection.execute(f"ALTER TABLE artifacts ADD COLUMN {column} {definition}")


def _migrate_research_schema_v6(connection: sqlite3.Connection) -> None:
    connection.executescript(_RESEARCH_EVALUATION_SCHEMA_SQL)


_RESEARCH_MIGRATION_STEPS = (
    _SchemaMigrationStep(1, "initial_research_schema", _noop_migration),
    _SchemaMigrationStep(2, "phase2_research_graph_schema", _noop_migration),
    _SchemaMigrationStep(3, "phase2_evidence_provenance_schema", _noop_migration),
    _SchemaMigrationStep(4, "registry_grade_research_schema_v4", _migrate_research_schema_v4),
    _SchemaMigrationStep(5, "artifact_audit_provenance_schema_v5", _migrate_research_schema_v5),
    _SchemaMigrationStep(6, "phase6_evaluation_calibration_schema", _migrate_research_schema_v6),
)


def _replace_instrument_children(
    connection: sqlite3.Connection, record: InstrumentRecord, now: datetime
) -> None:
    for table_name in (
        "instrument_aliases",
        "instrument_provider_ids",
        "instrument_related_instruments",
        "instrument_data_availability",
    ):
        _require_safe_table_name(table_name)
        connection.execute(
            f"DELETE FROM {table_name} WHERE instrument_id = ?", (record.instrument_id,)
        )

    for alias in record.aliases:
        _validate_required(alias, "alias")
        connection.execute(
            """
            INSERT INTO instrument_aliases (
                instrument_id, alias, alias_lower, created_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(instrument_id, alias_lower) DO UPDATE SET
                alias = excluded.alias
            """,
            (record.instrument_id, alias, _normalize_lookup(alias), _format_datetime(now)),
        )

    seen_provider_ids: set[tuple[str, str, str]] = set()
    for provider_id in record.provider_ids:
        provider = _json_required_text(provider_id, "provider")
        namespace = _json_optional_text(provider_id, "namespace") or "default"
        identifier = _json_required_text(provider_id, "identifier")
        provider_lower = _normalize_lookup(provider)
        namespace_lower = _normalize_lookup(namespace)
        identifier_lower = _normalize_lookup(identifier)
        provider_key = (provider_lower, namespace_lower, identifier_lower)
        if provider_key in seen_provider_ids:
            raise ValueError("instrument provider ids must be unique case-insensitively")
        seen_provider_ids.add(provider_key)
        existing_provider_row = connection.execute(
            """
            SELECT instrument_id FROM instrument_provider_ids
            WHERE provider_lower = ?
              AND namespace_lower = ?
              AND lower(identifier) = ?
            """,
            (provider_lower, namespace_lower, identifier_lower),
        ).fetchone()
        if (
            existing_provider_row is not None
            and existing_provider_row["instrument_id"] != record.instrument_id
        ):
            raise ValueError(
                "provider identifier is already assigned to "
                f"{existing_provider_row['instrument_id']}"
            )
        connection.execute(
            """
            INSERT INTO instrument_provider_ids (
                instrument_id, provider, provider_lower, namespace, namespace_lower,
                identifier, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider_lower, namespace_lower, identifier) DO UPDATE SET
                provider = excluded.provider,
                namespace = excluded.namespace,
                metadata_json = excluded.metadata_json
            """,
            (
                record.instrument_id,
                provider,
                provider_lower,
                namespace,
                namespace_lower,
                identifier,
                _dump_json(
                    _json_metadata_without(provider_id, {"provider", "namespace", "identifier"})
                ),
                _format_datetime(now),
            ),
        )

    for related in record.related_instruments:
        related_id = _json_required_text(related, "instrument_id")
        relationship = _json_required_text(related, "relationship")
        connection.execute(
            """
            INSERT INTO instrument_related_instruments (
                instrument_id, related_instrument_id, relationship, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(instrument_id, related_instrument_id, relationship) DO UPDATE SET
                metadata_json = excluded.metadata_json
            """,
            (
                record.instrument_id,
                related_id,
                relationship,
                _dump_json(_json_metadata_without(related, {"instrument_id", "relationship"})),
                _format_datetime(now),
            ),
        )

    for evidence in record.tradability_evidence:
        url = (
            _json_optional_text(evidence, "url")
            or _json_optional_text(evidence, "source_url")
            or _json_optional_text(evidence, "permalink")
        )
        _insert_tradability_evidence_once(
            connection,
            InstrumentTradabilityEvidenceRecord(
                instrument_id=record.instrument_id,
                provider=_json_required_text(evidence, "provider"),
                status=_json_required_text(evidence, "status"),
                retrieved_at=_json_optional_datetime(evidence, "retrieved_at") or now,
                source_query_id=_json_optional_text(evidence, "source_query_id"),
                url=url,
                raw_identifier=_json_optional_text(evidence, "raw_identifier"),
                extraction_confidence=_json_optional_float(evidence, "extraction_confidence"),
                metadata=_json_metadata_without(
                    evidence,
                    {
                        "provider",
                        "status",
                        "retrieved_at",
                        "source_query_id",
                        "url",
                        "source_url",
                        "permalink",
                        "raw_identifier",
                        "extraction_confidence",
                    },
                ),
            ),
        )

    for availability in record.data_availability:
        provider = _json_required_text(availability, "provider")
        data_type = _json_required_text(availability, "data_type")
        status = _json_required_text(availability, "status")
        as_of = (
            _json_optional_datetime(availability, "checked_at")
            or _json_optional_datetime(availability, "as_of")
            or now
        )
        connection.execute(
            """
            INSERT INTO instrument_data_availability (
                instrument_id, provider, provider_lower, data_type, status,
                as_of, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(instrument_id, provider_lower, data_type) DO UPDATE SET
                provider = excluded.provider,
                status = excluded.status,
                as_of = excluded.as_of,
                metadata_json = excluded.metadata_json
            """,
            (
                record.instrument_id,
                provider,
                _normalize_lookup(provider),
                data_type,
                status,
                _format_datetime(as_of),
                _dump_json(
                    _json_metadata_without(
                        availability,
                        {"provider", "data_type", "status", "checked_at", "as_of"},
                    )
                ),
            ),
        )


def _insert_tradability_evidence(
    connection: sqlite3.Connection, record: InstrumentTradabilityEvidenceRecord
) -> None:
    _validate_confidence(record.extraction_confidence)
    connection.execute(
        """
        INSERT INTO instrument_tradability_evidence (
            instrument_id, provider, provider_lower, status, retrieved_at,
            source_query_id, url, raw_identifier, extraction_confidence, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.instrument_id,
            record.provider,
            _normalize_lookup(record.provider),
            record.status,
            _format_datetime(record.retrieved_at),
            record.source_query_id,
            record.url,
            record.raw_identifier,
            record.extraction_confidence,
            _dump_json(record.metadata),
        ),
    )


def _insert_tradability_evidence_once(
    connection: sqlite3.Connection, record: InstrumentTradabilityEvidenceRecord
) -> None:
    metadata_json = _dump_json(record.metadata)
    existing = connection.execute(
        """
        SELECT evidence_rowid FROM instrument_tradability_evidence
        WHERE instrument_id = ?
          AND provider_lower = ?
          AND status = ?
          AND retrieved_at = ?
          AND COALESCE(source_query_id, '') = COALESCE(?, '')
          AND COALESCE(url, '') = COALESCE(?, '')
          AND COALESCE(raw_identifier, '') = COALESCE(?, '')
          AND COALESCE(extraction_confidence, -1.0) = COALESCE(?, -1.0)
          AND metadata_json = ?
        LIMIT 1
        """,
        (
            record.instrument_id,
            _normalize_lookup(record.provider),
            record.status,
            _format_datetime(record.retrieved_at),
            record.source_query_id,
            record.url,
            record.raw_identifier,
            record.extraction_confidence,
            metadata_json,
        ),
    ).fetchone()
    if existing is None:
        _insert_tradability_evidence(connection, record)


def _normalize_lookup(value: str) -> str:
    return value.strip().casefold()


_SAFE_SQL_TABLE_NAMES = frozenset(
    {
        "artifacts",
        "candidate_artifact_links",
        "candidate_evidence_links",
        "evidence_items",
        "instrument_aliases",
        "instrument_data_availability",
        "instrument_provider_ids",
        "instrument_related_instruments",
        "instrument_tradability_evidence",
        "instruments",
        "plan_artifact_links",
        "plan_commit_links",
        "planning_decisions",
        "planning_done_criteria",
        "planning_milestones",
        "planning_progress",
        "plans",
        "prediction_candidates",
        "research_runs",
        "schema_migrations",
        "source_queries",
        "tool_runs",
        "watchlist_items",
        "watchlists",
    }
)


def _require_safe_table_name(table_name: str) -> None:
    if table_name not in _SAFE_SQL_TABLE_NAMES:
        raise ValueError(f"unsupported SQLite table identifier: {table_name}")


def _open_sqlite_connection(path: Path, *, create: bool) -> sqlite3.Connection:
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path))
    else:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=rw", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    _require_safe_table_name(table_name)
    return {
        row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


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
    found_version = None if row is None else row["version"]
    if found_version != CURRENT_RESEARCH_SCHEMA_VERSION:
        raise RuntimeError(
            "SQLite prediction research database schema is not current "
            f"(found={found_version}, expected={CURRENT_RESEARCH_SCHEMA_VERSION})"
        )


def _ensure_planning_initialized(connection: sqlite3.Connection) -> None:
    try:
        row = connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
    except sqlite3.OperationalError as exc:
        raise RuntimeError("SQLite prediction planning database is not initialized") from exc
    found_version = None if row is None else row["version"]
    if found_version != CURRENT_PLANNING_SCHEMA_VERSION:
        raise RuntimeError(
            "SQLite prediction planning database schema is not current "
            f"(found={found_version}, expected={CURRENT_PLANNING_SCHEMA_VERSION})"
        )


def _validate_required(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _validate_relative_artifact_path(path: Path) -> None:
    if path.is_absolute():
        raise ValueError("artifact path must be relative")
    if any(part == ".." for part in path.parts):
        raise ValueError("artifact path must not contain parent traversal")


def _validate_confidence(value: float | None) -> None:
    if value is None:
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence values must be between 0.0 and 1.0")


def _validate_non_negative_count(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


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
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("datetimes stored in SQLite must be timezone-aware")
    return parsed.astimezone(UTC)


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


def _json_required_text(value: JsonObject, key: str) -> str:
    raw_value: object = value.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw_value


def _json_optional_text(value: JsonObject, key: str) -> str | None:
    raw_value: object = value.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError(f"{key} must be a string when provided")
    return raw_value


def _json_optional_float(value: JsonObject, key: str) -> float | None:
    raw_value: object = value.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"{key} must be numeric when provided")
    if not isinstance(raw_value, int | float):
        raise ValueError(f"{key} must be numeric when provided")
    return float(raw_value)


def _json_optional_datetime(value: JsonObject, key: str) -> datetime | None:
    raw_value: object = value.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, datetime):
        return raw_value
    if not isinstance(raw_value, str):
        raise ValueError(f"{key} must be an ISO timestamp when provided")
    return _parse_datetime(raw_value)


def _json_metadata_without(value: JsonObject, excluded_keys: set[str]) -> JsonObject:
    return {key: item for key, item in value.items() if key not in excluded_keys}


def _row_text(row: sqlite3.Row, column: str) -> str:
    return cast(str, row[column])


def _row_optional_text(row: sqlite3.Row, column: str) -> str | None:
    return cast(str | None, row[column])


def _row_optional_float(row: sqlite3.Row, column: str) -> float | None:
    return cast(float | None, row[column])


def _row_optional_int(row: sqlite3.Row, column: str) -> int | None:
    return cast(int | None, row[column])


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


def _tradability_evidence_from_row(row: sqlite3.Row) -> InstrumentTradabilityEvidenceRecord:
    return InstrumentTradabilityEvidenceRecord(
        instrument_id=_row_text(row, "instrument_id"),
        provider=_row_text(row, "provider"),
        status=_row_text(row, "status"),
        retrieved_at=_parse_datetime(_row_text(row, "retrieved_at")),
        source_query_id=_row_optional_text(row, "source_query_id"),
        url=_row_optional_text(row, "url"),
        raw_identifier=_row_optional_text(row, "raw_identifier"),
        extraction_confidence=_row_optional_float(row, "extraction_confidence"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _watchlist_from_row(row: sqlite3.Row) -> WatchlistRecord:
    return WatchlistRecord(
        watchlist_id=_row_text(row, "watchlist_id"),
        name=_row_text(row, "name"),
        description=_row_optional_text(row, "description"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _watchlist_item_from_row(row: sqlite3.Row) -> WatchlistItemRecord:
    return WatchlistItemRecord(
        watchlist_id=_row_text(row, "watchlist_id"),
        instrument_id=_row_text(row, "instrument_id"),
        sort_order=int(row["sort_order"]),
        notes=_row_optional_text(row, "notes"),
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
        produced_by=_row_optional_text(row, "produced_by"),
        record_count=_row_optional_int(row, "record_count"),
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
        tool_run_id=_row_optional_text(row, "tool_run_id"),
        source_query_id=_row_optional_text(row, "source_query_id"),
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
        provenance_json=_load_json_object(_row_text(row, "provenance_json")),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _prediction_candidate_from_row(row: sqlite3.Row) -> PredictionCandidateRecord:
    return PredictionCandidateRecord(
        candidate_id=_row_text(row, "candidate_id"),
        run_id=_row_optional_text(row, "run_id"),
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


def _candidate_evidence_link_from_row(row: sqlite3.Row) -> CandidateEvidenceLinkRecord:
    return CandidateEvidenceLinkRecord(
        candidate_id=_row_text(row, "candidate_id"),
        evidence_id=_row_text(row, "evidence_id"),
        relationship=_row_text(row, "relationship"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
        created_at=_parse_datetime(_row_text(row, "created_at")),
    )


def _candidate_artifact_link_from_row(row: sqlite3.Row) -> CandidateArtifactLinkRecord:
    return CandidateArtifactLinkRecord(
        candidate_id=_row_text(row, "candidate_id"),
        artifact_id=_row_text(row, "artifact_id"),
        relationship=_row_text(row, "relationship"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
        created_at=_parse_datetime(_row_text(row, "created_at")),
    )


def _prediction_evaluation_from_row(row: sqlite3.Row) -> PredictionEvaluationRecord:
    return PredictionEvaluationRecord(
        evaluation_id=_row_text(row, "evaluation_id"),
        run_id=_row_optional_text(row, "run_id"),
        candidate_id=_row_text(row, "candidate_id"),
        instrument_id=_row_text(row, "instrument_id"),
        symbol=_row_text(row, "symbol"),
        created_at=_parse_datetime(_row_text(row, "created_at")),
        prediction_type=_row_text(row, "prediction_type"),
        horizon=_row_text(row, "horizon"),
        direction=_row_optional_text(row, "direction"),
        status=_row_text(row, "status"),
        score=float(row["score"]),
        baseline_comparison=_load_json_object(_row_text(row, "baseline_comparison_json")),
        evidence_counts=_load_json_object(_row_text(row, "evidence_counts_json")),
        signal_counts=_load_json_object(_row_text(row, "signal_counts_json")),
        artifact_id=_row_optional_text(row, "artifact_id"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _prediction_outcome_from_row(row: sqlite3.Row) -> PredictionOutcomeRecord:
    return PredictionOutcomeRecord(
        outcome_id=_row_text(row, "outcome_id"),
        candidate_id=_row_text(row, "candidate_id"),
        instrument_id=_row_text(row, "instrument_id"),
        symbol=_row_text(row, "symbol"),
        prediction_type=_row_text(row, "prediction_type"),
        horizon=_row_text(row, "horizon"),
        evaluation_window_start=_parse_datetime(_row_text(row, "evaluation_window_start")),
        evaluation_window_end=_parse_datetime(_row_text(row, "evaluation_window_end")),
        status=_row_text(row, "status"),
        observed_result=_row_optional_text(row, "observed_result"),
        observed_at=_parse_optional_datetime(_row_optional_text(row, "observed_at")),
        result_summary=_row_optional_text(row, "result_summary"),
        result_value=_row_optional_float(row, "result_value"),
        baseline_value=_row_optional_float(row, "baseline_value"),
        limitations=_load_string_tuple(_row_text(row, "limitations_json")),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _prediction_outcome_evaluation_from_row(
    row: sqlite3.Row,
) -> PredictionOutcomeEvaluationRecord:
    return PredictionOutcomeEvaluationRecord(
        outcome_evaluation_id=_row_text(row, "outcome_evaluation_id"),
        run_id=_row_optional_text(row, "run_id"),
        outcome_id=_row_text(row, "outcome_id"),
        candidate_id=_row_text(row, "candidate_id"),
        instrument_id=_row_text(row, "instrument_id"),
        symbol=_row_text(row, "symbol"),
        evaluated_at=_parse_datetime(_row_text(row, "evaluated_at")),
        status=_row_text(row, "status"),
        quality_score=_row_optional_float(row, "quality_score"),
        baseline_comparison=_load_json_object(_row_text(row, "baseline_comparison_json")),
        artifact_id=_row_optional_text(row, "artifact_id"),
        limitations=_load_string_tuple(_row_text(row, "limitations_json")),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _outcome_evidence_link_from_row(row: sqlite3.Row) -> OutcomeEvidenceLinkRecord:
    return OutcomeEvidenceLinkRecord(
        outcome_id=_row_text(row, "outcome_id"),
        evidence_id=_row_text(row, "evidence_id"),
        relationship=_row_text(row, "relationship"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
        created_at=_parse_datetime(_row_text(row, "created_at")),
    )


def _outcome_artifact_link_from_row(row: sqlite3.Row) -> OutcomeArtifactLinkRecord:
    return OutcomeArtifactLinkRecord(
        outcome_id=_row_text(row, "outcome_id"),
        artifact_id=_row_text(row, "artifact_id"),
        relationship=_row_text(row, "relationship"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
        created_at=_parse_datetime(_row_text(row, "created_at")),
    )


def _outcome_evaluation_evidence_link_from_row(
    row: sqlite3.Row,
) -> OutcomeEvaluationEvidenceLinkRecord:
    return OutcomeEvaluationEvidenceLinkRecord(
        outcome_evaluation_id=_row_text(row, "outcome_evaluation_id"),
        evidence_id=_row_text(row, "evidence_id"),
        relationship=_row_text(row, "relationship"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
        created_at=_parse_datetime(_row_text(row, "created_at")),
    )


def _outcome_evaluation_artifact_link_from_row(
    row: sqlite3.Row,
) -> OutcomeEvaluationArtifactLinkRecord:
    return OutcomeEvaluationArtifactLinkRecord(
        outcome_evaluation_id=_row_text(row, "outcome_evaluation_id"),
        artifact_id=_row_text(row, "artifact_id"),
        relationship=_row_text(row, "relationship"),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
        created_at=_parse_datetime(_row_text(row, "created_at")),
    )


def _calibration_run_from_row(row: sqlite3.Row) -> CalibrationRunRecord:
    return CalibrationRunRecord(
        calibration_id=_row_text(row, "calibration_id"),
        run_id=_row_text(row, "run_id"),
        tool_run_id=_row_optional_text(row, "tool_run_id"),
        method_version=_row_text(row, "method_version"),
        created_at=_parse_datetime(_row_text(row, "created_at")),
        point_in_time_cutoff=_parse_datetime(_row_text(row, "point_in_time_cutoff")),
        cohort_query=_load_json_object(_row_text(row, "cohort_query_json")),
        source_outcome_evaluation_ids=_load_string_tuple(
            _row_text(row, "source_outcome_evaluation_ids_json")
        ),
        artifact_id=_row_optional_text(row, "artifact_id"),
        limitations=_load_string_tuple(_row_text(row, "limitations_json")),
        metadata=_load_json_object(_row_text(row, "metadata_json")),
    )


def _calibration_slice_from_row(row: sqlite3.Row) -> CalibrationSliceRecord:
    return CalibrationSliceRecord(
        slice_id=_row_text(row, "slice_id"),
        calibration_id=_row_text(row, "calibration_id"),
        signal_family=_row_optional_text(row, "signal_family"),
        prediction_type=_row_optional_text(row, "prediction_type"),
        horizon=_row_optional_text(row, "horizon"),
        cohort_label=_row_text(row, "cohort_label"),
        sample_count=int(row["sample_count"]),
        resolved_count=int(row["resolved_count"]),
        pending_count=int(row["pending_count"]),
        stale_count=int(row["stale_count"]),
        unavailable_count=int(row["unavailable_count"]),
        not_evaluable_count=int(row["not_evaluable_count"]),
        metrics=_load_json_object(_row_text(row, "metrics_json")),
        baseline_comparison=_load_json_object(_row_text(row, "baseline_comparison_json")),
        provenance=_load_json_object(_row_text(row, "provenance_json")),
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
    produced_by TEXT,
    record_count INTEGER CHECK(record_count IS NULL OR record_count >= 0),
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
CREATE INDEX IF NOT EXISTS idx_source_queries_tool_run_id ON source_queries(tool_run_id);
CREATE INDEX IF NOT EXISTS idx_source_queries_retrieved_at ON source_queries(retrieved_at);

CREATE TABLE IF NOT EXISTS evidence_items (
    evidence_id TEXT PRIMARY KEY CHECK(length(evidence_id) > 0),
    tool_run_id TEXT REFERENCES tool_runs(tool_run_id) ON DELETE SET NULL,
    source_query_id TEXT REFERENCES source_queries(source_query_id) ON DELETE SET NULL,
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
    provenance_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_evidence_provider ON evidence_items(provider);
CREATE INDEX IF NOT EXISTS idx_evidence_source_type ON evidence_items(source_type);
CREATE INDEX IF NOT EXISTS idx_evidence_retrieved_at ON evidence_items(retrieved_at);
CREATE INDEX IF NOT EXISTS idx_evidence_freshness ON evidence_items(freshness_status);

CREATE TABLE IF NOT EXISTS prediction_candidates (
    candidate_id TEXT PRIMARY KEY CHECK(length(candidate_id) > 0),
    run_id TEXT REFERENCES research_runs(run_id) ON DELETE SET NULL,
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

CREATE TABLE IF NOT EXISTS candidate_evidence_links (
    candidate_id TEXT NOT NULL REFERENCES prediction_candidates(candidate_id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL REFERENCES evidence_items(evidence_id) ON DELETE CASCADE,
    relationship TEXT NOT NULL CHECK(length(relationship) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY(candidate_id, evidence_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_candidate_evidence_links_evidence_id
ON candidate_evidence_links(evidence_id);

CREATE TABLE IF NOT EXISTS candidate_artifact_links (
    candidate_id TEXT NOT NULL REFERENCES prediction_candidates(candidate_id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    relationship TEXT NOT NULL CHECK(length(relationship) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY(candidate_id, artifact_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_candidate_artifact_links_artifact_id
ON candidate_artifact_links(artifact_id);
"""

_RESEARCH_REGISTRY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS instrument_aliases (
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id) ON DELETE CASCADE,
    alias TEXT NOT NULL CHECK(length(alias) > 0),
    alias_lower TEXT NOT NULL CHECK(length(alias_lower) > 0),
    created_at TEXT NOT NULL,
    PRIMARY KEY(instrument_id, alias_lower)
);

CREATE INDEX IF NOT EXISTS idx_instrument_aliases_alias_lower
ON instrument_aliases(alias_lower);

CREATE TABLE IF NOT EXISTS instrument_provider_ids (
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id) ON DELETE CASCADE,
    provider TEXT NOT NULL CHECK(length(provider) > 0),
    provider_lower TEXT NOT NULL CHECK(length(provider_lower) > 0),
    namespace TEXT NOT NULL CHECK(length(namespace) > 0),
    namespace_lower TEXT NOT NULL CHECK(length(namespace_lower) > 0),
    identifier TEXT NOT NULL CHECK(length(identifier) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY(provider_lower, namespace_lower, identifier)
);

CREATE INDEX IF NOT EXISTS idx_instrument_provider_ids_instrument
ON instrument_provider_ids(instrument_id);

CREATE TABLE IF NOT EXISTS instrument_related_instruments (
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id) ON DELETE CASCADE,
    related_instrument_id TEXT NOT NULL CHECK(length(related_instrument_id) > 0),
    relationship TEXT NOT NULL CHECK(length(relationship) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY(instrument_id, related_instrument_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_instrument_related_related_id
ON instrument_related_instruments(related_instrument_id);

CREATE TABLE IF NOT EXISTS instrument_tradability_evidence (
    evidence_rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id) ON DELETE CASCADE,
    provider TEXT NOT NULL CHECK(length(provider) > 0),
    provider_lower TEXT NOT NULL CHECK(length(provider_lower) > 0),
    status TEXT NOT NULL CHECK(length(status) > 0),
    retrieved_at TEXT NOT NULL,
    source_query_id TEXT REFERENCES source_queries(source_query_id) ON DELETE SET NULL,
    url TEXT,
    raw_identifier TEXT,
    extraction_confidence REAL CHECK(
        extraction_confidence IS NULL
        OR (extraction_confidence >= 0.0 AND extraction_confidence <= 1.0)
    ),
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_instrument_tradability_latest
ON instrument_tradability_evidence(instrument_id, provider_lower, retrieved_at);

CREATE TABLE IF NOT EXISTS instrument_data_availability (
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id) ON DELETE CASCADE,
    provider TEXT NOT NULL CHECK(length(provider) > 0),
    provider_lower TEXT NOT NULL CHECK(length(provider_lower) > 0),
    data_type TEXT NOT NULL CHECK(length(data_type) > 0),
    status TEXT NOT NULL CHECK(length(status) > 0),
    as_of TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(instrument_id, provider_lower, data_type)
);

CREATE INDEX IF NOT EXISTS idx_instrument_data_availability_provider
ON instrument_data_availability(provider_lower, data_type);

CREATE TABLE IF NOT EXISTS watchlists (
    watchlist_id TEXT PRIMARY KEY CHECK(length(watchlist_id) > 0),
    name TEXT NOT NULL CHECK(length(name) > 0),
    description TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_watchlists_name ON watchlists(name);

CREATE TABLE IF NOT EXISTS watchlist_items (
    watchlist_id TEXT NOT NULL REFERENCES watchlists(watchlist_id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id) ON DELETE CASCADE,
    sort_order INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY(watchlist_id, instrument_id)
);

CREATE INDEX IF NOT EXISTS idx_watchlist_items_instrument
ON watchlist_items(instrument_id);
"""

_RESEARCH_EVALUATION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prediction_evaluations (
    evaluation_id TEXT PRIMARY KEY CHECK(length(evaluation_id) > 0),
    run_id TEXT REFERENCES research_runs(run_id) ON DELETE SET NULL,
    candidate_id TEXT NOT NULL REFERENCES prediction_candidates(candidate_id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id) ON DELETE RESTRICT,
    symbol TEXT NOT NULL CHECK(length(symbol) > 0),
    created_at TEXT NOT NULL,
    prediction_type TEXT NOT NULL CHECK(length(prediction_type) > 0),
    horizon TEXT NOT NULL CHECK(length(horizon) > 0),
    direction TEXT,
    status TEXT NOT NULL CHECK(length(status) > 0),
    score REAL NOT NULL CHECK(score >= 0.0 AND score <= 1.0),
    baseline_comparison_json TEXT NOT NULL DEFAULT '{}',
    evidence_counts_json TEXT NOT NULL DEFAULT '{}',
    signal_counts_json TEXT NOT NULL DEFAULT '{}',
    artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_prediction_evaluations_run_id
ON prediction_evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_prediction_evaluations_candidate_id
ON prediction_evaluations(candidate_id);
CREATE INDEX IF NOT EXISTS idx_prediction_evaluations_artifact_id
ON prediction_evaluations(artifact_id);
CREATE INDEX IF NOT EXISTS idx_prediction_evaluations_created_at
ON prediction_evaluations(created_at);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    outcome_id TEXT PRIMARY KEY CHECK(length(outcome_id) > 0),
    candidate_id TEXT NOT NULL REFERENCES prediction_candidates(candidate_id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id) ON DELETE RESTRICT,
    symbol TEXT NOT NULL CHECK(length(symbol) > 0),
    prediction_type TEXT NOT NULL CHECK(length(prediction_type) > 0),
    horizon TEXT NOT NULL CHECK(length(horizon) > 0),
    evaluation_window_start TEXT NOT NULL,
    evaluation_window_end TEXT NOT NULL,
    status TEXT NOT NULL CHECK(length(status) > 0),
    observed_result TEXT,
    observed_at TEXT,
    result_summary TEXT,
    result_value REAL,
    baseline_value REAL,
    limitations_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(evaluation_window_end > evaluation_window_start)
);

CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_candidate_id
ON prediction_outcomes(candidate_id);
CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_instrument_id
ON prediction_outcomes(instrument_id);
CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_status
ON prediction_outcomes(status);
CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_window
ON prediction_outcomes(evaluation_window_start, evaluation_window_end);

CREATE TABLE IF NOT EXISTS prediction_outcome_evaluations (
    outcome_evaluation_id TEXT PRIMARY KEY CHECK(length(outcome_evaluation_id) > 0),
    run_id TEXT REFERENCES research_runs(run_id) ON DELETE SET NULL,
    outcome_id TEXT NOT NULL REFERENCES prediction_outcomes(outcome_id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL REFERENCES prediction_candidates(candidate_id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL REFERENCES instruments(instrument_id) ON DELETE RESTRICT,
    symbol TEXT NOT NULL CHECK(length(symbol) > 0),
    evaluated_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(length(status) > 0),
    quality_score REAL CHECK(
        quality_score IS NULL
        OR (quality_score >= 0.0 AND quality_score <= 1.0)
    ),
    baseline_comparison_json TEXT NOT NULL DEFAULT '{}',
    artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
    limitations_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prediction_outcome_evaluations_run_id
ON prediction_outcome_evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_prediction_outcome_evaluations_outcome_id
ON prediction_outcome_evaluations(outcome_id);
CREATE INDEX IF NOT EXISTS idx_prediction_outcome_evaluations_candidate_id
ON prediction_outcome_evaluations(candidate_id);
CREATE INDEX IF NOT EXISTS idx_prediction_outcome_evaluations_artifact_id
ON prediction_outcome_evaluations(artifact_id);

CREATE TABLE IF NOT EXISTS outcome_evidence_links (
    outcome_id TEXT NOT NULL REFERENCES prediction_outcomes(outcome_id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL REFERENCES evidence_items(evidence_id) ON DELETE CASCADE,
    relationship TEXT NOT NULL CHECK(length(relationship) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY(outcome_id, evidence_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_outcome_evidence_links_evidence_id
ON outcome_evidence_links(evidence_id);

CREATE TABLE IF NOT EXISTS outcome_artifact_links (
    outcome_id TEXT NOT NULL REFERENCES prediction_outcomes(outcome_id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    relationship TEXT NOT NULL CHECK(length(relationship) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY(outcome_id, artifact_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_outcome_artifact_links_artifact_id
ON outcome_artifact_links(artifact_id);

CREATE TABLE IF NOT EXISTS outcome_evaluation_evidence_links (
    outcome_evaluation_id TEXT NOT NULL
        REFERENCES prediction_outcome_evaluations(outcome_evaluation_id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL REFERENCES evidence_items(evidence_id) ON DELETE CASCADE,
    relationship TEXT NOT NULL CHECK(length(relationship) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY(outcome_evaluation_id, evidence_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_outcome_eval_evidence_links_evidence_id
ON outcome_evaluation_evidence_links(evidence_id);

CREATE TABLE IF NOT EXISTS outcome_evaluation_artifact_links (
    outcome_evaluation_id TEXT NOT NULL
        REFERENCES prediction_outcome_evaluations(outcome_evaluation_id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    relationship TEXT NOT NULL CHECK(length(relationship) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY(outcome_evaluation_id, artifact_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_outcome_eval_artifact_links_artifact_id
ON outcome_evaluation_artifact_links(artifact_id);

CREATE TABLE IF NOT EXISTS calibration_runs (
    calibration_id TEXT PRIMARY KEY CHECK(length(calibration_id) > 0),
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    tool_run_id TEXT REFERENCES tool_runs(tool_run_id) ON DELETE SET NULL,
    method_version TEXT NOT NULL CHECK(length(method_version) > 0),
    created_at TEXT NOT NULL,
    point_in_time_cutoff TEXT NOT NULL,
    cohort_query_json TEXT NOT NULL DEFAULT '{}',
    source_outcome_evaluation_ids_json TEXT NOT NULL DEFAULT '[]',
    artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
    limitations_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_calibration_runs_run_id
ON calibration_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_calibration_runs_tool_run_id
ON calibration_runs(tool_run_id);
CREATE INDEX IF NOT EXISTS idx_calibration_runs_artifact_id
ON calibration_runs(artifact_id);

CREATE TABLE IF NOT EXISTS calibration_slices (
    slice_id TEXT PRIMARY KEY CHECK(length(slice_id) > 0),
    calibration_id TEXT NOT NULL REFERENCES calibration_runs(calibration_id) ON DELETE CASCADE,
    signal_family TEXT,
    prediction_type TEXT,
    horizon TEXT,
    cohort_label TEXT NOT NULL CHECK(length(cohort_label) > 0),
    sample_count INTEGER NOT NULL CHECK(sample_count >= 0),
    resolved_count INTEGER NOT NULL CHECK(resolved_count >= 0),
    pending_count INTEGER NOT NULL CHECK(pending_count >= 0),
    stale_count INTEGER NOT NULL CHECK(stale_count >= 0),
    unavailable_count INTEGER NOT NULL CHECK(unavailable_count >= 0),
    not_evaluable_count INTEGER NOT NULL CHECK(not_evaluable_count >= 0),
    metrics_json TEXT NOT NULL DEFAULT '{}',
    baseline_comparison_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    CHECK(
        sample_count = resolved_count
            + pending_count
            + stale_count
            + unavailable_count
            + not_evaluable_count
    )
);

CREATE INDEX IF NOT EXISTS idx_calibration_slices_calibration_id
ON calibration_slices(calibration_id);
CREATE INDEX IF NOT EXISTS idx_calibration_slices_family
ON calibration_slices(signal_family);
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
    "CalibrationRunRecord",
    "CalibrationSliceRecord",
    "CandidateArtifactLinkRecord",
    "CandidateEvidenceLinkRecord",
    "EvidenceRecord",
    "InstrumentRecord",
    "InstrumentTradabilityEvidenceRecord",
    "OutcomeArtifactLinkRecord",
    "OutcomeEvaluationArtifactLinkRecord",
    "OutcomeEvaluationEvidenceLinkRecord",
    "OutcomeEvidenceLinkRecord",
    "PlanAcceptanceCriterionRecord",
    "PlanArtifactLinkRecord",
    "PlanCommitLinkRecord",
    "PlanDecisionRecord",
    "PlanMilestoneRecord",
    "PlanProgressRecord",
    "PlanRecord",
    "PlanningSQLiteStore",
    "PredictionCandidateRecord",
    "PredictionEvaluationRecord",
    "PredictionOutcomeEvaluationRecord",
    "PredictionOutcomeRecord",
    "ResearchRunRecord",
    "ResearchSQLiteStore",
    "SQLiteStore",
    "SourceQueryRecord",
    "ToolRunRecord",
    "WatchlistItemRecord",
    "WatchlistRecord",
    "initialize_database",
    "initialize_planning_database",
    "initialize_research_database",
]
