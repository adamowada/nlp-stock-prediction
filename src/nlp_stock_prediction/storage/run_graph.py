"""Run-scoped query implementation for research SQLite."""

from __future__ import annotations

import sqlite3


def fetch_tool_run_rows(connection: sqlite3.Connection, run_id: str) -> tuple[sqlite3.Row, ...]:
    rows = connection.execute(
        """
        SELECT * FROM tool_runs
        WHERE run_id = ?
        ORDER BY started_at, tool_run_id
        """,
        (run_id,),
    ).fetchall()
    return tuple(rows)


def fetch_artifact_rows(connection: sqlite3.Connection, run_id: str) -> tuple[sqlite3.Row, ...]:
    rows = connection.execute(
        """
        WITH run_evidence AS (
            SELECT DISTINCT evidence_items.evidence_id, evidence_items.artifact_id
            FROM evidence_items
            LEFT JOIN tool_runs direct_tool_runs
                ON evidence_items.tool_run_id = direct_tool_runs.tool_run_id
            LEFT JOIN source_queries
                ON evidence_items.source_query_id = source_queries.source_query_id
            LEFT JOIN tool_runs source_tool_runs
                ON source_queries.tool_run_id = source_tool_runs.tool_run_id
            LEFT JOIN artifacts evidence_artifacts
                ON evidence_items.artifact_id = evidence_artifacts.artifact_id
            LEFT JOIN tool_runs artifact_tool_runs
                ON evidence_artifacts.tool_run_id = artifact_tool_runs.tool_run_id
            LEFT JOIN candidate_evidence_links
                ON evidence_items.evidence_id = candidate_evidence_links.evidence_id
            LEFT JOIN prediction_candidates evidence_candidates
                ON candidate_evidence_links.candidate_id = evidence_candidates.candidate_id
            WHERE direct_tool_runs.run_id = ?
                OR source_tool_runs.run_id = ?
                OR artifact_tool_runs.run_id = ?
                OR evidence_candidates.run_id = ?
        )
        SELECT DISTINCT artifacts.* FROM artifacts
        LEFT JOIN tool_runs
            ON artifacts.tool_run_id = tool_runs.tool_run_id
        LEFT JOIN candidate_artifact_links
            ON artifacts.artifact_id = candidate_artifact_links.artifact_id
        LEFT JOIN prediction_candidates artifact_candidates
            ON candidate_artifact_links.candidate_id = artifact_candidates.candidate_id
        LEFT JOIN run_evidence
            ON artifacts.artifact_id = run_evidence.artifact_id
        WHERE tool_runs.run_id = ?
            OR artifact_candidates.run_id = ?
            OR run_evidence.evidence_id IS NOT NULL
        ORDER BY artifacts.created_at, artifacts.artifact_id
        """,
        (run_id, run_id, run_id, run_id, run_id, run_id),
    ).fetchall()
    return tuple(rows)


def fetch_source_query_rows(connection: sqlite3.Connection, run_id: str) -> tuple[sqlite3.Row, ...]:
    rows = connection.execute(
        """
        WITH run_evidence AS (
            SELECT DISTINCT evidence_items.evidence_id, evidence_items.source_query_id
            FROM evidence_items
            LEFT JOIN tool_runs direct_tool_runs
                ON evidence_items.tool_run_id = direct_tool_runs.tool_run_id
            LEFT JOIN source_queries evidence_source_queries
                ON evidence_items.source_query_id = evidence_source_queries.source_query_id
            LEFT JOIN tool_runs source_tool_runs
                ON evidence_source_queries.tool_run_id = source_tool_runs.tool_run_id
            LEFT JOIN artifacts
                ON evidence_items.artifact_id = artifacts.artifact_id
            LEFT JOIN tool_runs artifact_tool_runs
                ON artifacts.tool_run_id = artifact_tool_runs.tool_run_id
            LEFT JOIN candidate_evidence_links
                ON evidence_items.evidence_id = candidate_evidence_links.evidence_id
            LEFT JOIN prediction_candidates evidence_candidates
                ON candidate_evidence_links.candidate_id = evidence_candidates.candidate_id
            WHERE direct_tool_runs.run_id = ?
                OR source_tool_runs.run_id = ?
                OR artifact_tool_runs.run_id = ?
                OR evidence_candidates.run_id = ?
        )
        SELECT DISTINCT source_queries.* FROM source_queries
        LEFT JOIN tool_runs
            ON source_queries.tool_run_id = tool_runs.tool_run_id
        LEFT JOIN run_evidence
            ON source_queries.source_query_id = run_evidence.source_query_id
        WHERE tool_runs.run_id = ?
            OR run_evidence.evidence_id IS NOT NULL
        ORDER BY source_queries.retrieved_at, source_queries.source_query_id
        """,
        (run_id, run_id, run_id, run_id, run_id),
    ).fetchall()
    return tuple(rows)


def fetch_evidence_rows(connection: sqlite3.Connection, run_id: str) -> tuple[sqlite3.Row, ...]:
    rows = connection.execute(
        """
        SELECT DISTINCT evidence_items.* FROM evidence_items
        LEFT JOIN tool_runs direct_tool_runs
            ON evidence_items.tool_run_id = direct_tool_runs.tool_run_id
        LEFT JOIN source_queries
            ON evidence_items.source_query_id = source_queries.source_query_id
        LEFT JOIN tool_runs source_tool_runs
            ON source_queries.tool_run_id = source_tool_runs.tool_run_id
        LEFT JOIN artifacts
            ON evidence_items.artifact_id = artifacts.artifact_id
        LEFT JOIN tool_runs artifact_tool_runs
            ON artifacts.tool_run_id = artifact_tool_runs.tool_run_id
        LEFT JOIN candidate_evidence_links
            ON evidence_items.evidence_id = candidate_evidence_links.evidence_id
        LEFT JOIN prediction_candidates evidence_candidates
            ON candidate_evidence_links.candidate_id = evidence_candidates.candidate_id
        WHERE direct_tool_runs.run_id = ?
            OR source_tool_runs.run_id = ?
            OR artifact_tool_runs.run_id = ?
            OR evidence_candidates.run_id = ?
        ORDER BY evidence_items.retrieved_at, evidence_items.evidence_id
        """,
        (run_id, run_id, run_id, run_id),
    ).fetchall()
    return tuple(rows)


def fetch_prediction_candidate_rows(
    connection: sqlite3.Connection,
    run_id: str,
) -> tuple[sqlite3.Row, ...]:
    rows = connection.execute(
        """
        SELECT * FROM prediction_candidates
        WHERE run_id = ?
        ORDER BY created_at, candidate_id
        """,
        (run_id,),
    ).fetchall()
    return tuple(rows)


__all__ = [
    "fetch_artifact_rows",
    "fetch_evidence_rows",
    "fetch_prediction_candidate_rows",
    "fetch_source_query_rows",
    "fetch_tool_run_rows",
]
