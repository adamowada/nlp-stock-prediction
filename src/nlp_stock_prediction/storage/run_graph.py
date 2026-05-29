"""Run-scoped query implementation for research SQLite."""

from __future__ import annotations

import sqlite3

_RUN_SCOPE_CTES = """
run_candidates AS (
    SELECT candidate_id FROM prediction_candidates
    WHERE run_id = ?
),
evaluation_outcome_evaluations AS (
    SELECT outcome_evaluation_id FROM prediction_outcome_evaluations
    WHERE run_id = ?
),
evaluation_outcomes AS (
    SELECT DISTINCT outcome_id FROM prediction_outcome_evaluations
    WHERE outcome_evaluation_id IN (
        SELECT outcome_evaluation_id FROM evaluation_outcome_evaluations
    )
)
"""

_RUN_EVIDENCE_FROM = """
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
LEFT JOIN outcome_evidence_links
    ON evidence_items.evidence_id = outcome_evidence_links.evidence_id
LEFT JOIN outcome_evaluation_evidence_links
    ON evidence_items.evidence_id = outcome_evaluation_evidence_links.evidence_id
"""

_RUN_EVIDENCE_WHERE = """
WHERE direct_tool_runs.run_id = ?
    OR source_tool_runs.run_id = ?
    OR artifact_tool_runs.run_id = ?
    OR evidence_candidates.run_id = ?
    OR outcome_evidence_links.outcome_id IN (
        SELECT outcome_id FROM evaluation_outcomes
    )
    OR outcome_evaluation_evidence_links.outcome_evaluation_id IN (
        SELECT outcome_evaluation_id FROM evaluation_outcome_evaluations
    )
"""


def _run_scope_params(run_id: str) -> tuple[str, str]:
    return (run_id, run_id)


def _run_evidence_params(run_id: str) -> tuple[str, str, str, str]:
    return (run_id, run_id, run_id, run_id)


def _run_evidence_cte(selected_column: str, selected_alias: str) -> str:
    allowed_columns = {
        "evidence_items.artifact_id",
        "evidence_items.source_query_id",
    }
    allowed_aliases = {"artifact_id", "source_query_id"}
    if selected_column not in allowed_columns or selected_alias not in allowed_aliases:
        raise ValueError("unsupported run-evidence reachability selection")
    return f"""
    run_evidence AS (
        SELECT DISTINCT evidence_items.evidence_id, {selected_column} AS {selected_alias}
        {_RUN_EVIDENCE_FROM}
        {_RUN_EVIDENCE_WHERE}
    )
    """


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
        f"""
        WITH {_RUN_SCOPE_CTES},
        {_run_evidence_cte("evidence_items.artifact_id", "artifact_id")},
        evaluation_artifacts AS (
            SELECT artifact_id FROM prediction_evaluations
            WHERE artifact_id IS NOT NULL
                AND (
                    run_id = ?
                    OR candidate_id IN (SELECT candidate_id FROM run_candidates)
                )
            UNION
            SELECT artifact_id FROM outcome_artifact_links
            WHERE outcome_id IN (SELECT outcome_id FROM evaluation_outcomes)
            UNION
            SELECT artifact_id FROM prediction_outcome_evaluations
            WHERE artifact_id IS NOT NULL
                AND outcome_evaluation_id IN (
                    SELECT outcome_evaluation_id FROM evaluation_outcome_evaluations
                )
            UNION
            SELECT artifact_id FROM outcome_evaluation_artifact_links
            WHERE outcome_evaluation_id IN (
                SELECT outcome_evaluation_id FROM evaluation_outcome_evaluations
            )
            UNION
            SELECT artifact_id FROM calibration_runs
            WHERE artifact_id IS NOT NULL
                AND run_id = ?
            UNION
            SELECT artifact_id FROM calibration_drift_checks
            WHERE artifact_id IS NOT NULL
                AND run_id = ?
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
            OR artifacts.artifact_id IN (SELECT artifact_id FROM evaluation_artifacts)
        ORDER BY artifacts.created_at, artifacts.artifact_id
        """,
        (
            *_run_scope_params(run_id),
            *_run_evidence_params(run_id),
            run_id,
            run_id,
            run_id,
            run_id,
            run_id,
        ),
    ).fetchall()
    return tuple(rows)


def fetch_source_query_rows(connection: sqlite3.Connection, run_id: str) -> tuple[sqlite3.Row, ...]:
    rows = connection.execute(
        f"""
        WITH {_RUN_SCOPE_CTES},
        {_run_evidence_cte("evidence_items.source_query_id", "source_query_id")}
        SELECT DISTINCT source_queries.* FROM source_queries
        LEFT JOIN tool_runs
            ON source_queries.tool_run_id = tool_runs.tool_run_id
        LEFT JOIN run_evidence
            ON source_queries.source_query_id = run_evidence.source_query_id
        WHERE tool_runs.run_id = ?
            OR run_evidence.evidence_id IS NOT NULL
        ORDER BY source_queries.retrieved_at, source_queries.source_query_id
        """,
        (
            *_run_scope_params(run_id),
            *_run_evidence_params(run_id),
            run_id,
        ),
    ).fetchall()
    return tuple(rows)


def fetch_evidence_rows(connection: sqlite3.Connection, run_id: str) -> tuple[sqlite3.Row, ...]:
    rows = connection.execute(
        f"""
        WITH {_RUN_SCOPE_CTES}
        SELECT DISTINCT evidence_items.*
        {_RUN_EVIDENCE_FROM}
        {_RUN_EVIDENCE_WHERE}
        ORDER BY evidence_items.retrieved_at, evidence_items.evidence_id
        """,
        (*_run_scope_params(run_id), *_run_evidence_params(run_id)),
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
