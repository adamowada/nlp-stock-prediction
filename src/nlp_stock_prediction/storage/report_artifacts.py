"""Report-artifact index query helpers for SQLite storage."""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import cast


def fetch_report_artifact_row(
    connection: sqlite3.Connection,
    artifact_id: str,
) -> sqlite3.Row | None:
    return cast(
        sqlite3.Row | None,
        connection.execute(
            "SELECT * FROM report_artifact_index WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone(),
    )


def fetch_report_artifact_rows_for_run(
    connection: sqlite3.Connection,
    run_id: str,
) -> tuple[sqlite3.Row, ...]:
    rows = connection.execute(
        """
        SELECT * FROM report_artifact_index
        WHERE run_id = ?
        ORDER BY created_at,
            CASE artifact_type
                WHEN 'markdown_report' THEN 1
                WHEN 'json_report' THEN 2
                WHEN 'audit_manifest' THEN 3
                ELSE 4
            END,
            artifact_id
        """,
        (run_id,),
    ).fetchall()
    return tuple(rows)


def fetch_latest_report_artifact_row(
    connection: sqlite3.Connection,
    *,
    artifact_type: str,
    report_date: date | None = None,
    instrument_id: str | None = None,
    symbol: str | None = None,
) -> sqlite3.Row | None:
    filters = ["artifact_type = ?"]
    params: list[str] = [artifact_type]
    if report_date is not None:
        filters.append("report_date = ?")
        params.append(report_date.isoformat())
    if instrument_id is not None:
        filters.append("instrument_id = ?")
        params.append(instrument_id)
    if symbol is not None:
        filters.append("symbol = ?")
        params.append(symbol.strip().upper())
    return cast(
        sqlite3.Row | None,
        connection.execute(
            f"""
            SELECT * FROM report_artifact_index
            WHERE {" AND ".join(filters)}
            ORDER BY created_at DESC, run_id DESC, artifact_id DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone(),
    )


def fetch_latest_prior_report_artifact_row(
    connection: sqlite3.Connection,
    *,
    before_report_date: date,
    artifact_type: str,
    instrument_id: str | None = None,
    symbol: str | None = None,
) -> sqlite3.Row | None:
    filters = ["artifact_type = ?", "report_date < ?"]
    params: list[str] = [artifact_type, before_report_date.isoformat()]
    if instrument_id is not None:
        filters.append("instrument_id = ?")
        params.append(instrument_id)
    if symbol is not None:
        filters.append("symbol = ?")
        params.append(symbol.strip().upper())
    return cast(
        sqlite3.Row | None,
        connection.execute(
            f"""
            SELECT * FROM report_artifact_index
            WHERE {" AND ".join(filters)}
            ORDER BY report_date DESC, created_at DESC, run_id DESC, artifact_id DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone(),
    )


__all__ = [
    "fetch_latest_prior_report_artifact_row",
    "fetch_latest_report_artifact_row",
    "fetch_report_artifact_row",
    "fetch_report_artifact_rows_for_run",
]
