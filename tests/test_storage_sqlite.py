from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nlp_stock_prediction.storage import (
    ArtifactRecord,
    EvidenceRecord,
    InstrumentRecord,
    PlanDecisionRecord,
    PlanningSQLiteStore,
    PlanProgressRecord,
    PlanRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    SourceQueryRecord,
    SQLiteStore,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import (
    CURRENT_PLANNING_SCHEMA_VERSION,
    CURRENT_RESEARCH_SCHEMA_VERSION,
    DEFAULT_PLANNING_DATABASE_PATH,
)


def _research_store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "prediction-research.sqlite3")


def _planning_store(tmp_path: Path) -> PlanningSQLiteStore:
    return PlanningSQLiteStore(tmp_path / "plans" / "planning.sqlite3")


def _timestamp() -> datetime:
    return datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _table_names(store: SQLiteStore | PlanningSQLiteStore) -> set[str]:
    with store.connect() as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }


@pytest.mark.unit
def test_research_database_initialization_is_idempotent_and_excludes_planning(
    tmp_path: Path,
) -> None:
    store = _research_store(tmp_path)

    store.initialize()
    store.initialize()

    assert store.schema_version() == CURRENT_RESEARCH_SCHEMA_VERSION

    with store.connect() as connection:
        migration_count = connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
    assert migration_count == 1
    assert {
        "artifacts",
        "evidence_items",
        "instruments",
        "prediction_candidates",
        "research_runs",
        "source_queries",
        "tool_runs",
    }.issubset(_table_names(store))
    assert "plans" not in _table_names(store)

    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_prediction_candidate(
            PredictionCandidateRecord(
                candidate_id="candidate-missing-instrument",
                instrument_id="missing",
                prediction_horizon="5d",
                prediction_type="direction",
                scenario="Missing instrument should fail",
                status="watchlist",
            )
        )


@pytest.mark.unit
def test_planning_database_initialization_is_idempotent_and_excludes_research(
    tmp_path: Path,
) -> None:
    store = _planning_store(tmp_path)

    store.initialize()
    store.initialize()

    assert Path("plans/planning.sqlite3") == DEFAULT_PLANNING_DATABASE_PATH
    assert store.schema_version() == CURRENT_PLANNING_SCHEMA_VERSION

    with store.connect() as connection:
        migration_count = connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
    assert migration_count == 1
    assert {
        "plan_acceptance_criteria",
        "plan_artifact_links",
        "plan_commit_links",
        "plan_decisions",
        "plan_milestones",
        "plan_progress_events",
        "plans",
    }.issubset(_table_names(store))
    assert "artifacts" not in _table_names(store)
    assert "evidence_items" not in _table_names(store)


@pytest.mark.unit
def test_research_database_records_artifact_evidence_and_prediction_candidate(
    tmp_path: Path,
) -> None:
    store = _research_store(tmp_path)
    store.initialize()

    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="equity:NASDAQ:TSLA",
            symbol="TSLA",
            asset_class="stock",
            name="Tesla Inc.",
            venue="NASDAQ",
            aliases=("Tesla", "$TSLA"),
            tradability_source="user_watchlist",
            metadata={"sector": "consumer_discretionary"},
        )
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-2026-05-13",
            run_kind="daily_prediction_report",
            objective="daily prediction report",
            status="running",
            started_at=_timestamp(),
            metadata={"universe": "test"},
        )
    )
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-technical-tsla",
            run_id="run-2026-05-13",
            tool_name="technical_package",
            tool_version="0.1",
            status="ok",
            started_at=_timestamp(),
            completed_at=_timestamp(),
            inputs={"symbol": "TSLA"},
            warnings=("raw_timesfm_research_only",),
        )
    )
    store.record_source_query(
        SourceQueryRecord(
            source_query_id="query-news-tsla",
            tool_run_id="tool-technical-tsla",
            provider="example-news",
            query="TSLA latest catalyst",
            url="https://example.test/search?q=TSLA",
            retrieved_at=_timestamp(),
            metadata={"limit": 10},
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-technical-tsla",
            tool_run_id="tool-technical-tsla",
            artifact_type="technical_package",
            path=Path("artifacts/tools/technical-package/tsla.json"),
            sha256="a" * 64,
            schema_version="technical_package.v1",
            metadata={"latest_bar": "2026-05-12"},
            created_at=_timestamp(),
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-news-tsla",
            source_type="news_article",
            provider="example-news",
            url="https://example.test/tsla",
            query="TSLA latest catalyst",
            retrieved_at=_timestamp(),
            published_at=_timestamp(),
            instruments=("equity:NASDAQ:TSLA",),
            claim="Tesla announced a test catalyst.",
            extraction_confidence=0.82,
            source_reliability="known_publisher",
            freshness_status="fresh",
            artifact_id="artifact-technical-tsla",
            raw_excerpt="announced a test catalyst",
            metadata={"language": "en"},
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-tsla-5d",
            instrument_id="equity:NASDAQ:TSLA",
            prediction_horizon="5d",
            prediction_type="direction",
            scenario="Evidence leans bullish over the next week.",
            direction="bullish",
            confidence=0.63,
            status="moderate_confidence",
            evidence_for=("evidence-news-tsla",),
            evidence_against=(),
            signal_artifacts=("artifact-technical-tsla",),
            baseline={"comparison": "market_neutral"},
            uncertainty="single source catalyst in fixture test",
            metadata={"report_section": "top_candidates"},
        )
    )

    run = store.get_research_run("run-2026-05-13")
    tool_run = store.get_tool_run("tool-technical-tsla")
    source_query = store.get_source_query("query-news-tsla")
    instrument = store.get_instrument("equity:NASDAQ:TSLA")
    artifact = store.get_artifact("artifact-technical-tsla")
    evidence = store.get_evidence("evidence-news-tsla")
    candidate = store.get_prediction_candidate("candidate-tsla-5d")

    assert run is not None
    assert run.metadata["universe"] == "test"
    assert tool_run is not None
    assert tool_run.warnings == ("raw_timesfm_research_only",)
    assert source_query is not None
    assert source_query.metadata["limit"] == 10
    assert instrument is not None
    assert instrument.aliases == ("Tesla", "$TSLA")
    assert instrument.metadata["sector"] == "consumer_discretionary"
    assert artifact is not None
    assert artifact.path == Path("artifacts/tools/technical-package/tsla.json")
    assert artifact.metadata["latest_bar"] == "2026-05-12"
    assert evidence is not None
    assert evidence.instruments == ("equity:NASDAQ:TSLA",)
    assert evidence.extraction_confidence == pytest.approx(0.82)
    assert candidate is not None
    assert candidate.evidence_for == ("evidence-news-tsla",)
    assert candidate.signal_artifacts == ("artifact-technical-tsla",)
    assert candidate.baseline["comparison"] == "market_neutral"


@pytest.mark.unit
def test_planning_database_persists_structured_planning_state(tmp_path: Path) -> None:
    store = _planning_store(tmp_path)
    store.initialize()

    store.upsert_plan(
        PlanRecord(
            plan_id="plan-sqlite-layer",
            slug="sqlite-layer",
            title="SQLite Layer",
            goal="Create local operational memory.",
            status="active",
            priority=10,
            owner_agent="codex",
            non_goals={"visualization": False},
            context={"storage": "sqlite"},
        )
    )
    store.add_plan_decision(
        PlanDecisionRecord(
            decision_id="decision-store-plans",
            plan_id="plan-sqlite-layer",
            decision="Store active plans in SQLite.",
            rationale="Structured rows are easier for Codex to query and update.",
            alternatives_considered="Markdown plans",
            consequences="Markdown source docs stay stable.",
            decided_at=_timestamp(),
        )
    )
    store.add_plan_progress(
        PlanProgressRecord(
            progress_id="progress-schema",
            plan_id="plan-sqlite-layer",
            event_type="completed",
            summary="Schema initialized.",
            details="Added core operational tables.",
            linked_artifact_id="artifact-in-research-db",
            occurred_at=_timestamp(),
        )
    )

    plan = store.get_plan_by_slug("sqlite-layer")
    decisions = store.list_plan_decisions("plan-sqlite-layer")
    progress = store.list_plan_progress("plan-sqlite-layer")

    assert plan is not None
    assert plan.priority == 10
    assert plan.context["storage"] == "sqlite"
    assert decisions == (
        PlanDecisionRecord(
            decision_id="decision-store-plans",
            plan_id="plan-sqlite-layer",
            decision="Store active plans in SQLite.",
            rationale="Structured rows are easier for Codex to query and update.",
            alternatives_considered="Markdown plans",
            consequences="Markdown source docs stay stable.",
            decided_at=_timestamp(),
        ),
    )
    assert progress == (
        PlanProgressRecord(
            progress_id="progress-schema",
            plan_id="plan-sqlite-layer",
            event_type="completed",
            summary="Schema initialized.",
            details="Added core operational tables.",
            linked_artifact_id="artifact-in-research-db",
            occurred_at=_timestamp(),
        ),
    )


@pytest.mark.unit
def test_research_database_rejects_bad_confidence_and_naive_datetimes(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()

    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="crypto:BTC",
            symbol="BTC",
            asset_class="crypto",
        )
    )

    with pytest.raises(ValueError, match="confidence values"):
        store.upsert_prediction_candidate(
            PredictionCandidateRecord(
                candidate_id="candidate-bad-confidence",
                instrument_id="crypto:BTC",
                prediction_horizon="24h",
                prediction_type="direction",
                scenario="Bad confidence should fail.",
                status="watchlist",
                confidence=1.5,
            )
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        store.record_evidence(
            EvidenceRecord(
                evidence_id="evidence-naive-time",
                source_type="web",
                provider="search",
                retrieved_at=datetime(2026, 5, 13, 12, 0),
                claim="Naive timestamps should fail.",
            )
        )
