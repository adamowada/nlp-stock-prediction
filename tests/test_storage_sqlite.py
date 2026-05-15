from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import JsonObject
from nlp_stock_prediction.storage import (
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
    PlanningSQLiteStore,
    PlanProgressRecord,
    PlanRecord,
    PredictionCandidateRecord,
    PredictionEvaluationRecord,
    PredictionOutcomeEvaluationRecord,
    PredictionOutcomeRecord,
    ReportArtifactRecord,
    ResearchRunRecord,
    SourceQueryRecord,
    SQLiteStore,
    ToolRunRecord,
    WatchlistItemRecord,
    WatchlistRecord,
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


def _column_names(store: SQLiteStore, table_name: str) -> set[str]:
    with store.connect() as connection:
        return {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }


def _seed_phase6_prediction_graph(store: SQLiteStore) -> None:
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            asset_class="stock",
            name="Microsoft Corporation",
            venue="NASDAQ",
        )
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-phase6-eval",
            run_kind="prediction_evaluation",
            objective="evaluate prior prediction outcomes",
            status="running",
            started_at=_timestamp(),
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-msft-5d",
            run_id="run-phase6-eval",
            instrument_id="equity:NASDAQ:MSFT",
            prediction_horizon="swing",
            prediction_type="directional",
            scenario="Evidence supports a bullish five-day scenario.",
            direction="bullish",
            confidence=0.62,
            status="evidence_supported",
            baseline={"baseline_id": "market-neutral"},
        )
    )
    store.record_source_query(
        SourceQueryRecord(
            source_query_id="query-phase6-outcome",
            provider="example-market-data",
            query="MSFT 2026-05-18 close",
            url="https://example.test/market/msft/2026-05-18",
            retrieved_at=_timestamp(),
            metadata={"purpose": "outcome-observation"},
        )
    )
    for artifact_id, artifact_type in (
        ("artifact-phase6-evidence", "market_data"),
        ("artifact-phase6-evaluation", "prediction_evaluation"),
        ("artifact-phase6-outcome", "prediction_outcome"),
        ("artifact-phase6-review", "prediction_outcome_evaluation"),
        ("artifact-phase6-calibration", "calibration_summary"),
    ):
        store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                path=Path(f"artifacts/phase6/{artifact_id}.json"),
                sha256="e" * 64,
                schema_version=f"{artifact_type}.v1",
                produced_by="phase6_evaluation_calibration",
                record_count=1,
                metadata={"run_id": "run-phase6-eval"},
                created_at=_timestamp(),
            )
        )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-phase6-observed-close",
            source_type="market_data",
            provider="example-market-data",
            retrieved_at=_timestamp(),
            claim="Microsoft closed above the baseline comparison value.",
            source_query_id="query-phase6-outcome",
            url="https://example.test/market/msft/2026-05-18",
            query="MSFT 2026-05-18 close",
            instruments=("equity:NASDAQ:MSFT",),
            extraction_confidence=0.98,
            freshness_status="fresh",
            artifact_id="artifact-phase6-evidence",
            provenance_json={"source_query_id": "query-phase6-outcome"},
        )
    )


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
    assert migration_count == CURRENT_RESEARCH_SCHEMA_VERSION
    assert {
        "artifacts",
        "calibration_runs",
        "calibration_slices",
        "candidate_artifact_links",
        "candidate_evidence_links",
        "evidence_items",
        "instruments",
        "instrument_aliases",
        "instrument_data_availability",
        "instrument_provider_ids",
        "instrument_related_instruments",
        "instrument_tradability_evidence",
        "prediction_candidates",
        "prediction_evaluations",
        "prediction_outcome_evaluations",
        "prediction_outcomes",
        "report_artifact_index",
        "research_runs",
        "outcome_artifact_links",
        "outcome_evaluation_artifact_links",
        "outcome_evaluation_evidence_links",
        "outcome_evidence_links",
        "source_queries",
        "tool_runs",
        "watchlist_items",
        "watchlists",
    }.issubset(_table_names(store))
    assert "plans" not in _table_names(store)

    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_prediction_candidate(
            PredictionCandidateRecord(
                candidate_id="candidate-missing-instrument",
                instrument_id="missing",
                prediction_horizon="swing",
                prediction_type="directional",
                scenario="Missing instrument should fail",
                status="watchlist",
            )
        )


@pytest.mark.unit
def test_research_database_migrates_v2_runtime_graph_columns_idempotently(
    tmp_path: Path,
) -> None:
    store = _research_store(tmp_path)
    with store.connect(create=True) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );
            INSERT INTO schema_migrations(version, name, applied_at)
            VALUES (2, 'instrument_provenance_research_schema', '2026-05-13T00:00:00+00:00');

            CREATE TABLE instruments (
                instrument_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                name TEXT,
                asset_class TEXT NOT NULL,
                venue TEXT,
                aliases_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE research_runs (
                run_id TEXT PRIMARY KEY,
                run_kind TEXT NOT NULL,
                objective TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE tool_runs (
                tool_run_id TEXT PRIMARY KEY,
                run_id TEXT REFERENCES research_runs(run_id) ON DELETE SET NULL,
                tool_name TEXT NOT NULL,
                tool_version TEXT NOT NULL,
                status TEXT NOT NULL,
                inputs_json TEXT NOT NULL DEFAULT '{}',
                started_at TEXT NOT NULL,
                completed_at TEXT,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                error_message TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY,
                tool_run_id TEXT REFERENCES tool_runs(tool_run_id) ON DELETE SET NULL,
                artifact_type TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE source_queries (
                source_query_id TEXT PRIMARY KEY,
                tool_run_id TEXT REFERENCES tool_runs(tool_run_id) ON DELETE SET NULL,
                provider TEXT NOT NULL,
                query TEXT NOT NULL,
                url TEXT,
                retrieved_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE evidence_items (
                evidence_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                provider TEXT NOT NULL,
                url TEXT,
                query TEXT,
                retrieved_at TEXT NOT NULL,
                published_at TEXT,
                instruments_json TEXT NOT NULL DEFAULT '[]',
                claim TEXT NOT NULL,
                extraction_confidence REAL,
                source_reliability TEXT,
                freshness_status TEXT NOT NULL DEFAULT 'unknown',
                artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
                raw_excerpt TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE prediction_candidates (
                candidate_id TEXT PRIMARY KEY,
                instrument_id TEXT NOT NULL
                    REFERENCES instruments(instrument_id) ON DELETE RESTRICT,
                prediction_horizon TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                scenario TEXT NOT NULL,
                direction TEXT,
                confidence REAL,
                status TEXT NOT NULL,
                evidence_for_json TEXT NOT NULL DEFAULT '[]',
                evidence_against_json TEXT NOT NULL DEFAULT '[]',
                signal_artifacts_json TEXT NOT NULL DEFAULT '[]',
                baseline_json TEXT NOT NULL DEFAULT '{}',
                uncertainty TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

    store.initialize()
    store.initialize()

    assert store.schema_version() == CURRENT_RESEARCH_SCHEMA_VERSION
    assert "run_id" in _column_names(store, "prediction_candidates")
    assert {
        "tool_run_id",
        "source_query_id",
        "provenance_json",
    }.issubset(_column_names(store, "evidence_items"))
    assert {"produced_by", "record_count"}.issubset(_column_names(store, "artifacts"))
    assert {
        "calibration_runs",
        "calibration_slices",
        "candidate_artifact_links",
        "candidate_evidence_links",
        "instrument_aliases",
        "instrument_data_availability",
        "instrument_provider_ids",
        "instrument_related_instruments",
        "instrument_tradability_evidence",
        "outcome_artifact_links",
        "outcome_evaluation_artifact_links",
        "outcome_evaluation_evidence_links",
        "outcome_evidence_links",
        "prediction_evaluations",
        "prediction_outcome_evaluations",
        "prediction_outcomes",
        "report_artifact_index",
        "watchlist_items",
        "watchlists",
    }.issubset(_table_names(store))
    with store.connect() as connection:
        migration_count = connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
    assert migration_count == CURRENT_RESEARCH_SCHEMA_VERSION


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
            provider_ids=(
                {"provider": "example-market", "identifier": "TSLA", "namespace": "ticker"},
            ),
            related_instruments=(
                {
                    "instrument_id": "equity:NASDAQ:TSLA_OPTIONS",
                    "relationship": "derivative-chain",
                },
            ),
            tradability_evidence=(
                {
                    "provider": "example-broker",
                    "status": "available",
                    "raw_identifier": "tsla-availability",
                },
            ),
            data_availability=(
                {
                    "provider": "example-market",
                    "data_type": "daily_ohlcv",
                    "status": "available",
                },
            ),
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
            produced_by="technical_package",
            record_count=3,
            metadata={"latest_bar": "2026-05-12"},
            created_at=_timestamp(),
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-news-tsla",
            tool_run_id="tool-technical-tsla",
            source_query_id="query-news-tsla",
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
            provenance_json={
                "provider": "example-news",
                "source_query_id": "query-news-tsla",
            },
            metadata={"language": "en"},
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-tsla-5d",
            run_id="run-2026-05-13",
            instrument_id="equity:NASDAQ:TSLA",
            prediction_horizon="swing",
            prediction_type="directional",
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
    store.link_candidate_evidence(
        CandidateEvidenceLinkRecord(
            candidate_id="candidate-tsla-5d",
            evidence_id="evidence-news-tsla",
            relationship="supports",
            metadata={"stance": "for"},
            created_at=_timestamp(),
        )
    )
    store.link_candidate_evidence(
        CandidateEvidenceLinkRecord(
            candidate_id="candidate-tsla-5d",
            evidence_id="evidence-news-tsla",
            relationship="supports",
            metadata={"stance": "for"},
            created_at=_timestamp(),
        )
    )
    store.link_candidate_artifact(
        CandidateArtifactLinkRecord(
            candidate_id="candidate-tsla-5d",
            artifact_id="artifact-technical-tsla",
            relationship="signal",
            metadata={"section": "technical_package"},
            created_at=_timestamp(),
        )
    )

    run = store.get_research_run("run-2026-05-13")
    tool_run = store.get_tool_run("tool-technical-tsla")
    source_query = store.get_source_query("query-news-tsla")
    instrument = store.get_instrument("equity:NASDAQ:TSLA")
    artifact = store.get_artifact("artifact-technical-tsla")
    evidence = store.get_evidence("evidence-news-tsla")
    candidate = store.get_prediction_candidate("candidate-tsla-5d")
    candidate_evidence_links = store.list_candidate_evidence_links("candidate-tsla-5d")
    candidate_artifact_links = store.list_candidate_artifact_links("candidate-tsla-5d")

    assert run is not None
    assert run.metadata["universe"] == "test"
    assert tool_run is not None
    assert tool_run.warnings == ("raw_timesfm_research_only",)
    assert source_query is not None
    assert source_query.metadata["limit"] == 10
    assert instrument is not None
    assert instrument.aliases == ("Tesla", "$TSLA")
    assert instrument.provider_ids[0]["provider"] == "example-market"
    assert instrument.related_instruments[0]["relationship"] == "derivative-chain"
    assert instrument.tradability_evidence[0]["raw_identifier"] == "tsla-availability"
    assert instrument.data_availability[0]["data_type"] == "daily_ohlcv"
    assert instrument.metadata["sector"] == "consumer_discretionary"
    assert artifact is not None
    assert artifact.path == Path("artifacts/tools/technical-package/tsla.json")
    assert artifact.produced_by == "technical_package"
    assert artifact.record_count == 3
    assert artifact.metadata["latest_bar"] == "2026-05-12"
    assert evidence is not None
    assert evidence.tool_run_id == "tool-technical-tsla"
    assert evidence.source_query_id == "query-news-tsla"
    assert evidence.instruments == ("equity:NASDAQ:TSLA",)
    assert evidence.extraction_confidence == pytest.approx(0.82)
    assert evidence.provenance_json["source_query_id"] == "query-news-tsla"
    assert candidate is not None
    assert candidate.run_id == "run-2026-05-13"
    assert candidate.evidence_for == ("evidence-news-tsla",)
    assert candidate.signal_artifacts == ("artifact-technical-tsla",)
    assert candidate.baseline["comparison"] == "market_neutral"
    assert store.list_tool_runs_for_run("run-2026-05-13") == (tool_run,)
    assert store.list_source_queries_for_run("run-2026-05-13") == (source_query,)
    assert store.list_artifacts_for_run("run-2026-05-13") == (artifact,)
    assert store.list_evidence_for_run("run-2026-05-13") == (evidence,)
    assert store.list_prediction_candidates_for_run("run-2026-05-13") == (candidate,)
    assert candidate_evidence_links == (
        CandidateEvidenceLinkRecord(
            candidate_id="candidate-tsla-5d",
            evidence_id="evidence-news-tsla",
            relationship="supports",
            metadata={"stance": "for"},
            created_at=_timestamp(),
        ),
    )
    assert candidate_artifact_links == (
        CandidateArtifactLinkRecord(
            candidate_id="candidate-tsla-5d",
            artifact_id="artifact-technical-tsla",
            relationship="signal",
            metadata={"section": "technical_package"},
            created_at=_timestamp(),
        ),
    )


@pytest.mark.unit
def test_phase6_evaluation_calibration_records_round_trip_and_extend_run_graph(
    tmp_path: Path,
) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    _seed_phase6_prediction_graph(store)

    evaluation = PredictionEvaluationRecord(
        evaluation_id="evaluation-msft-5d",
        run_id="run-phase6-eval",
        candidate_id="candidate-msft-5d",
        instrument_id="equity:NASDAQ:MSFT",
        symbol="MSFT",
        created_at=_timestamp(),
        prediction_type="directional",
        horizon="swing",
        direction="bullish",
        status="evidence_supported",
        score=0.68,
        baseline_comparison={
            "baseline_id": "market-neutral",
            "verdict": "above_baseline",
        },
        evidence_counts={"supporting_source_evidence": 1},
        signal_counts={"technical": 0, "news": 0},
        artifact_id="artifact-phase6-evaluation",
    )
    outcome = PredictionOutcomeRecord(
        outcome_id="outcome-msft-5d",
        candidate_id="candidate-msft-5d",
        instrument_id="equity:NASDAQ:MSFT",
        symbol="MSFT",
        prediction_type="directional",
        horizon="swing",
        evaluation_window_start=datetime(2026, 5, 13, 20, 0, tzinfo=UTC),
        evaluation_window_end=datetime(2026, 5, 18, 20, 0, tzinfo=UTC),
        status="observed",
        observed_result="supported",
        observed_at=datetime(2026, 5, 18, 20, 0, tzinfo=UTC),
        result_summary="The observed close was above the comparison baseline.",
        result_value=433.25,
        baseline_value=428.1,
        metadata={"target_id": "target-msft-5d"},
    )
    outcome_evaluation = PredictionOutcomeEvaluationRecord(
        outcome_evaluation_id="outcome-evaluation-msft-5d",
        run_id="run-phase6-eval",
        outcome_id="outcome-msft-5d",
        candidate_id="candidate-msft-5d",
        instrument_id="equity:NASDAQ:MSFT",
        symbol="MSFT",
        evaluated_at=datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
        status="confirmed",
        quality_score=0.74,
        baseline_comparison={"verdict": "above_baseline"},
        artifact_id="artifact-phase6-review",
    )
    calibration_run = CalibrationRunRecord(
        calibration_id="calibration-msft-cohort",
        run_id="run-phase6-eval",
        method_version="phase6-evalcal.v1",
        created_at=datetime(2026, 5, 18, 22, 0, tzinfo=UTC),
        point_in_time_cutoff=datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
        cohort_query={"prediction_type": "directional", "horizon": "swing"},
        source_outcome_evaluation_ids=("outcome-evaluation-msft-5d",),
        artifact_id="artifact-phase6-calibration",
        limitations=("single resolved record in unit test",),
    )
    calibration_slice = CalibrationSliceRecord(
        slice_id="calibration-msft-cohort-overall",
        calibration_id="calibration-msft-cohort",
        cohort_label="overall",
        sample_count=1,
        resolved_count=1,
        metrics={"brier_score": 0.06, "accuracy": 1.0},
        baseline_comparison={"verdict": "above_baseline"},
        provenance={"source_outcome_evaluation_ids": ["outcome-evaluation-msft-5d"]},
    )

    store.record_prediction_evaluation(evaluation)
    store.upsert_prediction_outcome(outcome)
    store.link_outcome_evidence(
        OutcomeEvidenceLinkRecord(
            outcome_id="outcome-msft-5d",
            evidence_id="evidence-phase6-observed-close",
            relationship="observes",
            metadata={"field": "close"},
            created_at=_timestamp(),
        )
    )
    store.link_outcome_artifact(
        OutcomeArtifactLinkRecord(
            outcome_id="outcome-msft-5d",
            artifact_id="artifact-phase6-outcome",
            relationship="outcome_payload",
            created_at=_timestamp(),
        )
    )
    store.upsert_prediction_outcome_evaluation(outcome_evaluation)
    store.link_outcome_evaluation_evidence(
        OutcomeEvaluationEvidenceLinkRecord(
            outcome_evaluation_id="outcome-evaluation-msft-5d",
            evidence_id="evidence-phase6-observed-close",
            relationship="supports_review",
            created_at=_timestamp(),
        )
    )
    store.link_outcome_evaluation_artifact(
        OutcomeEvaluationArtifactLinkRecord(
            outcome_evaluation_id="outcome-evaluation-msft-5d",
            artifact_id="artifact-phase6-review",
            relationship="review_payload",
            created_at=_timestamp(),
        )
    )
    store.record_calibration_run(calibration_run)
    store.record_calibration_slice(calibration_slice)

    assert store.get_prediction_evaluation("evaluation-msft-5d") == evaluation
    assert store.list_prediction_evaluations_for_run("run-phase6-eval") == (evaluation,)
    assert store.list_prediction_evaluations_for_candidate("candidate-msft-5d") == (evaluation,)
    assert store.get_prediction_outcome("outcome-msft-5d") == outcome
    assert store.list_prediction_outcomes_for_candidate("candidate-msft-5d") == (outcome,)
    assert store.list_outcome_evidence_links("outcome-msft-5d") == (
        OutcomeEvidenceLinkRecord(
            outcome_id="outcome-msft-5d",
            evidence_id="evidence-phase6-observed-close",
            relationship="observes",
            metadata={"field": "close"},
            created_at=_timestamp(),
        ),
    )
    assert store.list_outcome_artifact_links("outcome-msft-5d") == (
        OutcomeArtifactLinkRecord(
            outcome_id="outcome-msft-5d",
            artifact_id="artifact-phase6-outcome",
            relationship="outcome_payload",
            created_at=_timestamp(),
        ),
    )
    assert (
        store.get_prediction_outcome_evaluation("outcome-evaluation-msft-5d") == outcome_evaluation
    )
    assert store.list_outcome_evaluations_for_run("run-phase6-eval") == (outcome_evaluation,)
    assert store.list_outcome_evaluation_evidence_links("outcome-evaluation-msft-5d") == (
        OutcomeEvaluationEvidenceLinkRecord(
            outcome_evaluation_id="outcome-evaluation-msft-5d",
            evidence_id="evidence-phase6-observed-close",
            relationship="supports_review",
            created_at=_timestamp(),
        ),
    )
    assert store.list_outcome_evaluation_artifact_links("outcome-evaluation-msft-5d") == (
        OutcomeEvaluationArtifactLinkRecord(
            outcome_evaluation_id="outcome-evaluation-msft-5d",
            artifact_id="artifact-phase6-review",
            relationship="review_payload",
            created_at=_timestamp(),
        ),
    )
    assert store.get_calibration_run("calibration-msft-cohort") == calibration_run
    assert store.list_calibration_runs_for_run("run-phase6-eval") == (calibration_run,)
    assert store.list_calibration_slices("calibration-msft-cohort") == (calibration_slice,)

    assert {
        artifact.artifact_id for artifact in store.list_artifacts_for_run("run-phase6-eval")
    } == {
        "artifact-phase6-calibration",
        "artifact-phase6-evaluation",
        "artifact-phase6-evidence",
        "artifact-phase6-outcome",
        "artifact-phase6-review",
    }
    assert tuple(
        evidence.evidence_id for evidence in store.list_evidence_for_run("run-phase6-eval")
    ) == ("evidence-phase6-observed-close",)
    assert tuple(
        query.source_query_id for query in store.list_source_queries_for_run("run-phase6-eval")
    ) == ("query-phase6-outcome",)

    store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-phase6-failed",
            run_id="run-phase6-eval",
            tool_name="phase6_evaluation",
            tool_version="0.1",
            status="failed",
            started_at=_timestamp(),
            error_message="simulated failure after partial writes",
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-phase6-failed",
            tool_run_id="tool-phase6-failed",
            artifact_type="prediction_evaluation",
            path=Path("artifacts/phase6/failed-evaluation.json"),
            sha256="f" * 64,
            schema_version="prediction_evaluation.v1",
            created_at=_timestamp(),
        )
    )
    store.record_prediction_evaluation(
        PredictionEvaluationRecord(
            evaluation_id="evaluation-msft-failed",
            run_id="run-phase6-eval",
            candidate_id="candidate-msft-5d",
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            created_at=_timestamp(),
            prediction_type="directional",
            horizon="swing",
            status="insufficient_evidence",
            score=0.1,
            artifact_id="artifact-phase6-failed",
        )
    )
    store.record_calibration_run(
        CalibrationRunRecord(
            calibration_id="calibration-msft-failed",
            run_id="run-phase6-eval",
            tool_run_id="tool-phase6-failed",
            method_version="phase6-evalcal.v1",
            created_at=datetime(2026, 5, 18, 23, 0, tzinfo=UTC),
            point_in_time_cutoff=datetime(2026, 5, 18, 22, 0, tzinfo=UTC),
            artifact_id="artifact-phase6-failed",
        )
    )

    store.delete_tool_run_outputs("tool-phase6-failed")

    assert store.get_prediction_evaluation("evaluation-msft-failed") is None
    assert store.get_calibration_run("calibration-msft-failed") is None
    assert store.get_artifact("artifact-phase6-failed") is None
    assert store.get_tool_run("tool-phase6-failed") is None

    with pytest.raises(ValueError, match="sample_count"):
        store.record_calibration_slice(
            CalibrationSliceRecord(
                slice_id="calibration-msft-invalid",
                calibration_id="calibration-msft-cohort",
                cohort_label="invalid",
                sample_count=2,
                resolved_count=1,
            )
        )


@pytest.mark.unit
def test_phase6_storage_rejects_incoherent_outcome_evaluation_links(
    tmp_path: Path,
) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    _seed_phase6_prediction_graph(store)
    store.upsert_prediction_outcome(
        PredictionOutcomeRecord(
            outcome_id="outcome-msft-5d",
            candidate_id="candidate-msft-5d",
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            prediction_type="directional",
            horizon="swing",
            evaluation_window_start=datetime(2026, 5, 13, 20, 0, tzinfo=UTC),
            evaluation_window_end=datetime(2026, 5, 18, 20, 0, tzinfo=UTC),
            status="observed",
            observed_result="supported",
            observed_at=datetime(2026, 5, 18, 20, 0, tzinfo=UTC),
        )
    )

    with pytest.raises(ValueError, match="candidate_id must match outcome"):
        store.upsert_prediction_outcome_evaluation(
            PredictionOutcomeEvaluationRecord(
                outcome_evaluation_id="outcome-evaluation-mismatch",
                run_id="run-phase6-eval",
                outcome_id="outcome-msft-5d",
                candidate_id="candidate-other",
                instrument_id="equity:NASDAQ:MSFT",
                symbol="MSFT",
                evaluated_at=datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
                status="confirmed",
                quality_score=0.74,
            )
        )


@pytest.mark.unit
def test_prediction_evaluation_rejects_candidate_run_instrument_and_symbol_drift(
    tmp_path: Path,
) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    _seed_phase6_prediction_graph(store)

    with pytest.raises(ValueError, match="candidate/instrument mismatch"):
        store.record_prediction_evaluation(
            PredictionEvaluationRecord(
                evaluation_id="evaluation-msft-wrong-instrument",
                run_id="run-phase6-eval",
                candidate_id="candidate-msft-5d",
                instrument_id="equity:NASDAQ:AAPL",
                symbol="AAPL",
                created_at=_timestamp(),
                prediction_type="directional",
                horizon="swing",
                status="insufficient_evidence",
                score=0.1,
            )
        )

    with pytest.raises(ValueError, match="symbol must match instrument"):
        store.record_prediction_evaluation(
            PredictionEvaluationRecord(
                evaluation_id="evaluation-msft-wrong-symbol",
                run_id="run-phase6-eval",
                candidate_id="candidate-msft-5d",
                instrument_id="equity:NASDAQ:MSFT",
                symbol="AAPL",
                created_at=_timestamp(),
                prediction_type="directional",
                horizon="swing",
                status="insufficient_evidence",
                score=0.1,
            )
        )


@pytest.mark.unit
def test_calibration_run_rejects_cross_run_outcome_evaluation_sources(
    tmp_path: Path,
) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    _seed_phase6_prediction_graph(store)
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-other",
            run_kind="prediction_evaluation",
            objective="other run",
            status="completed",
            started_at=_timestamp(),
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-other-msft-5d",
            run_id="run-other",
            instrument_id="equity:NASDAQ:MSFT",
            prediction_horizon="swing",
            prediction_type="directional",
            scenario="Other run candidate.",
            status="evidence_supported",
        )
    )
    store.upsert_prediction_outcome(
        PredictionOutcomeRecord(
            outcome_id="outcome-other-msft-5d",
            candidate_id="candidate-other-msft-5d",
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            prediction_type="directional",
            horizon="swing",
            evaluation_window_start=datetime(2026, 5, 13, 20, 0, tzinfo=UTC),
            evaluation_window_end=datetime(2026, 5, 18, 20, 0, tzinfo=UTC),
            status="observed",
            observed_result="supported",
            observed_at=datetime(2026, 5, 18, 20, 0, tzinfo=UTC),
        )
    )
    store.upsert_prediction_outcome_evaluation(
        PredictionOutcomeEvaluationRecord(
            outcome_evaluation_id="outcome-evaluation-other-msft-5d",
            run_id="run-other",
            outcome_id="outcome-other-msft-5d",
            candidate_id="candidate-other-msft-5d",
            instrument_id="equity:NASDAQ:MSFT",
            symbol="MSFT",
            evaluated_at=datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
            status="confirmed",
            quality_score=0.74,
        )
    )

    with pytest.raises(ValueError, match="must match run_id"):
        store.record_calibration_run(
            CalibrationRunRecord(
                calibration_id="calibration-cross-run",
                run_id="run-phase6-eval",
                method_version="phase6-evalcal.v1",
                created_at=datetime(2026, 5, 18, 22, 0, tzinfo=UTC),
                point_in_time_cutoff=datetime(2026, 5, 18, 21, 0, tzinfo=UTC),
                source_outcome_evaluation_ids=("outcome-evaluation-other-msft-5d",),
            )
        )


@pytest.mark.unit
def test_record_artifact_rejects_absolute_and_parent_traversal_paths(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()

    with pytest.raises(ValueError, match="relative"):
        store.record_artifact(
            ArtifactRecord(
                artifact_id="artifact-invalid-path-absolute",
                artifact_type="provider_result",
                path=tmp_path / "absolute.json",
                sha256="b" * 64,
                schema_version="unit.v1",
            )
        )
    with pytest.raises(ValueError, match="parent traversal"):
        store.record_artifact(
            ArtifactRecord(
                artifact_id="artifact-invalid-path-traversal",
                artifact_type="provider_result",
                path=Path("artifacts/../leak.json"),
                sha256="b" * 64,
                schema_version="unit.v1",
            )
        )
    with pytest.raises(ValueError, match="relative"):
        store.record_artifact(
            ArtifactRecord(
                artifact_id="artifact-invalid-path-rooted",
                artifact_type="provider_result",
                path=Path("\\tmp\\leak.json"),
                sha256="b" * 64,
                schema_version="unit.v1",
            )
        )
    with pytest.raises(ValueError, match="relative"):
        store.record_artifact(
            ArtifactRecord(
                artifact_id="artifact-invalid-path-drive",
                artifact_type="provider_result",
                path=Path("C:\\tmp\\leak.json"),
                sha256="b" * 64,
                schema_version="unit.v1",
            )
        )
    with pytest.raises(ValueError, match="name a file"):
        store.record_artifact(
            ArtifactRecord(
                artifact_id="artifact-invalid-path-current",
                artifact_type="provider_result",
                path=Path("."),
                sha256="b" * 64,
                schema_version="unit.v1",
            )
        )


@pytest.mark.unit
def test_artifact_upsert_refreshes_created_at_for_run_graph_ordering(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    earlier = _timestamp()
    later = earlier.replace(hour=13)
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-refresh",
            artifact_type="provider_result",
            path=Path("artifacts/refresh.json"),
            sha256="c" * 64,
            schema_version="unit.v1",
            created_at=earlier,
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-refresh",
            artifact_type="provider_result",
            path=Path("artifacts/refresh.json"),
            sha256="d" * 64,
            schema_version="unit.v1",
            created_at=later,
        )
    )

    artifact = store.get_artifact("artifact-refresh")
    assert artifact is not None
    assert artifact.created_at == later


@pytest.mark.unit
def test_candidate_links_are_repaired_when_evidence_and_artifacts_arrive_later(
    tmp_path: Path,
) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    now = _timestamp()
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-link-repair",
            run_kind="unit",
            objective="repair candidate links",
            status="running",
            started_at=now,
        )
    )
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-link-repair",
            run_id="run-link-repair",
            tool_name="unit_tool",
            tool_version="unit.v1",
            status="successful",
            started_at=now,
        )
    )
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="instrument:unit:tsla",
            symbol="TSLA",
            asset_class="stock",
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-link-repair",
            run_id="run-link-repair",
            instrument_id="instrument:unit:tsla",
            prediction_horizon="swing",
            prediction_type="directional",
            scenario="Candidate references records that arrive later.",
            status="insufficient_evidence",
            evidence_for=("evidence-late",),
            signal_artifacts=("artifact-late",),
        )
    )

    assert store.list_candidate_evidence_links("candidate-link-repair") == ()
    assert store.list_candidate_artifact_links("candidate-link-repair") == ()

    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-late",
            tool_run_id="tool-link-repair",
            artifact_type="technical_package",
            path=Path("artifacts/late.json"),
            sha256="e" * 64,
            schema_version="technical_package.v1",
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-late",
            tool_run_id="tool-link-repair",
            source_type="news_article",
            provider="unit-news",
            retrieved_at=now,
            claim="Late evidence arrived after candidate storage.",
        )
    )

    assert tuple(
        link.evidence_id for link in store.list_candidate_evidence_links("candidate-link-repair")
    ) == ("evidence-late",)
    assert tuple(
        link.artifact_id for link in store.list_candidate_artifact_links("candidate-link-repair")
    ) == ("artifact-late",)


@pytest.mark.unit
def test_report_artifact_index_round_trips_and_finds_latest_bundle(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    earlier = _timestamp()
    later = earlier.replace(hour=13)
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-report-index",
            run_kind="daily_prediction_report",
            objective="index final report artifacts",
            status="completed",
            started_at=earlier,
            completed_at=later,
            metadata={"run_date": "2026-05-13"},
        )
    )
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-render-report-index",
            run_id="run-report-index",
            tool_name="render_prediction_report",
            tool_version="phase5.report-index.v1",
            status="successful",
            started_at=earlier,
            completed_at=later,
            inputs={"symbol": "TSLA"},
        )
    )
    for artifact_id, artifact_type, path, digest in (
        (
            "artifact-report-md-index",
            "markdown_report",
            Path("reports/2026-05-13/tsla/report.md"),
            "a" * 64,
        ),
        (
            "artifact-report-json-index",
            "json_report",
            Path("reports/2026-05-13/tsla/report.json"),
            "b" * 64,
        ),
        (
            "artifact-report-audit-index",
            "audit_manifest",
            Path("reports/2026-05-13/tsla/audit/audit-manifest.json"),
            "c" * 64,
        ),
    ):
        store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                tool_run_id="tool-render-report-index",
                artifact_type=artifact_type,
                path=path,
                sha256=digest,
                schema_version="phase5-report.v1",
                produced_by="render_prediction_report",
                metadata={"run_id": "run-report-index"},
                created_at=later,
            )
        )
        store.record_report_artifact(
            ReportArtifactRecord(
                artifact_id=artifact_id,
                run_id="run-report-index",
                tool_run_id="tool-render-report-index",
                artifact_type=artifact_type,
                path=path,
                sha256=digest,
                schema_version="phase5-report.v1",
                report_schema_version="daily-report.v2",
                report_date=date(2026, 5, 13),
                instrument_id="instrument:equity:us:tsla",
                symbol="tsla",
                report_data_mode="offline_fixture",
                source_run_started_at=earlier,
                source_run_completed_at=later,
                metadata={"candidate_count": 1},
                created_at=later,
            )
        )

    report_artifacts = store.list_report_artifacts_for_run("run-report-index")
    assert tuple(artifact.artifact_type for artifact in report_artifacts) == (
        "markdown_report",
        "json_report",
        "audit_manifest",
    )
    assert report_artifacts[1] == ReportArtifactRecord(
        artifact_id="artifact-report-json-index",
        run_id="run-report-index",
        tool_run_id="tool-render-report-index",
        artifact_type="json_report",
        path=Path("reports/2026-05-13/tsla/report.json"),
        sha256="b" * 64,
        schema_version="phase5-report.v1",
        report_schema_version="daily-report.v2",
        report_date=date(2026, 5, 13),
        instrument_id="instrument:equity:us:tsla",
        symbol="TSLA",
        report_data_mode="offline_fixture",
        source_run_started_at=earlier,
        source_run_completed_at=later,
        metadata={"candidate_count": 1},
        created_at=later,
    )
    latest = store.get_latest_report_artifact(
        report_date=date(2026, 5, 13),
        instrument_id="instrument:equity:us:tsla",
        artifact_type="json_report",
    )
    assert latest == report_artifacts[1]
    assert store.get_latest_report_artifact(symbol="TSLA") == report_artifacts[1]
    assert (
        store.get_latest_prior_report_artifact(
            before_report_date=date(2026, 5, 14),
            symbol="TSLA",
        )
        == report_artifacts[1]
    )
    assert (
        store.get_latest_prior_report_artifact(
            before_report_date=date(2026, 5, 13),
            symbol="TSLA",
        )
        is None
    )
    assert store.list_latest_report_artifact_bundle(symbol="TSLA") == report_artifacts

    columns = _column_names(store, "report_artifact_index")
    assert "report_body" not in columns
    assert "payload_json" not in columns
    with store.connect() as connection:
        stored_metadata = connection.execute(
            """
            SELECT metadata_json FROM report_artifact_index
            WHERE artifact_id = 'artifact-report-json-index'
            """
        ).fetchone()[0]
    assert "candidate_count" in stored_metadata
    assert "prediction_candidates" not in stored_metadata


@pytest.mark.unit
def test_report_artifact_index_must_match_artifact_ledger(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    now = _timestamp()
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-report-index-mismatch",
            run_kind="daily_prediction_report",
            objective="index final report artifacts",
            status="completed",
            started_at=now,
        )
    )
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-render-report-mismatch",
            run_id="run-report-index-mismatch",
            tool_name="render_prediction_report",
            tool_version="phase5-report.v1",
            status="successful",
            started_at=now,
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-report-json-mismatch",
            tool_run_id="tool-render-report-mismatch",
            artifact_type="json_report",
            path=Path("reports/report.json"),
            sha256="f" * 64,
            schema_version="phase5-report.v1",
            produced_by="render_prediction_report",
            created_at=now,
        )
    )

    with pytest.raises(ValueError, match="artifact ledger"):
        store.record_report_artifact(
            ReportArtifactRecord(
                artifact_id="artifact-report-json-mismatch",
                run_id="run-report-index-mismatch",
                tool_run_id="tool-render-report-mismatch",
                artifact_type="json_report",
                path=Path("reports/other.json"),
                sha256="f" * 64,
                schema_version="phase5-report.v1",
                report_schema_version="daily-report.v2",
                report_date=date(2026, 5, 13),
                report_data_mode="offline_fixture",
                source_run_started_at=now,
            )
        )


@pytest.mark.unit
def test_report_artifact_index_must_match_artifact_tool_run(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    now = _timestamp()
    for run_id in ("run-report-index-source", "run-report-index-wrong"):
        store.upsert_research_run(
            ResearchRunRecord(
                run_id=run_id,
                run_kind="daily_prediction_report",
                objective="index final report artifacts",
                status="completed",
                started_at=now,
            )
        )
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-render-report-run-source",
            run_id="run-report-index-source",
            tool_name="render_prediction_report",
            tool_version="phase5-report.v1",
            status="successful",
            started_at=now,
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-report-json-run-source",
            tool_run_id="tool-render-report-run-source",
            artifact_type="json_report",
            path=Path("reports/report.json"),
            sha256="f" * 64,
            schema_version="phase5-report.v1",
            produced_by="render_prediction_report",
            created_at=now,
        )
    )

    with pytest.raises(ValueError, match="run_id must match artifact tool run"):
        store.record_report_artifact(
            ReportArtifactRecord(
                artifact_id="artifact-report-json-run-source",
                run_id="run-report-index-wrong",
                tool_run_id="tool-render-report-run-source",
                artifact_type="json_report",
                path=Path("reports/report.json"),
                sha256="f" * 64,
                schema_version="phase5-report.v1",
                report_schema_version="daily-report.v2",
                report_date=date(2026, 5, 13),
                report_data_mode="live",
                source_run_started_at=now,
            )
        )


@pytest.mark.unit
def test_storage_rejects_non_finite_json_values(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()

    with pytest.raises(ValueError, match="Out of range float values"):
        store.record_source_query(
            SourceQueryRecord(
                source_query_id="query-non-finite-json",
                provider="unit-provider",
                query="MSFT latest quote",
                retrieved_at=_timestamp(),
                metadata={"bad_value": float("nan")},
            )
        )


@pytest.mark.unit
def test_instrument_registry_tables_round_trip_and_query_helpers(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()

    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="equity:NASDAQ:TSLA",
            symbol="TSLA",
            asset_class="stock",
            name="Tesla Inc.",
            venue="NASDAQ",
            aliases=("Tesla", "$TSLA", "common-alias"),
            provider_ids=(
                {
                    "provider": "ExampleMarket",
                    "namespace": "ticker",
                    "identifier": "TSLA",
                    "region": "US",
                },
                {
                    "provider": "FIGI",
                    "namespace": "composite",
                    "identifier": "BBG000N9MNX3",
                },
            ),
            related_instruments=(
                {
                    "instrument_id": "equity:NASDAQ:TSLA_OPTIONS",
                    "relationship": "derivative-chain",
                    "source": "fixture",
                },
            ),
            tradability_evidence=(
                {
                    "provider": "ExampleBroker",
                    "status": "available",
                    "retrieved_at": _timestamp().isoformat(),
                    "source_url": "https://example.test/tradability/tsla",
                    "raw_identifier": "tsla-availability",
                    "extraction_confidence": 0.9,
                    "restriction": "none",
                },
            ),
            data_availability=(
                {
                    "provider": "ExampleMarket",
                    "data_type": "daily_ohlcv",
                    "status": "available",
                    "checked_at": _timestamp().isoformat(),
                    "lag": "1d",
                },
            ),
            metadata={"sector": "consumer_discretionary"},
        )
    )
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="crypto:BTC",
            symbol="BTC",
            asset_class="crypto",
            aliases=("Bitcoin", "common-alias"),
            provider_ids=(
                {
                    "provider": "ExampleMarket",
                    "namespace": "ticker",
                    "identifier": "BTC-USD",
                },
            ),
        )
    )

    assert tuple(item.instrument_id for item in store.list_instruments()) == (
        "crypto:BTC",
        "equity:NASDAQ:TSLA",
    )
    assert tuple(
        item.instrument_id for item in store.find_instruments_by_symbol_or_alias("common-alias")
    ) == ("crypto:BTC", "equity:NASDAQ:TSLA")
    tesla_matches = store.find_instruments_by_symbol_or_alias("tesla")
    assert tuple(item.instrument_id for item in tesla_matches) == ("equity:NASDAQ:TSLA",)
    assert store.find_instrument_by_provider_id(
        "examplemarket", "ticker", "TSLA"
    ) == store.get_instrument("equity:NASDAQ:TSLA")
    assert store.find_instrument_by_provider_id("FIGI", "composite", "missing") is None
    assert tuple(item.instrument_id for item in store.list_instruments_by_asset_class("STOCK")) == (
        "equity:NASDAQ:TSLA",
    )

    latest = store.get_latest_tradability_evidence("equity:NASDAQ:TSLA", "examplebroker")
    assert latest == InstrumentTradabilityEvidenceRecord(
        instrument_id="equity:NASDAQ:TSLA",
        provider="ExampleBroker",
        status="available",
        retrieved_at=_timestamp(),
        url="https://example.test/tradability/tsla",
        raw_identifier="tsla-availability",
        extraction_confidence=0.9,
        metadata={"restriction": "none"},
    )

    with store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM instrument_aliases").fetchone()[0] == 5
        assert connection.execute("SELECT count(*) FROM instrument_provider_ids").fetchone()[0] == 3
        assert (
            connection.execute("SELECT count(*) FROM instrument_data_availability").fetchone()[0]
            == 1
        )


@pytest.mark.unit
def test_instrument_upsert_replaces_normalized_children_idempotently(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()

    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="etf:NYSEARCA:SPY",
            symbol="SPY",
            asset_class="etf",
            aliases=("SPDR S&P 500 ETF", "SPY ETF"),
            provider_ids=(
                {"provider": "ExampleMarket", "namespace": "ticker", "identifier": "SPY"},
            ),
        )
    )
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="etf:NYSEARCA:SPY",
            symbol="SPY",
            asset_class="etf",
            aliases=("SPY Trust",),
            provider_ids=(
                {"provider": "ExampleMarket", "namespace": "ticker", "identifier": "SPY"},
            ),
        )
    )

    assert (
        tuple(item.instrument_id for item in store.find_instruments_by_symbol_or_alias("SPY ETF"))
        == ()
    )
    spy_trust_matches = store.find_instruments_by_symbol_or_alias("SPY Trust")
    assert tuple(item.instrument_id for item in spy_trust_matches) == ("etf:NYSEARCA:SPY",)
    with store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM instrument_aliases").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM instrument_provider_ids").fetchone()[0] == 1


@pytest.mark.unit
def test_watchlists_round_trip_and_list_instruments(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="equity:NASDAQ:NVDA",
            symbol="NVDA",
            asset_class="stock",
        )
    )
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="etf:NYSEARCA:QQQ",
            symbol="QQQ",
            asset_class="etf",
        )
    )
    store.upsert_watchlist(
        WatchlistRecord(
            watchlist_id="watchlist-ai",
            name="AI Research",
            description="Phase 3 fixture watchlist",
            metadata={"owner": "tests"},
        )
    )
    store.upsert_watchlist_item(
        WatchlistItemRecord(
            watchlist_id="watchlist-ai",
            instrument_id="equity:NASDAQ:NVDA",
            sort_order=2,
            notes="single-name exposure",
            metadata={"reason": "semis"},
        )
    )
    store.upsert_watchlist_item(
        WatchlistItemRecord(
            watchlist_id="watchlist-ai",
            instrument_id="etf:NYSEARCA:QQQ",
            sort_order=1,
        )
    )
    store.upsert_watchlist_item(
        WatchlistItemRecord(
            watchlist_id="watchlist-ai",
            instrument_id="equity:NASDAQ:NVDA",
            sort_order=3,
            notes="updated note",
        )
    )

    assert store.get_watchlist("watchlist-ai") == WatchlistRecord(
        watchlist_id="watchlist-ai",
        name="AI Research",
        description="Phase 3 fixture watchlist",
        metadata={"owner": "tests"},
    )
    assert store.list_watchlists() == (
        WatchlistRecord(
            watchlist_id="watchlist-ai",
            name="AI Research",
            description="Phase 3 fixture watchlist",
            metadata={"owner": "tests"},
        ),
    )
    assert store.list_watchlist_items("watchlist-ai") == (
        WatchlistItemRecord(
            watchlist_id="watchlist-ai",
            instrument_id="etf:NYSEARCA:QQQ",
            sort_order=1,
        ),
        WatchlistItemRecord(
            watchlist_id="watchlist-ai",
            instrument_id="equity:NASDAQ:NVDA",
            sort_order=3,
            notes="updated note",
        ),
    )
    watchlist_instruments = store.list_watchlist_instruments("watchlist-ai")
    assert tuple(item.instrument_id for item in watchlist_instruments) == (
        "etf:NYSEARCA:QQQ",
        "equity:NASDAQ:NVDA",
    )


@pytest.mark.unit
def test_append_tradability_evidence_returns_latest_by_provider(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="crypto:ETH",
            symbol="ETH",
            asset_class="crypto",
        )
    )

    store.append_tradability_evidence(
        InstrumentTradabilityEvidenceRecord(
            instrument_id="crypto:ETH",
            provider="ExampleBroker",
            status="unavailable",
            retrieved_at=datetime(2026, 5, 12, 12, 0, tzinfo=UTC),
            metadata={"reason": "fixture-old"},
        )
    )
    store.append_tradability_evidence(
        InstrumentTradabilityEvidenceRecord(
            instrument_id="crypto:ETH",
            provider="ExampleBroker",
            status="available",
            retrieved_at=_timestamp(),
            url="https://example.test/eth",
            metadata={"reason": "fixture-new"},
        )
    )

    assert store.get_latest_tradability_evidence(
        "crypto:ETH", "examplebroker"
    ) == InstrumentTradabilityEvidenceRecord(
        instrument_id="crypto:ETH",
        provider="ExampleBroker",
        status="available",
        retrieved_at=_timestamp(),
        url="https://example.test/eth",
        metadata={"reason": "fixture-new"},
    )


@pytest.mark.unit
def test_instrument_upsert_does_not_erase_appended_tradability_evidence(
    tmp_path: Path,
) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="crypto:SOL",
            symbol="SOL",
            asset_class="crypto",
        )
    )
    store.append_tradability_evidence(
        InstrumentTradabilityEvidenceRecord(
            instrument_id="crypto:SOL",
            provider="ExampleBroker",
            status="available",
            retrieved_at=_timestamp(),
        )
    )

    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="crypto:SOL",
            symbol="SOL",
            asset_class="crypto",
            aliases=("Solana",),
        )
    )

    assert store.get_latest_tradability_evidence(
        "crypto:SOL",
        "ExampleBroker",
    ) == InstrumentTradabilityEvidenceRecord(
        instrument_id="crypto:SOL",
        provider="ExampleBroker",
        status="available",
        retrieved_at=_timestamp(),
    )


@pytest.mark.unit
def test_provider_identifier_conflict_is_rejected(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    provider_id: JsonObject = {
        "provider": "ExampleMarket",
        "namespace": "ticker",
        "identifier": "TSLA",
    }
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="equity:NASDAQ:TSLA",
            symbol="TSLA",
            asset_class="stock",
            provider_ids=(provider_id,),
        )
    )

    with pytest.raises(ValueError, match="provider identifier is already assigned"):
        store.upsert_instrument(
            InstrumentRecord(
                instrument_id="equity:NYSE:TSLA",
                symbol="TSLA",
                asset_class="stock",
                provider_ids=(provider_id,),
            )
        )


@pytest.mark.unit
def test_run_scoped_lists_follow_candidate_links_without_tool_runs(tmp_path: Path) -> None:
    store = _research_store(tmp_path)
    store.initialize()
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id="etf:NYSEARCA:SPY",
            symbol="SPY",
            asset_class="etf",
        )
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-link-only",
            run_kind="candidate_graph",
            objective="link-only graph test",
            status="completed",
            started_at=_timestamp(),
        )
    )
    store.record_source_query(
        SourceQueryRecord(
            source_query_id="query-link-only",
            provider="example-search",
            query="SPY macro context",
            retrieved_at=_timestamp(),
        )
    )
    store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-link-only",
            artifact_type="evidence_bundle",
            path=Path("artifacts/link-only.json"),
            sha256="b" * 64,
            schema_version="bundle.v1",
            created_at=_timestamp(),
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-link-only",
            source_query_id="query-link-only",
            source_type="search_result",
            provider="example-search",
            retrieved_at=_timestamp(),
            claim="SPY has link-only evidence in this fixture.",
            artifact_id="artifact-link-only",
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-link-only",
            run_id="run-link-only",
            instrument_id="etf:NYSEARCA:SPY",
            prediction_horizon="intraday",
            prediction_type="directional",
            scenario="Candidate link graph should scope evidence.",
            status="low_confidence",
        )
    )
    store.link_candidate_evidence(
        CandidateEvidenceLinkRecord(
            candidate_id="candidate-link-only",
            evidence_id="evidence-link-only",
            relationship="supports",
            created_at=_timestamp(),
        )
    )
    store.link_candidate_artifact(
        CandidateArtifactLinkRecord(
            candidate_id="candidate-link-only",
            artifact_id="artifact-link-only",
            relationship="source_artifact",
            created_at=_timestamp(),
        )
    )

    assert store.list_tool_runs_for_run("run-link-only") == ()
    assert tuple(
        item.source_query_id for item in store.list_source_queries_for_run("run-link-only")
    ) == ("query-link-only",)
    assert tuple(item.artifact_id for item in store.list_artifacts_for_run("run-link-only")) == (
        "artifact-link-only",
    )
    assert tuple(item.evidence_id for item in store.list_evidence_for_run("run-link-only")) == (
        "evidence-link-only",
    )


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
    store.upsert_plan_milestone(
        PlanMilestoneRecord(
            milestone_id="milestone-schema",
            plan_id="plan-sqlite-layer",
            title="Schema initialized",
            status="completed",
            sort_order=1,
            details={"tables": 12},
        )
    )
    store.upsert_plan_acceptance_criterion(
        PlanAcceptanceCriterionRecord(
            criterion_id="criterion-storage-round-trip",
            plan_id="plan-sqlite-layer",
            description="Planning state round-trips through typed storage methods.",
            status="met",
            verification_command="python -m pytest tests/test_storage_sqlite.py",
        )
    )
    store.link_plan_artifact(
        PlanArtifactLinkRecord(
            plan_id="plan-sqlite-layer",
            artifact_id="artifact-in-research-db",
            relationship="supports",
        )
    )
    store.link_plan_commit(
        PlanCommitLinkRecord(
            plan_id="plan-sqlite-layer",
            commit_sha="a" * 40,
            relationship="implements",
        )
    )

    plan = store.get_plan_by_slug("sqlite-layer")
    decisions = store.list_plan_decisions("plan-sqlite-layer")
    progress = store.list_plan_progress("plan-sqlite-layer")
    milestones = store.list_plan_milestones("plan-sqlite-layer")
    criteria = store.list_plan_acceptance_criteria("plan-sqlite-layer")
    artifact_links = store.list_plan_artifact_links("plan-sqlite-layer")
    commit_links = store.list_plan_commit_links("plan-sqlite-layer")

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
    assert milestones == (
        PlanMilestoneRecord(
            milestone_id="milestone-schema",
            plan_id="plan-sqlite-layer",
            title="Schema initialized",
            status="completed",
            sort_order=1,
            details={"tables": 12},
        ),
    )
    assert criteria == (
        PlanAcceptanceCriterionRecord(
            criterion_id="criterion-storage-round-trip",
            plan_id="plan-sqlite-layer",
            description="Planning state round-trips through typed storage methods.",
            status="met",
            verification_command="python -m pytest tests/test_storage_sqlite.py",
        ),
    )
    assert artifact_links == (
        PlanArtifactLinkRecord(
            plan_id="plan-sqlite-layer",
            artifact_id="artifact-in-research-db",
            relationship="supports",
        ),
    )
    assert commit_links == (
        PlanCommitLinkRecord(
            plan_id="plan-sqlite-layer",
            commit_sha="a" * 40,
            relationship="implements",
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
                prediction_horizon="intraday",
                prediction_type="directional",
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
