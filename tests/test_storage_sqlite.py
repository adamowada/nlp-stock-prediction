from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts import JsonObject
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    CandidateArtifactLinkRecord,
    CandidateEvidenceLinkRecord,
    EvidenceRecord,
    InstrumentRecord,
    InstrumentTradabilityEvidenceRecord,
    PlanAcceptanceCriterionRecord,
    PlanArtifactLinkRecord,
    PlanCommitLinkRecord,
    PlanDecisionRecord,
    PlanMilestoneRecord,
    PlanningSQLiteStore,
    PlanProgressRecord,
    PlanRecord,
    PredictionCandidateRecord,
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
        "research_runs",
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
                prediction_horizon="5d",
                prediction_type="direction",
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
                provider_ids_json TEXT NOT NULL DEFAULT '[]',
                related_instruments_json TEXT NOT NULL DEFAULT '[]',
                tradability_evidence_json TEXT NOT NULL DEFAULT '[]',
                data_availability_json TEXT NOT NULL DEFAULT '[]',
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
        "candidate_artifact_links",
        "candidate_evidence_links",
        "instrument_aliases",
        "instrument_data_availability",
        "instrument_provider_ids",
        "instrument_related_instruments",
        "instrument_tradability_evidence",
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
            prediction_horizon="1d",
            prediction_type="direction",
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
