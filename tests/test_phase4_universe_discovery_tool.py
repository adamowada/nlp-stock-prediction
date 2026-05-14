from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    AssetClass,
    InstrumentQuery,
    InstrumentUniverseRequest,
    JsonObject,
    Watchlist,
    WatchlistEntry,
)
from nlp_stock_prediction.orchestration.artifacts import ArtifactWriter
from nlp_stock_prediction.orchestration.context import RunContext
from nlp_stock_prediction.orchestration.phase4_universe_discovery import (
    PHASE4_FIXTURE_PROVIDER,
    PHASE4_TOOL_NAME,
    PHASE4_UNIVERSE_SCHEMA_VERSION,
    Phase4UniverseDiscoveryTool,
)
from nlp_stock_prediction.storage import ResearchRunRecord, SQLiteStore

pytestmark = pytest.mark.unit


def _now() -> datetime:
    return datetime(2026, 5, 13, 21, 0, tzinfo=UTC)


def _context(repo_root: Path) -> RunContext:
    output_dir = repo_root / "reports"
    report_dir = output_dir / "2026-05-13"
    audit_dir = report_dir / "audit"
    return RunContext(
        run_id="run-phase4-universe",
        run_date=date(2026, 5, 13),
        generated_at=_now(),
        timezone="UTC",
        output_dir=output_dir,
        report_dir=report_dir,
        audit_dir=audit_dir,
        command_args={"offline": True},
        artifact_writer=ArtifactWriter(
            base_dir=audit_dir,
            created_at=_now(),
            produced_by="phase4-test",
        ),
    )


def _store(repo_root: Path) -> SQLiteStore:
    return SQLiteStore(repo_root / "data" / "prediction-research.sqlite3")


def _seed_run(store: SQLiteStore, context: RunContext) -> None:
    store.initialize()
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=context.run_id,
            run_kind="phase4_universe_discovery_test",
            objective="Fixture-backed Phase 4 universe discovery test",
            status="running",
            started_at=context.generated_at,
        )
    )


def test_phase4_fixture_discovery_persists_run_graph_and_indexed_instruments(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    context = _context(repo_root)
    store = _store(repo_root)
    _seed_run(store, context)
    request = InstrumentUniverseRequest(
        request_id="phase4-fixture-universe",
        as_of=_now(),
        queries=(
            InstrumentQuery(query="TSLA"),
            InstrumentQuery(query="AI"),
            InstrumentQuery(query="MISSING"),
            InstrumentQuery(query="PRIVATE:SPACEX", asset_class=AssetClass.UNKNOWN),
        ),
    )

    result = Phase4UniverseDiscoveryTool().run(
        request=request,
        context=context,
        store=store,
        repo_root=repo_root,
    )

    tool_run = store.get_tool_run(result.tool_run_id)
    artifact = store.get_artifact(result.artifact_id)
    source_queries = store.list_source_queries_for_run(context.run_id)
    tsla = store.get_instrument("instrument:equity:us:tsla")
    provider_match = store.find_instrument_by_provider_id(
        PHASE4_FIXTURE_PROVIDER,
        "fixture-symbol",
        "TSLA",
    )

    assert tool_run is not None
    assert tool_run.tool_name == PHASE4_TOOL_NAME
    assert tool_run.run_id == context.run_id
    tool_request = cast(JsonObject, tool_run.inputs["request"])
    assert tool_request["request_id"] == request.request_id
    assert any("AI" in warning for warning in tool_run.warnings)
    assert any("MISSING" in warning for warning in tool_run.warnings)
    assert any("unknown asset class" in warning for warning in tool_run.warnings)

    assert artifact is not None
    assert artifact.tool_run_id == result.tool_run_id
    assert artifact.artifact_type == "instrument_universe"
    assert artifact.schema_version == PHASE4_UNIVERSE_SCHEMA_VERSION
    assert artifact.metadata["instrument_ids"] == ["instrument:equity:us:tsla"]
    artifact_payload = json.loads((repo_root / artifact.path).read_text(encoding="utf-8"))
    assert artifact_payload["schema_version"] == PHASE4_UNIVERSE_SCHEMA_VERSION
    assert artifact_payload["run_id"] == context.run_id
    assert artifact_payload["universe"]["instrument_ids"] == ["instrument:equity:us:tsla"]

    assert len(source_queries) == 4
    assert {record.query for record in source_queries} == {
        "AI",
        "MISSING",
        "PRIVATE:SPACEX",
        "TSLA",
    }
    ai_source_query = next(record for record in source_queries if record.query == "AI")
    assert ai_source_query.provider == PHASE4_FIXTURE_PROVIDER
    assert ai_source_query.tool_run_id == result.tool_run_id
    assert ai_source_query.metadata["matched_instrument_ids"] == [
        "instrument:equity:us:ai",
        "instrument:crypto:ai-usd",
    ]

    assert tsla is not None
    assert tsla.aliases == ("TESLA", "TSLA.US")
    assert tsla.metadata["phase4_universe_discovery"] is True
    assert tuple(
        item.instrument_id for item in store.find_instruments_by_symbol_or_alias("tesla")
    ) == ("instrument:equity:us:tsla",)
    assert provider_match is not None
    assert provider_match.instrument_id == "instrument:equity:us:tsla"
    assert result.instrument_records[0].instrument_id == "instrument:equity:us:tsla"


def test_phase4_fixture_discovery_keeps_ambiguous_unavailable_and_unsupported_explicit(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    context = _context(repo_root)
    store = _store(repo_root)
    _seed_run(store, context)
    request = InstrumentUniverseRequest(
        request_id="phase4-resolution-outcomes",
        as_of=_now(),
        queries=(
            InstrumentQuery(query="AI"),
            InstrumentQuery(query="MISSING"),
            InstrumentQuery(query="PRIVATE:SPACEX", asset_class=AssetClass.UNKNOWN),
        ),
    )

    result = Phase4UniverseDiscoveryTool().run(
        request=request,
        context=context,
        store=store,
        repo_root=repo_root,
    )
    resolutions = {resolution.query: resolution for resolution in result.universe.resolutions}

    assert result.universe.instrument_ids == ()
    assert resolutions["AI"].status.value == "ambiguous"
    assert resolutions["AI"].selected_instrument_id is None
    assert tuple(instrument.instrument_id for instrument in resolutions["AI"].matches) == (
        "instrument:crypto:ai-usd",
        "instrument:equity:us:ai",
    )
    assert resolutions["MISSING"].status.value == "unavailable"
    assert resolutions["MISSING"].selected_instrument_id is None
    assert resolutions["MISSING"].matches == ()
    assert resolutions["PRIVATE:SPACEX"].status.value == "unsupported"
    assert resolutions["PRIVATE:SPACEX"].selected_instrument_id is None
    assert resolutions["PRIVATE:SPACEX"].matches == ()
    assert result.instrument_records == ()


def test_phase4_fixture_discovery_resolves_provider_hinted_fixture_identifier(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    context = _context(repo_root)
    store = _store(repo_root)
    _seed_run(store, context)
    request = InstrumentUniverseRequest(
        request_id="phase4-provider-hinted-fixture",
        as_of=_now(),
        queries=(
            InstrumentQuery(
                query="Tesla Inc",
                asset_class=AssetClass.STOCK,
                provider=PHASE4_FIXTURE_PROVIDER,
                provider_namespace="fixture-symbol",
                provider_identifier="TSLA",
            ),
        ),
    )

    result = Phase4UniverseDiscoveryTool().run(
        request=request,
        context=context,
        store=store,
        repo_root=repo_root,
    )

    assert result.universe.instrument_ids == ("instrument:equity:us:tsla",)
    assert result.universe.resolutions[0].status.value == "resolved"
    source_query = store.list_source_queries_for_run(context.run_id)[0]
    assert source_query.metadata["matched_instrument_ids"] == ["instrument:equity:us:tsla"]


def test_phase4_fixture_discovery_requested_id_does_not_override_unrelated_query(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    context = _context(repo_root)
    store = _store(repo_root)
    _seed_run(store, context)
    request = InstrumentUniverseRequest(
        request_id="phase4-requested-id-mismatch",
        as_of=_now(),
        watchlists=(
            Watchlist(
                watchlist_id="phase4-watchlist",
                name="Phase 4 Watchlist",
                entries=(
                    WatchlistEntry(
                        query=InstrumentQuery(query="SPY", asset_class=AssetClass.ETF),
                        requested_instrument_id="instrument:equity:us:tsla",
                    ),
                ),
            ),
        ),
    )

    result = Phase4UniverseDiscoveryTool().run(
        request=request,
        context=context,
        store=store,
        repo_root=repo_root,
    )

    assert result.universe.instrument_ids == ("instrument:etf:us:spy",)
    assert result.universe.resolutions[0].selected_instrument_id == "instrument:etf:us:spy"
    source_query = store.list_source_queries_for_run(context.run_id)[0]
    assert source_query.metadata["requested_instrument_id"] == "instrument:equity:us:tsla"
    assert source_query.metadata["matched_instrument_ids"] == ["instrument:etf:us:spy"]
