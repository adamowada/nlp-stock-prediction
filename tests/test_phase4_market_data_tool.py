from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import NoReturn

import pytest

from nlp_stock_prediction.contracts import (
    CredentialState,
    FreshnessStatus,
    MarketDataProvider,
    MarketDataRequest,
    MarketSnapshot,
    PriceBar,
    ProviderResult,
    ProviderStatus,
    RetrievalMethod,
    WarningCode,
)
from nlp_stock_prediction.orchestration.phase4_market_data import (
    PHASE4_MARKET_DATA_SCHEMA_VERSION,
    Phase4MarketDataTool,
    load_phase4_market_data_artifact,
)
from nlp_stock_prediction.providers._base import provider_result
from nlp_stock_prediction.providers.candlecharts import CandlechartsMarketDataProvider
from nlp_stock_prediction.providers.market import AlphaVantageMarketDataProvider
from nlp_stock_prediction.storage import ArtifactRecord, ResearchRunRecord, SQLiteStore

RUN_ID = "phase4-market-data-run"
RUN_DATE = date(2026, 5, 11)
NOW = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "raw" / "candlecharts"


def _html(name: str) -> str:
    return FIXTURE_ROOT.joinpath(name).read_text(encoding="utf-8")


def _store(tmp_path: Path, run_id: str = RUN_ID) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "prediction-research.sqlite3")
    store.initialize()
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=run_id,
            run_kind="phase4_tool_test",
            objective="exercise Phase 4 market data tool",
            status="running",
            started_at=NOW,
            metadata={"run_date": RUN_DATE.isoformat()},
        )
    )
    return store


def _tool(
    *,
    tmp_path: Path,
    store: SQLiteStore,
    provider: MarketDataProvider,
) -> Phase4MarketDataTool:
    return Phase4MarketDataTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_DATE.isoformat() / "audit",
        provider=provider,
        now=lambda: NOW,
    )


class _StaticMarketDataProvider:
    provider_name = "alpha-vantage-market-data"

    def __init__(self, snapshot: MarketSnapshot) -> None:
        self._snapshot = snapshot

    def fetch_daily_candles(self, request: MarketDataRequest) -> ProviderResult[MarketSnapshot]:
        return provider_result(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=NOW,
            credential_state=CredentialState.NOT_REQUIRED,
            data=self._snapshot,
            raw_snapshot_id="raw-static-market-data",
        )

    def health(self) -> NoReturn:
        raise NotImplementedError


def _bar(
    *,
    ticker: str = "TSLA",
    timestamp: date = RUN_DATE,
    close: str = "184.25",
) -> PriceBar:
    close_decimal = Decimal(close)
    return PriceBar(
        ticker=ticker,
        timestamp=timestamp,
        open=Decimal("181.00"),
        high=max(Decimal("185.00"), close_decimal),
        low=Decimal("180.00"),
        close=close_decimal,
        volume=123_456,
    )


class _FailingArtifactStore(SQLiteStore):
    def record_artifact(self, record: ArtifactRecord) -> None:
        del record
        raise RuntimeError("artifact insert failed")


@pytest.mark.unit
def test_phase4_market_data_tool_writes_artifact_and_sqlite_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    fixture_path = FIXTURE_ROOT / "public_ohlcv_tsla.html"
    provider = CandlechartsMarketDataProvider(html=_html(fixture_path.name), now=lambda: NOW)

    result = _tool(tmp_path=tmp_path, store=store, provider=provider).run(
        run_id=RUN_ID,
        run_date=RUN_DATE,
        symbol="TSLA",
        instrument_id="instrument:equity:us:tsla",
        source_url=fixture_path,
    )

    assert result.provider_result.status == ProviderStatus.OK
    assert result.artifact_payload.schema_version == PHASE4_MARKET_DATA_SCHEMA_VERSION
    assert result.artifact_payload.status == ProviderStatus.OK
    assert result.artifact_payload.freshness_status == FreshnessStatus.FRESH
    assert result.latest_usable_bar is not None
    assert result.latest_usable_bar.timestamp == RUN_DATE
    assert result.artifact_payload.bar_count == 2
    assert result.artifact_payload.provenance.source_query_id == result.source_query_id
    assert result.artifact_payload.provenance.raw_snapshot_id is not None
    assert Path(result.artifact.path).exists()

    loaded = load_phase4_market_data_artifact(Path(result.artifact.path))
    assert loaded.artifact_id == result.artifact.artifact_id
    assert loaded.provider_result.status == ProviderStatus.OK
    assert loaded.bars[0].ticker == "TSLA"

    tool_run = store.get_tool_run(result.tool_run_id)
    source_query = store.get_source_query(result.source_query_id)
    artifact = store.get_artifact(result.artifact.artifact_id)

    assert tool_run is not None
    assert tool_run.status == "successful"
    assert tool_run.inputs["symbol"] == "TSLA"
    assert source_query is not None
    assert source_query.provider == "candlecharts-market-data"
    assert source_query.url == str(fixture_path)
    assert source_query.metadata["status"] == "ok"
    assert artifact is not None
    assert artifact.artifact_type == "market_data"
    assert artifact.schema_version == PHASE4_MARKET_DATA_SCHEMA_VERSION
    assert artifact.produced_by == "phase4_market_data"
    assert artifact.record_count == 2
    assert artifact.metadata["latest_usable_bar"] == RUN_DATE.isoformat()
    assert store.list_source_queries_for_run(RUN_ID) == (source_query,)
    assert store.list_artifacts_for_run(RUN_ID) == (artifact,)


@pytest.mark.unit
def test_phase4_market_data_rolls_back_file_and_rows_when_artifact_index_fails(
    tmp_path: Path,
) -> None:
    store = _FailingArtifactStore(tmp_path / "prediction-research.sqlite3")
    store.initialize()
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=RUN_ID,
            run_kind="phase4_tool_test",
            objective="exercise Phase 4 market data rollback",
            status="running",
            started_at=NOW,
            metadata={"run_date": RUN_DATE.isoformat()},
        )
    )
    artifact_dir = tmp_path / "reports" / RUN_DATE.isoformat() / "audit"
    provider = CandlechartsMarketDataProvider(
        html=_html("public_ohlcv_tsla.html"),
        now=lambda: NOW,
    )

    with pytest.raises(RuntimeError, match="artifact insert failed"):
        Phase4MarketDataTool(
            store=store,
            repo_root=tmp_path,
            artifact_dir=artifact_dir,
            provider=provider,
            now=lambda: NOW,
        ).run(
            run_id=RUN_ID,
            run_date=RUN_DATE,
            symbol="TSLA",
            instrument_id="instrument:equity:us:tsla",
        )

    tool_runs = store.list_tool_runs_for_run(RUN_ID)
    assert len(tool_runs) == 1
    assert tool_runs[0].status == "failed"
    assert tool_runs[0].error_message == "artifact insert failed"
    assert store.list_source_queries_for_run(RUN_ID) == ()
    assert store.list_artifacts_for_run(RUN_ID) == ()
    assert not tuple(artifact_dir.rglob("*.json")) if artifact_dir.exists() else True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider_factory", "expected_status", "expected_freshness", "expected_warning_code"),
    (
        (
            lambda: CandlechartsMarketDataProvider(
                html=_html("stale_ohlcv_tsla.html"),
                now=lambda: NOW,
                stale_after_days=5,
            ),
            ProviderStatus.STALE,
            FreshnessStatus.STALE,
            WarningCode.STALE_DATA,
        ),
        (
            lambda: CandlechartsMarketDataProvider(
                html=_html("malformed_ohlcv_tsla.html"),
                now=lambda: NOW,
            ),
            ProviderStatus.MALFORMED,
            FreshnessStatus.UNKNOWN,
            WarningCode.MALFORMED_RESPONSE,
        ),
        (
            lambda: CandlechartsMarketDataProvider(now=lambda: NOW),
            ProviderStatus.EMPTY,
            FreshnessStatus.MISSING,
            WarningCode.NO_DATA,
        ),
        (
            lambda: AlphaVantageMarketDataProvider(api_key=None, now=lambda: NOW),
            ProviderStatus.UNCONFIGURED,
            FreshnessStatus.MISSING,
            WarningCode.MISSING_CREDENTIALS,
        ),
    ),
)
def test_phase4_market_data_tool_records_warning_artifacts_for_provider_problems(
    tmp_path: Path,
    provider_factory: Callable[[], MarketDataProvider],
    expected_status: ProviderStatus,
    expected_freshness: FreshnessStatus,
    expected_warning_code: WarningCode,
) -> None:
    run_id = f"{RUN_ID}-{expected_status.value}"
    store = _store(tmp_path, run_id=run_id)

    result = _tool(tmp_path=tmp_path, store=store, provider=provider_factory()).run(
        run_id=run_id,
        run_date=RUN_DATE,
        symbol="TSLA",
    )

    assert result.provider_result.status == expected_status
    assert result.artifact_payload.status == expected_status
    assert result.artifact_payload.freshness_status == expected_freshness
    assert result.artifact_payload.warnings
    assert result.artifact_payload.warnings[0].code == expected_warning_code
    assert Path(result.artifact.path).exists()

    tool_run = store.get_tool_run(result.tool_run_id)
    artifact = store.get_artifact(result.artifact.artifact_id)
    source_query = store.get_source_query(result.source_query_id)

    assert tool_run is not None
    expected_tool_status = (
        "partial"
        if expected_status == ProviderStatus.STALE
        else "failed"
        if expected_status in {ProviderStatus.MALFORMED, ProviderStatus.UNCONFIGURED}
        else "empty"
    )
    assert tool_run.status == expected_tool_status
    assert tool_run.warnings
    assert artifact is not None
    warning_count = artifact.metadata["warning_count"]
    assert isinstance(warning_count, int)
    assert warning_count >= 1
    assert source_query is not None
    assert source_query.metadata["status"] == expected_status.value


@pytest.mark.unit
def test_phase4_market_data_rejects_snapshot_ticker_mismatch(tmp_path: Path) -> None:
    store = _store(tmp_path)
    provider = _StaticMarketDataProvider(MarketSnapshot(ticker="MSFT", bars=(_bar(ticker="MSFT"),)))

    result = _tool(tmp_path=tmp_path, store=store, provider=provider).run(
        run_id=RUN_ID,
        run_date=RUN_DATE,
        symbol="TSLA",
    )

    assert result.provider_result.status == ProviderStatus.EMPTY
    assert result.artifact_payload.bar_count == 0
    assert result.artifact_payload.latest_usable_bar is None
    assert result.artifact_payload.metadata["rejected_snapshot"] is True
    assert result.artifact_payload.metadata["returned_snapshot_ticker"] == "MSFT"
    assert {warning.code for warning in result.artifact_payload.warnings} >= {
        WarningCode.SCHEMA_MISMATCH,
        WarningCode.NO_DATA,
    }


@pytest.mark.unit
def test_phase4_market_data_filters_mismatched_and_future_bars(tmp_path: Path) -> None:
    store = _store(tmp_path)
    provider = _StaticMarketDataProvider(
        MarketSnapshot(
            ticker="TSLA",
            bars=(
                _bar(timestamp=RUN_DATE, close="184.25"),
                _bar(ticker="MSFT", timestamp=RUN_DATE, close="420.00"),
                _bar(timestamp=date(2026, 5, 12), close="190.00"),
            ),
        )
    )

    result = _tool(tmp_path=tmp_path, store=store, provider=provider).run(
        run_id=RUN_ID,
        run_date=RUN_DATE,
        symbol="TSLA",
    )

    assert result.provider_result.status == ProviderStatus.PARTIAL
    assert result.artifact_payload.bar_count == 1
    assert result.latest_usable_bar is not None
    assert result.latest_usable_bar.timestamp == RUN_DATE
    assert result.artifact_payload.metadata["rejected_bar_count"] == 1
    assert result.artifact_payload.metadata["excluded_future_bar_count"] == 1
    future_bars = result.artifact_payload.metadata["excluded_future_bars"]
    assert isinstance(future_bars, list)
    first_future_bar = future_bars[0]
    assert isinstance(first_future_bar, dict)
    assert first_future_bar["timestamp"] == "2026-05-12"
    assert {warning.code for warning in result.artifact_payload.warnings} >= {
        WarningCode.SCHEMA_MISMATCH,
        WarningCode.PARTIAL_DATA,
    }


@pytest.mark.unit
def test_phase4_market_data_non_fixture_provider_does_not_default_to_fixture_provenance(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    provider = _StaticMarketDataProvider(MarketSnapshot(ticker="TSLA", bars=(_bar(),)))

    result = _tool(tmp_path=tmp_path, store=store, provider=provider).run(
        run_id=RUN_ID,
        run_date=RUN_DATE,
        symbol="TSLA",
    )

    assert result.artifact_payload.provenance.retrieval_method == RetrievalMethod.OFFICIAL_API
    assert result.artifact_payload.provenance.url is None
    source_query = store.get_source_query(result.source_query_id)
    assert source_query is not None
    assert source_query.url is None
