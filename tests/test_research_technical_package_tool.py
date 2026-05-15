from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import pytest

from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    FreshnessStatus,
    ProviderStatus,
    WarningCode,
)
from nlp_stock_prediction.ml.timesfm.contracts import TimesFmForecastArtifact
from nlp_stock_prediction.orchestration.report_data_modes import (
    OFFLINE_FIXTURE_REPORT_DATA_MODE,
    report_data_mode_metadata,
)
from nlp_stock_prediction.orchestration.research_market_data import (
    MarketDataToolResult,
    ResearchMarketDataTool,
)
from nlp_stock_prediction.orchestration.research_technical_package import (
    RESEARCH_TECHNICAL_PACKAGE_SCHEMA_VERSION,
    TECHNICAL_PACKAGE_PREDICTION_POLICY,
    ResearchTechnicalPackageTool,
    load_research_technical_package_artifact,
)
from nlp_stock_prediction.providers.candlecharts import CandlechartsMarketDataProvider
from nlp_stock_prediction.providers.market import AlphaVantageMarketDataProvider
from nlp_stock_prediction.storage import ResearchRunRecord, SQLiteStore

RUN_ID = "research-technical-package-run"
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
            run_kind="research_tool_test",
            objective="exercise Research Stage technical package tool",
            status="running",
            started_at=NOW,
            metadata={
                "run_date": RUN_DATE.isoformat(),
                **report_data_mode_metadata(OFFLINE_FIXTURE_REPORT_DATA_MODE),
            },
        )
    )
    return store


def _audit_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports" / RUN_DATE.isoformat() / "audit"


def _market_data_result(tmp_path: Path, store: SQLiteStore) -> MarketDataToolResult:
    provider = CandlechartsMarketDataProvider(
        html=_html("public_ohlcv_tsla.html"),
        now=lambda: NOW,
    )
    return ResearchMarketDataTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_audit_dir(tmp_path),
        provider=provider,
        now=lambda: NOW,
    ).run(
        run_id=RUN_ID,
        run_date=RUN_DATE,
        symbol="TSLA",
        instrument_id="instrument:equity:us:tsla",
        source_url=FIXTURE_ROOT / "public_ohlcv_tsla.html",
    )


def _timesfm_artifact(
    *,
    status: Literal["usable", "weak", "unavailable"] = "usable",
    ticker: str = "TSLA",
) -> TimesFmForecastArtifact:
    return TimesFmForecastArtifact(
        status=status,
        ticker=ticker,
        model_id="timesfm-fixture",
        model_revision="test",
        dataset_hash="dataset-fixture-hash",
        input_hash="input-fixture-hash",
        forecast_timestamp=NOW,
        target_field="close",
        context_start=datetime(2026, 5, 8, tzinfo=UTC),
        context_end=NOW,
        forecast_horizon_sessions=2,
        point_forecast=(185.0, 186.0) if status != "unavailable" else (),
        expected_return=0.012,
        interval_width=0.04,
        directional_probability_proxy=0.58,
        uncertainty_score=0.35,
        warning_ids=("timesfm-fixture-warning",) if status == "unavailable" else (),
        metadata={"fixture": True},
    )


@pytest.mark.unit
def test_research_technical_package_computes_indicators_and_indexes_artifact(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    market_result = _market_data_result(tmp_path, store)

    result = ResearchTechnicalPackageTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_audit_dir(tmp_path),
        now=lambda: NOW,
    ).run(
        run_id=RUN_ID,
        symbol="TSLA",
        market_data=Path(market_result.artifact.path),
        timesfm_sidecar=_timesfm_artifact(),
        instrument_id="instrument:equity:us:tsla",
    )

    artifact_payload = result.artifact_payload
    assert artifact_payload.schema_version == RESEARCH_TECHNICAL_PACKAGE_SCHEMA_VERSION
    assert artifact_payload.status == "ok"
    assert artifact_payload.freshness_status == FreshnessStatus.FRESH
    assert artifact_payload.market_data_status == ProviderStatus.OK
    assert artifact_payload.market_data_artifact_id is not None
    assert artifact_payload.market_data_artifact_path is not None
    assert Path(artifact_payload.market_data_artifact_path) == Path(market_result.artifact.path)
    assert artifact_payload.market_data_artifact_sha256 == market_result.artifact.sha256
    assert artifact_payload.market_data_source_query_id == market_result.source_query_id
    assert artifact_payload.latest_usable_bar is not None
    assert artifact_payload.baseline_context.bar_count == 2
    assert artifact_payload.baseline_context.close_return_1d == pytest.approx(0.017956)
    assert {metric.name for metric in artifact_payload.deterministic_indicators} >= {
        "sma-20",
        "rsi-14",
        "relative-volume",
    }
    assert artifact_payload.timesfm_sidecar is not None
    assert artifact_payload.timesfm_applied_to_analysis is True
    assert artifact_payload.technical_analysis.ml_signal is not None
    assert artifact_payload.prediction_policy == TECHNICAL_PACKAGE_PREDICTION_POLICY
    assert "cannot create a reportable prediction" in artifact_payload.prediction_policy
    assert Path(result.artifact.path).exists()

    loaded = load_research_technical_package_artifact(Path(result.artifact.path))
    assert loaded.artifact_id == result.artifact.artifact_id
    assert loaded.technical_analysis.ticker == "TSLA"
    assert loaded.timesfm_applied_to_analysis is True

    tool_run = store.get_tool_run(result.tool_run_id)
    artifact = store.get_artifact(result.artifact.artifact_id)

    assert tool_run is not None
    assert tool_run.status == "successful"
    assert tool_run.inputs["market_data_artifact_id"] == artifact_payload.market_data_artifact_id
    assert artifact is not None
    assert artifact.artifact_type == "technical_package"
    assert artifact.schema_version == RESEARCH_TECHNICAL_PACKAGE_SCHEMA_VERSION
    assert artifact.metadata["timesfm_applied_to_analysis"] is True
    assert artifact in store.list_artifacts_for_run(RUN_ID)


@pytest.mark.unit
def test_research_technical_package_does_not_apply_timesfm_without_market_bars(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    market_result = ResearchMarketDataTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_audit_dir(tmp_path),
        provider=AlphaVantageMarketDataProvider(api_key=None, now=lambda: NOW),
        now=lambda: NOW,
    ).run(run_id=RUN_ID, run_date=RUN_DATE, symbol="TSLA")

    result = ResearchTechnicalPackageTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_audit_dir(tmp_path),
        now=lambda: NOW,
    ).run(
        run_id=RUN_ID,
        symbol="TSLA",
        market_data=market_result.provider_result,
        timesfm_sidecar=_timesfm_artifact(),
    )

    assert result.artifact_payload.status == "unavailable"
    assert result.artifact_payload.freshness_status == FreshnessStatus.MISSING
    assert result.artifact_payload.timesfm_sidecar is not None
    assert result.artifact_payload.timesfm_applied_to_analysis is False
    assert result.technical_analysis.ml_signal is None
    assert result.technical_analysis.signal == AnalysisSignal.UNKNOWN
    assert any(
        warning.code == WarningCode.UNSUPPORTED_CLAIM and "sole source" in warning.message
        for warning in result.artifact_payload.warnings
    )
    assert any(
        "deterministic OHLCV indicators were unavailable" in assumption
        for assumption in result.artifact_payload.assumptions
    )


@pytest.mark.unit
def test_research_technical_package_malformed_market_artifact_path_warns_not_crashes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    result = ResearchTechnicalPackageTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_audit_dir(tmp_path),
        now=lambda: NOW,
    ).run(
        run_id=RUN_ID,
        symbol="TSLA",
        market_data=tmp_path / "missing-market-data.json",
    )

    assert result.artifact_payload.status == "unavailable"
    assert result.artifact_payload.freshness_status == FreshnessStatus.MISSING
    assert result.artifact_payload.market_data_status is None
    assert result.artifact_payload.warnings
    assert any(
        "could not be loaded" in warning.message for warning in result.artifact_payload.warnings
    )
    assert Path(result.artifact.path).exists()
    tool_run = store.get_tool_run(result.tool_run_id)
    assert tool_run is not None
    assert tool_run.status == "partial"
    assert tool_run.warnings


@pytest.mark.unit
def test_research_technical_package_resolves_relative_market_artifact_path(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    market_result = _market_data_result(tmp_path, store)
    market_record = store.get_artifact(market_result.artifact.artifact_id)
    assert market_record is not None

    result = ResearchTechnicalPackageTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_audit_dir(tmp_path),
        now=lambda: NOW,
    ).run(
        run_id=RUN_ID,
        symbol="TSLA",
        market_data=Path(market_record.path),
    )

    assert result.artifact_payload.status == "ok"
    assert result.artifact_payload.baseline_context.bar_count == 2
    artifact_path = result.artifact_payload.market_data_artifact_path
    assert artifact_path is not None
    assert Path(artifact_path) == tmp_path / market_record.path


@pytest.mark.unit
def test_research_technical_package_does_not_apply_unavailable_timesfm_sidecar(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    market_result = _market_data_result(tmp_path, store)

    result = ResearchTechnicalPackageTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_audit_dir(tmp_path),
        now=lambda: NOW,
    ).run(
        run_id=RUN_ID,
        symbol="TSLA",
        market_data=market_result.provider_result,
        timesfm_sidecar=_timesfm_artifact(status="unavailable"),
    )

    assert result.artifact_payload.status == "warning"
    assert result.artifact_payload.timesfm_sidecar is None
    assert result.artifact_payload.timesfm_applied_to_analysis is False
    assert result.technical_analysis.ml_signal is None
    assert any(
        warning.code == WarningCode.NO_DATA and "TimesFM sidecar was unavailable" in warning.message
        for warning in result.artifact_payload.warnings
    )


@pytest.mark.unit
def test_research_technical_package_rejects_timesfm_sidecar_ticker_mismatch(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    market_result = _market_data_result(tmp_path, store)

    result = ResearchTechnicalPackageTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=_audit_dir(tmp_path),
        now=lambda: NOW,
    ).run(
        run_id=RUN_ID,
        symbol="TSLA",
        market_data=market_result.provider_result,
        timesfm_sidecar=_timesfm_artifact(ticker="NVDA"),
    )

    assert result.artifact_payload.status == "warning"
    assert result.artifact_payload.timesfm_sidecar is None
    assert result.artifact_payload.timesfm_applied_to_analysis is False
    assert result.technical_analysis.ml_signal is None
    assert any(
        warning.code == WarningCode.SCHEMA_MISMATCH and "NVDA" in warning.message
        for warning in result.artifact_payload.warnings
    )
