from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.cli import build_parser, build_research_config
from nlp_stock_prediction.contracts import (
    EvidenceRequest,
    FundamentalsProvider,
    MacroProvider,
    MarketDataRequest,
    MarketSnapshot,
    NewsProvider,
    ProviderHealth,
    ProviderResult,
    RedditProvider,
    RunConfig,
    SourceEvidence,
    XProvider,
)
from nlp_stock_prediction.orchestration import (
    LIVE_REPORT_DATA_MODE,
    OFFLINE_FIXTURE_REPORT_DATA_MODE,
    Phase4Service,
    Phase4ToolExecutionError,
)
from nlp_stock_prediction.orchestration.phase4_live_providers import (
    Phase4LiveProviderFactoryProtocol,
)
from nlp_stock_prediction.orchestration.phase4_universe_discovery import (
    Phase4LiveSymbolUniverseProvider,
    UniverseDiscoveryProvider,
)
from nlp_stock_prediction.orchestration.report_data_modes import (
    ReportInputProvenance,
    find_non_live_report_input_violations,
    report_data_mode_metadata,
)
from nlp_stock_prediction.pipeline import (
    LIVE_ORCHESTRATION_DISABLED_MESSAGE,
    generate_daily_report,
)
from nlp_stock_prediction.providers._base import missing_credentials_result, no_data_result
from nlp_stock_prediction.storage import InstrumentRecord, PredictionCandidateRecord, ToolRunRecord

RUN_DATE = date(2026, 5, 13)
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_pipeline_refuses_live_config_without_fixture_or_dummy_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="refusing to use fixture or dummy fallback"):
        generate_daily_report(
            RunConfig(
                run_date=RUN_DATE,
                output_dir=tmp_path / "reports",
                offline=False,
            )
        )

    assert "fixture or dummy fallback" in LIVE_ORCHESTRATION_DISABLED_MESSAGE


@pytest.mark.unit
def test_cli_research_live_mode_is_explicit_and_machine_marked(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "research",
            "--date",
            RUN_DATE.isoformat(),
            "--output",
            str(tmp_path / "reports"),
            "--symbol",
            "TSLA",
            "--live",
        ]
    )

    config = build_research_config(args)

    assert config.offline is False
    assert config.source_mode == "live"
    assert config.live_providers is True


@pytest.mark.unit
def test_report_input_provenance_reads_stamped_modes() -> None:
    live_provenance = ReportInputProvenance.from_metadata(
        record_type="artifact",
        record_id="artifact-live",
        metadata=report_data_mode_metadata(LIVE_REPORT_DATA_MODE),
    )

    assert live_provenance.observed_modes == (
        LIVE_REPORT_DATA_MODE,
        LIVE_REPORT_DATA_MODE,
        LIVE_REPORT_DATA_MODE,
    )
    assert live_provenance.non_live_violations() == ()

    fixture_provenance = ReportInputProvenance.from_metadata(
        record_type="artifact",
        record_id="artifact-fixture",
        metadata=report_data_mode_metadata(OFFLINE_FIXTURE_REPORT_DATA_MODE),
    )

    assert fixture_provenance.includes_non_live_mode is True
    assert {violation.field for violation in fixture_provenance.non_live_violations()} == {
        "report_data_mode",
        "provider_mode",
        "input_data_mode",
    }


@pytest.mark.integration
def test_offline_phase4_report_inputs_are_machine_marked(tmp_path: Path) -> None:
    bundle = generate_daily_report(
        RunConfig(
            run_date="2026-05-11",
            output_dir=tmp_path / "reports",
            offline=True,
        )
    )

    payload = json.loads(bundle.json_path.read_text(encoding="utf-8"))
    audit_payload = json.loads(bundle.audit_manifest_path.read_text(encoding="utf-8"))

    assert payload["command_args"]["report_data_mode"] == OFFLINE_FIXTURE_REPORT_DATA_MODE
    assert payload["audit_manifest"]["command_args"]["report_data_mode"] == (
        OFFLINE_FIXTURE_REPORT_DATA_MODE
    )
    assert audit_payload["command_args"]["report_data_mode"] == OFFLINE_FIXTURE_REPORT_DATA_MODE
    tool_records = cast(Any, bundle.tool_records)
    assert any(
        record.inputs.get("report_data_mode") == OFFLINE_FIXTURE_REPORT_DATA_MODE
        for record in tool_records
    )


@pytest.mark.integration
def test_live_phase4_report_without_inputs_renders_insufficient_evidence(
    tmp_path: Path,
) -> None:
    service = Phase4Service(repo_root=tmp_path)
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/live-empty",
        symbol="TSLA",
        report_data_mode=LIVE_REPORT_DATA_MODE,
    )

    rendered = service.render_prediction_report(run_id=str(started["run_id"]), symbol="TSLA")

    payload = json.loads(Path(str(rendered["json_path"])).read_text(encoding="utf-8"))
    assert rendered["report_data_mode"] == LIVE_REPORT_DATA_MODE
    assert payload["command_args"]["report_data_mode"] == LIVE_REPORT_DATA_MODE
    assert payload["prediction_candidates"] == []
    assert payload["insufficient_evidence"]["provider_names"] == ["live-providers"]
    instruments = cast(list[dict[str, Any]], payload["instruments"])
    assert instruments[0]["instrument_id"].startswith("instrument:live:unknown:")
    assert instruments[0]["provider_ids"][0]["provider"] == "live-symbol-directory"


@pytest.mark.integration
def test_live_phase4_report_rejects_fixture_or_dummy_leakage(tmp_path: Path) -> None:
    service = Phase4Service(repo_root=tmp_path)
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/live-leakage",
        symbol="TSLA",
        report_data_mode=LIVE_REPORT_DATA_MODE,
    )
    run_id = str(started["run_id"])
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-fixture-leakage",
            run_id=run_id,
            tool_name="phase4_universe_discovery",
            tool_version="phase4.fixture.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "TSLA", "report_data_mode": OFFLINE_FIXTURE_REPORT_DATA_MODE},
        )
    )

    with pytest.raises(Phase4ToolExecutionError) as exc_info:
        service.render_prediction_report(run_id=run_id, symbol="TSLA")

    assert isinstance(exc_info.value.original_error, ValueError)
    assert "live report assembly cannot use non-live report inputs" in str(
        exc_info.value.original_error
    )


@pytest.mark.integration
def test_live_phase4_flow_uses_live_providers_and_degrades_to_insufficient_evidence(
    tmp_path: Path,
) -> None:
    service = Phase4Service(
        repo_root=tmp_path,
        live_provider_factory=_UnitLiveProviderFactory(),
    )

    result = service.run_live_phase4_flow(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/live-no-evidence",
        symbol="TSLA",
    )
    run_id = str(result["run_id"])
    run = service.store.get_research_run(run_id)
    assert run is not None

    violations = find_non_live_report_input_violations(store=service.store, run=run)
    assert violations == ()

    report_payload = cast(dict[str, object], result["report"])
    json_path = Path(str(report_payload["json_path"]))
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["command_args"]["report_data_mode"] == LIVE_REPORT_DATA_MODE
    assert payload["prediction_candidates"] == []
    assert payload["insufficient_evidence"]["provider_names"][0] == "live-providers"
    assert "tool:phase4_news_catalyst" in payload["insufficient_evidence"]["provider_names"]
    assert any(
        health["provider_name"] == "tool:phase4_news_catalyst" and health["status"] == "failed"
        for health in payload["provider_health"]
    )
    assert any(
        run.tool_name == "phase4_prediction_candidate_synthesis" and run.status == "empty"
        for run in service.store.list_tool_runs_for_run(run_id)
    )


@pytest.mark.integration
def test_live_phase4_flow_prefers_live_instrument_identity_when_fixture_records_exist(
    tmp_path: Path,
) -> None:
    service = Phase4Service(
        repo_root=tmp_path,
        fixture_root=REPO_ROOT,
        database_path=Path("data") / "prediction-research.sqlite3",
        live_provider_factory=_UnitLiveProviderFactory(),
    )
    service.run_offline_phase4_flow(
        run_date="2026-05-12",
        output_dir="reports/offline-before-live",
        symbol="TSLA",
    )

    result = service.run_live_phase4_flow(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/live-after-offline",
        symbol="TSLA",
    )

    report_payload = cast(dict[str, object], result["report"])
    payload = json.loads(Path(str(report_payload["json_path"])).read_text(encoding="utf-8"))
    instruments = cast(list[dict[str, Any]], payload["instruments"])
    assert instruments[0]["instrument_id"].startswith("instrument:live:")
    assert instruments[0]["instrument_id"] != "instrument:codex:TSLA"
    run = service.store.get_research_run(str(result["run_id"]))
    assert run is not None
    assert find_non_live_report_input_violations(store=service.store, run=run) == ()


@pytest.mark.integration
def test_live_boundary_scans_candidate_instrument_provider_records(tmp_path: Path) -> None:
    service = Phase4Service(repo_root=tmp_path)
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/live-fixture-instrument-record",
        symbol="TSLA",
        report_data_mode=LIVE_REPORT_DATA_MODE,
    )
    run_id = str(started["run_id"])
    service.store.upsert_instrument(
        InstrumentRecord(
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            asset_class="stock",
            name="Tesla Inc.",
            provider_ids=(
                {
                    "provider": "phase4-fixture-universe",
                    "id_type": "fixture-symbol",
                    "identifier": "TSLA",
                },
            ),
        )
    )
    service.store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-live-fixture-instrument",
            run_id=run_id,
            instrument_id="instrument:equity:us:tsla",
            prediction_horizon="swing",
            prediction_type="directional",
            scenario="TSLA live candidate must not reuse fixture instrument provenance.",
            status="insufficient_evidence",
            confidence=0.18,
            direction="mixed",
            baseline={"summary": "No directional edge is assumed without source-backed evidence."},
            uncertainty="Live provider evidence is unavailable.",
            metadata={"symbol": "TSLA"},
        )
    )
    run = service.store.get_research_run(run_id)
    assert run is not None

    violations = find_non_live_report_input_violations(store=service.store, run=run)

    assert any(
        violation.record_type == "instrument_provider_id"
        and violation.field == "provider"
        and violation.value == "phase4-fixture-universe"
        for violation in violations
    )


class _UnitLiveProviderFactory(Phase4LiveProviderFactoryProtocol):
    def universe_provider(self) -> UniverseDiscoveryProvider:
        return Phase4LiveSymbolUniverseProvider()

    def market_data_provider(self, symbol: str) -> _UnitLiveMarketDataProvider:
        del symbol
        return _UnitLiveMarketDataProvider()

    def market_data_source_query_url(self, symbol: str) -> str:
        return f"https://live.example.invalid/market-data?symbol={symbol.strip().upper()}"

    def reddit_provider(self) -> RedditProvider | None:
        return None

    def x_provider(self, symbol: str) -> XProvider | None:
        del symbol
        return None

    def news_providers(self, symbol: str) -> tuple[NewsProvider, ...]:
        del symbol
        return (_UnitLiveMissingNewsProvider(),)

    def fundamentals_providers(self, symbol: str) -> tuple[FundamentalsProvider, ...]:
        del symbol
        return ()

    def macro_providers(self, symbol: str) -> tuple[MacroProvider, ...]:
        del symbol
        return ()


class _UnitLiveMarketDataProvider:
    provider_name = "unit-live-market-data"

    def fetch_daily_candles(
        self,
        request: MarketDataRequest,
    ) -> ProviderResult[MarketSnapshot]:
        return no_data_result(
            provider_name=self.provider_name,
            request=request,
            fetched_at=NOW,
            message="Unit live market provider returned no bars.",
        )

    def health(self) -> ProviderHealth:
        return self.fetch_daily_candles(
            MarketDataRequest(
                request_id="unit-live-market-health",
                run_date=RUN_DATE,
                tickers=("TSLA",),
            )
        ).health


class _UnitLiveMissingNewsProvider:
    provider_name = "unit-live-news"

    def fetch_articles(
        self,
        request: EvidenceRequest,
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        return missing_credentials_result(
            provider_name=self.provider_name,
            request=request,
            fetched_at=NOW,
            credential_name="unit live news API key",
        )

    def health(self) -> ProviderHealth:
        return self.fetch_articles(
            EvidenceRequest(
                request_id="unit-live-news-health",
                run_date=RUN_DATE,
                tickers=("TSLA",),
            )
        ).health
