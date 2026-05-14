from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    AssetClass,
    Direction,
    EvidenceReference,
    FundamentalsProvider,
    FundamentalsRequest,
    FundamentalsSnapshot,
    InstrumentQuery,
    InstrumentUniverseRequest,
    JsonObject,
    MacroProvider,
    MacroRequest,
    MacroSeries,
    MacroSnapshot,
    PredictionCandidate,
    PredictionChangeTrigger,
    PredictionStatus,
    ProviderHealth,
    ProviderMetric,
    ProviderResult,
    ProviderStatus,
    SourceEvidence,
    TimeHorizon,
)
from nlp_stock_prediction.evaluation import (
    attach_evaluation_metadata,
    write_prediction_evaluation_artifact,
)
from nlp_stock_prediction.orchestration import (
    ArtifactWriter,
    Phase4Service,
    Phase4UniverseDiscoveryTool,
    RunContext,
)
from nlp_stock_prediction.orchestration.phase4_fundamentals import (
    run_phase4_fundamentals_tool,
)
from nlp_stock_prediction.orchestration.phase4_market_data import Phase4MarketDataTool
from nlp_stock_prediction.orchestration.phase4_sector_macro import (
    run_phase4_sector_macro_tool,
)
from nlp_stock_prediction.orchestration.phase4_technical_package import (
    Phase4TechnicalPackageTool,
)
from nlp_stock_prediction.providers.candlecharts import CandlechartsMarketDataProvider
from nlp_stock_prediction.storage import (
    PredictionCandidateRecord,
)

RUN_DATE = date(2026, 5, 13)
STARTED_AT = datetime(2026, 5, 13, 14, 0, tzinfo=UTC)
COMPLETED_AT = datetime(2026, 5, 13, 14, 5, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.e2e
def test_phase4_service_e2e_runs_complete_fixture_backed_tool_suite(
    tmp_path: Path,
) -> None:
    service = Phase4Service(
        repo_root=tmp_path,
        fixture_root=REPO_ROOT,
        database_path=Path("data") / "prediction-research.sqlite3",
    )

    result = service.run_offline_phase4_flow(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/phase4-full-tool-suite",
        symbol="TSLA",
    )

    run_id = str(result["run_id"])
    report = cast(dict[str, object], result["report"])
    tool_runs = service.store.list_tool_runs_for_run(run_id)
    tool_names = {record.tool_name for record in tool_runs}
    tool_statuses = {record.tool_name: record.status for record in tool_runs}
    tool_warnings = {record.tool_name: record.warnings for record in tool_runs}
    artifact_types = {
        record.artifact_type for record in service.store.list_artifacts_for_run(run_id)
    }
    evidence_providers = {record.provider for record in service.store.list_evidence_for_run(run_id)}
    candidates = service.store.list_prediction_candidates_for_run(run_id)
    json_path = Path(str(report["json_path"]))
    markdown_path = Path(str(report["markdown_path"]))
    report_payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert {
        "phase4_universe_discovery",
        "phase4_market_data",
        "phase4_technical_package",
        "phase4_social_evidence",
        "phase4_news_catalyst",
        "phase4_fundamentals",
        "phase4_sector_macro",
        "phase4_prediction_candidate_synthesis",
        "phase4_prediction_evaluation",
        "render_prediction_report",
    }.issubset(tool_names)
    assert tool_statuses["phase4_social_evidence"] == "successful"
    assert tool_statuses["phase4_news_catalyst"] in {"successful", "partial"}
    if tool_statuses["phase4_news_catalyst"] == "partial":
        assert any("stale_data" in warning for warning in tool_warnings["phase4_news_catalyst"])
    assert tool_statuses["phase4_fundamentals"] == "successful"
    assert tool_statuses["phase4_prediction_candidate_synthesis"] == "successful"
    assert {
        "instrument_universe",
        "market_data",
        "technical_package",
        "normalized_evidence",
        "analysis_context",
        "prediction_input",
        "prediction_evaluation",
        "markdown_report",
        "json_report",
        "audit_manifest",
    }.issubset(artifact_types)
    assert {
        "reddit",
        "fixture-x-recent-search",
        "fixture-news",
        "fixture-sec-edgar",
    }.issubset(evidence_providers)
    assert candidates
    assert candidates[0].status == PredictionStatus.EVIDENCE_SUPPORTED.value
    assert candidates[0].evidence_for
    assert json_path.parent.name == "tsla"
    assert markdown_path.parent == json_path.parent
    assert report_payload["prediction_candidates"][0]["metadata"]["prediction_evaluation"]
    assert "Recommendation:" not in markdown
    assert "Trade instruction:" not in markdown


@pytest.mark.e2e
def test_phase4_real_tool_suite_e2e_runs_fixture_backed_production_gate(
    tmp_path: Path,
) -> None:
    service = Phase4Service(
        repo_root=tmp_path,
        database_path=Path("data") / "prediction-research.sqlite3",
    )
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/phase4-real-tool-suite",
        symbol="TSLA",
        objective="Fixture-backed Phase 4 production gate for TSLA.",
    )
    run_id = str(started["run_id"])
    audit_dir = Path(str(started["audit_dir"]))
    context = _real_tool_context(tmp_path, started)
    store = service.store

    universe_result = Phase4UniverseDiscoveryTool().run(
        request=InstrumentUniverseRequest(
            request_id="phase4-real-tool-suite-universe",
            as_of=STARTED_AT,
            queries=(
                InstrumentQuery(query="TSLA", asset_class=AssetClass.STOCK),
                InstrumentQuery(query="SPY", asset_class=AssetClass.ETF),
            ),
            metadata={"gate": "phase4-real-tool-suite"},
        ),
        context=context,
        store=store,
        repo_root=tmp_path,
    )
    tsla_instrument_id = "instrument:equity:us:tsla"

    market_result = Phase4MarketDataTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=audit_dir,
        provider=CandlechartsMarketDataProvider(
            html=_candlecharts_fixture("public_ohlcv_tsla.html"),
            now=lambda: STARTED_AT,
        ),
        now=lambda: STARTED_AT,
    ).run(
        run_id=run_id,
        run_date=RUN_DATE,
        symbol="TSLA",
        instrument_id=tsla_instrument_id,
        source_url=_candlecharts_fixture_path("public_ohlcv_tsla.html"),
    )
    technical_result = Phase4TechnicalPackageTool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=audit_dir,
        now=lambda: STARTED_AT,
    ).run(
        run_id=run_id,
        symbol="TSLA",
        market_data=market_result,
        instrument_id=tsla_instrument_id,
    )
    fundamentals_result = run_phase4_fundamentals_tool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=audit_dir,
        run_id=run_id,
        symbol="TSLA",
        run_date=RUN_DATE,
        generated_at=STARTED_AT,
        providers=(cast(FundamentalsProvider, _FixtureFundamentalsProvider()),),
        instrument_id=tsla_instrument_id,
    )
    sector_macro_result = run_phase4_sector_macro_tool(
        store=store,
        repo_root=tmp_path,
        artifact_dir=audit_dir,
        run_id=run_id,
        symbol="TSLA",
        run_date=RUN_DATE,
        generated_at=STARTED_AT,
        target_snapshot=_fundamentals_snapshot("TSLA"),
        macro_providers=(cast(MacroProvider, _FixtureMacroProvider()),),
        instrument_id=tsla_instrument_id,
        peers=(_fundamentals_snapshot("F"),),
        sector="Consumer Discretionary",
        benchmark_symbol="SPY",
        benchmark_metrics=(
            ProviderMetric(
                name="benchmark_return_20d",
                value=Decimal("0.018"),
                unit="pct",
                as_of=RUN_DATE,
            ),
        ),
        macro_series_ids=("FEDFUNDS",),
        horizon=TimeHorizon.MONTHLY,
    )
    source_evidence = _source_evidence_from_payloads(
        fundamentals_result.artifact_payload,
        sector_macro_result.artifact_payload,
    )
    candidate = _real_tool_candidate(
        instrument_id=tsla_instrument_id,
        evidence_id=fundamentals_result.evidence_ids[0],
        signal_artifact_id=technical_result.artifact.artifact_id,
    )
    store.upsert_prediction_candidate(
        _candidate_record_from_contract(run_id=run_id, candidate=candidate)
    )
    evaluation, evaluation_artifact = write_prediction_evaluation_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=audit_dir,
        run_id=run_id,
        candidate=candidate,
        evidence_sources=source_evidence,
        created_at=COMPLETED_AT,
    )
    evaluated_candidate = attach_evaluation_metadata(
        candidate,
        evaluation,
        artifact=evaluation_artifact,
    )
    store.upsert_prediction_candidate(
        _candidate_record_from_contract(run_id=run_id, candidate=evaluated_candidate)
    )

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    tool_names = {record.tool_name for record in store.list_tool_runs_for_run(run_id)}
    artifact_types = {record.artifact_type for record in store.list_artifacts_for_run(run_id)}
    report_payload = json.loads(Path(str(rendered["json_path"])).read_text(encoding="utf-8"))
    markdown = Path(str(rendered["markdown_path"])).read_text(encoding="utf-8")

    assert universe_result.instrument_ids == (tsla_instrument_id, "instrument:etf:us:spy")
    assert {
        "phase4_universe_discovery",
        "phase4_market_data",
        "phase4_technical_package",
        "phase4_fundamentals",
        "phase4_sector_macro",
        "phase4_prediction_evaluation",
        "render_prediction_report",
    }.issubset(tool_names)
    assert {
        "instrument_universe",
        "market_data",
        "technical_package",
        "analysis_context",
        "prediction_evaluation",
        "markdown_report",
        "json_report",
        "audit_manifest",
    }.issubset(artifact_types)
    assert store.list_evidence_for_run(run_id)
    assert store.list_prediction_candidates_for_run(run_id)[0].candidate_id == (
        candidate.candidate_id
    )
    assert evaluation.status == PredictionStatus.EVIDENCE_SUPPORTED
    assert (
        report_payload["prediction_candidates"][0]["metadata"]["prediction_evaluation"][
            "artifact_id"
        ]
        == evaluation_artifact.artifact_id
    )
    assert "market-data/tsla.json" in markdown
    assert "Recommendation:" not in markdown
    assert "Trade instruction:" not in markdown
    assert "dummy structural tools" not in markdown


@dataclass(frozen=True)
class _FixtureFundamentalsProvider:
    provider_name: str = "phase4-fixture-fundamentals"

    def fetch_fundamentals(
        self,
        request: FundamentalsRequest,
    ) -> ProviderResult[FundamentalsSnapshot]:
        snapshot = _fundamentals_snapshot(request.tickers[0] if request.tickers else "TSLA")
        return ProviderResult[FundamentalsSnapshot](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=STARTED_AT,
            data=snapshot,
            health=_provider_health(self.provider_name),
            raw_snapshot_id=f"raw-{self.provider_name}-{snapshot.ticker.lower()}",
            cache_key=f"fixture:{snapshot.ticker}",
        )

    def health(self) -> ProviderHealth:
        return _provider_health(self.provider_name)


@dataclass(frozen=True)
class _FixtureMacroProvider:
    provider_name: str = "phase4-fixture-macro"

    def fetch_macro(self, request: MacroRequest) -> ProviderResult[MacroSnapshot]:
        series_id = request.series_ids[0] if request.series_ids else "FEDFUNDS"
        snapshot = MacroSnapshot(
            series=(
                MacroSeries(
                    series_id=series_id,
                    name="Fixture federal funds rate",
                    values=(
                        ProviderMetric(
                            name="fed_funds_rate",
                            value=Decimal("4.25"),
                            unit="pct",
                            as_of=RUN_DATE,
                        ),
                    ),
                ),
            )
        )
        return ProviderResult[MacroSnapshot](
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=STARTED_AT,
            data=snapshot,
            health=_provider_health(self.provider_name),
            raw_snapshot_id=f"raw-{self.provider_name}-{series_id.lower()}",
            cache_key=f"fixture:{series_id}",
        )

    def health(self) -> ProviderHealth:
        return _provider_health(self.provider_name)


def _real_tool_context(repo_root: Path, started: Mapping[str, object]) -> RunContext:
    run_dir = Path(str(started["run_dir"]))
    audit_dir = Path(str(started["audit_dir"]))
    return RunContext(
        run_id=str(started["run_id"]),
        run_date=RUN_DATE,
        generated_at=STARTED_AT,
        timezone="UTC",
        output_dir=Path(str(started["output_dir"])),
        report_dir=run_dir,
        audit_dir=audit_dir,
        command_args={"offline": True, "gate": "phase4-real-tool-suite"},
        artifact_writer=ArtifactWriter(
            base_dir=audit_dir,
            created_at=STARTED_AT,
            produced_by="phase4-real-tool-suite",
        ),
    )


def _provider_health(provider_name: str) -> ProviderHealth:
    return ProviderHealth(
        provider_name=provider_name,
        status=ProviderStatus.OK,
        checked_at=STARTED_AT,
    )


def _fundamentals_snapshot(symbol: str) -> FundamentalsSnapshot:
    normalized = symbol.strip().upper()
    return FundamentalsSnapshot(
        ticker=normalized,
        company_name=f"{normalized} Fixture Company",
        metrics=(
            ProviderMetric(
                name="revenue_growth",
                value=Decimal("0.12") if normalized == "TSLA" else Decimal("0.04"),
                unit="ratio",
                as_of=RUN_DATE,
                metadata={"source_url": f"https://example.test/fundamentals/{normalized}"},
            ),
            ProviderMetric(
                name="gross_margin",
                value=Decimal("0.18") if normalized == "TSLA" else Decimal("0.09"),
                unit="ratio",
                as_of=RUN_DATE,
                metadata={"source_url": f"https://example.test/fundamentals/{normalized}"},
            ),
        ),
    )


def _real_tool_candidate(
    *,
    instrument_id: str,
    evidence_id: str,
    signal_artifact_id: str,
) -> PredictionCandidate:
    return PredictionCandidate(
        candidate_id="candidate-phase4-real-tool-tsla",
        instrument_id=instrument_id,
        symbol="TSLA",
        horizon=TimeHorizon.SWING,
        direction=Direction.MIXED,
        status=PredictionStatus.EVIDENCE_SUPPORTED,
        thesis=(
            "TSLA fixture scenario has source-attributed fundamental support with technical "
            "and sector/macro context preserved as sidecar evidence."
        ),
        baseline="No directional edge is assumed without attributed source evidence.",
        confidence=0.61,
        evidence_for=(EvidenceReference(evidence_id=evidence_id),),
        signal_artifact_ids=(signal_artifact_id,),
        assumptions=("Fixture providers are deterministic and offline.",),
        uncertainties=("Provider fixtures do not represent live market conditions.",),
        change_triggers=(
            PredictionChangeTrigger(
                trigger_id="change-phase4-real-tool-live-evidence",
                summary="Fresh live provider evidence would change the fixture scenario support.",
                trigger_type="provider_refresh",
                evidence=(EvidenceReference(evidence_id=evidence_id),),
                artifact_ids=(signal_artifact_id,),
                rationale="The Phase 4 gate uses controlled fixture providers.",
            ),
        ),
        metadata={"symbol": "TSLA", "gate": "phase4-real-tool-suite"},
    )


def _candidate_record_from_contract(
    *,
    run_id: str,
    candidate: PredictionCandidate,
) -> PredictionCandidateRecord:
    return PredictionCandidateRecord(
        candidate_id=candidate.candidate_id,
        run_id=run_id,
        instrument_id=candidate.instrument_id,
        prediction_horizon=candidate.horizon.value,
        prediction_type="directional",
        scenario=candidate.thesis,
        status=candidate.status.value,
        confidence=candidate.confidence,
        direction=candidate.direction.value,
        evidence_for=tuple(reference.evidence_id for reference in candidate.evidence_for),
        evidence_against=tuple(reference.evidence_id for reference in candidate.evidence_against),
        signal_artifacts=candidate.signal_artifact_ids,
        baseline={"summary": candidate.baseline},
        uncertainty="; ".join(candidate.uncertainties),
        metadata=candidate.metadata,
    )


def _source_evidence_from_payloads(
    *payloads: JsonObject | None,
) -> tuple[SourceEvidence, ...]:
    records: list[SourceEvidence] = []
    for payload in payloads:
        if payload is None:
            continue
        raw_records = payload.get("source_evidence")
        if not isinstance(raw_records, list):
            continue
        records.extend(SourceEvidence.model_validate(record) for record in raw_records)
    return tuple(records)


def _candlecharts_fixture_path(name: str) -> Path:
    return Path(__file__).parent / "fixtures" / "raw" / "candlecharts" / name


def _candlecharts_fixture(name: str) -> str:
    return _candlecharts_fixture_path(name).read_text(encoding="utf-8")
