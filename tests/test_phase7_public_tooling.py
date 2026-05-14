from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from nlp_stock_prediction.cli import build_parser, main
from nlp_stock_prediction.contracts import (
    FreshnessStatus,
    JsonObject,
    RetrievalMethod,
    SourceKind,
    SourceProvenance,
)
from nlp_stock_prediction.orchestration.phase6_service import (
    Phase6Service,
    build_phase6_tool_registry,
)
from nlp_stock_prediction.storage import (
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    SQLiteStore,
    ToolRunRecord,
)

RUN_ID = "run-phase7-public-evaluation"
INSTRUMENT_ID = "instrument:equity:us:msft"
NOW = datetime(2026, 5, 14, 18, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 5, 14, 20, 0, tzinfo=UTC)
WINDOW_START = datetime(2026, 5, 15, 13, 30, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 18, 20, 0, tzinfo=UTC)
OBSERVED_AT = datetime(2026, 5, 18, 20, 15, tzinfo=UTC)
EVALUATED_AT = datetime(2026, 5, 18, 21, 0, tzinfo=UTC)


class _FakeMcpServer:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def tool(self) -> Any:
        def decorator(func: Any) -> Any:
            self.registered.append(func.__name__)
            return func

        return decorator


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "data" / "prediction-research.sqlite3")
    store.initialize()
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id=INSTRUMENT_ID,
            symbol="MSFT",
            asset_class="stock",
            name="Microsoft Corporation",
        )
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=RUN_ID,
            run_kind="evaluation_hardening",
            objective="Exercise the public evaluation interface.",
            status="running",
            started_at=NOW,
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    return store


def _persist_candidate(store: SQLiteStore) -> None:
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-public-msft",
            run_id=RUN_ID,
            instrument_id=INSTRUMENT_ID,
            prediction_horizon="swing",
            prediction_type="directional",
            scenario="Stored source-backed directional scenario for public evaluation.",
            direction="bullish",
            confidence=0.68,
            status="evidence_supported",
            evidence_for=("evidence-public-support",),
            baseline={
                "baseline_id": "no_directional_edge",
                "baseline_summary": "No directional edge without source-backed evidence.",
                "baseline_score": 0.5,
                "candidate_score": 0.68,
                "score_delta": 0.18,
                "verdict": "above_baseline",
            },
        )
    )
    for evidence_id in ("evidence-public-support", "evidence-public-outcome"):
        store.record_evidence(
            EvidenceRecord(
                evidence_id=evidence_id,
                source_type=SourceKind.MARKET_DATA.value,
                provider="verified-market-data",
                retrieved_at=CUTOFF if evidence_id.endswith("support") else OBSERVED_AT,
                published_at=CUTOFF if evidence_id.endswith("support") else OBSERVED_AT,
                instruments=(INSTRUMENT_ID,),
                claim=f"{evidence_id} was available to the evaluation workflow.",
                freshness_status=FreshnessStatus.FRESH.value,
            )
        )


def _persist_outcome_evaluation(tmp_path: Path) -> Phase6Service:
    store = _store(tmp_path)
    _persist_candidate(store)
    service = Phase6Service(repo_root=tmp_path)
    service.phase6_point_in_time_outcome_evaluation(
        run_id=RUN_ID,
        candidate_id="candidate-public-msft",
        point_in_time_cutoff=CUTOFF.isoformat(),
        evaluation_window_start=WINDOW_START.isoformat(),
        evaluation_window_end=WINDOW_END.isoformat(),
        observed_result="supported",
        observed_at=OBSERVED_AT.isoformat(),
        result_summary="MSFT outcome evidence supported the stored scenario.",
        outcome_evidence_ids=("evidence-public-outcome",),
        created_at=OBSERVED_AT.isoformat(),
        evaluated_at=EVALUATED_AT.isoformat(),
    )
    return service


def _persist_live_evidence(store: SQLiteStore) -> None:
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-public-live-provider",
            run_id=RUN_ID,
            tool_name="phase4_market_data",
            tool_version="phase4.market-data.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "MSFT"},
        )
    )
    provenance = SourceProvenance(
        provider_name="alpha-vantage-market-data",
        source_kind=SourceKind.MARKET_DATA,
        retrieval_method=RetrievalMethod.OFFICIAL_API,
        fetched_at=NOW,
        observed_at=NOW,
        source_url="https://example.com/alpha-vantage/MSFT",
        permalink="https://example.com/alpha-vantage/MSFT",
        raw_identifier="MSFT",
        raw_snapshot_id="raw-live-msft",
        query="MSFT",
        freshness_status=FreshnessStatus.FRESH,
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-public-live-market",
            tool_run_id="tool-public-live-provider",
            source_type=SourceKind.MARKET_DATA.value,
            provider="alpha-vantage-market-data",
            url=provenance.source_url,
            query="MSFT",
            retrieved_at=NOW,
            published_at=NOW,
            instruments=(INSTRUMENT_ID,),
            claim="MSFT live market data was retrieved from an official provider.",
            extraction_confidence=0.96,
            source_reliability="provider_metric",
            freshness_status=FreshnessStatus.FRESH.value,
            provenance_json=cast(JsonObject, provenance.model_dump(mode="json")),
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )


@pytest.mark.unit
def test_evaluation_group_parser_lists_public_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["evaluation", "--help"])

    assert exc.value.code == 0
    help_output = capsys.readouterr().out
    for command in (
        "inspect",
        "materialize-outcome",
        "load-outcomes",
        "outcome-summary",
        "stale-artifacts",
        "source-reliability",
        "provider-playbook",
        "calibration",
        "walk-forward",
        "ablation",
        "calibration-drift",
    ):
        assert command in help_output
    assert "dummy" not in help_output.lower()
    assert "fixture" not in help_output.lower()


@pytest.mark.unit
def test_evaluation_cli_requires_existing_explicit_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "evaluation",
            "--repo-root",
            str(tmp_path),
            "--database",
            str(tmp_path / "missing.sqlite3"),
            "inspect",
            "--run-id",
            RUN_ID,
        ]
    )

    assert code == 3
    assert (
        "--database must reference an existing research SQLite database" in capsys.readouterr().err
    )


@pytest.mark.unit
def test_evaluation_cli_inspect_reads_explicit_run_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = _store(tmp_path)

    code = main(
        [
            "evaluation",
            "--repo-root",
            str(tmp_path),
            "--database",
            str(store.path),
            "inspect",
            "--run-id",
            RUN_ID,
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == RUN_ID
    assert payload["artifact_count"] == 0


@pytest.mark.unit
def test_phase7_public_service_writes_real_evaluation_artifacts(tmp_path: Path) -> None:
    service = _persist_outcome_evaluation(tmp_path)
    audit_dir = tmp_path / "reports" / RUN_ID / "audit"

    outcome_summary = service.evaluation_outcome_summary(
        run_id=RUN_ID,
        artifact_dir=audit_dir.as_posix(),
        created_at=EVALUATED_AT.isoformat(),
    )
    stale_artifacts = service.evaluation_stale_artifacts(
        run_id=RUN_ID,
        artifact_dir=audit_dir.as_posix(),
        reviewed_at=EVALUATED_AT.isoformat(),
    )
    provider_playbooks = service.evaluation_provider_playbook(
        run_id=RUN_ID,
        artifact_dir=audit_dir.as_posix(),
        created_at=EVALUATED_AT.isoformat(),
    )

    assert cast(int, outcome_summary["summary_count"]) == 1
    assert cast(int, stale_artifacts["review_count"]) >= 2
    assert cast(int, provider_playbooks["playbook_count"]) >= 1
    for result in (outcome_summary, stale_artifacts):
        artifact_path = tmp_path / str(result["artifact_path"])
        assert artifact_path.exists()
    assert all(
        (tmp_path / path).exists() for path in cast(list[str], provider_playbooks["artifact_paths"])
    )


@pytest.mark.unit
def test_phase7_public_service_writes_source_reliability_notes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _persist_live_evidence(store)
    service = Phase6Service(repo_root=tmp_path)

    result = service.evaluation_source_reliability(
        run_id=RUN_ID,
        artifact_dir=(tmp_path / "reports" / RUN_ID / "audit").as_posix(),
        created_at=NOW.isoformat(),
    )

    assert result["note_count"] == 1
    assert result["reliability_counts"] == {"high": 1}
    assert (tmp_path / cast(list[str], result["artifact_paths"])[0]).exists()


@pytest.mark.unit
def test_phase7_registry_and_mcp_expose_phase_neutral_evaluation_tools(tmp_path: Path) -> None:
    from nlp_stock_prediction.orchestration.phase6_mcp_registration import (
        PHASE6_MCP_TOOL_NAMES,
        register_phase6_mcp_tools,
    )

    plan = build_phase6_tool_registry().as_plan()
    tool_names = [str(tool["tool_name"]) for tool in cast(list[dict[str, object]], plan["tools"])]
    server = _FakeMcpServer()

    register_phase6_mcp_tools(server, Phase6Service(repo_root=tmp_path))

    assert tool_names == [
        "evaluation_materialize_outcome",
        "evaluation_load_outcomes",
        "evaluation_ablation",
        "evaluation_walk_forward",
        "evaluation_outcome_summary",
        "evaluation_stale_artifacts",
        "evaluation_source_reliability",
        "evaluation_provider_playbook",
        "evaluation_calibration",
        "evaluation_calibration_drift",
        "evaluation_inspect",
    ]
    assert list(PHASE6_MCP_TOOL_NAMES) == server.registered
    assert server.registered[0] == "list_evaluation_tool_plan"
    assert not any(
        "dummy" in name or "fixture" in name or "scaffold" in name for name in tool_names
    )


@pytest.mark.unit
def test_codex_mcp_server_registers_public_evaluation_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlp_stock_prediction.codex_mcp import build_server
    from nlp_stock_prediction.orchestration.phase4_mcp_registration import PHASE4_MCP_TOOL_NAMES
    from nlp_stock_prediction.orchestration.phase6_mcp_registration import PHASE6_MCP_TOOL_NAMES

    fastmcp_module = ModuleType("mcp.server.fastmcp")

    class _FastMCP(_FakeMcpServer):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    fastmcp_module.FastMCP = _FastMCP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp", ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)

    server = build_server(repo_root=tmp_path, database_path=Path("data/test.sqlite3"))

    assert server.registered == [*PHASE4_MCP_TOOL_NAMES, *PHASE6_MCP_TOOL_NAMES]
