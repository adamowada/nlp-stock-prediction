from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import pytest

from nlp_stock_prediction.contracts import (
    AuditArtifact,
    AuditManifest,
    CodexEvidenceImport,
    DailyReport,
    DataFreshnessSummary,
    DataReference,
    Direction,
    EvidenceReference,
    FreshnessStatus,
    Instrument,
    InstrumentReportSection,
    InstrumentUniverse,
    OrchestratorRunSummary,
    PredictionCandidate,
    PredictionStatus,
    ProviderHealth,
    ProviderStatus,
    ResearchObjective,
    ResearchToolSpec,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
    TimeHorizon,
    ToolExecutionResult,
    ToolInvocation,
)
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    CandidateArtifactLinkRecord,
    CandidateEvidenceLinkRecord,
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    SourceQueryRecord,
    SQLiteStore,
    ToolRunRecord,
)

RUN_DATE = date(2026, 5, 13)
STARTED_AT = datetime(2026, 5, 13, 14, 0, tzinfo=UTC)
COMPLETED_AT = datetime(2026, 5, 13, 14, 5, tzinfo=UTC)

_ReferenceType = Literal[
    "raw_snapshot",
    "normalized_evidence",
    "extraction",
    "analysis",
    "scoring_input",
    "report",
    "audit_artifact",
]


@dataclass(frozen=True)
class _Phase4FixtureRun:
    summary: OrchestratorRunSummary
    report: DailyReport
    markdown_path: Path
    json_path: Path
    audit_manifest_path: Path


@pytest.mark.e2e
def test_phase4_fixture_tool_suite_e2e_scaffold_persists_artifacts_evidence_and_report(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "prediction-research.sqlite3")
    store.initialize()

    fixture_run = _run_phase4_fixture_tool_suite(tmp_path, store)

    tool_names = tuple(tool.tool_name for tool in fixture_run.summary.selected_tools)
    assert tool_names == (
        "phase4_fixture_universe_discovery",
        "phase4_fixture_market_data",
        "phase4_fixture_news_catalyst",
        "phase4_fixture_social_evidence",
        "phase4_fixture_prediction_evaluation",
        "phase4_fixture_report",
    )
    assert fixture_run.summary.status == "succeeded"
    assert fixture_run.summary.prediction_candidate_ids == ("candidate-phase4-tsla-weekly",)

    stored_tool_runs = store.list_tool_runs_for_run(fixture_run.summary.run_id)
    stored_artifacts = store.list_artifacts_for_run(fixture_run.summary.run_id)
    stored_evidence = store.list_evidence_for_run(fixture_run.summary.run_id)
    stored_candidates = store.list_prediction_candidates_for_run(fixture_run.summary.run_id)

    assert {tool.tool_name for tool in stored_tool_runs} == set(tool_names)
    assert {artifact.artifact_type for artifact in stored_artifacts} >= {
        "instrument_universe",
        "provider_result",
        "normalized_evidence",
        "prediction_input",
        "markdown_report",
        "json_report",
        "audit_manifest",
    }
    assert {record.evidence_id for record in stored_evidence} >= {
        "fixture-evidence-sector-xly",
        "phase4-news-tsla-deliveries",
        "phase4-social-tsla-valuation-risk",
    }
    assert stored_candidates[0].candidate_id == "candidate-phase4-tsla-weekly"
    assert stored_candidates[0].evidence_for == ("phase4-news-tsla-deliveries",)
    assert stored_candidates[0].evidence_against == ("phase4-social-tsla-valuation-risk",)

    markdown = fixture_run.markdown_path.read_text(encoding="utf-8")
    report_payload = json.loads(fixture_run.json_path.read_text(encoding="utf-8"))
    audit_payload = json.loads(fixture_run.audit_manifest_path.read_text(encoding="utf-8"))

    assert "## Evidence Ledger" in markdown
    assert "phase4-news-tsla-deliveries" in markdown
    assert "Recommendation:" not in markdown
    assert "Trade instruction:" not in markdown
    assert report_payload["prediction_candidates"][0]["status"] == "contradicted"
    assert report_payload["prediction_candidates"][0]["evidence_against"][0]["evidence_id"] == (
        "phase4-social-tsla-valuation-risk"
    )
    assert any(
        artifact["artifact_type"] == "markdown_report" for artifact in audit_payload["artifacts"]
    )


@pytest.mark.e2e
def test_phase4_fixture_tool_suite_e2e_scaffold_keeps_partial_failures_visible(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "prediction-research.sqlite3")
    store.initialize()

    fixture_run = _run_phase4_fixture_tool_suite(tmp_path, store, fail_social_tool=True)

    failed_results = [
        result for result in fixture_run.summary.tool_results if result.status == "failed"
    ]
    assert fixture_run.summary.status == "partial"
    assert fixture_run.summary.warnings == (
        "phase4_fixture_social_evidence failed; report used remaining fixture evidence.",
    )
    assert failed_results[0].error_message == "fixture social evidence provider unavailable"
    assert fixture_run.markdown_path.exists()

    stored_social_tool = store.get_tool_run("tool-phase4-social-evidence")
    stored_candidates = store.list_prediction_candidates_for_run(fixture_run.summary.run_id)

    assert stored_social_tool is not None
    assert stored_social_tool.status == "failed"
    assert stored_social_tool.error_message == "fixture social evidence provider unavailable"
    assert stored_candidates[0].status == "evidence_supported"
    assert stored_candidates[0].evidence_against == ()


def _run_phase4_fixture_tool_suite(
    repo_root: Path,
    store: SQLiteStore,
    *,
    fail_social_tool: bool = False,
) -> _Phase4FixtureRun:
    """Fixture-backed scaffold for future Phase 4 tool-suite orchestration tests."""

    run_id = "phase4-fixture-2026-05-13-tsla"
    artifact_root = repo_root / "artifacts" / "phase4-fixture"
    objective = ResearchObjective(
        objective_id="objective-phase4-fixture-tsla",
        run_date=RUN_DATE,
        prompt="Fixture-backed Phase 4 research flow for TSLA with cross-tool provenance.",
        requested_symbols=("TSLA", "BTC/USD", "ESM6"),
        time_horizon=TimeHorizon.WEEKLY,
        offline=True,
        max_tool_calls=8,
        metadata={"scaffold": True, "phase": 4},
    )
    selected_tools = _phase4_tool_specs()
    invocation_by_name = _phase4_invocations(objective)

    store.upsert_research_run(
        ResearchRunRecord(
            run_id=run_id,
            run_kind="phase4_fixture_tool_suite",
            objective=objective.prompt,
            status="running",
            started_at=STARTED_AT,
            metadata={"objective_id": objective.objective_id, "scaffold": True},
        )
    )

    universe = _load_universe_fixture()
    for instrument in universe.instruments:
        store.upsert_instrument(_instrument_record(instrument))

    universe_ref = _write_json_artifact(
        repo_root,
        artifact_root,
        "artifact-phase4-universe",
        "analysis",
        universe.model_dump(mode="json"),
    )
    market_evidence = _market_context_evidence()
    market_ref = _write_json_artifact(
        repo_root,
        artifact_root,
        "artifact-phase4-market-context",
        "analysis",
        {"evidence_ids": [market_evidence.evidence_id], "role": "sector-baseline"},
    )
    news_evidence = _news_evidence()
    news_ref = _write_json_artifact(
        repo_root,
        artifact_root,
        "artifact-phase4-news-evidence",
        "normalized_evidence",
        {"evidence": [news_evidence.model_dump(mode="json")]},
    )

    social_evidence = _social_evidence()
    social_ref = _write_json_artifact(
        repo_root,
        artifact_root,
        "artifact-phase4-social-evidence",
        "normalized_evidence",
        {"evidence": [social_evidence.model_dump(mode="json")]},
    )

    evidence_sources = [market_evidence, news_evidence]
    if not fail_social_tool:
        evidence_sources.append(social_evidence)

    candidate = _prediction_candidate(
        evidence_against=() if fail_social_tool else (_social_evidence_ref(),)
    )
    scoring_ref = _write_json_artifact(
        repo_root,
        artifact_root,
        "artifact-phase4-prediction-input",
        "scoring_input",
        candidate.model_dump(mode="json"),
    )
    audit_manifest = _audit_manifest(
        run_id,
        (
            (universe_ref, "instrument_universe"),
            (market_ref, "provider_result"),
            (news_ref, "normalized_evidence"),
            (scoring_ref, "prediction_input"),
            *_optional_artifact_tuple(social_ref, fail_social_tool),
            (
                _planned_reference(
                    "artifact-phase4-markdown-report",
                    "report",
                    "artifacts/phase4-fixture/artifact-phase4-markdown-report.md",
                ),
                "markdown_report",
            ),
            (
                _planned_reference(
                    "artifact-phase4-json-report",
                    "report",
                    "artifacts/phase4-fixture/artifact-phase4-json-report.json",
                ),
                "json_report",
            ),
            (
                _planned_reference(
                    "artifact-phase4-audit-manifest",
                    "audit_artifact",
                    "artifacts/phase4-fixture/artifact-phase4-audit-manifest.json",
                ),
                "audit_manifest",
            ),
        ),
    )
    report = _daily_report(
        run_id=run_id,
        objective=objective,
        universe=universe,
        evidence=tuple(evidence_sources),
        candidate=candidate,
        audit_manifest=audit_manifest,
        partial=fail_social_tool,
    )

    markdown_path = artifact_root / "artifact-phase4-markdown-report.md"
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    markdown_ref = _reference_from_path(
        repo_root,
        markdown_path,
        "artifact-phase4-markdown-report",
        "report",
    )
    json_path = artifact_root / "artifact-phase4-json-report.json"
    json_path.write_text(render_json_report(report), encoding="utf-8")
    json_ref = _reference_from_path(
        repo_root,
        json_path,
        "artifact-phase4-json-report",
        "report",
    )
    audit_manifest_path = artifact_root / "artifact-phase4-audit-manifest.json"
    audit_manifest_path.write_text(
        f"{audit_manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    audit_ref = _reference_from_path(
        repo_root,
        audit_manifest_path,
        "artifact-phase4-audit-manifest",
        "audit_artifact",
    )

    _record_tool_run(
        store,
        run_id,
        invocation_by_name["phase4_fixture_universe_discovery"],
        status="succeeded",
        artifacts=((universe_ref, "instrument_universe", "instrument_universe.v1"),),
    )
    _record_tool_run(
        store,
        run_id,
        invocation_by_name["phase4_fixture_market_data"],
        status="succeeded",
        artifacts=((market_ref, "provider_result", "provider_result.v1"),),
        evidence=(market_evidence,),
    )
    _record_tool_run(
        store,
        run_id,
        invocation_by_name["phase4_fixture_news_catalyst"],
        status="succeeded",
        artifacts=((news_ref, "normalized_evidence", "normalized_evidence.v1"),),
        evidence=(news_evidence,),
    )
    if fail_social_tool:
        _record_tool_run(
            store,
            run_id,
            invocation_by_name["phase4_fixture_social_evidence"],
            status="failed",
            error_message="fixture social evidence provider unavailable",
        )
    else:
        _record_tool_run(
            store,
            run_id,
            invocation_by_name["phase4_fixture_social_evidence"],
            status="succeeded",
            artifacts=((social_ref, "normalized_evidence", "normalized_evidence.v1"),),
            evidence=(social_evidence,),
        )
    _record_tool_run(
        store,
        run_id,
        invocation_by_name["phase4_fixture_prediction_evaluation"],
        status="succeeded",
        artifacts=((scoring_ref, "prediction_input", "prediction_input.v1"),),
    )
    _record_candidate(
        store,
        run_id,
        candidate,
        scoring_ref,
        evidence_against=() if fail_social_tool else (social_evidence.evidence_id,),
    )
    _record_tool_run(
        store,
        run_id,
        invocation_by_name["phase4_fixture_report"],
        status="succeeded",
        artifacts=(
            (markdown_ref, "markdown_report", "markdown_report.v1"),
            (json_ref, "json_report", "json_report.v1"),
            (audit_ref, "audit_manifest", "audit_manifest.v1"),
        ),
    )

    status: Literal["succeeded", "partial"] = "partial" if fail_social_tool else "succeeded"
    warnings = (
        ("phase4_fixture_social_evidence failed; report used remaining fixture evidence.",)
        if fail_social_tool
        else ()
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=run_id,
            run_kind="phase4_fixture_tool_suite",
            objective=objective.prompt,
            status=status,
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
            metadata={"objective_id": objective.objective_id, "scaffold": True},
        )
    )

    tool_results = _tool_results(
        invocation_by_name,
        universe_ref,
        market_ref,
        news_ref,
        social_ref,
        scoring_ref,
        markdown_ref,
        json_ref,
        audit_ref,
        fail_social_tool=fail_social_tool,
    )
    evidence_import = CodexEvidenceImport(
        import_id="import-phase4-fixture-evidence",
        objective_id=objective.objective_id,
        imported_at=COMPLETED_AT,
        evidence=tuple(evidence_sources),
        artifact_refs=(market_ref, news_ref, *((social_ref,) if not fail_social_tool else ())),
        originating_invocation_id="invoke-phase4-news-catalyst",
        notes="Fixture import stands in for Phase 4 tool outputs until tool branches merge.",
    )
    summary = OrchestratorRunSummary(
        run_id=run_id,
        objective=objective,
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
        status=status,
        selected_tools=selected_tools,
        invocations=tuple(invocation_by_name.values()),
        tool_results=tool_results,
        evidence_imports=(evidence_import,),
        report_artifacts=(markdown_ref, json_ref, audit_ref),
        prediction_candidate_ids=(candidate.candidate_id,),
        warnings=warnings,
    )
    return _Phase4FixtureRun(
        summary=summary,
        report=report,
        markdown_path=markdown_path,
        json_path=json_path,
        audit_manifest_path=audit_manifest_path,
    )


def _phase4_tool_specs() -> tuple[ResearchToolSpec, ...]:
    return (
        _tool_spec(
            "phase4_fixture_universe_discovery",
            "Resolve mixed-asset fixture universe through Phase 3 contracts.",
            ("instrument_universe",),
        ),
        _tool_spec(
            "phase4_fixture_market_data",
            "Write market and sector-baseline context as provider-result artifacts.",
            ("provider_result",),
        ),
        _tool_spec(
            "phase4_fixture_news_catalyst",
            "Normalize fixture news/catalyst evidence with source provenance.",
            ("normalized_evidence",),
        ),
        _tool_spec(
            "phase4_fixture_social_evidence",
            "Normalize fixture social evidence and expose recoverable failures.",
            ("normalized_evidence",),
        ),
        _tool_spec(
            "phase4_fixture_prediction_evaluation",
            "Score evidence into a prediction candidate without trading instructions.",
            ("prediction_input",),
        ),
        _tool_spec(
            "phase4_fixture_report",
            "Render Markdown, JSON, and audit-manifest report artifacts.",
            ("markdown_report", "json_report", "audit_manifest"),
        ),
    )


def _tool_spec(
    tool_name: str,
    description: str,
    artifact_kinds: tuple[str, ...],
) -> ResearchToolSpec:
    return ResearchToolSpec(
        tool_name=tool_name,
        tool_version="phase4-scaffold.v1",
        description=description,
        input_schema={"type": "object"},
        output_contract="fixture-backed Phase 4 scaffold artifact envelope",
        artifact_kinds=artifact_kinds,
        offline_capable=True,
        deterministic=True,
        live_capable=False,
        requires_network=False,
        metadata={"phase": 4, "scaffold": True},
    )


def _phase4_invocations(objective: ResearchObjective) -> dict[str, ToolInvocation]:
    return {
        "phase4_fixture_universe_discovery": _invocation(
            objective,
            "phase4_fixture_universe_discovery",
            "invoke-phase4-universe",
            {"fixture": "mixed_asset_universe"},
        ),
        "phase4_fixture_market_data": _invocation(
            objective,
            "phase4_fixture_market_data",
            "invoke-phase4-market-data",
            {"symbols": ["TSLA", "SPY", "XLY"]},
            depends_on=("invoke-phase4-universe",),
        ),
        "phase4_fixture_news_catalyst": _invocation(
            objective,
            "phase4_fixture_news_catalyst",
            "invoke-phase4-news-catalyst",
            {"query": "TSLA delivery expectations"},
            depends_on=("invoke-phase4-universe",),
        ),
        "phase4_fixture_social_evidence": _invocation(
            objective,
            "phase4_fixture_social_evidence",
            "invoke-phase4-social-evidence",
            {"query": "TSLA valuation risk"},
            depends_on=("invoke-phase4-universe",),
        ),
        "phase4_fixture_prediction_evaluation": _invocation(
            objective,
            "phase4_fixture_prediction_evaluation",
            "invoke-phase4-prediction-evaluation",
            {"candidate_id": "candidate-phase4-tsla-weekly"},
            depends_on=(
                "invoke-phase4-market-data",
                "invoke-phase4-news-catalyst",
                "invoke-phase4-social-evidence",
            ),
        ),
        "phase4_fixture_report": _invocation(
            objective,
            "phase4_fixture_report",
            "invoke-phase4-report",
            {"formats": ["markdown", "json", "audit_manifest"]},
            depends_on=("invoke-phase4-prediction-evaluation",),
        ),
    }


def _invocation(
    objective: ResearchObjective,
    tool_name: str,
    invocation_id: str,
    arguments: dict[str, object],
    *,
    depends_on: tuple[str, ...] = (),
) -> ToolInvocation:
    return ToolInvocation(
        invocation_id=invocation_id,
        objective_id=objective.objective_id,
        tool_name=tool_name,
        tool_version="phase4-scaffold.v1",
        requested_at=STARTED_AT,
        arguments=arguments,
        mode="offline",
        depends_on_invocation_ids=depends_on,
        metadata={"scaffold": True},
    )


def _load_universe_fixture() -> InstrumentUniverse:
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "tools"
        / "universe_discovery"
        / "mixed_asset_universe.json"
    )
    return InstrumentUniverse.model_validate_json(fixture_path.read_text(encoding="utf-8"))


def _instrument_record(instrument: Instrument) -> InstrumentRecord:
    return InstrumentRecord(
        instrument_id=instrument.instrument_id,
        symbol=instrument.symbol,
        asset_class=instrument.asset_class.value,
        name=instrument.display_name,
        venue=instrument.venue,
        aliases=instrument.aliases,
        provider_ids=tuple(
            provider.model_dump(mode="json") for provider in instrument.provider_ids
        ),
        related_instruments=tuple(
            related.model_dump(mode="json") for related in instrument.related_instruments
        ),
        tradability_evidence=tuple(
            evidence.model_dump(mode="json") for evidence in instrument.tradability_evidence
        ),
        data_availability=tuple(
            availability.model_dump(mode="json") for availability in instrument.data_availability
        ),
        metadata=instrument.metadata,
    )


def _market_context_evidence() -> SourceEvidence:
    text = "Fixture market context notes XLY was used as a sector proxy for TSLA."
    return SourceEvidence(
        evidence_id="fixture-evidence-sector-xly",
        source_kind=SourceKind.MARKET_DATA,
        ticker="XLY",
        title="Fixture sector baseline",
        text=text,
        created_at=STARTED_AT,
        permalink="https://example.test/market/xly-sector-baseline",
        instrument_id="instrument:etf:us:xly",
        matched_tickers=("TSLA", "XLY"),
        matched_instrument_ids=("instrument:equity:us:tsla", "instrument:etf:us:xly"),
        provenance=_provenance(
            provider_name="phase4-fixture-market",
            source_kind=SourceKind.MARKET_DATA,
            source_url="https://example.test/market/xly-sector-baseline",
            raw_identifier="fixture-market-xly-sector-baseline",
            raw_snapshot_id="artifact-phase4-market-context",
            query="TSLA sector baseline",
        ),
        metadata={"stance": "baseline", "scaffold": True},
    )


def _news_evidence() -> SourceEvidence:
    text = (
        "Fixture news says TSLA delivery expectations improved while analysts still noted "
        "execution risk."
    )
    return SourceEvidence(
        evidence_id="phase4-news-tsla-deliveries",
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="TSLA",
        title="Fixture TSLA delivery expectations",
        text=text,
        created_at=STARTED_AT,
        permalink="https://example.test/news/tsla-delivery-expectations",
        instrument_id="instrument:equity:us:tsla",
        matched_tickers=("TSLA",),
        matched_instrument_ids=("instrument:equity:us:tsla",),
        provenance=_provenance(
            provider_name="phase4-fixture-news",
            source_kind=SourceKind.NEWS_ARTICLE,
            source_url="https://example.test/news/tsla-delivery-expectations",
            raw_identifier="fixture-news-tsla-deliveries",
            raw_snapshot_id="artifact-phase4-news-evidence",
            query="TSLA delivery expectations",
        ),
        metadata={"stance": "supports", "scaffold": True},
    )


def _social_evidence() -> SourceEvidence:
    text = "Fixture discussion says TSLA enthusiasm is high but valuation concerns remain visible."
    return SourceEvidence(
        evidence_id="phase4-social-tsla-valuation-risk",
        source_kind=SourceKind.REDDIT_POST,
        ticker="TSLA",
        title="Fixture TSLA discussion risk",
        text=text,
        created_at=STARTED_AT,
        permalink="https://example.test/reddit/tsla-valuation-risk",
        instrument_id="instrument:equity:us:tsla",
        matched_tickers=("TSLA",),
        matched_instrument_ids=("instrument:equity:us:tsla",),
        provenance=_provenance(
            provider_name="phase4-fixture-social",
            source_kind=SourceKind.REDDIT_POST,
            source_url="https://example.test/reddit/tsla-valuation-risk",
            raw_identifier="fixture-social-tsla-valuation-risk",
            raw_snapshot_id="artifact-phase4-social-evidence",
            query="TSLA valuation risk",
        ),
        metadata={"stance": "contradicts", "scaffold": True},
    )


def _provenance(
    *,
    provider_name: str,
    source_kind: SourceKind,
    source_url: str,
    raw_identifier: str,
    raw_snapshot_id: str,
    query: str,
) -> SourceProvenance:
    return SourceProvenance(
        provider_name=provider_name,
        source_kind=source_kind,
        retrieval_method=RetrievalMethod.FIXTURE,
        fetched_at=STARTED_AT,
        observed_at=STARTED_AT,
        source_url=source_url,
        permalink=source_url,
        raw_identifier=raw_identifier,
        raw_snapshot_id=raw_snapshot_id,
        query=query,
        freshness_status=FreshnessStatus.FRESH,
        provider_metadata={"fixture": True, "phase": 4},
    )


def _prediction_candidate(
    *,
    evidence_against: tuple[EvidenceReference, ...],
) -> PredictionCandidate:
    status = (
        PredictionStatus.CONTRADICTED if evidence_against else PredictionStatus.EVIDENCE_SUPPORTED
    )
    direction = Direction.MIXED if evidence_against else Direction.BULLISH
    return PredictionCandidate(
        candidate_id="candidate-phase4-tsla-weekly",
        instrument_id="instrument:equity:us:tsla",
        symbol="TSLA",
        horizon=TimeHorizon.WEEKLY,
        direction=direction,
        status=status,
        thesis=(
            "TSLA has fixture-backed upside catalyst evidence, with valuation risk tracked "
            "as dissenting context."
        ),
        baseline="SPY and XLY fixture context provide broad and sector baselines.",
        confidence=0.52,
        evidence_for=(_news_evidence_ref(),),
        evidence_against=evidence_against,
        assumptions=("Fixture artifacts are deterministic and offline.",),
        uncertainties=("Live provider branches may change scoring weights after integration.",),
        signal_artifact_ids=("artifact-phase4-market-context", "artifact-phase4-prediction-input"),
        metadata={"scaffold": True, "phase": 4},
    )


def _news_evidence_ref() -> EvidenceReference:
    return EvidenceReference(
        evidence_id="phase4-news-tsla-deliveries",
        quote="delivery expectations improved",
        relevance=0.82,
    )


def _social_evidence_ref() -> EvidenceReference:
    return EvidenceReference(
        evidence_id="phase4-social-tsla-valuation-risk",
        quote="valuation concerns remain visible",
        relevance=0.74,
    )


def _daily_report(
    *,
    run_id: str,
    objective: ResearchObjective,
    universe: InstrumentUniverse,
    evidence: tuple[SourceEvidence, ...],
    candidate: PredictionCandidate,
    audit_manifest: AuditManifest,
    partial: bool,
) -> DailyReport:
    sections = tuple(
        InstrumentReportSection(
            instrument_id=instrument.instrument_id,
            symbol=instrument.symbol,
            display_name=instrument.display_name,
            observed_discussion_summary=(
                "Fixture discussion includes both catalyst and dissenting context."
                if instrument.symbol == "TSLA"
                else "No instrument-specific fixture discussion in this scaffold."
            ),
            social_news_summary=(
                "News supports the TSLA catalyst; social evidence is "
                f"{'unavailable' if partial else 'dissenting'}."
                if instrument.symbol == "TSLA"
                else "Used for universe, market, or macro context."
            ),
            analysis_summary=(
                "Phase 4 scaffold combines universe, market, news, social, scoring, and report "
                "artifacts."
            ),
            prediction_candidate_ids=_section_candidate_ids(instrument, candidate),
            evidence=(
                (_news_evidence_ref(),)
                if instrument.instrument_id == candidate.instrument_id
                else ()
            ),
            data_quality={"fixture": True, "partial": partial},
        )
        for instrument in universe.instruments
    )
    return DailyReport(
        schema_version="daily-report.phase4-fixture-scaffold.v1",
        run_id=run_id,
        report_date=RUN_DATE,
        generated_at=COMPLETED_AT,
        timezone="UTC",
        objective=objective.prompt,
        universe="Mixed-asset Phase 4 fixture scaffold.",
        instruments=universe.instruments,
        data_freshness=DataFreshnessSummary(
            as_of=STARTED_AT,
            summary="Fixture data is deterministic and offline.",
            missing_provider_names=("phase4-fixture-social",) if partial else (),
        ),
        provider_health=(
            ProviderHealth(
                provider_name="phase4-fixture-suite",
                status=ProviderStatus.PARTIAL if partial else ProviderStatus.OK,
                checked_at=COMPLETED_AT,
            ),
        ),
        evidence_sources=evidence,
        instrument_resolutions=universe.resolutions,
        instrument_sections=sections,
        prediction_candidates=(candidate,),
        audit_manifest=audit_manifest,
    )


def _section_candidate_ids(
    instrument: Instrument,
    candidate: PredictionCandidate,
) -> tuple[str, ...]:
    if instrument.instrument_id != candidate.instrument_id:
        return ()
    return (candidate.candidate_id,)


def _write_json_artifact(
    repo_root: Path,
    artifact_root: Path,
    reference_id: str,
    reference_type: _ReferenceType,
    payload: object,
) -> DataReference:
    path = artifact_root / f"{reference_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return _reference_from_path(repo_root, path, reference_id, reference_type)


def _reference_from_path(
    repo_root: Path,
    path: Path,
    reference_id: str,
    reference_type: _ReferenceType,
) -> DataReference:
    return DataReference(
        reference_id=reference_id,
        reference_type=reference_type,
        path=path.relative_to(repo_root).as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata={"fixture": True, "phase": 4},
    )


def _planned_reference(
    reference_id: str,
    reference_type: _ReferenceType,
    path: str,
) -> DataReference:
    return DataReference(
        reference_id=reference_id,
        reference_type=reference_type,
        path=path,
        metadata={"fixture": True, "phase": 4, "planned": True},
    )


def _optional_artifact_tuple(
    social_ref: DataReference,
    fail_social_tool: bool,
) -> tuple[tuple[DataReference, str], ...]:
    if fail_social_tool:
        return ()
    return ((social_ref, "normalized_evidence"),)


def _audit_manifest(
    run_id: str,
    artifacts: tuple[tuple[DataReference, str], ...],
) -> AuditManifest:
    return AuditManifest(
        run_id=run_id,
        schema_version="audit-manifest.phase4-scaffold.v1",
        created_at=COMPLETED_AT,
        artifacts=tuple(
            AuditArtifact(
                artifact_id=reference.reference_id,
                artifact_type=artifact_type,
                path=reference.path or "",
                created_at=COMPLETED_AT,
                produced_by="phase4-fixture-tool-suite",
                sha256=reference.sha256,
                metadata={"scaffold": True},
            )
            for reference, artifact_type in artifacts
        ),
        provider_run_ids=("phase4-fixture-suite",),
        prompt_versions={"scaffold": "phase4-scaffold.v1"},
        command_args={"offline": True, "run_date": RUN_DATE.isoformat()},
        prediction_trace_ids=("candidate-phase4-tsla-weekly",),
    )


def _record_tool_run(
    store: SQLiteStore,
    run_id: str,
    invocation: ToolInvocation,
    *,
    status: Literal["succeeded", "failed"],
    artifacts: tuple[tuple[DataReference, str, str], ...] = (),
    evidence: tuple[SourceEvidence, ...] = (),
    error_message: str | None = None,
) -> None:
    tool_run_id = invocation.invocation_id.replace("invoke", "tool")
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=invocation.tool_name,
            tool_version=invocation.tool_version,
            status=status,
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
            inputs=invocation.arguments,
            error_message=error_message,
        )
    )
    for reference, artifact_type, schema_version in artifacts:
        _record_artifact(store, reference, tool_run_id, artifact_type, schema_version)
    artifact_id = artifacts[0][0].reference_id if artifacts else None
    for source in evidence:
        _record_source_query(store, source, tool_run_id)
        store.record_evidence(_evidence_record(source, tool_run_id, artifact_id))


def _record_artifact(
    store: SQLiteStore,
    reference: DataReference,
    tool_run_id: str,
    artifact_type: str,
    schema_version: str,
) -> None:
    assert reference.path is not None
    assert reference.sha256 is not None
    store.record_artifact(
        ArtifactRecord(
            artifact_id=reference.reference_id,
            tool_run_id=tool_run_id,
            artifact_type=artifact_type,
            path=Path(reference.path),
            sha256=reference.sha256,
            schema_version=schema_version,
            metadata=reference.metadata,
            created_at=COMPLETED_AT,
        )
    )


def _record_source_query(
    store: SQLiteStore,
    source: SourceEvidence,
    tool_run_id: str,
) -> None:
    query_id = f"query-{source.evidence_id}"
    store.record_source_query(
        SourceQueryRecord(
            source_query_id=query_id,
            tool_run_id=tool_run_id,
            provider=source.provenance.provider_name,
            query=source.provenance.query or source.evidence_id,
            url=source.provenance.source_url,
            retrieved_at=source.provenance.fetched_at,
            metadata={"fixture": True},
        )
    )


def _evidence_record(
    source: SourceEvidence,
    tool_run_id: str,
    artifact_id: str | None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=source.evidence_id,
        source_type=source.source_kind.value,
        provider=source.provenance.provider_name,
        retrieved_at=source.provenance.fetched_at,
        claim=source.text,
        tool_run_id=tool_run_id,
        source_query_id=f"query-{source.evidence_id}",
        url=source.provenance.source_url,
        query=source.provenance.query,
        published_at=source.created_at,
        instruments=source.matched_instrument_ids,
        extraction_confidence=0.8,
        source_reliability="fixture",
        freshness_status=source.provenance.freshness_status.value,
        artifact_id=artifact_id,
        raw_excerpt=source.text[:80],
        provenance_json=source.provenance.model_dump(mode="json"),
        metadata=source.metadata,
    )


def _record_candidate(
    store: SQLiteStore,
    run_id: str,
    candidate: PredictionCandidate,
    scoring_ref: DataReference,
    *,
    evidence_against: tuple[str, ...],
) -> None:
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id=candidate.candidate_id,
            run_id=run_id,
            instrument_id=candidate.instrument_id,
            prediction_horizon=candidate.horizon.value,
            prediction_type="scenario",
            scenario=candidate.thesis,
            status=candidate.status.value,
            confidence=candidate.confidence,
            direction=candidate.direction.value,
            evidence_for=tuple(reference.evidence_id for reference in candidate.evidence_for),
            evidence_against=evidence_against,
            signal_artifacts=candidate.signal_artifact_ids,
            baseline={"summary": candidate.baseline},
            uncertainty="; ".join(candidate.uncertainties),
            metadata=candidate.metadata,
        )
    )
    for reference in candidate.evidence_for:
        store.link_candidate_evidence(
            CandidateEvidenceLinkRecord(
                candidate_id=candidate.candidate_id,
                evidence_id=reference.evidence_id,
                relationship="supports",
                created_at=COMPLETED_AT,
            )
        )
    for evidence_id in evidence_against:
        store.link_candidate_evidence(
            CandidateEvidenceLinkRecord(
                candidate_id=candidate.candidate_id,
                evidence_id=evidence_id,
                relationship="contradicts",
                created_at=COMPLETED_AT,
            )
        )
    store.link_candidate_artifact(
        CandidateArtifactLinkRecord(
            candidate_id=candidate.candidate_id,
            artifact_id=scoring_ref.reference_id,
            relationship="scoring_input",
            created_at=COMPLETED_AT,
        )
    )


def _tool_results(
    invocation_by_name: dict[str, ToolInvocation],
    universe_ref: DataReference,
    market_ref: DataReference,
    news_ref: DataReference,
    social_ref: DataReference,
    scoring_ref: DataReference,
    markdown_ref: DataReference,
    json_ref: DataReference,
    audit_ref: DataReference,
    *,
    fail_social_tool: bool,
) -> tuple[ToolExecutionResult, ...]:
    social_result = (
        ToolExecutionResult(
            result_id="result-phase4-social-evidence-failed",
            invocation=invocation_by_name["phase4_fixture_social_evidence"],
            status="failed",
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
            error_message="fixture social evidence provider unavailable",
        )
        if fail_social_tool
        else _tool_result(
            "result-phase4-social-evidence",
            invocation_by_name["phase4_fixture_social_evidence"],
            (social_ref,),
            ("phase4-social-tsla-valuation-risk",),
        )
    )
    return (
        _tool_result(
            "result-phase4-universe",
            invocation_by_name["phase4_fixture_universe_discovery"],
            (universe_ref,),
            (),
        ),
        _tool_result(
            "result-phase4-market-data",
            invocation_by_name["phase4_fixture_market_data"],
            (market_ref,),
            ("fixture-evidence-sector-xly",),
        ),
        _tool_result(
            "result-phase4-news-catalyst",
            invocation_by_name["phase4_fixture_news_catalyst"],
            (news_ref,),
            ("phase4-news-tsla-deliveries",),
        ),
        social_result,
        _tool_result(
            "result-phase4-prediction-evaluation",
            invocation_by_name["phase4_fixture_prediction_evaluation"],
            (scoring_ref,),
            (),
        ),
        _tool_result(
            "result-phase4-report",
            invocation_by_name["phase4_fixture_report"],
            (markdown_ref, json_ref, audit_ref),
            (),
        ),
    )


def _tool_result(
    result_id: str,
    invocation: ToolInvocation,
    artifact_refs: tuple[DataReference, ...],
    evidence_ids: tuple[str, ...],
) -> ToolExecutionResult:
    return ToolExecutionResult(
        result_id=result_id,
        invocation=invocation,
        status="succeeded",
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
        artifact_refs=artifact_refs,
        evidence_ids=evidence_ids,
    )
