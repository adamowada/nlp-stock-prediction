from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from nlp_stock_prediction.contracts import (
    AssetClass,
    FreshnessStatus,
    Instrument,
    InstrumentResolution,
    InstrumentResolutionStatus,
    InstrumentUniverse,
    RetrievalMethod,
    SourceEvidence,
    SourceKind,
    SourceProvenance,
)
from nlp_stock_prediction.orchestration import Phase4Service
from nlp_stock_prediction.orchestration.phase4_universe_discovery import (
    phase4_universe_artifact_payload,
)
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ToolRunRecord,
)

RUN_DATE = date(2026, 5, 13)
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_report_assembly_traces_candidates_to_stored_evidence_and_artifacts(
    tmp_path: Path,
) -> None:
    service = Phase4Service(
        repo_root=tmp_path,
        fixture_root=REPO_ROOT,
        database_path=Path("data") / "prediction-research.sqlite3",
    )

    result = service.run_offline_phase4_flow(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/phase5-report-assembly",
        symbol="TSLA",
    )

    report = cast(dict[str, object], result["report"])
    payload = _read_report_payload(report)
    candidates = cast(list[dict[str, Any]], payload["prediction_candidates"])
    candidate = candidates[0]
    candidate_id = cast(str, candidate["candidate_id"])
    evidence_for = cast(list[dict[str, Any]], candidate["evidence_for"])
    candidate_metadata = cast(dict[str, Any], candidate["metadata"])
    evaluation_metadata = cast(dict[str, Any], candidate_metadata["prediction_evaluation"])
    evaluation_artifact_id = cast(str, evaluation_metadata["artifact_id"])

    audit_manifest = cast(dict[str, Any], payload["audit_manifest"])
    audit_artifacts = cast(list[dict[str, Any]], audit_manifest["artifacts"])
    artifacts = {artifact["artifact_id"]: artifact for artifact in audit_artifacts}
    assert all(artifact["path"] for artifact in artifacts.values())
    assert all(artifact["sha256"] for artifact in artifacts.values())
    assert artifacts[evaluation_artifact_id]["artifact_type"] == "prediction_evaluation"

    source_references = cast(list[dict[str, Any]], payload["source_references"])
    assert any(
        reference["reference_type"] == "prediction_evaluation"
        and evaluation_artifact_id in reference["artifact_ids"]
        and candidate_id in reference["candidate_ids"]
        for reference in source_references
    )
    assert any(
        reference["reference_type"] == "source_evidence"
        and evidence_for[0]["evidence_id"] in reference["evidence_ids"]
        and reference["candidate_ids"] == [candidate_id]
        for reference in source_references
    )

    traces = {
        trace["claim_id"]: trace
        for trace in cast(list[dict[str, Any]], payload["material_claim_traces"])
    }
    prediction_input_ids = [
        artifact_id
        for artifact_id, artifact in artifacts.items()
        if artifact["artifact_type"] == "prediction_input"
    ]
    assert traces[f"claim-{candidate_id}-baseline"]["claim_type"] == "baseline"
    assert evaluation_artifact_id in traces[f"claim-{candidate_id}-evaluation"]["artifact_ids"]
    assert any(
        artifact_id in traces[f"claim-{candidate_id}"]["artifact_ids"]
        for artifact_id in prediction_input_ids
    )


@pytest.mark.integration
def test_report_assembly_missing_candidate_artifact_becomes_insufficient_evidence(
    tmp_path: Path,
) -> None:
    service, run_id = _service_with_started_run(tmp_path)
    service.store.upsert_prediction_candidate(
        _candidate(signal_artifacts=("artifact-missing-technical",))
    )

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    assert rendered["status"] == "empty"
    assert rendered["candidate_count"] == 0
    assert rendered["stored_candidate_count"] == 1
    assert rendered["excluded_candidate_ids"] == ["candidate-stage3-assembly"]
    rendered_warnings = cast(list[str], rendered["warnings"])
    assert any("Missing required artifact" in warning for warning in rendered_warnings)

    payload = _read_report_payload(rendered)
    insufficient = cast(dict[str, Any], payload["insufficient_evidence"])
    assert payload["prediction_candidates"] == []
    assert any(
        "Missing required artifact artifact-missing-technical" in reason
        for reason in insufficient["blocking_reasons"]
    )
    assert "required tool artifacts" in insufficient["missing_evidence_types"]
    assert insufficient["metadata"]["missing_artifact_ids"] == ["artifact-missing-technical"]
    assert "report-assembly" in insufficient["provider_names"]
    provider_health = cast(list[dict[str, Any]], payload["provider_health"])
    assert any(
        health["provider_name"] == "report-assembly" and health["status"] == "malformed"
        for health in provider_health
    )
    audit_health = cast(list[dict[str, Any]], payload["audit_manifest"]["provider_health"])
    assert any(
        health["provider_name"] == "report-assembly" and health["status"] == "malformed"
        for health in audit_health
    )


@pytest.mark.integration
def test_report_assembly_malformed_required_artifact_becomes_insufficient_evidence(
    tmp_path: Path,
) -> None:
    service, run_id = _service_with_started_run(tmp_path)
    artifact_id = "artifact-bad-technical"
    artifact_path = _write_bad_technical_artifact(service=service, run_id=run_id, tmp_path=tmp_path)
    service.store.upsert_prediction_candidate(_candidate(signal_artifacts=(artifact_id,)))
    service.store.record_artifact(
        ArtifactRecord(
            artifact_id=artifact_id,
            artifact_type="technical_package",
            path=artifact_path.relative_to(tmp_path),
            sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            schema_version="phase4.technical-package.v1",
            tool_run_id="tool-bad-technical",
            produced_by="phase4_technical_package",
            record_count=1,
            metadata={"symbol": "TSLA"},
            created_at=NOW,
        )
    )

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    assert rendered["status"] == "empty"
    assert rendered["candidate_count"] == 0
    assert rendered["excluded_candidate_ids"] == ["candidate-stage3-assembly"]
    rendered_warnings = cast(list[str], rendered["warnings"])
    assert any("Artifact payload is malformed" in warning for warning in rendered_warnings)

    payload = _read_report_payload(rendered)
    audit_manifest = cast(dict[str, Any], payload["audit_manifest"])
    audit_artifacts = cast(list[dict[str, Any]], audit_manifest["artifacts"])
    artifact = next(
        artifact for artifact in audit_artifacts if artifact["artifact_id"] == artifact_id
    )
    assert artifact["metadata"]["assembly_required"] is True
    assert artifact["metadata"]["assembly_status"] == "warning"
    insufficient = cast(dict[str, Any], payload["insufficient_evidence"])
    assert any(
        "Artifact payload is malformed" in reason for reason in insufficient["blocking_reasons"]
    )
    assert artifact_id in insufficient["artifact_ids"]
    assert "complete provider outputs" in insufficient["missing_evidence_types"]


@pytest.mark.integration
def test_report_assembly_failed_provider_run_is_visible_in_report_and_audit_manifest(
    tmp_path: Path,
) -> None:
    service, run_id = _service_with_started_run(tmp_path)
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-stage8-malformed-page",
            run_id=run_id,
            tool_name="phase4_news_catalyst",
            tool_version="phase4.news.v1",
            status="failed",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "TSLA"},
            warnings=("Malformed provider page returned no readable article text.",),
            error_message="Malformed provider page returned no readable article text.",
        )
    )

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    payload = _read_report_payload(rendered)
    provider_health = cast(list[dict[str, Any]], payload["provider_health"])
    assert any(
        health["provider_name"] == "tool:phase4_news_catalyst" and health["status"] == "failed"
        for health in provider_health
    )
    audit_health = cast(list[dict[str, Any]], payload["audit_manifest"]["provider_health"])
    assert any(
        health["provider_name"] == "tool:phase4_news_catalyst" and health["status"] == "failed"
        for health in audit_health
    )
    insufficient = cast(dict[str, Any], payload["insufficient_evidence"])
    assert "complete provider outputs" in insufficient["missing_evidence_types"]
    markdown = Path(str(rendered["markdown_path"])).read_text(encoding="utf-8")
    assert "Malformed provider page returned no readable article text." in markdown


@pytest.mark.integration
def test_report_assembly_stale_evidence_renders_structured_insufficient_evidence(
    tmp_path: Path,
) -> None:
    service, run_id = _service_with_started_run(tmp_path)
    evidence = _source_evidence(
        "evidence-stage8-stale",
        text="A stale source claims Tesla demand was improving last quarter.",
        freshness_status=FreshnessStatus.STALE,
    )
    _record_evidence(service, run_id=run_id, evidence=evidence)

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    payload = _read_report_payload(rendered)
    insufficient = cast(dict[str, Any], payload["insufficient_evidence"])
    assert payload["prediction_candidates"] == []
    assert insufficient["evidence"][0]["evidence_id"] == evidence.evidence_id
    assert "fresh source evidence" in insufficient["missing_evidence_types"]
    assert payload["data_freshness"]["stale_provider_names"] == ["fixture-news"]
    markdown = Path(str(rendered["markdown_path"])).read_text(encoding="utf-8")
    assert "`evidence-stage8-stale` news_article via fixture-news" in markdown
    assert "Report-authored scenario" not in markdown


@pytest.mark.integration
def test_report_assembly_surfaces_ambiguous_and_unsupported_instrument_resolutions(
    tmp_path: Path,
) -> None:
    service, run_id = _service_with_started_run(tmp_path)
    artifact_path = _write_universe_resolution_artifact(
        service=service,
        run_id=run_id,
        tmp_path=tmp_path,
    )
    service.store.record_artifact(
        ArtifactRecord(
            artifact_id="artifact-stage8-universe-resolution",
            artifact_type="instrument_universe",
            path=artifact_path.relative_to(tmp_path),
            sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            schema_version="phase4.instrument-universe.v1",
            tool_run_id="tool-stage8-universe",
            produced_by="phase4_universe_discovery",
            record_count=2,
            metadata={"symbol": "TSLA"},
            created_at=NOW,
        )
    )

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    payload = _read_report_payload(rendered)
    statuses = {
        resolution["query"]: resolution["status"]
        for resolution in cast(list[dict[str, Any]], payload["instrument_resolutions"])
    }
    assert statuses["AI"] == "ambiguous"
    assert statuses["OTC:MISSING"] == "unsupported"
    insufficient = cast(dict[str, Any], payload["insufficient_evidence"])
    assert "resolved supported instrument identity" in insufficient["missing_evidence_types"]
    assert any(
        "Instrument query AI is ambiguous" in reason for reason in insufficient["blocking_reasons"]
    )
    markdown = Path(str(rendered["markdown_path"])).read_text(encoding="utf-8")
    assert "## Universe Resolution" in markdown
    assert "`AI`: ambiguous" in markdown
    assert "`OTC:MISSING`: unsupported" in markdown


@pytest.mark.integration
def test_report_assembly_preserves_contradictory_evidence_as_contradicted_candidate(
    tmp_path: Path,
) -> None:
    service, run_id = _service_with_started_run(tmp_path)
    support = _source_evidence(
        "evidence-stage8-support",
        text="A cited source claims Tesla deliveries improved sequentially.",
        freshness_status=FreshnessStatus.FRESH,
    )
    conflict = _source_evidence(
        "evidence-stage8-conflict",
        text="A cited source claims Tesla margin pressure remains elevated.",
        freshness_status=FreshnessStatus.FRESH,
    )
    _record_evidence(service, run_id=run_id, evidence=support)
    _record_evidence(service, run_id=run_id, evidence=conflict)
    service.store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-stage8-contradiction",
            run_id=run_id,
            instrument_id="instrument:equity:us:tsla",
            prediction_horizon="swing",
            prediction_type="directional",
            scenario="TSLA has mixed source evidence and should remain a contested scenario.",
            status="evidence_supported",
            confidence=0.52,
            direction="mixed",
            evidence_for=(support.evidence_id,),
            evidence_against=(conflict.evidence_id,),
            baseline={"summary": "No directional edge is assumed without source-backed evidence."},
            uncertainty="Contradictory source evidence limits confidence.",
            metadata={"symbol": "TSLA"},
        )
    )

    rendered = service.render_prediction_report(run_id=run_id, symbol="TSLA")

    payload = _read_report_payload(rendered)
    candidate = cast(list[dict[str, Any]], payload["prediction_candidates"])[0]
    assert candidate["status"] == "contradicted"
    assert candidate["evidence_against"][0]["evidence_id"] == conflict.evidence_id
    assert candidate["dissenting_evidence"][0]["impact"] == "contradicts"
    assert (
        "Source evidence is observed material, not automatically true." in candidate["assumptions"]
    )
    markdown = Path(str(rendered["markdown_path"])).read_text(encoding="utf-8")
    assert "Status: contradicted" in markdown
    assert "Evidence against: `evidence-stage8-conflict`" in markdown
    assert "Dissenting evidence: contradicts" in markdown


def _service_with_started_run(tmp_path: Path) -> tuple[Phase4Service, str]:
    service = Phase4Service(repo_root=tmp_path)
    started = service.start_research_run(
        run_date=RUN_DATE.isoformat(),
        output_dir="reports/phase5-report-assembly",
        symbol="TSLA",
        objective="Stage 3 report assembly test.",
    )
    service.store.upsert_instrument(
        InstrumentRecord(
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            asset_class="stock",
            name="Tesla Inc.",
        )
    )
    return service, str(started["run_id"])


def _candidate(*, signal_artifacts: tuple[str, ...]) -> PredictionCandidateRecord:
    return PredictionCandidateRecord(
        candidate_id="candidate-stage3-assembly",
        run_id=f"phase4-{RUN_DATE.isoformat()}-tsla",
        instrument_id="instrument:equity:us:tsla",
        prediction_horizon="swing",
        prediction_type="directional",
        scenario="TSLA report assembly candidate depends on stored artifact integrity.",
        status="insufficient_evidence",
        confidence=0.18,
        direction="mixed",
        signal_artifacts=signal_artifacts,
        baseline={"summary": "No directional edge is assumed without source-backed evidence."},
        uncertainty="Artifact availability and schema validity limit confidence.",
        metadata={"symbol": "TSLA"},
    )


def _write_bad_technical_artifact(
    *,
    service: Phase4Service,
    run_id: str,
    tmp_path: Path,
) -> Path:
    run = service.store.get_research_run(run_id)
    assert run is not None
    audit_dir = Path(str(run.metadata["output_dir"]))
    if not audit_dir.is_absolute():
        audit_dir = tmp_path / audit_dir
    artifact_path = (
        audit_dir / RUN_DATE.isoformat() / "tsla" / "audit" / "technical-package" / "bad.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("{not-json", encoding="utf-8")
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-bad-technical",
            run_id=run_id,
            tool_name="phase4_technical_package",
            tool_version="phase4.technical-package.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "TSLA"},
        )
    )
    return artifact_path


def _source_evidence(
    evidence_id: str,
    *,
    text: str,
    freshness_status: FreshnessStatus,
) -> SourceEvidence:
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="TSLA",
        text=text,
        created_at=NOW,
        permalink=f"https://example.test/{evidence_id}",
        matched_tickers=("TSLA",),
        matched_instrument_ids=("instrument:equity:us:tsla",),
        instrument_id="instrument:equity:us:tsla",
        provenance=SourceProvenance(
            provider_name="fixture-news",
            source_kind=SourceKind.NEWS_ARTICLE,
            retrieval_method=RetrievalMethod.FIXTURE,
            fetched_at=NOW,
            observed_at=NOW,
            source_url=f"https://example.test/{evidence_id}",
            permalink=f"https://example.test/{evidence_id}",
            raw_identifier=evidence_id,
            raw_snapshot_id=f"raw-{evidence_id}",
            freshness_status=freshness_status,
        ),
    )


def _record_evidence(
    service: Phase4Service,
    *,
    run_id: str,
    evidence: SourceEvidence,
) -> None:
    tool_run_id = f"tool-{evidence.evidence_id}"
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name="stage8_test_evidence",
            tool_version="test.v1",
            status="successful",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "TSLA"},
        )
    )
    service.store.record_evidence(
        EvidenceRecord(
            evidence_id=evidence.evidence_id,
            tool_run_id=tool_run_id,
            source_type=evidence.source_kind.value,
            provider=evidence.provenance.provider_name,
            retrieved_at=evidence.provenance.fetched_at,
            published_at=evidence.created_at,
            instruments=evidence.matched_instrument_ids,
            claim=evidence.text,
            url=evidence.provenance.source_url,
            freshness_status=evidence.provenance.freshness_status.value,
            metadata={"source_evidence": evidence.model_dump(mode="json")},
        )
    )


def _write_universe_resolution_artifact(
    *,
    service: Phase4Service,
    run_id: str,
    tmp_path: Path,
) -> Path:
    run = service.store.get_research_run(run_id)
    assert run is not None
    output_dir = Path(str(run.metadata["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = tmp_path / output_dir
    artifact_path = output_dir / RUN_DATE.isoformat() / "tsla" / "audit" / "universe.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    ai_equity = Instrument(
        instrument_id="instrument:equity:us:ai",
        symbol="AI",
        display_name="C3.ai Inc.",
        asset_class=AssetClass.STOCK,
    )
    ai_crypto = Instrument(
        instrument_id="instrument:crypto:ai-token",
        symbol="AI",
        display_name="AI Token",
        asset_class=AssetClass.CRYPTO,
    )
    universe = InstrumentUniverse(
        request_id="stage8-universe-resolution",
        generated_at=NOW,
        resolutions=(
            InstrumentResolution(
                query="AI",
                status=InstrumentResolutionStatus.AMBIGUOUS,
                matches=(ai_equity, ai_crypto),
                warnings=("AI maps to multiple instruments; no default selected.",),
            ),
            InstrumentResolution(
                query="OTC:MISSING",
                status=InstrumentResolutionStatus.UNSUPPORTED,
                warnings=("OTC fixture symbol is unsupported for this report.",),
            ),
        ),
        instruments=(ai_equity, ai_crypto),
    )
    payload = phase4_universe_artifact_payload(run_id=run_id, universe=universe)
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    service.store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-stage8-universe",
            run_id=run_id,
            tool_name="phase4_universe_discovery",
            tool_version="phase4.universe.v1",
            status="partial",
            started_at=NOW,
            completed_at=NOW,
            inputs={"symbol": "TSLA"},
            warnings=("Ambiguous and unsupported universe inputs were retained.",),
        )
    )
    return artifact_path


def _read_report_payload(report: Mapping[str, object]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path(str(report["json_path"])).read_text(encoding="utf-8")),
    )
