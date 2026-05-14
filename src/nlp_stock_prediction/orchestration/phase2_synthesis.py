"""Prediction candidate synthesis for Phase 2 smoke runs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import Direction, TimeHorizon
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase2_common import (
    Phase2RunPaths,
    stable_digest,
    symbol_slug,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase2_evidence import evidence_stance_from_record
from nlp_stock_prediction.storage.records import (
    CandidateArtifactLinkRecord,
    CandidateEvidenceLinkRecord,
    PredictionCandidateRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore


def synthesize_prediction_candidates(
    *,
    store: SQLiteStore,
    repo_root: Path,
    paths: Phase2RunPaths,
    run_id: str,
    symbol: str,
    ensure_instrument: Callable[[], object],
) -> JsonObject:
    now = utc_now()
    evidence = store.list_evidence_for_run(run_id)
    instrument_id = f"instrument:codex:{symbol.upper()}"
    if not store.get_instrument(instrument_id):
        ensure_instrument()
    evidence_for = tuple(
        record.evidence_id
        for record in evidence
        if evidence_stance_from_record(record) == "supports"
    )
    evidence_against = tuple(
        record.evidence_id
        for record in evidence
        if evidence_stance_from_record(record) == "contradicts"
    )
    evidence_ids = evidence_for + evidence_against
    candidate_id = f"candidate-{symbol_slug(symbol)}-{stable_digest(run_id)[:8]}"
    scenario = _candidate_scenario(
        symbol=symbol,
        evidence_for=bool(evidence_for),
        evidence_against=bool(evidence_against),
    )
    status = (
        "contradicted"
        if evidence_against
        else "moderate_confidence"
        if evidence_for
        else "insufficient_evidence"
    )
    confidence: float | None = (
        0.22 if evidence_against and not evidence_for else 0.28 if evidence_against else 0.36
    )
    if not evidence_ids:
        confidence = None
    candidate = PredictionCandidateRecord(
        candidate_id=candidate_id,
        run_id=run_id,
        instrument_id=instrument_id,
        prediction_horizon=TimeHorizon.SWING.value,
        prediction_type="scenario",
        scenario=scenario,
        direction=Direction.MIXED.value,
        confidence=confidence,
        status=status,
        evidence_for=evidence_for[:3],
        evidence_against=evidence_against[:3],
        baseline={"comparison": "no directional edge"},
        uncertainty="Dummy tools are structural validation only.",
        metadata={"phase2_mcp": True, "symbol": symbol.upper()},
    )
    payload: JsonObject = {
        "schema_version": "phase2-candidate-synthesis.v1",
        "run_id": run_id,
        "records": [
            {
                "candidate_id": candidate.candidate_id,
                "scenario": candidate.scenario,
                "evidence_for": list(candidate.evidence_for),
                "evidence_against": list(candidate.evidence_against),
            }
        ],
    }
    artifact_id = f"artifact-prediction-inputs-{stable_digest(run_id)}"
    tool_run_id = f"tool-candidate-synthesis-{run_id}"
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name="synthesize_prediction_candidates",
            tool_version="phase2.v1",
            status="ok",
            started_at=now,
            completed_at=now,
            inputs={"symbol": symbol, "evidence_count": len(evidence)},
        )
    )
    ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=paths.audit_dir,
        created_at=now,
        produced_by="synthesize_prediction_candidates",
        tool_run_id=tool_run_id,
        schema_version="phase2-candidate-synthesis.v1",
    ).write_json(
        artifact_id=artifact_id,
        artifact_type="prediction_input",
        filename="prediction-inputs.json",
        payload=payload,
        metadata={"candidate_id": candidate_id},
    )
    store.upsert_prediction_candidate(candidate)
    for evidence_id in candidate.evidence_for:
        store.link_candidate_evidence(
            CandidateEvidenceLinkRecord(
                candidate_id=candidate_id,
                evidence_id=evidence_id,
                relationship="supports",
                metadata={"source": "phase2_mcp_synthesis"},
                created_at=now,
            )
        )
    for evidence_id in candidate.evidence_against:
        store.link_candidate_evidence(
            CandidateEvidenceLinkRecord(
                candidate_id=candidate_id,
                evidence_id=evidence_id,
                relationship="contradicts",
                metadata={"source": "phase2_mcp_synthesis"},
                created_at=now,
            )
        )
    store.link_candidate_artifact(
        CandidateArtifactLinkRecord(
            candidate_id=candidate_id,
            artifact_id=artifact_id,
            relationship="prediction_input",
            metadata={"source": "phase2_mcp_synthesis"},
            created_at=now,
        )
    )
    return {"run_id": run_id, "candidate_id": candidate_id, "artifact_id": artifact_id}


def _candidate_scenario(*, symbol: str, evidence_for: bool, evidence_against: bool) -> str:
    normalized = symbol.upper()
    if evidence_for and evidence_against:
        return (
            f"Live-search evidence for {normalized} is contradictory, so the Phase 2 "
            "scenario remains contested and conservative."
        )
    if evidence_against:
        return (
            f"Live-search evidence for {normalized} is mostly contradictory, so no "
            "supported directional scenario is produced."
        )
    if evidence_for:
        return (
            f"Live-search evidence for {normalized} is sufficient for a monitored "
            "prediction scenario, but dummy tools keep confidence conservative."
        )
    return f"Insufficient evidence for {normalized} after dummy tool execution."


__all__ = ["synthesize_prediction_candidates"]
