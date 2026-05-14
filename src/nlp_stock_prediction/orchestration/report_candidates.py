"""Phase-neutral conversion helpers for stored prediction candidates."""

from __future__ import annotations

from nlp_stock_prediction.contracts import (
    Direction,
    EvidenceReference,
    PredictionCandidate,
    PredictionStatus,
    SourceEvidence,
    TimeHorizon,
)
from nlp_stock_prediction.storage.records import PredictionCandidateRecord


def prediction_candidate_from_record(
    candidate: PredictionCandidateRecord,
    evidence_sources: tuple[SourceEvidence, ...],
) -> PredictionCandidate:
    """Return a report/evaluation candidate from durable SQLite state."""

    evidence_by_id = {record.evidence_id: record for record in evidence_sources}
    evidence_for_refs = tuple(
        EvidenceReference(
            evidence_id=evidence_id,
            quote=evidence_by_id[evidence_id].text[:180] if evidence_id in evidence_by_id else None,
            relevance=0.76,
        )
        for evidence_id in candidate.evidence_for
    )
    evidence_against_refs = tuple(
        EvidenceReference(
            evidence_id=evidence_id,
            quote=evidence_by_id[evidence_id].text[:180] if evidence_id in evidence_by_id else None,
            relevance=0.76,
        )
        for evidence_id in candidate.evidence_against
    )
    status = _candidate_status(candidate, evidence_for_refs, evidence_against_refs)
    symbol = candidate.metadata.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        symbol = candidate.instrument_id.rsplit(":", 1)[-1]
    baseline = candidate.baseline.get("summary")
    if not isinstance(baseline, str) or not baseline.strip():
        baseline = "No directional edge is assumed without source-backed evidence."
    uncertainty = candidate.uncertainty or "Evidence coverage and freshness may limit confidence."
    return PredictionCandidate(
        candidate_id=candidate.candidate_id,
        instrument_id=candidate.instrument_id,
        symbol=symbol,
        horizon=_time_horizon(candidate.prediction_horizon),
        direction=_direction(candidate.direction),
        status=status,
        thesis=candidate.scenario,
        baseline=baseline,
        confidence=candidate.confidence or 0.0,
        evidence_for=evidence_for_refs,
        evidence_against=evidence_against_refs,
        assumptions=("Source evidence is observed material, not automatically true.",),
        uncertainties=(uncertainty,),
        signal_artifact_ids=candidate.signal_artifacts,
        metadata=candidate.metadata,
    )


def update_candidate_record_from_contract(
    stored: PredictionCandidateRecord,
    candidate: PredictionCandidate,
) -> PredictionCandidateRecord:
    """Preserve stored fields while applying report-facing evaluation updates."""

    return PredictionCandidateRecord(
        candidate_id=stored.candidate_id,
        run_id=stored.run_id,
        instrument_id=stored.instrument_id,
        prediction_horizon=stored.prediction_horizon,
        prediction_type=stored.prediction_type,
        scenario=stored.scenario,
        status=candidate.status.value,
        confidence=stored.confidence,
        direction=stored.direction,
        evidence_for=stored.evidence_for,
        evidence_against=stored.evidence_against,
        signal_artifacts=stored.signal_artifacts,
        baseline=stored.baseline,
        uncertainty=stored.uncertainty,
        metadata=candidate.metadata,
    )


def _candidate_status(
    candidate: PredictionCandidateRecord,
    evidence_for_refs: tuple[EvidenceReference, ...],
    evidence_against_refs: tuple[EvidenceReference, ...],
) -> PredictionStatus:
    try:
        return PredictionStatus(candidate.status)
    except ValueError:
        if evidence_against_refs:
            return PredictionStatus.CONTRADICTED
        if evidence_for_refs:
            return PredictionStatus.EVIDENCE_SUPPORTED
        return PredictionStatus.INSUFFICIENT_EVIDENCE


def _direction(value: str | None) -> Direction:
    if value is None:
        return Direction.MIXED
    try:
        return Direction(value)
    except ValueError:
        return Direction.MIXED


def _time_horizon(value: str) -> TimeHorizon:
    try:
        return TimeHorizon(value)
    except ValueError:
        return TimeHorizon.SWING


__all__ = [
    "prediction_candidate_from_record",
    "update_candidate_record_from_contract",
]
