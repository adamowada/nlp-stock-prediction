"""Phase-neutral conversion helpers for stored prediction candidates."""

from __future__ import annotations

from nlp_stock_prediction.contracts import (
    Direction,
    DissentingEvidence,
    EvidenceReference,
    PredictionCandidate,
    PredictionStatus,
    PredictionType,
    SignalArtifactReference,
    SourceEvidence,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.signal_artifact_references import (
    legacy_signal_artifact_reference,
    metadata_signal_artifact_references,
)
from nlp_stock_prediction.storage.records import PredictionCandidateRecord


def prediction_candidate_from_record(
    candidate: PredictionCandidateRecord,
    evidence_sources: tuple[SourceEvidence, ...],
    *,
    include_missing_references: bool = False,
    prefer_evaluated_references: bool = False,
) -> PredictionCandidate:
    """Return a report/evaluation candidate from durable SQLite state."""

    evidence_by_id = {record.evidence_id: record for record in evidence_sources}
    evidence_for_ids, evidence_against_ids = _candidate_evidence_ids(
        candidate,
        prefer_evaluated_references=prefer_evaluated_references,
    )
    evidence_for_refs = _evidence_references(
        evidence_for_ids,
        evidence_by_id,
        include_missing_references=include_missing_references,
    )
    evidence_against_refs = _evidence_references(
        evidence_against_ids,
        evidence_by_id,
        include_missing_references=include_missing_references,
    )
    status = _candidate_status(candidate, evidence_for_refs, evidence_against_refs)
    if prefer_evaluated_references:
        status = _evaluated_status(candidate) or status
    symbol = candidate.metadata.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        symbol = candidate.instrument_id.rsplit(":", 1)[-1]
    baseline = candidate.baseline.get("summary")
    if not isinstance(baseline, str) or not baseline.strip():
        baseline = "No directional edge is assumed without source-backed evidence."
    uncertainty = candidate.uncertainty or "Evidence coverage and freshness may limit confidence."
    signal_artifacts = _signal_artifact_references(candidate)
    signal_artifact_ids = tuple(
        dict.fromkeys(
            (
                *candidate.signal_artifacts,
                *(reference.artifact_id for reference in signal_artifacts),
            )
        )
    )
    return PredictionCandidate(
        candidate_id=candidate.candidate_id,
        instrument_id=candidate.instrument_id,
        symbol=symbol,
        prediction_type=_prediction_type(candidate.prediction_type),
        horizon=_time_horizon(candidate.prediction_horizon),
        direction=_direction(candidate.direction),
        status=status,
        thesis=candidate.scenario,
        baseline=baseline,
        confidence=candidate.confidence or 0.0,
        evidence_for=evidence_for_refs,
        evidence_against=evidence_against_refs,
        dissenting_evidence=_dissenting_evidence(evidence_against_refs),
        assumptions=("Source evidence is observed material, not automatically true.",),
        uncertainties=(uncertainty,),
        change_trigger_limitations=(
            "Stored candidate records do not yet include structured change-trigger inputs.",
        ),
        signal_artifact_ids=signal_artifact_ids,
        signal_artifacts=signal_artifacts,
        metadata=candidate.metadata,
    )


def _candidate_evidence_ids(
    candidate: PredictionCandidateRecord,
    *,
    prefer_evaluated_references: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if prefer_evaluated_references:
        metadata = candidate.metadata.get("prediction_evaluation")
        if isinstance(metadata, dict):
            evidence_for_ids = _string_tuple(metadata.get("evidence_for_ids"))
            evidence_against_ids = _string_tuple(metadata.get("evidence_against_ids"))
            if evidence_for_ids or evidence_against_ids:
                return evidence_for_ids, evidence_against_ids
    return candidate.evidence_for, candidate.evidence_against


def _evidence_references(
    evidence_ids: tuple[str, ...],
    evidence_by_id: dict[str, SourceEvidence],
    *,
    include_missing_references: bool,
) -> tuple[EvidenceReference, ...]:
    references: list[EvidenceReference] = []
    for evidence_id in evidence_ids:
        if evidence_id not in evidence_by_id and not include_missing_references:
            continue
        references.append(EvidenceReference(evidence_id=evidence_id))
    return tuple(references)


def _evaluated_status(candidate: PredictionCandidateRecord) -> PredictionStatus | None:
    metadata = candidate.metadata.get("prediction_evaluation")
    if not isinstance(metadata, dict):
        return None
    status = metadata.get("status")
    if not isinstance(status, str):
        return None
    try:
        return PredictionStatus(status)
    except ValueError:
        return None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


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
    if evidence_against_refs:
        return PredictionStatus.CONTRADICTED
    try:
        return PredictionStatus(candidate.status)
    except ValueError:
        if evidence_for_refs:
            return PredictionStatus.EVIDENCE_SUPPORTED
        return PredictionStatus.INSUFFICIENT_EVIDENCE


def _dissenting_evidence(
    evidence_against_refs: tuple[EvidenceReference, ...],
) -> tuple[DissentingEvidence, ...]:
    if not evidence_against_refs:
        return ()
    return (
        DissentingEvidence(
            summary="Stored source evidence contradicts or materially limits the scenario.",
            evidence=evidence_against_refs,
            impact="contradicts",
        ),
    )


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


def _prediction_type(value: str) -> PredictionType:
    try:
        return PredictionType(value)
    except ValueError:
        return PredictionType.DIRECTIONAL


def _signal_artifact_references(
    candidate: PredictionCandidateRecord,
) -> tuple[SignalArtifactReference, ...]:
    references: list[SignalArtifactReference] = []
    seen: set[str] = set()
    for reference in metadata_signal_artifact_references(
        candidate.metadata.get("signal_artifacts")
    ):
        if reference.artifact_id in seen:
            continue
        seen.add(reference.artifact_id)
        references.append(reference)
    for artifact_id in candidate.signal_artifacts:
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        references.append(
            legacy_signal_artifact_reference(artifact_id).model_copy(
                update={"metadata": {"derived_from_stored_candidate": True}}
            )
        )
    return tuple(references)


__all__ = [
    "prediction_candidate_from_record",
    "update_candidate_record_from_contract",
]
