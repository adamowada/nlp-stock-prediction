"""Phase 4 prediction-quality scoring."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from nlp_stock_prediction.contracts.analysis import AnalysisBundle
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import PredictionStatus
from nlp_stock_prediction.contracts.evaluation import (
    BaselineComparison,
    EvaluationEvidenceCounts,
    PredictionEvaluation,
    PredictionEvaluationArtifactPayload,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.provenance import EvidenceReference
from nlp_stock_prediction.contracts.report import AuditArtifact, PredictionCandidate
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.storage.records import (
    CandidateArtifactLinkRecord,
    CandidateEvidenceLinkRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

_BASELINE_SCORE = 0.5
_TOOL_NAME = "phase4_prediction_evaluation"
_TOOL_VERSION = "phase4.v1"


def evaluate_prediction_candidate(
    candidate: PredictionCandidate,
    *,
    evidence_sources: tuple[SourceEvidence, ...],
    analysis_bundle: AnalysisBundle | None = None,
    created_at: datetime | None = None,
    evaluation_id: str | None = None,
) -> PredictionEvaluation:
    """Score a candidate as prediction quality using attributable source evidence."""

    evaluated_at = _aware_utc(created_at)
    source_ids = frozenset(evidence.evidence_id for evidence in evidence_sources)
    supporting_refs, missing_supporting = _partition_attributed_refs(
        candidate.evidence_for,
        source_ids,
    )
    contradicting_refs, missing_contradicting = _partition_attributed_refs(
        candidate.evidence_against,
        source_ids,
    )
    missing_refs = (*missing_supporting, *missing_contradicting)
    ml_signal_count = _ml_signal_count(analysis_bundle)
    counts = EvaluationEvidenceCounts(
        supporting_source_evidence=len(supporting_refs),
        contradicting_source_evidence=len(contradicting_refs),
        missing_source_references=len(missing_refs),
        technical_signal_artifacts=len(candidate.signal_artifact_ids),
        ml_signal_count=ml_signal_count,
        supporting_reference_ids=tuple(reference.evidence_id for reference in supporting_refs),
        contradicting_reference_ids=tuple(
            reference.evidence_id for reference in contradicting_refs
        ),
        missing_reference_ids=tuple(reference.evidence_id for reference in missing_refs),
    )
    status = _evaluation_status(counts)
    score = _evaluation_score(candidate, counts, status)
    uncertainty = _uncertainty_context(candidate, counts, analysis_bundle)
    baseline = _baseline_comparison(candidate, score)
    return PredictionEvaluation(
        evaluation_id=evaluation_id or _evaluation_id(candidate.candidate_id, evaluated_at),
        candidate_id=candidate.candidate_id,
        instrument_id=candidate.instrument_id,
        symbol=candidate.symbol,
        created_at=evaluated_at,
        horizon=candidate.horizon,
        direction=candidate.direction,
        status=status,
        score=score,
        baseline_comparison=baseline,
        uncertainty=uncertainty,
        evidence_counts=counts,
        evidence_for=supporting_refs,
        evidence_against=contradicting_refs,
        signal_artifact_ids=candidate.signal_artifact_ids,
        metadata={
            "candidate_declared_status": candidate.status.value,
            "analysis_bundle_id": analysis_bundle.analysis_id if analysis_bundle else None,
            "technical_or_ml_support_is_sidecar_only": True,
        },
    )


def write_prediction_evaluation_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    candidate: PredictionCandidate,
    evidence_sources: tuple[SourceEvidence, ...],
    analysis_bundle: AnalysisBundle | None = None,
    created_at: datetime | None = None,
    artifact_filename: str | None = None,
) -> tuple[PredictionEvaluation, AuditArtifact]:
    """Evaluate a stored candidate, write a stable artifact, and index it in SQLite."""

    stored_candidate = store.get_prediction_candidate(candidate.candidate_id)
    if stored_candidate is None:
        raise ValueError("prediction evaluation requires a stored prediction candidate")

    evaluated_at = _aware_utc(created_at)
    digest = _stable_digest(f"{run_id}:{candidate.candidate_id}:{evaluated_at.isoformat()}")
    evaluation = evaluate_prediction_candidate(
        candidate,
        evidence_sources=evidence_sources,
        analysis_bundle=analysis_bundle,
        created_at=evaluated_at,
        evaluation_id=f"evaluation-{_slug(candidate.candidate_id)}-{digest[:8]}",
    )
    tool_run_id = f"tool-prediction-evaluation-{digest[:12]}"
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=_TOOL_NAME,
            tool_version=_TOOL_VERSION,
            status="ok",
            started_at=evaluated_at,
            completed_at=evaluated_at,
            inputs={
                "candidate_id": candidate.candidate_id,
                "source_evidence_count": len(evidence_sources),
                "analysis_bundle_id": analysis_bundle.analysis_id if analysis_bundle else None,
            },
        )
    )
    payload = PredictionEvaluationArtifactPayload(
        run_id=run_id,
        created_at=evaluated_at,
        evaluation=evaluation,
        source_evidence_ids=tuple(evidence.evidence_id for evidence in evidence_sources),
        metadata={
            "quality_language": evaluation.quality_language.model_dump(mode="json"),
            "baseline_comparison": evaluation.baseline_comparison.model_dump(mode="json"),
            "evidence_counts": evaluation.evidence_counts.model_dump(mode="json"),
        },
    )
    artifact_id = f"artifact-prediction-evaluation-{_slug(candidate.candidate_id)}-{digest[:8]}"
    filename = artifact_filename or f"prediction-evaluation-{_slug(candidate.candidate_id)}.json"
    artifact = ArtifactIndex.for_directory(
        store=store,
        repo_root=repo_root,
        base_dir=artifact_dir,
        created_at=evaluated_at,
        produced_by=_TOOL_NAME,
        tool_run_id=tool_run_id,
        schema_version=payload.schema_version,
    ).write_json(
        artifact_id=artifact_id,
        artifact_type="prediction_evaluation",
        filename=filename,
        payload=cast(JsonObject, payload.model_dump(mode="json")),
        record_count=1,
        metadata={
            "run_id": run_id,
            "candidate_id": candidate.candidate_id,
            "evaluation_id": evaluation.evaluation_id,
            "status": evaluation.status.value,
            "quality_language": evaluation.quality_language.model_dump(mode="json"),
        },
    )
    store.link_candidate_artifact(
        CandidateArtifactLinkRecord(
            candidate_id=candidate.candidate_id,
            artifact_id=artifact.artifact_id,
            relationship="prediction_evaluation",
            metadata={
                "evaluation_id": evaluation.evaluation_id,
                "status": evaluation.status.value,
            },
            created_at=evaluated_at,
        )
    )
    _link_available_evidence(
        store=store,
        candidate_id=candidate.candidate_id,
        evaluation=evaluation,
        created_at=evaluated_at,
    )
    return evaluation, artifact


def attach_evaluation_metadata(
    candidate: PredictionCandidate,
    evaluation: PredictionEvaluation,
    *,
    artifact: AuditArtifact | None = None,
) -> PredictionCandidate:
    """Return a candidate copy with report-facing evaluation metadata attached."""

    metadata: JsonObject = dict(candidate.metadata)
    metadata["prediction_evaluation"] = _evaluation_metadata(evaluation, artifact)
    return candidate.model_copy(update={"metadata": metadata})


def _partition_attributed_refs(
    references: tuple[EvidenceReference, ...],
    source_ids: frozenset[str],
) -> tuple[tuple[EvidenceReference, ...], tuple[EvidenceReference, ...]]:
    attributed: list[EvidenceReference] = []
    missing: list[EvidenceReference] = []
    for reference in references:
        if reference.evidence_id in source_ids:
            attributed.append(reference)
        else:
            missing.append(reference)
    return tuple(attributed), tuple(missing)


def _evaluation_status(counts: EvaluationEvidenceCounts) -> PredictionStatus:
    if counts.contradicting_source_evidence > 0:
        return PredictionStatus.CONTRADICTED
    if counts.supporting_source_evidence > 0:
        return PredictionStatus.EVIDENCE_SUPPORTED
    return PredictionStatus.INSUFFICIENT_EVIDENCE


def _evaluation_score(
    candidate: PredictionCandidate,
    counts: EvaluationEvidenceCounts,
    status: PredictionStatus,
) -> float:
    support_depth = min(counts.supporting_source_evidence, 3) / 3.0
    conflict_depth = min(counts.contradicting_source_evidence, 3) / 3.0
    missing_depth = min(counts.missing_source_references, 3) / 3.0
    signal_context = min(
        0.09,
        (counts.technical_signal_artifacts + counts.ml_signal_count) * 0.03,
    )
    if status == PredictionStatus.EVIDENCE_SUPPORTED:
        score = 0.48 + (support_depth * 0.24) + (candidate.confidence * 0.24)
        score -= missing_depth * 0.12
        return _round_score(score)
    if status == PredictionStatus.CONTRADICTED:
        score = 0.22 + (support_depth * 0.18) + (candidate.confidence * 0.18)
        score -= conflict_depth * 0.22
        score -= missing_depth * 0.08
        return min(0.49, _round_score(score))
    score = candidate.confidence * 0.18 + signal_context
    return min(0.24, _round_score(score))


def _baseline_comparison(candidate: PredictionCandidate, score: float) -> BaselineComparison:
    delta = round(score - _BASELINE_SCORE, 6)
    if delta > 0.05:
        verdict = "above_baseline"
    elif delta < -0.05:
        verdict = "below_baseline"
    else:
        verdict = "near_baseline"
    return BaselineComparison(
        baseline_id="no_directional_edge",
        baseline_summary=candidate.baseline,
        baseline_score=_BASELINE_SCORE,
        candidate_score=score,
        score_delta=delta,
        verdict=verdict,
    )


def _uncertainty_context(
    candidate: PredictionCandidate,
    counts: EvaluationEvidenceCounts,
    analysis_bundle: AnalysisBundle | None,
) -> tuple[str, ...]:
    uncertainty = list(candidate.uncertainties)
    if counts.supporting_source_evidence == 0:
        uncertainty.append(
            "No attributable source evidence supports the scenario; technical and ML context "
            "remain sidecar inputs."
        )
    if counts.contradicting_source_evidence > 0:
        uncertainty.append("Contradicting source evidence reduces prediction-quality support.")
    if counts.missing_source_references > 0:
        uncertainty.append(
            "Some cited evidence references were not present in the source evidence inputs."
        )
    if analysis_bundle is not None and analysis_bundle.contradictions:
        uncertainty.append("Analysis bundle reports contradictions that require review.")
    return tuple(dict.fromkeys(uncertainty))


def _ml_signal_count(analysis_bundle: AnalysisBundle | None) -> int:
    if analysis_bundle is None or analysis_bundle.technical is None:
        return 0
    return 1 if analysis_bundle.technical.ml_signal is not None else 0


def _link_available_evidence(
    *,
    store: SQLiteStore,
    candidate_id: str,
    evaluation: PredictionEvaluation,
    created_at: datetime,
) -> None:
    for reference in evaluation.evidence_for:
        if store.get_evidence(reference.evidence_id) is None:
            continue
        store.link_candidate_evidence(
            CandidateEvidenceLinkRecord(
                candidate_id=candidate_id,
                evidence_id=reference.evidence_id,
                relationship="evaluation_supports",
                metadata={"evaluation_id": evaluation.evaluation_id},
                created_at=created_at,
            )
        )
    for reference in evaluation.evidence_against:
        if store.get_evidence(reference.evidence_id) is None:
            continue
        store.link_candidate_evidence(
            CandidateEvidenceLinkRecord(
                candidate_id=candidate_id,
                evidence_id=reference.evidence_id,
                relationship="evaluation_contradicts",
                metadata={"evaluation_id": evaluation.evaluation_id},
                created_at=created_at,
            )
        )


def _evaluation_metadata(
    evaluation: PredictionEvaluation,
    artifact: AuditArtifact | None,
) -> JsonObject:
    metadata: JsonObject = {
        "evaluation_id": evaluation.evaluation_id,
        "status": evaluation.status.value,
        "score": evaluation.score,
        "baseline_verdict": evaluation.baseline_comparison.verdict,
        "quality_label": evaluation.quality_language.report_label,
        "quality_language": evaluation.quality_language.model_dump(mode="json"),
        "evidence_counts": evaluation.evidence_counts.model_dump(mode="json"),
    }
    if artifact is not None:
        metadata["artifact_id"] = artifact.artifact_id
        metadata["artifact_path"] = artifact.path
        metadata["artifact_sha256"] = artifact.sha256
    return metadata


def _evaluation_id(candidate_id: str, created_at: datetime) -> str:
    digest = _stable_digest(f"{candidate_id}:{created_at.isoformat()}")
    return f"evaluation-{_slug(candidate_id)}-{digest[:8]}"


def _aware_utc(value: datetime | None) -> datetime:
    candidate = value or datetime.now(UTC)
    if candidate.tzinfo is None or candidate.utcoffset() is None:
        raise ValueError("evaluation timestamps must be timezone-aware")
    return candidate.astimezone(UTC)


def _round_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "prediction"


__all__ = [
    "attach_evaluation_metadata",
    "evaluate_prediction_candidate",
    "write_prediction_evaluation_artifact",
]
