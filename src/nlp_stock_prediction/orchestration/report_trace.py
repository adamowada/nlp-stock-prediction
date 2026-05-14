"""Report source-reference and material-claim trace construction."""

from __future__ import annotations

from dataclasses import dataclass

from nlp_stock_prediction.contracts.report import (
    AuditArtifact,
    MaterialClaimTrace,
    PriorOutcomeReview,
    ReportSourceReference,
)


@dataclass(frozen=True)
class ReportTrace:
    """Traceability records derived from report inputs."""

    source_references: tuple[ReportSourceReference, ...]
    material_claim_traces: tuple[MaterialClaimTrace, ...]


def build_report_trace(
    *,
    evidence_sources: tuple[object, ...],
    prediction_candidates: tuple[object, ...],
    audit_artifacts: tuple[AuditArtifact, ...],
    prior_outcome_reviews: tuple[PriorOutcomeReview, ...],
) -> ReportTrace:
    """Build report-level references and candidate claim traces."""

    source_references = _report_source_references(
        evidence_sources=evidence_sources,
        prediction_candidates=prediction_candidates,
        audit_artifacts=audit_artifacts,
        prior_outcome_reviews=prior_outcome_reviews,
    )
    return ReportTrace(
        source_references=source_references,
        material_claim_traces=_material_claim_traces(
            prediction_candidates=prediction_candidates,
            source_references=source_references,
        ),
    )


def _report_source_references(
    *,
    evidence_sources: tuple[object, ...],
    prediction_candidates: tuple[object, ...],
    audit_artifacts: tuple[AuditArtifact, ...],
    prior_outcome_reviews: tuple[PriorOutcomeReview, ...],
) -> tuple[ReportSourceReference, ...]:
    references: list[ReportSourceReference] = []
    candidate_ids = tuple(
        candidate_id
        for candidate_id in (
            getattr(candidate, "candidate_id", None) for candidate in prediction_candidates
        )
        if isinstance(candidate_id, str) and candidate_id
    )
    for evidence in evidence_sources:
        evidence_id = getattr(evidence, "evidence_id", None)
        if not isinstance(evidence_id, str) or not evidence_id:
            continue
        references.append(
            ReportSourceReference(
                reference_id=f"source-ref-{evidence_id}",
                label=f"Source evidence {evidence_id}",
                reference_type="source_evidence",
                evidence_ids=(evidence_id,),
                candidate_ids=_candidate_ids_for_evidence(
                    evidence_id,
                    prediction_candidates,
                ),
            )
        )
    for artifact in audit_artifacts:
        if artifact.artifact_type not in {
            "prediction_evaluation",
            "technical_package",
            "calibration_summary",
            "signal_family_ablation",
            "walk_forward_evaluation",
        }:
            continue
        reference_type = (
            "prediction_evaluation"
            if artifact.artifact_type == "prediction_evaluation"
            else "tool_artifact"
        )
        references.append(
            ReportSourceReference(
                reference_id=f"source-ref-{artifact.artifact_id}",
                label=f"Artifact {artifact.artifact_id}",
                reference_type=reference_type,
                artifact_ids=(artifact.artifact_id,),
                candidate_ids=_candidate_ids_for_artifact(
                    artifact,
                    prediction_candidates,
                    fallback_candidate_ids=candidate_ids,
                ),
                metadata={"artifact_type": artifact.artifact_type},
            )
        )
    for review in prior_outcome_reviews:
        references.append(
            ReportSourceReference(
                reference_id=f"source-ref-{review.review_id}",
                label=f"Prior outcome review {review.review_id}",
                reference_type="prior_outcome",
                evidence_ids=tuple(reference.evidence_id for reference in review.outcome_evidence),
                artifact_ids=review.artifact_ids,
                candidate_ids=((review.candidate_id,) if review.candidate_id else ()),
                prior_outcome_review_ids=(review.review_id,),
                metadata={"status": review.status},
            )
        )
    return tuple(references)


def _material_claim_traces(
    *,
    prediction_candidates: tuple[object, ...],
    source_references: tuple[ReportSourceReference, ...],
) -> tuple[MaterialClaimTrace, ...]:
    if not prediction_candidates:
        return ()
    traces: list[MaterialClaimTrace] = []
    for candidate in prediction_candidates:
        candidate_id = getattr(candidate, "candidate_id", None)
        thesis = getattr(candidate, "thesis", None)
        evidence_for = getattr(candidate, "evidence_for", ())
        evidence_against = getattr(candidate, "evidence_against", ())
        signal_artifact_ids = getattr(candidate, "signal_artifact_ids", ())
        if not isinstance(candidate_id, str) or not isinstance(thesis, str):
            continue
        prior_outcome_review_ids = _candidate_prior_outcome_review_ids(candidate)
        source_reference_ids = _source_reference_ids_for_candidate(
            candidate,
            source_references,
        )
        has_trace_references = bool(
            evidence_for
            or evidence_against
            or signal_artifact_ids
            or source_reference_ids
            or prior_outcome_review_ids
        )
        traces.append(
            MaterialClaimTrace(
                claim_id=f"claim-{candidate_id}",
                claim=thesis,
                claim_type="analysis" if has_trace_references else "labeled_inference",
                evidence=tuple(evidence_for) + tuple(evidence_against),
                artifact_ids=tuple(signal_artifact_ids),
                source_reference_ids=source_reference_ids,
                candidate_ids=(candidate_id,),
                prior_outcome_review_ids=prior_outcome_review_ids,
                rationale=(
                    None
                    if has_trace_references
                    else "Candidate is carried as structured insufficient-evidence context."
                ),
            )
        )
    return tuple(traces)


def _candidate_ids_for_evidence(
    evidence_id: str,
    prediction_candidates: tuple[object, ...],
) -> tuple[str, ...]:
    candidate_ids: list[str] = []
    for candidate in prediction_candidates:
        candidate_id = getattr(candidate, "candidate_id", None)
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        if evidence_id in _candidate_evidence_ids(candidate):
            candidate_ids.append(candidate_id)
    return tuple(dict.fromkeys(candidate_ids))


def _candidate_ids_for_artifact(
    artifact: AuditArtifact,
    prediction_candidates: tuple[object, ...],
    *,
    fallback_candidate_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if artifact.artifact_type in {
        "calibration_summary",
        "signal_family_ablation",
        "walk_forward_evaluation",
    }:
        source_candidate_ids = artifact.metadata.get("source_candidate_ids")
        if isinstance(source_candidate_ids, list):
            allowed = set(fallback_candidate_ids)
            return tuple(
                dict.fromkeys(
                    candidate_id
                    for candidate_id in source_candidate_ids
                    if isinstance(candidate_id, str) and candidate_id and candidate_id in allowed
                )
            )
        return ()
    candidate_ids: list[str] = []
    for candidate in prediction_candidates:
        candidate_id = getattr(candidate, "candidate_id", None)
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        if artifact.artifact_id in _candidate_artifact_ids(candidate):
            candidate_ids.append(candidate_id)
    return tuple(dict.fromkeys(candidate_ids))


def _source_reference_ids_for_candidate(
    candidate: object,
    source_references: tuple[ReportSourceReference, ...],
) -> tuple[str, ...]:
    candidate_id = getattr(candidate, "candidate_id", None)
    if not isinstance(candidate_id, str) or not candidate_id:
        return ()
    evidence_ids = set(_candidate_evidence_ids(candidate))
    artifact_ids = set(_candidate_artifact_ids(candidate))
    prior_review_ids = set(_candidate_prior_outcome_review_ids(candidate))
    reference_ids: list[str] = []
    for reference in source_references:
        if candidate_id in reference.candidate_ids:
            reference_ids.append(reference.reference_id)
            continue
        if evidence_ids.intersection(reference.evidence_ids):
            reference_ids.append(reference.reference_id)
            continue
        if artifact_ids.intersection(reference.artifact_ids):
            reference_ids.append(reference.reference_id)
            continue
        if prior_review_ids.intersection(reference.prior_outcome_review_ids):
            reference_ids.append(reference.reference_id)
    return tuple(dict.fromkeys(reference_ids))


def _candidate_evidence_ids(candidate: object) -> tuple[str, ...]:
    ids: list[str] = []
    for field_name in ("evidence_for", "evidence_against"):
        for reference in getattr(candidate, field_name, ()):
            evidence_id = getattr(reference, "evidence_id", None)
            if isinstance(evidence_id, str) and evidence_id:
                ids.append(evidence_id)
    return tuple(dict.fromkeys(ids))


def _candidate_artifact_ids(candidate: object) -> tuple[str, ...]:
    ids: list[str] = []
    for artifact_id in getattr(candidate, "signal_artifact_ids", ()):
        if isinstance(artifact_id, str) and artifact_id:
            ids.append(artifact_id)
    for reference in getattr(candidate, "signal_artifacts", ()):
        artifact_id = getattr(reference, "artifact_id", None)
        if isinstance(artifact_id, str) and artifact_id:
            ids.append(artifact_id)
    return tuple(dict.fromkeys(ids))


def _candidate_prior_outcome_review_ids(candidate: object) -> tuple[str, ...]:
    return tuple(
        review_id
        for review_id in getattr(candidate, "prior_outcome_review_ids", ())
        if isinstance(review_id, str) and review_id
    )


__all__ = ["ReportTrace", "build_report_trace"]
