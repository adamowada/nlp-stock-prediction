"""Stored run-graph assembly helpers for final prediction reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from nlp_stock_prediction.contracts import CredentialState
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import (
    ProviderStatus,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.contracts.evaluation import PredictionEvaluationArtifactPayload
from nlp_stock_prediction.contracts.instruments import InstrumentUniverse
from nlp_stock_prediction.contracts.provenance import (
    EvidenceReference,
    ProviderHealth,
    ProviderWarning,
)
from nlp_stock_prediction.contracts.report import (
    AuditArtifact,
    MaterialClaimTrace,
    ReportSourceReference,
)
from nlp_stock_prediction.ml.timesfm.contracts import TimesFmForecastArtifact
from nlp_stock_prediction.orchestration.artifact_policy import (
    ALLOWED_ARTIFACT_TYPES,
    JSON_ARTIFACT_TYPES,
    ArtifactType,
    source_reference_type_for_artifact,
)
from nlp_stock_prediction.orchestration.phase2_common import file_sha256, utc_now
from nlp_stock_prediction.orchestration.phase4_market_data import (
    load_phase4_market_data_artifact,
)
from nlp_stock_prediction.orchestration.phase4_technical_package import (
    load_phase4_technical_package_artifact,
)
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    EvidenceRecord,
    PredictionCandidateRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

_TOOL_HEALTH_STATUSES = {"failed", "partial", "empty", "skipped"}


@dataclass(frozen=True)
class ReportAssemblyState:
    """Validated stored inputs and trace maps used by report rendering."""

    audit_artifacts: tuple[AuditArtifact, ...]
    usable_candidate_records: tuple[PredictionCandidateRecord, ...]
    excluded_candidate_reasons: dict[str, tuple[str, ...]]
    warnings: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    provider_health: tuple[ProviderHealth, ...]
    candidate_artifact_ids: dict[str, tuple[str, ...]]
    candidate_evidence_ids: dict[str, tuple[str, ...]]
    artifact_candidate_ids: dict[str, tuple[str, ...]]
    evidence_candidate_ids: dict[str, tuple[str, ...]]
    missing_artifact_ids: tuple[str, ...]
    missing_evidence_ids: tuple[str, ...]

    @property
    def excluded_candidate_ids(self) -> tuple[str, ...]:
        return tuple(self.excluded_candidate_reasons)

    @property
    def usable_candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.usable_candidate_records)


@dataclass(frozen=True)
class _CandidateRequirements:
    artifact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass
class _AssemblyBuilder:
    store: SQLiteStore
    repo_root: Path
    artifact_records: tuple[ArtifactRecord, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    candidate_records: tuple[PredictionCandidateRecord, ...]
    tool_runs: tuple[ToolRunRecord, ...]
    warnings: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    excluded_candidate_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    candidate_artifact_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    candidate_evidence_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    artifact_candidate_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    evidence_candidate_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    missing_artifact_ids: list[str] = field(default_factory=list)
    missing_evidence_ids: list[str] = field(default_factory=list)

    def build(self) -> ReportAssemblyState:
        artifacts_by_id = {record.artifact_id: record for record in self.artifact_records}
        evidence_by_id = {record.evidence_id: record for record in self.evidence_records}
        requirements_by_candidate = {
            candidate.candidate_id: self._candidate_requirements(
                candidate,
                artifacts_by_id=artifacts_by_id,
                evidence_by_id=evidence_by_id,
            )
            for candidate in self.candidate_records
        }
        required_artifact_ids = tuple(
            dict.fromkeys(
                (
                    *(
                        artifact_id
                        for requirements in requirements_by_candidate.values()
                        for artifact_id in requirements.artifact_ids
                    ),
                    *(
                        record.artifact_id
                        for record in self.artifact_records
                        if record.artifact_type == "instrument_universe"
                    ),
                )
            )
        )
        validation_by_artifact_id: dict[str, tuple[str, ...]] = {}
        audit_artifacts = self._audit_artifacts(
            required_artifact_ids=required_artifact_ids,
            validation_by_artifact_id=validation_by_artifact_id,
        )
        missing_required_artifacts = tuple(
            artifact_id
            for artifact_id in required_artifact_ids
            if artifact_id not in artifacts_by_id
        )
        for artifact_id in missing_required_artifacts:
            validation_by_artifact_id[artifact_id] = (f"Missing required artifact {artifact_id}.",)
            self.missing_artifact_ids.append(artifact_id)

        usable_candidates: list[PredictionCandidateRecord] = []
        for candidate in self.candidate_records:
            requirements = requirements_by_candidate[candidate.candidate_id]
            reasons = self._candidate_blocking_reasons(
                candidate,
                requirements=requirements,
                evidence_by_id=evidence_by_id,
                artifacts_by_id=artifacts_by_id,
                validation_by_artifact_id=validation_by_artifact_id,
            )
            if reasons:
                self.excluded_candidate_reasons[candidate.candidate_id] = reasons
                continue
            usable_candidates.append(candidate)

        for candidate_id, reasons in self.excluded_candidate_reasons.items():
            joined_reasons = "; ".join(reasons)
            reason = f"Candidate {candidate_id} excluded during report assembly: {joined_reasons}"
            self.blocking_reasons.append(reason)
            self.warnings.append(reason)

        return ReportAssemblyState(
            audit_artifacts=tuple(audit_artifacts),
            usable_candidate_records=tuple(usable_candidates),
            excluded_candidate_reasons=dict(self.excluded_candidate_reasons),
            warnings=tuple(dict.fromkeys(self.warnings)),
            blocking_reasons=tuple(dict.fromkeys(self.blocking_reasons)),
            provider_health=self._assembly_provider_health(),
            candidate_artifact_ids=dict(self.candidate_artifact_ids),
            candidate_evidence_ids=dict(self.candidate_evidence_ids),
            artifact_candidate_ids=dict(self.artifact_candidate_ids),
            evidence_candidate_ids=dict(self.evidence_candidate_ids),
            missing_artifact_ids=tuple(dict.fromkeys(self.missing_artifact_ids)),
            missing_evidence_ids=tuple(dict.fromkeys(self.missing_evidence_ids)),
        )

    def _candidate_requirements(
        self,
        candidate: PredictionCandidateRecord,
        *,
        artifacts_by_id: dict[str, ArtifactRecord],
        evidence_by_id: dict[str, EvidenceRecord],
    ) -> _CandidateRequirements:
        artifact_ids: list[str] = []
        evidence_ids = _candidate_evidence_ids(candidate)
        for evidence_id in evidence_ids:
            self._append_map_value(
                self.evidence_candidate_ids,
                evidence_id,
                candidate.candidate_id,
            )
            evidence = evidence_by_id.get(evidence_id)
            if evidence is not None and evidence.artifact_id:
                artifact_ids.append(evidence.artifact_id)

        artifact_ids.extend(candidate.signal_artifacts)
        for evidence_link in self.store.list_candidate_evidence_links(candidate.candidate_id):
            self._append_map_value(
                self.evidence_candidate_ids,
                evidence_link.evidence_id,
                candidate.candidate_id,
            )
            if evidence_link.evidence_id not in evidence_ids:
                evidence_ids = tuple(dict.fromkeys((*evidence_ids, evidence_link.evidence_id)))
            evidence = evidence_by_id.get(evidence_link.evidence_id)
            if evidence is not None and evidence.artifact_id:
                artifact_ids.append(evidence.artifact_id)
        self.candidate_evidence_ids[candidate.candidate_id] = evidence_ids
        for artifact_id in _evaluation_artifact_ids(candidate):
            artifact_ids.append(artifact_id)
        for artifact_link in self.store.list_candidate_artifact_links(candidate.candidate_id):
            artifact_ids.append(artifact_link.artifact_id)

        unique_artifact_ids = tuple(dict.fromkeys(artifact_ids))
        self.candidate_artifact_ids[candidate.candidate_id] = unique_artifact_ids
        for artifact_id in unique_artifact_ids:
            if artifact_id in artifacts_by_id:
                self._append_map_value(
                    self.artifact_candidate_ids,
                    artifact_id,
                    candidate.candidate_id,
                )
        return _CandidateRequirements(
            artifact_ids=unique_artifact_ids,
            evidence_ids=evidence_ids,
        )

    def _audit_artifacts(
        self,
        *,
        required_artifact_ids: tuple[str, ...],
        validation_by_artifact_id: dict[str, tuple[str, ...]],
    ) -> list[AuditArtifact]:
        required = set(required_artifact_ids)
        audit_artifacts: list[AuditArtifact] = []
        for record in self.artifact_records:
            validation_warnings = self._validate_artifact(record)
            validation_by_artifact_id[record.artifact_id] = validation_warnings
            if validation_warnings:
                message = f"Artifact {record.artifact_id}: {'; '.join(validation_warnings)}"
                self.warnings.append(message)
                if record.artifact_id in required:
                    self.blocking_reasons.append(message)
            artifact = _audit_artifact_from_record(
                record,
                self.repo_root,
                required=record.artifact_id in required,
                validation_warnings=validation_warnings,
            )
            if artifact is not None:
                audit_artifacts.append(artifact)
        return audit_artifacts

    def _validate_artifact(self, record: ArtifactRecord) -> tuple[str, ...]:
        if record.artifact_type not in ALLOWED_ARTIFACT_TYPES:
            return (f"Unknown artifact type {record.artifact_type}.",)
        path = _artifact_path(record, self.repo_root)
        warnings: list[str] = []
        if not path.exists():
            return (f"Artifact file is missing at {path.as_posix()}.",)
        actual_sha256 = file_sha256(path)
        if actual_sha256 != record.sha256:
            warnings.append(
                f"Artifact sha256 mismatch: expected {record.sha256}, observed {actual_sha256}."
            )
        if record.artifact_type == "markdown_report":
            return tuple(warnings)
        if record.artifact_type in JSON_ARTIFACT_TYPES:
            try:
                _validate_json_artifact_payload(record, path)
            except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
                warnings.append(f"Artifact payload is malformed: {exc}.")
        return tuple(warnings)

    def _candidate_blocking_reasons(
        self,
        candidate: PredictionCandidateRecord,
        *,
        requirements: _CandidateRequirements,
        evidence_by_id: dict[str, EvidenceRecord],
        artifacts_by_id: dict[str, ArtifactRecord],
        validation_by_artifact_id: dict[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        for evidence_id in requirements.evidence_ids:
            if evidence_id not in evidence_by_id:
                reasons.append(f"Missing required evidence {evidence_id}.")
                self.missing_evidence_ids.append(evidence_id)
        for artifact_id in requirements.artifact_ids:
            if artifact_id not in artifacts_by_id:
                reasons.append(f"Missing required artifact {artifact_id}.")
                self.missing_artifact_ids.append(artifact_id)
                continue
            artifact_warnings = validation_by_artifact_id.get(artifact_id, ())
            if artifact_warnings:
                joined_warnings = "; ".join(artifact_warnings)
                reasons.append(f"Artifact {artifact_id} is not usable: {joined_warnings}")
        return tuple(dict.fromkeys(reasons))

    def _assembly_provider_health(self) -> tuple[ProviderHealth, ...]:
        now = utc_now()
        health: list[ProviderHealth] = []
        if self.warnings or self.blocking_reasons:
            severity = WarningSeverity.ERROR if self.blocking_reasons else WarningSeverity.WARNING
            status = ProviderStatus.MALFORMED if self.blocking_reasons else ProviderStatus.PARTIAL
            health.append(
                ProviderHealth(
                    provider_name="report-assembly",
                    status=status,
                    checked_at=now,
                    credential_state=CredentialState.NOT_REQUIRED,
                    warnings=tuple(
                        ProviderWarning(
                            code=WarningCode.MALFORMED_RESPONSE
                            if self.blocking_reasons
                            else WarningCode.PARTIAL_DATA,
                            severity=severity,
                            provider_name="report-assembly",
                            message=warning[:400],
                            occurred_at=now,
                        )
                        for warning in tuple(dict.fromkeys(self.warnings))
                    ),
                )
            )
        for tool_run in self.tool_runs:
            if tool_run.status not in _TOOL_HEALTH_STATUSES:
                continue
            tool_messages = (
                *tool_run.warnings,
                *((tool_run.error_message,) if tool_run.error_message else ()),
            )
            messages = tuple(dict.fromkeys(tool_messages))
            health.append(
                ProviderHealth(
                    provider_name=f"tool:{tool_run.tool_name}",
                    status=_provider_status_for_tool_run(tool_run.status),
                    checked_at=tool_run.completed_at or tool_run.started_at,
                    credential_state=CredentialState.NOT_REQUIRED,
                    warnings=tuple(
                        ProviderWarning(
                            code=_warning_code_for_tool_run(tool_run.status),
                            severity=WarningSeverity.ERROR
                            if tool_run.status == "failed"
                            else WarningSeverity.WARNING,
                            provider_name=f"tool:{tool_run.tool_name}",
                            message=message[:400],
                            occurred_at=tool_run.completed_at or tool_run.started_at,
                            metadata={"tool_run_id": tool_run.tool_run_id},
                        )
                        for message in messages
                    ),
                )
            )
        return tuple(health)

    @staticmethod
    def _append_map_value(target: dict[str, tuple[str, ...]], key: str, value: str) -> None:
        target[key] = tuple(dict.fromkeys((*target.get(key, ()), value)))


def prepare_report_assembly_state(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_records: tuple[ArtifactRecord, ...],
    evidence_records: tuple[EvidenceRecord, ...],
    candidate_records: tuple[PredictionCandidateRecord, ...],
    tool_runs: tuple[ToolRunRecord, ...],
) -> ReportAssemblyState:
    """Validate stored report inputs and return trace maps for rendering."""

    return _AssemblyBuilder(
        store=store,
        repo_root=repo_root,
        artifact_records=artifact_records,
        evidence_records=evidence_records,
        candidate_records=candidate_records,
        tool_runs=tool_runs,
    ).build()


def report_source_references(
    *,
    evidence_sources: tuple[object, ...],
    audit_artifacts: tuple[AuditArtifact, ...],
    provider_health: tuple[ProviderHealth, ...],
    assembly_state: ReportAssemblyState,
) -> tuple[ReportSourceReference, ...]:
    """Create deterministic evidence/artifact/provider references from stored inputs."""

    references: list[ReportSourceReference] = []
    reference_ids: set[str] = set()
    for evidence in evidence_sources:
        evidence_id = getattr(evidence, "evidence_id", None)
        if not isinstance(evidence_id, str) or not evidence_id:
            continue
        candidate_ids = _usable_candidate_ids(
            assembly_state,
            assembly_state.evidence_candidate_ids.get(evidence_id, ()),
        )
        if not candidate_ids and len(references) >= 5:
            continue
        references.append(
            ReportSourceReference(
                reference_id=_unique_reference_id(
                    f"source-ref-{evidence_id}",
                    reference_ids,
                ),
                label=f"Source evidence {evidence_id}",
                reference_type="source_evidence",
                evidence_ids=(evidence_id,),
                candidate_ids=candidate_ids,
            )
        )
    for artifact in audit_artifacts:
        references.append(
            ReportSourceReference(
                reference_id=_unique_reference_id(
                    f"source-ref-{artifact.artifact_id}",
                    reference_ids,
                ),
                label=f"Artifact {artifact.artifact_id}",
                reference_type=source_reference_type_for_artifact(artifact.artifact_type),
                evidence_ids=_artifact_evidence_ids(artifact),
                artifact_ids=(artifact.artifact_id,),
                candidate_ids=_usable_candidate_ids(
                    assembly_state,
                    assembly_state.artifact_candidate_ids.get(artifact.artifact_id, ()),
                ),
                metadata={
                    "artifact_type": artifact.artifact_type,
                    "path": artifact.path,
                    "sha256": artifact.sha256,
                },
            )
        )
    for health in provider_health:
        reliability_note_ids = _provider_reliability_note_ids(
            audit_artifacts,
            provider_name=health.provider_name,
        )
        references.append(
            ReportSourceReference(
                reference_id=_unique_reference_id(
                    f"source-ref-provider-{_reference_slug(health.provider_name)}",
                    reference_ids,
                ),
                label=f"Provider health {health.provider_name}",
                reference_type="provider_health",
                provider_names=(health.provider_name,),
                metadata={
                    "status": health.status.value,
                    "source_reliability_note_ids": list(reliability_note_ids),
                },
            )
        )
    return tuple(references)


def _artifact_evidence_ids(artifact: AuditArtifact) -> tuple[str, ...]:
    evidence_id = artifact.metadata.get("evidence_id")
    evidence_ids = artifact.metadata.get("evidence_ids")
    values: list[str] = []
    if isinstance(evidence_id, str) and evidence_id:
        values.append(evidence_id)
    if isinstance(evidence_ids, list | tuple):
        values.extend(value for value in evidence_ids if isinstance(value, str) and value)
    return tuple(dict.fromkeys(values))


def _provider_reliability_note_ids(
    audit_artifacts: tuple[AuditArtifact, ...],
    *,
    provider_name: str,
) -> tuple[str, ...]:
    note_ids: list[str] = []
    for artifact in audit_artifacts:
        if artifact.artifact_type != "source_reliability_note":
            continue
        if artifact.metadata.get("provider") != provider_name:
            continue
        note_id = artifact.metadata.get("note_id")
        if isinstance(note_id, str) and note_id:
            note_ids.append(note_id)
    return tuple(dict.fromkeys(note_ids))


def material_claim_traces(
    *,
    prediction_candidates: tuple[object, ...],
    source_references: tuple[ReportSourceReference, ...],
    assembly_state: ReportAssemblyState,
) -> tuple[MaterialClaimTrace, ...]:
    """Trace candidate thesis, baseline, and evaluation claims to stored sources."""

    if not prediction_candidates:
        return ()
    source_reference_ids_by_candidate: dict[str, tuple[str, ...]] = {}
    for reference in source_references:
        for reference_candidate_id in reference.candidate_ids:
            source_reference_ids_by_candidate[reference_candidate_id] = tuple(
                dict.fromkeys(
                    (
                        *source_reference_ids_by_candidate.get(reference_candidate_id, ()),
                        reference.reference_id,
                    )
                )
            )

    traces: list[MaterialClaimTrace] = []
    for candidate in prediction_candidates:
        candidate_id = getattr(candidate, "candidate_id", None)
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        thesis = getattr(candidate, "thesis", None)
        baseline = getattr(candidate, "baseline", None)
        evidence = tuple(getattr(candidate, "evidence_for", ())) + tuple(
            getattr(candidate, "evidence_against", ())
        )
        artifact_ids = assembly_state.candidate_artifact_ids.get(candidate_id, ())
        source_ref_ids = source_reference_ids_by_candidate.get(candidate_id, ())
        if isinstance(thesis, str) and thesis:
            traces.append(
                MaterialClaimTrace(
                    claim_id=f"claim-{candidate_id}",
                    claim=thesis,
                    claim_type=(
                        "analysis"
                        if (evidence or artifact_ids or source_ref_ids)
                        else "labeled_inference"
                    ),
                    evidence=cast(tuple[EvidenceReference, ...], evidence),
                    artifact_ids=artifact_ids,
                    source_reference_ids=source_ref_ids,
                    candidate_ids=(candidate_id,),
                    rationale=(
                        None
                        if (evidence or artifact_ids or source_ref_ids)
                        else "Candidate is carried as structured insufficient-evidence context."
                    ),
                )
            )
        if isinstance(baseline, str) and baseline:
            traces.append(
                MaterialClaimTrace(
                    claim_id=f"claim-{candidate_id}-baseline",
                    claim=baseline,
                    claim_type="baseline",
                    artifact_ids=artifact_ids,
                    source_reference_ids=source_ref_ids,
                    candidate_ids=(candidate_id,),
                )
            )
        evaluation_artifact_ids: tuple[str, ...] = ()
        evaluation_metadata = getattr(candidate, "metadata", {}).get("prediction_evaluation")
        if isinstance(evaluation_metadata, dict):
            evaluation_artifact_id = evaluation_metadata.get("artifact_id")
            if isinstance(evaluation_artifact_id, str) and evaluation_artifact_id:
                evaluation_artifact_ids = tuple(
                    dict.fromkeys((*evaluation_artifact_ids, evaluation_artifact_id))
                )
            score = evaluation_metadata.get("score")
            claim = f"Prediction quality evaluation for {candidate_id}" + (
                f" recorded score {score}." if score is not None else "."
            )
            evaluation_id = evaluation_metadata.get("evaluation_id")
            evaluation_trace_metadata: JsonObject = {}
            if isinstance(evaluation_id, str):
                evaluation_trace_metadata["evaluation_id"] = evaluation_id
            traces.append(
                MaterialClaimTrace(
                    claim_id=f"claim-{candidate_id}-evaluation",
                    claim=claim,
                    claim_type="prediction_evaluation",
                    evidence=cast(tuple[EvidenceReference, ...], evidence),
                    artifact_ids=evaluation_artifact_ids or artifact_ids,
                    source_reference_ids=source_ref_ids,
                    candidate_ids=(candidate_id,),
                    metadata=evaluation_trace_metadata,
                )
            )
    return tuple(traces)


def _audit_artifact_from_record(
    record: ArtifactRecord,
    repo_root: Path,
    *,
    required: bool,
    validation_warnings: tuple[str, ...],
) -> AuditArtifact | None:
    if record.artifact_type not in ALLOWED_ARTIFACT_TYPES:
        return None
    path = _artifact_path(record, repo_root)
    metadata: JsonObject = {
        **record.metadata,
        "assembly_required": required,
        "assembly_status": "warning" if validation_warnings else "ok",
    }
    if validation_warnings:
        metadata["assembly_warnings"] = list(validation_warnings)
    return AuditArtifact(
        artifact_id=record.artifact_id,
        artifact_type=cast(ArtifactType, record.artifact_type),
        path=path.as_posix(),
        created_at=record.created_at or utc_now(),
        produced_by=record.produced_by or "phase2-mcp",
        sha256=record.sha256,
        record_count=record.record_count,
        metadata=metadata,
    )


def _validate_json_artifact_payload(record: ArtifactRecord, path: Path) -> None:
    if record.artifact_type == "market_data":
        load_phase4_market_data_artifact(path)
        return
    if record.artifact_type == "technical_package":
        load_phase4_technical_package_artifact(path)
        return
    if record.artifact_type == "prediction_evaluation":
        PredictionEvaluationArtifactPayload.model_validate_json(path.read_text(encoding="utf-8"))
        return
    if record.artifact_type == "instrument_universe":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("instrument_universe artifact payload must be an object")
        universe_payload = payload.get("universe")
        if isinstance(universe_payload, dict):
            universe_payload = dict(universe_payload)
            universe_payload.pop("instrument_ids", None)
        InstrumentUniverse.model_validate(universe_payload)
        return
    if record.artifact_type == "ml_forecast":
        TimesFmForecastArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact payload must be an object")


def _candidate_evidence_ids(candidate: PredictionCandidateRecord) -> tuple[str, ...]:
    metadata = candidate.metadata.get("prediction_evaluation")
    if isinstance(metadata, dict):
        evidence_for_ids = _string_tuple(metadata.get("evidence_for_ids"))
        evidence_against_ids = _string_tuple(metadata.get("evidence_against_ids"))
        if evidence_for_ids or evidence_against_ids:
            return tuple(dict.fromkeys((*evidence_for_ids, *evidence_against_ids)))
    return tuple(dict.fromkeys((*candidate.evidence_for, *candidate.evidence_against)))


def _evaluation_artifact_ids(candidate: PredictionCandidateRecord) -> tuple[str, ...]:
    metadata = candidate.metadata.get("prediction_evaluation")
    if not isinstance(metadata, dict):
        return ()
    artifact_id = metadata.get("artifact_id")
    return (artifact_id,) if isinstance(artifact_id, str) and artifact_id else ()


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _artifact_path(record: ArtifactRecord, repo_root: Path) -> Path:
    return record.path if record.path.is_absolute() else repo_root / record.path


def _usable_candidate_ids(
    assembly_state: ReportAssemblyState,
    candidate_ids: tuple[str, ...],
) -> tuple[str, ...]:
    usable_ids = set(assembly_state.usable_candidate_ids)
    return tuple(candidate_id for candidate_id in candidate_ids if candidate_id in usable_ids)


def _provider_status_for_tool_run(status: str) -> ProviderStatus:
    if status == "failed":
        return ProviderStatus.FAILED
    if status == "partial":
        return ProviderStatus.PARTIAL
    return ProviderStatus.EMPTY


def _warning_code_for_tool_run(status: str) -> WarningCode:
    if status == "failed":
        return WarningCode.UPSTREAM_UNAVAILABLE
    if status == "partial":
        return WarningCode.PARTIAL_DATA
    return WarningCode.NO_DATA


def _reference_slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")


def _unique_reference_id(reference_id: str, seen: set[str]) -> str:
    if reference_id not in seen:
        seen.add(reference_id)
        return reference_id
    index = 2
    while f"{reference_id}-{index}" in seen:
        index += 1
    unique = f"{reference_id}-{index}"
    seen.add(unique)
    return unique


__all__ = [
    "ReportAssemblyState",
    "material_claim_traces",
    "prepare_report_assembly_state",
    "report_source_references",
]
