"""Live Phase 6 evaluation and calibration tooling over persisted research runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, NonEmptyStr
from nlp_stock_prediction.contracts.enums import (
    PredictionType,
    SignalArtifactFamily,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.provenance import EvidenceReference
from nlp_stock_prediction.evaluation.ablation import (
    PHASE6_ABLATION_TOOL_NAME,
    PHASE6_ABLATION_TOOL_VERSION,
    write_signal_family_ablation_artifact,
)
from nlp_stock_prediction.evaluation.calibration import (
    DEFAULT_CALIBRATION_BIN_EDGES,
    PHASE6_CALIBRATION_TOOL_NAME,
    PHASE6_CALIBRATION_TOOL_VERSION,
    write_calibration_summary_artifact,
)
from nlp_stock_prediction.evaluation.outcomes import (
    PHASE6_OUTCOME_TOOL_NAME,
    PHASE6_OUTCOME_TOOL_VERSION,
    build_prediction_evaluation_target,
    write_point_in_time_outcome_evaluation_artifacts,
)
from nlp_stock_prediction.evaluation.walk_forward import (
    PHASE6_WALK_FORWARD_TOOL_NAME,
    PHASE6_WALK_FORWARD_TOOL_VERSION,
    write_walk_forward_evaluation_artifact,
)
from nlp_stock_prediction.orchestration.phase2_common import Phase2WritePolicy
from nlp_stock_prediction.orchestration.phase6_sources import (
    Phase6OutcomeEvaluationSource,
    load_phase6_outcome_evaluation_sources,
)
from nlp_stock_prediction.storage.records import (
    ResearchRunRecord,
)
from nlp_stock_prediction.storage.sqlite import (
    DEFAULT_RESEARCH_DATABASE_PATH,
    SQLiteStore,
    initialize_research_database,
)

PHASE6_STAGE_ORDER: tuple[str, ...] = ("prepare", "evaluate", "summarize", "inspect")
Phase6Stage = Literal["prepare", "evaluate", "summarize", "inspect"]

PHASE6_OUTCOME_EVALUATION_TOOL_ID = "phase6.point_in_time_outcome_evaluation"
PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_ID = "phase6.load_persisted_outcome_evaluations"
PHASE6_ABLATION_TOOL_ID = "phase6.signal_family_ablation"
PHASE6_WALK_FORWARD_TOOL_ID = "phase6.walk_forward_evaluation"
PHASE6_CALIBRATION_TOOL_ID = "phase6.calibration_summary"
PHASE6_INSPECT_TOOL_ID = "phase6.inspect_run"
PHASE6_OUTCOME_EVALUATION_PUBLIC_TOOL_NAME = "phase6_point_in_time_outcome_evaluation"
PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_NAME = "phase6_load_outcome_evaluations"
PHASE6_INSPECT_TOOL_NAME = "inspect_phase6_run"


class Phase6ToolMetadata(ContractModel):
    """Canonical metadata for one live Phase 6 evaluation tool."""

    tool_id: NonEmptyStr
    tool_name: NonEmptyStr
    tool_version: NonEmptyStr
    stage: Phase6Stage
    description: NonEmptyStr
    artifact_kinds: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    offline_capable: bool = True
    live_capable: bool = True
    requires_network: bool = False
    dependencies: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tool_metadata(self) -> Phase6ToolMetadata:
        if self.tool_id in self.dependencies:
            raise ValueError("phase6 tools cannot depend on themselves")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("phase6 tool dependencies must be unique")
        if self.requires_network:
            raise ValueError("phase6 evaluation tools must run from persisted research data")
        return self

    def as_plan_item(self) -> JsonObject:
        return {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "version": self.tool_version,
            "tool_version": self.tool_version,
            "stage": self.stage,
            "description": self.description,
            "artifact_kinds": list(self.artifact_kinds),
            "offline_capable": self.offline_capable,
            "live_capable": self.live_capable,
            "requires_network": self.requires_network,
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }


class Phase6ToolRegistry:
    """Deterministic Phase 6 tool metadata registry."""

    def __init__(self, tools: Iterable[Phase6ToolMetadata] = ()) -> None:
        self._tools: dict[str, Phase6ToolMetadata] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Phase6ToolMetadata) -> None:
        if tool.tool_id in self._tools:
            raise ValueError(f"phase6 tool is already registered: {tool.tool_id}")
        if any(
            item.tool_name == tool.tool_name and item.tool_version == tool.tool_version
            for item in self._tools.values()
        ):
            raise ValueError(
                f"phase6 tool name/version is already registered: "
                f"{tool.tool_name}@{tool.tool_version}"
            )
        self._tools[tool.tool_id] = tool

    def get(self, tool_id: str) -> Phase6ToolMetadata:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"phase6 tool is not registered: {tool_id}") from exc

    def specs(self) -> tuple[Phase6ToolMetadata, ...]:
        return self.ordered_for_stages(PHASE6_STAGE_ORDER)

    def ordered_for_stages(
        self,
        stage_order: Sequence[str] = PHASE6_STAGE_ORDER,
    ) -> tuple[Phase6ToolMetadata, ...]:
        stages = tuple(stage_order)
        if len(set(stages)) != len(stages):
            raise ValueError("stage_order entries must be unique")
        stage_index = {stage: index for index, stage in enumerate(stages)}
        unknown_stages = sorted(
            {tool.stage for tool in self._tools.values() if tool.stage not in stage_index}
        )
        if unknown_stages:
            joined = ", ".join(unknown_stages)
            raise ValueError(f"phase6 tools use stages missing from stage_order: {joined}")
        self._validate_dependencies()
        order_index = {tool_id: index for index, tool_id in enumerate(self._tools)}
        return tuple(
            tool
            for _tool_id, tool in sorted(
                self._tools.items(),
                key=lambda item: (stage_index[item[1].stage], order_index[item[0]]),
            )
        )

    def as_plan(self) -> JsonObject:
        return {
            "tools": [tool.as_plan_item() for tool in self.ordered_for_stages()],
            "stage_order": list(PHASE6_STAGE_ORDER),
        }

    def _validate_dependencies(self) -> None:
        known_ids = set(self._tools)
        missing: dict[str, tuple[str, ...]] = {}
        for tool in self._tools.values():
            unknown = tuple(
                dependency for dependency in tool.dependencies if dependency not in known_ids
            )
            if unknown:
                missing[tool.tool_id] = unknown
        if missing:
            details = "; ".join(
                f"{tool_id}: {', '.join(dependencies)}"
                for tool_id, dependencies in sorted(missing.items())
            )
            raise ValueError(f"phase6 tool dependencies are not registered: {details}")


def build_phase6_tool_registry() -> Phase6ToolRegistry:
    """Build the default Phase 6 metadata registry."""

    return Phase6ToolRegistry(
        (
            Phase6ToolMetadata(
                tool_id=PHASE6_OUTCOME_EVALUATION_TOOL_ID,
                tool_name=PHASE6_OUTCOME_EVALUATION_PUBLIC_TOOL_NAME,
                tool_version=PHASE6_OUTCOME_TOOL_VERSION,
                stage="evaluate",
                description=(
                    "Persist a point-in-time outcome evaluation for one stored prediction "
                    "candidate using attributed post-window evidence and artifacts."
                ),
                artifact_kinds=("prediction_outcome", "prediction_outcome_evaluation"),
                metadata={"source_tool_name": PHASE6_OUTCOME_TOOL_NAME},
            ),
            Phase6ToolMetadata(
                tool_id=PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_ID,
                tool_name=PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_NAME,
                tool_version="phase6.persisted-outcome-evaluations.v1",
                stage="evaluate",
                description=(
                    "Load persisted point-in-time outcome-evaluation artifacts for a research "
                    "run without introducing new fixtures or provider calls."
                ),
                artifact_kinds=("prediction_outcome_evaluation",),
                metadata={"source_tool_name": PHASE6_OUTCOME_TOOL_NAME},
            ),
            Phase6ToolMetadata(
                tool_id=PHASE6_ABLATION_TOOL_ID,
                tool_name=PHASE6_ABLATION_TOOL_NAME,
                tool_version=PHASE6_ABLATION_TOOL_VERSION,
                stage="evaluate",
                description=(
                    "Persist signal-family included-versus-excluded outcome quality slices from "
                    "stored outcome evaluations."
                ),
                artifact_kinds=("signal_family_ablation",),
                dependencies=(PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_ID,),
            ),
            Phase6ToolMetadata(
                tool_id=PHASE6_WALK_FORWARD_TOOL_ID,
                tool_name=PHASE6_WALK_FORWARD_TOOL_NAME,
                tool_version=PHASE6_WALK_FORWARD_TOOL_VERSION,
                stage="evaluate",
                description=(
                    "Persist chronological walk-forward folds from stored outcome evaluations."
                ),
                artifact_kinds=("walk_forward_evaluation",),
                dependencies=(PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_ID,),
            ),
            Phase6ToolMetadata(
                tool_id=PHASE6_CALIBRATION_TOOL_ID,
                tool_name=PHASE6_CALIBRATION_TOOL_NAME,
                tool_version=PHASE6_CALIBRATION_TOOL_VERSION,
                stage="summarize",
                description=(
                    "Persist reliability bins, cohort metrics, and calibration slices from "
                    "stored outcome evaluations."
                ),
                artifact_kinds=("calibration_summary",),
                dependencies=(PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_ID,),
            ),
            Phase6ToolMetadata(
                tool_id=PHASE6_INSPECT_TOOL_ID,
                tool_name=PHASE6_INSPECT_TOOL_NAME,
                tool_version="phase6.inspect.v1",
                stage="inspect",
                description="Inspect persisted Phase 6 run counts and calibration artifacts.",
                dependencies=(PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_ID,),
                metadata={"final_only": True},
            ),
        )
    )


def phase6_evaluation_tool_plan(
    registry: Phase6ToolRegistry | None = None,
) -> JsonObject:
    """Return a registry-derived Phase 6 tool plan."""

    return (registry or build_phase6_tool_registry()).as_plan()


@dataclass(frozen=True)
class Phase6Service:
    """Stateful Phase 6 service over the research SQLite run graph."""

    repo_root: Path = Path(".")
    database_path: Path = DEFAULT_RESEARCH_DATABASE_PATH
    extra_write_roots: tuple[Path, ...] = ()
    registry: Phase6ToolRegistry = field(default_factory=build_phase6_tool_registry)
    _store: SQLiteStore = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", self.repo_root.resolve())
        object.__setattr__(
            self,
            "extra_write_roots",
            tuple(path.resolve() for path in self.extra_write_roots),
        )
        object.__setattr__(
            self,
            "_store",
            initialize_research_database(self._resolve_write_path(self.database_path)),
        )

    @property
    def write_policy(self) -> Phase2WritePolicy:
        return Phase2WritePolicy(self.repo_root, extra_allowed_roots=self.extra_write_roots)

    @property
    def store(self) -> SQLiteStore:
        return self._store

    def list_phase6_tool_plan(self) -> JsonObject:
        return phase6_evaluation_tool_plan(self.registry)

    def phase6_point_in_time_outcome_evaluation(
        self,
        *,
        run_id: str,
        candidate_id: str,
        point_in_time_cutoff: str,
        evaluation_window_start: str,
        evaluation_window_end: str,
        artifact_dir: str | None = None,
        report_date: str | None = None,
        status: str | None = None,
        observed_result: str | None = None,
        observed_at: str | None = None,
        result_summary: str | None = None,
        result_value: float | None = None,
        baseline_value: float | None = None,
        outcome_evidence_ids: Sequence[str] = (),
        market_artifact_ids: Sequence[str] = (),
        limitations: Sequence[str] = (),
        created_at: str | None = None,
        evaluated_at: str | None = None,
    ) -> JsonObject:
        self._require_run(run_id)
        target = build_prediction_evaluation_target(
            store=self.store,
            run_id=run_id,
            candidate_id=candidate_id,
            point_in_time_cutoff=_parse_aware_datetime(
                point_in_time_cutoff,
                "point_in_time_cutoff",
            ),
            evaluation_window_start=_parse_aware_datetime(
                evaluation_window_start,
                "evaluation_window_start",
            ),
            evaluation_window_end=_parse_aware_datetime(
                evaluation_window_end,
                "evaluation_window_end",
            ),
            report_date=_parse_optional_date(report_date, "report_date"),
        )
        evidence = tuple(
            EvidenceReference(evidence_id=evidence_id)
            for evidence_id in _non_empty_unique_strings(
                outcome_evidence_ids,
                "outcome_evidence_ids",
            )
        )
        result = write_point_in_time_outcome_evaluation_artifacts(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir_for_run(run_id=run_id, artifact_dir=artifact_dir),
            run_id=run_id,
            target=target,
            status=status,
            observed_result=observed_result,
            observed_at=(
                None
                if observed_at is None or not observed_at.strip()
                else _parse_aware_datetime(observed_at, "observed_at")
            ),
            result_summary=result_summary,
            result_value=result_value,
            baseline_value=baseline_value,
            outcome_evidence=evidence,
            market_artifact_ids=_non_empty_unique_strings(
                market_artifact_ids,
                "market_artifact_ids",
            ),
            limitations=_non_empty_unique_strings(limitations, "limitations"),
            created_at=(
                None
                if created_at is None or not created_at.strip()
                else _parse_aware_datetime(created_at, "created_at")
            ),
            evaluated_at=(
                None
                if evaluated_at is None or not evaluated_at.strip()
                else _parse_aware_datetime(evaluated_at, "evaluated_at")
            ),
        )
        return {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "target_id": result.target.target_id,
            "tool_run_id": result.tool_run_id,
            "outcome_id": result.outcome.outcome_id,
            "outcome_evaluation_id": result.outcome_evaluation.outcome_evaluation_id,
            "status": result.outcome_evaluation.status.value,
            "quality_score": result.outcome_evaluation.quality_score,
            "outcome_artifact_id": result.outcome_artifact.artifact_id,
            "outcome_evaluation_artifact_id": result.outcome_evaluation_artifact.artifact_id,
            "outcome_artifact_path": Path(result.outcome_artifact.path).as_posix(),
            "outcome_evaluation_artifact_path": Path(
                result.outcome_evaluation_artifact.path
            ).as_posix(),
            "limitations": list(result.outcome_evaluation.limitations),
        }

    def phase6_load_outcome_evaluations(self, *, run_id: str) -> JsonObject:
        self._require_run(run_id)
        sources = self._outcome_sources(run_id)
        return {
            "run_id": run_id,
            "outcome_evaluation_count": len(sources),
            "source_outcome_evaluation_ids": [
                source.outcome_evaluation.outcome_evaluation_id for source in sources
            ],
            "source_artifact_ids": [source.artifact.artifact_id for source in sources],
            "sources": [source.as_summary() for source in sources],
        }

    def phase6_signal_family_ablation(
        self,
        *,
        run_id: str,
        cohort_id: str,
        point_in_time_cutoff: str,
        artifact_dir: str | None = None,
        families: Sequence[str] | None = None,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> JsonObject:
        self._require_run(run_id)
        sources = self._outcome_sources(run_id)
        result = write_signal_family_ablation_artifact(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir_from_sources(run_id=run_id, artifact_dir=artifact_dir),
            run_id=run_id,
            cohort_id=cohort_id,
            inputs=tuple(source.ablation_input for source in sources),
            point_in_time_cutoff=_parse_aware_datetime(
                point_in_time_cutoff,
                "point_in_time_cutoff",
            ),
            families=_signal_families(families),
            prediction_type=_prediction_type(prediction_type),
            horizon=_horizon(horizon),
        )
        return {
            "run_id": run_id,
            "cohort_id": cohort_id,
            "calibration_id": result.calibration_id,
            "tool_run_id": result.tool_run_id,
            "artifact_id": result.artifact.artifact_id,
            "artifact_path": Path(result.artifact.path).as_posix(),
            "ablation_count": len(result.ablations),
            "calibration_slice_count": len(result.calibration_slices),
            "source_outcome_evaluation_ids": list(
                result.artifact_payload.source_outcome_evaluation_ids
            ),
            "limitations": list(result.artifact_payload.limitations),
        }

    def phase6_walk_forward_evaluation(
        self,
        *,
        run_id: str,
        cohort_id: str,
        point_in_time_cutoff: str,
        minimum_train_size: int,
        artifact_dir: str | None = None,
        test_size: int = 1,
        step_size: int = 1,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> JsonObject:
        self._require_run(run_id)
        sources = self._outcome_sources(run_id)
        result = write_walk_forward_evaluation_artifact(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir_from_sources(run_id=run_id, artifact_dir=artifact_dir),
            run_id=run_id,
            cohort_id=cohort_id,
            outcome_evaluations=tuple(source.outcome_evaluation for source in sources),
            point_in_time_cutoff=_parse_aware_datetime(
                point_in_time_cutoff,
                "point_in_time_cutoff",
            ),
            minimum_train_size=minimum_train_size,
            test_size=test_size,
            step_size=step_size,
            prediction_type=_prediction_type(prediction_type),
            horizon=_horizon(horizon),
        )
        return {
            "run_id": run_id,
            "cohort_id": cohort_id,
            "calibration_id": result.calibration_id,
            "tool_run_id": result.tool_run_id,
            "artifact_id": result.artifact.artifact_id,
            "artifact_path": Path(result.artifact.path).as_posix(),
            "fold_count": len(result.folds),
            "sample_count": result.artifact_payload.sample_count,
            "resolved_count": result.artifact_payload.resolved_count,
            "calibration_slice_count": len(result.calibration_slices),
            "source_outcome_evaluation_ids": list(
                result.artifact_payload.source_outcome_evaluation_ids
            ),
            "excluded_outcome_evaluation_ids": list(
                result.artifact_payload.excluded_outcome_evaluation_ids
            ),
            "limitations": list(result.artifact_payload.limitations),
        }

    def phase6_calibration_summary(
        self,
        *,
        run_id: str,
        cohort_id: str,
        as_of: str,
        artifact_dir: str | None = None,
        bin_edges: Sequence[float] = DEFAULT_CALIBRATION_BIN_EDGES,
        families: Sequence[str] | None = None,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> JsonObject:
        self._require_run(run_id)
        sources = self._outcome_sources(run_id)
        result = write_calibration_summary_artifact(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir_from_sources(run_id=run_id, artifact_dir=artifact_dir),
            run_id=run_id,
            cohort_id=cohort_id,
            inputs=tuple(source.calibration_input for source in sources),
            as_of=_parse_aware_datetime(as_of, "as_of"),
            bin_edges=tuple(bin_edges),
            families=_signal_families(families),
            prediction_type=_prediction_type(prediction_type),
            horizon=_horizon(horizon),
        )
        return {
            "run_id": run_id,
            "cohort_id": cohort_id,
            "calibration_id": result.calibration_id,
            "tool_run_id": result.tool_run_id,
            "artifact_id": result.artifact.artifact_id,
            "artifact_path": Path(result.artifact.path).as_posix(),
            "sample_count": result.summary.sample_count,
            "resolved_count": result.summary.resolved_count,
            "brier_score": result.summary.brier_score,
            "expected_calibration_error": result.summary.expected_calibration_error,
            "calibration_slice_count": len(result.calibration_slices),
            "source_outcome_evaluation_ids": list(result.summary.source_outcome_evaluation_ids),
            "excluded_outcome_evaluation_ids": list(
                result.artifact_payload.excluded_outcome_evaluation_ids
            ),
            "limitations": list(result.summary.limitations),
        }

    def inspect_phase6_run(self, *, run_id: str) -> JsonObject:
        run = self._require_run(run_id)
        outcome_evaluations = self.store.list_outcome_evaluations_for_run(run_id)
        calibration_runs = self.store.list_calibration_runs_for_run(run_id)
        tool_runs = self.store.list_tool_runs_for_run(run_id)
        registered_tool_names = {tool.tool_name for tool in self.registry.specs()}
        raw_status_counts: dict[str, int] = {}
        for record in outcome_evaluations:
            raw_status_counts[record.status] = raw_status_counts.get(record.status, 0) + 1
        return {
            "run_id": run.run_id,
            "status": run.status,
            "outcome_evaluation_count": len(outcome_evaluations),
            "outcome_evaluation_status_counts": cast(JsonObject, raw_status_counts),
            "calibration_run_count": len(calibration_runs),
            "calibration_slice_count": sum(
                len(self.store.list_calibration_slices(record.calibration_id))
                for record in calibration_runs
            ),
            "phase6_tool_run_count": len(
                [record for record in tool_runs if str(record.tool_name) in registered_tool_names]
            ),
            "artifact_count": len(self.store.list_artifacts_for_run(run_id)),
            "calibration_runs": [
                {
                    "calibration_id": record.calibration_id,
                    "method_version": record.method_version,
                    "artifact_id": record.artifact_id,
                    "source_outcome_evaluation_ids": list(record.source_outcome_evaluation_ids),
                    "limitations": list(record.limitations),
                }
                for record in calibration_runs
            ],
        }

    def _require_run(self, run_id: str) -> ResearchRunRecord:
        run = self.store.get_research_run(run_id)
        if run is None:
            raise ValueError(f"research run does not exist: {run_id}")
        return run

    def _outcome_sources(self, run_id: str) -> tuple[Phase6OutcomeEvaluationSource, ...]:
        return load_phase6_outcome_evaluation_sources(
            store=self.store,
            repo_root=self.repo_root,
            run_id=run_id,
        )

    def _artifact_dir_for_run(self, *, run_id: str, artifact_dir: str | None) -> Path:
        if artifact_dir is not None and artifact_dir.strip():
            return self._resolve_write_path(Path(artifact_dir))
        return self._resolve_write_path(Path("reports") / run_id / "audit")

    def _artifact_dir_from_sources(self, *, run_id: str, artifact_dir: str | None) -> Path:
        if artifact_dir is not None and artifact_dir.strip():
            return self._resolve_write_path(Path(artifact_dir))
        return self._infer_artifact_dir(run_id)

    def _infer_artifact_dir(self, run_id: str) -> Path:
        sources = self._outcome_sources(run_id)
        for source in sources:
            for parent in (source.artifact_path.parent, *source.artifact_path.parents):
                if parent.name == "audit":
                    return self._resolve_write_path(parent)
        return self._resolve_write_path(Path("reports") / run_id / "audit")

    def _resolve_write_path(self, path: Path) -> Path:
        return self.write_policy.resolve(path)


def _parse_aware_datetime(value: str, field_name: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone offset")
    return parsed.astimezone(UTC)


def _parse_optional_date(value: str | None, field_name: str) -> date | None:
    if value is None or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc


def _prediction_type(value: str | None) -> PredictionType | None:
    if value is None or not value.strip():
        return None
    try:
        return PredictionType(value.strip())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in PredictionType)
        raise ValueError(f"prediction_type must be one of: {allowed}") from exc


def _horizon(value: str | None) -> TimeHorizon | None:
    if value is None or not value.strip():
        return None
    try:
        return TimeHorizon(value.strip())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in TimeHorizon)
        raise ValueError(f"horizon must be one of: {allowed}") from exc


def _signal_families(values: Sequence[str] | None) -> tuple[SignalArtifactFamily, ...] | None:
    if values is None:
        return None
    families: list[SignalArtifactFamily] = []
    for value in values:
        try:
            families.append(SignalArtifactFamily(value.strip()))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in SignalArtifactFamily)
            raise ValueError(f"families must be one of: {allowed}") from exc
    return tuple(dict.fromkeys(families))


def _non_empty_unique_strings(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    resolved: list[str] = []
    for value in values:
        item = value.strip()
        if not item:
            raise ValueError(f"{field_name} must not include empty values")
        resolved.append(item)
    return tuple(dict.fromkeys(resolved))


__all__ = [
    "PHASE6_ABLATION_TOOL_ID",
    "PHASE6_CALIBRATION_TOOL_ID",
    "PHASE6_INSPECT_TOOL_ID",
    "PHASE6_INSPECT_TOOL_NAME",
    "PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_ID",
    "PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_NAME",
    "PHASE6_OUTCOME_EVALUATION_PUBLIC_TOOL_NAME",
    "PHASE6_OUTCOME_EVALUATION_TOOL_ID",
    "PHASE6_STAGE_ORDER",
    "PHASE6_WALK_FORWARD_TOOL_ID",
    "Phase6OutcomeEvaluationSource",
    "Phase6Service",
    "Phase6ToolMetadata",
    "Phase6ToolRegistry",
    "build_phase6_tool_registry",
    "load_phase6_outcome_evaluation_sources",
    "phase6_evaluation_tool_plan",
]
