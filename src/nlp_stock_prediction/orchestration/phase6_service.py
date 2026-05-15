"""Public evaluation and calibration tooling over persisted research runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import ContractModel, JsonObject, NonEmptyStr
from nlp_stock_prediction.contracts.enums import (
    Direction,
    PredictionType,
    SignalArtifactFamily,
    TimeHorizon,
)
from nlp_stock_prediction.contracts.evaluation import OutcomeReviewSummary
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
from nlp_stock_prediction.evaluation.common import (
    digest,
    phase7_freshness_records_from_target,
    slug,
)
from nlp_stock_prediction.evaluation.drift import (
    PHASE7_CALIBRATION_DRIFT_TOOL_NAME,
    PHASE7_CALIBRATION_DRIFT_TOOL_VERSION,
    CalibrationDriftThresholds,
    write_calibration_drift_check_artifact,
)
from nlp_stock_prediction.evaluation.execution import evaluation_artifact_execution
from nlp_stock_prediction.evaluation.freshness import (
    review_artifact_file_freshness,
    review_evidence_aging,
    write_artifact_freshness_review_artifact,
    write_evidence_aging_summary_artifact,
)
from nlp_stock_prediction.evaluation.live_outcomes import (
    PHASE7_LIVE_OUTCOME_TOOL_NAME,
    PHASE7_LIVE_OUTCOME_TOOL_VERSION,
    DefaultLiveOutcomeProviderFactory,
    LiveOutcomeProviderFactory,
    materialize_live_prediction_outcome_artifacts,
)
from nlp_stock_prediction.evaluation.outcomes import (
    PHASE6_OUTCOME_TOOL_NAME,
    build_prediction_evaluation_target,
    write_point_in_time_outcome_evaluation_artifacts,
)
from nlp_stock_prediction.evaluation.walk_forward import (
    PHASE6_WALK_FORWARD_TOOL_NAME,
    PHASE6_WALK_FORWARD_TOOL_VERSION,
    write_walk_forward_evaluation_artifact,
)
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.phase2_common import Phase2WritePolicy
from nlp_stock_prediction.orchestration.phase6_sources import (
    Phase6OutcomeEvaluationSource,
    load_phase6_outcome_evaluation_sources,
)
from nlp_stock_prediction.orchestration.report_data_modes import (
    LIVE_REPORT_DATA_MODE,
    find_non_live_report_input_violations,
    report_data_mode_from_run,
    report_data_mode_metadata_for_run_id,
)
from nlp_stock_prediction.reliability import (
    build_source_reliability_notes,
    default_provider_replacement_playbooks,
    write_provider_replacement_playbook_artifacts,
    write_source_reliability_note_artifacts,
)
from nlp_stock_prediction.storage.records import (
    ResearchRunRecord,
    ToolRunRecord,
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
PHASE7_LIVE_OUTCOME_MATERIALIZATION_TOOL_ID = "phase7.live_outcome_materialization"
PHASE6_OUTCOME_EVALUATION_PUBLIC_TOOL_NAME = "phase6_point_in_time_outcome_evaluation"
PHASE7_LIVE_OUTCOME_MATERIALIZATION_PUBLIC_TOOL_NAME = "phase7_live_outcome_materialization"
PHASE6_LOAD_OUTCOME_EVALUATIONS_TOOL_NAME = "phase6_load_outcome_evaluations"
PHASE6_INSPECT_TOOL_NAME = "inspect_phase6_run"

EVALUATION_MATERIALIZE_OUTCOME_TOOL_ID = "evaluation.materialize_outcome"
EVALUATION_LOAD_OUTCOMES_TOOL_ID = "evaluation.load_outcomes"
EVALUATION_OUTCOME_SUMMARY_TOOL_ID = "evaluation.outcome_summary"
EVALUATION_STALE_ARTIFACTS_TOOL_ID = "evaluation.stale_artifacts"
EVALUATION_EVIDENCE_AGING_TOOL_ID = "evaluation.evidence_aging"
EVALUATION_SOURCE_RELIABILITY_TOOL_ID = "evaluation.source_reliability"
EVALUATION_PROVIDER_PLAYBOOK_TOOL_ID = "evaluation.provider_playbook"
EVALUATION_ABLATION_TOOL_ID = "evaluation.ablation"
EVALUATION_WALK_FORWARD_TOOL_ID = "evaluation.walk_forward"
EVALUATION_CALIBRATION_TOOL_ID = "evaluation.calibration"
EVALUATION_CALIBRATION_DRIFT_TOOL_ID = "evaluation.calibration_drift"
EVALUATION_INSPECT_TOOL_ID = "evaluation.inspect"

EVALUATION_MATERIALIZE_OUTCOME_TOOL_NAME = "evaluation_materialize_outcome"
EVALUATION_LOAD_OUTCOMES_TOOL_NAME = "evaluation_load_outcomes"
EVALUATION_OUTCOME_SUMMARY_TOOL_NAME = "evaluation_outcome_summary"
EVALUATION_STALE_ARTIFACTS_TOOL_NAME = "evaluation_stale_artifacts"
EVALUATION_EVIDENCE_AGING_TOOL_NAME = "evaluation_evidence_aging"
EVALUATION_SOURCE_RELIABILITY_TOOL_NAME = "evaluation_source_reliability"
EVALUATION_PROVIDER_PLAYBOOK_TOOL_NAME = "evaluation_provider_playbook"
EVALUATION_ABLATION_TOOL_NAME = "evaluation_ablation"
EVALUATION_WALK_FORWARD_TOOL_NAME = "evaluation_walk_forward"
EVALUATION_CALIBRATION_TOOL_NAME = "evaluation_calibration"
EVALUATION_CALIBRATION_DRIFT_TOOL_NAME = "evaluation_calibration_drift"
EVALUATION_INSPECT_TOOL_NAME = "evaluation_inspect"

EVALUATION_OUTCOME_SUMMARY_TOOL_VERSION = "evaluation.outcome-summary.v1"
EVALUATION_STALE_ARTIFACTS_TOOL_VERSION = "evaluation.stale-artifacts.v1"
EVALUATION_EVIDENCE_AGING_TOOL_VERSION = "evaluation.evidence-aging.v1"
EVALUATION_SOURCE_RELIABILITY_TOOL_VERSION = "evaluation.source-reliability.v1"
EVALUATION_PROVIDER_PLAYBOOK_TOOL_VERSION = "evaluation.provider-playbook.v1"
EVALUATION_INSPECT_TOOL_VERSION = "evaluation.inspect.v1"


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
    """Build the default public evaluation metadata registry."""

    return Phase6ToolRegistry(
        (
            Phase6ToolMetadata(
                tool_id=EVALUATION_MATERIALIZE_OUTCOME_TOOL_ID,
                tool_name=EVALUATION_MATERIALIZE_OUTCOME_TOOL_NAME,
                tool_version=PHASE7_LIVE_OUTCOME_TOOL_VERSION,
                stage="evaluate",
                description=(
                    "Fetch or reuse real post-window market data and materialize an outcome "
                    "without caller-supplied result shortcuts."
                ),
                artifact_kinds=(
                    "market_data",
                    "prediction_outcome",
                    "prediction_outcome_evaluation",
                ),
                offline_capable=False,
                live_capable=True,
                requires_network=True,
                metadata={"source_tool_name": PHASE7_LIVE_OUTCOME_TOOL_NAME},
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_LOAD_OUTCOMES_TOOL_ID,
                tool_name=EVALUATION_LOAD_OUTCOMES_TOOL_NAME,
                tool_version="evaluation.load-outcomes.v1",
                stage="evaluate",
                description=(
                    "Load persisted point-in-time outcome-evaluation artifacts for a research "
                    "run without introducing new fixtures or provider calls."
                ),
                artifact_kinds=("prediction_outcome_evaluation",),
                metadata={"source_tool_name": PHASE6_OUTCOME_TOOL_NAME},
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_OUTCOME_SUMMARY_TOOL_ID,
                tool_name=EVALUATION_OUTCOME_SUMMARY_TOOL_NAME,
                tool_version=EVALUATION_OUTCOME_SUMMARY_TOOL_VERSION,
                stage="summarize",
                description=(
                    "Persist a cross-run outcome review summary from stored outcome "
                    "evaluations, including evidence, artifact, and freshness context."
                ),
                artifact_kinds=("outcome_review_summary",),
                dependencies=(EVALUATION_LOAD_OUTCOMES_TOOL_ID,),
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_STALE_ARTIFACTS_TOOL_ID,
                tool_name=EVALUATION_STALE_ARTIFACTS_TOOL_NAME,
                tool_version=EVALUATION_STALE_ARTIFACTS_TOOL_VERSION,
                stage="summarize",
                description=(
                    "Review persisted run artifacts for stale, missing, malformed, "
                    "hash-mismatched, or unknown freshness states."
                ),
                artifact_kinds=("artifact_freshness_review",),
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_EVIDENCE_AGING_TOOL_ID,
                tool_name=EVALUATION_EVIDENCE_AGING_TOOL_NAME,
                tool_version=EVALUATION_EVIDENCE_AGING_TOOL_VERSION,
                stage="summarize",
                description=(
                    "Review stored evidence for aged-out, stale, missing, superseded, "
                    "provider-replaced, or malformed provenance states."
                ),
                artifact_kinds=("evidence_aging_summary",),
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_SOURCE_RELIABILITY_TOOL_ID,
                tool_name=EVALUATION_SOURCE_RELIABILITY_TOOL_NAME,
                tool_version=EVALUATION_SOURCE_RELIABILITY_TOOL_VERSION,
                stage="summarize",
                description=(
                    "Persist source reliability notes from stored live evidence provenance."
                ),
                artifact_kinds=("source_reliability_note",),
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_PROVIDER_PLAYBOOK_TOOL_ID,
                tool_name=EVALUATION_PROVIDER_PLAYBOOK_TOOL_NAME,
                tool_version=EVALUATION_PROVIDER_PLAYBOOK_TOOL_VERSION,
                stage="summarize",
                description=(
                    "Persist provider replacement playbooks that preserve required "
                    "provenance and freshness semantics."
                ),
                artifact_kinds=("provider_replacement_playbook",),
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_ABLATION_TOOL_ID,
                tool_name=EVALUATION_ABLATION_TOOL_NAME,
                tool_version=PHASE6_ABLATION_TOOL_VERSION,
                stage="evaluate",
                description=(
                    "Persist signal-family included-versus-excluded outcome quality slices from "
                    "stored outcome evaluations."
                ),
                artifact_kinds=("signal_family_ablation",),
                dependencies=(EVALUATION_LOAD_OUTCOMES_TOOL_ID,),
                metadata={"source_tool_name": PHASE6_ABLATION_TOOL_NAME},
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_WALK_FORWARD_TOOL_ID,
                tool_name=EVALUATION_WALK_FORWARD_TOOL_NAME,
                tool_version=PHASE6_WALK_FORWARD_TOOL_VERSION,
                stage="evaluate",
                description=(
                    "Persist chronological walk-forward folds from stored outcome evaluations."
                ),
                artifact_kinds=("walk_forward_evaluation",),
                dependencies=(EVALUATION_LOAD_OUTCOMES_TOOL_ID,),
                metadata={"source_tool_name": PHASE6_WALK_FORWARD_TOOL_NAME},
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_CALIBRATION_TOOL_ID,
                tool_name=EVALUATION_CALIBRATION_TOOL_NAME,
                tool_version=PHASE6_CALIBRATION_TOOL_VERSION,
                stage="summarize",
                description=(
                    "Persist reliability bins, cohort metrics, and calibration slices from "
                    "stored outcome evaluations."
                ),
                artifact_kinds=("calibration_summary",),
                dependencies=(EVALUATION_LOAD_OUTCOMES_TOOL_ID,),
                metadata={"source_tool_name": PHASE6_CALIBRATION_TOOL_NAME},
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_CALIBRATION_DRIFT_TOOL_ID,
                tool_name=EVALUATION_CALIBRATION_DRIFT_TOOL_NAME,
                tool_version=PHASE7_CALIBRATION_DRIFT_TOOL_VERSION,
                stage="summarize",
                description=(
                    "Persist a calibration drift check comparing two stored calibration "
                    "summaries without recomputing or backfilling observations."
                ),
                artifact_kinds=("calibration_drift_check",),
                dependencies=(EVALUATION_CALIBRATION_TOOL_ID,),
                metadata={"source_tool_name": PHASE7_CALIBRATION_DRIFT_TOOL_NAME},
            ),
            Phase6ToolMetadata(
                tool_id=EVALUATION_INSPECT_TOOL_ID,
                tool_name=EVALUATION_INSPECT_TOOL_NAME,
                tool_version=EVALUATION_INSPECT_TOOL_VERSION,
                stage="inspect",
                description="Inspect persisted evaluation run counts and audit artifacts.",
                metadata={"final_only": True},
            ),
        )
    )


def phase6_evaluation_tool_plan(
    registry: Phase6ToolRegistry | None = None,
) -> JsonObject:
    """Return a registry-derived public evaluation tool plan."""

    return (registry or build_phase6_tool_registry()).as_plan()


@dataclass(frozen=True)
class Phase6Service:
    """Stateful Phase 6 service over the research SQLite run graph."""

    repo_root: Path = Path(".")
    database_path: Path = DEFAULT_RESEARCH_DATABASE_PATH
    extra_write_roots: tuple[Path, ...] = ()
    registry: Phase6ToolRegistry = field(default_factory=build_phase6_tool_registry)
    live_outcome_provider_factory: LiveOutcomeProviderFactory | None = None
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

    def list_evaluation_tool_plan(self) -> JsonObject:
        return phase6_evaluation_tool_plan(self.registry)

    def list_phase6_tool_plan(self) -> JsonObject:
        return self.list_evaluation_tool_plan()

    def evaluation_materialize_outcome(
        self,
        *,
        run_id: str,
        candidate_id: str,
        point_in_time_cutoff: str,
        evaluation_window_start: str,
        evaluation_window_end: str,
        artifact_dir: str | None = None,
        report_date: str | None = None,
        market_artifact_ids: Sequence[str] = (),
        created_at: str | None = None,
        evaluated_at: str | None = None,
    ) -> JsonObject:
        required_artifact_dir = _required_artifact_dir(artifact_dir)
        return self.phase7_live_outcome_materialization(
            run_id=run_id,
            candidate_id=candidate_id,
            point_in_time_cutoff=point_in_time_cutoff,
            evaluation_window_start=evaluation_window_start,
            evaluation_window_end=evaluation_window_end,
            artifact_dir=required_artifact_dir,
            report_date=report_date,
            market_artifact_ids=market_artifact_ids,
            created_at=created_at,
            evaluated_at=evaluated_at,
        )

    def evaluation_load_outcomes(self, *, run_id: str) -> JsonObject:
        return self.phase6_load_outcome_evaluations(run_id=run_id)

    def evaluation_ablation(
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
        required_artifact_dir = _required_artifact_dir(artifact_dir)
        return self.phase6_signal_family_ablation(
            run_id=run_id,
            cohort_id=cohort_id,
            point_in_time_cutoff=point_in_time_cutoff,
            artifact_dir=required_artifact_dir,
            families=families,
            prediction_type=prediction_type,
            horizon=horizon,
        )

    def evaluation_walk_forward(
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
        required_artifact_dir = _required_artifact_dir(artifact_dir)
        return self.phase6_walk_forward_evaluation(
            run_id=run_id,
            cohort_id=cohort_id,
            point_in_time_cutoff=point_in_time_cutoff,
            minimum_train_size=minimum_train_size,
            artifact_dir=required_artifact_dir,
            test_size=test_size,
            step_size=step_size,
            prediction_type=prediction_type,
            horizon=horizon,
        )

    def evaluation_calibration(
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
        required_artifact_dir = _required_artifact_dir(artifact_dir)
        return self.phase6_calibration_summary(
            run_id=run_id,
            cohort_id=cohort_id,
            as_of=as_of,
            artifact_dir=required_artifact_dir,
            bin_edges=bin_edges,
            families=families,
            prediction_type=prediction_type,
            horizon=horizon,
        )

    def evaluation_calibration_drift(
        self,
        *,
        run_id: str,
        prior_calibration_id: str,
        current_calibration_id: str,
        as_of: str,
        artifact_dir: str | None = None,
        signal_family: str | None = None,
        min_resolved_count: int = 10,
        watch_delta: float = 0.05,
        degraded_delta: float = 0.10,
        improved_delta: float = 0.10,
    ) -> JsonObject:
        required_artifact_dir = _required_artifact_dir(artifact_dir)
        return self.phase7_calibration_drift_check(
            run_id=run_id,
            prior_calibration_id=prior_calibration_id,
            current_calibration_id=current_calibration_id,
            as_of=as_of,
            artifact_dir=required_artifact_dir,
            signal_family=signal_family,
            min_resolved_count=min_resolved_count,
            watch_delta=watch_delta,
            degraded_delta=degraded_delta,
            improved_delta=improved_delta,
        )

    def evaluation_inspect(self, *, run_id: str) -> JsonObject:
        return self.inspect_phase6_run(run_id=run_id)

    def phase7_live_outcome_materialization(
        self,
        *,
        run_id: str,
        candidate_id: str,
        point_in_time_cutoff: str,
        evaluation_window_start: str,
        evaluation_window_end: str,
        artifact_dir: str | None = None,
        report_date: str | None = None,
        market_artifact_ids: Sequence[str] = (),
        created_at: str | None = None,
        evaluated_at: str | None = None,
    ) -> JsonObject:
        self._require_live_run(run_id)
        resolved_created_at = (
            None
            if created_at is None or not created_at.strip()
            else _parse_aware_datetime(created_at, "created_at")
        )
        resolved_evaluated_at = (
            datetime.now(UTC)
            if evaluated_at is None or not evaluated_at.strip()
            else _parse_aware_datetime(evaluated_at, "evaluated_at")
        )
        result = materialize_live_prediction_outcome_artifacts(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir_for_run(run_id=run_id, artifact_dir=artifact_dir),
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
            market_artifact_ids=_non_empty_unique_strings(
                market_artifact_ids,
                "market_artifact_ids",
            ),
            created_at=resolved_created_at,
            evaluated_at=resolved_evaluated_at,
            provider_factory=(
                self.live_outcome_provider_factory
                or DefaultLiveOutcomeProviderFactory(
                    cache_root=self.repo_root / "data" / "provider-cache",
                    now=lambda: resolved_evaluated_at,
                )
            ),
        )
        return {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "target_id": result.target.target_id,
            "evaluation_attempt_id": result.evaluation_attempt_id,
            "tool_run_id": result.written.tool_run_id,
            "outcome_id": result.outcome.outcome_id,
            "outcome_evaluation_id": result.outcome_evaluation.outcome_evaluation_id,
            "outcome_status": result.outcome.status.value,
            "observed_result": (
                result.outcome.observed_result.value if result.outcome.observed_result else None
            ),
            "status": result.outcome_evaluation.status.value,
            "quality_score": result.outcome_evaluation.quality_score,
            "observed_at": (
                None
                if result.outcome.observed_at is None
                else result.outcome.observed_at.isoformat()
            ),
            "result_value": result.outcome.result_value,
            "baseline_value": result.outcome.baseline_value,
            "market_artifact_ids": list(result.market_artifact_ids),
            "outcome_evidence_ids": list(result.outcome_evidence_ids),
            "provider_attempts": [dict(attempt) for attempt in result.provider_attempts],
            "outcome_artifact_id": result.outcome_artifact_id,
            "outcome_evaluation_artifact_id": result.outcome_evaluation_artifact_id,
            "outcome_artifact_path": Path(result.written.outcome_artifact.path).as_posix(),
            "outcome_evaluation_artifact_path": Path(
                result.written.outcome_evaluation_artifact.path
            ).as_posix(),
            "limitations": list(result.outcome_evaluation.limitations),
        }

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
            repo_root=self.repo_root,
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

    def evaluation_outcome_summary(
        self,
        *,
        run_id: str,
        artifact_dir: str | None = None,
        created_at: str | None = None,
    ) -> JsonObject:
        self._require_run(run_id)
        required_artifact_dir = _required_artifact_dir(artifact_dir)
        created = _parse_optional_aware_datetime(created_at, "created_at") or datetime.now(UTC)
        sources = self._outcome_sources(run_id)
        summaries = tuple(
            _outcome_review_summary_from_source(source, created_at=created) for source in sources
        )
        source_outcome_evaluation_ids = tuple(
            summary.outcome_evaluation_id for summary in summaries
        )
        tool_run_id = _evaluation_tool_run_id(
            tool_name=EVALUATION_OUTCOME_SUMMARY_TOOL_NAME,
            run_id=run_id,
            created_at=created,
            material=source_outcome_evaluation_ids,
        )
        inputs: JsonObject = {
            "run_id": run_id,
            "source_outcome_evaluation_ids": list(source_outcome_evaluation_ids),
        }
        artifact_id = _evaluation_artifact_id(
            prefix="outcome-summary",
            run_id=run_id,
            created_at=created,
            material=source_outcome_evaluation_ids,
        )
        payload: JsonObject = {
            "schema_version": "outcome-review-summary-artifact.v1",
            "run_id": run_id,
            "created_at": created.isoformat(),
            "outcome_review_summaries": [
                cast(JsonObject, summary.model_dump(mode="json")) for summary in summaries
            ],
            "metadata": {
                "source_artifact_ids": [source.artifact.artifact_id for source in sources],
                "source_outcome_evaluation_ids": list(source_outcome_evaluation_ids),
            },
        }
        artifact_base_dir = self._artifact_dir_for_run(
            run_id=run_id, artifact_dir=required_artifact_dir
        )
        with evaluation_artifact_execution(
            store=self.store,
            artifact_roots=(artifact_base_dir,),
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=EVALUATION_OUTCOME_SUMMARY_TOOL_NAME,
            tool_version=EVALUATION_OUTCOME_SUMMARY_TOOL_VERSION,
            started_at=created,
            inputs=inputs,
        ):
            artifact = ArtifactIndex.for_directory(
                store=self.store,
                repo_root=self.repo_root,
                base_dir=artifact_base_dir,
                created_at=created,
                produced_by=EVALUATION_OUTCOME_SUMMARY_TOOL_NAME,
                tool_run_id=tool_run_id,
                schema_version="outcome-review-summary-artifact.v1",
            ).write_json(
                artifact_id=artifact_id,
                artifact_type="outcome_review_summary",
                filename=(
                    "outcome-summary/"
                    f"{slug(run_id, allow_file_safe_punctuation=True)}-{artifact_id[-8:]}.json"
                ),
                payload=payload,
                record_count=len(summaries),
                metadata={
                    "run_id": run_id,
                    "summary_count": len(summaries),
                    "summary_ids": [summary.summary_id for summary in summaries],
                    "source_outcome_evaluation_ids": list(source_outcome_evaluation_ids),
                },
            )
            self._record_successful_tool_run(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name=EVALUATION_OUTCOME_SUMMARY_TOOL_NAME,
                tool_version=EVALUATION_OUTCOME_SUMMARY_TOOL_VERSION,
                at=created,
                inputs=inputs,
            )
        return {
            "run_id": run_id,
            "tool_run_id": tool_run_id,
            "artifact_id": artifact.artifact_id,
            "artifact_path": Path(artifact.path).as_posix(),
            "summary_count": len(summaries),
            "summary_ids": [summary.summary_id for summary in summaries],
            "source_outcome_evaluation_ids": list(source_outcome_evaluation_ids),
        }

    def evaluation_stale_artifacts(
        self,
        *,
        run_id: str,
        artifact_dir: str | None = None,
        reviewed_at: str | None = None,
    ) -> JsonObject:
        self._require_run(run_id)
        required_artifact_dir = _required_artifact_dir(artifact_dir)
        reviewed = _parse_optional_aware_datetime(reviewed_at, "reviewed_at") or datetime.now(UTC)
        artifacts = self.store.list_artifacts_for_run(run_id)
        reviews = tuple(
            review_artifact_file_freshness(
                artifact=artifact,
                reviewed_at=reviewed,
                repo_root=self.repo_root,
            )
            for artifact in artifacts
        )
        artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
        tool_run_id = _evaluation_tool_run_id(
            tool_name=EVALUATION_STALE_ARTIFACTS_TOOL_NAME,
            run_id=run_id,
            created_at=reviewed,
            material=artifact_ids,
        )
        inputs: JsonObject = {"run_id": run_id, "artifact_ids": list(artifact_ids)}
        artifact_base_dir = self._artifact_dir_for_run(
            run_id=run_id, artifact_dir=required_artifact_dir
        )
        with evaluation_artifact_execution(
            store=self.store,
            artifact_roots=(artifact_base_dir,),
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=EVALUATION_STALE_ARTIFACTS_TOOL_NAME,
            tool_version=EVALUATION_STALE_ARTIFACTS_TOOL_VERSION,
            started_at=reviewed,
            inputs=inputs,
        ):
            written = write_artifact_freshness_review_artifact(
                store=self.store,
                repo_root=self.repo_root,
                artifact_dir=artifact_base_dir,
                run_id=run_id,
                reviews=reviews,
                created_at=reviewed,
                tool_run_id=tool_run_id,
                metadata={"source_artifact_ids": list(artifact_ids)},
            )
            self._record_successful_tool_run(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name=EVALUATION_STALE_ARTIFACTS_TOOL_NAME,
                tool_version=EVALUATION_STALE_ARTIFACTS_TOOL_VERSION,
                at=reviewed,
                inputs=inputs,
            )
        freshness_counts: dict[str, int] = {}
        for review in reviews:
            freshness_counts[review.freshness_status] = (
                freshness_counts.get(review.freshness_status, 0) + 1
            )
        return {
            "run_id": run_id,
            "tool_run_id": tool_run_id,
            "artifact_id": written.artifact.artifact_id,
            "artifact_path": Path(written.artifact.path).as_posix(),
            "review_count": len(reviews),
            "freshness_counts": cast(JsonObject, freshness_counts),
            "reviewed_artifact_ids": list(artifact_ids),
        }

    def evaluation_evidence_aging(
        self,
        *,
        run_id: str,
        artifact_dir: str | None = None,
        reviewed_at: str | None = None,
    ) -> JsonObject:
        self._require_run(run_id)
        required_artifact_dir = _required_artifact_dir(artifact_dir)
        reviewed = _parse_optional_aware_datetime(reviewed_at, "reviewed_at") or datetime.now(UTC)
        evidence_records = self.store.list_evidence_for_run(run_id)
        reviews = tuple(
            review_evidence_aging(evidence=record, reviewed_at=reviewed)
            for record in evidence_records
        )
        evidence_ids = tuple(record.evidence_id for record in evidence_records)
        tool_run_id = _evaluation_tool_run_id(
            tool_name=EVALUATION_EVIDENCE_AGING_TOOL_NAME,
            run_id=run_id,
            created_at=reviewed,
            material=evidence_ids,
        )
        inputs: JsonObject = {"run_id": run_id, "evidence_ids": list(evidence_ids)}
        artifact_base_dir = self._artifact_dir_for_run(
            run_id=run_id, artifact_dir=required_artifact_dir
        )
        with evaluation_artifact_execution(
            store=self.store,
            artifact_roots=(artifact_base_dir,),
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=EVALUATION_EVIDENCE_AGING_TOOL_NAME,
            tool_version=EVALUATION_EVIDENCE_AGING_TOOL_VERSION,
            started_at=reviewed,
            inputs=inputs,
        ):
            written = write_evidence_aging_summary_artifact(
                store=self.store,
                repo_root=self.repo_root,
                artifact_dir=artifact_base_dir,
                run_id=run_id,
                aging_records=reviews,
                created_at=reviewed,
                tool_run_id=tool_run_id,
                metadata={"source_evidence_ids": list(evidence_ids)},
            )
            self._record_successful_tool_run(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name=EVALUATION_EVIDENCE_AGING_TOOL_NAME,
                tool_version=EVALUATION_EVIDENCE_AGING_TOOL_VERSION,
                at=reviewed,
                inputs=inputs,
            )
        aging_counts: dict[str, int] = {}
        for review in reviews:
            aging_counts[review.age_status] = aging_counts.get(review.age_status, 0) + 1
        return {
            "run_id": run_id,
            "tool_run_id": tool_run_id,
            "artifact_id": written.artifact.artifact_id,
            "artifact_path": Path(written.artifact.path).as_posix(),
            "review_count": len(reviews),
            "aging_counts": cast(JsonObject, aging_counts),
            "reviewed_evidence_ids": list(evidence_ids),
        }

    def evaluation_source_reliability(
        self,
        *,
        run_id: str,
        artifact_dir: str | None = None,
        created_at: str | None = None,
    ) -> JsonObject:
        self._require_live_run(run_id)
        required_artifact_dir = _required_artifact_dir(artifact_dir)
        created = _parse_optional_aware_datetime(created_at, "created_at") or datetime.now(UTC)
        evidence_records = self.store.list_evidence_for_run(run_id)
        if not evidence_records:
            raise ValueError(f"No evidence records were found for run: {run_id}")
        notes = build_source_reliability_notes(evidence_records)
        tool_run_id = _evaluation_tool_run_id(
            tool_name=EVALUATION_SOURCE_RELIABILITY_TOOL_NAME,
            run_id=run_id,
            created_at=created,
            material=tuple(note.note_id for note in notes),
        )
        inputs: JsonObject = {
            "run_id": run_id,
            "evidence_ids": [record.evidence_id for record in evidence_records],
        }
        artifact_base_dir = self._artifact_dir_for_run(
            run_id=run_id, artifact_dir=required_artifact_dir
        )
        with evaluation_artifact_execution(
            store=self.store,
            artifact_roots=(artifact_base_dir,),
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=EVALUATION_SOURCE_RELIABILITY_TOOL_NAME,
            tool_version=EVALUATION_SOURCE_RELIABILITY_TOOL_VERSION,
            started_at=created,
            inputs=inputs,
        ):
            artifacts = write_source_reliability_note_artifacts(
                store=self.store,
                repo_root=self.repo_root,
                artifact_dir=artifact_base_dir,
                run_id=run_id,
                tool_run_id=tool_run_id,
                notes=notes,
                created_at=created,
            )
            self._record_successful_tool_run(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name=EVALUATION_SOURCE_RELIABILITY_TOOL_NAME,
                tool_version=EVALUATION_SOURCE_RELIABILITY_TOOL_VERSION,
                at=created,
                inputs=inputs,
            )
        reliability_counts: dict[str, int] = {}
        for note in notes:
            reliability_counts[note.reliability] = reliability_counts.get(note.reliability, 0) + 1
        return {
            "run_id": run_id,
            "tool_run_id": tool_run_id,
            "note_count": len(notes),
            "artifact_ids": [artifact.artifact_id for artifact in artifacts],
            "artifact_paths": [Path(artifact.path).as_posix() for artifact in artifacts],
            "note_ids": [note.note_id for note in notes],
            "reliability_counts": cast(JsonObject, reliability_counts),
        }

    def evaluation_provider_playbook(
        self,
        *,
        run_id: str,
        artifact_dir: str | None = None,
        created_at: str | None = None,
    ) -> JsonObject:
        self._require_live_run(run_id)
        required_artifact_dir = _required_artifact_dir(artifact_dir)
        created = _parse_optional_aware_datetime(created_at, "created_at") or datetime.now(UTC)
        playbooks = default_provider_replacement_playbooks(created_at=created)
        tool_run_id = _evaluation_tool_run_id(
            tool_name=EVALUATION_PROVIDER_PLAYBOOK_TOOL_NAME,
            run_id=run_id,
            created_at=created,
            material=tuple(playbook.playbook_id for playbook in playbooks),
        )
        inputs: JsonObject = {
            "run_id": run_id,
            "playbook_ids": [item.playbook_id for item in playbooks],
        }
        artifact_base_dir = self._artifact_dir_for_run(
            run_id=run_id, artifact_dir=required_artifact_dir
        )
        with evaluation_artifact_execution(
            store=self.store,
            artifact_roots=(artifact_base_dir,),
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=EVALUATION_PROVIDER_PLAYBOOK_TOOL_NAME,
            tool_version=EVALUATION_PROVIDER_PLAYBOOK_TOOL_VERSION,
            started_at=created,
            inputs=inputs,
        ):
            artifacts = write_provider_replacement_playbook_artifacts(
                store=self.store,
                repo_root=self.repo_root,
                artifact_dir=artifact_base_dir,
                run_id=run_id,
                tool_run_id=tool_run_id,
                playbooks=playbooks,
                created_at=created,
            )
            self._record_successful_tool_run(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name=EVALUATION_PROVIDER_PLAYBOOK_TOOL_NAME,
                tool_version=EVALUATION_PROVIDER_PLAYBOOK_TOOL_VERSION,
                at=created,
                inputs=inputs,
            )
        compatibility_counts: dict[str, int] = {}
        for playbook in playbooks:
            for note in playbook.compatibility_notes:
                compatibility_counts[note.compatibility_status] = (
                    compatibility_counts.get(note.compatibility_status, 0) + 1
                )
        return {
            "run_id": run_id,
            "tool_run_id": tool_run_id,
            "playbook_count": len(playbooks),
            "artifact_ids": [artifact.artifact_id for artifact in artifacts],
            "artifact_paths": [Path(artifact.path).as_posix() for artifact in artifacts],
            "playbook_ids": [playbook.playbook_id for playbook in playbooks],
            "compatibility_counts": cast(JsonObject, compatibility_counts),
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

    def phase7_calibration_drift_check(
        self,
        *,
        run_id: str,
        prior_calibration_id: str,
        current_calibration_id: str,
        as_of: str,
        artifact_dir: str | None = None,
        signal_family: str | None = None,
        min_resolved_count: int = 10,
        watch_delta: float = 0.05,
        degraded_delta: float = 0.10,
        improved_delta: float = 0.10,
    ) -> JsonObject:
        self._require_run(run_id)
        result = write_calibration_drift_check_artifact(
            store=self.store,
            repo_root=self.repo_root,
            artifact_dir=self._artifact_dir_for_run(run_id=run_id, artifact_dir=artifact_dir),
            run_id=run_id,
            prior_calibration_id=prior_calibration_id,
            current_calibration_id=current_calibration_id,
            as_of=_parse_aware_datetime(as_of, "as_of"),
            signal_family=(
                None
                if signal_family is None or not signal_family.strip()
                else _single_signal_family(signal_family)
            ),
            thresholds=CalibrationDriftThresholds(
                min_resolved_count=min_resolved_count,
                watch_delta=watch_delta,
                degraded_delta=degraded_delta,
                improved_delta=improved_delta,
            ),
        )
        return {
            "run_id": run_id,
            "drift_check_id": result.drift_check_id,
            "tool_run_id": result.tool_run_id,
            "artifact_id": result.artifact.artifact_id,
            "artifact_path": Path(result.artifact.path).as_posix(),
            "prior_calibration_id": prior_calibration_id,
            "current_calibration_id": current_calibration_id,
            "drift_status": result.drift_check.drift_status,
            "metric_deltas": dict(result.drift_check.metric_deltas),
            "source_calibration_artifact_ids": list(
                result.drift_check.source_calibration_artifact_ids
            ),
            "source_outcome_evaluation_ids": list(result.drift_check.source_outcome_evaluation_ids),
            "limitations": list(result.drift_check.limitations),
        }

    def inspect_phase6_run(self, *, run_id: str) -> JsonObject:
        run = self._require_run(run_id)
        outcome_evaluations = self.store.list_outcome_evaluations_for_run(run_id)
        calibration_runs = self.store.list_calibration_runs_for_run(run_id)
        calibration_drift_checks = self.store.list_calibration_drift_checks_for_run(run_id)
        tool_runs = self.store.list_tool_runs_for_run(run_id)
        registered_tool_names = _registry_tool_run_names(self.registry)
        raw_status_counts: dict[str, int] = {}
        for record in outcome_evaluations:
            raw_status_counts[record.status] = raw_status_counts.get(record.status, 0) + 1
        return {
            "run_id": run.run_id,
            "status": run.status,
            "outcome_evaluation_count": len(outcome_evaluations),
            "outcome_evaluation_status_counts": cast(JsonObject, raw_status_counts),
            "calibration_run_count": len(calibration_runs),
            "calibration_drift_check_count": len(calibration_drift_checks),
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
            "calibration_drift_checks": [
                {
                    "drift_check_id": record.drift_check_id,
                    "drift_status": record.drift_status,
                    "artifact_id": record.artifact_id,
                    "prior_calibration_id": record.prior_calibration_id,
                    "current_calibration_id": record.current_calibration_id,
                    "source_calibration_artifact_ids": list(record.source_calibration_artifact_ids),
                    "limitations": list(record.limitations),
                }
                for record in calibration_drift_checks
            ],
        }

    def _require_run(self, run_id: str) -> ResearchRunRecord:
        run = self.store.get_research_run(run_id)
        if run is None:
            raise ValueError(f"research run does not exist: {run_id}")
        return run

    def _require_live_run(self, run_id: str) -> ResearchRunRecord:
        run = self._require_run(run_id)
        mode = report_data_mode_from_run(run)
        if mode != LIVE_REPORT_DATA_MODE:
            raise ValueError(f"evaluation tool requires a live research run: {run_id}")
        violations = find_non_live_report_input_violations(store=self.store, run=run)
        if violations:
            details = "; ".join(violation.as_text() for violation in violations[:8])
            extra_count = len(violations) - 8
            if extra_count > 0:
                details = f"{details}; and {extra_count} more"
            raise ValueError("live evaluation tools cannot use non-live inputs: " + details)
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

    def _record_successful_tool_run(
        self,
        *,
        tool_run_id: str,
        run_id: str,
        tool_name: str,
        tool_version: str,
        at: datetime,
        inputs: JsonObject,
    ) -> None:
        self.store.record_tool_run(
            ToolRunRecord(
                tool_run_id=tool_run_id,
                run_id=run_id,
                tool_name=tool_name,
                tool_version=tool_version,
                status="successful",
                started_at=at,
                completed_at=at,
                inputs={**inputs, **report_data_mode_metadata_for_run_id(self.store, run_id)},
            )
        )


def _parse_aware_datetime(value: str, field_name: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone offset")
    return parsed.astimezone(UTC)


def _required_artifact_dir(artifact_dir: str | None) -> str:
    if artifact_dir is None or not artifact_dir.strip():
        raise ValueError("artifact_dir is required for public evaluation writer tools")
    return artifact_dir


def _parse_optional_aware_datetime(value: str | None, field_name: str) -> datetime | None:
    if value is None or not value.strip():
        return None
    return _parse_aware_datetime(value, field_name)


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


def _single_signal_family(value: str) -> SignalArtifactFamily:
    try:
        return SignalArtifactFamily(value.strip())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SignalArtifactFamily)
        raise ValueError(f"signal_family must be one of: {allowed}") from exc


def _non_empty_unique_strings(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    resolved: list[str] = []
    for value in values:
        item = value.strip()
        if not item:
            raise ValueError(f"{field_name} must not include empty values")
        resolved.append(item)
    return tuple(dict.fromkeys(resolved))


def _registry_tool_run_names(registry: Phase6ToolRegistry) -> set[str]:
    names: set[str] = set()
    for tool in registry.specs():
        names.add(tool.tool_name)
        source_tool_name = tool.metadata.get("source_tool_name")
        if isinstance(source_tool_name, str) and source_tool_name.strip():
            names.add(source_tool_name.strip())
    return names


def _outcome_review_summary_from_source(
    source: Phase6OutcomeEvaluationSource,
    *,
    created_at: datetime,
) -> OutcomeReviewSummary:
    target = source.payload.target
    outcome_evaluation = source.payload.outcome_evaluation
    outcome = outcome_evaluation.outcome
    phase7_records = phase7_freshness_records_from_target(target)
    outcome_evidence_ids = tuple(
        dict.fromkeys(
            (
                *(reference.evidence_id for reference in outcome.outcome_evidence),
                *(reference.evidence_id for reference in outcome_evaluation.evidence),
            )
        )
    )
    artifact_ids = tuple(
        dict.fromkeys(
            (
                source.artifact.artifact_id,
                *(artifact.artifact_id for artifact in target.signal_artifacts),
                *outcome.artifact_ids,
                *outcome_evaluation.artifact_ids,
            )
        )
    )
    summary_material = "|".join(
        (
            source.record.run_id or target.run_id,
            outcome_evaluation.outcome_evaluation_id,
            created_at.isoformat(),
        )
    )
    limitations = tuple(
        dict.fromkeys(
            (
                *outcome.limitations,
                *outcome_evaluation.limitations,
                *phase7_records.limitations,
            )
        )
    )
    if outcome_evaluation.quality_score is None and not limitations:
        limitations = ("Outcome evaluation has not resolved to a quality score.",)
    return OutcomeReviewSummary(
        summary_id=(
            "outcome-summary-"
            f"{slug(outcome_evaluation.outcome_evaluation_id)}-{digest(summary_material)[:8]}"
        ),
        run_id=source.record.run_id or target.run_id,
        candidate_id=outcome_evaluation.candidate_id,
        instrument_id=outcome_evaluation.instrument_id,
        symbol=outcome_evaluation.symbol,
        prediction_type=target.prediction_type,
        horizon=target.horizon,
        direction=target.direction or Direction.UNKNOWN,
        created_at=created_at,
        prior_run_id=None,
        outcome_id=outcome.outcome_id,
        outcome_evaluation_id=outcome_evaluation.outcome_evaluation_id,
        outcome_status=outcome.status,
        outcome_evaluation_status=outcome_evaluation.status,
        quality_score=outcome_evaluation.quality_score,
        outcome_evidence_ids=outcome_evidence_ids,
        artifact_ids=artifact_ids,
        evidence_aging_records=phase7_records.evidence_aging_records,
        artifact_freshness_reviews=phase7_records.artifact_freshness_reviews,
        source_calibration_artifact_ids=(),
        limitations=limitations,
        metadata={
            "target_id": target.target_id,
            "source_artifact_id": source.artifact.artifact_id,
            "source_artifact_path": source.artifact_path.as_posix(),
        },
    )


def _evaluation_tool_run_id(
    *,
    tool_name: str,
    run_id: str,
    created_at: datetime,
    material: Sequence[str],
) -> str:
    resolved_material = "|".join((tool_name, run_id, created_at.isoformat(), *material))
    return (
        f"tool-{slug(tool_name)}-{slug(run_id, allow_file_safe_punctuation=True)}-"
        f"{digest(resolved_material)[:8]}"
    )


def _evaluation_artifact_id(
    *,
    prefix: str,
    run_id: str,
    created_at: datetime,
    material: Sequence[str],
) -> str:
    resolved_material = "|".join((prefix, run_id, created_at.isoformat(), *material))
    return (
        f"artifact-{prefix}-{slug(run_id, allow_file_safe_punctuation=True)}-"
        f"{digest(resolved_material)[:8]}"
    )


__all__ = [
    "EVALUATION_ABLATION_TOOL_NAME",
    "EVALUATION_CALIBRATION_DRIFT_TOOL_NAME",
    "EVALUATION_CALIBRATION_TOOL_NAME",
    "EVALUATION_INSPECT_TOOL_NAME",
    "EVALUATION_LOAD_OUTCOMES_TOOL_NAME",
    "EVALUATION_MATERIALIZE_OUTCOME_TOOL_NAME",
    "EVALUATION_OUTCOME_SUMMARY_TOOL_NAME",
    "EVALUATION_PROVIDER_PLAYBOOK_TOOL_NAME",
    "EVALUATION_SOURCE_RELIABILITY_TOOL_NAME",
    "EVALUATION_STALE_ARTIFACTS_TOOL_NAME",
    "EVALUATION_WALK_FORWARD_TOOL_NAME",
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
    "PHASE7_LIVE_OUTCOME_MATERIALIZATION_PUBLIC_TOOL_NAME",
    "PHASE7_LIVE_OUTCOME_MATERIALIZATION_TOOL_ID",
    "Phase6OutcomeEvaluationSource",
    "Phase6Service",
    "Phase6ToolMetadata",
    "Phase6ToolRegistry",
    "build_phase6_tool_registry",
    "load_phase6_outcome_evaluation_sources",
    "phase6_evaluation_tool_plan",
]
