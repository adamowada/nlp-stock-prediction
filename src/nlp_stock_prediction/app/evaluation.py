"""Evaluation command builder and dispatcher for the terminal app."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.evaluation.calibration import DEFAULT_CALIBRATION_BIN_EDGES
from nlp_stock_prediction.orchestration.phase6_service import Phase6Service


class EvaluationServiceProtocol(Protocol):
    def evaluation_inspect(self, *, run_id: str) -> JsonObject: ...

    def evaluation_materialize_outcome(
        self,
        *,
        run_id: str,
        candidate_id: str,
        point_in_time_cutoff: str,
        evaluation_window_start: str,
        evaluation_window_end: str,
        artifact_dir: str,
        report_date: str | None = None,
        market_artifact_ids: tuple[str, ...] = (),
        created_at: str | None = None,
        evaluated_at: str | None = None,
    ) -> JsonObject: ...

    def evaluation_load_outcomes(self, *, run_id: str) -> JsonObject: ...

    def evaluation_outcome_summary(
        self,
        *,
        run_id: str,
        artifact_dir: str,
        created_at: str | None = None,
    ) -> JsonObject: ...

    def evaluation_stale_artifacts(
        self,
        *,
        run_id: str,
        artifact_dir: str,
        reviewed_at: str | None = None,
    ) -> JsonObject: ...

    def evaluation_evidence_aging(
        self,
        *,
        run_id: str,
        artifact_dir: str,
        reviewed_at: str | None = None,
    ) -> JsonObject: ...

    def evaluation_source_reliability(
        self,
        *,
        run_id: str,
        artifact_dir: str,
        created_at: str | None = None,
    ) -> JsonObject: ...

    def evaluation_provider_playbook(
        self,
        *,
        run_id: str,
        artifact_dir: str,
        created_at: str | None = None,
    ) -> JsonObject: ...

    def evaluation_calibration(
        self,
        *,
        run_id: str,
        cohort_id: str,
        as_of: str,
        artifact_dir: str,
        bin_edges: tuple[float, ...] = DEFAULT_CALIBRATION_BIN_EDGES,
        families: list[str] | None = None,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> JsonObject: ...

    def evaluation_walk_forward(
        self,
        *,
        run_id: str,
        cohort_id: str,
        point_in_time_cutoff: str,
        minimum_train_size: int,
        artifact_dir: str,
        test_size: int = 1,
        step_size: int = 1,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> JsonObject: ...

    def evaluation_ablation(
        self,
        *,
        run_id: str,
        cohort_id: str,
        point_in_time_cutoff: str,
        artifact_dir: str,
        families: list[str] | None = None,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> JsonObject: ...

    def evaluation_calibration_drift(
        self,
        *,
        run_id: str,
        prior_calibration_id: str,
        current_calibration_id: str,
        as_of: str,
        artifact_dir: str,
        signal_family: str | None = None,
        min_resolved_count: int = 10,
        watch_delta: float = 0.05,
        degraded_delta: float = 0.10,
        improved_delta: float = 0.10,
    ) -> JsonObject: ...


@dataclass(frozen=True)
class EvaluationField:
    name: str
    label: str
    required: bool = True
    default: str | None = None


@dataclass(frozen=True)
class EvaluationCommandSpec:
    name: str
    label: str
    fields: tuple[EvaluationField, ...]


COMMON_RUN_FIELD = EvaluationField("run_id", "Run ID")
ARTIFACT_FIELD = EvaluationField("artifact_dir", "Audit/artifact directory")

EVALUATION_COMMAND_SPECS: tuple[EvaluationCommandSpec, ...] = (
    EvaluationCommandSpec("inspect", "Inspect run", (COMMON_RUN_FIELD,)),
    EvaluationCommandSpec(
        "materialize-outcome",
        "Materialize live outcome",
        (
            COMMON_RUN_FIELD,
            EvaluationField("candidate_id", "Candidate ID"),
            EvaluationField("point_in_time_cutoff", "Point-in-time cutoff"),
            EvaluationField("evaluation_window_start", "Evaluation window start"),
            EvaluationField("evaluation_window_end", "Evaluation window end"),
            ARTIFACT_FIELD,
            EvaluationField("report_date", "Report date", required=False),
            EvaluationField("market_artifact_ids", "Market artifact IDs, comma-separated", False),
            EvaluationField("created_at", "Created at", required=False),
            EvaluationField("evaluated_at", "Evaluated at", required=False),
        ),
    ),
    EvaluationCommandSpec("load-outcomes", "Load outcomes", (COMMON_RUN_FIELD,)),
    EvaluationCommandSpec(
        "outcome-summary",
        "Outcome summary",
        (COMMON_RUN_FIELD, ARTIFACT_FIELD, EvaluationField("created_at", "Created at", False)),
    ),
    EvaluationCommandSpec(
        "stale-artifacts",
        "Artifact freshness",
        (COMMON_RUN_FIELD, ARTIFACT_FIELD, EvaluationField("reviewed_at", "Reviewed at", False)),
    ),
    EvaluationCommandSpec(
        "evidence-aging",
        "Evidence aging",
        (COMMON_RUN_FIELD, ARTIFACT_FIELD, EvaluationField("reviewed_at", "Reviewed at", False)),
    ),
    EvaluationCommandSpec(
        "source-reliability",
        "Source reliability",
        (COMMON_RUN_FIELD, ARTIFACT_FIELD, EvaluationField("created_at", "Created at", False)),
    ),
    EvaluationCommandSpec(
        "provider-playbook",
        "Provider playbook",
        (COMMON_RUN_FIELD, ARTIFACT_FIELD, EvaluationField("created_at", "Created at", False)),
    ),
    EvaluationCommandSpec(
        "calibration",
        "Calibration",
        (
            COMMON_RUN_FIELD,
            EvaluationField("cohort_id", "Cohort ID"),
            EvaluationField("as_of", "As of"),
            ARTIFACT_FIELD,
            EvaluationField("bin_edges", "Bin edges, comma-separated", False),
            EvaluationField("families", "Signal families, comma-separated", False),
            EvaluationField("prediction_type", "Prediction type", False),
            EvaluationField("horizon", "Horizon", False),
        ),
    ),
    EvaluationCommandSpec(
        "walk-forward",
        "Walk-forward",
        (
            COMMON_RUN_FIELD,
            EvaluationField("cohort_id", "Cohort ID"),
            EvaluationField("point_in_time_cutoff", "Point-in-time cutoff"),
            EvaluationField("minimum_train_size", "Minimum train size"),
            ARTIFACT_FIELD,
            EvaluationField("test_size", "Test size", False, "1"),
            EvaluationField("step_size", "Step size", False, "1"),
            EvaluationField("prediction_type", "Prediction type", False),
            EvaluationField("horizon", "Horizon", False),
        ),
    ),
    EvaluationCommandSpec(
        "ablation",
        "Signal-family ablation",
        (
            COMMON_RUN_FIELD,
            EvaluationField("cohort_id", "Cohort ID"),
            EvaluationField("point_in_time_cutoff", "Point-in-time cutoff"),
            ARTIFACT_FIELD,
            EvaluationField("families", "Signal families, comma-separated", False),
            EvaluationField("prediction_type", "Prediction type", False),
            EvaluationField("horizon", "Horizon", False),
        ),
    ),
    EvaluationCommandSpec(
        "calibration-drift",
        "Calibration drift",
        (
            COMMON_RUN_FIELD,
            EvaluationField("prior_calibration_id", "Prior calibration ID"),
            EvaluationField("current_calibration_id", "Current calibration ID"),
            EvaluationField("as_of", "As of"),
            ARTIFACT_FIELD,
            EvaluationField("signal_family", "Signal family", False),
            EvaluationField("min_resolved_count", "Min resolved count", False, "10"),
            EvaluationField("watch_delta", "Watch delta", False, "0.05"),
            EvaluationField("degraded_delta", "Degraded delta", False, "0.10"),
            EvaluationField("improved_delta", "Improved delta", False, "0.10"),
        ),
    ),
)


def command_spec(name: str) -> EvaluationCommandSpec:
    for spec in EVALUATION_COMMAND_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(f"unknown evaluation command: {name}")


def build_evaluation_service(
    repo_root: Path,
    database_path: Path,
    extra_write_roots: tuple[Path, ...] = (),
) -> Phase6Service:
    return Phase6Service(
        repo_root=repo_root,
        database_path=database_path,
        extra_write_roots=extra_write_roots,
    )


def run_evaluation_action(
    service: EvaluationServiceProtocol,
    command: str,
    values: Mapping[str, object],
) -> JsonObject:
    if command == "inspect":
        return service.evaluation_inspect(run_id=_required(values, "run_id"))
    if command == "materialize-outcome":
        return service.evaluation_materialize_outcome(
            run_id=_required(values, "run_id"),
            candidate_id=_required(values, "candidate_id"),
            point_in_time_cutoff=_required(values, "point_in_time_cutoff"),
            evaluation_window_start=_required(values, "evaluation_window_start"),
            evaluation_window_end=_required(values, "evaluation_window_end"),
            artifact_dir=_required(values, "artifact_dir"),
            report_date=_optional(values, "report_date"),
            market_artifact_ids=tuple(_list(values, "market_artifact_ids")),
            created_at=_optional(values, "created_at"),
            evaluated_at=_optional(values, "evaluated_at"),
        )
    if command == "load-outcomes":
        return service.evaluation_load_outcomes(run_id=_required(values, "run_id"))
    if command == "outcome-summary":
        return service.evaluation_outcome_summary(
            run_id=_required(values, "run_id"),
            artifact_dir=_required(values, "artifact_dir"),
            created_at=_optional(values, "created_at"),
        )
    if command == "stale-artifacts":
        return service.evaluation_stale_artifacts(
            run_id=_required(values, "run_id"),
            artifact_dir=_required(values, "artifact_dir"),
            reviewed_at=_optional(values, "reviewed_at"),
        )
    if command == "evidence-aging":
        return service.evaluation_evidence_aging(
            run_id=_required(values, "run_id"),
            artifact_dir=_required(values, "artifact_dir"),
            reviewed_at=_optional(values, "reviewed_at"),
        )
    if command == "source-reliability":
        return service.evaluation_source_reliability(
            run_id=_required(values, "run_id"),
            artifact_dir=_required(values, "artifact_dir"),
            created_at=_optional(values, "created_at"),
        )
    if command == "provider-playbook":
        return service.evaluation_provider_playbook(
            run_id=_required(values, "run_id"),
            artifact_dir=_required(values, "artifact_dir"),
            created_at=_optional(values, "created_at"),
        )
    if command == "calibration":
        bin_edges = _float_tuple(values, "bin_edges") or DEFAULT_CALIBRATION_BIN_EDGES
        return service.evaluation_calibration(
            run_id=_required(values, "run_id"),
            cohort_id=_required(values, "cohort_id"),
            as_of=_required(values, "as_of"),
            artifact_dir=_required(values, "artifact_dir"),
            bin_edges=bin_edges,
            families=_optional_list(values, "families"),
            prediction_type=_optional(values, "prediction_type"),
            horizon=_optional(values, "horizon"),
        )
    if command == "walk-forward":
        return service.evaluation_walk_forward(
            run_id=_required(values, "run_id"),
            cohort_id=_required(values, "cohort_id"),
            point_in_time_cutoff=_required(values, "point_in_time_cutoff"),
            minimum_train_size=_int(values, "minimum_train_size"),
            artifact_dir=_required(values, "artifact_dir"),
            test_size=_int(values, "test_size", 1),
            step_size=_int(values, "step_size", 1),
            prediction_type=_optional(values, "prediction_type"),
            horizon=_optional(values, "horizon"),
        )
    if command == "ablation":
        return service.evaluation_ablation(
            run_id=_required(values, "run_id"),
            cohort_id=_required(values, "cohort_id"),
            point_in_time_cutoff=_required(values, "point_in_time_cutoff"),
            artifact_dir=_required(values, "artifact_dir"),
            families=_optional_list(values, "families"),
            prediction_type=_optional(values, "prediction_type"),
            horizon=_optional(values, "horizon"),
        )
    if command == "calibration-drift":
        return service.evaluation_calibration_drift(
            run_id=_required(values, "run_id"),
            prior_calibration_id=_required(values, "prior_calibration_id"),
            current_calibration_id=_required(values, "current_calibration_id"),
            as_of=_required(values, "as_of"),
            artifact_dir=_required(values, "artifact_dir"),
            signal_family=_optional(values, "signal_family"),
            min_resolved_count=_int(values, "min_resolved_count", 10),
            watch_delta=_float(values, "watch_delta", 0.05),
            degraded_delta=_float(values, "degraded_delta", 0.10),
            improved_delta=_float(values, "improved_delta", 0.10),
        )
    raise ValueError(f"unknown evaluation command: {command}")


def _required(values: Mapping[str, object], key: str) -> str:
    value = _optional(values, key)
    if value is None:
        raise ValueError(f"{key} is required")
    return value


def _optional(values: Mapping[str, object], key: str) -> str | None:
    value = values.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    return None


def _list(values: Mapping[str, object], key: str) -> list[str]:
    value = values.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _optional_list(values: Mapping[str, object], key: str) -> list[str] | None:
    items = _list(values, key)
    return items or None


def _float_tuple(values: Mapping[str, object], key: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _list(values, key))


def _int(values: Mapping[str, object], key: str, default: int | None = None) -> int:
    text = _optional(values, key)
    if text is None:
        if default is None:
            raise ValueError(f"{key} is required")
        return default
    return int(text)


def _float(values: Mapping[str, object], key: str, default: float) -> float:
    text = _optional(values, key)
    return default if text is None else float(text)


def as_json_object(value: object) -> JsonObject:
    return cast(JsonObject, value)


__all__ = [
    "EVALUATION_COMMAND_SPECS",
    "EvaluationCommandSpec",
    "EvaluationField",
    "EvaluationServiceProtocol",
    "build_evaluation_service",
    "command_spec",
    "run_evaluation_action",
]
