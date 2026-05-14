"""FastMCP registration adapter for Phase 6 evaluation and calibration tools."""

from __future__ import annotations

from typing import Any

from nlp_stock_prediction.evaluation.calibration import DEFAULT_CALIBRATION_BIN_EDGES
from nlp_stock_prediction.orchestration.phase6_service import (
    Phase6Service,
    build_phase6_tool_registry,
)

PHASE6_MCP_TOOL_NAMES: tuple[str, ...] = (
    "list_phase6_tool_plan",
    *(tool.tool_name for tool in build_phase6_tool_registry().specs()),
)


def register_phase6_mcp_tools(server: Any, service: Phase6Service) -> None:
    """Register Phase 6 tools on a FastMCP-compatible server."""

    def list_phase6_tool_plan() -> dict[str, object]:
        """List the real Phase 6 tools Codex should call."""

        return dict(service.list_phase6_tool_plan())

    def phase7_live_outcome_materialization(
        run_id: str,
        candidate_id: str,
        point_in_time_cutoff: str,
        evaluation_window_start: str,
        evaluation_window_end: str,
        artifact_dir: str | None = None,
        report_date: str | None = None,
        market_artifact_ids: list[str] | None = None,
        created_at: str | None = None,
        evaluated_at: str | None = None,
    ) -> dict[str, object]:
        """Materialize an outcome from real post-window market data."""

        return dict(
            service.phase7_live_outcome_materialization(
                run_id=run_id,
                candidate_id=candidate_id,
                point_in_time_cutoff=point_in_time_cutoff,
                evaluation_window_start=evaluation_window_start,
                evaluation_window_end=evaluation_window_end,
                artifact_dir=artifact_dir,
                report_date=report_date,
                market_artifact_ids=() if market_artifact_ids is None else market_artifact_ids,
                created_at=created_at,
                evaluated_at=evaluated_at,
            )
        )

    def phase6_point_in_time_outcome_evaluation(
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
        outcome_evidence_ids: list[str] | None = None,
        market_artifact_ids: list[str] | None = None,
        limitations: list[str] | None = None,
        created_at: str | None = None,
        evaluated_at: str | None = None,
    ) -> dict[str, object]:
        """Persist a point-in-time outcome evaluation for a stored prediction candidate."""

        return dict(
            service.phase6_point_in_time_outcome_evaluation(
                run_id=run_id,
                candidate_id=candidate_id,
                point_in_time_cutoff=point_in_time_cutoff,
                evaluation_window_start=evaluation_window_start,
                evaluation_window_end=evaluation_window_end,
                artifact_dir=artifact_dir,
                report_date=report_date,
                status=status,
                observed_result=observed_result,
                observed_at=observed_at,
                result_summary=result_summary,
                result_value=result_value,
                baseline_value=baseline_value,
                outcome_evidence_ids=() if outcome_evidence_ids is None else outcome_evidence_ids,
                market_artifact_ids=() if market_artifact_ids is None else market_artifact_ids,
                limitations=() if limitations is None else limitations,
                created_at=created_at,
                evaluated_at=evaluated_at,
            )
        )

    def phase6_load_outcome_evaluations(run_id: str) -> dict[str, object]:
        """Load persisted outcome-evaluation artifacts for a research run."""

        return dict(service.phase6_load_outcome_evaluations(run_id=run_id))

    def phase6_signal_family_ablation(
        run_id: str,
        cohort_id: str,
        point_in_time_cutoff: str,
        artifact_dir: str | None = None,
        families: list[str] | None = None,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> dict[str, object]:
        """Persist signal-family ablation metrics from stored outcome evaluations."""

        return dict(
            service.phase6_signal_family_ablation(
                run_id=run_id,
                cohort_id=cohort_id,
                point_in_time_cutoff=point_in_time_cutoff,
                artifact_dir=artifact_dir,
                families=families,
                prediction_type=prediction_type,
                horizon=horizon,
            )
        )

    def phase6_walk_forward_evaluation(
        run_id: str,
        cohort_id: str,
        point_in_time_cutoff: str,
        minimum_train_size: int,
        artifact_dir: str | None = None,
        test_size: int = 1,
        step_size: int = 1,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> dict[str, object]:
        """Persist chronological walk-forward folds from stored outcome evaluations."""

        return dict(
            service.phase6_walk_forward_evaluation(
                run_id=run_id,
                cohort_id=cohort_id,
                point_in_time_cutoff=point_in_time_cutoff,
                minimum_train_size=minimum_train_size,
                artifact_dir=artifact_dir,
                test_size=test_size,
                step_size=step_size,
                prediction_type=prediction_type,
                horizon=horizon,
            )
        )

    def phase6_calibration_summary(
        run_id: str,
        cohort_id: str,
        as_of: str,
        artifact_dir: str | None = None,
        bin_edges: list[float] | None = None,
        families: list[str] | None = None,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> dict[str, object]:
        """Persist calibration reliability bins from stored outcome evaluations."""

        return dict(
            service.phase6_calibration_summary(
                run_id=run_id,
                cohort_id=cohort_id,
                as_of=as_of,
                artifact_dir=artifact_dir,
                bin_edges=(
                    DEFAULT_CALIBRATION_BIN_EDGES if bin_edges is None else tuple(bin_edges)
                ),
                families=families,
                prediction_type=prediction_type,
                horizon=horizon,
            )
        )

    def inspect_phase6_run(run_id: str) -> dict[str, object]:
        """Inspect stored Phase 6 counts for verification."""

        return dict(service.inspect_phase6_run(run_id=run_id))

    functions = {
        "list_phase6_tool_plan": list_phase6_tool_plan,
        "phase7_live_outcome_materialization": phase7_live_outcome_materialization,
        "phase6_point_in_time_outcome_evaluation": phase6_point_in_time_outcome_evaluation,
        "phase6_load_outcome_evaluations": phase6_load_outcome_evaluations,
        "phase6_signal_family_ablation": phase6_signal_family_ablation,
        "phase6_walk_forward_evaluation": phase6_walk_forward_evaluation,
        "phase6_calibration_summary": phase6_calibration_summary,
        "inspect_phase6_run": inspect_phase6_run,
    }
    for tool_name in PHASE6_MCP_TOOL_NAMES:
        server.tool()(functions[tool_name])


__all__ = ["PHASE6_MCP_TOOL_NAMES", "register_phase6_mcp_tools"]
