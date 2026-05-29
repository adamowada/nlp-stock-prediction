"""FastMCP registration adapter for public evaluation tools."""

from __future__ import annotations

from typing import Any

from nlp_stock_prediction.evaluation.calibration import DEFAULT_CALIBRATION_BIN_EDGES
from nlp_stock_prediction.orchestration.evaluation_service import (
    EvaluationService,
    build_evaluation_tool_registry,
)

EVALUATION_MCP_TOOL_NAMES: tuple[str, ...] = (
    "list_evaluation_tool_plan",
    *(tool.tool_name for tool in build_evaluation_tool_registry().specs()),
)
EVALUATION_MCP_TOOL_NAMES = EVALUATION_MCP_TOOL_NAMES


def register_evaluation_mcp_tools(server: Any, service: EvaluationService) -> None:
    """Register public evaluation tools on a FastMCP-compatible server."""

    def list_evaluation_tool_plan() -> dict[str, object]:
        """List the real public evaluation tools Codex should call."""

        return dict(service.list_evaluation_tool_plan())

    def evaluation_materialize_outcome(
        run_id: str,
        candidate_id: str,
        point_in_time_cutoff: str,
        evaluation_window_start: str,
        evaluation_window_end: str,
        artifact_dir: str,
        report_date: str | None = None,
        market_artifact_ids: list[str] | None = None,
        created_at: str | None = None,
        evaluated_at: str | None = None,
    ) -> dict[str, object]:
        """Materialize an outcome from real post-window market data."""

        return dict(
            service.evaluation_materialize_outcome(
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

    def evaluation_load_outcomes(run_id: str) -> dict[str, object]:
        """Load persisted outcome-evaluation artifacts for a research run."""

        return dict(service.evaluation_load_outcomes(run_id=run_id))

    def evaluation_outcome_summary(
        run_id: str,
        artifact_dir: str,
        created_at: str | None = None,
    ) -> dict[str, object]:
        """Persist cross-run outcome review summaries from stored outcomes."""

        return dict(
            service.evaluation_outcome_summary(
                run_id=run_id,
                artifact_dir=artifact_dir,
                created_at=created_at,
            )
        )

    def evaluation_stale_artifacts(
        run_id: str,
        artifact_dir: str,
        reviewed_at: str | None = None,
    ) -> dict[str, object]:
        """Persist artifact freshness reviews for a research run."""

        return dict(
            service.evaluation_stale_artifacts(
                run_id=run_id,
                artifact_dir=artifact_dir,
                reviewed_at=reviewed_at,
            )
        )

    def evaluation_evidence_aging(
        run_id: str,
        artifact_dir: str,
        reviewed_at: str | None = None,
    ) -> dict[str, object]:
        """Persist evidence aging reviews for a research run."""

        return dict(
            service.evaluation_evidence_aging(
                run_id=run_id,
                artifact_dir=artifact_dir,
                reviewed_at=reviewed_at,
            )
        )

    def evaluation_source_reliability(
        run_id: str,
        artifact_dir: str,
        created_at: str | None = None,
    ) -> dict[str, object]:
        """Persist source reliability notes from stored live evidence."""

        return dict(
            service.evaluation_source_reliability(
                run_id=run_id,
                artifact_dir=artifact_dir,
                created_at=created_at,
            )
        )

    def evaluation_provider_playbook(
        run_id: str,
        artifact_dir: str,
        created_at: str | None = None,
    ) -> dict[str, object]:
        """Persist provider replacement playbooks."""

        return dict(
            service.evaluation_provider_playbook(
                run_id=run_id,
                artifact_dir=artifact_dir,
                created_at=created_at,
            )
        )

    def evaluation_ablation(
        run_id: str,
        cohort_id: str,
        point_in_time_cutoff: str,
        artifact_dir: str,
        families: list[str] | None = None,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> dict[str, object]:
        """Persist signal-family ablation metrics from stored outcome evaluations."""

        return dict(
            service.evaluation_ablation(
                run_id=run_id,
                cohort_id=cohort_id,
                point_in_time_cutoff=point_in_time_cutoff,
                artifact_dir=artifact_dir,
                families=families,
                prediction_type=prediction_type,
                horizon=horizon,
            )
        )

    def evaluation_walk_forward(
        run_id: str,
        cohort_id: str,
        point_in_time_cutoff: str,
        minimum_train_size: int,
        artifact_dir: str,
        test_size: int = 1,
        step_size: int = 1,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> dict[str, object]:
        """Persist chronological walk-forward folds from stored outcome evaluations."""

        return dict(
            service.evaluation_walk_forward(
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

    def evaluation_calibration(
        run_id: str,
        cohort_id: str,
        as_of: str,
        artifact_dir: str,
        bin_edges: list[float] | None = None,
        families: list[str] | None = None,
        prediction_type: str | None = None,
        horizon: str | None = None,
    ) -> dict[str, object]:
        """Persist calibration reliability bins from stored outcome evaluations."""

        return dict(
            service.evaluation_calibration(
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

    def evaluation_calibration_drift(
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
    ) -> dict[str, object]:
        """Persist a calibration drift check for two stored calibration summaries."""

        return dict(
            service.evaluation_calibration_drift(
                run_id=run_id,
                prior_calibration_id=prior_calibration_id,
                current_calibration_id=current_calibration_id,
                as_of=as_of,
                artifact_dir=artifact_dir,
                signal_family=signal_family,
                min_resolved_count=min_resolved_count,
                watch_delta=watch_delta,
                degraded_delta=degraded_delta,
                improved_delta=improved_delta,
            )
        )

    def evaluation_inspect(run_id: str) -> dict[str, object]:
        """Inspect stored evaluation counts for verification."""

        return dict(service.evaluation_inspect(run_id=run_id))

    functions = {
        "list_evaluation_tool_plan": list_evaluation_tool_plan,
        "evaluation_materialize_outcome": evaluation_materialize_outcome,
        "evaluation_load_outcomes": evaluation_load_outcomes,
        "evaluation_outcome_summary": evaluation_outcome_summary,
        "evaluation_stale_artifacts": evaluation_stale_artifacts,
        "evaluation_evidence_aging": evaluation_evidence_aging,
        "evaluation_source_reliability": evaluation_source_reliability,
        "evaluation_provider_playbook": evaluation_provider_playbook,
        "evaluation_ablation": evaluation_ablation,
        "evaluation_walk_forward": evaluation_walk_forward,
        "evaluation_calibration": evaluation_calibration,
        "evaluation_calibration_drift": evaluation_calibration_drift,
        "evaluation_inspect": evaluation_inspect,
    }
    for tool_name in EVALUATION_MCP_TOOL_NAMES:
        server.tool()(functions[tool_name])


__all__ = ["EVALUATION_MCP_TOOL_NAMES", "register_evaluation_mcp_tools"]
