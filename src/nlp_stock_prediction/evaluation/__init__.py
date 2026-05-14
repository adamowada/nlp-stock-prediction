"""Prediction-quality evaluation tools."""

from nlp_stock_prediction.evaluation.ablation import (
    PHASE6_ABLATION_TOOL_NAME,
    PHASE6_ABLATION_TOOL_VERSION,
    SignalFamilyAblationArtifactPayload,
    SignalFamilyAblationArtifacts,
    SignalFamilyAblationInput,
    compute_signal_family_ablations,
    signal_family_ablation_inputs_from_outcome_artifacts,
    write_signal_family_ablation_artifact,
)
from nlp_stock_prediction.evaluation.outcomes import (
    PHASE6_OUTCOME_TOOL_NAME,
    PHASE6_OUTCOME_TOOL_VERSION,
    PointInTimeOutcomeEvaluationArtifacts,
    build_prediction_evaluation_target,
    build_prediction_outcome,
    evaluate_prediction_outcome,
    write_point_in_time_outcome_evaluation_artifacts,
)
from nlp_stock_prediction.evaluation.prediction import (
    attach_evaluation_metadata,
    evaluate_prediction_candidate,
    write_prediction_evaluation_artifact,
)
from nlp_stock_prediction.evaluation.walk_forward import (
    PHASE6_WALK_FORWARD_TOOL_NAME,
    PHASE6_WALK_FORWARD_TOOL_VERSION,
    WalkForwardEvaluationArtifactPayload,
    WalkForwardEvaluationArtifacts,
    WalkForwardFold,
    compute_walk_forward_folds,
    walk_forward_evaluations_from_outcome_artifacts,
    write_walk_forward_evaluation_artifact,
)

__all__ = [
    "PHASE6_ABLATION_TOOL_NAME",
    "PHASE6_ABLATION_TOOL_VERSION",
    "PHASE6_OUTCOME_TOOL_NAME",
    "PHASE6_OUTCOME_TOOL_VERSION",
    "PHASE6_WALK_FORWARD_TOOL_NAME",
    "PHASE6_WALK_FORWARD_TOOL_VERSION",
    "PointInTimeOutcomeEvaluationArtifacts",
    "SignalFamilyAblationArtifactPayload",
    "SignalFamilyAblationArtifacts",
    "SignalFamilyAblationInput",
    "WalkForwardEvaluationArtifactPayload",
    "WalkForwardEvaluationArtifacts",
    "WalkForwardFold",
    "attach_evaluation_metadata",
    "build_prediction_evaluation_target",
    "build_prediction_outcome",
    "compute_signal_family_ablations",
    "compute_walk_forward_folds",
    "evaluate_prediction_candidate",
    "evaluate_prediction_outcome",
    "signal_family_ablation_inputs_from_outcome_artifacts",
    "walk_forward_evaluations_from_outcome_artifacts",
    "write_point_in_time_outcome_evaluation_artifacts",
    "write_prediction_evaluation_artifact",
    "write_signal_family_ablation_artifact",
    "write_walk_forward_evaluation_artifact",
]
