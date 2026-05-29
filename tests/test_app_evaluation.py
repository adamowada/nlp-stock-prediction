from __future__ import annotations

from typing import Any

import pytest

from nlp_stock_prediction.app.evaluation import EVALUATION_COMMAND_SPECS, run_evaluation_action
from nlp_stock_prediction.contracts.base import JsonObject

pytestmark = pytest.mark.unit


class FakeEvaluationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _record(self, name: str, **kwargs: object) -> JsonObject:
        self.calls.append((name, kwargs))
        return {"command": name, "run_id": str(kwargs.get("run_id", ""))}

    def evaluation_inspect(self, *, run_id: str) -> JsonObject:
        return self._record("inspect", run_id=run_id)

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
    ) -> JsonObject:
        return self._record(
            "materialize-outcome",
            run_id=run_id,
            candidate_id=candidate_id,
            point_in_time_cutoff=point_in_time_cutoff,
            evaluation_window_start=evaluation_window_start,
            evaluation_window_end=evaluation_window_end,
            artifact_dir=artifact_dir,
            report_date=report_date,
            market_artifact_ids=market_artifact_ids,
            created_at=created_at,
            evaluated_at=evaluated_at,
        )

    def evaluation_load_outcomes(self, *, run_id: str) -> JsonObject:
        return self._record("load-outcomes", run_id=run_id)

    def evaluation_outcome_summary(
        self,
        *,
        run_id: str,
        artifact_dir: str,
        created_at: str | None = None,
    ) -> JsonObject:
        return self._record(
            "outcome-summary", run_id=run_id, artifact_dir=artifact_dir, created_at=created_at
        )

    def evaluation_stale_artifacts(
        self,
        *,
        run_id: str,
        artifact_dir: str,
        reviewed_at: str | None = None,
    ) -> JsonObject:
        return self._record(
            "stale-artifacts", run_id=run_id, artifact_dir=artifact_dir, reviewed_at=reviewed_at
        )

    def evaluation_evidence_aging(
        self,
        *,
        run_id: str,
        artifact_dir: str,
        reviewed_at: str | None = None,
    ) -> JsonObject:
        return self._record(
            "evidence-aging", run_id=run_id, artifact_dir=artifact_dir, reviewed_at=reviewed_at
        )

    def evaluation_source_reliability(
        self,
        *,
        run_id: str,
        artifact_dir: str,
        created_at: str | None = None,
    ) -> JsonObject:
        return self._record(
            "source-reliability", run_id=run_id, artifact_dir=artifact_dir, created_at=created_at
        )

    def evaluation_provider_playbook(
        self,
        *,
        run_id: str,
        artifact_dir: str,
        created_at: str | None = None,
    ) -> JsonObject:
        return self._record(
            "provider-playbook", run_id=run_id, artifact_dir=artifact_dir, created_at=created_at
        )

    def evaluation_calibration(self, **kwargs: Any) -> JsonObject:
        return self._record("calibration", **kwargs)

    def evaluation_walk_forward(self, **kwargs: Any) -> JsonObject:
        return self._record("walk-forward", **kwargs)

    def evaluation_ablation(self, **kwargs: Any) -> JsonObject:
        return self._record("ablation", **kwargs)

    def evaluation_calibration_drift(self, **kwargs: Any) -> JsonObject:
        return self._record("calibration-drift", **kwargs)


def test_evaluation_dispatch_exposes_full_command_set() -> None:
    assert [spec.name for spec in EVALUATION_COMMAND_SPECS] == [
        "inspect",
        "materialize-outcome",
        "load-outcomes",
        "outcome-summary",
        "stale-artifacts",
        "evidence-aging",
        "source-reliability",
        "provider-playbook",
        "calibration",
        "walk-forward",
        "ablation",
        "calibration-drift",
    ]


def test_evaluation_dispatch_maps_full_builder_values() -> None:
    service = FakeEvaluationService()

    result = run_evaluation_action(
        service,
        "walk-forward",
        {
            "run_id": "run-1",
            "cohort_id": "cohort",
            "point_in_time_cutoff": "2026-05-15T00:00:00+00:00",
            "minimum_train_size": 3,
            "artifact_dir": "reports/run/audit",
            "test_size": 2,
            "step_size": 1,
        },
    )

    assert result == {"command": "walk-forward", "run_id": "run-1"}
    assert service.calls[-1][1]["minimum_train_size"] == 3
    assert service.calls[-1][1]["test_size"] == 2


def test_evaluation_dispatch_parses_lists_and_optional_values() -> None:
    service = FakeEvaluationService()

    run_evaluation_action(
        service,
        "calibration",
        {
            "run_id": "run-1",
            "cohort_id": "cohort",
            "as_of": "2026-05-22T00:00:00+00:00",
            "artifact_dir": "reports/run/audit",
            "bin_edges": "0,0.5,1",
            "families": "news,technicals",
        },
    )

    call = service.calls[-1][1]
    assert call["bin_edges"] == (0.0, 0.5, 1.0)
    assert call["families"] == ["news", "technicals"]
