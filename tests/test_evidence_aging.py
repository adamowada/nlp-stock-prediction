from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    Direction,
    FreshnessStatus,
    PredictionEvaluationTarget,
    PredictionOutcomeEvaluation,
    PredictionOutcomeStatus,
    PredictionType,
    SourceKind,
)
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.evaluation import (
    build_prediction_evaluation_target,
    build_prediction_outcome,
    evaluate_prediction_outcome,
)
from nlp_stock_prediction.evaluation.common import filter_targeted_outcome_inputs
from nlp_stock_prediction.evaluation.freshness import (
    FreshnessPolicy,
    review_evidence_aging,
    write_evidence_aging_summary_artifact,
)
from nlp_stock_prediction.storage import (
    ArtifactRecord,
    CandidateArtifactLinkRecord,
    EvidenceRecord,
    InstrumentRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    SQLiteStore,
)

RUN_ID = "run-reliability-aging"
INSTRUMENT_ID = "equity:NASDAQ:MSFT"
REVIEWED_AT = datetime(2026, 5, 14, 20, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 5, 13, 20, 0, tzinfo=UTC)
WINDOW_START = datetime(2026, 5, 14, 21, 0, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 21, 20, 0, tzinfo=UTC)


@dataclass(frozen=True)
class _TargetedOutcome:
    target: PredictionEvaluationTarget
    outcome_evaluation: PredictionOutcomeEvaluation


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "prediction-research.sqlite3")
    store.initialize()
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id=INSTRUMENT_ID,
            symbol="MSFT",
            asset_class="stock",
            name="Microsoft Corporation",
        )
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=RUN_ID,
            run_kind="daily_prediction_report",
            objective="Generate an evidence-backed MSFT prediction report.",
            status="completed",
            started_at=datetime(2026, 5, 13, 12, 0, tzinfo=UTC),
            completed_at=CUTOFF,
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    artifact_inputs: tuple[tuple[str, datetime, JsonObject], ...] = (
        (
            "artifact-msft-old-news",
            datetime(2026, 4, 20, 13, 0, tzinfo=UTC),
            {"provider": "Dow Jones Newswires"},
        ),
        (
            "artifact-msft-technical-date-only",
            datetime(2026, 5, 13, 20, 0, tzinfo=UTC),
            {
                "provider": "NASDAQ Basic",
                "signal_family": "technicals",
                "latest_usable_bar": "2026-05-13",
            },
        ),
    )
    for artifact_id, created_at, metadata in artifact_inputs:
        store.record_artifact(
            ArtifactRecord(
                artifact_id=artifact_id,
                artifact_type="normalized_evidence"
                if artifact_id.endswith("news")
                else "technical_package",
                path=Path(f"artifacts/evaluation/{artifact_id}.json"),
                sha256="d" * 64,
                schema_version="reliability-test.v1",
                produced_by="reliability_aging_review",
                record_count=1,
                created_at=created_at,
                metadata=metadata,
            )
        )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-msft-old-news",
            source_type=SourceKind.NEWS_ARTICLE.value,
            provider="Dow Jones Newswires",
            retrieved_at=datetime(2026, 4, 20, 14, 0, tzinfo=UTC),
            published_at=datetime(2026, 4, 20, 13, 0, tzinfo=UTC),
            instruments=(INSTRUMENT_ID,),
            claim="Older attributed evidence supported a prior MSFT scenario.",
            freshness_status=FreshnessStatus.FRESH.value,
            artifact_id="artifact-msft-old-news",
            url="https://www.dowjones.com/example/msft-older-context",
        )
    )
    store.record_evidence(
        EvidenceRecord(
            evidence_id="evidence-msft-provider-replaced",
            source_type=SourceKind.MARKET_DATA.value,
            provider="Legacy Market Feed",
            retrieved_at=datetime(2026, 5, 13, 19, 0, tzinfo=UTC),
            instruments=(INSTRUMENT_ID,),
            claim="Legacy feed captured the pre-window market state.",
            freshness_status=FreshnessStatus.FRESH.value,
            metadata={"replacement_provider": "NASDAQ Basic"},
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id="candidate-msft-aging",
            run_id=RUN_ID,
            instrument_id=INSTRUMENT_ID,
            prediction_horizon="swing",
            prediction_type=PredictionType.DIRECTIONAL.value,
            scenario="MSFT had an evidence-backed directional setup.",
            direction=Direction.BULLISH.value,
            confidence=0.61,
            status="evidence_supported",
            evidence_for=("evidence-msft-old-news", "evidence-msft-provider-replaced"),
            signal_artifacts=("artifact-msft-technical-date-only",),
            baseline={"baseline_id": "market-neutral", "baseline_score": 0.5},
        )
    )
    store.link_candidate_artifact(
        CandidateArtifactLinkRecord(
            candidate_id="candidate-msft-aging",
            artifact_id="artifact-msft-technical-date-only",
            relationship="signal",
            created_at=CUTOFF,
        )
    )
    return store


@pytest.mark.unit
def test_prior_evidence_ages_out_with_structured_limitations(tmp_path: Path) -> None:
    store = _store(tmp_path)

    record = review_evidence_aging(
        evidence=store.get_evidence("evidence-msft-old-news"),
        reviewed_at=REVIEWED_AT,
        policy=FreshnessPolicy(evidence_max_age=timedelta(days=7)),
    )

    assert record.age_status == "aged_out"
    assert record.freshness_status == FreshnessStatus.STALE
    assert record.source_artifact_id == "artifact-msft-old-news"
    assert any("exceeded" in item for item in record.limitations)


@pytest.mark.unit
def test_provider_replaced_evidence_links_old_and_new_sources(tmp_path: Path) -> None:
    store = _store(tmp_path)

    record = review_evidence_aging(
        evidence=store.get_evidence("evidence-msft-provider-replaced"),
        reviewed_at=REVIEWED_AT,
        replacement_provider="NASDAQ Basic",
        replacement_evidence_id="evidence-msft-nasdaq-replacement",
    )

    assert record.age_status == "provider_replaced"
    assert record.provider == "Legacy Market Feed"
    assert record.replacement_provider == "NASDAQ Basic"
    assert record.replacement_evidence_id == "evidence-msft-nasdaq-replacement"


@pytest.mark.unit
def test_missing_source_freshness_remains_missing_even_when_recent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    evidence = store.get_evidence("evidence-msft-provider-replaced")
    assert evidence is not None
    evidence = EvidenceRecord(
        **{
            **evidence.__dict__,
            "metadata": {},
            "freshness_status": FreshnessStatus.MISSING.value,
        }
    )

    record = review_evidence_aging(evidence=evidence, reviewed_at=REVIEWED_AT)

    assert record.age_status == "missing"
    assert record.freshness_status == FreshnessStatus.MISSING
    assert any("missing freshness" in item for item in record.limitations)


@pytest.mark.unit
def test_target_freezing_carries_structured_aging_and_freshness_metadata(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    target = build_prediction_evaluation_target(
        store=store,
        run_id=RUN_ID,
        candidate_id="candidate-msft-aging",
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
    )

    freshness = target.metadata["reliability_freshness"]
    assert isinstance(freshness, dict)
    aging_records = cast(list[JsonObject], freshness["evidence_aging_records"])
    artifact_reviews = cast(list[JsonObject], freshness["artifact_freshness_reviews"])
    assert isinstance(aging_records, list)
    assert isinstance(artifact_reviews, list)
    assert {record["evidence_id"] for record in aging_records} == {
        "evidence-msft-old-news",
        "evidence-msft-provider-replaced",
    }
    assert any(record["age_status"] == "aged_out" for record in aging_records)
    artifact_metadata = cast(JsonObject, artifact_reviews[0]["metadata"])
    assert artifact_metadata["date_only_as_of_normalized"] is True
    assert any("evidence-msft-old-news" in item for item in target.limitations)


@pytest.mark.unit
def test_cohort_selection_carries_structured_reliability_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = build_prediction_evaluation_target(
        store=store,
        run_id=RUN_ID,
        candidate_id="candidate-msft-aging",
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
    )
    outcome = build_prediction_outcome(
        target=target,
        status=PredictionOutcomeStatus.PENDING,
        limitations=("Evaluation window is not complete.",),
    )
    outcome_evaluation = evaluate_prediction_outcome(
        target=target,
        outcome=outcome,
        evaluated_at=WINDOW_START,
        limitations=("Evaluation window is not complete.",),
    )

    cohort = filter_targeted_outcome_inputs(
        (_TargetedOutcome(target=target, outcome_evaluation=outcome_evaluation),),
        as_of=WINDOW_START,
        prediction_type=None,
        horizon=None,
        cutoff_reason="Outcome evaluation after cutoff",
    )

    assert cohort.excluded_outcome_evaluation_ids == ()
    assert {record.evidence_id for record in cohort.reliability_evidence_aging_records} == {
        "evidence-msft-old-news",
        "evidence-msft-provider-replaced",
    }
    assert {review.artifact_id for review in cohort.reliability_artifact_freshness_reviews} == {
        "artifact-msft-old-news",
        "artifact-msft-technical-date-only",
    }
    evidence_artifact_review = next(
        review
        for review in cohort.reliability_artifact_freshness_reviews
        if review.artifact_id == "artifact-msft-old-news"
    )
    assert evidence_artifact_review.source_evidence_ids == ("evidence-msft-old-news",)


@pytest.mark.unit
def test_evidence_aging_summary_writes_indexed_artifact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    aging_record = review_evidence_aging(
        evidence=store.get_evidence("evidence-msft-old-news"),
        reviewed_at=REVIEWED_AT,
        policy=FreshnessPolicy(evidence_max_age=timedelta(days=7)),
    )

    written = write_evidence_aging_summary_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        aging_records=(aging_record,),
        created_at=REVIEWED_AT,
    )

    assert written.artifact.artifact_type == "evidence_aging_summary"
    assert store.get_artifact(written.artifact.artifact_id) is not None
    payload = json.loads(Path(written.artifact.path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "evidence-aging-summary-artifact.v1"
    assert payload["evidence_aging_records"][0]["age_status"] == "aged_out"
