from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.evaluation.freshness import (
    FreshnessPolicy,
    review_artifact_file_freshness,
    review_artifact_freshness,
    write_artifact_freshness_review_artifact,
)
from nlp_stock_prediction.storage import ArtifactRecord, ResearchRunRecord, SQLiteStore

RUN_ID = "run-reliability-freshness"
REVIEWED_AT = datetime(2026, 5, 14, 21, 0, tzinfo=UTC)


def _store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "prediction-research.sqlite3")
    store.initialize()
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=RUN_ID,
            run_kind="evaluation_hardening",
            objective="Review persisted artifact freshness.",
            status="running",
            started_at=datetime(2026, 5, 14, 20, 0, tzinfo=UTC),
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    return store


def _artifact(
    *,
    artifact_id: str = "artifact-market-msft-date-only",
    artifact_type: str = "market_data",
    created_at: datetime = datetime(2026, 5, 14, 20, 0, tzinfo=UTC),
    metadata: JsonObject | None = None,
    sha256: str = "a" * 64,
) -> ArtifactRecord:
    artifact_metadata: JsonObject = {
        "provider": "NASDAQ Basic",
        "instrument_id": "equity:NASDAQ:MSFT",
        **({} if metadata is None else metadata),
    }
    return ArtifactRecord(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        path=Path(f"artifacts/evaluation/{artifact_id}.json"),
        sha256=sha256,
        schema_version=f"{artifact_type}.v1",
        produced_by="reliability_freshness_review",
        record_count=1,
        created_at=created_at,
        metadata=artifact_metadata,
    )


@pytest.mark.unit
def test_date_only_latest_usable_bar_is_normalized_deterministically() -> None:
    review = review_artifact_freshness(
        artifact=_artifact(metadata={"latest_usable_bar": "2026-05-14"}),
        reviewed_at=REVIEWED_AT,
        policy=FreshnessPolicy(artifact_max_age=timedelta(days=2)),
        source_evidence_ids=("evidence-msft-close",),
    )

    assert review.freshness_status == "fresh"
    assert review.as_of == datetime(2026, 5, 14, tzinfo=UTC)
    assert review.provider == "NASDAQ Basic"
    assert review.source_evidence_ids == ("evidence-msft-close",)
    assert review.metadata["as_of_source"] == "latest_usable_bar"
    assert review.metadata["date_only_as_of_normalized"] is True
    assert any("Date-only" in item for item in review.limitations)


@pytest.mark.unit
def test_artifact_hash_mismatch_blocks_freshness_use() -> None:
    review = review_artifact_freshness(
        artifact=_artifact(sha256="b" * 64, metadata={"as_of": REVIEWED_AT.isoformat()}),
        reviewed_at=REVIEWED_AT,
        expected_sha256="c" * 64,
    )

    assert review.freshness_status == "hash_mismatch"
    assert review.sha256 == "b" * 64
    assert review.expected_sha256 == "c" * 64
    assert any("hash" in item.lower() for item in review.limitations)


@pytest.mark.unit
def test_malformed_and_missing_artifacts_remain_auditable() -> None:
    malformed = review_artifact_freshness(
        artifact=_artifact(metadata={"as_of": "not-an-iso-timestamp"}),
        reviewed_at=REVIEWED_AT,
    )
    missing = review_artifact_freshness(
        artifact=None,
        artifact_id="artifact-missing-market-data",
        reviewed_at=REVIEWED_AT,
    )

    assert malformed.freshness_status == "malformed"
    assert malformed.as_of is None
    assert any("as_of" in item for item in malformed.limitations)
    assert missing.freshness_status == "missing"
    assert missing.artifact_type == "unknown"
    assert missing.limitations == ("Artifact was not present in the research database.",)


@pytest.mark.unit
def test_artifact_freshness_review_writes_indexed_audit_artifact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    artifact = _artifact(metadata={"observed_at": REVIEWED_AT.isoformat()})
    store.record_artifact(artifact)
    review = review_artifact_freshness(artifact=artifact, reviewed_at=REVIEWED_AT)

    written = write_artifact_freshness_review_artifact(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        reviews=(review,),
        created_at=REVIEWED_AT,
    )

    assert written.artifact.artifact_type == "artifact_freshness_review"
    assert store.get_artifact(written.artifact.artifact_id) is not None
    payload = json.loads(Path(written.artifact.path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "artifact-freshness-review-artifact.v1"
    assert payload["artifact_freshness_reviews"][0]["artifact_id"] == artifact.artifact_id


@pytest.mark.unit
def test_artifact_file_freshness_reads_current_file_state(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "evaluation" / "artifact-live.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text('{"schema_version":"market-data.v1"}\n', encoding="utf-8")
    artifact = _artifact(
        artifact_id="artifact-live-file-state",
        metadata={"observed_at": REVIEWED_AT.isoformat()},
        sha256="0" * 64,
    )
    artifact = ArtifactRecord(
        **{
            **artifact.__dict__,
            "path": artifact_path.relative_to(tmp_path),
        }
    )

    hash_mismatch = review_artifact_file_freshness(
        artifact=artifact,
        reviewed_at=REVIEWED_AT,
        repo_root=tmp_path,
    )
    artifact_path.write_text("{not json", encoding="utf-8")
    malformed = review_artifact_file_freshness(
        artifact=ArtifactRecord(
            **{
                **artifact.__dict__,
                "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            }
        ),
        reviewed_at=REVIEWED_AT,
        repo_root=tmp_path,
    )
    artifact_path.unlink()
    missing = review_artifact_file_freshness(
        artifact=artifact,
        reviewed_at=REVIEWED_AT,
        repo_root=tmp_path,
    )

    assert hash_mismatch.freshness_status == "hash_mismatch"
    assert hash_mismatch.expected_sha256 == "0" * 64
    assert malformed.freshness_status == "malformed"
    assert missing.freshness_status == "missing"
