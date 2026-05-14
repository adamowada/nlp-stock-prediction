from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from nlp_stock_prediction.contracts import (
    AuditArtifact,
    FreshnessStatus,
    JsonObject,
    ProviderHealth,
    ProviderStatus,
    RetrievalMethod,
    SourceKind,
    SourceProvenance,
)
from nlp_stock_prediction.orchestration.report_assembly import (
    ReportAssemblyState,
    report_source_references,
)
from nlp_stock_prediction.reliability import (
    build_source_reliability_note,
    write_source_reliability_note_artifacts,
)
from nlp_stock_prediction.storage import (
    EvidenceRecord,
    ResearchRunRecord,
    ToolRunRecord,
    initialize_database,
)

NOW = datetime(2026, 5, 14, 18, 0, tzinfo=UTC)


def test_official_api_evidence_builds_high_reliability_note() -> None:
    evidence = _evidence_record(
        provider="alpha-vantage-market-data",
        source_kind=SourceKind.MARKET_DATA,
        retrieval_method=RetrievalMethod.OFFICIAL_API,
        extraction_confidence=0.98,
        source_reliability="provider_metric",
    )

    note = build_source_reliability_note(evidence)

    assert note.reliability == "high"
    assert note.provider == "alpha-vantage-market-data"
    assert note.retrieval_method == RetrievalMethod.OFFICIAL_API
    assert note.freshness_status == FreshnessStatus.FRESH
    assert note.extraction_confidence == 0.98
    assert note.evidence_ids == (evidence.evidence_id,)
    assert note.related_artifact_ids == ("artifact-live-alpha-vantage",)
    assert note.limitations == ()


def test_public_scrape_missing_observed_timestamp_builds_limited_note() -> None:
    evidence = _evidence_record(
        provider="reddit-public-page",
        source_kind=SourceKind.REDDIT_POST,
        retrieval_method=RetrievalMethod.PUBLIC_SCRAPE,
        observed_at=None,
        extraction_confidence=0.72,
        source_reliability="public_scrape",
    )

    note = build_source_reliability_note(evidence)

    assert note.reliability == "low"
    assert note.retrieval_method == RetrievalMethod.PUBLIC_SCRAPE
    assert any("observed timestamp" in limitation for limitation in note.limitations)
    assert any("public scraping" in limitation for limitation in note.limitations)


def test_missing_external_traceability_rejects_reliability_note() -> None:
    evidence = _evidence_record(
        provider="ap-news",
        source_kind=SourceKind.NEWS_ARTICLE,
        retrieval_method=RetrievalMethod.OFFICIAL_API,
    )
    evidence = EvidenceRecord(
        **{
            **evidence.__dict__,
            "url": None,
            "provenance_json": {
                "provider_name": "ap-news",
                "source_kind": SourceKind.NEWS_ARTICLE.value,
                "retrieval_method": RetrievalMethod.OFFICIAL_API.value,
                "fetched_at": NOW.isoformat(),
                "freshness_status": FreshnessStatus.FRESH.value,
            },
        }
    )

    with pytest.raises(ValueError, match="traceability"):
        build_source_reliability_note(evidence)


def test_non_live_source_evidence_rejects_reliability_note() -> None:
    evidence = _evidence_record(
        provider="fixture-market-data",
        source_kind=SourceKind.MARKET_DATA,
        retrieval_method=RetrievalMethod.FIXTURE,
    )

    with pytest.raises(ValueError, match="fixture retrieval"):
        build_source_reliability_note(evidence)


def test_source_reliability_notes_write_audit_artifacts(tmp_path: Path) -> None:
    store = initialize_database(tmp_path / "data" / "prediction-research.sqlite3")
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-live-msft",
            run_kind="phase7_stage6_test",
            objective="Write source reliability note artifacts for a live report.",
            status="completed",
            started_at=NOW,
            completed_at=NOW,
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id="tool-live-report",
            run_id="run-live-msft",
            tool_name="render_prediction_report",
            tool_version="phase7.stage6.test",
            status="ok",
            started_at=NOW,
            completed_at=NOW,
            inputs={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    note = build_source_reliability_note(
        _evidence_record(
            evidence_id="evidence-live-msft-market",
            provider="alpha-vantage-market-data",
            source_kind=SourceKind.MARKET_DATA,
            retrieval_method=RetrievalMethod.OFFICIAL_API,
            extraction_confidence=0.96,
        )
    )

    artifacts = write_source_reliability_note_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / "audit",
        run_id="run-live-msft",
        tool_run_id="tool-live-report",
        notes=(note,),
        created_at=NOW,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.artifact_type == "source_reliability_note"
    assert artifact.metadata["evidence_id"] == "evidence-live-msft-market"
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    assert payload["note_id"] == note.note_id
    assert payload["reliability"] == "high"
    ledger_record = store.get_artifact(artifact.artifact_id)
    assert ledger_record is not None
    assert ledger_record.artifact_type == "source_reliability_note"


def test_source_reliability_artifacts_propagate_to_report_source_references() -> None:
    artifact = AuditArtifact(
        artifact_id="artifact-source-reliability-msft",
        artifact_type="source_reliability_note",
        path="reports/audit/source-reliability/source-reliability-msft.json",
        created_at=NOW,
        produced_by="phase7_source_reliability",
        sha256="a" * 64,
        record_count=1,
        metadata={
            "note_id": "source-reliability-msft",
            "provider": "alpha-vantage-market-data",
            "evidence_id": "evidence-live-msft-market",
        },
    )
    references = report_source_references(
        evidence_sources=(),
        audit_artifacts=(artifact,),
        provider_health=(
            ProviderHealth(
                provider_name="alpha-vantage-market-data",
                status=ProviderStatus.OK,
                checked_at=NOW,
            ),
        ),
        assembly_state=ReportAssemblyState(
            audit_artifacts=(),
            usable_candidate_records=(),
            excluded_candidate_reasons={},
            warnings=(),
            blocking_reasons=(),
            provider_health=(),
            candidate_artifact_ids={},
            candidate_evidence_ids={},
            artifact_candidate_ids={},
            evidence_candidate_ids={},
            missing_artifact_ids=(),
            missing_evidence_ids=(),
        ),
    )

    reliability_reference = next(
        reference
        for reference in references
        if reference.artifact_ids == ("artifact-source-reliability-msft",)
    )
    provider_reference = next(
        reference for reference in references if reference.reference_type == "provider_health"
    )

    assert reliability_reference.evidence_ids == ("evidence-live-msft-market",)
    assert provider_reference.metadata["source_reliability_note_ids"] == ["source-reliability-msft"]


def _evidence_record(
    *,
    evidence_id: str = "evidence-live-alpha-vantage",
    provider: str,
    source_kind: SourceKind,
    retrieval_method: RetrievalMethod,
    observed_at: datetime | None = NOW,
    extraction_confidence: float | None = 0.9,
    source_reliability: str = "provider_normalized",
) -> EvidenceRecord:
    provenance = SourceProvenance(
        provider_name=provider,
        source_kind=source_kind,
        retrieval_method=retrieval_method,
        fetched_at=NOW,
        observed_at=observed_at,
        source_url=f"https://example.com/{provider}/MSFT",
        permalink=f"https://example.com/{provider}/MSFT",
        raw_identifier="MSFT",
        raw_snapshot_id="raw-live-msft",
        query="MSFT",
        freshness_status=FreshnessStatus.FRESH,
    )
    return EvidenceRecord(
        evidence_id=evidence_id,
        tool_run_id="tool-live-provider",
        source_query_id="query-live-provider",
        source_type=source_kind.value,
        provider=provider,
        url=provenance.source_url,
        query="MSFT",
        retrieved_at=NOW,
        published_at=observed_at,
        instruments=("instrument:equity:us:msft",),
        claim="MSFT source evidence was retrieved from a live provider.",
        extraction_confidence=extraction_confidence,
        source_reliability=source_reliability,
        freshness_status=FreshnessStatus.FRESH.value,
        artifact_id="artifact-live-alpha-vantage",
        raw_excerpt="MSFT source evidence was retrieved from a live provider.",
        provenance_json=cast(JsonObject, provenance.model_dump(mode="json")),
        metadata={
            "report_data_mode": "live",
            "provider_mode": "live",
            "source_evidence": {
                "evidence_id": evidence_id,
                "source_kind": source_kind.value,
                "ticker": "MSFT",
                "text": "MSFT source evidence was retrieved from a live provider.",
                "created_at": observed_at.isoformat() if observed_at is not None else None,
                "permalink": provenance.permalink,
                "matched_tickers": ["MSFT"],
                "matched_instrument_ids": ["instrument:equity:us:msft"],
                "provenance": provenance.model_dump(mode="json"),
                "metadata": {
                    "extraction_confidence": extraction_confidence,
                    "source_reliability": source_reliability,
                },
            },
        },
    )
