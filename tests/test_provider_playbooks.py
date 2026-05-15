from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from nlp_stock_prediction.contracts import AuditArtifact
from nlp_stock_prediction.reliability import (
    ProviderReplacementSpec,
    build_provider_compatibility_note,
    build_provider_replacement_playbook,
    default_provider_replacement_playbooks,
    write_provider_replacement_playbook_artifacts,
)
from nlp_stock_prediction.storage import ResearchRunRecord, ToolRunRecord, initialize_database

NOW = datetime(2026, 5, 14, 18, 30, tzinfo=UTC)


def test_market_provider_replacement_playbook_preserves_limited_compatibility() -> None:
    spec = ProviderReplacementSpec(
        provider_family="market_data",
        source_provider="alpha-vantage-market-data",
        replacement_provider="candlecharts-market-data",
        preserved_fields=(
            "provider",
            "source_url",
            "raw_identifier",
            "raw_snapshot_id",
            "retrieved_at",
            "freshness_status",
            "artifact_type",
            "schema_version",
        ),
        expected_artifact_type="market_data",
        replacement_artifact_type="market_data",
        artifact_schema_versions=("research-market-data.v1",),
        credential_requirements=("NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT",),
        unsupported_modes=("widget_only_without_ohlcv",),
    )

    playbook = build_provider_replacement_playbook(spec, created_at=NOW)

    assert playbook.provider_family == "market_data"
    assert playbook.compatibility_notes[0].compatibility_status == ("compatible_with_limitations")
    assert playbook.compatibility_notes[0].missing_fields == ("observed_at",)
    assert "widget_only_without_ohlcv" in playbook.unsupported_modes
    assert any("observed_at" in limitation for limitation in playbook.limitations)


def test_missing_source_trace_fields_are_not_evaluable() -> None:
    note = build_provider_compatibility_note(
        ProviderReplacementSpec(
            provider_family="news",
            source_provider="ap-news",
            replacement_provider="minimal-news-feed",
            preserved_fields=("provider", "retrieved_at", "published_at"),
            expected_artifact_type="normalized_evidence",
            replacement_artifact_type="normalized_evidence",
        ),
        checked_at=NOW,
    )

    assert note.compatibility_status == "not_evaluable"
    assert "source_url" in note.missing_fields
    assert "raw_identifier" in note.missing_fields
    assert any("provenance trace" in limitation for limitation in note.limitations)


def test_default_provider_replacement_playbooks_cover_provider_families() -> None:
    playbooks = default_provider_replacement_playbooks(created_at=NOW)
    families = {playbook.provider_family for playbook in playbooks}

    assert {
        "market_data",
        "news",
        "social",
        "fundamentals",
        "macro",
        "scraping",
    }.issubset(families)
    assert all(playbook.compatibility_notes for playbook in playbooks)
    assert all(playbook.report_data_mode == "live" for playbook in playbooks)


def test_provider_replacement_playbooks_write_audit_artifacts(tmp_path: Path) -> None:
    store = initialize_database(tmp_path / "data" / "prediction-research.sqlite3")
    store.upsert_research_run(
        ResearchRunRecord(
            run_id="run-live-msft",
            run_kind="reliability_stage6_test",
            objective="Write provider replacement playbook artifacts for a live report.",
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
            tool_version="reliability.stage6.test",
            status="ok",
            started_at=NOW,
            completed_at=NOW,
            inputs={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    playbook = default_provider_replacement_playbooks(created_at=NOW)[0]

    artifacts = write_provider_replacement_playbook_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / "audit",
        run_id="run-live-msft",
        tool_run_id="tool-live-report",
        playbooks=(playbook,),
        created_at=NOW,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.artifact_type == "provider_replacement_playbook"
    assert artifact.metadata["provider_family"] == playbook.provider_family
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    assert payload["playbook_id"] == playbook.playbook_id
    assert payload["compatibility_notes"][0]["compatibility_status"] in {
        "compatible",
        "compatible_with_limitations",
    }
    ledger_record = store.get_artifact(artifact.artifact_id)
    assert ledger_record is not None
    assert ledger_record.artifact_type == "provider_replacement_playbook"


def test_provider_replacement_playbook_artifact_ids_are_run_scoped(tmp_path: Path) -> None:
    store = initialize_database(tmp_path / "data" / "prediction-research.sqlite3")
    playbook = default_provider_replacement_playbooks(created_at=NOW)[0]
    artifacts: list[AuditArtifact] = []
    for run_id in ("run-live-msft-a", "run-live-msft-b"):
        store.upsert_research_run(
            ResearchRunRecord(
                run_id=run_id,
                run_kind="reliability_stage6_test",
                objective="Write provider replacement playbook artifacts for a live report.",
                status="completed",
                started_at=NOW,
                completed_at=NOW,
                metadata={"report_data_mode": "live", "provider_mode": "live"},
            )
        )
        artifacts.extend(
            write_provider_replacement_playbook_artifacts(
                store=store,
                repo_root=tmp_path,
                artifact_dir=tmp_path / "reports" / run_id / "audit",
                run_id=run_id,
                tool_run_id=None,
                playbooks=(playbook,),
                created_at=NOW,
            )
        )

    assert len({artifact.artifact_id for artifact in artifacts}) == 2
