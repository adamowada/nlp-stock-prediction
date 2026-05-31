from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.cli import main
from nlp_stock_prediction.contracts import (
    BatchRunConfig,
    PredictionStatus,
    ResearchViabilityRankingReport,
    ResearchViabilityTarget,
    RunConfig,
)
from nlp_stock_prediction.orchestration.orchestration_common import symbol_slug
from nlp_stock_prediction.orchestration.report_bundle import BatchReportBundle, ReportBundle
from nlp_stock_prediction.orchestration.research_batch import generate_batch_research_reports
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle

pytestmark = pytest.mark.unit


RUN_DATE = date(2026, 5, 12)
GENERATED_AT = datetime(2026, 5, 12, 21, 0, tzinfo=UTC)


def test_batch_run_config_normalizes_and_rejects_duplicate_symbols(tmp_path: Path) -> None:
    config = BatchRunConfig(
        run_date=RUN_DATE,
        output_dir=tmp_path / "reports",
        symbols=("tsla", "btc:usd"),
        offline=True,
        source_mode="offline",
    )

    assert config.symbols == ("TSLA", "BTC:USD")

    with pytest.raises(ValidationError, match="unique"):
        BatchRunConfig(
            run_date=RUN_DATE,
            output_dir=tmp_path / "reports",
            symbols=("TSLA", "tsla"),
            offline=True,
            source_mode="offline",
        )


def test_batch_research_generates_ranked_viability_artifacts(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_generator(config: RunConfig) -> ReportBundle:
        calls.append(config.symbol)
        return _fake_bundle(config, weak=config.symbol == "WEAK")

    bundle = generate_batch_research_reports(
        BatchRunConfig(
            run_date=RUN_DATE,
            output_dir=tmp_path / "reports",
            symbols=("WEAK", "TSLA"),
            offline=True,
            source_mode="offline",
            max_workers=2,
        ),
        report_generator=fake_generator,
    )

    payload = json.loads(bundle.ranking_json_path.read_text(encoding="utf-8"))
    markdown = bundle.ranking_markdown_path.read_text(encoding="utf-8")

    assert sorted(calls) == ["TSLA", "WEAK"]
    assert bundle.ranking_report.ranked_targets[0].symbol == "TSLA"
    assert bundle.ranking_report.ranked_targets[0].rank == 1
    assert bundle.ranking_report.ranked_targets[1].symbol == "WEAK"
    assert bundle.ranking_report.ranked_targets[0].viability_score > (
        bundle.ranking_report.ranked_targets[1].viability_score
    )
    assert payload["schema_version"] == "research-batch-ranking.v1"
    assert payload["generated_at"] == "2026-05-12T21:00:00Z"
    assert payload["metadata"]["max_workers"] == 2
    assert "not a trading instruction" in markdown
    assert "TSLA" in markdown
    assert "WEAK" in markdown


def test_batch_research_rejects_all_failed_runs(tmp_path: Path) -> None:
    def failing_generator(_config: RunConfig) -> ReportBundle:
        raise ValueError("fixture run already exists")

    with pytest.raises(ValueError, match="all batch research runs failed"):
        generate_batch_research_reports(
            BatchRunConfig(
                run_date=RUN_DATE,
                output_dir=tmp_path / "reports",
                symbols=("TSLA", "MSFT"),
                offline=True,
                source_mode="offline",
                max_workers=2,
            ),
            report_generator=failing_generator,
        )

    assert not (tmp_path / "reports" / "2026-05-12" / "batch").exists()


def test_cli_research_batch_parses_symbols_and_prints_ranking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "reports"

    def fake_batch(config: BatchRunConfig) -> BatchReportBundle:
        assert config.symbols == ("TSLA", "MSFT", "NVDA")
        ranking_dir = config.output_dir.resolve() / config.run_date.isoformat() / "batch"
        ranking_dir.mkdir(parents=True)
        markdown_path = ranking_dir / "viability-ranking.md"
        json_path = ranking_dir / "viability-ranking.json"
        markdown_path.write_text("# ranking\n", encoding="utf-8")
        json_path.write_text("{}\n", encoding="utf-8")
        report = ResearchViabilityRankingReport(
            generated_at=GENERATED_AT,
            run_date=config.run_date,
            mode="offline_fixture",
            ranked_targets=(
                ResearchViabilityTarget(
                    rank=1,
                    symbol="MSFT",
                    viability_score=0.72,
                    report_status="evidence_supported",
                    markdown_path=(config.output_dir / "msft" / "report.md").as_posix(),
                    json_path=(config.output_dir / "msft" / "report.json").as_posix(),
                    rationale=("Evidence-supported target for test output.",),
                ),
            ),
        )
        return BatchReportBundle(
            output_dir=config.output_dir.resolve(),
            ranking_markdown_path=markdown_path,
            ranking_json_path=json_path,
            ranking_report=report,
            report_bundles=(),
        )

    monkeypatch.setattr(
        "nlp_stock_prediction.cli.generate_ranked_research_reports",
        fake_batch,
    )

    exit_code = main(
        [
            "research-batch",
            "--date",
            "2026-05-12",
            "--symbols",
            "TSLA,MSFT",
            "NVDA",
            "--output",
            str(output_dir),
            "--offline",
            "--max-workers",
            "3",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert f"Wrote batch ranking Markdown: {output_dir.resolve()}" in captured.out
    assert "Ranked research targets: 1" in captured.out
    assert "1. MSFT viability 0.720 (evidence_supported)" in captured.out


def _fake_bundle(config: RunConfig, *, weak: bool) -> ReportBundle:
    fixture = build_offline_fixture_bundle(
        RunConfig(
            run_date=config.run_date,
            output_dir=config.output_dir,
            symbol="TSLA",
            offline=True,
            source_mode="offline",
        )
    )
    report = fixture.report
    candidate = report.prediction_candidates[0]
    if weak:
        candidate = candidate.model_copy(
            update={
                "status": PredictionStatus.INSUFFICIENT_EVIDENCE,
                "confidence": 0.12,
                "evidence_for": (),
                "evidence_against": (),
                "dissenting_evidence": (),
                "signal_artifact_ids": (),
                "signal_artifacts": (),
                "uncertainties": ("Only thin source coverage is available.",),
                "uncertainty_drivers": (),
                "change_triggers": (),
                "change_trigger_limitations": (
                    "More attributable evidence is required before ranking higher.",
                ),
                "metadata": {},
            }
        )
    else:
        metadata = dict(candidate.metadata)
        metadata["prediction_evaluation"] = {"score": 0.86}
        candidate = candidate.model_copy(update={"metadata": metadata})
    report = report.model_copy(update={"prediction_candidates": (candidate,)})
    report_dir = (
        config.output_dir.resolve() / config.run_date.isoformat() / symbol_slug(config.symbol)
    )
    audit_dir = report_dir / "audit"
    audit_dir.mkdir(parents=True)
    markdown_path = report_dir / "report.md"
    json_path = report_dir / "report.json"
    audit_manifest_path = audit_dir / "audit-manifest.json"
    markdown_path.write_text("# report\n", encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    audit_manifest_path.write_text("{}\n", encoding="utf-8")
    return ReportBundle(
        report_dir=report_dir,
        markdown_path=markdown_path,
        json_path=json_path,
        audit_dir=audit_dir,
        audit_manifest_path=audit_manifest_path,
        report=report,
        tool_records=(),
        database_path=(
            config.output_dir.resolve() / "data" / f"{symbol_slug(config.symbol)}.sqlite3"
        ),
    )
