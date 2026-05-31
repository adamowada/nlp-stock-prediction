"""Batch research orchestration and viability ranking artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from nlp_stock_prediction.contracts import (
    BatchRunConfig,
    DailyReport,
    PredictionCandidate,
    PredictionStatus,
    ResearchViabilityRankingReport,
    ResearchViabilityTarget,
    RunConfig,
)
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.orchestration.context import deterministic_generated_at
from nlp_stock_prediction.orchestration.report_bundle import BatchReportBundle, ReportBundle

ReportGenerator = Callable[[RunConfig], ReportBundle]


def generate_batch_research_reports(
    config: BatchRunConfig,
    *,
    report_generator: ReportGenerator,
) -> BatchReportBundle:
    """Generate per-symbol reports concurrently and write an aggregate viability ranking."""

    live_requested = config.source_mode == "live" or config.live_providers
    if not config.offline and not live_requested:
        raise ValueError(
            "Batch research requires --offline or --live; refusing to use disabled providers."
        )
    if config.offline and live_requested:
        raise ValueError("BatchRunConfig cannot request both offline fixtures and live providers.")

    symbols = tuple(config.symbols)
    max_workers = min(config.max_workers, len(symbols))
    completed: dict[str, ReportBundle] = {}
    failures: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_symbol = {
            executor.submit(report_generator, _single_run_config(config, symbol)): symbol
            for symbol in symbols
        }
        for future in as_completed(future_by_symbol):
            symbol = future_by_symbol[future]
            try:
                completed[symbol] = future.result()
            except Exception as exc:
                failures[symbol] = str(exc)

    if failures and not completed:
        raise ValueError(f"all batch research runs failed: {_failure_summary(failures, symbols)}")

    ranking_report = _ranking_report(
        config=config,
        completed=completed,
        failures=failures,
        generated_at=(
            deterministic_generated_at(config.run_date) if config.offline else datetime.now(UTC)
        ),
    )
    ranking_json_path, ranking_markdown_path = _write_ranking_artifacts(
        output_dir=config.output_dir,
        report=ranking_report,
    )
    ordered_bundles = tuple(completed[symbol] for symbol in symbols if symbol in completed)
    return BatchReportBundle(
        output_dir=config.output_dir.resolve(),
        ranking_markdown_path=ranking_markdown_path,
        ranking_json_path=ranking_json_path,
        ranking_report=ranking_report,
        report_bundles=ordered_bundles,
    )


def rank_report_viability(*, symbol: str, bundle: ReportBundle) -> ResearchViabilityTarget:
    """Rank one completed report as follow-up research viability, not trading appeal."""

    report = bundle.report
    warnings = _report_warnings(bundle)
    if not report.prediction_candidates:
        return ResearchViabilityTarget(
            rank=1,
            symbol=symbol,
            viability_score=0.0,
            report_status="insufficient_evidence",
            evidence_for_count=0,
            evidence_against_count=0,
            signal_artifact_count=0,
            markdown_path=bundle.markdown_path.as_posix(),
            json_path=bundle.json_path.as_posix(),
            database_path=bundle.database_path.as_posix() if bundle.database_path else None,
            rationale=(
                "No prediction candidates were emitted; this target needs more attributable "
                "evidence before deeper ranking.",
            ),
            warnings=warnings,
            metadata=_target_metadata(report),
        )

    scored = tuple(
        (_candidate_viability_score(candidate), candidate)
        for candidate in report.prediction_candidates
    )
    score, candidate = max(
        scored,
        key=lambda item: (item[0], item[1].confidence, item[1].candidate_id),
    )
    evidence_against_count = len(candidate.evidence_against) + sum(
        1 for dissent in candidate.dissenting_evidence if dissent.impact == "contradicts"
    )
    evaluation_score = _candidate_evaluation_score(candidate)
    return ResearchViabilityTarget(
        rank=1,
        symbol=symbol,
        viability_score=score,
        report_status=candidate.status.value,
        candidate_id=candidate.candidate_id,
        candidate_status=candidate.status,
        candidate_confidence=candidate.confidence,
        evaluation_score=evaluation_score,
        evidence_for_count=len(candidate.evidence_for),
        evidence_against_count=evidence_against_count,
        signal_artifact_count=_signal_artifact_count(candidate),
        markdown_path=bundle.markdown_path.as_posix(),
        json_path=bundle.json_path.as_posix(),
        database_path=bundle.database_path.as_posix() if bundle.database_path else None,
        rationale=_candidate_rationale(candidate, score, evaluation_score),
        warnings=warnings,
        metadata=_target_metadata(report),
    )


def _single_run_config(config: BatchRunConfig, symbol: str) -> RunConfig:
    return RunConfig(
        run_date=config.run_date,
        output_dir=config.output_dir,
        symbol=symbol,
        fixture_dir=config.fixture_dir,
        cache_dir=config.cache_dir,
        offline=config.offline,
        source_mode=config.source_mode,
        live_providers=config.live_providers,
    )


def _failure_summary(failures: dict[str, str], symbols: tuple[str, ...]) -> str:
    details = [f"{symbol}: {failures[symbol]}" for symbol in symbols if symbol in failures]
    if len(details) <= 3:
        return "; ".join(details)
    return "; ".join((*details[:3], f"{len(details) - 3} more failed"))


def _ranking_report(
    *,
    config: BatchRunConfig,
    completed: dict[str, ReportBundle],
    failures: dict[str, str],
    generated_at: datetime,
) -> ResearchViabilityRankingReport:
    initial_targets = tuple(
        rank_report_viability(symbol=symbol, bundle=bundle) for symbol, bundle in completed.items()
    )
    input_order = {symbol: index for index, symbol in enumerate(config.symbols)}
    ordered_targets = sorted(
        initial_targets,
        key=lambda target: (-target.viability_score, input_order.get(target.symbol, 0)),
    )
    ranked_targets = tuple(
        target.model_copy(update={"rank": rank})
        for rank, target in enumerate(ordered_targets, start=1)
    )
    failed_targets = tuple(
        ResearchViabilityTarget(
            symbol=symbol,
            research_status="failed",
            viability_score=0.0,
            rationale=("The per-symbol research run failed before a report could be ranked.",),
            error_message=failures[symbol],
            metadata={"input_order": input_order[symbol]},
        )
        for symbol in config.symbols
        if symbol in failures
    )
    return ResearchViabilityRankingReport(
        generated_at=generated_at,
        run_date=config.run_date,
        mode="offline_fixture" if config.offline else "live",
        ranked_targets=ranked_targets,
        failed_targets=failed_targets,
        metadata={
            "symbol_count": len(config.symbols),
            "completed_count": len(ranked_targets),
            "failed_count": len(failed_targets),
            "max_workers": config.max_workers,
            "ranking_policy": "research_viability_not_trade_advice",
        },
    )


def _candidate_viability_score(candidate: PredictionCandidate) -> float:
    evaluation_or_confidence = _candidate_evaluation_score(candidate)
    base_score = (
        candidate.confidence if evaluation_or_confidence is None else evaluation_or_confidence
    )
    status_component = {
        PredictionStatus.EVIDENCE_SUPPORTED: 0.32,
        PredictionStatus.CONTRADICTED: 0.12,
        PredictionStatus.INSUFFICIENT_EVIDENCE: 0.04,
        PredictionStatus.UNAVAILABLE: 0.0,
    }[candidate.status]
    support_component = 0.14 * min(len(candidate.evidence_for), 4) / 4
    artifact_component = 0.12 * min(_signal_artifact_count(candidate), 6) / 6
    contradiction_count = len(candidate.evidence_against) + sum(
        1 for dissent in candidate.dissenting_evidence if dissent.impact == "contradicts"
    )
    contradiction_penalty = 0.25 * min(contradiction_count, 4) / 4
    score = status_component + (0.42 * base_score) + support_component + artifact_component
    return round(max(0.0, min(1.0, score - contradiction_penalty)), 6)


def _candidate_evaluation_score(candidate: PredictionCandidate) -> float | None:
    metadata = candidate.metadata.get("prediction_evaluation")
    if not isinstance(metadata, dict):
        return None
    raw_score = metadata.get("score")
    if isinstance(raw_score, bool):
        return None
    if isinstance(raw_score, int | float):
        return max(0.0, min(1.0, float(raw_score)))
    return None


def _signal_artifact_count(candidate: PredictionCandidate) -> int:
    typed_ids = tuple(reference.artifact_id for reference in candidate.signal_artifacts)
    return len(tuple(dict.fromkeys((*candidate.signal_artifact_ids, *typed_ids))))


def _candidate_rationale(
    candidate: PredictionCandidate,
    score: float,
    evaluation_score: float | None,
) -> tuple[str, ...]:
    rationale = [
        f"Viability score {score:.3f} from candidate status {candidate.status.value}.",
        (
            f"Prediction evaluation score {evaluation_score:.3f} was available."
            if evaluation_score is not None
            else f"Candidate confidence {candidate.confidence:.3f} was used as the score input."
        ),
        (
            f"{len(candidate.evidence_for)} supporting evidence references and "
            f"{len(candidate.evidence_against)} opposing evidence references were cited."
        ),
    ]
    signal_count = _signal_artifact_count(candidate)
    if signal_count:
        rationale.append(f"{signal_count} signal artifacts were linked to the candidate.")
    if candidate.uncertainties:
        rationale.append("Uncertainty context is present and remains part of the ranking.")
    return tuple(rationale)


def _target_metadata(report: DailyReport) -> JsonObject:
    return {
        "run_id": report.run_id,
        "objective": report.objective,
        "instrument_count": len(report.instruments),
        "candidate_count": len(report.prediction_candidates),
        "evidence_source_count": len(report.evidence_sources),
        "provider_count": len(report.provider_health),
    }


def _report_warnings(bundle: ReportBundle) -> tuple[str, ...]:
    warnings: list[str] = []
    if bundle.report.insufficient_evidence_summary:
        warnings.append(bundle.report.insufficient_evidence_summary)
    for record in bundle.tool_records:
        warnings.extend(str(warning) for warning in getattr(record, "warnings", ()))
    for health in bundle.report.provider_health:
        if health.status.value != "ok":
            warnings.append(f"{health.provider_name} provider status was {health.status.value}.")
    return tuple(dict.fromkeys(warnings))


def _write_ranking_artifacts(
    *,
    output_dir: Path,
    report: ResearchViabilityRankingReport,
) -> tuple[Path, Path]:
    batch_dir = output_dir.resolve() / report.run_date.isoformat() / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    json_path = batch_dir / "viability-ranking.json"
    markdown_path = batch_dir / "viability-ranking.md"
    payload = report.model_dump(mode="json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_ranking_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _render_ranking_markdown(report: ResearchViabilityRankingReport) -> str:
    lines = [
        "# Batch Research Viability Ranking",
        "",
        (
            "This ranks evidence-backed research viability for follow-up analysis. "
            "It is not a trading instruction, position-sizing output, or recommendation."
        ),
        "",
        f"- Report date: {report.run_date.isoformat()}",
        f"- Mode: {report.mode}",
        f"- Generated at: {report.generated_at.isoformat()}",
        "",
        "| Rank | Symbol | Score | Status | Candidate | Evidence For | Evidence Against | Report |",
        "| ---: | --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for target in report.ranked_targets:
        report_path = target.markdown_path or ""
        candidate_id = target.candidate_id or ""
        lines.append(
            "| "
            f"{target.rank} | "
            f"{target.symbol} | "
            f"{target.viability_score:.3f} | "
            f"{target.report_status or ''} | "
            f"{candidate_id} | "
            f"{target.evidence_for_count} | "
            f"{target.evidence_against_count} | "
            f"{report_path} |"
        )
    if report.failed_targets:
        lines.extend(["", "## Failed Targets", ""])
        for target in report.failed_targets:
            lines.append(f"- {target.symbol}: {target.error_message}")
    lines.append("")
    lines.append("## Ranking Rationale")
    lines.append("")
    for target in report.ranked_targets:
        lines.append(f"### {target.rank}. {target.symbol}")
        lines.extend(f"- {item}" for item in target.rationale)
        if target.warnings:
            lines.extend(f"- Warning: {warning}" for warning in target.warnings)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ReportGenerator",
    "generate_batch_research_reports",
    "rank_report_viability",
]
