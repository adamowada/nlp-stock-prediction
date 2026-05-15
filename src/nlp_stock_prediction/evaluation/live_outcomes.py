"""Live market-data materialization for Phase 7 outcome evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast

from nlp_stock_prediction.contracts import (
    Direction,
    EvidenceReference,
    FreshnessStatus,
    JsonObject,
    MarketDataProvider,
    PredictionEvaluationTarget,
    PredictionOutcome,
    PredictionOutcomeEvaluation,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
    PriceBar,
    ProviderStatus,
    SourceKind,
)
from nlp_stock_prediction.contracts.live_validation import (
    require_live_metadata,
    require_live_retrieval_method,
    text_is_non_live,
)
from nlp_stock_prediction.evaluation.common import aware_utc, digest, slug
from nlp_stock_prediction.evaluation.outcomes import (
    PointInTimeOutcomeEvaluationArtifacts,
    build_prediction_evaluation_target,
    write_point_in_time_outcome_evaluation_artifacts,
)
from nlp_stock_prediction.ml.ohlcv import calendar_date
from nlp_stock_prediction.orchestration.live_market_data import (
    SUPPORTED_LIVE_MARKET_OUTCOME_ASSET_CLASSES,
    LiveMarketDataSelector,
    yahoo_finance_chart_source_url,
)
from nlp_stock_prediction.orchestration.live_market_data import (
    LiveMarketDataSelection as LiveOutcomeMarketDataSelection,
)
from nlp_stock_prediction.orchestration.phase4_market_data import (
    PHASE4_MARKET_DATA_TOOL_NAME,
    MarketDataToolResult,
    Phase4MarketDataArtifact,
    Phase4MarketDataTool,
    load_phase4_market_data_artifact,
)
from nlp_stock_prediction.providers.market import YahooFinanceChartMarketDataProvider
from nlp_stock_prediction.storage.records import (
    ArtifactRecord,
    EvaluationAttemptRecord,
    EvidenceRecord,
    InstrumentRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE7_LIVE_OUTCOME_TOOL_NAME = "phase7_live_outcome_materialization"
PHASE7_LIVE_OUTCOME_TOOL_VERSION = "phase7.live-outcome-materialization.v1"
SUPPORTED_LIVE_OUTCOME_ASSET_CLASSES = SUPPORTED_LIVE_MARKET_OUTCOME_ASSET_CLASSES


class LiveOutcomeProviderFactory(Protocol):
    """Provider-selection surface for live outcome materialization."""

    def market_data_selections(
        self,
        *,
        symbol: str,
        instrument: InstrumentRecord,
    ) -> tuple[LiveOutcomeMarketDataSelection, ...]: ...


@dataclass(frozen=True)
class DefaultLiveOutcomeProviderFactory:
    """Build stock/ETF live outcome providers from real credentials and public sources."""

    cache_root: Path | None = None
    env: Mapping[str, str] | None = None
    now: Callable[[], datetime] | None = None

    def market_data_selections(
        self,
        *,
        symbol: str,
        instrument: InstrumentRecord,
    ) -> tuple[LiveOutcomeMarketDataSelection, ...]:
        return LiveMarketDataSelector(
            cache_root=self.cache_root,
            env=self.env,
            now=self.now,
        ).outcome_selections(
            symbol=symbol,
            asset_class=instrument.asset_class,
        )


@dataclass(frozen=True)
class LiveOutcomeMaterializationArtifacts:
    """Persisted output of one live outcome-materialization attempt."""

    written: PointInTimeOutcomeEvaluationArtifacts
    evaluation_attempt_id: str
    market_artifact_ids: tuple[str, ...]
    outcome_evidence_ids: tuple[str, ...]
    provider_attempts: tuple[JsonObject, ...]

    @property
    def target(self) -> PredictionEvaluationTarget:
        return self.written.target

    @property
    def outcome(self) -> PredictionOutcome:
        return self.written.outcome

    @property
    def outcome_evaluation(self) -> PredictionOutcomeEvaluation:
        return self.written.outcome_evaluation

    @property
    def outcome_artifact_id(self) -> str:
        return self.written.outcome_artifact.artifact_id

    @property
    def outcome_evaluation_artifact_id(self) -> str:
        return self.written.outcome_evaluation_artifact.artifact_id


@dataclass(frozen=True)
class _Observation:
    observed_result: PredictionOutcomeResult
    observed_at: datetime
    result_value: float
    baseline_value: float
    result_summary: str
    evidence_id: str
    artifact_payload: Phase4MarketDataArtifact


@dataclass(frozen=True)
class _InspectionResult:
    observation: _Observation | None
    status: PredictionOutcomeStatus
    limitations: tuple[str, ...]


def materialize_live_prediction_outcome_artifacts(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    candidate_id: str,
    point_in_time_cutoff: datetime,
    evaluation_window_start: datetime,
    evaluation_window_end: datetime,
    report_date: date | None = None,
    evaluated_at: datetime | None = None,
    created_at: datetime | None = None,
    provider_factory: LiveOutcomeProviderFactory | None = None,
    market_artifact_ids: Sequence[str] = (),
) -> LiveOutcomeMaterializationArtifacts:
    """Fetch or reuse real market data and persist a no-shortcut outcome evaluation."""

    evaluated = aware_utc(evaluated_at or datetime.now(UTC), "evaluated_at")
    created = aware_utc(created_at or evaluated, "created_at")
    target = build_prediction_evaluation_target(
        store=store,
        run_id=run_id,
        candidate_id=candidate_id,
        point_in_time_cutoff=point_in_time_cutoff,
        evaluation_window_start=evaluation_window_start,
        evaluation_window_end=evaluation_window_end,
        report_date=report_date,
        repo_root=repo_root,
    )
    candidate = store.get_prediction_candidate(candidate_id)
    if candidate is None:
        raise ValueError(f"prediction candidate does not exist: {candidate_id}")
    instrument = store.get_instrument(target.instrument_id)
    if instrument is None:
        raise ValueError(f"instrument does not exist: {target.instrument_id}")
    provided_market_artifact_ids = tuple(dict.fromkeys(market_artifact_ids))
    preloaded_market_artifacts = tuple(
        _load_live_market_artifact(
            store=store,
            repo_root=repo_root,
            target=target,
            artifact_id=artifact_id,
        )
        for artifact_id in provided_market_artifact_ids
    )

    run_identity = _attempt_identity(
        target=target,
        evaluated_at=evaluated,
        market_artifact_ids=provided_market_artifact_ids,
    )
    attempt_id = (
        "attempt-phase7-live-outcome-"
        f"{slug(candidate_id, fallback='candidate', allow_file_safe_punctuation=True)}-"
        f"{run_identity[:10]}"
    )
    tool_run_id = (
        "tool-phase7-live-outcome-"
        f"{slug(candidate_id, fallback='candidate', allow_file_safe_punctuation=True)}-"
        f"{run_identity[:10]}"
    )
    outcome_id = (
        "outcome-live-"
        f"{slug(candidate_id, fallback='candidate', allow_file_safe_punctuation=True)}-"
        f"{run_identity[:10]}"
    )
    outcome_evaluation_id = (
        "outcome-evaluation-live-"
        f"{slug(candidate_id, fallback='candidate', allow_file_safe_punctuation=True)}-"
        f"{run_identity[:10]}"
    )

    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=PHASE7_LIVE_OUTCOME_TOOL_NAME,
            tool_version=PHASE7_LIVE_OUTCOME_TOOL_VERSION,
            status="running",
            started_at=created,
            inputs={
                "candidate_id": candidate_id,
                "target_id": target.target_id,
                "symbol": target.symbol,
                "market_artifact_ids": list(market_artifact_ids),
            },
        )
    )
    _record_attempt(
        store=store,
        attempt_id=attempt_id,
        run_id=run_id,
        source_run_id=candidate.run_id,
        tool_run_id=tool_run_id,
        target=target,
        outcome_id=None,
        status="running",
        started_at=created,
        completed_at=None,
        metadata={"target_id": target.target_id},
    )

    provider_attempts: list[JsonObject] = []
    observed_market_artifact_ids: list[str] = []
    limitations: list[str] = []
    observation: _Observation | None = None
    outcome_status = PredictionOutcomeStatus.UNAVAILABLE

    if instrument.asset_class.strip().lower() not in SUPPORTED_LIVE_OUTCOME_ASSET_CLASSES:
        limitations.append(
            f"Live outcome materialization is unavailable for asset class "
            f"{instrument.asset_class!r}; class-specific live adapters are not implemented."
        )
    elif preloaded_market_artifacts:
        for artifact_id, payload in zip(
            provided_market_artifact_ids,
            preloaded_market_artifacts,
            strict=True,
        ):
            observed_market_artifact_ids.append(artifact_id)
            inspection = _inspect_market_payload(
                target=target,
                artifact_payload=payload,
                evaluated_at=evaluated,
            )
            provider_attempts.append(
                _provider_attempt_summary(
                    role="reused",
                    artifact_id=artifact_id,
                    payload=payload,
                    inspection=inspection,
                )
            )
            limitations.extend(inspection.limitations)
            if inspection.observation is not None:
                observation = inspection.observation
                outcome_status = PredictionOutcomeStatus.OBSERVED
                break
            if inspection.status == PredictionOutcomeStatus.STALE:
                outcome_status = PredictionOutcomeStatus.STALE
    else:
        selections = (
            provider_factory or DefaultLiveOutcomeProviderFactory(now=lambda: evaluated)
        ).market_data_selections(
            symbol=target.symbol,
            instrument=instrument,
        )
        if not selections:
            limitations.append("No live market-data providers are configured for this instrument.")
        for index, selection in enumerate(selections, start=1):
            try:
                tool_result = _fetch_market_artifact(
                    store=store,
                    repo_root=repo_root,
                    artifact_dir=artifact_dir,
                    run_id=run_id,
                    target=target,
                    evaluated_at=evaluated,
                    attempt_digest=run_identity,
                    index=index,
                    selection=selection,
                )
            except Exception as exc:
                message = (
                    f"{_provider_name(selection.provider)} provider call failed during live "
                    f"outcome materialization: {exc}"
                )
                limitations.append(message)
                provider_attempts.append(
                    {
                        "provider": _provider_name(selection.provider),
                        "role": selection.role,
                        "status": "failed",
                        "limitation": message,
                    }
                )
                continue
            observed_market_artifact_ids.append(tool_result.artifact.artifact_id)
            inspection = _inspect_market_payload(
                target=target,
                artifact_payload=tool_result.artifact_payload,
                evaluated_at=evaluated,
            )
            provider_attempts.append(
                _provider_attempt_summary(
                    role=selection.role,
                    artifact_id=tool_result.artifact.artifact_id,
                    payload=tool_result.artifact_payload,
                    inspection=inspection,
                )
            )
            limitations.extend(inspection.limitations)
            if inspection.observation is not None:
                observation = inspection.observation
                outcome_status = PredictionOutcomeStatus.OBSERVED
                break
            if inspection.status == PredictionOutcomeStatus.STALE:
                outcome_status = PredictionOutcomeStatus.STALE

    evidence_ids: tuple[str, ...] = ()
    observed_result: PredictionOutcomeResult | None = None
    observed_at: datetime | None = None
    result_summary: str | None = None
    result_value: float | None = None
    baseline_value: float | None = None
    if observation is not None:
        _record_outcome_evidence(
            store=store,
            tool_run_id=tool_run_id,
            target=target,
            observation=observation,
        )
        evidence_ids = (observation.evidence_id,)
        observed_result = observation.observed_result
        observed_at = observation.observed_at
        result_summary = observation.result_summary
        result_value = observation.result_value
        baseline_value = observation.baseline_value
        limitations = _fallback_limitations_only(limitations)
    elif outcome_status == PredictionOutcomeStatus.STALE:
        limitations.append("Live market data was stale for the requested evaluation window.")
    else:
        limitations.append(
            "No live market-data artifact contained both cutoff and post-window bars."
        )

    written = write_point_in_time_outcome_evaluation_artifacts(
        store=store,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        run_id=run_id,
        target=target,
        status=outcome_status,
        observed_result=observed_result,
        observed_at=observed_at,
        result_summary=result_summary,
        result_value=result_value,
        baseline_value=baseline_value,
        outcome_evidence=tuple(
            EvidenceReference(evidence_id=evidence_id) for evidence_id in evidence_ids
        ),
        market_artifact_ids=tuple(dict.fromkeys(observed_market_artifact_ids)),
        limitations=tuple(dict.fromkeys(limitations)),
        metadata=cast(
            JsonObject,
            {
                "source": PHASE7_LIVE_OUTCOME_TOOL_NAME,
                "provider_attempts": provider_attempts,
            },
        ),
        created_at=created,
        evaluated_at=evaluated,
        tool_run_id=tool_run_id,
        record_tool_run=False,
        evaluation_attempt_id=attempt_id,
        outcome_id=outcome_id,
        outcome_evaluation_id=outcome_evaluation_id,
        outcome_artifact_filename=(
            "prediction-outcomes/live/"
            f"{slug(candidate_id, fallback='candidate', allow_file_safe_punctuation=True)}-"
            f"{run_identity[:8]}.json"
        ),
        outcome_evaluation_artifact_filename=(
            "prediction-outcome-evaluations/live/"
            f"{slug(candidate_id, fallback='candidate', allow_file_safe_punctuation=True)}-"
            f"{run_identity[:8]}.json"
        ),
    )

    final_status = _final_attempt_status(outcome_status)
    _record_attempt(
        store=store,
        attempt_id=attempt_id,
        run_id=run_id,
        source_run_id=candidate.run_id,
        tool_run_id=tool_run_id,
        target=target,
        outcome_id=outcome_id,
        status=final_status,
        started_at=created,
        completed_at=evaluated,
        metadata=cast(
            JsonObject,
            {
                "target_id": target.target_id,
                "outcome_id": written.outcome.outcome_id,
                "outcome_evaluation_id": written.outcome_evaluation.outcome_evaluation_id,
                "market_artifact_ids": list(observed_market_artifact_ids),
                "outcome_evidence_ids": list(evidence_ids),
                "provider_attempts": provider_attempts,
            },
        ),
    )
    store.record_tool_run(
        ToolRunRecord(
            tool_run_id=tool_run_id,
            run_id=run_id,
            tool_name=PHASE7_LIVE_OUTCOME_TOOL_NAME,
            tool_version=PHASE7_LIVE_OUTCOME_TOOL_VERSION,
            status=final_status,
            started_at=created,
            completed_at=evaluated,
            inputs={
                "candidate_id": candidate_id,
                "target_id": target.target_id,
                "symbol": target.symbol,
                "market_artifact_ids": list(observed_market_artifact_ids),
                "outcome_evidence_ids": list(evidence_ids),
            },
            warnings=written.outcome_evaluation.limitations,
        )
    )
    return LiveOutcomeMaterializationArtifacts(
        written=written,
        evaluation_attempt_id=attempt_id,
        market_artifact_ids=tuple(dict.fromkeys(observed_market_artifact_ids)),
        outcome_evidence_ids=evidence_ids,
        provider_attempts=tuple(provider_attempts),
    )


def _fetch_market_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    artifact_dir: Path,
    run_id: str,
    target: PredictionEvaluationTarget,
    evaluated_at: datetime,
    attempt_digest: str,
    index: int,
    selection: LiveOutcomeMarketDataSelection,
) -> MarketDataToolResult:
    request_id = f"phase7-live-outcome-market-data-{attempt_digest[:12]}-{index}"
    provider_slug = slug(_provider_name(selection.provider), fallback="provider")
    source_url = _source_url_for_selection(
        selection=selection,
        target=target,
        evaluated_at=evaluated_at,
    )
    tool = Phase4MarketDataTool(
        store=store,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
        provider=selection.provider,
        now=lambda: evaluated_at,
        retrieval_method=selection.retrieval_method,
    )
    result = tool.run(
        run_id=run_id,
        run_date=evaluated_at.date(),
        symbol=target.symbol,
        instrument_id=target.instrument_id,
        request_id=request_id,
        source_url=source_url,
        options={"phase7_live_outcome_role": selection.role, **selection.options},
        tool_run_id=f"tool-phase7-market-data-{provider_slug}-{attempt_digest[:10]}-{index}",
        artifact_id=f"artifact-phase7-market-data-{provider_slug}-{attempt_digest[:10]}-{index}",
        artifact_filename=(
            "market-data/live-outcomes/"
            f"{slug(target.candidate_id, fallback='candidate')}-{provider_slug}-"
            f"{attempt_digest[:8]}-{index}.json"
        ),
        source_query_id=f"query-phase7-market-data-{provider_slug}-{attempt_digest[:10]}-{index}",
    )
    _validate_live_market_payload(
        store=store,
        artifact=store.get_artifact(result.artifact.artifact_id),
        payload=result.artifact_payload,
        target=target,
    )
    return result


def _inspect_market_payload(
    *,
    target: PredictionEvaluationTarget,
    artifact_payload: Phase4MarketDataArtifact,
    evaluated_at: datetime,
) -> _InspectionResult:
    limitations: list[str] = []
    provider = artifact_payload.provenance.provider_name
    if _normalize_symbol(artifact_payload.symbol) != _normalize_symbol(target.symbol):
        return _InspectionResult(
            observation=None,
            status=PredictionOutcomeStatus.UNAVAILABLE,
            limitations=(
                f"{provider} market artifact symbol {artifact_payload.symbol} did not match "
                f"target symbol {target.symbol}.",
            ),
        )
    if artifact_payload.status == ProviderStatus.STALE or (
        artifact_payload.freshness_status == FreshnessStatus.STALE
    ):
        return _InspectionResult(
            observation=None,
            status=PredictionOutcomeStatus.STALE,
            limitations=(f"{provider} market artifact was stale for {target.symbol}.",),
        )
    if artifact_payload.status not in {ProviderStatus.OK, ProviderStatus.PARTIAL}:
        return _InspectionResult(
            observation=None,
            status=PredictionOutcomeStatus.UNAVAILABLE,
            limitations=(
                f"{provider} market artifact status was {artifact_payload.status.value}.",
                *tuple(warning.message for warning in artifact_payload.warnings),
            ),
        )
    for warning in artifact_payload.warnings:
        limitations.append(f"{provider} warning: {warning.message}")

    evaluated_date = evaluated_at.date()
    bars = tuple(
        bar
        for bar in artifact_payload.bars
        if _normalize_symbol(bar.ticker) == _normalize_symbol(target.symbol)
        and calendar_date(bar.timestamp) <= evaluated_date
    )
    if not bars:
        return _InspectionResult(
            observation=None,
            status=PredictionOutcomeStatus.UNAVAILABLE,
            limitations=(
                f"{provider} market artifact contained no usable bars for {target.symbol}.",
            ),
        )

    cutoff_date = target.point_in_time_cutoff.date()
    result_date = target.evaluation_window_end.date()
    baseline_candidates = [bar for bar in bars if calendar_date(bar.timestamp) <= cutoff_date]
    result_candidates = [bar for bar in bars if calendar_date(bar.timestamp) >= result_date]
    if not baseline_candidates:
        return _InspectionResult(
            observation=None,
            status=PredictionOutcomeStatus.UNAVAILABLE,
            limitations=(
                f"{provider} market artifact did not contain a cutoff comparison bar on or "
                f"before {cutoff_date.isoformat()}.",
            ),
        )
    if not result_candidates:
        return _InspectionResult(
            observation=None,
            status=PredictionOutcomeStatus.UNAVAILABLE,
            limitations=(
                f"{provider} market artifact did not contain a post-window usable bar on or "
                f"after {result_date.isoformat()}.",
            ),
        )

    baseline_bar = max(baseline_candidates, key=lambda bar: calendar_date(bar.timestamp))
    result_bar = min(result_candidates, key=lambda bar: calendar_date(bar.timestamp))
    baseline_value = _bar_value(baseline_bar)
    result_value = _bar_value(result_bar)
    observed_result = _observed_result_for_direction(
        direction=target.direction,
        baseline_value=baseline_value,
        result_value=result_value,
    )
    observed_at = max(_bar_observed_at(result_bar), target.evaluation_window_end)
    evidence_id = _outcome_evidence_id(target=target, artifact_payload=artifact_payload)
    result_bar_date = calendar_date(result_bar.timestamp).isoformat()
    baseline_bar_date = calendar_date(baseline_bar.timestamp).isoformat()
    summary = (
        f"{target.symbol} daily adjusted close for {result_bar_date} was "
        f"{_format_decimal(result_value)} versus cutoff comparison close "
        f"{_format_decimal(baseline_value)} from {baseline_bar_date}."
    )
    return _InspectionResult(
        observation=_Observation(
            observed_result=observed_result,
            observed_at=observed_at,
            result_value=float(result_value),
            baseline_value=float(baseline_value),
            result_summary=summary,
            evidence_id=evidence_id,
            artifact_payload=artifact_payload,
        ),
        status=PredictionOutcomeStatus.OBSERVED,
        limitations=tuple(limitations),
    )


def _record_outcome_evidence(
    *,
    store: SQLiteStore,
    tool_run_id: str,
    target: PredictionEvaluationTarget,
    observation: _Observation,
) -> None:
    payload = observation.artifact_payload
    provenance = payload.provenance
    evidence = EvidenceRecord(
        evidence_id=observation.evidence_id,
        source_type=SourceKind.MARKET_DATA.value,
        provider=provenance.provider_name,
        retrieved_at=provenance.retrieved_at,
        published_at=observation.observed_at,
        claim=observation.result_summary,
        tool_run_id=tool_run_id,
        source_query_id=provenance.source_query_id,
        url=provenance.url,
        query=provenance.query,
        instruments=(target.instrument_id,),
        extraction_confidence=provenance.extraction_confidence,
        freshness_status=FreshnessStatus.FRESH.value,
        artifact_id=payload.artifact_id,
        provenance_json={
            "retrieval_method": provenance.retrieval_method.value,
            "raw_identifier": provenance.raw_identifier,
            "raw_snapshot_id": provenance.raw_snapshot_id,
            "cache_key": provenance.cache_key,
        },
        metadata={
            "target_id": target.target_id,
            "source": PHASE7_LIVE_OUTCOME_TOOL_NAME,
            "result_value": observation.result_value,
            "baseline_value": observation.baseline_value,
            "observed_result": observation.observed_result.value,
        },
    )
    store.record_evidence(evidence)


def _record_attempt(
    *,
    store: SQLiteStore,
    attempt_id: str,
    run_id: str,
    source_run_id: str | None,
    tool_run_id: str,
    target: PredictionEvaluationTarget,
    outcome_id: str | None,
    status: str,
    started_at: datetime,
    completed_at: datetime | None,
    metadata: JsonObject,
) -> None:
    store.record_evaluation_attempt(
        EvaluationAttemptRecord(
            evaluation_attempt_id=attempt_id,
            run_id=run_id,
            source_run_id=source_run_id,
            tool_run_id=tool_run_id,
            attempt_kind="outcome_evaluation",
            subject_id=target.candidate_id,
            candidate_id=target.candidate_id,
            outcome_id=outcome_id,
            instrument_id=target.instrument_id,
            symbol=target.symbol,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            metadata=metadata,
        )
    )


def _load_live_market_artifact(
    *,
    store: SQLiteStore,
    repo_root: Path,
    target: PredictionEvaluationTarget,
    artifact_id: str,
) -> Phase4MarketDataArtifact:
    artifact = store.get_artifact(artifact_id)
    if artifact is None:
        raise ValueError(f"market artifact does not exist: {artifact_id}")
    if artifact.artifact_type != "market_data":
        raise ValueError(f"market artifact must be market_data: {artifact_id}")
    if artifact.produced_by != PHASE4_MARKET_DATA_TOOL_NAME:
        raise ValueError(
            f"live market artifact must be produced by the Phase 4 market-data tool: {artifact_id}"
        )
    if artifact.tool_run_id is None:
        raise ValueError(f"live market artifact is missing tool_run_id: {artifact_id}")
    tool_run = store.get_tool_run(artifact.tool_run_id)
    if tool_run is None:
        raise ValueError(
            f"live market artifact references a missing tool run: {artifact.tool_run_id}"
        )
    if tool_run.run_id != target.run_id:
        raise ValueError(
            "live market artifact tool run_id must match the evaluation target run_id: "
            f"{tool_run.run_id} != {target.run_id}"
        )
    require_live_metadata("market artifact", artifact.artifact_id, artifact.metadata)
    require_live_metadata("market tool run", tool_run.tool_run_id, tool_run.inputs)
    path = artifact.path if artifact.path.is_absolute() else repo_root / artifact.path
    if not path.exists():
        raise ValueError(f"live market artifact file is missing: {path.as_posix()}")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != artifact.sha256:
        raise ValueError(f"live market artifact hash does not match storage row: {artifact_id}")
    payload = load_phase4_market_data_artifact(path)
    _validate_live_market_payload(
        store=store,
        artifact=artifact,
        payload=payload,
        target=target,
    )
    return payload


def _validate_live_market_payload(
    *,
    store: SQLiteStore,
    artifact: ArtifactRecord | None,
    payload: Phase4MarketDataArtifact,
    target: PredictionEvaluationTarget,
) -> None:
    if artifact is None:
        raise ValueError(f"live market artifact was not indexed: {payload.artifact_id}")
    if payload.run_id != target.run_id:
        raise ValueError(
            "live market artifact payload run_id must match the evaluation target run_id: "
            f"{payload.run_id} != {target.run_id}"
        )
    if payload.tool_run_id != artifact.tool_run_id:
        raise ValueError(
            "live market artifact payload tool_run_id must match the indexed artifact row."
        )
    if payload.artifact_id != artifact.artifact_id:
        raise ValueError(
            "live market artifact payload artifact_id must match the indexed artifact row."
        )
    require_live_retrieval_method(
        record_type="live market artifact",
        record_id=artifact.artifact_id,
        retrieval_method=payload.provenance.retrieval_method,
    )
    if not payload.provenance.source_query_id:
        raise ValueError(f"live market artifact is missing source_query_id: {artifact.artifact_id}")
    if not payload.provenance.url or not payload.provenance.url.startswith(("http://", "https://")):
        raise ValueError(
            f"live market artifact is missing a reproducible source URL: {artifact.artifact_id}"
        )
    if not payload.provenance.raw_identifier or not payload.provenance.raw_snapshot_id:
        raise ValueError(
            f"live market artifact is missing raw provider identifiers: {artifact.artifact_id}"
        )
    if text_is_non_live(payload.provenance.provider_name):
        raise ValueError(
            f"live market artifact provider is not allowed: {payload.provenance.provider_name}"
        )
    source_query = store.get_source_query(payload.provenance.source_query_id)
    if source_query is None:
        raise ValueError(
            "live market artifact references a missing source query: "
            f"{payload.provenance.source_query_id}"
        )
    if source_query.url != payload.provenance.url:
        raise ValueError("live market artifact source query URL does not match payload URL")
    if source_query.provider != payload.provenance.provider_name:
        raise ValueError("live market artifact source query provider does not match payload")
    require_live_metadata(
        "market source query", source_query.source_query_id, source_query.metadata
    )


def _source_url_for_selection(
    *,
    selection: LiveOutcomeMarketDataSelection,
    target: PredictionEvaluationTarget,
    evaluated_at: datetime,
) -> str | Path | None:
    provider_name = _provider_name(selection.provider)
    if provider_name == YahooFinanceChartMarketDataProvider.provider_name:
        end_date = evaluated_at.date()
        return yahoo_finance_chart_source_url(
            target.symbol,
            start_date=end_date - timedelta(days=370),
            end_date=end_date,
        )
    return selection.source_url


def _provider_attempt_summary(
    *,
    role: str,
    artifact_id: str,
    payload: Phase4MarketDataArtifact,
    inspection: _InspectionResult,
) -> JsonObject:
    return {
        "provider": payload.provenance.provider_name,
        "role": role,
        "artifact_id": artifact_id,
        "status": payload.status.value,
        "freshness_status": payload.freshness_status.value,
        "bar_count": payload.bar_count,
        "observed": inspection.observation is not None,
        "limitations": list(inspection.limitations),
    }


def _attempt_identity(
    *,
    target: PredictionEvaluationTarget,
    evaluated_at: datetime,
    market_artifact_ids: tuple[str, ...],
) -> str:
    artifact_identity = ",".join(market_artifact_ids) if market_artifact_ids else "fetch"
    return digest(
        "|".join(
            (
                target.run_id,
                target.target_id,
                target.candidate_id,
                evaluated_at.isoformat(),
                artifact_identity,
            )
        )
    )


def _observed_result_for_direction(
    *,
    direction: Direction,
    baseline_value: Decimal,
    result_value: Decimal,
) -> PredictionOutcomeResult:
    if result_value == baseline_value:
        return PredictionOutcomeResult.NEUTRAL
    if direction == Direction.BULLISH:
        return (
            PredictionOutcomeResult.SUPPORTED
            if result_value > baseline_value
            else PredictionOutcomeResult.NOT_SUPPORTED
        )
    if direction == Direction.BEARISH:
        return (
            PredictionOutcomeResult.SUPPORTED
            if result_value < baseline_value
            else PredictionOutcomeResult.NOT_SUPPORTED
        )
    if direction == Direction.NEUTRAL:
        return PredictionOutcomeResult.NEUTRAL
    return PredictionOutcomeResult.INSUFFICIENT_DATA


def _bar_value(bar: PriceBar) -> Decimal:
    return bar.adjusted_close or bar.close


def _bar_observed_at(bar: PriceBar) -> datetime:
    timestamp = bar.timestamp
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC)
    return datetime.combine(calendar_date(timestamp), time(20, 0), tzinfo=UTC)


def _outcome_evidence_id(
    *,
    target: PredictionEvaluationTarget,
    artifact_payload: Phase4MarketDataArtifact,
) -> str:
    evidence_digest = digest(
        "|".join(
            (
                target.target_id,
                artifact_payload.artifact_id,
                artifact_payload.provenance.provider_name,
            )
        )
    )
    return (
        "evidence-live-outcome-"
        f"{slug(target.candidate_id, fallback='candidate', allow_file_safe_punctuation=True)}-"
        f"{evidence_digest[:10]}"
    )


def _format_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _provider_name(provider: MarketDataProvider) -> str:
    return str(getattr(provider, "provider_name", provider.__class__.__name__))


def _fallback_limitations_only(limitations: Sequence[str]) -> list[str]:
    return [
        item
        for item in tuple(dict.fromkeys(limitations))
        if "status was" in item.lower()
        or "provider call failed" in item.lower()
        or "warning:" in item.lower()
    ]


def _final_attempt_status(status: PredictionOutcomeStatus) -> str:
    if status == PredictionOutcomeStatus.OBSERVED:
        return "completed"
    if status == PredictionOutcomeStatus.STALE:
        return "stale"
    return "blocked"


__all__ = [
    "PHASE7_LIVE_OUTCOME_TOOL_NAME",
    "PHASE7_LIVE_OUTCOME_TOOL_VERSION",
    "SUPPORTED_LIVE_OUTCOME_ASSET_CLASSES",
    "DefaultLiveOutcomeProviderFactory",
    "LiveOutcomeMarketDataSelection",
    "LiveOutcomeMaterializationArtifacts",
    "LiveOutcomeProviderFactory",
    "materialize_live_prediction_outcome_artifacts",
]
