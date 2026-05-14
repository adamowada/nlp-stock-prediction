from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

import pytest

from nlp_stock_prediction.contracts import (
    CredentialState,
    Direction,
    MarketDataRequest,
    MarketSnapshot,
    PredictionOutcomeResult,
    PredictionOutcomeStatus,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.evaluation.live_outcomes import (
    LiveOutcomeMarketDataSelection,
    materialize_live_prediction_outcome_artifacts,
)
from nlp_stock_prediction.orchestration.phase6_service import Phase6Service
from nlp_stock_prediction.providers._base import provider_health, provider_result, provider_warning
from nlp_stock_prediction.storage import (
    InstrumentRecord,
    PredictionCandidateRecord,
    ResearchRunRecord,
    SQLiteStore,
)

RUN_ID = "run-phase7-live-outcome-materialization"
INSTRUMENT_ID = "instrument:equity:us:msft"
CANDIDATE_ID = "candidate-msft-directional-live-outcome"
NOW = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 5, 13, 20, 0, tzinfo=UTC)
WINDOW_START = datetime(2026, 5, 14, 13, 30, tzinfo=UTC)
WINDOW_END = datetime(2026, 5, 18, 20, 0, tzinfo=UTC)
EVALUATED_AT = datetime(2026, 5, 19, 21, 0, tzinfo=UTC)


class _SelectionFactory(Protocol):
    def market_data_selections(
        self,
        *,
        symbol: str,
        instrument: InstrumentRecord,
    ) -> tuple[LiveOutcomeMarketDataSelection, ...]: ...


class _StaticMarketDataProvider:
    provider_name = "verified-live-market-data"

    def __init__(
        self,
        *,
        status: ProviderStatus = ProviderStatus.OK,
        bars: Sequence[tuple[date, Decimal]] = (),
    ) -> None:
        self.status = status
        self.bars = tuple(bars)
        self.calls = 0

    def fetch_daily_candles(
        self,
        request: MarketDataRequest,
    ) -> ProviderResult[MarketSnapshot]:
        self.calls += 1
        warnings: tuple[ProviderWarning, ...] = ()
        if self.status != ProviderStatus.OK:
            warnings = (
                provider_warning(
                    provider_name=self.provider_name,
                    code=(
                        WarningCode.STALE_DATA
                        if self.status == ProviderStatus.STALE
                        else WarningCode.UPSTREAM_UNAVAILABLE
                    ),
                    severity=WarningSeverity.WARNING,
                    message=f"{self.provider_name} returned {self.status.value}.",
                    occurred_at=EVALUATED_AT,
                ),
            )
        snapshot = None
        if self.bars:
            from nlp_stock_prediction.contracts import PriceBar

            snapshot = MarketSnapshot(
                ticker=request.tickers[0],
                bars=tuple(
                    PriceBar(
                        ticker=request.tickers[0],
                        timestamp=bar_date,
                        open=value,
                        high=value,
                        low=value,
                        close=value,
                        adjusted_close=value,
                        volume=1_000_000,
                    )
                    for bar_date, value in self.bars
                ),
            )
        return provider_result(
            provider_name=self.provider_name,
            status=self.status,
            request=request,
            fetched_at=EVALUATED_AT,
            credential_state=CredentialState.NOT_REQUIRED,
            data=snapshot,
            warnings=warnings,
        )

    def health(self) -> ProviderHealth:
        return provider_health(
            provider_name=self.provider_name,
            status=self.status,
            checked_at=EVALUATED_AT,
            credential_state=CredentialState.NOT_REQUIRED,
        )


class _StaticSelectionFactory:
    def __init__(self, *selections: LiveOutcomeMarketDataSelection) -> None:
        self.selections = selections
        self.calls = 0

    def market_data_selections(
        self,
        *,
        symbol: str,
        instrument: InstrumentRecord,
    ) -> tuple[LiveOutcomeMarketDataSelection, ...]:
        self.calls += 1
        assert symbol == "MSFT"
        assert instrument.instrument_id == INSTRUMENT_ID
        return tuple(self.selections)


class _NoProviderFactory:
    calls = 0

    def market_data_selections(
        self,
        *,
        symbol: str,
        instrument: InstrumentRecord,
    ) -> tuple[LiveOutcomeMarketDataSelection, ...]:
        self.calls += 1
        raise AssertionError("provider factory should not be called")


def _store(tmp_path: Path, *, asset_class: str = "stock") -> SQLiteStore:
    store = SQLiteStore(tmp_path / "data" / "prediction-research.sqlite3")
    store.initialize()
    store.upsert_instrument(
        InstrumentRecord(
            instrument_id=INSTRUMENT_ID,
            symbol="MSFT",
            asset_class=asset_class,
            name="Microsoft Corporation",
            venue="NASDAQ",
        )
    )
    store.upsert_research_run(
        ResearchRunRecord(
            run_id=RUN_ID,
            run_kind="phase7_live_outcome_materialization",
            objective="Materialize an observed prediction outcome from live market data.",
            status="running",
            started_at=NOW,
            metadata={"report_data_mode": "live", "provider_mode": "live"},
        )
    )
    store.upsert_prediction_candidate(
        PredictionCandidateRecord(
            candidate_id=CANDIDATE_ID,
            run_id=RUN_ID,
            instrument_id=INSTRUMENT_ID,
            prediction_horizon="swing",
            prediction_type="directional",
            scenario="MSFT ends the fixed evaluation window above the cutoff comparison close.",
            direction=Direction.BULLISH.value,
            confidence=0.64,
            status="evidence_supported",
            baseline={
                "baseline_id": "cutoff_daily_close",
                "summary": "Compare the observed window-end close with the cutoff daily close.",
                "baseline_score": 0.5,
                "candidate_score": 0.64,
                "verdict": "above_baseline",
            },
        )
    )
    return store


@pytest.mark.unit
def test_live_outcome_materialization_fetches_real_provider_artifact_and_scores_window(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    provider = _StaticMarketDataProvider(
        bars=(
            (date(2026, 5, 13), Decimal("100")),
            (date(2026, 5, 18), Decimal("103")),
            (date(2026, 5, 19), Decimal("110")),
        )
    )
    factory = _StaticSelectionFactory(
        LiveOutcomeMarketDataSelection(
            provider=provider,
            source_url="https://example.test/msft/daily",
            role="primary",
        )
    )

    result = materialize_live_prediction_outcome_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        candidate_id=CANDIDATE_ID,
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        evaluated_at=EVALUATED_AT,
        provider_factory=factory,
    )

    assert provider.calls == 1
    assert result.outcome.status == PredictionOutcomeStatus.OBSERVED
    assert result.outcome.observed_result == PredictionOutcomeResult.SUPPORTED
    assert result.outcome.result_value == 103.0
    assert result.outcome.baseline_value == 100.0
    assert result.outcome.observed_at == WINDOW_END
    assert result.market_artifact_ids
    assert result.outcome_evidence_ids
    assert store.get_evidence(result.outcome_evidence_ids[0]) is not None

    outcome_record = store.get_prediction_outcome(result.outcome.outcome_id)
    review_record = store.get_prediction_outcome_evaluation(
        result.outcome_evaluation.outcome_evaluation_id
    )
    assert outcome_record is not None
    assert review_record is not None
    assert outcome_record.evaluation_attempt_id == result.evaluation_attempt_id
    assert review_record.evaluation_attempt_id == result.evaluation_attempt_id
    assert store.get_evaluation_attempt(result.evaluation_attempt_id) is not None


@pytest.mark.unit
def test_live_outcome_materialization_can_reuse_existing_market_artifact(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    provider = _StaticMarketDataProvider(
        bars=((date(2026, 5, 13), Decimal("100")), (date(2026, 5, 18), Decimal("103")))
    )
    artifact_dir = tmp_path / "reports" / RUN_ID / "audit"
    first = materialize_live_prediction_outcome_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=artifact_dir,
        run_id=RUN_ID,
        candidate_id=CANDIDATE_ID,
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        evaluated_at=EVALUATED_AT,
        provider_factory=_StaticSelectionFactory(
            LiveOutcomeMarketDataSelection(provider=provider, role="primary")
        ),
    )

    second = materialize_live_prediction_outcome_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=artifact_dir,
        run_id=RUN_ID,
        candidate_id=CANDIDATE_ID,
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        evaluated_at=EVALUATED_AT,
        provider_factory=_NoProviderFactory(),
        market_artifact_ids=first.market_artifact_ids,
    )

    assert second.outcome.status == PredictionOutcomeStatus.OBSERVED
    assert second.outcome.result_value == 103.0
    assert provider.calls == 1


@pytest.mark.unit
def test_live_outcome_materialization_records_unavailable_provider_without_shortcuts(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    provider = _StaticMarketDataProvider(status=ProviderStatus.UNCONFIGURED)

    result = materialize_live_prediction_outcome_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        candidate_id=CANDIDATE_ID,
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        evaluated_at=EVALUATED_AT,
        provider_factory=_StaticSelectionFactory(
            LiveOutcomeMarketDataSelection(provider=provider, role="primary")
        ),
    )

    assert result.outcome.status == PredictionOutcomeStatus.UNAVAILABLE
    assert result.outcome.observed_result is None
    assert result.outcome.result_value is None
    assert result.outcome.baseline_value is None
    assert result.market_artifact_ids
    assert result.outcome_evidence_ids == ()
    assert "unconfigured" in " ".join(result.outcome.limitations).lower()
    assert store.get_prediction_outcome(result.outcome.outcome_id) is not None


@pytest.mark.unit
def test_live_outcome_materialization_rejects_pre_window_payloads(tmp_path: Path) -> None:
    store = _store(tmp_path)
    provider = _StaticMarketDataProvider(
        bars=((date(2026, 5, 13), Decimal("100")), (date(2026, 5, 16), Decimal("102")))
    )

    result = materialize_live_prediction_outcome_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        candidate_id=CANDIDATE_ID,
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        evaluated_at=EVALUATED_AT,
        provider_factory=_StaticSelectionFactory(
            LiveOutcomeMarketDataSelection(provider=provider, role="primary")
        ),
    )

    assert result.outcome.status == PredictionOutcomeStatus.UNAVAILABLE
    assert any("post-window usable bar" in item for item in result.outcome.limitations)


@pytest.mark.unit
def test_live_outcome_materialization_marks_unsupported_asset_classes_unavailable(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, asset_class="crypto")
    factory = _NoProviderFactory()

    result = materialize_live_prediction_outcome_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        candidate_id=CANDIDATE_ID,
        point_in_time_cutoff=CUTOFF,
        evaluation_window_start=WINDOW_START,
        evaluation_window_end=WINDOW_END,
        evaluated_at=EVALUATED_AT,
        provider_factory=factory,
    )

    assert factory.calls == 0
    assert result.outcome.status == PredictionOutcomeStatus.UNAVAILABLE
    assert result.market_artifact_ids == ()
    assert any("crypto" in item for item in result.outcome.limitations)


@pytest.mark.unit
def test_phase6_service_exposes_live_materialization_without_observed_result_shortcut(
    tmp_path: Path,
) -> None:
    _store(tmp_path)
    service = Phase6Service(
        repo_root=tmp_path,
        live_outcome_provider_factory=_StaticSelectionFactory(
            LiveOutcomeMarketDataSelection(
                provider=_StaticMarketDataProvider(
                    bars=(
                        (date(2026, 5, 13), Decimal("100")),
                        (date(2026, 5, 18), Decimal("103")),
                    )
                ),
                role="primary",
            )
        ),
    )

    result = service.phase7_live_outcome_materialization(
        run_id=RUN_ID,
        candidate_id=CANDIDATE_ID,
        point_in_time_cutoff=CUTOFF.isoformat(),
        evaluation_window_start=WINDOW_START.isoformat(),
        evaluation_window_end=WINDOW_END.isoformat(),
        evaluated_at=EVALUATED_AT.isoformat(),
    )

    assert result["outcome_status"] == "observed"
    assert result["observed_result"] == "supported"
    assert result["result_value"] == 103.0
    assert result["baseline_value"] == 100.0

    with pytest.raises(TypeError):
        service.phase7_live_outcome_materialization(  # type: ignore[call-arg]
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            point_in_time_cutoff=CUTOFF.isoformat(),
            evaluation_window_start=WINDOW_START.isoformat(),
            evaluation_window_end=WINDOW_END.isoformat(),
            observed_result="supported",
        )


@pytest.mark.live_api
def test_live_outcome_materialization_opt_in_live_api_smoke(tmp_path: Path) -> None:
    if os.environ.get("NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS") != "1":
        pytest.skip("Set NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS=1 to run live outcome smoke.")
    from nlp_stock_prediction.evaluation.live_outcomes import DefaultLiveOutcomeProviderFactory

    store = _store(tmp_path)
    factory = DefaultLiveOutcomeProviderFactory()

    result = materialize_live_prediction_outcome_artifacts(
        store=store,
        repo_root=tmp_path,
        artifact_dir=tmp_path / "reports" / RUN_ID / "audit",
        run_id=RUN_ID,
        candidate_id=CANDIDATE_ID,
        point_in_time_cutoff=datetime(2026, 5, 13, 12, 30, tzinfo=UTC),
        evaluation_window_start=datetime(2026, 5, 13, 13, 30, tzinfo=UTC),
        evaluation_window_end=datetime(2026, 5, 13, 20, 0, tzinfo=UTC),
        evaluated_at=datetime.now(UTC),
        provider_factory=factory,
    )

    if result.outcome.status != PredictionOutcomeStatus.OBSERVED:
        pytest.fail(
            "live outcome materialization did not produce an observed market outcome: "
            + "; ".join(result.outcome.limitations)
        )
    assert result.market_artifact_ids
    assert result.outcome_evidence_ids
