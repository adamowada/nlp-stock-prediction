"""Fixture-backed Reddit provider for deterministic Lane A tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from nlp_stock_prediction.contracts import (
    CredentialState,
    EvidenceRequest,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    RetrievalMethod,
    SourceEvidence,
    TickerDiscoveryRequest,
    TickerDiscoveryResult,
    TickerDiscoveryStatus,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.reddit.discovery import discover_tickers_from_devvit_html
from nlp_stock_prediction.reddit.evidence import normalize_reddit_evidence


@dataclass(frozen=True, slots=True)
class FixtureRedditProvider:
    """A deterministic Reddit provider backed by in-memory fixtures."""

    ticker_card_html: str
    discussion_records: Sequence[Mapping[str, object]]
    fetched_at: datetime
    raw_ticker_snapshot_id: str
    raw_discussion_snapshot_id: str
    provider_name: str = "reddit"
    latency_ms: int | None = 0

    def discover_tickers(
        self,
        request: TickerDiscoveryRequest,
    ) -> ProviderResult[TickerDiscoveryResult]:
        discovery = discover_tickers_from_devvit_html(
            self.ticker_card_html,
            request=request,
            fetched_at=self.fetched_at,
            raw_snapshot_id=self.raw_ticker_snapshot_id,
            retrieval_method=RetrievalMethod.FIXTURE,
            provider_name=self.provider_name,
        )
        status = (
            ProviderStatus.OK
            if discovery.status == TickerDiscoveryStatus.VALID
            else ProviderStatus.PARTIAL
        )
        health = self._health(status=status, warnings=discovery.warnings)
        return ProviderResult[TickerDiscoveryResult](
            provider_name=self.provider_name,
            status=status,
            request=request,
            fetched_at=self.fetched_at,
            data=discovery,
            warnings=discovery.warnings,
            health=health,
            raw_snapshot_id=self.raw_ticker_snapshot_id,
            cache_key=f"reddit:ticker-card:{request.run_date.isoformat()}",
        )

    def fetch_discussion(
        self,
        request: EvidenceRequest,
    ) -> ProviderResult[tuple[SourceEvidence, ...]]:
        evidence = normalize_reddit_evidence(
            self.discussion_records,
            request=request,
            fetched_at=self.fetched_at,
            raw_snapshot_id=self.raw_discussion_snapshot_id,
            retrieval_method=RetrievalMethod.FIXTURE,
            provider_name=self.provider_name,
        )
        if evidence:
            return ProviderResult[tuple[SourceEvidence, ...]](
                provider_name=self.provider_name,
                status=ProviderStatus.OK,
                request=request,
                fetched_at=self.fetched_at,
                data=evidence,
                health=self._health(status=ProviderStatus.OK),
                raw_snapshot_id=self.raw_discussion_snapshot_id,
                cache_key=f"reddit:discussion:{request.run_date.isoformat()}",
            )

        warning = ProviderWarning(
            code=WarningCode.NO_DATA,
            severity=WarningSeverity.INFO,
            message="Reddit discussion fixture produced no high-confidence ticker evidence.",
            provider_name=self.provider_name,
            occurred_at=self.fetched_at,
            raw_snapshot_id=self.raw_discussion_snapshot_id,
            metadata={"request_id": request.request_id},
        )
        return ProviderResult[tuple[SourceEvidence, ...]](
            provider_name=self.provider_name,
            status=ProviderStatus.EMPTY,
            request=request,
            fetched_at=self.fetched_at,
            data=None,
            warnings=(warning,),
            health=self._health(status=ProviderStatus.EMPTY, warnings=(warning,)),
            raw_snapshot_id=self.raw_discussion_snapshot_id,
            cache_key=f"reddit:discussion:{request.run_date.isoformat()}",
        )

    def health(self) -> ProviderHealth:
        return self._health(status=ProviderStatus.OK)

    def _health(
        self,
        *,
        status: ProviderStatus,
        warnings: tuple[ProviderWarning, ...] = (),
    ) -> ProviderHealth:
        return ProviderHealth(
            provider_name=self.provider_name,
            status=status,
            checked_at=self.fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            latency_ms=self.latency_ms,
            last_success_at=self.fetched_at if status == ProviderStatus.OK else None,
            warnings=warnings,
        )


__all__ = ["FixtureRedditProvider"]
