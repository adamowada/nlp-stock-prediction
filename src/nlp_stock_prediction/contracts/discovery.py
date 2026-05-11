"""Ticker discovery contracts."""

from __future__ import annotations

from datetime import date

from pydantic import Field, model_validator

from nlp_stock_prediction.contracts.base import ContractModel, NonEmptyStr, TickerSymbol
from nlp_stock_prediction.contracts.enums import TickerDiscoveryStatus
from nlp_stock_prediction.contracts.provenance import ProviderWarning, SourceProvenance


class TickerCandidate(ContractModel):
    """Raw ticker candidate found in the WSB Devvit ticker card."""

    symbol: TickerSymbol
    raw_identifier: NonEmptyStr
    raw_text: str | None = None
    first_seen_rank: int = Field(ge=0)
    source_url: str | None = None
    provenance: SourceProvenance


class TickerDiscoveryResult(ContractModel):
    """Normalized result of daily ticker-card discovery."""

    run_date: date
    status: TickerDiscoveryStatus
    candidates: tuple[TickerCandidate, ...] = Field(default_factory=tuple)
    tickers: tuple[TickerSymbol, ...] = Field(default_factory=tuple)
    warnings: tuple[ProviderWarning, ...] = Field(default_factory=tuple)
    raw_snapshot_id: str | None = None

    @model_validator(mode="after")
    def validate_valid_discovery(self) -> TickerDiscoveryResult:
        unique = tuple(dict.fromkeys(self.tickers))
        if len(unique) != len(self.tickers):
            raise ValueError("TickerDiscoveryResult.tickers must already be unique")
        if self.status == TickerDiscoveryStatus.VALID and len(self.tickers) != 6:
            raise ValueError("valid ticker discovery requires exactly six unique tickers")
        if self.status == TickerDiscoveryStatus.VALID and not self.raw_snapshot_id:
            raise ValueError("valid ticker discovery requires raw_snapshot_id")
        if self.status == TickerDiscoveryStatus.VALID:
            candidate_tickers = tuple(
                dict.fromkeys(candidate.symbol for candidate in self.candidates)
            )
            if candidate_tickers != self.tickers:
                raise ValueError("valid ticker discovery candidates must match tickers in order")
        if self.status != TickerDiscoveryStatus.VALID and not self.warnings:
            raise ValueError("invalid ticker discovery must include at least one warning")
        return self


__all__ = ["TickerCandidate", "TickerDiscoveryResult"]
