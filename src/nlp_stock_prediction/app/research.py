"""Research workflow helpers for the terminal app."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from nlp_stock_prediction.app.settings import AppMode, AppSettings
from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.orchestration.report_bundle import ReportBundle
from nlp_stock_prediction.pipeline import generate_daily_report

ReportGenerator = Callable[[RunConfig], ReportBundle]


@dataclass(frozen=True)
class ResearchRequest:
    symbol: str
    run_date: date
    mode: AppMode
    output_dir: Path
    cache_dir: Path | None = None
    fixture_dir: Path | None = None

    @classmethod
    def from_settings(cls, settings: AppSettings, *, today: date | None = None) -> ResearchRequest:
        return cls(
            symbol=settings.default_symbol,
            run_date=today or date.today(),
            mode=settings.default_mode,
            output_dir=settings.output_dir,
            cache_dir=settings.cache_dir,
        )

    def to_run_config(self) -> RunConfig:
        offline = self.mode == "offline"
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("Symbol is required.")
        return RunConfig(
            run_date=self.run_date,
            output_dir=self.output_dir,
            symbol=symbol,
            fixture_dir=self.fixture_dir,
            cache_dir=self.cache_dir,
            offline=offline,
            source_mode="offline" if offline else "live",
            live_providers=not offline,
        )


def run_research_request(
    request: ResearchRequest,
    *,
    report_generator: ReportGenerator = generate_daily_report,
) -> ReportBundle:
    return report_generator(request.to_run_config())


__all__ = ["ReportGenerator", "ResearchRequest", "run_research_request"]
