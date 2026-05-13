"""CLI orchestration for deterministic research report generation."""

from __future__ import annotations

from nlp_stock_prediction.contracts.providers import RunConfig
from nlp_stock_prediction.orchestration import (
    DUMMY_ORCHESTRATION_DISABLED_MESSAGE,
    ReportBundle,
    generate_dummy_report_bundle,
)

LIVE_ORCHESTRATION_DISABLED_MESSAGE = DUMMY_ORCHESTRATION_DISABLED_MESSAGE


def generate_daily_report(config: RunConfig) -> ReportBundle:
    """Generate a deterministic Phase 2 orchestrated research report bundle."""

    return generate_dummy_report_bundle(config)


__all__ = ["LIVE_ORCHESTRATION_DISABLED_MESSAGE", "ReportBundle", "generate_daily_report"]
