"""Report rendering and fixture-backed audit writing helpers."""

from nlp_stock_prediction.reporting.fixtures import (
    OfflineFixtureBundle,
    build_offline_fixture_bundle,
)
from nlp_stock_prediction.reporting.json import render_json_report
from nlp_stock_prediction.reporting.markdown import render_markdown_report
from nlp_stock_prediction.reporting.view import ReportView

__all__ = [
    "OfflineFixtureBundle",
    "ReportView",
    "build_offline_fixture_bundle",
    "render_json_report",
    "render_markdown_report",
]
