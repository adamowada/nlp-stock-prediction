"""Macro context synthesis by strategy horizon."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from nlp_stock_prediction.analysis._metrics import (
    analysis_metric,
    metric_decimal,
    provider_metric_map,
)
from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    MacroContext,
    MacroSnapshot,
    MetricValue,
    ProviderMetric,
    TimeHorizon,
)

SHORT_HORIZONS = {
    TimeHorizon.INTRADAY,
    TimeHorizon.SWING,
    TimeHorizon.WEEKLY,
    TimeHorizon.EARNINGS_EVENT,
}


def analyze_macro_context(
    snapshot: MacroSnapshot,
    *,
    horizon: TimeHorizon,
    as_of: date | datetime,
) -> MacroContext:
    """Summarize macro support and conflict for a strategy horizon."""

    metrics = _flatten_macro_metrics(snapshot)
    supportive_factors: list[str] = []
    conflicting_factors: list[str] = []
    fed_funds_change = metric_decimal(metrics, "fed_funds_change_3m", "rates_change_3m")
    cpi_change = metric_decimal(metrics, "cpi_yoy_change_3m", "inflation_change_3m")
    vix = metric_decimal(metrics, "vix", "vix_close")
    gdp_growth = metric_decimal(metrics, "gdp_growth", "real_gdp_growth")
    yield_curve = metric_decimal(metrics, "yield_curve_10y2y", "ten_two_spread")
    context_metrics = _context_metrics(
        as_of=as_of,
        fed_funds_change=fed_funds_change,
        cpi_change=cpi_change,
        vix=vix,
        gdp_growth=gdp_growth,
        yield_curve=yield_curve,
    )

    if fed_funds_change is not None:
        if fed_funds_change <= Decimal("-0.10"):
            supportive_factors.append("Rates are easing, which can support risk appetite.")
        elif fed_funds_change >= Decimal("0.10"):
            conflicting_factors.append("Rising rates can pressure speculative risk appetite.")
    if cpi_change is not None:
        if cpi_change <= Decimal("-0.10"):
            supportive_factors.append("Inflation trend is cooling.")
        elif cpi_change >= Decimal("0.10"):
            conflicting_factors.append("Inflation trend is heating up.")
    if horizon in SHORT_HORIZONS and vix is not None:
        if vix >= Decimal("28"):
            conflicting_factors.append("High volatility can overwhelm short-horizon setups.")
        elif vix <= Decimal("20"):
            supportive_factors.append("Contained volatility supports short-horizon execution.")
    if horizon in {TimeHorizon.MONTHLY, TimeHorizon.UNKNOWN} and gdp_growth is not None:
        if gdp_growth > Decimal("0"):
            supportive_factors.append("Positive growth supports the monthly horizon.")
        else:
            conflicting_factors.append("Contracting growth conflicts with the monthly horizon.")
    if yield_curve is not None and yield_curve < Decimal("0"):
        conflicting_factors.append("Inverted yield curve keeps recession risk in view.")

    signal = _signal(tuple(supportive_factors), tuple(conflicting_factors))
    confidence = _confidence(len(context_metrics), supportive_factors, conflicting_factors)
    summary = _summary(signal, horizon, supportive_factors, conflicting_factors)

    return MacroContext(
        as_of=as_of,
        horizon=horizon,
        summary=summary,
        signal=signal,
        confidence=confidence,
        metrics=context_metrics,
        supportive_factors=tuple(supportive_factors),
        conflicting_factors=tuple(conflicting_factors),
    )


def _flatten_macro_metrics(snapshot: MacroSnapshot) -> dict[str, ProviderMetric]:
    flattened: dict[str, ProviderMetric] = {}
    for series in snapshot.series:
        flattened.update(provider_metric_map(series.values))
    return flattened


def _context_metrics(
    *,
    as_of: date | datetime,
    fed_funds_change: Decimal | None,
    cpi_change: Decimal | None,
    vix: Decimal | None,
    gdp_growth: Decimal | None,
    yield_curve: Decimal | None,
) -> tuple[MetricValue, ...]:
    metrics: list[MetricValue] = []
    for name, value in (
        ("fed-funds-change-3m", fed_funds_change),
        ("cpi-yoy-change-3m", cpi_change),
        ("vix", vix),
        ("gdp-growth", gdp_growth),
        ("yield-curve-10y2y", yield_curve),
    ):
        if value is not None:
            metrics.append(analysis_metric(name, value, as_of=as_of))
    return tuple(metrics)


def _signal(
    supportive_factors: tuple[str, ...],
    conflicting_factors: tuple[str, ...],
) -> AnalysisSignal:
    if supportive_factors and conflicting_factors:
        return AnalysisSignal.MIXED
    if supportive_factors:
        return AnalysisSignal.SUPPORTS
    if conflicting_factors:
        return AnalysisSignal.CONFLICTS
    return AnalysisSignal.NEUTRAL


def _confidence(
    metric_count: int,
    supportive_factors: list[str],
    conflicting_factors: list[str],
) -> float:
    factor_count = len(supportive_factors) + len(conflicting_factors)
    return round(min(0.85, 0.25 + metric_count * 0.07 + factor_count * 0.08), 4)


def _summary(
    signal: AnalysisSignal,
    horizon: TimeHorizon,
    supportive_factors: list[str],
    conflicting_factors: list[str],
) -> str:
    support_text = "; ".join(supportive_factors) if supportive_factors else "no clear supports"
    conflict_text = "; ".join(conflicting_factors) if conflicting_factors else "no clear conflicts"
    return f"Macro context is {signal.value} for {horizon.value}: {support_text}; {conflict_text}."
