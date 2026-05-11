"""Company fundamental synthesis from provider-supplied facts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal

from nlp_stock_prediction.analysis._metrics import (
    analysis_metric,
    metric_decimal,
    metric_text,
    metric_values_from_provider,
    provider_metric_map,
)
from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    FundamentalAnalysis,
    FundamentalsSnapshot,
    MetricValue,
    ProviderMetric,
)


def analyze_fundamentals(
    snapshot: FundamentalsSnapshot,
    *,
    as_of: date | datetime | None = None,
) -> FundamentalAnalysis:
    """Summarize valuation, quality, growth, balance sheet, and event context."""

    metrics = provider_metric_map(snapshot.metrics)
    valuation_score, valuation_summary = _valuation(metrics)
    profitability_score, profitability_summary = _profitability(metrics)
    growth_score, growth_summary = _growth(metrics)
    balance_score, balance_summary = _balance_sheet(metrics)
    earnings_timing = _earnings_timing(metrics)
    notable_filings = _notable_filings(metrics)
    component_scores = (
        valuation_score,
        profitability_score,
        growth_score,
        balance_score,
    )
    known_scores = tuple(score for score in component_scores if score is not None)
    confidence = round(min(1.0, 0.25 + (len(known_scores) * 0.17)), 4)
    average_score = sum(known_scores) / len(known_scores) if known_scores else 0.5
    signal = _fundamental_signal(
        average_score=average_score,
        valuation_score=valuation_score,
        profitability_score=profitability_score,
        growth_score=growth_score,
        balance_score=balance_score,
    )
    derived_metrics = _derived_metrics(
        as_of=as_of,
        valuation_score=valuation_score,
        profitability_score=profitability_score,
        growth_score=growth_score,
        balance_score=balance_score,
    )

    return FundamentalAnalysis(
        ticker=snapshot.ticker,
        summary=(
            f"{signal.value.title()} fundamental context: {valuation_summary} "
            f"{profitability_summary} {growth_summary} {balance_summary}"
        ),
        signal=signal,
        confidence=confidence,
        metrics=metric_values_from_provider(snapshot.metrics) + derived_metrics,
        valuation_summary=valuation_summary,
        profitability_summary=profitability_summary,
        growth_summary=growth_summary,
        balance_sheet_risk=balance_summary,
        earnings_timing=earnings_timing,
        notable_filings=notable_filings,
    )


def _valuation(metrics: Mapping[str, ProviderMetric]) -> tuple[float | None, str]:
    pe_ratio = metric_decimal(metrics, "pe_ratio", "trailing_pe", "forward_pe")
    price_to_sales = metric_decimal(metrics, "price_to_sales", "ps_ratio")
    if pe_ratio is None and price_to_sales is None:
        return None, "Valuation data is unavailable."
    scores: list[float] = []
    phrases: list[str] = []
    if pe_ratio is not None:
        if pe_ratio <= Decimal("20"):
            scores.append(0.85)
            phrases.append(f"P/E {pe_ratio:g} is reasonable")
        elif pe_ratio <= Decimal("40"):
            scores.append(0.55)
            phrases.append(f"P/E {pe_ratio:g} is elevated")
        else:
            scores.append(0.20)
            phrases.append(f"P/E {pe_ratio:g} implies premium valuation risk")
    if price_to_sales is not None:
        if price_to_sales <= Decimal("5"):
            scores.append(0.80)
            phrases.append(f"price/sales {price_to_sales:g} is contained")
        elif price_to_sales <= Decimal("12"):
            scores.append(0.50)
            phrases.append(f"price/sales {price_to_sales:g} is elevated")
        else:
            scores.append(0.20)
            phrases.append(f"price/sales {price_to_sales:g} reinforces premium valuation")
    return sum(scores) / len(scores), "Valuation: " + "; ".join(phrases) + "."


def _profitability(metrics: Mapping[str, ProviderMetric]) -> tuple[float | None, str]:
    net_margin = metric_decimal(metrics, "net_margin", "profit_margin")
    operating_margin = metric_decimal(metrics, "operating_margin")
    return_on_equity = metric_decimal(metrics, "return_on_equity", "roe")
    values = tuple(
        value for value in (net_margin, operating_margin, return_on_equity) if value is not None
    )
    if not values:
        return None, "Profitability data is unavailable."
    margin = max(values)
    if margin >= Decimal("0.20"):
        score = 0.90
        label = "highly profitable"
    elif margin > Decimal("0"):
        score = 0.65
        label = "profitable"
    else:
        score = 0.20
        label = "unprofitable"
    return score, f"Profitability: company is {label} with best margin/ROE metric {margin:g}."


def _growth(metrics: Mapping[str, ProviderMetric]) -> tuple[float | None, str]:
    revenue_growth = metric_decimal(metrics, "revenue_growth_yoy", "revenue_growth")
    eps_growth = metric_decimal(metrics, "eps_growth_yoy", "earnings_growth")
    values = tuple(value for value in (revenue_growth, eps_growth) if value is not None)
    if not values:
        return None, "Growth data is unavailable."
    average_growth = sum(values) / Decimal(len(values))
    if average_growth >= Decimal("0.20"):
        score = 0.90
        label = "strong growth"
    elif average_growth > Decimal("0"):
        score = 0.65
        label = "positive growth"
    else:
        score = 0.25
        label = "contracting growth"
    return score, f"Growth: {label} at roughly {average_growth:.1%} year over year."


def _balance_sheet(metrics: Mapping[str, ProviderMetric]) -> tuple[float | None, str]:
    debt_to_equity = metric_decimal(metrics, "debt_to_equity")
    current_ratio = metric_decimal(metrics, "current_ratio")
    if debt_to_equity is None and current_ratio is None:
        return None, "Balance sheet risk data is unavailable."
    score = 0.5
    parts: list[str] = []
    if debt_to_equity is not None:
        if debt_to_equity <= Decimal("0.75"):
            score += 0.25
            parts.append(f"debt/equity {debt_to_equity:g} is manageable")
        elif debt_to_equity >= Decimal("2"):
            score -= 0.25
            parts.append(f"debt/equity {debt_to_equity:g} is elevated")
        else:
            parts.append(f"debt/equity {debt_to_equity:g} is moderate")
    if current_ratio is not None:
        if current_ratio >= Decimal("1.5"):
            score += 0.20
            parts.append(f"current ratio {current_ratio:g} supports liquidity")
        elif current_ratio < Decimal("1"):
            score -= 0.20
            parts.append(f"current ratio {current_ratio:g} is tight")
        else:
            parts.append(f"current ratio {current_ratio:g} is adequate")
    bounded_score = min(max(score, 0.0), 1.0)
    return bounded_score, "balance sheet risk: " + "; ".join(parts) + "."


def _earnings_timing(metrics: Mapping[str, ProviderMetric]) -> str | None:
    days = metric_decimal(metrics, "days_until_earnings")
    if days is None:
        text = metric_text(metrics, "next_earnings_date", "earnings_date")
        return f"Earnings timing: next event around {text}." if text else None
    day_count = int(days)
    if day_count == 0:
        return "Earnings timing: expected today."
    if day_count > 0:
        return f"Earnings timing: expected in {day_count} days."
    return f"Earnings timing: last reported {abs(day_count)} days ago."


def _notable_filings(metrics: Mapping[str, ProviderMetric]) -> tuple[str, ...]:
    filings: list[str] = []
    for name in ("latest_filing_type", "recent_filing_type", "filing_type"):
        filing = metric_text(metrics, name)
        if filing and filing not in filings:
            filings.append(filing)
    return tuple(filings)


def _fundamental_signal(
    *,
    average_score: float,
    valuation_score: float | None,
    profitability_score: float | None,
    growth_score: float | None,
    balance_score: float | None,
) -> AnalysisSignal:
    strong_positive_count = sum(
        1
        for score in (profitability_score, growth_score, balance_score)
        if score is not None and score >= 0.70
    )
    has_premium_valuation = valuation_score is not None and valuation_score <= 0.30
    if has_premium_valuation and strong_positive_count >= 2:
        return AnalysisSignal.MIXED
    if average_score >= 0.67:
        return AnalysisSignal.SUPPORTS
    if average_score <= 0.40:
        return AnalysisSignal.CONFLICTS
    return AnalysisSignal.MIXED


def _derived_metrics(
    *,
    as_of: date | datetime | None,
    valuation_score: float | None,
    profitability_score: float | None,
    growth_score: float | None,
    balance_score: float | None,
) -> tuple[MetricValue, ...]:
    return tuple(
        analysis_metric(name, None if value is None else round(value, 4), as_of=as_of)
        for name, value in (
            ("valuation-score", valuation_score),
            ("profitability-score", profitability_score),
            ("growth-score", growth_score),
            ("balance-sheet-score", balance_score),
        )
        if value is not None
    )
