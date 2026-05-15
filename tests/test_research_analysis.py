from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from nlp_stock_prediction.analysis.fundamentals import analyze_fundamentals
from nlp_stock_prediction.analysis.macro import analyze_macro_context
from nlp_stock_prediction.analysis.sector import analyze_sector_context
from nlp_stock_prediction.analysis.technical import analyze_technical_snapshot
from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    FundamentalsSnapshot,
    MacroSeries,
    MacroSnapshot,
    MarketSnapshot,
    PriceBar,
    ProviderMetric,
    TimeHorizon,
)

RUN_DATE = date(2026, 5, 11)


def _decimal(value: float | int | str) -> Decimal:
    return Decimal(str(value))


def _uptrend_market_snapshot() -> MarketSnapshot:
    bars: list[PriceBar] = []
    previous_close = Decimal("100")
    for index in range(60):
        day = RUN_DATE - timedelta(days=59 - index)
        base_close = Decimal("100") + Decimal(index) * Decimal("1.35")
        open_price = previous_close + Decimal("0.30")
        if index == 59:
            open_price = previous_close * Decimal("1.035")
            close = open_price * Decimal("1.025")
            volume = 3_800_000
        else:
            close = base_close
            volume = 950_000 + index * 8_000
        high = max(open_price, close) + Decimal("1.10")
        low = min(open_price, close) - Decimal("0.90")
        bars.append(
            PriceBar(
                ticker="tsla",
                timestamp=day,
                open=open_price.quantize(Decimal("0.01")),
                high=high.quantize(Decimal("0.01")),
                low=low.quantize(Decimal("0.01")),
                close=close.quantize(Decimal("0.01")),
                volume=volume,
            )
        )
        previous_close = close
    return MarketSnapshot(ticker="TSLA", bars=tuple(bars))


def _fundamentals_snapshot(ticker: str, **metrics: object) -> FundamentalsSnapshot:
    return FundamentalsSnapshot(
        ticker=ticker,
        company_name=f"{ticker.upper()} Inc.",
        metrics=tuple(
            ProviderMetric(name=name, value=value, unit=None, as_of=RUN_DATE)
            for name, value in metrics.items()
        ),
    )


@pytest.mark.unit
def test_technical_analysis_computes_indicators_and_price_context() -> None:
    analysis = analyze_technical_snapshot(_uptrend_market_snapshot(), as_of=RUN_DATE)

    metric_by_name = {metric.name: metric for metric in analysis.metrics}

    assert analysis.ticker == "TSLA"
    assert analysis.signal == AnalysisSignal.SUPPORTS
    assert analysis.trend == "uptrend"
    assert analysis.support_levels
    assert analysis.resistance_levels
    assert "above-average" in (analysis.volume_summary or "")
    assert "gap up" in (analysis.gap_summary or "")
    assert "bullish" in (analysis.candlestick_summary or "")
    assert {"sma-20", "sma-50", "rsi-14", "macd-line", "macd-signal"} <= set(metric_by_name)
    assert {"macd-histogram", "relative-volume", "average-true-range-pct"} <= set(metric_by_name)
    assert float(metric_by_name["rsi-14"].value or 0) > 70.0
    assert float(metric_by_name["macd-histogram"].value or 0) > 0.0
    assert float(metric_by_name["relative-volume"].value or 0) > 2.0


@pytest.mark.unit
def test_fundamental_analysis_summarizes_valuation_quality_and_event_risk() -> None:
    snapshot = _fundamentals_snapshot(
        "NVDA",
        pe_ratio=Decimal("72"),
        price_to_sales=Decimal("18"),
        net_margin=Decimal("0.48"),
        revenue_growth_yoy=Decimal("0.62"),
        debt_to_equity=Decimal("0.35"),
        current_ratio=Decimal("3.4"),
        days_until_earnings=5,
        latest_filing_type="10-Q",
    )

    analysis = analyze_fundamentals(snapshot, as_of=RUN_DATE)

    assert analysis.ticker == "NVDA"
    assert analysis.signal == AnalysisSignal.MIXED
    assert "premium" in (analysis.valuation_summary or "")
    assert "profitable" in (analysis.profitability_summary or "")
    assert "growth" in (analysis.growth_summary or "")
    assert "balance sheet" in (analysis.balance_sheet_risk or "")
    assert "5 days" in (analysis.earnings_timing or "")
    assert analysis.notable_filings == ("10-Q",)


@pytest.mark.unit
def test_fundamental_analysis_unknown_without_provider_metrics() -> None:
    analysis = analyze_fundamentals(FundamentalsSnapshot(ticker="NVDA"), as_of=RUN_DATE)

    assert analysis.signal == AnalysisSignal.UNKNOWN
    assert analysis.confidence == 0.0
    assert analysis.metrics == ()
    assert "unavailable" in analysis.summary


@pytest.mark.unit
def test_fundamental_analysis_scales_percent_string_metrics() -> None:
    analysis = analyze_fundamentals(
        _fundamentals_snapshot("NVDA", net_margin="48%", revenue_growth_yoy="62%"),
        as_of=RUN_DATE,
    )

    assert analysis.signal == AnalysisSignal.SUPPORTS
    assert "0.48" in (analysis.profitability_summary or "")
    assert "62.0%" in (analysis.growth_summary or "")


@pytest.mark.unit
def test_sector_context_compares_against_peers_or_uses_etf_proxy() -> None:
    target = _fundamentals_snapshot(
        "NVDA",
        pe_ratio=Decimal("45"),
        net_margin=Decimal("0.50"),
        revenue_growth_yoy=Decimal("0.70"),
    )
    peers = (
        _fundamentals_snapshot(
            "AMD",
            pe_ratio=Decimal("36"),
            net_margin=Decimal("0.12"),
            revenue_growth_yoy=Decimal("0.10"),
        ),
        _fundamentals_snapshot(
            "INTC",
            pe_ratio=Decimal("28"),
            net_margin=Decimal("0.05"),
            revenue_growth_yoy=Decimal("-0.03"),
        ),
    )

    peer_context = analyze_sector_context(target, peers=peers, sector="Semiconductors")

    assert peer_context.signal == AnalysisSignal.SUPPORTS
    assert peer_context.peers == ("AMD", "INTC")
    assert peer_context.benchmark_symbol is None
    assert "peers" in peer_context.summary
    assert any(metric.name == "peer-median-revenue-growth-yoy" for metric in peer_context.metrics)

    proxy_context = analyze_sector_context(
        target,
        sector="Semiconductors",
        benchmark_symbol="SMH",
        benchmark_metrics=(
            ProviderMetric(
                name="sector_etf_return_20d",
                value=Decimal("0.035"),
                unit="pct",
                as_of=RUN_DATE,
            ),
        ),
    )

    assert proxy_context.signal == AnalysisSignal.SUPPORTS
    assert proxy_context.peers == ()
    assert proxy_context.benchmark_symbol == "SMH"
    assert any("ETF proxy" in assumption for assumption in proxy_context.assumptions)


@pytest.mark.unit
def test_macro_context_maps_conditions_to_strategy_horizons() -> None:
    snapshot = MacroSnapshot(
        series=(
            MacroSeries(
                series_id="FEDFUNDS",
                name="Fed funds rate",
                values=(
                    ProviderMetric(
                        name="fed_funds_change_3m",
                        value=Decimal("-0.25"),
                        unit="pct",
                        as_of=RUN_DATE,
                    ),
                ),
            ),
            MacroSeries(
                series_id="CPIAUCSL",
                name="CPI inflation",
                values=(
                    ProviderMetric(
                        name="cpi_yoy_change_3m",
                        value=Decimal("-0.30"),
                        unit="pct",
                        as_of=RUN_DATE,
                    ),
                ),
            ),
            MacroSeries(
                series_id="VIXCLS",
                name="VIX",
                values=(ProviderMetric(name="vix", value=Decimal("31"), as_of=RUN_DATE),),
            ),
            MacroSeries(
                series_id="A191RL1Q225SBEA",
                name="Real GDP growth",
                values=(
                    ProviderMetric(
                        name="gdp_growth",
                        value=Decimal("2.4"),
                        unit="pct",
                        as_of=RUN_DATE,
                    ),
                ),
            ),
        )
    )

    weekly = analyze_macro_context(snapshot, horizon=TimeHorizon.WEEKLY, as_of=RUN_DATE)
    monthly = analyze_macro_context(snapshot, horizon=TimeHorizon.MONTHLY, as_of=RUN_DATE)

    assert weekly.horizon == TimeHorizon.WEEKLY
    assert monthly.horizon == TimeHorizon.MONTHLY
    assert weekly.signal == AnalysisSignal.MIXED
    assert monthly.signal == AnalysisSignal.SUPPORTS
    assert any("volatility" in factor for factor in weekly.conflicting_factors)
    assert any("growth" in factor for factor in monthly.supportive_factors)
