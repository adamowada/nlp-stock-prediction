"""Technical analysis derived from deterministic daily candle snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from itertools import pairwise

from nlp_stock_prediction.analysis._metrics import analysis_metric, clamp
from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    MarketSnapshot,
    MetricValue,
    PriceBar,
    TechnicalAnalysis,
)


def analyze_technical_snapshot(
    snapshot: MarketSnapshot,
    *,
    as_of: date | datetime | None = None,
) -> TechnicalAnalysis:
    """Create a contract-shaped technical analysis from daily price bars."""

    bars = tuple(sorted(snapshot.bars, key=_bar_sort_key))
    if not bars:
        return TechnicalAnalysis(
            ticker=snapshot.ticker,
            summary="No daily market bars were available for technical analysis.",
            signal=AnalysisSignal.UNKNOWN,
            confidence=0.0,
            assumptions=("Technical analysis requires daily OHLCV bars.",),
        )

    closes = [_as_float(bar.close) for bar in bars]
    latest = bars[-1]
    latest_close = closes[-1]
    sma_20 = _simple_moving_average(closes, 20)
    sma_50 = _simple_moving_average(closes, 50)
    previous_sma_20 = _simple_moving_average(closes[:-1], 20)
    rsi = _relative_strength_index(closes)
    macd_line, macd_signal, macd_histogram = _macd(closes)
    relative_volume = _relative_volume(bars)
    atr_pct = _average_true_range_pct(bars)
    gap_pct = _latest_gap_pct(bars)
    trend = _trend(latest_close, sma_20, sma_50, previous_sma_20, macd_histogram)
    signal = _signal_from_trend(trend)
    support, resistance = _support_resistance(bars)
    volume_summary = _volume_summary(relative_volume)
    volatility_summary = _volatility_summary(atr_pct)
    gap_summary = _gap_summary(gap_pct)
    candlestick_summary = _candlestick_summary(latest)
    metrics = _technical_metrics(
        as_of=as_of,
        sma_20=sma_20,
        sma_50=sma_50,
        rsi=rsi,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        relative_volume=relative_volume,
        atr_pct=atr_pct,
        gap_pct=gap_pct,
    )
    confidence = _confidence(len(bars), relative_volume, trend)

    return TechnicalAnalysis(
        ticker=snapshot.ticker,
        summary=(
            f"Technical setup is a {trend}: close {latest.close} versus "
            f"SMA20 {sma_20:.2f} and SMA50 {sma_50:.2f}; RSI {rsi:.1f}; "
            f"MACD histogram {macd_histogram:.2f}."
        ),
        signal=signal,
        confidence=confidence,
        metrics=metrics,
        trend=trend,
        support_levels=support,
        resistance_levels=resistance,
        volume_summary=volume_summary,
        volatility_summary=volatility_summary,
        gap_summary=gap_summary,
        candlestick_summary=candlestick_summary,
    )


def _bar_sort_key(bar: PriceBar) -> int:
    timestamp = bar.timestamp
    if isinstance(timestamp, datetime):
        return timestamp.date().toordinal()
    return timestamp.toordinal()


def _as_float(value: Decimal) -> float:
    return float(value)


def _simple_moving_average(values: Sequence[float], window: int) -> float:
    if not values:
        return 0.0
    bounded_window = min(window, len(values))
    return sum(values[-bounded_window:]) / bounded_window


def _relative_strength_index(closes: Sequence[float], period: int = 14) -> float:
    if len(closes) < 2:
        return 50.0
    changes = [current - previous for previous, current in pairwise(closes)]
    recent_changes = changes[-period:]
    gains = [change for change in recent_changes if change > 0]
    losses = [-change for change in recent_changes if change < 0]
    average_gain = sum(gains) / period
    average_loss = sum(losses) / period
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _ema(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    multiplier = 2.0 / (period + 1.0)
    ema_values = [values[0]]
    for value in values[1:]:
        ema_values.append((value - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def _macd(closes: Sequence[float]) -> tuple[float, float, float]:
    if not closes:
        return 0.0, 0.0, 0.0
    ema_12 = _ema(closes, 12)
    ema_26 = _ema(closes, 26)
    macd_values = [fast - slow for fast, slow in zip(ema_12, ema_26, strict=False)]
    signal_values = _ema(macd_values, 9)
    line = macd_values[-1]
    signal = signal_values[-1]
    return line, signal, line - signal


def _relative_volume(bars: Sequence[PriceBar], window: int = 20) -> float:
    if len(bars) < 2:
        return 1.0
    prior_bars = bars[:-1][-window:]
    average_volume = sum(bar.volume for bar in prior_bars) / len(prior_bars)
    if average_volume == 0:
        return 1.0
    return bars[-1].volume / average_volume


def _average_true_range_pct(bars: Sequence[PriceBar], period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    true_ranges: list[float] = []
    for previous, current in pairwise(bars):
        high = _as_float(current.high)
        low = _as_float(current.low)
        previous_close = _as_float(previous.close)
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    latest_close = _as_float(bars[-1].close)
    if latest_close == 0:
        return 0.0
    recent_ranges = true_ranges[-period:]
    return (sum(recent_ranges) / len(recent_ranges)) / latest_close


def _latest_gap_pct(bars: Sequence[PriceBar]) -> float:
    if len(bars) < 2:
        return 0.0
    previous_close = _as_float(bars[-2].close)
    if previous_close == 0:
        return 0.0
    return (_as_float(bars[-1].open) - previous_close) / previous_close


def _trend(
    latest_close: float,
    sma_20: float,
    sma_50: float,
    previous_sma_20: float,
    macd_histogram: float,
) -> str:
    rising_short_average = sma_20 >= previous_sma_20
    if latest_close > sma_20 > sma_50 and rising_short_average and macd_histogram >= 0:
        return "uptrend"
    if latest_close < sma_20 < sma_50 and not rising_short_average and macd_histogram <= 0:
        return "downtrend"
    return "range"


def _signal_from_trend(trend: str) -> AnalysisSignal:
    if trend == "uptrend":
        return AnalysisSignal.SUPPORTS
    if trend == "downtrend":
        return AnalysisSignal.CONFLICTS
    return AnalysisSignal.MIXED


def _support_resistance(
    bars: Sequence[PriceBar],
    window: int = 20,
) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
    recent = bars[-min(window, len(bars)) :]
    support = min(bar.low for bar in recent)
    resistance = max(bar.high for bar in recent)
    return (support,), (resistance,)


def _volume_summary(relative_volume: float) -> str:
    if relative_volume >= 1.25:
        return f"Latest volume is above-average at {relative_volume:.2f}x the recent baseline."
    if relative_volume <= 0.75:
        return f"Latest volume is below-average at {relative_volume:.2f}x the recent baseline."
    return f"Latest volume is near average at {relative_volume:.2f}x the recent baseline."


def _volatility_summary(atr_pct: float) -> str:
    if atr_pct >= 0.05:
        label = "elevated"
    elif atr_pct <= 0.02:
        label = "contained"
    else:
        label = "moderate"
    return f"ATR-style volatility is {label} at {atr_pct:.2%} of latest close."


def _gap_summary(gap_pct: float) -> str:
    if gap_pct >= 0.01:
        return f"Latest session opened with a gap up of {gap_pct:.2%}."
    if gap_pct <= -0.01:
        return f"Latest session opened with a gap down of {abs(gap_pct):.2%}."
    return f"Latest session opened without a material gap ({gap_pct:.2%})."


def _candlestick_summary(bar: PriceBar) -> str:
    open_price = _as_float(bar.open)
    close = _as_float(bar.close)
    high = _as_float(bar.high)
    low = _as_float(bar.low)
    full_range = max(high - low, 0.01)
    body_pct = abs(close - open_price) / full_range
    if close > open_price:
        direction = "bullish"
    elif close < open_price:
        direction = "bearish"
    else:
        direction = "neutral"
    size = "wide-range" if body_pct >= 0.55 else "small-body"
    return f"Latest candle is a {direction} {size} candle with body {body_pct:.0%} of range."


def _technical_metrics(
    *,
    as_of: date | datetime | None,
    sma_20: float,
    sma_50: float,
    rsi: float,
    macd_line: float,
    macd_signal: float,
    macd_histogram: float,
    relative_volume: float,
    atr_pct: float,
    gap_pct: float,
) -> tuple[MetricValue, ...]:
    return (
        analysis_metric("sma-20", round(sma_20, 4), as_of=as_of),
        analysis_metric("sma-50", round(sma_50, 4), as_of=as_of),
        analysis_metric("rsi-14", round(rsi, 4), as_of=as_of),
        analysis_metric("macd-line", round(macd_line, 4), as_of=as_of),
        analysis_metric("macd-signal", round(macd_signal, 4), as_of=as_of),
        analysis_metric("macd-histogram", round(macd_histogram, 4), as_of=as_of),
        analysis_metric("relative-volume", round(relative_volume, 4), unit="x", as_of=as_of),
        analysis_metric("average-true-range-pct", round(atr_pct, 6), unit="pct", as_of=as_of),
        analysis_metric("latest-gap-pct", round(gap_pct, 6), unit="pct", as_of=as_of),
    )


def _confidence(bar_count: int, relative_volume: float, trend: str) -> float:
    history_score = clamp(bar_count / 60.0)
    volume_score = clamp(relative_volume / 2.0)
    trend_score = 0.85 if trend != "range" else 0.55
    return round((history_score * 0.45) + (volume_score * 0.20) + (trend_score * 0.35), 4)
