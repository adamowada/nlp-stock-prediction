"""Sector and peer-relative context synthesis."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal

from nlp_stock_prediction.analysis._metrics import (
    analysis_metric,
    median,
    metric_decimal,
    provider_metric_map,
)
from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    FundamentalsSnapshot,
    MetricValue,
    ProviderMetric,
    SectorContext,
)


def analyze_sector_context(
    snapshot: FundamentalsSnapshot,
    *,
    peers: Sequence[FundamentalsSnapshot] = (),
    sector: str | None = None,
    benchmark_symbol: str | None = None,
    benchmark_metrics: Sequence[ProviderMetric] = (),
    as_of: date | datetime | None = None,
) -> SectorContext:
    """Compare a ticker to peer fundamentals or a sector ETF proxy."""

    if peers:
        return _peer_context(snapshot, peers=peers, sector=sector, as_of=as_of)
    return _proxy_context(
        snapshot,
        sector=sector,
        benchmark_symbol=benchmark_symbol,
        benchmark_metrics=benchmark_metrics,
        as_of=as_of,
    )


def _peer_context(
    snapshot: FundamentalsSnapshot,
    *,
    peers: Sequence[FundamentalsSnapshot],
    sector: str | None,
    as_of: date | datetime | None,
) -> SectorContext:
    target_metrics = provider_metric_map(snapshot.metrics)
    peer_maps = [provider_metric_map(peer.metrics) for peer in peers]
    target_pe = metric_decimal(target_metrics, "pe_ratio", "forward_pe", "trailing_pe")
    target_margin = metric_decimal(target_metrics, "net_margin", "profit_margin")
    target_growth = metric_decimal(target_metrics, "revenue_growth_yoy", "revenue_growth")
    peer_pe = median(_peer_values(peer_maps, "pe_ratio", "forward_pe", "trailing_pe"))
    peer_margin = median(_peer_values(peer_maps, "net_margin", "profit_margin"))
    peer_growth = median(_peer_values(peer_maps, "revenue_growth_yoy", "revenue_growth"))
    support_count = 0
    conflict_count = 0
    phrases: list[str] = []
    derived_metrics: list[MetricValue] = []

    if target_pe is not None and peer_pe is not None:
        derived_metrics.append(analysis_metric("peer-median-pe-ratio", peer_pe, as_of=as_of))
        if target_pe <= peer_pe * Decimal("1.15"):
            support_count += 1
            phrases.append(f"valuation is near peer median P/E {peer_pe:g}")
        else:
            conflict_count += 1
            phrases.append(f"valuation is premium to peer median P/E {peer_pe:g}")
    if target_margin is not None and peer_margin is not None:
        derived_metrics.append(analysis_metric("peer-median-net-margin", peer_margin, as_of=as_of))
        if target_margin >= peer_margin:
            support_count += 1
            phrases.append(f"profitability leads peers at {target_margin:.1%}")
        else:
            conflict_count += 1
            phrases.append(f"profitability trails peers at {target_margin:.1%}")
    if target_growth is not None and peer_growth is not None:
        derived_metrics.append(
            analysis_metric("peer-median-revenue-growth-yoy", peer_growth, as_of=as_of)
        )
        if target_growth >= peer_growth:
            support_count += 1
            phrases.append(f"growth leads peers at {target_growth:.1%}")
        else:
            conflict_count += 1
            phrases.append(f"growth trails peers at {target_growth:.1%}")

    signal = _signal(support_count, conflict_count)
    confidence = _confidence(support_count, conflict_count, len(peers))

    return SectorContext(
        ticker=snapshot.ticker,
        summary="Sector peer context: " + "; ".join(phrases or ["peer metrics are sparse"]) + ".",
        signal=signal,
        confidence=confidence,
        metrics=tuple(derived_metrics),
        sector=sector,
        peers=tuple(peer.ticker for peer in peers),
    )


def _proxy_context(
    snapshot: FundamentalsSnapshot,
    *,
    sector: str | None,
    benchmark_symbol: str | None,
    benchmark_metrics: Sequence[ProviderMetric],
    as_of: date | datetime | None,
) -> SectorContext:
    metrics = provider_metric_map(benchmark_metrics)
    relative_return = metric_decimal(
        metrics,
        "sector_etf_return_20d",
        "benchmark_return_20d",
        "relative_strength_20d",
    )
    assumptions = (
        ("ETF proxy used because peer fundamentals were unavailable.",)
        if benchmark_symbol
        else ("Sector context is limited because neither peers nor ETF proxy data were available.",)
    )
    if relative_return is None:
        signal = AnalysisSignal.NEUTRAL if benchmark_symbol else AnalysisSignal.UNKNOWN
        summary = "Sector ETF proxy data is unavailable."
        confidence = 0.25
        context_metrics: tuple[MetricValue, ...] = ()
    else:
        signal = AnalysisSignal.SUPPORTS if relative_return > 0 else AnalysisSignal.CONFLICTS
        summary = f"Sector ETF proxy return is {relative_return:.1%} over the comparison window."
        confidence = 0.55
        context_metrics = (
            analysis_metric("sector-etf-return-20d", relative_return, unit="pct", as_of=as_of),
        )

    return SectorContext(
        ticker=snapshot.ticker,
        summary=summary,
        signal=signal,
        confidence=confidence,
        metrics=context_metrics,
        sector=sector,
        benchmark_symbol=benchmark_symbol,
        assumptions=assumptions,
    )


def _peer_values(
    peer_maps: Sequence[dict[str, ProviderMetric]],
    *names: str,
) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for peer_metrics in peer_maps:
        value = metric_decimal(peer_metrics, *names)
        if value is not None:
            values.append(value)
    return tuple(values)


def _signal(support_count: int, conflict_count: int) -> AnalysisSignal:
    if support_count > conflict_count:
        return AnalysisSignal.SUPPORTS
    if conflict_count > support_count:
        return AnalysisSignal.CONFLICTS
    if support_count or conflict_count:
        return AnalysisSignal.MIXED
    return AnalysisSignal.UNKNOWN


def _confidence(support_count: int, conflict_count: int, peer_count: int) -> float:
    compared_metrics = support_count + conflict_count
    return round(min(0.85, 0.25 + compared_metrics * 0.15 + min(peer_count, 4) * 0.05), 4)
