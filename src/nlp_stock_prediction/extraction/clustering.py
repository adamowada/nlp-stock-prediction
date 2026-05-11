"""Cluster near-duplicate observed strategy extractions."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from nlp_stock_prediction.contracts import EvidenceReference, StrategyCluster, StrategyExtraction

_GENERIC_CATALYST_WORDS = {
    "call",
    "calls",
    "into",
    "next",
    "option",
    "options",
    "report",
    "the",
    "weekly",
}


def cluster_strategies(strategies: Iterable[StrategyExtraction]) -> tuple[StrategyCluster, ...]:
    """Group near-duplicate strategies by ticker, direction, instrument, horizon, and catalyst."""

    grouped: dict[tuple[str, str, str, str, str], list[StrategyExtraction]] = defaultdict(list)
    for strategy in strategies:
        catalyst = _normalize_catalyst(strategy.catalyst)
        key = (
            strategy.ticker,
            strategy.direction.value,
            strategy.instrument.value,
            strategy.time_horizon.value,
            catalyst or "",
        )
        grouped[key].append(strategy)

    clusters: list[StrategyCluster] = []
    for key, members in grouped.items():
        ticker, direction, instrument, time_horizon, catalyst = key
        evidence = _dedupe_evidence(
            reference for member in members for reference in member.evidence
        )
        confidence = sum(member.confidence for member in members) / len(members)
        clusters.append(
            StrategyCluster(
                cluster_id=_cluster_id(
                    ticker=ticker,
                    direction=direction,
                    instrument=instrument,
                    time_horizon=time_horizon,
                    catalyst=catalyst,
                ),
                ticker=ticker,
                direction=members[0].direction,
                instrument=members[0].instrument,
                time_horizon=members[0].time_horizon,
                catalyst_summary=catalyst or None,
                member_strategy_ids=tuple(member.strategy_id for member in members),
                evidence=evidence,
                confidence=confidence,
                warnings=tuple(warning for member in members for warning in member.warnings),
            )
        )

    return tuple(clusters)


def _normalize_catalyst(catalyst: str | None) -> str | None:
    if catalyst is None or not catalyst.strip():
        return None

    normalized = catalyst.strip().lower()
    if normalized in {"er", "earnings", "earnings report", "earnings call"}:
        return "earnings"
    if "earnings" in normalized:
        return "earnings"
    if "cpi" in normalized:
        return "cpi"
    if "fomc" in normalized or "fed" in normalized:
        return "fed"

    tokens = []
    for token in re.sub(r"[^a-z0-9]+", " ", normalized).split():
        if token not in _GENERIC_CATALYST_WORDS:
            tokens.append(token)
    if not tokens:
        return None
    return " ".join(tokens)


def _cluster_id(
    *,
    ticker: str,
    direction: str,
    instrument: str,
    time_horizon: str,
    catalyst: str,
) -> str:
    raw = "-".join((ticker.lower(), direction, instrument, time_horizon, catalyst or "general"))
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return f"cluster-{slug}"


def _dedupe_evidence(references: Iterable[EvidenceReference]) -> tuple[EvidenceReference, ...]:
    seen: set[tuple[str, str | None, int | None, int | None]] = set()
    deduped: list[EvidenceReference] = []
    for reference in references:
        key = (reference.evidence_id, reference.quote, reference.start_char, reference.end_char)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(reference)
    return tuple(deduped)


__all__ = ["cluster_strategies"]
