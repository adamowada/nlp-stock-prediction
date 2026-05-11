"""Cluster near-duplicate observed strategy extractions."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime

from nlp_stock_prediction.contracts import (
    EvidenceReference,
    JsonObject,
    ProviderWarning,
    StrategyCluster,
    StrategyExtraction,
    WarningCode,
    WarningSeverity,
)

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
_DEFAULT_WARNING_TIME = datetime(1970, 1, 1, tzinfo=UTC)
_HIGH_SARCASM_JOKE_RISK_THRESHOLD = 0.75


def cluster_strategies(
    strategies: Iterable[StrategyExtraction],
    *,
    occurred_at: datetime | None = None,
    warning_provider_name: str = "strategy-clustering",
) -> tuple[StrategyCluster, ...]:
    """Group near-duplicate strategies by ticker, direction, instrument, horizon, and catalyst."""

    strategy_tuple = tuple(strategies)
    grouped: dict[tuple[str, str, str, str, str], list[StrategyExtraction]] = defaultdict(list)
    for strategy in strategy_tuple:
        catalyst = _normalize_catalyst(strategy.catalyst)
        key = (
            strategy.ticker,
            strategy.direction.value,
            strategy.instrument.value,
            strategy.time_horizon.value,
            catalyst or "",
        )
        grouped[key].append(strategy)

    conflicting_directions = _conflicting_source_directions(strategy_tuple)
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
                warnings=(
                    *tuple(warning for member in members for warning in member.warnings),
                    *_cluster_warnings(
                        members=members,
                        conflict_key=(ticker, instrument, time_horizon, catalyst),
                        conflicting_directions=conflicting_directions,
                        occurred_at=occurred_at or _DEFAULT_WARNING_TIME,
                        provider_name=warning_provider_name,
                    ),
                ),
            )
        )

    return tuple(clusters)


def _conflicting_source_directions(
    strategies: tuple[StrategyExtraction, ...],
) -> dict[tuple[str, str, str, str], tuple[str, ...]]:
    directions_by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for strategy in strategies:
        key = (
            strategy.ticker,
            strategy.instrument.value,
            strategy.time_horizon.value,
            _normalize_catalyst(strategy.catalyst) or "",
        )
        directions_by_key[key].add(strategy.direction.value)
    return {
        key: tuple(sorted(directions))
        for key, directions in directions_by_key.items()
        if {"bullish", "bearish"}.issubset(directions)
    }


def _cluster_warnings(
    *,
    members: list[StrategyExtraction],
    conflict_key: tuple[str, str, str, str],
    conflicting_directions: dict[tuple[str, str, str, str], tuple[str, ...]],
    occurred_at: datetime,
    provider_name: str,
) -> tuple[ProviderWarning, ...]:
    warnings: list[ProviderWarning] = []
    max_sarcasm_joke_risk = max(member.sarcasm_joke_risk for member in members)
    if max_sarcasm_joke_risk >= _HIGH_SARCASM_JOKE_RISK_THRESHOLD:
        warnings.append(
            _warning(
                code=WarningCode.UNSUPPORTED_CLAIM,
                message=(
                    "High sarcasm/joke risk in cited discussion; treat this observed strategy "
                    "as watch-only until corroborated."
                ),
                occurred_at=occurred_at,
                provider_name=provider_name,
                metadata={
                    "risk_type": "high_sarcasm_joke_risk",
                    "risk": round(max_sarcasm_joke_risk, 4),
                    "threshold": _HIGH_SARCASM_JOKE_RISK_THRESHOLD,
                    "member_strategy_ids": [member.strategy_id for member in members],
                },
            )
        )
    directions = conflicting_directions.get(conflict_key)
    if directions is not None:
        warnings.append(
            _warning(
                code=WarningCode.PARTIAL_DATA,
                message=(
                    "Conflicting source evidence produced opposing directions for the same "
                    "ticker, instrument, horizon, and catalyst."
                ),
                occurred_at=occurred_at,
                provider_name=provider_name,
                metadata={
                    "risk_type": "conflicting_source_evidence",
                    "directions": list(directions),
                    "member_strategy_ids": [member.strategy_id for member in members],
                },
            )
        )
    return tuple(warnings)


def _warning(
    *,
    code: WarningCode,
    message: str,
    occurred_at: datetime,
    provider_name: str,
    metadata: JsonObject,
) -> ProviderWarning:
    return ProviderWarning(
        code=code,
        severity=WarningSeverity.WARNING,
        message=message,
        provider_name=provider_name,
        occurred_at=occurred_at,
        metadata=metadata,
    )


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
