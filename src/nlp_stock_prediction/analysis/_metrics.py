"""Shared numeric metric helpers for deterministic analysis modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from nlp_stock_prediction.contracts import MetricValue, ProviderMetric


def normalize_metric_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def provider_metric_map(metrics: Iterable[ProviderMetric]) -> dict[str, ProviderMetric]:
    return {normalize_metric_name(metric.name): metric for metric in metrics}


def metric_decimal(metrics: Mapping[str, ProviderMetric], *names: str) -> Decimal | None:
    for name in names:
        metric = metrics.get(normalize_metric_name(name))
        if metric is None:
            continue
        value = decimal_from_value(metric.value)
        if value is not None:
            return value
    return None


def metric_text(metrics: Mapping[str, ProviderMetric], *names: str) -> str | None:
    for name in names:
        metric = metrics.get(normalize_metric_name(name))
        if metric is not None and metric.value is not None:
            text = str(metric.value).strip()
            if text:
                return text
    return None


def decimal_from_value(value: Decimal | float | int | str | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value)) if value == value else None
    text = value.strip().replace(",", "")
    is_percentage = text.endswith("%")
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return None
    return parsed / Decimal("100") if is_percentage else parsed


def metric_values_from_provider(metrics: Sequence[ProviderMetric]) -> tuple[MetricValue, ...]:
    return tuple(
        MetricValue(
            name=metric.name,
            value=metric.value,
            unit=metric.unit,
            as_of=metric.as_of,
            metadata=metric.metadata,
        )
        for metric in metrics
    )


def analysis_metric(
    name: str,
    value: Decimal | float | int | str | None,
    *,
    unit: str | None = None,
    as_of: date | datetime | None = None,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> MetricValue:
    return MetricValue(
        name=name,
        value=value,
        unit=unit,
        as_of=as_of,
        metadata={} if metadata is None else metadata,
    )


def median(values: Sequence[Decimal]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal("2")


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(max(value, minimum), maximum)
