"""Schema constants for evidence-grounded strategy extraction."""

from __future__ import annotations

STRATEGY_EXTRACTION_SCHEMA_VERSION = "strategy-extraction-v1"
STRATEGY_EXTRACTION_PROMPT_VERSION = "lane-c-discussed-strategies-v1"

STRATEGY_EXTRACTION_REQUIRED_FIELDS = frozenset(
    {
        "strategy_id",
        "ticker",
        "label",
        "direction",
        "instrument",
        "position_type",
        "time_horizon",
        "catalyst",
        "risk_or_hedge",
        "slang_terms",
        "evidence",
        "confidence",
        "sarcasm_joke_risk",
    }
)

__all__ = [
    "STRATEGY_EXTRACTION_PROMPT_VERSION",
    "STRATEGY_EXTRACTION_REQUIRED_FIELDS",
    "STRATEGY_EXTRACTION_SCHEMA_VERSION",
]
