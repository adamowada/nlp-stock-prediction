"""Shared enum taxonomy for public contracts."""

from __future__ import annotations

from enum import StrEnum


class RetrievalMethod(StrEnum):
    OFFICIAL_API = "official_api"
    PUBLIC_SCRAPE = "public_scrape"
    FIXTURE = "fixture"
    LLM = "llm"
    DERIVED = "derived"


class SourceKind(StrEnum):
    REDDIT_POST = "reddit_post"
    REDDIT_COMMENT = "reddit_comment"
    REDDIT_TICKER_CARD = "reddit_ticker_card"
    X_POST = "x_post"
    NEWS_ARTICLE = "news_article"
    MARKET_DATA = "market_data"
    FUNDAMENTAL_DATA = "fundamental_data"
    SEC_FILING = "sec_filing"
    MACRO_SERIES = "macro_series"
    LLM_EXTRACTION = "llm_extraction"
    INTERNAL_ANALYSIS = "internal_analysis"


class FreshnessStatus(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"
    UNKNOWN = "unknown"


class ProviderStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    PARTIAL = "partial"
    STALE = "stale"
    UNCONFIGURED = "unconfigured"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"
    MALFORMED = "malformed"


class CredentialState(StrEnum):
    CONFIGURED = "configured"
    MISSING = "missing"
    INVALID = "invalid"
    NOT_REQUIRED = "not_required"


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class WarningCode(StrEnum):
    MISSING_CREDENTIALS = "missing_credentials"
    AUTH_FAILED = "auth_failed"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT = "timeout"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    MALFORMED_RESPONSE = "malformed_response"
    SCHEMA_MISMATCH = "schema_mismatch"
    SCRAPING_DRIFT = "scraping_drift"
    STALE_DATA = "stale_data"
    NO_DATA = "no_data"
    PARTIAL_DATA = "partial_data"
    LLM_SCHEMA_INVALID = "llm_schema_invalid"
    LLM_EVIDENCE_MISMATCH = "llm_evidence_mismatch"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class TickerDiscoveryStatus(StrEnum):
    VALID = "valid"
    TOO_FEW_UNIQUE = "too_few_unique"
    TOO_MANY_UNIQUE = "too_many_unique"
    DUPLICATES_REMOVED = "duplicates_removed"
    MALFORMED_SOURCE = "malformed_source"


class Direction(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class InstrumentType(StrEnum):
    SHARES = "shares"
    CALL_OPTION = "call_option"
    PUT_OPTION = "put_option"
    OPTION_SPREAD = "option_spread"
    CASH_SECURED_PUT = "cash_secured_put"
    COVERED_CALL = "covered_call"
    UNKNOWN = "unknown"


class PositionType(StrEnum):
    LONG = "long"
    SHORT = "short"
    DEFINED_RISK = "defined_risk"
    INCOME = "income"
    WATCH_ONLY = "watch_only"


class TimeHorizon(StrEnum):
    INTRADAY = "intraday"
    SWING = "swing"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    EARNINGS_EVENT = "earnings_event"
    UNKNOWN = "unknown"


class RecommendationAction(StrEnum):
    QUALIFIED = "qualified"
    WATCH = "watch"
    AVOID = "avoid"
    NO_TRADE = "no_trade"


class RiskProfile(StrEnum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"


class AnalysisSignal(StrEnum):
    SUPPORTS = "supports"
    CONFLICTS = "conflicts"
    MIXED = "mixed"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


__all__ = [
    "AnalysisSignal",
    "CredentialState",
    "Direction",
    "FreshnessStatus",
    "InstrumentType",
    "PositionType",
    "ProviderStatus",
    "RecommendationAction",
    "RetrievalMethod",
    "RiskProfile",
    "SourceKind",
    "TickerDiscoveryStatus",
    "TimeHorizon",
    "WarningCode",
    "WarningSeverity",
]
