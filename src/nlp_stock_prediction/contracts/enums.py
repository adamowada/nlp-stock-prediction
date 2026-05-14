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


class AssetClass(StrEnum):
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "crypto"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    FUTURES = "futures"
    FUND = "fund"
    INDEX = "index"
    PROXY = "proxy"
    UNKNOWN = "unknown"


class InstrumentResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"


class TradabilityStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


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


class PredictionStatus(StrEnum):
    EVIDENCE_SUPPORTED = "evidence_supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNAVAILABLE = "unavailable"


class PredictionType(StrEnum):
    DIRECTIONAL = "directional"
    VOLATILITY = "volatility"
    EVENT = "event"
    RELATIVE_VALUE = "relative_value"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class SignalArtifactFamily(StrEnum):
    TECHNICALS = "technicals"
    TIMESFM = "timesfm"
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    SECTOR_MACRO = "sector_macro"


class PredictionOutcomeStatus(StrEnum):
    OBSERVED = "observed"
    PENDING = "pending"
    UNAVAILABLE = "unavailable"
    STALE = "stale"


class PredictionOutcomeResult(StrEnum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    NEUTRAL = "neutral"
    INSUFFICIENT_DATA = "insufficient_data"


class PredictionOutcomeEvaluationStatus(StrEnum):
    CONFIRMED = "confirmed"
    MISSED = "missed"
    MIXED = "mixed"
    INCONCLUSIVE = "inconclusive"
    PENDING = "pending"
    STALE = "stale"
    NOT_EVALUABLE = "not_evaluable"


class AnalysisSignal(StrEnum):
    SUPPORTS = "supports"
    CONFLICTS = "conflicts"
    MIXED = "mixed"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


__all__ = [
    "AnalysisSignal",
    "AssetClass",
    "CredentialState",
    "Direction",
    "FreshnessStatus",
    "InstrumentResolutionStatus",
    "InstrumentType",
    "PositionType",
    "PredictionOutcomeEvaluationStatus",
    "PredictionOutcomeResult",
    "PredictionOutcomeStatus",
    "PredictionStatus",
    "PredictionType",
    "ProviderStatus",
    "RetrievalMethod",
    "SignalArtifactFamily",
    "SourceKind",
    "TickerDiscoveryStatus",
    "TimeHorizon",
    "TradabilityStatus",
    "WarningCode",
    "WarningSeverity",
]
