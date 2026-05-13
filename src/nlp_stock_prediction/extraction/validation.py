"""Validate schema-shaped LLM strategy output against normalized evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    Direction,
    EvidenceReference,
    InstrumentType,
    ProviderWarning,
    SourceEvidence,
    StrategyExtraction,
    WarningCode,
    WarningSeverity,
)

_EVIDENCE_REFERENCE_FIELDS = {"evidence_id", "quote", "start_char", "end_char", "relevance"}
_RECOMMENDATION_FIELDS = {
    "action",
    "candidate_id",
    "entry_logic",
    "invalidation_criteria",
    "recommendation_action",
    "risk_plan",
    "score",
    "score_input_ids",
    "thesis",
}
_RECOMMENDATION_PHRASES = (
    "app recommends",
    "our recommendation",
    "recommend buying",
    "recommend selling",
    "we recommend",
    "qualified trade",
    "trade candidate",
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "around",
    "as",
    "at",
    "because",
    "before",
    "but",
    "for",
    "from",
    "if",
    "in",
    "into",
    "near",
    "next",
    "of",
    "on",
    "or",
    "print",
    "report",
    "term",
    "the",
    "to",
    "tomorrow",
    "with",
}
_INSTRUMENT_KEYWORDS = {
    InstrumentType.CALL_OPTION: ("call", "calls", "call option", "weeklies", "weekly"),
    InstrumentType.PUT_OPTION: ("put", "puts", "put option"),
    InstrumentType.OPTION_SPREAD: ("spread", "spreads", "debit spread", "credit spread"),
    InstrumentType.CASH_SECURED_PUT: ("cash-secured put", "cash secured put", "csp"),
    InstrumentType.COVERED_CALL: ("covered call", "covered calls"),
    InstrumentType.SHARES: ("shares", "stock", "common", "long", "short", "buy", "sell"),
    InstrumentType.UNKNOWN: (),
}
_DIRECTION_KEYWORDS = {
    Direction.BULLISH: (
        "bull",
        "bullish",
        "buy",
        "call",
        "calls",
        "long",
        "moon",
        "rip",
        "squeeze",
        "upside",
    ),
    Direction.BEARISH: ("bear", "bearish", "downside", "dump", "put", "puts", "short"),
    Direction.NEUTRAL: (),
    Direction.MIXED: (),
    Direction.UNKNOWN: (),
}


@dataclass(frozen=True)
class ExtractionValidationResult:
    """Accepted strategies plus warnings for rejected LLM records."""

    strategies: tuple[StrategyExtraction, ...]
    warnings: tuple[ProviderWarning, ...]
    rejected_count: int


@dataclass(frozen=True)
class _RejectedStrategy(Exception):
    code: WarningCode
    message: str
    metadata: Mapping[str, object]


def parse_llm_json_response(raw_response: str) -> tuple[dict[str, object], ...]:
    """Parse a JSON response that is either a strategy list or a strategy envelope."""

    loaded = json.loads(raw_response)
    if isinstance(loaded, list):
        raw_strategies = loaded
    elif isinstance(loaded, dict) and "strategies" in loaded:
        raw_strategies = loaded["strategies"]
    else:
        raise ValueError("LLM response must be a list or an object with a strategies list")

    if not isinstance(raw_strategies, list):
        raise ValueError("LLM strategies field must be a list")

    parsed: list[dict[str, object]] = []
    for index, item in enumerate(raw_strategies):
        if not isinstance(item, dict):
            raise ValueError(f"LLM strategy at index {index} must be an object")
        parsed.append(dict(item))
    return tuple(parsed)


def validate_llm_strategy_payloads(
    payloads: Iterable[Mapping[str, object]],
    *,
    evidence: Iterable[SourceEvidence],
    provider_name: str,
    occurred_at: datetime,
) -> ExtractionValidationResult:
    """Validate schema-shaped LLM output and reject unsupported extracted claims.

    The accepted output describes observed discussion only. Anything that looks like the app's own
    recommendation, omits evidence, cites unknown evidence, or quotes text not present in the
    normalized evidence body is rejected with a provider warning.
    """

    evidence_by_id = {record.evidence_id: record for record in evidence}
    strategies: list[StrategyExtraction] = []
    warnings: list[ProviderWarning] = []

    for index, payload in enumerate(payloads):
        strategy_id = _payload_identifier(payload, index)
        try:
            normalized_payload, cited_records = _normalize_payload(
                payload,
                evidence_by_id=evidence_by_id,
                index=index,
            )
            strategy = StrategyExtraction.model_validate(normalized_payload)
            _validate_supported_claims(strategy, cited_records, index=index)
        except _RejectedStrategy as error:
            warnings.append(
                _warning(
                    code=error.code,
                    message=error.message,
                    provider_name=provider_name,
                    occurred_at=occurred_at,
                    metadata={
                        "strategy_index": index,
                        "strategy_id": strategy_id,
                        **_json_safe_metadata(error.metadata),
                    },
                )
            )
            continue
        except ValidationError as error:
            warnings.append(
                _warning(
                    code=WarningCode.LLM_SCHEMA_INVALID,
                    message=f"LLM strategy payload failed schema validation: {error}",
                    provider_name=provider_name,
                    occurred_at=occurred_at,
                    metadata={"strategy_index": index, "strategy_id": strategy_id},
                )
            )
            continue

        strategies.append(strategy)

    return ExtractionValidationResult(
        strategies=tuple(strategies),
        warnings=tuple(warnings),
        rejected_count=len(warnings),
    )


def _payload_identifier(payload: Mapping[str, object], index: int) -> str:
    strategy_id = payload.get("strategy_id")
    if isinstance(strategy_id, str) and strategy_id.strip():
        return strategy_id.strip()
    return f"llm-strategy-index-{index}"


def _normalize_payload(
    payload: Mapping[str, object],
    *,
    evidence_by_id: Mapping[str, SourceEvidence],
    index: int,
) -> tuple[dict[str, object], tuple[SourceEvidence, ...]]:
    recommendation_fields = sorted(set(payload).intersection(_RECOMMENDATION_FIELDS))
    if recommendation_fields:
        raise _RejectedStrategy(
            code=WarningCode.UNSUPPORTED_CLAIM,
            message="LLM output described an app recommendation instead of observed discussion.",
            metadata={"recommendation_fields": recommendation_fields},
        )

    label = payload.get("label")
    if isinstance(label, str) and _contains_recommendation_phrase(label):
        raise _RejectedStrategy(
            code=WarningCode.UNSUPPORTED_CLAIM,
            message="LLM strategy label used recommendation language.",
            metadata={"label": label},
        )

    ticker = _payload_ticker(payload, index)
    references, cited_records = _normalize_evidence_references(
        payload.get("evidence"),
        evidence_by_id=evidence_by_id,
        ticker=ticker,
        index=index,
    )

    normalized = dict(payload)
    normalized["ticker"] = ticker
    normalized["evidence"] = references
    return normalized, cited_records


def _payload_ticker(payload: Mapping[str, object], index: int) -> str:
    raw_ticker = payload.get("ticker")
    if not isinstance(raw_ticker, str) or not raw_ticker.strip():
        raise _RejectedStrategy(
            code=WarningCode.LLM_SCHEMA_INVALID,
            message="LLM strategy payload is missing a ticker.",
            metadata={"strategy_index": index},
        )
    return raw_ticker.strip().upper()


def _normalize_evidence_references(
    raw_references: object,
    *,
    evidence_by_id: Mapping[str, SourceEvidence],
    ticker: str,
    index: int,
) -> tuple[tuple[EvidenceReference, ...], tuple[SourceEvidence, ...]]:
    if not isinstance(raw_references, Sequence) or isinstance(raw_references, str | bytes):
        raise _RejectedStrategy(
            code=WarningCode.LLM_EVIDENCE_MISMATCH,
            message="LLM strategy payload must include one or more evidence references.",
            metadata={"strategy_index": index},
        )
    if not raw_references:
        raise _RejectedStrategy(
            code=WarningCode.LLM_EVIDENCE_MISMATCH,
            message="LLM strategy payload did not cite any evidence.",
            metadata={"strategy_index": index},
        )

    references: list[EvidenceReference] = []
    cited_records: list[SourceEvidence] = []
    for reference_index, raw_reference in enumerate(raw_references):
        if not isinstance(raw_reference, Mapping):
            raise _RejectedStrategy(
                code=WarningCode.LLM_SCHEMA_INVALID,
                message="LLM evidence reference must be an object.",
                metadata={"evidence_reference_index": reference_index},
            )
        unknown_fields = sorted(set(raw_reference).difference(_EVIDENCE_REFERENCE_FIELDS))
        if unknown_fields:
            raise _RejectedStrategy(
                code=WarningCode.LLM_SCHEMA_INVALID,
                message="LLM evidence reference included fields outside the frozen contract.",
                metadata={"unknown_fields": unknown_fields},
            )
        evidence_id = _reference_evidence_id(raw_reference, reference_index)
        record = evidence_by_id.get(evidence_id)
        if record is None:
            raise _RejectedStrategy(
                code=WarningCode.LLM_EVIDENCE_MISMATCH,
                message="LLM strategy cited an unknown evidence_id.",
                metadata={"evidence_id": evidence_id},
            )
        if not _evidence_mentions_ticker(record, ticker):
            raise _RejectedStrategy(
                code=WarningCode.LLM_EVIDENCE_MISMATCH,
                message="LLM strategy cited evidence that does not mention the strategy ticker.",
                metadata={"evidence_id": evidence_id, "ticker": ticker},
            )

        quote = _reference_quote(raw_reference, reference_index)
        start_char, end_char, normalized_quote = _locate_quote(
            body=record.text,
            quote=quote,
            raw_start=raw_reference.get("start_char"),
            raw_end=raw_reference.get("end_char"),
            evidence_id=evidence_id,
        )
        try:
            reference = EvidenceReference(
                evidence_id=evidence_id,
                quote=normalized_quote,
                start_char=start_char,
                end_char=end_char,
                relevance=raw_reference.get("relevance"),
            )
        except ValidationError as error:
            raise _RejectedStrategy(
                code=WarningCode.LLM_SCHEMA_INVALID,
                message=f"LLM evidence reference failed schema validation: {error}",
                metadata={"evidence_id": evidence_id},
            ) from error

        references.append(reference)
        cited_records.append(record)

    return tuple(references), tuple(cited_records)


def _reference_evidence_id(raw_reference: Mapping[str, object], reference_index: int) -> str:
    raw_evidence_id = raw_reference.get("evidence_id")
    if not isinstance(raw_evidence_id, str) or not raw_evidence_id.strip():
        raise _RejectedStrategy(
            code=WarningCode.LLM_EVIDENCE_MISMATCH,
            message="LLM evidence reference is missing evidence_id.",
            metadata={"evidence_reference_index": reference_index},
        )
    return raw_evidence_id.strip()


def _reference_quote(raw_reference: Mapping[str, object], reference_index: int) -> str:
    raw_quote = raw_reference.get("quote")
    if not isinstance(raw_quote, str) or not raw_quote.strip():
        raise _RejectedStrategy(
            code=WarningCode.LLM_EVIDENCE_MISMATCH,
            message="LLM evidence reference is missing a quote.",
            metadata={"evidence_reference_index": reference_index},
        )
    return raw_quote.strip()


def _locate_quote(
    *,
    body: str,
    quote: str,
    raw_start: object,
    raw_end: object,
    evidence_id: str,
) -> tuple[int, int, str]:
    if raw_start is not None or raw_end is not None:
        if not isinstance(raw_start, int) or not isinstance(raw_end, int):
            raise _RejectedStrategy(
                code=WarningCode.LLM_EVIDENCE_MISMATCH,
                message="LLM evidence quote offsets must both be integers when provided.",
                metadata={"evidence_id": evidence_id},
            )
        if raw_start < 0 or raw_end < raw_start or raw_end > len(body):
            raise _RejectedStrategy(
                code=WarningCode.LLM_EVIDENCE_MISMATCH,
                message="LLM evidence quote offsets are outside the evidence body.",
                metadata={"evidence_id": evidence_id},
            )
        if body[raw_start:raw_end] != quote:
            raise _RejectedStrategy(
                code=WarningCode.LLM_EVIDENCE_MISMATCH,
                message="LLM evidence quote offsets do not match the evidence body.",
                metadata={"evidence_id": evidence_id},
            )
        return raw_start, raw_end, quote

    start_char = body.find(quote)
    if start_char < 0:
        start_char = body.lower().find(quote.lower())
    if start_char < 0:
        raise _RejectedStrategy(
            code=WarningCode.LLM_EVIDENCE_MISMATCH,
            message="LLM evidence quote was not found in the normalized evidence body.",
            metadata={"evidence_id": evidence_id, "quote": quote},
        )

    end_char = start_char + len(quote)
    return start_char, end_char, body[start_char:end_char]


def _evidence_mentions_ticker(record: SourceEvidence, ticker: str) -> bool:
    tickers = {matched.upper() for matched in record.matched_tickers}
    if record.ticker is not None:
        tickers.add(record.ticker.upper())
    return not tickers or ticker.upper() in tickers


def _validate_supported_claims(
    strategy: StrategyExtraction,
    cited_records: tuple[SourceEvidence, ...],
    *,
    index: int,
) -> None:
    support_text = _support_text(strategy, cited_records)

    instrument_keywords = _INSTRUMENT_KEYWORDS[strategy.instrument]
    if instrument_keywords and not _contains_any(support_text, instrument_keywords):
        raise _RejectedStrategy(
            code=WarningCode.UNSUPPORTED_CLAIM,
            message="LLM strategy instrument was not supported by cited evidence.",
            metadata={"strategy_index": index, "instrument": strategy.instrument.value},
        )

    direction_keywords = _DIRECTION_KEYWORDS[strategy.direction]
    if direction_keywords and not _contains_any(support_text, direction_keywords):
        raise _RejectedStrategy(
            code=WarningCode.UNSUPPORTED_CLAIM,
            message="LLM strategy direction was not supported by cited evidence.",
            metadata={"strategy_index": index, "direction": strategy.direction.value},
        )

    if strategy.catalyst:
        catalyst_tokens = _meaningful_tokens(strategy.catalyst)
        if catalyst_tokens and not any(token in support_text for token in catalyst_tokens):
            raise _RejectedStrategy(
                code=WarningCode.UNSUPPORTED_CLAIM,
                message="LLM strategy catalyst was not supported by cited evidence.",
                metadata={"strategy_index": index, "catalyst": strategy.catalyst},
            )


def _support_text(strategy: StrategyExtraction, cited_records: tuple[SourceEvidence, ...]) -> str:
    parts = [record.text for record in cited_records]
    parts.extend(reference.quote or "" for reference in strategy.evidence)
    return " ".join(parts).lower()


def _meaningful_tokens(value: str) -> tuple[str, ...]:
    tokens = []
    for raw_token in value.lower().replace("/", " ").replace("-", " ").split():
        token = "".join(character for character in raw_token if character.isalnum())
        if len(token) >= 2 and token not in _STOPWORDS:
            tokens.append(token)
    return tuple(tokens)


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _contains_recommendation_phrase(value: str) -> bool:
    lowered = value.lower()
    return any(phrase in lowered for phrase in _RECOMMENDATION_PHRASES)


def _warning(
    *,
    code: WarningCode,
    message: str,
    provider_name: str,
    occurred_at: datetime,
    metadata: Mapping[str, object],
) -> ProviderWarning:
    severity = (
        WarningSeverity.ERROR
        if code in {WarningCode.LLM_SCHEMA_INVALID, WarningCode.LLM_EVIDENCE_MISMATCH}
        else WarningSeverity.WARNING
    )
    return ProviderWarning(
        code=code,
        severity=severity,
        message=message,
        provider_name=provider_name,
        occurred_at=occurred_at,
        metadata=_json_safe_metadata(metadata),
    )


def _json_safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool) or value is None:
            safe[key] = value
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            safe[key] = [str(item) for item in value]
        else:
            safe[key] = str(value)
    return safe


__all__ = [
    "ExtractionValidationResult",
    "parse_llm_json_response",
    "validate_llm_strategy_payloads",
]
