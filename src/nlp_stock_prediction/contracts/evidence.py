"""Normalized evidence contracts."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from nlp_stock_prediction.contracts.base import (
    AwareDatetime,
    ContractModel,
    JsonObject,
    NonEmptyStr,
    TickerSymbol,
)
from nlp_stock_prediction.contracts.enums import SourceKind
from nlp_stock_prediction.contracts.provenance import SourceProvenance


class TextSpan(ContractModel):
    """Character span in provider text that matched a ticker or evidence quote."""

    text: NonEmptyStr
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_span(self) -> TextSpan:
        if self.end_char < self.start_char:
            raise ValueError("end_char must be greater than or equal to start_char")
        return self


class SourceEvidence(ContractModel):
    """Cross-provider normalized evidence record."""

    evidence_id: NonEmptyStr
    source_kind: SourceKind
    ticker: TickerSymbol | None = None
    title: str | None = None
    text: NonEmptyStr
    author_hash: str | None = None
    created_at: AwareDatetime | None = None
    score: int | None = None
    permalink: str | None = None
    instrument_id: NonEmptyStr | None = None
    matched_tickers: tuple[TickerSymbol, ...] = Field(default_factory=tuple)
    matched_instrument_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    match_spans: tuple[TextSpan, ...] = Field(default_factory=tuple)
    provenance: SourceProvenance
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("matched_tickers")
    @classmethod
    def remove_duplicate_tickers(cls, tickers: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(tickers))

    @field_validator("matched_instrument_ids")
    @classmethod
    def remove_duplicate_instrument_ids(cls, instrument_ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(instrument_ids))

    @model_validator(mode="after")
    def validate_instrument_traceability(self) -> SourceEvidence:
        if self.source_kind != self.provenance.source_kind:
            raise ValueError("source_kind must match provenance.source_kind")
        if (
            self.instrument_id is not None
            and self.matched_instrument_ids
            and self.instrument_id not in self.matched_instrument_ids
        ):
            raise ValueError("instrument_id must be included in matched_instrument_ids")
        for span in self.match_spans:
            if span.end_char > len(self.text):
                raise ValueError("match_spans must stay within evidence text")
            if self.text[span.start_char : span.end_char] != span.text:
                raise ValueError("match_spans text must match evidence text")
        return self


__all__ = ["SourceEvidence", "TextSpan"]
