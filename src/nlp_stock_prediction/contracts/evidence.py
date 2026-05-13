"""Normalized evidence contracts."""

from __future__ import annotations

from pydantic import Field, model_validator

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
    matched_tickers: tuple[TickerSymbol, ...] = Field(default_factory=tuple)
    match_spans: tuple[TextSpan, ...] = Field(default_factory=tuple)
    provenance: SourceProvenance
    metadata: JsonObject = Field(default_factory=dict)


__all__ = ["SourceEvidence", "TextSpan"]
