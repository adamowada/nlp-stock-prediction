"""Shared Phase 2 orchestration helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from nlp_stock_prediction.contracts import AssetClass, Instrument, TradabilityStatus
from nlp_stock_prediction.contracts.instruments import (
    InstrumentDataAvailability,
    ProviderInstrumentId,
    TradabilityEvidence,
)
from nlp_stock_prediction.storage import ResearchRunRecord

ALLOWED_WRITE_ROOTS = ("reports", "artifacts", "data", "cache")


@dataclass(frozen=True)
class Phase2RunPaths:
    output_dir: Path
    run_dir: Path
    audit_dir: Path
    report_path: Path
    json_path: Path
    audit_manifest_path: Path


@dataclass(frozen=True)
class Phase2WritePolicy:
    """Resolve Phase 2 write paths while enforcing ignored repo roots."""

    repo_root: Path
    allowed_roots: tuple[str, ...] = ALLOWED_WRITE_ROOTS

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", self.repo_root.resolve())

    def resolve(self, path: Path) -> Path:
        resolved = path if path.is_absolute() else self.repo_root / path
        resolved = resolved.resolve()
        allowed = tuple((self.repo_root / root).resolve() for root in self.allowed_roots)
        if not any(is_relative_to(resolved, root) or resolved == root for root in allowed):
            roots = ", ".join(root.as_posix() for root in allowed)
            raise ValueError(f"Phase 2 MCP writes are limited to: {roots}")
        return resolved

    def run_paths(self, run_date: date, output_dir: str) -> Phase2RunPaths:
        output_path = self.resolve(Path(output_dir))
        run_dir = output_path / run_date.isoformat()
        audit_dir = run_dir / "audit"
        return Phase2RunPaths(
            output_dir=output_path,
            run_dir=run_dir,
            audit_dir=audit_dir,
            report_path=run_dir / "report.md",
            json_path=run_dir / "report.json",
            audit_manifest_path=audit_dir / "audit-manifest.json",
        )


def phase2_instrument(symbol: str, retrieved_at: datetime) -> Instrument:
    normalized = symbol.upper()
    instrument_id = f"instrument:codex:{normalized}"
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        display_name=f"{normalized} Codex smoke instrument",
        asset_class=AssetClass.STOCK,
        provider_ids=(
            ProviderInstrumentId(
                provider="codex-smoke",
                identifier=normalized,
                namespace="symbol",
            ),
        ),
        tradability_evidence=(
            TradabilityEvidence(
                provider="codex-smoke",
                status=TradabilityStatus.UNKNOWN,
                retrieved_at=retrieved_at,
                raw_identifier=f"{normalized}:codex-smoke",
                notes="Smoke records researchability only, not trading availability.",
            ),
        ),
        data_availability=(
            InstrumentDataAvailability(
                provider="codex-smoke",
                data_type="dummy_tool_context",
                status=TradabilityStatus.AVAILABLE,
                checked_at=retrieved_at,
                provider_identifier=normalized,
            ),
        ),
        metadata={"phase2_mcp": True},
    )


def phase2_run_id(run_date: date, symbol: str) -> str:
    return f"codex-smoke-{run_date.isoformat()}-{symbol_slug(symbol)}"


def symbol_slug(symbol: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", symbol.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "symbol"


def source_evidence_ticker(symbol: str) -> str | None:
    normalized = symbol.strip().upper()
    return normalized if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", normalized) else None


def normalize_evidence_stance(stance: str | None, claim: str) -> str:
    if stance is not None and stance.strip():
        normalized = stance.strip().lower().replace("_", "-")
        if normalized in {"support", "supports", "supportive", "for", "bullish", "positive"}:
            return "supports"
        if normalized in {
            "against",
            "contradict",
            "contradicts",
            "contradictory",
            "conflict",
            "conflicts",
            "bearish",
            "negative",
        }:
            return "contradicts"
        if normalized in {"neutral", "mixed", "unclear", "context"}:
            return "neutral"
        raise ValueError("stance must be supports, contradicts, or neutral")
    return "neutral"


def run_date_from_run(run: ResearchRunRecord) -> date:
    raw = run.metadata.get("run_date")
    if isinstance(raw, str):
        return date.fromisoformat(raw)
    return run.started_at.date()


def parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "ALLOWED_WRITE_ROOTS",
    "Phase2RunPaths",
    "Phase2WritePolicy",
    "file_sha256",
    "is_relative_to",
    "normalize_evidence_stance",
    "parse_optional_datetime",
    "phase2_instrument",
    "phase2_run_id",
    "run_date_from_run",
    "source_evidence_ticker",
    "stable_digest",
    "symbol_slug",
    "utc_now",
]
