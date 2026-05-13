"""Run context shared by orchestration tools."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from nlp_stock_prediction.contracts import JsonObject, RunConfig
from nlp_stock_prediction.orchestration.artifacts import ArtifactWriter


def deterministic_generated_at(run_date: date) -> datetime:
    """Return the deterministic UTC timestamp used by offline orchestration runs."""

    return datetime(run_date.year, run_date.month, run_date.day, 21, 0, tzinfo=UTC)


@dataclass(frozen=True)
class RunContext:
    """Immutable context available to every tool in a staged run."""

    run_id: str
    run_date: date
    generated_at: datetime
    timezone: str
    output_dir: Path
    report_dir: Path
    audit_dir: Path
    command_args: JsonObject
    artifact_writer: ArtifactWriter

    @classmethod
    def from_config(
        cls,
        config: RunConfig,
        *,
        run_id: str | None = None,
        generated_at: datetime | None = None,
        timezone: str = "UTC",
        produced_by: str = "dummy-orchestrator",
    ) -> RunContext:
        resolved_run_id = run_id or f"research-{config.run_date.isoformat()}"
        resolved_generated_at = generated_at or deterministic_generated_at(config.run_date)
        report_dir = config.output_dir / config.run_date.isoformat()
        audit_dir = report_dir / "audit"
        return cls(
            run_id=resolved_run_id,
            run_date=config.run_date,
            generated_at=resolved_generated_at,
            timezone=timezone,
            output_dir=config.output_dir,
            report_dir=report_dir,
            audit_dir=audit_dir,
            command_args=_command_args(config),
            artifact_writer=ArtifactWriter(
                base_dir=audit_dir,
                created_at=resolved_generated_at,
                produced_by=produced_by,
            ),
        )


def _command_args(config: RunConfig) -> JsonObject:
    return {
        "run_date": config.run_date.isoformat(),
        "output_dir": str(config.output_dir),
        "fixture_dir": str(config.fixture_dir) if config.fixture_dir else None,
        "cache_dir": str(config.cache_dir) if config.cache_dir else None,
        "offline": config.offline,
        "source_mode": config.source_mode,
        "live_providers": config.live_providers,
    }


__all__ = ["RunContext", "deterministic_generated_at"]
