"""Prepared report view for renderers."""

from __future__ import annotations

from dataclasses import dataclass

from nlp_stock_prediction.contracts.instruments import Instrument
from nlp_stock_prediction.contracts.provenance import DataReference, ProviderWarning
from nlp_stock_prediction.contracts.report import (
    AuditArtifact,
    AuditManifest,
    DailyReport,
    InstrumentReportSection,
    PredictionCandidate,
)


@dataclass(frozen=True)
class ReportView:
    """Deep renderer Interface over a validated daily report."""

    report: DailyReport
    instruments_by_id: dict[str, Instrument]
    candidates_by_id: dict[str, PredictionCandidate]
    provider_warnings: tuple[ProviderWarning, ...]
    audit_artifacts: tuple[AuditArtifact, ...]
    audit_reference: DataReference | None

    @classmethod
    def from_report(cls, report: DailyReport) -> ReportView:
        manifest = report.audit_manifest
        audit_artifacts = manifest.artifacts if isinstance(manifest, AuditManifest) else ()
        audit_reference = manifest if isinstance(manifest, DataReference) else None
        return cls(
            report=report,
            instruments_by_id={
                instrument.instrument_id: instrument for instrument in report.instruments
            },
            candidates_by_id={
                candidate.candidate_id: candidate for candidate in report.prediction_candidates
            },
            provider_warnings=tuple(
                warning for health in report.provider_health for warning in health.warnings
            ),
            audit_artifacts=audit_artifacts,
            audit_reference=audit_reference,
        )

    def instrument_for_section(self, section: InstrumentReportSection) -> Instrument | None:
        return self.instruments_by_id.get(section.instrument_id)

    def candidates_for_section(
        self,
        section: InstrumentReportSection,
    ) -> tuple[PredictionCandidate, ...]:
        return tuple(
            self.candidates_by_id[candidate_id]
            for candidate_id in section.prediction_candidate_ids
            if candidate_id in self.candidates_by_id
        )


__all__ = ["ReportView"]
