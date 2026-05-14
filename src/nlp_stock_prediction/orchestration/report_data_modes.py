"""Runtime data-mode boundaries for prediction report inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nlp_stock_prediction.contracts.base import JsonObject, JsonValue
from nlp_stock_prediction.storage.records import ResearchRunRecord
from nlp_stock_prediction.storage.sqlite import SQLiteStore

type ReportDataMode = Literal["live", "offline_fixture", "dummy_smoke", "codex_smoke"]

LIVE_REPORT_DATA_MODE: ReportDataMode = "live"
OFFLINE_FIXTURE_REPORT_DATA_MODE: ReportDataMode = "offline_fixture"
DUMMY_SMOKE_REPORT_DATA_MODE: ReportDataMode = "dummy_smoke"
CODEX_SMOKE_REPORT_DATA_MODE: ReportDataMode = "codex_smoke"

REPORT_DATA_MODE_KEY = "report_data_mode"
PROVIDER_MODE_KEY = "provider_mode"
INPUT_DATA_MODE_KEY = "input_data_mode"

KNOWN_REPORT_DATA_MODES = frozenset(
    {
        LIVE_REPORT_DATA_MODE,
        OFFLINE_FIXTURE_REPORT_DATA_MODE,
        DUMMY_SMOKE_REPORT_DATA_MODE,
        CODEX_SMOKE_REPORT_DATA_MODE,
    }
)
NON_LIVE_REPORT_DATA_MODES = frozenset(
    {
        OFFLINE_FIXTURE_REPORT_DATA_MODE,
        DUMMY_SMOKE_REPORT_DATA_MODE,
        CODEX_SMOKE_REPORT_DATA_MODE,
    }
)

_MODE_KEYS = frozenset(
    {
        REPORT_DATA_MODE_KEY,
        PROVIDER_MODE_KEY,
        INPUT_DATA_MODE_KEY,
        "source_mode",
        "retrieval_method",
        "report_input_mode",
        "artifact_data_mode",
        "tool_data_mode",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "provider",
        "provider_name",
        "produced_by",
        "schema_version",
        "source",
        "tool_name",
        "tool_version",
    }
)
_BOOLEAN_MARKER_KEYS = frozenset({"dummy", "fixture", "fixtures", "smoke"})
_NON_LIVE_TEXT_MARKERS = ("fixture", "dummy", "smoke")
_NON_LIVE_MODE_MARKERS = frozenset(
    {
        "offline",
        "offline_fixture",
        "fixture",
        "fixtures",
        "fixture_backed",
        "dummy",
        "dummy_smoke",
        "codex_smoke",
        "smoke",
        "test",
    }
)


@dataclass(frozen=True)
class ReportInputBoundaryViolation:
    """One non-live input discovered while validating a live report assembly."""

    record_type: str
    record_id: str
    field: str
    value: str

    def as_text(self) -> str:
        return f"{self.record_type} {self.record_id} has {self.field}={self.value!r}"


def report_data_mode_metadata(
    mode: ReportDataMode,
    *,
    provider_mode: ReportDataMode | None = None,
) -> JsonObject:
    """Return the canonical JSON metadata stamped on report inputs and outputs."""

    normalized_mode = normalize_report_data_mode(mode)
    normalized_provider_mode = normalize_report_data_mode(provider_mode or mode)
    return {
        REPORT_DATA_MODE_KEY: normalized_mode,
        PROVIDER_MODE_KEY: normalized_provider_mode,
        "live_report_inputs": normalized_mode == LIVE_REPORT_DATA_MODE,
    }


def merge_report_data_mode_metadata(
    metadata: JsonObject | None,
    mode: ReportDataMode,
    *,
    provider_mode: ReportDataMode | None = None,
) -> JsonObject:
    """Merge report data-mode metadata without replacing explicit caller metadata."""

    return {
        **report_data_mode_metadata(mode, provider_mode=provider_mode),
        **({} if metadata is None else metadata),
    }


def report_data_mode_from_run(
    run: ResearchRunRecord,
    *,
    default: ReportDataMode | None = None,
) -> ReportDataMode:
    """Resolve a run's report data mode, including legacy run-kind inference."""

    raw_mode = run.metadata.get(REPORT_DATA_MODE_KEY)
    if raw_mode is not None:
        return normalize_report_data_mode(raw_mode)
    if default is not None:
        return default
    run_kind = run.run_kind.lower()
    if "codex_smoke" in run_kind or "codex-smoke" in run_kind:
        return CODEX_SMOKE_REPORT_DATA_MODE
    if "dummy" in run_kind:
        return DUMMY_SMOKE_REPORT_DATA_MODE
    if "phase4" in run_kind:
        return OFFLINE_FIXTURE_REPORT_DATA_MODE
    raise ValueError(f"research run is missing report_data_mode metadata: {run.run_id}")


def report_data_mode_metadata_from_run(run: ResearchRunRecord | None) -> JsonObject:
    if run is None:
        return {}
    return report_data_mode_metadata(report_data_mode_from_run(run))


def report_data_mode_metadata_for_run_id(store: SQLiteStore, run_id: str) -> JsonObject:
    return report_data_mode_metadata_from_run(store.get_research_run(run_id))


def normalize_report_data_mode(value: object) -> ReportDataMode:
    if isinstance(value, str) and value in KNOWN_REPORT_DATA_MODES:
        return value
    raise ValueError(
        "report data mode must be one of: " + ", ".join(sorted(KNOWN_REPORT_DATA_MODES))
    )


def enforce_live_report_input_boundary(
    *,
    store: SQLiteStore,
    run: ResearchRunRecord,
) -> tuple[ReportInputBoundaryViolation, ...]:
    """Raise when a live report run tries to assemble fixture, dummy, or smoke inputs."""

    report_data_mode = report_data_mode_from_run(run)
    if report_data_mode != LIVE_REPORT_DATA_MODE:
        return ()
    violations = find_non_live_report_input_violations(store=store, run=run)
    if violations:
        details = "; ".join(violation.as_text() for violation in violations[:8])
        extra_count = len(violations) - 8
        if extra_count > 0:
            details = f"{details}; and {extra_count} more"
        raise ValueError("live report assembly cannot use non-live report inputs: " + details)
    return ()


def find_non_live_report_input_violations(
    *,
    store: SQLiteStore,
    run: ResearchRunRecord,
) -> tuple[ReportInputBoundaryViolation, ...]:
    violations: list[ReportInputBoundaryViolation] = []
    violations.extend(_metadata_violations("research_run", run.run_id, run.metadata))
    for tool_run in store.list_tool_runs_for_run(run.run_id):
        violations.extend(
            _text_marker_violations(
                record_type="tool_run",
                record_id=tool_run.tool_run_id,
                fields={
                    "tool_name": tool_run.tool_name,
                    "tool_version": tool_run.tool_version,
                },
            )
        )
        violations.extend(_metadata_violations("tool_run", tool_run.tool_run_id, tool_run.inputs))
    for artifact in store.list_artifacts_for_run(run.run_id):
        violations.extend(
            _text_marker_violations(
                record_type="artifact",
                record_id=artifact.artifact_id,
                fields={
                    "schema_version": artifact.schema_version,
                    "produced_by": artifact.produced_by or "",
                },
            )
        )
        violations.extend(_metadata_violations("artifact", artifact.artifact_id, artifact.metadata))
    for source_query in store.list_source_queries_for_run(run.run_id):
        violations.extend(
            _text_marker_violations(
                record_type="source_query",
                record_id=source_query.source_query_id,
                fields={"provider": source_query.provider},
            )
        )
        violations.extend(
            _metadata_violations(
                "source_query",
                source_query.source_query_id,
                source_query.metadata,
            )
        )
    for evidence in store.list_evidence_for_run(run.run_id):
        violations.extend(
            _text_marker_violations(
                record_type="evidence",
                record_id=evidence.evidence_id,
                fields={"provider": evidence.provider},
            )
        )
        violations.extend(_metadata_violations("evidence", evidence.evidence_id, evidence.metadata))
        violations.extend(
            _metadata_violations(
                "evidence_provenance",
                evidence.evidence_id,
                evidence.provenance_json,
            )
        )
    for candidate in store.list_prediction_candidates_for_run(run.run_id):
        violations.extend(
            _text_marker_violations(
                record_type="prediction_candidate",
                record_id=candidate.candidate_id,
                fields={
                    "instrument_id": candidate.instrument_id,
                    "prediction_horizon": candidate.prediction_horizon,
                    "prediction_type": candidate.prediction_type,
                    "scenario": candidate.scenario,
                    "signal_artifacts": " ".join(candidate.signal_artifacts),
                },
            )
        )
        violations.extend(
            _metadata_violations(
                "prediction_candidate",
                candidate.candidate_id,
                candidate.metadata,
            )
        )
        violations.extend(
            _metadata_violations(
                "prediction_candidate_baseline",
                candidate.candidate_id,
                candidate.baseline,
            )
        )
        instrument = store.get_instrument(candidate.instrument_id)
        if instrument is not None:
            violations.extend(_instrument_violations(instrument.instrument_id, instrument.metadata))
            violations.extend(
                _structured_record_violations(
                    "instrument_provider_id",
                    instrument.instrument_id,
                    instrument.provider_ids,
                )
            )
            violations.extend(
                _structured_record_violations(
                    "instrument_tradability",
                    instrument.instrument_id,
                    instrument.tradability_evidence,
                )
            )
            violations.extend(
                _structured_record_violations(
                    "instrument_data_availability",
                    instrument.instrument_id,
                    instrument.data_availability,
                )
            )
    return tuple(_dedupe_violations(violations))


def _metadata_violations(
    record_type: str,
    record_id: str,
    metadata: JsonObject,
) -> list[ReportInputBoundaryViolation]:
    violations: list[ReportInputBoundaryViolation] = []
    for key, value in metadata.items():
        if key in _MODE_KEYS and _mode_value_is_non_live(value):
            violations.append(ReportInputBoundaryViolation(record_type, record_id, key, str(value)))
            continue
        if key in _PROVENANCE_KEYS and isinstance(value, str) and _text_is_non_live(value):
            violations.append(ReportInputBoundaryViolation(record_type, record_id, key, value))
            continue
        if key.lower() in _BOOLEAN_MARKER_KEYS and value is True:
            violations.append(ReportInputBoundaryViolation(record_type, record_id, key, str(value)))
            continue
        if isinstance(value, dict) and key in {"metadata", "provider_metadata", "provenance"}:
            violations.extend(
                _metadata_violations(
                    record_type,
                    record_id,
                    value,
                )
            )
    return violations


def _text_marker_violations(
    *,
    record_type: str,
    record_id: str,
    fields: dict[str, str],
) -> list[ReportInputBoundaryViolation]:
    return [
        ReportInputBoundaryViolation(record_type, record_id, field, value)
        for field, value in fields.items()
        if _text_is_non_live(value)
    ]


def _instrument_violations(
    instrument_id: str,
    metadata: JsonObject,
) -> list[ReportInputBoundaryViolation]:
    violations = _text_marker_violations(
        record_type="instrument",
        record_id=instrument_id,
        fields={"instrument_id": instrument_id},
    )
    violations.extend(_metadata_violations("instrument", instrument_id, metadata))
    return violations


def _structured_record_violations(
    record_type: str,
    record_id: str,
    values: tuple[JsonObject, ...],
) -> list[ReportInputBoundaryViolation]:
    violations: list[ReportInputBoundaryViolation] = []
    for index, value in enumerate(values):
        violations.extend(_metadata_violations(record_type, f"{record_id}:{index}", value))
    return violations


def _mode_value_is_non_live(value: JsonValue) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().replace("-", "_")
    return normalized in _NON_LIVE_MODE_MARKERS


def _text_is_non_live(value: str) -> bool:
    normalized = value.strip().lower()
    return any(marker in normalized for marker in _NON_LIVE_TEXT_MARKERS)


def _dedupe_violations(
    violations: list[ReportInputBoundaryViolation],
) -> tuple[ReportInputBoundaryViolation, ...]:
    deduped: dict[tuple[str, str, str, str], ReportInputBoundaryViolation] = {}
    for violation in violations:
        deduped[
            (
                violation.record_type,
                violation.record_id,
                violation.field,
                violation.value,
            )
        ] = violation
    return tuple(deduped.values())


__all__ = [
    "CODEX_SMOKE_REPORT_DATA_MODE",
    "DUMMY_SMOKE_REPORT_DATA_MODE",
    "INPUT_DATA_MODE_KEY",
    "KNOWN_REPORT_DATA_MODES",
    "LIVE_REPORT_DATA_MODE",
    "NON_LIVE_REPORT_DATA_MODES",
    "OFFLINE_FIXTURE_REPORT_DATA_MODE",
    "PROVIDER_MODE_KEY",
    "REPORT_DATA_MODE_KEY",
    "ReportDataMode",
    "ReportInputBoundaryViolation",
    "enforce_live_report_input_boundary",
    "find_non_live_report_input_violations",
    "merge_report_data_mode_metadata",
    "normalize_report_data_mode",
    "report_data_mode_from_run",
    "report_data_mode_metadata",
    "report_data_mode_metadata_for_run_id",
    "report_data_mode_metadata_from_run",
]
