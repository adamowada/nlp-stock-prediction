"""Ignored local state for the persistent terminal app."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from nlp_stock_prediction.app.settings import AppSettings
from nlp_stock_prediction.contracts import DailyReport
from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.orchestration.report_bundle import ReportBundle
from nlp_stock_prediction.reporting.json import load_json_report

APP_STATE_SCHEMA_VERSION = "terminal-app-state.v2"
APP_STATE_PATH = Path("data/app-state.json")


@dataclass(frozen=True)
class ReportIndexEntry:
    report_id: str
    run_id: str
    symbol: str
    report_date: date
    generated_at: datetime
    report_data_mode: str
    markdown_path: Path
    json_path: Path
    audit_dir: Path
    audit_manifest_path: Path | None = None
    database_path: Path | None = None
    candidate_count: int = 0
    provider_warning_count: int = 0
    status_summary: str = "unknown"

    @classmethod
    def from_report(
        cls,
        *,
        repo_root: Path,
        report: DailyReport,
        json_path: Path,
        markdown_path: Path | None = None,
        audit_dir: Path | None = None,
        audit_manifest_path: Path | None = None,
        database_path: Path | None = None,
    ) -> ReportIndexEntry:
        symbol = report.instruments[0].symbol if report.instruments else "UNKNOWN"
        mode = str(report.command_args.get("report_data_mode") or "unknown")
        resolved_json = _resolve_repo_path(repo_root, json_path)
        resolved_database_path = database_path or _database_path_from_report(report)
        report_id = f"{report.run_id}:{_state_path(repo_root, resolved_json).as_posix()}"
        warning_count = sum(len(health.warnings) for health in report.provider_health)
        status_summary = _status_summary(report)
        inferred_markdown = markdown_path or resolved_json.with_suffix(".md")
        inferred_audit_dir = audit_dir or resolved_json.parent / "audit"
        inferred_manifest = audit_manifest_path or inferred_audit_dir / "audit-manifest.json"
        return cls(
            report_id=report_id,
            run_id=report.run_id,
            symbol=symbol.upper(),
            report_date=report.report_date,
            generated_at=report.generated_at,
            report_data_mode=mode,
            markdown_path=_state_path(repo_root, inferred_markdown),
            json_path=_state_path(repo_root, resolved_json),
            audit_dir=_state_path(repo_root, inferred_audit_dir),
            audit_manifest_path=_state_path(repo_root, inferred_manifest),
            database_path=(
                _state_path(repo_root, resolved_database_path)
                if resolved_database_path is not None
                else None
            ),
            candidate_count=len(report.prediction_candidates),
            provider_warning_count=warning_count,
            status_summary=status_summary,
        )

    @classmethod
    def from_json(cls, payload: object) -> ReportIndexEntry | None:
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                report_id=_required_string(payload, "report_id"),
                run_id=_required_string(payload, "run_id"),
                symbol=_required_string(payload, "symbol").upper(),
                report_date=date.fromisoformat(_required_string(payload, "report_date")),
                generated_at=datetime.fromisoformat(_required_string(payload, "generated_at")),
                report_data_mode=_required_string(payload, "report_data_mode"),
                markdown_path=Path(_required_string(payload, "markdown_path")),
                json_path=Path(_required_string(payload, "json_path")),
                audit_dir=Path(_required_string(payload, "audit_dir")),
                audit_manifest_path=_optional_path(payload.get("audit_manifest_path")),
                database_path=_optional_path(payload.get("database_path")),
                candidate_count=_int(payload.get("candidate_count")),
                provider_warning_count=_int(payload.get("provider_warning_count")),
                status_summary=_required_string(payload, "status_summary"),
            )
        except KeyError, ValueError:
            return None

    def to_json(self) -> JsonObject:
        payload: JsonObject = {
            "report_id": self.report_id,
            "run_id": self.run_id,
            "symbol": self.symbol,
            "report_date": self.report_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "report_data_mode": self.report_data_mode,
            "markdown_path": self.markdown_path.as_posix(),
            "json_path": self.json_path.as_posix(),
            "audit_dir": self.audit_dir.as_posix(),
            "candidate_count": self.candidate_count,
            "provider_warning_count": self.provider_warning_count,
            "status_summary": self.status_summary,
        }
        if self.audit_manifest_path is not None:
            payload["audit_manifest_path"] = self.audit_manifest_path.as_posix()
        if self.database_path is not None:
            payload["database_path"] = self.database_path.as_posix()
        return payload

    def resolve_json_path(self, repo_root: Path) -> Path:
        return _resolve_repo_path(repo_root, self.json_path)

    def resolve_markdown_path(self, repo_root: Path) -> Path:
        return _resolve_repo_path(repo_root, self.markdown_path)

    def resolve_audit_dir(self, repo_root: Path) -> Path:
        return _resolve_repo_path(repo_root, self.audit_dir)

    def resolve_database_path(self, repo_root: Path) -> Path | None:
        if self.database_path is None:
            return None
        return _resolve_repo_path(repo_root, self.database_path)


@dataclass(frozen=True)
class CodexSessionRecord:
    session_key: str
    session_id: str
    report_id: str | None
    database_path: Path | None
    transcript_path: Path
    updated_at: datetime

    @classmethod
    def from_json(cls, payload: object) -> CodexSessionRecord | None:
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                session_key=_required_string(payload, "session_key"),
                session_id=_required_string(payload, "session_id"),
                report_id=_optional_string(payload.get("report_id")),
                database_path=_optional_path(payload.get("database_path")),
                transcript_path=Path(_required_string(payload, "transcript_path")),
                updated_at=datetime.fromisoformat(_required_string(payload, "updated_at")),
            )
        except KeyError, ValueError:
            return None

    def to_json(self) -> JsonObject:
        payload: JsonObject = {
            "session_key": self.session_key,
            "session_id": self.session_id,
            "transcript_path": self.transcript_path.as_posix(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.report_id is not None:
            payload["report_id"] = self.report_id
        if self.database_path is not None:
            payload["database_path"] = self.database_path.as_posix()
        return payload


@dataclass(frozen=True)
class AppState:
    settings: AppSettings = field(default_factory=AppSettings)
    reports: tuple[ReportIndexEntry, ...] = ()
    selected_report_id: str | None = None
    codex_sessions: tuple[CodexSessionRecord, ...] = ()

    @classmethod
    def from_json(cls, payload: object) -> AppState:
        if not isinstance(payload, dict):
            return cls()
        reports = tuple(
            entry
            for item in _list(payload.get("reports"))
            if (entry := ReportIndexEntry.from_json(item)) is not None
        )
        sessions = tuple(
            session
            for item in _list(payload.get("codex_sessions"))
            if (session := CodexSessionRecord.from_json(item)) is not None
        )
        settings = AppSettings.from_json(payload.get("settings"))
        if payload.get("schema_version") != APP_STATE_SCHEMA_VERSION:
            settings = _migrate_legacy_settings(settings)
        return cls(
            settings=settings,
            reports=_sort_reports(reports),
            selected_report_id=_optional_string(payload.get("selected_report_id")),
            codex_sessions=sessions,
        )

    def to_json(self) -> JsonObject:
        payload: JsonObject = {
            "schema_version": APP_STATE_SCHEMA_VERSION,
            "settings": self.settings.to_json(),
            "reports": [entry.to_json() for entry in self.reports],
            "codex_sessions": [session.to_json() for session in self.codex_sessions],
        }
        if self.selected_report_id is not None:
            payload["selected_report_id"] = self.selected_report_id
        return payload

    def selected_report(self) -> ReportIndexEntry | None:
        if self.selected_report_id is None:
            return self.reports[0] if self.reports else None
        return next(
            (entry for entry in self.reports if entry.report_id == self.selected_report_id),
            self.reports[0] if self.reports else None,
        )

    def with_reports(self, reports: tuple[ReportIndexEntry, ...]) -> AppState:
        selected = self.selected_report_id
        if selected is not None and all(entry.report_id != selected for entry in reports):
            selected = reports[0].report_id if reports else None
        return replace(self, reports=_sort_reports(reports), selected_report_id=selected)

    def upsert_report(self, entry: ReportIndexEntry) -> AppState:
        merged = {item.report_id: item for item in self.reports}
        merged[entry.report_id] = entry
        return replace(
            self,
            reports=_sort_reports(tuple(merged.values())),
            selected_report_id=entry.report_id,
        )

    def select_report(self, report_id: str | None) -> AppState:
        if report_id is None:
            return replace(self, selected_report_id=None)
        if any(entry.report_id == report_id for entry in self.reports):
            return replace(self, selected_report_id=report_id)
        return self

    def upsert_codex_session(self, record: CodexSessionRecord) -> AppState:
        sessions = {item.session_key: item for item in self.codex_sessions}
        sessions[record.session_key] = record
        return replace(self, codex_sessions=tuple(sessions.values()))

    def codex_session(self, session_key: str) -> CodexSessionRecord | None:
        return next(
            (record for record in self.codex_sessions if record.session_key == session_key),
            None,
        )


def load_app_state(repo_root: Path, path: Path = APP_STATE_PATH) -> AppState:
    state_path = _resolve_repo_path(repo_root, path)
    if not state_path.exists():
        return AppState()
    try:
        return AppState.from_json(json.loads(state_path.read_text(encoding="utf-8")))
    except OSError, json.JSONDecodeError:
        return AppState()


def save_app_state(repo_root: Path, state: AppState, path: Path = APP_STATE_PATH) -> None:
    state_path = _resolve_repo_path(repo_root, path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state.to_json(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def refresh_report_index(repo_root: Path, state: AppState) -> AppState:
    known_by_json = {
        _state_path(repo_root, entry.resolve_json_path(repo_root)).as_posix(): entry
        for entry in state.reports
    }
    reports: dict[str, ReportIndexEntry] = {}
    for report_root in _report_scan_roots(repo_root, state):
        for json_path in report_root.glob("**/report.json"):
            try:
                report = load_json_report(json_path.read_text(encoding="utf-8"))
            except OSError, ValueError:
                continue
            key = _state_path(repo_root, json_path).as_posix()
            previous = known_by_json.get(key)
            entry = ReportIndexEntry.from_report(
                repo_root=repo_root,
                report=report,
                json_path=json_path,
                database_path=previous.database_path if previous else None,
            )
            reports[entry.report_id] = entry
    for entry in state.reports:
        if entry.resolve_json_path(repo_root).exists():
            reports.setdefault(entry.report_id, entry)
    return state.with_reports(tuple(reports.values()))


def report_entry_from_bundle(repo_root: Path, bundle: ReportBundle) -> ReportIndexEntry:
    return ReportIndexEntry.from_report(
        repo_root=repo_root,
        report=bundle.report,
        json_path=bundle.json_path,
        markdown_path=bundle.markdown_path,
        audit_dir=bundle.audit_dir,
        audit_manifest_path=bundle.audit_manifest_path,
        database_path=bundle.database_path,
    )


def _status_summary(report: DailyReport) -> str:
    if report.prediction_candidates:
        statuses = sorted({candidate.status.value for candidate in report.prediction_candidates})
        return ", ".join(statuses)
    if report.insufficient_evidence is not None:
        return report.insufficient_evidence.summary
    return report.insufficient_evidence_summary or "No prediction scenario emitted."


def _sort_reports(reports: tuple[ReportIndexEntry, ...]) -> tuple[ReportIndexEntry, ...]:
    return tuple(sorted(reports, key=lambda entry: entry.generated_at, reverse=True))


def _migrate_legacy_settings(settings: AppSettings) -> AppSettings:
    if settings.default_symbol == "TSLA":
        return settings.with_updates(default_symbol="")
    return settings


def _database_path_from_report(report: DailyReport) -> Path | None:
    for key in ("database_path", "research_database_path", "runtime_database_path"):
        value = report.command_args.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.strip())
    return None


def _report_scan_roots(repo_root: Path, state: AppState) -> tuple[Path, ...]:
    candidates = [
        repo_root / "reports",
        _resolve_repo_path(repo_root, state.settings.output_dir),
    ]
    for entry in state.reports:
        json_path = entry.resolve_json_path(repo_root)
        if len(json_path.parents) >= 3:
            candidates.append(json_path.parents[2])

    roots: dict[str, Path] = {}
    for candidate in candidates:
        resolved = candidate.resolve()
        roots[resolved.as_posix()] = resolved
    return tuple(roots.values())


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _state_path(repo_root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    try:
        return resolved.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return resolved


def _required_string(payload: dict[Any, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_path(value: object) -> Path | None:
    text = _optional_string(value)
    return Path(text) if text is not None else None


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "APP_STATE_PATH",
    "APP_STATE_SCHEMA_VERSION",
    "AppState",
    "CodexSessionRecord",
    "ReportIndexEntry",
    "load_app_state",
    "refresh_report_index",
    "report_entry_from_bundle",
    "save_app_state",
    "utc_now",
]
