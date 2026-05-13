"""MCP-facing Phase 2 module for real Codex smoke runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from nlp_stock_prediction.contracts import JsonObject
from nlp_stock_prediction.orchestration.phase2_common import (
    ALLOWED_WRITE_ROOTS,
    Phase2RunPaths,
    Phase2WritePolicy,
    phase2_run_id,
    run_date_from_run,
    utc_now,
)
from nlp_stock_prediction.orchestration.phase2_dummy_tools import (
    run_phase2_dummy_analysis_tool,
    run_phase2_dummy_universe_tool,
)
from nlp_stock_prediction.orchestration.phase2_evidence import record_codex_search_evidence
from nlp_stock_prediction.orchestration.phase2_report import render_phase2_prediction_report
from nlp_stock_prediction.orchestration.phase2_synthesis import synthesize_prediction_candidates
from nlp_stock_prediction.orchestration.phase2_tool_plan import phase2_research_tool_plan
from nlp_stock_prediction.storage import (
    ResearchRunRecord,
    SQLiteStore,
    initialize_research_database,
)


@dataclass(frozen=True)
class Phase2McpService:
    """Stateful Phase 2 interface exposed through MCP and tests."""

    repo_root: Path = Path(".")
    database_path: Path = Path("data/prediction-research.sqlite3")
    _store: SQLiteStore = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", self.repo_root.resolve())
        object.__setattr__(
            self,
            "_store",
            initialize_research_database(self._resolve_write_path(self.database_path)),
        )

    @property
    def write_policy(self) -> Phase2WritePolicy:
        return Phase2WritePolicy(self.repo_root)

    @property
    def store(self) -> SQLiteStore:
        return self._store

    def start_research_run(
        self,
        *,
        run_date: str,
        output_dir: str,
        symbol: str,
        objective: str | None = None,
    ) -> JsonObject:
        parsed_date = date.fromisoformat(run_date)
        normalized_symbol = symbol.strip().upper()
        run_id = phase2_run_id(parsed_date, normalized_symbol)
        paths = self._paths(parsed_date, output_dir)
        now = utc_now()
        self.store.upsert_research_run(
            ResearchRunRecord(
                run_id=run_id,
                run_kind="codex_smoke_prediction_report",
                objective=objective or f"Codex smoke prediction research for {normalized_symbol}",
                status="running",
                started_at=now,
                metadata={
                    "run_date": parsed_date.isoformat(),
                    "symbol": normalized_symbol,
                    "output_dir": paths.output_dir.as_posix(),
                    "phase": "phase2_codex_orchestrator",
                },
            )
        )
        return {
            "run_id": run_id,
            "run_date": parsed_date.isoformat(),
            "symbol": normalized_symbol,
            "output_dir": paths.output_dir.as_posix(),
            "run_dir": paths.run_dir.as_posix(),
            "audit_dir": paths.audit_dir.as_posix(),
            "database_path": self._resolve_write_path(self.database_path).as_posix(),
        }

    def list_research_tool_plan(self) -> JsonObject:
        return phase2_research_tool_plan()

    def record_codex_search_evidence(
        self,
        *,
        run_id: str,
        symbol: str,
        title: str,
        url: str,
        claim: str,
        query: str,
        published_at: str | None = None,
        stance: str | None = None,
    ) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        with self.store.transaction():
            return record_codex_search_evidence(
                store=self.store,
                repo_root=self.repo_root,
                paths=paths,
                run_id=run_id,
                symbol=normalized_symbol,
                title=title,
                url=url,
                claim=claim,
                query=query,
                published_at=published_at,
                stance=stance,
            )

    def run_dummy_universe_tool(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        with self.store.transaction():
            return run_phase2_dummy_universe_tool(
                store=self.store,
                repo_root=self.repo_root,
                paths=paths,
                run_id=run_id,
                symbol=normalized_symbol,
            )

    def run_dummy_analysis_tool(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        with self.store.transaction():
            return run_phase2_dummy_analysis_tool(
                store=self.store,
                repo_root=self.repo_root,
                paths=paths,
                run_id=run_id,
                symbol=normalized_symbol,
            )

    def synthesize_prediction_candidates(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        with self.store.transaction():
            return synthesize_prediction_candidates(
                store=self.store,
                repo_root=self.repo_root,
                paths=paths,
                run_id=run_id,
                symbol=normalized_symbol,
                ensure_instrument=lambda: self.run_dummy_universe_tool(
                    run_id=run_id,
                    symbol=normalized_symbol,
                ),
            )

    def render_prediction_report(self, *, run_id: str, symbol: str) -> JsonObject:
        run = self._require_run(run_id)
        normalized_symbol = self._validated_symbol(run, symbol)
        run_date = run_date_from_run(run)
        paths = self._paths(run_date, str(run.metadata["output_dir"]))
        with self.store.transaction():
            if not self.store.list_prediction_candidates_for_run(run_id):
                self.synthesize_prediction_candidates(run_id=run_id, symbol=normalized_symbol)
            return render_phase2_prediction_report(
                store=self.store,
                repo_root=self.repo_root,
                run=run,
                paths=paths,
                run_date=run_date,
                symbol=normalized_symbol,
            )

    def inspect_research_run(self, *, run_id: str) -> JsonObject:
        run = self._require_run(run_id)
        store = self.store
        return {
            "run_id": run.run_id,
            "status": run.status,
            "tool_run_count": len(store.list_tool_runs_for_run(run_id)),
            "artifact_count": len(store.list_artifacts_for_run(run_id)),
            "source_query_count": len(store.list_source_queries_for_run(run_id)),
            "evidence_count": len(store.list_evidence_for_run(run_id)),
            "candidate_count": len(store.list_prediction_candidates_for_run(run_id)),
            "has_codex_search_evidence": any(
                record.provider == "codex-web-search"
                for record in store.list_evidence_for_run(run_id)
            ),
        }

    def _require_run(self, run_id: str) -> ResearchRunRecord:
        run = self.store.get_research_run(run_id)
        if run is None:
            raise ValueError(f"research run does not exist: {run_id}")
        return run

    def _validated_symbol(self, run: ResearchRunRecord, symbol: str) -> str:
        stored_symbol = run.metadata.get("symbol")
        if not isinstance(stored_symbol, str) or not stored_symbol.strip():
            raise ValueError(f"research run is missing stored symbol metadata: {run.run_id}")
        normalized_stored_symbol = stored_symbol.strip().upper()
        normalized_symbol = symbol.strip().upper()
        if normalized_symbol != normalized_stored_symbol:
            raise ValueError(
                f"symbol {normalized_symbol} does not match research run symbol "
                f"{normalized_stored_symbol}"
            )
        return normalized_stored_symbol

    def _paths(self, run_date: date, output_dir: str) -> Phase2RunPaths:
        return self.write_policy.run_paths(run_date, output_dir)

    def _resolve_write_path(self, path: Path) -> Path:
        return self.write_policy.resolve(path)


__all__ = ["ALLOWED_WRITE_ROOTS", "Phase2McpService", "Phase2RunPaths"]
