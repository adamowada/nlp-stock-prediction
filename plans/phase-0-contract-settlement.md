# Phase 0 Contract Settlement

## Goal

Implement the Phase 0 contract surface so later Codex subagents can work in parallel from a
stable, shared Python package contract.

## Non-goals

- Do not implement provider adapters, analysis formulas, recommendation scoring, report rendering,
  live scraping, live API access, or worktree lane branches.
- Do not complete the Phase 1 comprehensive contract test harness.
- Do not add real-money brokerage execution.

## Context

- The roadmap is `docs/multi-milestone-plan.md`.
- The frozen contract reference is `docs/contracts.md`.
- The package uses the canonical module invocation `python -m nlp_stock_prediction`.
- Phase 0 may scaffold package files needed to define public contracts.

## Milestones

### Milestone 1: Minimal Package Scaffold

- Changes: add `pyproject.toml`, `src/nlp_stock_prediction/`, module entrypoint, and typed package marker.
- Files likely affected: `pyproject.toml`, `src/nlp_stock_prediction/`.
- Verification: package imports and CLI help work once dependencies are installed.

### Milestone 2: Public Contract Models

- Changes: define Pydantic models and enums for provenance, provider health, ticker discovery,
  evidence, extraction, analysis, recommendations, reports, provider protocols, and fixtures.
- Files likely affected: `src/nlp_stock_prediction/contracts/`.
- Verification: schema smoke tests and serialization checks pass.

### Milestone 3: Contract Documentation And Status

- Changes: document frozen contracts, fixture layout, phase status, and pending Phase 1 work.
- Files likely affected: `docs/contracts.md`, `docs/multi-milestone-plan.md`, `README.md`.
- Verification: docs agree that Phase 1 and parallel implementation lanes have not started.

## Acceptance Criteria

- [x] Public models are defined at API boundaries.
- [x] Provider protocols are defined without concrete provider implementations.
- [x] Report, audit, fixture, provider health, warning, and CLI contracts are documented.
- [x] Test marker taxonomy is registered in project configuration.
- [x] Phase 0 status is recorded without claiming Phase 1 or parallel lanes have started.
- [x] Worktree usage is documented in a runbook for the later parallel implementation phase.
- [ ] Comprehensive contract tests are added in Phase 1.

## Verification Commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/  # exits 3 in Phase 0
```

## Decision Log

- 2026-05-11-12-46: Phase 0 will use code-backed Pydantic contracts instead of documentation-only contracts, because later worktree lanes need importable shared types and provider protocols.
- 2026-05-11-12-46: Runtime dependencies are limited to Pydantic; provider/network dependencies remain lane-owned and are not added in Phase 0.
- 2026-05-11-12-46: CLI uses stdlib `argparse` and the canonical module invocation; no console script is frozen in Phase 0.
- 2026-05-11-12-46: Provider methods use a synchronous `ProviderResult[T]` envelope. Expected upstream failures are modeled as status/warnings instead of exceptions.
- 2026-05-11-12-46: Phase 0 completion does not complete the full contract gate. Phase 1 contract tests remain required before parallel implementation worktrees begin.
- 2026-05-11-13-05: Provider contracts return normalized provider facts only; analysis outputs are owned by Lane D and are not provider protocol return types.
- 2026-05-11-13-05: The worktree operating procedure is documented in `docs/worktree-runbook.md`; Phase 2 remains blocked until Phase 1 is complete.

## Progress Log

- 2026-05-11-12-46: Reviewed roadmap, testing plan, README, and planning template; confirmed the repo started with planning documents only.
- 2026-05-11-12-46: Coordinated read-only subagent reviews for architecture, provider semantics, packaging, reporting contracts, and phase-status documentation.
- 2026-05-11-12-46: Added minimal package scaffold, public contracts, fixture contract, contract docs, and Phase 0 smoke tests.
- 2026-05-11-12-46: Recorded that Phase 1 comprehensive contract harness and parallel implementation lanes remain pending.
- 2026-05-11-13-05: Tightened provider-result, JSON metadata, and ticker-discovery invariants based on Phase 0 review findings.
- 2026-05-11-13-05: Added a concrete git worktree runbook with branch names, commands, handoff template, verification commands, and merge protocol.
