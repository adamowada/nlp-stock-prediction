# Phase 2 Parallel Implementation

## Goal

Implement the Phase 2 application lanes from the frozen `phase-1-contract-gate` tag, then merge them
through `codex/integration-v1` into a fixture-backed CLI that can generate evidence-grounded Markdown,
JSON, and audit reports.

## Non-goals

- Do not change frozen public contracts unless a blocker is escalated and recorded first.
- Do not add real-money brokerage execution or auto-trading.
- Do not make default tests depend on credentials, internet access, or live provider quota.
- Do not require live API, live scraping, or live LLM checks for normal handoff.

## Context

- Base ref: `phase-1-contract-gate` (`5b3d6b5`).
- Integration branch/worktree: `codex/integration-v1` at `C:\Users\adams\projects\nlp-stock-prediction-integration`.
- Lane branches/worktrees:
  - `codex/lane-a-reddit-evidence` at `C:\Users\adams\projects\nlp-stock-prediction-lane-a`.
  - `codex/lane-b-providers` at `C:\Users\adams\projects\nlp-stock-prediction-lane-b`.
  - `codex/lane-c-extraction` at `C:\Users\adams\projects\nlp-stock-prediction-lane-c`.
  - `codex/lane-d-analysis-scoring` at `C:\Users\adams\projects\nlp-stock-prediction-lane-d`.
  - `codex/lane-e-report-cli` at `C:\Users\adams\projects\nlp-stock-prediction-lane-e`.
  - `codex/lane-f-reliability-ci` at `C:\Users\adams\projects\nlp-stock-prediction-lane-f`.
- Shared contracts live under `src/nlp_stock_prediction/contracts/` and are treated as read-only
  during lane work.
- The project is TDD-first; each lane should add focused deterministic tests before or alongside
  implementation.

## Milestones

### Milestone 1: Parallel Lane Implementation

- Changes: Assign six subagents to lanes A-F, each on its own branch and worktree.
- Files likely affected: New implementation modules, fixtures, and tests outside frozen contracts.
- Verification: Lane-specific tests plus default fast suite where feasible.

### Milestone 2: Lane Review And Stabilization

- Changes: Review each lane diff for contract drift, ownership violations, missing provenance,
  evidence gaps, and deterministic test coverage.
- Files likely affected: Lane branches only, with fixes made in the owning worktree.
- Verification: `python -m pytest`, `ruff check .`, `ruff format --check .`, and `mypy .` per lane
  where feasible.

### Milestone 3: Controlled Integration

- Changes: Merge lanes into `codex/integration-v1` in dependency order A, B, C, D, E, F.
- Files likely affected: Integrated package modules, tests, docs, fixture artifacts.
- Verification: Run default fast suite after each merge and full lint/typecheck before final handoff.

### Milestone 4: Fixture-Backed Report Generation

- Changes: Verify `python -m nlp_stock_prediction run --date 2026-05-11 --output reports/` writes
  Markdown, JSON, and audit artifacts from deterministic fixtures/offline providers.
- Files likely affected: CLI orchestration, report rendering, fixtures, e2e tests.
- Verification: `python -m pytest -m e2e` and direct CLI smoke run.

## Acceptance criteria

- [ ] Six lane worktrees and branches are created from `phase-1-contract-gate`.
- [ ] Each lane has tests covering its implemented behavior.
- [ ] Shared contracts are not modified without an explicit decision log entry.
- [ ] Lane handoffs list changed files, tests, verification, known gaps, and contract issues.
- [ ] Reviews are completed before integration merges.
- [ ] Lanes are merged into `codex/integration-v1` in dependency order.
- [ ] Default deterministic tests pass on the integration branch.
- [ ] `ruff check .`, `ruff format --check .`, and `mypy .` pass on the integration branch.
- [ ] Fixture-backed report generation produces Markdown, JSON, and audit artifacts.
- [ ] Documentation is updated for behavior, configuration, commands, or report structure changes.

## Verification commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
```

## Decision log

- 2026-05-11-14-16: Created one integration worktree plus six lane worktrees from
  `phase-1-contract-gate` to preserve the frozen contract base and avoid cross-lane working tree
  conflicts.
- 2026-05-11-14-16: Adopted the roadmap lane split A-F without changing ownership boundaries,
  because it already maps cleanly to provider/evidence, extraction, analysis, report, and reliability
  responsibilities.

## Progress log

- 2026-05-11-14-16: Confirmed `main` is clean and points at `phase-1-contract-gate`; created
  `codex/integration-v1` and six lane branches/worktrees from that tag.
- 2026-05-11-14-18: Created shared Phase 2 venv at
  `C:\Users\adams\projects\nlp-stock-prediction-phase2-venv` because the shell PATH pointed at an
  unrelated project venv. Baseline integration checks passed: `python -m pytest` reported
  166 passed and 3 expected skips; `ruff check .`, `ruff format --check .`, and `mypy .` passed.
- 2026-05-11-14-28: Lane C handoff received on `codex/lane-c-extraction` at commit `4bcb742`.
  Coordinator review found no contract drift or ownership issue. Independent checks passed:
  focused Lane C tests reported 9 passed and 1 expected skip; `ruff check .`,
  `ruff format --check .`, and `mypy .` passed. Lane C is ready to merge after lanes A and B.
- 2026-05-11-14-34: Lane A handoff received on `codex/lane-a-reddit-evidence` at commit
  `15319fa`. Coordinator review found no contract drift or ownership issue. Independent checks
  passed: focused Lane A tests reported 14 passed; `ruff check .`, `ruff format --check .`, and
  `mypy .` passed. Lane A is ready to merge first.
- 2026-05-11-14-35: Lane F handoff received on `codex/lane-f-reliability-ci` at commit `f787ea8`.
  Coordinator review found no contract drift or ownership issue. Independent checks passed:
  focused Lane F tests reported 15 passed and 2 expected skips; `ruff check .`,
  `ruff format --check .`, and `mypy .` passed. Lane F is ready to merge after lanes A-E.
- 2026-05-11-14-39: Lane D handoff received on `codex/lane-d-analysis-scoring` at commit
  `0c584cf`. Coordinator review found no contract drift or ownership issue. Independent checks
  passed: focused Lane D tests reported 9 passed; `ruff check .`, `ruff format --check .`, and
  `mypy .` passed. Lane D is ready to merge after Lane C.
- 2026-05-11-14-43: Lane B handoff received on `codex/lane-b-providers` at commit `16795f2`.
  Coordinator review found no contract drift, but found a FRED mapping exception path that could
  escape instead of returning `ProviderResult` warnings. Sent Lane B back for a targeted patch and
  test before integration.
- 2026-05-11-14-46: Lane E handoff received on `codex/lane-e-report-cli` at commit `1cc1907`.
  Coordinator review found no contract drift or ownership issue. Independent checks passed:
  focused CLI/report tests reported 28 passed; e2e selection reported 2 passed and 1 expected
  legacy placeholder skip; `ruff check .`, `ruff format --check .`, and `mypy .` passed. Lane E is
  ready to merge after Lane D.
