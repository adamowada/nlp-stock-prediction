# Lane E Report CLI

## Goal

Implement the Phase 2 Lane E surface so `python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline` produces a deterministic daily report bundle with `report.md`, `report.json`, and audit artifacts while preserving the frozen report contracts.

## Non-goals

- Do not modify `src/nlp_stock_prediction/contracts/`.
- Do not implement live provider, extraction, analysis, or scoring integrations owned by other lanes.
- Do not add new third-party dependencies.
- Do not introduce brokerage execution or real-money trading actions.

## Context

Lane E owns CLI command wiring, Markdown and JSON rendering, audit artifact writing, and fixture-backed end-to-end report generation. The Phase 1 branch starts from tag `phase-1-contract-gate` and currently exits with code `3` for `run`. The reporting implementation must build against frozen contracts and deterministic offline data because other Phase 2 lane implementations are not visible in this worktree.

Relevant files:

- `src/nlp_stock_prediction/cli.py`
- `src/nlp_stock_prediction/pipeline.py`
- `src/nlp_stock_prediction/reporting/`
- `tests/`
- `docs/contracts.md`
- `docs/testing-plan.md`

## Milestones

### Milestone 1: TDD Coverage

- Changes: Add renderer, audit, and CLI e2e tests for deterministic offline generation.
- Files likely affected: `tests/test_lane_e_report_rendering.py`, `tests/test_lane_e_cli_e2e.py`.
- Verification: Targeted tests fail before implementation and pass after implementation.

### Milestone 2: Report Construction And Rendering

- Changes: Add deterministic report builder, Markdown renderer, JSON renderer, and audit writer.
- Files likely affected: `src/nlp_stock_prediction/pipeline.py`, `src/nlp_stock_prediction/reporting/`.
- Verification: JSON validates as `DailyReport`; Markdown includes required header, six ticker sections, final strategies/no-trade, disclaimer, and audit artifacts.

### Milestone 3: CLI Wiring

- Changes: Replace contract-gate exit for `run` with report generation and clear stderr/stdout behavior.
- Files likely affected: `src/nlp_stock_prediction/cli.py`, `README.md` if usage/status changes.
- Verification: `python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline` exits 0 and writes expected files.

### Milestone 4: Verification And Handoff

- Changes: Format/lint/typecheck fixes within Lane E ownership, local commit if coherent.
- Files likely affected: implementation and tests only.
- Verification: Run requested commands as practical and record results in the final handoff.

## Acceptance criteria

- [x] Behavior change is implemented.
- [x] Relevant tests are added or updated.
- [x] Existing tests continue to pass.
- [x] Lint/typecheck/build pass where applicable.
- [x] Documentation is updated if behavior or usage changes.
- [x] No frozen contract files are modified.
- [x] CLI writes `report.md`, `report.json`, and `audit/` artifacts for offline deterministic runs.

## Verification commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
python -m pytest -m e2e
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
```

## Decision log

- 2026-05-11-14-19: Keep Lane E deterministic by generating a complete offline fixture report in code, with audit artifacts exposing raw snapshots, normalized evidence, extracted strategies, scoring inputs, and final reports. This preserves integration points for other lanes without adding dependencies or touching contracts.

## Progress log

- 2026-05-11-14-19: Read AGENTS, roadmap, testing plan, contracts, worktree runbook, CLI, report contracts, and Phase 1 tests. Confirmed branch `codex/lane-e-report-cli` starts at `phase-1-contract-gate` and worktree status is clean.
- 2026-05-11-14-32: Added Lane E renderer/e2e tests, implemented deterministic offline report generation, updated CLI wiring and README, verified no frozen contract diffs, and ran the requested deterministic suites successfully.
