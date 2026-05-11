# Phase 1 Contract Test Harness

## Goal

Complete the Phase 1 contract gate by adding deterministic tests for public schema behavior,
package imports, CLI command shape, provider protocol semantics, fixture manifests, and daily report
shape.

## Non-goals

- Do not implement live provider adapters, scraping, LLM calls, analysis formulas, report rendering,
  recommendation scoring, or audit writing.
- Do not start Phase 2 lane implementation worktrees from the contract base.
- Do not add real-money brokerage execution or auto-trading behavior.
- Do not require network access, API credentials, or live dependency availability for the default
  test suite.

## Context

- Phase 0 contracts live under `src/nlp_stock_prediction/contracts/`.
- The canonical CLI is `python -m nlp_stock_prediction`.
- Phase 1 should make the contract gate testable without changing the frozen contract intent.
- Parallel product lanes remain outside this plan; the six subagents are assigned only to contract
  harness workstreams.

## Milestones

### Milestone 1: Schema And Import Harness

- Changes: add comprehensive import/export and Pydantic invariant tests for public contract models.
- Files likely affected: `tests/test_phase1_import_contracts.py`,
  `tests/test_phase1_schema_contracts.py`.
- Verification: `python -m pytest -m "schema or unit"`.

### Milestone 2: Provider And Fixture Harness

- Changes: add fake protocol implementations, provider-result failure-shape tests, request
  serialization tests, and fixture manifest tests.
- Files likely affected: `tests/test_phase1_provider_contracts.py`,
  `tests/test_phase1_fixture_contracts.py`.
- Verification: `python -m pytest -m "contract or schema"`.

### Milestone 3: CLI And Report Shape Harness

- Changes: add CLI subprocess and parser tests, plus daily report JSON/Markdown shape contract tests.
- Files likely affected: `tests/test_phase1_cli_contracts.py`,
  `tests/test_phase1_report_contracts.py`.
- Verification: `python -m pytest -m "unit or schema or e2e"`.

### Milestone 4: Documentation And Gate Status

- Changes: update status docs, fixture README, and this plan's progress/decision log.
- Files likely affected: `docs/multi-milestone-plan.md`, `docs/contracts.md`,
  `tests/fixtures/README.md`, `README.md`, this plan.
- Verification: docs align with the implemented test gate and Phase 2 remains a separate next step.

## Acceptance Criteria

- [ ] Import/export tests cover the public `nlp_stock_prediction.contracts` namespace.
- [ ] Schema tests cover key invariants for evidence, extraction, analysis, recommendations, reports,
      provider health, and fixture manifests.
- [ ] CLI tests cover help, valid run parsing, invalid dates/capital, offline/fixture/cache options,
      and Phase 1's intentional run exit code.
- [ ] Provider contract tests use deterministic fake providers and verify graceful failure envelopes.
- [ ] Report-shape tests verify exactly six ticker sections, no-trade summaries, disclaimer fields,
      provider health, data freshness, and audit manifest serialization.
- [ ] Default tests do not require network access or credentials.
- [ ] `python -m pytest`, `ruff check .`, `ruff format --check .`, `mypy .`, and canonical CLI checks
      pass.
- [ ] Documentation states Phase 1 is complete and Phase 2 implementation worktrees are the next
      milestone.

## Verification Commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/
```

## Decision Log

- 2026-05-11-13-16: Phase 1 will use fixture-backed fake providers and schema-shaped sample reports
  rather than concrete live adapters, because Phase 1 freezes and verifies public contracts without
  starting Phase 2 implementation lanes.
- 2026-05-11-13-16: The six subagents are assigned to disjoint contract-harness files instead of
  product implementation worktrees, preserving the roadmap rule that Phase 2 lanes start only after
  the contract gate is complete.

## Progress Log

- 2026-05-11-13-16: Reviewed Phase 0 contracts, roadmap, testing plan, and runbook; started the Phase
  1 execution plan.
