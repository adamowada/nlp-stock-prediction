# Phase 1 Contract Test Harness

Status: complete. This is a historical Phase 1 plan; Phase 2 integration and Phase 3 local V1
acceptance have since been completed.

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
- Verification: docs align with the implemented test gate and, at the time, Phase 2 remains a
  separate next step.

## Acceptance Criteria

- [x] Import/export tests cover the public `nlp_stock_prediction.contracts` namespace.
- [x] Schema tests cover key invariants for evidence, extraction, analysis, recommendations, reports,
      provider health, fixture manifests, provenance completeness, and cross-object integrity.
- [x] CLI tests cover help, valid run parsing, invalid dates/capital, offline/fixture/cache options,
      and Phase 1's intentional run exit code.
- [x] Provider contract tests use deterministic fake providers and verify graceful failure envelopes.
- [x] Report-shape tests verify exactly six ticker sections, no-trade summaries, disclaimer fields,
      provider health, data freshness, audit manifest serialization, recommendation linkage, and a
      minimal Markdown section outline.
- [x] Default tests do not require network access or credentials.
- [x] `python -m pytest`, `ruff check .`, `ruff format --check .`, `mypy .`, and canonical CLI checks
      pass.
- [x] Documentation stated Phase 1 was complete and Phase 2 implementation worktrees were the next
      milestone at the time of Phase 1 completion.

## Verification Commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/  # expected exit code 3
```

## Decision Log

- 2026-05-11-13-16: Phase 1 will use fixture-backed fake providers and schema-shaped sample reports
  rather than concrete live adapters, because Phase 1 freezes and verifies public contracts without
  starting Phase 2 implementation lanes.
- 2026-05-11-13-16: The six subagents are assigned to disjoint contract-harness files instead of
  product implementation worktrees, preserving the roadmap rule that Phase 2 lanes start only after
  the contract gate is complete.
- 2026-05-11-13-23: Phase 1 includes a small single-threaded contract revision for discovered
  guardrail gaps: `None` is no longer coerced into strings/tickers, evidence reference spans must be
  monotonic, score breakdowns require components, and v1 disclaimers must keep educational,
  not-financial-advice, and no-auto-trading flags enabled.
- 2026-05-11-13-42: Phase 1 owns cross-object contract integrity that later lanes rely on:
  qualified candidates must pass risk/score gates, report recommendation IDs must link to matching
  candidates, provider/fixture envelopes must be coherent, and external provenance must carry an
  auditable source trail. Lane E still owns rendering and file writing.

## Progress Log

- 2026-05-11-13-16: Reviewed Phase 0 contracts, roadmap, testing plan, and runbook; started the Phase
  1 execution plan.
- 2026-05-11-13-17: Committed ACP `689c3fe` with the Phase 1 execution plan.
- 2026-05-11-13-21: Integrated six subagent slices for import, schema, provider, fixture, CLI, and
  report contract harness coverage.
- 2026-05-11-13-22: Committed ACP `298e9a4` with Phase 1 tests and contract guardrail fixes after
  `python -m pytest` passed with 153 tests.
- 2026-05-11-13-23: Updated README, contracts docs, roadmap, and worktree runbook to mark Phase 1
  complete and Phase 2 ready.
- 2026-05-11-13-24: Verified the final tree with the default suite, non-live suite, ruff, format
  check, mypy, CLI help, and the expected contract-gate `run` exit code.
- 2026-05-11-13-42: Addressed deep review findings with stricter CLI parsing, provenance, provider,
  fixture, recommendation, report-linkage, Markdown-outline, network-guard, and marker-placeholder
  tests.
