# Phase 3 Integration And Live Smoke

## Goal

Complete the V1 local CLI application so it can generate an auditable daily stock opportunity report
from deterministic fixtures by default, preserve evidence and provider provenance throughout the
report bundle, and expose opt-in live smoke checks for provider/API/scraping edges when credentials
and network access are configured.

End of Phase 3 means the CLI is complete for local V1 use: it writes Markdown, JSON, and audit
artifacts; handles no-trade and degraded-provider outcomes clearly; and documents the remaining
live-provider requirements. It does not mean automated brokerage execution, production hosting, or
unqualified investment advice.

## Non-goals

- Do not add real-money brokerage execution or auto-trading.
- Do not make default tests depend on credentials, internet access, live LLM quota, or provider
  availability.
- Do not treat Reddit, X/Twitter, news, or other public discussion as fact without attribution.
- Do not weaken evidence, provenance, risk-gate, disclaimer, or audit requirements to make reports
  easier to generate.
- Do not broaden Phase 3 into new product surfaces beyond the local V1 CLI without an explicit plan
  update.

## Context

- Current branch: `feature/integration-and-hardening`.
- Phase 0 and Phase 1 established the frozen public contracts and deterministic contract harness.
- Phase 2 implementation lanes A-F have been integrated into the main application surface.
- The strongest Phase 2 source of truth is `plans/phase-2-parallel-implementation.md`, whose final
  progress log records 230 passing tests, lint/typecheck success, CLI help success, and an offline
  smoke run that wrote `report.md`, `report.json`, and audit artifacts.
- The ambient shell may point to an unrelated Python environment; use the repository virtual
  environment if needed, for example `.\.venv\Scripts\python.exe -m pytest`.
- Stage 0 does not require `.env` files or API keys. Future live checks must escalate missing
  credentials or required environment variables before treating them as blockers.

## Milestones

### Stage 0: Setup And Status Cleanup

- Changes: create this Phase 3 execution plan, align status/source-of-truth docs, remove stale
  pre-Phase-3 placeholder wording, and record the current verification baseline.
- Files likely affected: `plans/phase-3-integration-live-smoke.md`, `README.md`,
  `docs/multi-milestone-plan.md`, `docs/contracts.md`, `docs/testing-plan.md`,
  `docs/worktree-runbook.md`, `plans/phase-2-parallel-implementation.md`, and stale placeholder
  tests/messages.
- Done when:
  - This plan includes goal, non-goals, context, milestones, acceptance criteria, verification
    commands, decision log, and progress log.
  - Current docs agree that Phase 2 is integrated and Phase 3 is active.
  - The Phase 3 baseline verification is recorded.
  - Stale pre-Phase-3 "not implemented" wording is removed or narrowed to accurate live-only gaps.
  - The repo is clean after ACP.

### Stage 1: Deterministic Integration Baseline

- Changes: verify the merged app end to end without live dependencies and resolve any fixture,
  report-contract, audit, or skip-marker drift.
- Files likely affected: CLI orchestration, fixture report builders, report renderers, audit
  writing, tests, and docs.
- Done when:
  - Full default test suite passes.
  - `ruff check .`, `ruff format --check .`, and `mypy .` pass.
  - Offline CLI run creates `report.md`, `report.json`, and `audit/`.
  - Generated JSON validates against the public report contract.
  - Audit artifacts include raw snapshots, normalized evidence, extracted strategies, analysis
    contexts, scoring inputs, final report data, and the audit manifest.

### Stage 2: CLI And Configuration Hardening

- Changes: make local CLI usage, offline/live separation, env handling, and degraded-provider
  behavior explicit and ergonomic.
- Files likely affected: CLI parser/help, run configuration, provider setup docs, README, tests.
- Done when:
  - CLI help documents date, output, capital, risk profile, fixture/cache paths, offline behavior,
    and live-provider limitations.
  - Missing credentials produce structured provider warnings or documented skips instead of
    crashes.
  - Offline mode remains deterministic and network-free.
  - Optional `.env` or environment-variable usage is documented without committing secrets.
  - Non-offline behavior fails clearly until live orchestration is deliberately enabled.

### Stage 3: Live Smoke Wiring

- Changes: validate real provider edges through explicitly marked, quota-conscious smoke tests.
- Files likely affected: live smoke tests, provider adapters, reliability helpers, docs, CI
  schedule notes.
- Done when:
  - Live API, live scraping, and live LLM smoke checks are opt-in through explicit environment
    variables.
  - Each live smoke test has a narrow assertion and actionable skip/failure message.
  - Missing credentials are escalated to the user when live checks are requested.
  - Provider failures return structured health/warning output.
  - At least one successful configured live API/scraping smoke path is documented when credentials
    and network access are available.

### Stage 4: End-To-End Report Quality Pass

- Changes: review report content for usefulness, traceability, internal consistency, and small
  account risk posture.
- Files likely affected: report fixtures, renderers, scoring summaries, audit payloads, tests,
  documentation.
- Done when:
  - Markdown report includes expected sections for all six discovered tickers.
  - JSON preserves evidence IDs, provider metadata, freshness, confidence inputs, risks, and
    recommendation inputs.
  - Observed discussion is clearly separated from app analysis and recommendations.
  - Recommendations cite evidence and pass risk gates.
  - No-trade days are represented cleanly.
  - Provider warnings and stale/missing data are visible in report output.
  - Educational-only, non-advice, no-auto-trading disclaimers are present.

### Stage 5: Risk, Reliability, And Failure Drills

- Changes: deliberately exercise malformed, missing, stale, conflicting, unsupported, and sarcastic
  input scenarios.
- Files likely affected: provider fixtures, reliability tests, extraction/scoring tests, e2e tests,
  docs.
- Done when tests cover:
  - Malformed Reddit ticker-card HTML.
  - Duplicate or insufficient tickers.
  - Missing provider data.
  - Rate-limit and provider-unavailable responses.
  - Stale market or macro data.
  - Unsupported recommendations.
  - Conflicting evidence.
  - Joke/sarcasm risk.
  - No qualified strategies.

### Stage 6: Final V1 Acceptance Gate

- Changes: run and record the full deterministic gate, plus opt-in live gates when credentials are
  available.
- Files likely affected: active Phase 3 plan, README, docs, tests, final report artifacts used for
  validation.
- Done when these pass from a clean checkout:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
```

And, when credentials/network access are configured:

```sh
python -m pytest -m live_api
python -m pytest -m live_scraping
```

## Acceptance criteria

- [x] Phase 3 execution plan exists and is decision-complete.
- [x] Phase 2 integration status is reflected in current source-of-truth docs.
- [x] Stage 0 verification baseline is recorded.
- [x] Stale pre-Phase-3 placeholder wording is removed or narrowed.
- [ ] CLI configuration and live/offline behavior are hardened for V1 use.
- [ ] Opt-in live smoke checks have documented credential requirements and actionable skips.
- [ ] End-to-end report QA confirms evidence, provenance, warnings, confidence inputs, and
      disclaimers are preserved.
- [ ] Failure drills cover malformed, missing, stale, conflicting, unsupported, and no-trade
      scenarios.
- [ ] Final V1 acceptance gate passes from a clean checkout.

## Verification commands

Use the repository virtual environment if the ambient shell points at another project:

```sh
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -m "not live_api and not live_scraping"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy .
.\.venv\Scripts\python.exe -m nlp_stock_prediction --help
.\.venv\Scripts\python.exe -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
```

Live checks remain opt-in:

```sh
.\.venv\Scripts\python.exe -m pytest -m live_api
.\.venv\Scripts\python.exe -m pytest -m live_scraping
```

## Decision log

- 2026-05-11-15-15: Phase 3 will use one primary coordinator thread with parallel read-only scouts
  or narrow side tasks only when they do not create shared-contract or integration conflicts.
- 2026-05-11-15-15: Stage 0 does not require API keys or `.env` files; missing credentials become a
  blocker only when a user-requested live check cannot run without them.
- 2026-05-11-15-15: A complete Phase 3 V1 CLI means local deterministic report generation plus
  opt-in live smoke coverage and documented live limitations, not brokerage execution or production
  deployment.

## Progress log

- 2026-05-11-15-15: Started Stage 0 on `feature/integration-and-hardening`; confirmed Phase 2
  implementation is already merged and identified doc/test drift from stale pre-Phase-2 wording.
- 2026-05-11-15-18: Completed Stage 0 doc alignment: added this Phase 3 plan, refreshed README,
  roadmap, contract, testing, worktree, and AGENTS command docs, marked the Phase 2 checklist
  complete, removed stale marker placeholders, and narrowed live-only gap messages. No `.env` or API
  key blocker was encountered for Stage 0.
- 2026-05-11-15-18: Stage 0 verification passed with the repo venv: full pytest reported
  235 passed and 3 opt-in live skips; non-live pytest reported 235 passed and 3 deselected; e2e
  reported 3 passed; `ruff check .`, `ruff format --check .`, `mypy .`, CLI help, live marker
  selections, and offline CLI smoke all passed. The offline smoke wrote Markdown, JSON, and audit
  artifacts to a temporary directory that was removed after verification.
