# Phase 3 Integration And Live Smoke

Status: complete. This is the historical Phase 3 local V1 acceptance plan; later follow-on work on
`feature/release-v1` added opt-in `--source-mode scrape --live-providers` evidence collection. Any
older notes below that describe live-provider report orchestration as outside V1 refer to the Phase
3 acceptance boundary at that time, not to the current branch behavior.

## Goal

Complete the V1 local CLI application so it can generate an auditable daily stock opportunity report
from deterministic fixtures by default, preserve evidence and provider provenance throughout the
report bundle, and expose opt-in live smoke checks for provider/API/scraping edges when credentials
and network access are configured.

End of Phase 3 means the CLI is complete for local V1 use: it writes Markdown, JSON, and audit
artifacts; handles outcomes where nothing qualifies and degraded-provider outcomes clearly; and
documents the remaining live-provider requirements.

## Non-goals

- Do not make default tests depend on credentials, internet access, live LLM quota, or provider
  availability.
- Do not treat Reddit, X/Twitter, news, or other public discussion as fact without attribution.
- Do not weaken evidence, provenance, risk-gate, or audit requirements to make reports
  easier to generate.
- Do not broaden Phase 3 into new product surfaces beyond the local V1 CLI without an explicit plan
  update.

## Context

- Original Phase 3 branch: `feature/integration-and-hardening`; current release integration branch:
  `feature/release-v1`.
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
  - Current docs agree that Phase 2 is integrated and Phase 3 has entered integration hardening.
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
    and live-provider configuration.
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
  - Live API and live scraping smoke checks are opt-in through explicit environment variables, and
    the live LLM smoke gate stays explicit/reserved until a live adapter and credential contract are
    enabled.
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
  - Days with no qualified opportunities are represented cleanly.
  - Provider warnings and stale/missing data are visible in report output.

### Stage 5: Risk, Reliability, And Failure Drills

- Changes: deliberately exercise malformed, missing, stale, conflicting, unsupported, and sarcastic
  input scenarios.
- Files likely affected: provider fixtures, reliability tests, extraction and scoring tests, e2e tests,
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
  - Source-of-truth docs agree on Stage 5 status and the active failure-drill matrix.

### Stage 6: Final V1 Acceptance Gate

- Changes: run and record the full deterministic gate, plus opt-in live gates when credentials are
  available.
- Files likely affected: active Phase 3 plan, README, docs, tests, final report artifacts used for
  validation.
- Done when these pass from a clean checkout:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
python -m pytest -m e2e
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
- [x] CLI configuration and live/offline behavior are hardened for V1 use.
- [x] Opt-in live smoke checks have documented credential requirements and actionable skips/failures.
- [x] Stage 3 live SEC API and configured public scraping smoke paths have narrow assertions and
      documented environment variables.
- [x] End-to-end report QA confirms evidence, provenance, warnings, and confidence inputs are
      preserved.
- [x] Failure drills cover all Stage 5 scenarios listed above.
- [x] Final V1 acceptance gate passes from a clean checkout.

## Verification commands

Use the repository virtual environment if the ambient shell points at another project:

```sh
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -m "not live_api and not live_scraping"
.\.venv\Scripts\python.exe -m pytest -m e2e
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
  opt-in live smoke coverage and documented provider configuration.
- 2026-05-11-15-50: Stage 3 keeps live LLM smoke as an explicit reserved gate because the current
  V1 CLI has fixture-backed LLM validation but no live LLM adapter or credential contract.
- 2026-05-11-15-50: Live smoke tests skip only when the global live opt-in is absent; once live
  checks are requested, missing per-check environment variables fail with actionable messages.
- 2026-05-11-16-00: Stage 4 report JSON includes `evidence_sources` so every cited evidence ID can
  resolve to normalized evidence provenance, provider metadata, raw snapshot IDs, and freshness.
- 2026-05-11-16-00: Markdown report recommendations should show the same practical QA inputs as
  JSON: catalysts, risks, assumptions, contradictions, confidence inputs, risk controls, score
  components, penalties, evidence IDs, and score input IDs.
- 2026-05-11-16-10: Stage 5 treats high sarcasm/joke risk and conflicting source evidence as
  extraction/cluster warnings that add score penalties and failed gates, keeping affected setups
  watch-only instead of silently qualified.
- 2026-05-11-16-25: Stage 6 final acceptance kept live-provider report orchestration outside the
  Phase 3 local V1 boundary; configured SEC API and public scraping smoke checks passed as opt-in
  gates, while live LLM smoke remains a reserved skip until a live adapter and credential contract
  exist.

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
- 2026-05-11-15-26: Started Stage 1 deterministic integration baseline. Read-only audits found that
  the pipeline-level report test covered most artifact requirements, while the subprocess CLI smoke
  needed stronger report-contract and audit-manifest assertions. Also found a few historical
  phase/lane plans whose old "next step" wording could be mistaken for current status.
- 2026-05-11-15-26: Strengthened `tests/test_lane_e_cli_e2e.py` so deterministic e2e coverage checks
  exact audit file coverage, validates `report.json` and `audit-manifest.json`, verifies manifest
  artifact IDs/types/paths/record counts/checksums, confirms final report references, checks core
  raw/evidence/extraction/analysis/scoring payload semantics, and applies report/audit validation
  to the actual subprocess CLI run.
- 2026-05-11-15-26: Completed Stage 1 doc drift cleanup by marking older Phase 0, Phase 1, Lane E,
  and Lane D plans as historical where their original "next step" wording could conflict with the
  current Phase 3 source of truth. No `.env` or API key blocker was encountered for Stage 1.
- 2026-05-11-15-26: Stage 1 verification passed with the repo venv: full pytest reported
  235 passed and 3 opt-in live skips; non-live pytest reported 235 passed and 3 deselected; e2e
  reported 3 passed; `ruff check .`, `ruff format --check .`, `mypy .`, CLI help, live marker
  selections, and direct offline CLI smoke all passed. The offline smoke wrote Markdown, JSON, and
  audit artifacts to a temporary directory that was removed after verification.
- 2026-05-11-15-35: Started Stage 2 CLI/configuration hardening. Read-only audits found sparse
  `run --help` text, undocumented env/.env conventions, missing Alpha Vantage credential-absence
  coverage, a non-offline example in the roadmap, and older Phase 2 lane plans without historical
  status headers.
- 2026-05-11-15-35: Hardened `run --help` for date/output/capital/risk-profile/fixture/cache/offline
  behavior and live-provider configuration, added `docs/configuration.md` and `.env.example`,
  documented live-smoke and provider credential conventions without secrets, clarified that
  `--fixture-dir` and `--cache-dir` are metadata/reserved in current offline runs, added Alpha
  Vantage missing-credential tests, and aligned drift in README, contracts, testing, roadmap, AGENTS,
  live LLM skip text, and historical Phase 2 lane plans. No blocker requiring user-supplied
  credentials was encountered.
- 2026-05-11-15-36: Stage 2 verification passed with the repo venv: full pytest reported
  239 passed and 3 opt-in live skips; non-live pytest reported 239 passed and 3 deselected; e2e
  reported 3 passed; `ruff check .`, `ruff format --check .`, `mypy .`, top-level CLI help,
  `run --help`, live marker selections, and direct offline CLI smoke all passed. The offline smoke
  wrote Markdown, JSON, and audit artifacts to a temporary directory that was removed after
  verification.
- 2026-05-11-15-50: Started Stage 3 live smoke wiring. Parallel read-only audits found LLM scaffold
  wording drift, missing configured scraping path documentation, broad live assertions, and testing
  docs that overstated current live-provider coverage.
- 2026-05-11-15-50: Tightened Stage 3 live smoke behavior: SEC live API now validates JSON record
  shape, configured public scraping now requires expected literal text, live network failures use
  actionable messages, missing per-check environment variables fail after global opt-in, live LLM
  remains an explicit reserved gate, and reliability tests cover missing-credential health
  envelopes.
- 2026-05-11-15-50: Confirmed successful configured live paths with the repo venv and network
  access: `pytest -m live_api tests/test_lane_f_live_smoke.py` passed against the SEC company
  tickers endpoint with a non-secret User-Agent, and the live scraping marker path passed against
  `https://example.com/` with expected text `Example Domain`.
- 2026-05-11-15-51: Stage 3 verification passed with the repo venv: full pytest reported
  244 passed and 3 opt-in live skips; non-live pytest reported 244 passed and 3 deselected; e2e
  reported 3 passed; `ruff check .`, `ruff format --check .`, `mypy .`, top-level CLI help,
  `run --help`, configured live SEC API smoke, configured live scraping smoke, and direct offline
  CLI smoke all passed. The offline smoke wrote Markdown, JSON, and audit artifacts to a temporary
  workspace directory that was removed after verification.
- 2026-05-11-16-00: Started Stage 4 report quality pass. Parallel read-only audits found report JSON
  evidence-reference provenance gaps, CLI coverage gaps for runs where nothing qualifies, loose per-ticker Markdown section
  assertions, missing/degraded provider visibility under-test, and doc drift that still described
  Stage 3 as active.
- 2026-05-11-16-00: Added `DailyReport.evidence_sources` with contract validation that cited
  evidence IDs resolve when the source map is present; expanded offline fixture report JSON with
  normalized evidence provenance; made degraded provider visibility include stale market data and an
  unconfigured supplemental SEC provider; expanded Markdown candidate detail; added scoped
  per-ticker Markdown assertions and CLI coverage for runs where nothing qualifies.
- 2026-05-11-16-01: Stage 4 verification passed with the repo venv: full pytest reported
  247 passed and 3 opt-in live skips; non-live pytest reported 247 passed and 3 deselected; e2e
  reported 4 passed; `ruff check .`, `ruff format --check .`, `mypy .`, top-level CLI help,
  `run --help`, live marker selections, configured live SEC API smoke, configured live scraping
  smoke, and direct offline CLI smoke all passed. The offline smoke wrote Markdown, JSON, and audit
  artifacts to a temporary workspace directory that was removed after verification.
- 2026-05-11-16-10: Started Stage 5 risk, reliability, and failure drills. Parallel read-only
  audits found weak adapter-level rate-limit/unavailable and empty-data drills, no explicit stale
  Alpha Vantage market drill, conflicting-analysis coverage without conflicting source-evidence
  behavior, weak high-sarcasm behavior, and source-of-truth docs still describing Stage 5 as
  pending.
- 2026-05-11-16-12: Added concrete provider-adapter drills for empty news payloads, 429
  rate-limit envelopes, 503 upstream-unavailable envelopes, and stale Alpha Vantage candles; added
  unsupported-instrument scoring coverage; added cluster warnings plus scoring penalties/gates for
  high sarcasm/joke risk and conflicting source evidence; aligned README, roadmap, contracts,
  testing docs, AGENTS, and this plan. No blocker requiring credentials or `.env` values was
  encountered.
- 2026-05-11-16-14: Stage 5 verification passed with the repo venv: full pytest reported
  256 passed and 3 opt-in live skips; non-live pytest reported 256 passed and 3 deselected; e2e
  reported 4 passed and 255 deselected; `ruff check .`, `ruff format --check .`, `mypy .`,
  top-level CLI help, `run --help`, and direct offline CLI smoke all passed. The offline smoke wrote
  Markdown, JSON, and audit artifacts to a temporary directory that was removed after verification.
- 2026-05-11-16-25: Started Stage 6 final V1 acceptance gate from a clean branch. Read-only audits
  found source-of-truth docs still saying acceptance was pending, the active plan missing the
  explicit `pytest -m e2e` command, and the live-smoke CI job missing
  `NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT`.
- 2026-05-11-16-28: Completed Stage 6 doc and CI alignment: marked Phase 3 local V1 acceptance
  complete in README, roadmap, contract docs, historical plans, and this plan; added e2e to the
  Stage 6 gate; and wired the live-scraping expected-text secret into CI. No `.env`, API key, or
  credential blocker was encountered. The first live SEC retry received HTTP 403 with a generic
  non-contact User-Agent, then passed with a contact-style non-secret User-Agent.
- 2026-05-11-16-29: Stage 6 verification passed with the repo venv: full pytest reported
  256 passed and 3 opt-in live skips; non-live pytest reported 256 passed and 3 deselected; e2e
  reported 4 passed and 255 deselected; `ruff check .`, `ruff format --check .`, `mypy .`,
  top-level CLI help, `run --help`, configured live API reported 1 passed, 1 reserved live-LLM skip,
  and 257 deselected; configured live scraping reported 1 passed and 258 deselected; and direct
  offline CLI smoke wrote Markdown, JSON, and audit artifacts to a temporary directory that was
  removed after verification.
- 2026-05-12-00-00: Follow-on work after this Phase 3 plan added fixture-backed scrape mode and
  explicit live-provider scrape evidence collection on `feature/release-v1`. Current active status
  is tracked in `plans/scraping-ml-agent-analysis.md`.
