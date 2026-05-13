# Phase 2 Lane F Reliability CI

Status: complete. This is a historical Phase 2 lane plan.

## Goal

Implement Lane F reliability, live-test gating, CI, and provider guardrails without changing
frozen public contracts. The lane should provide reusable helpers for provider retries,
rate-limit/timeouts/upstream failures, graceful degradation result envelopes, provider health
messages, official API versus public scraping provenance checks, opt-in live smoke scaffolding, and
report guardrails.

## Non-goals

- Do not modify `src/nlp_stock_prediction/contracts/`.
- Do not implement concrete provider adapters owned by Lane A or Lane B.
- Do not implement report rendering or CLI orchestration owned by Lane E.
- Do not add new third-party dependencies.
- Do not make default tests require network, live credentials, or provider quota.

## Context

- Branch: `codex/lane-f-reliability-ci`.
- Starting ref: `phase-1-contract-gate` / `5b3d6b5a7e6f3b0ca230eb8a59bd9d8ce40b1a80`.
- Provider contracts already define `ProviderResult`, `ProviderHealth`, `ProviderWarning`,
  `ProviderStatus`, `WarningCode`, and `RetrievalMethod`.
- Default tests block socket access in `tests/conftest.py` unless a test is marked `live_api` or
  `live_scraping` and `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS=1`.
- Lane F should add helpers under `src/nlp_stock_prediction/reliability/` and shared provider
  modules, plus focused tests and CI workflow wiring.

## Milestones

### Milestone 1: Reliability Helpers

- Changes:
  - Add retry/backoff policy objects with deterministic sleeper/clock injection.
  - Map timeouts, rate limits, upstream failures, auth/config failures, malformed responses, and
    no-data cases to explicit provider warnings and result envelopes.
  - Provide provider health and warning message helpers.
- Files likely affected:
  - `src/nlp_stock_prediction/reliability/`
  - `tests/test_lane_f_reliability.py`
- Verification:
  - `python -m pytest tests/test_lane_f_reliability.py`

### Milestone 2: Provider Guardrails

- Changes:
  - Add provenance helper to distinguish official API data from public scraping fallback data.
- Files likely affected:
  - `src/nlp_stock_prediction/providers/scraping.py`
  - `tests/test_lane_f_reliability.py`
- Verification:
  - `python -m pytest tests/test_lane_f_reliability.py`

### Milestone 3: Live Smoke And CI Wiring

- Changes:
  - Add deterministic live smoke scaffolding that skips unless explicitly opted in.
  - Add CI workflow running default deterministic tests, lint, format check, and mypy.
  - Add scheduled/manual live workflow commands that remain opt-in by marker.
- Files likely affected:
  - `tests/test_lane_f_live_smoke.py`
  - `.github/workflows/ci.yml`
- Verification:
  - `python -m pytest -m "not live_api and not live_scraping"`
  - `ruff check .`
  - `ruff format --check .`
  - `mypy .`

## Acceptance criteria

- [x] Behavior change is implemented.
- [x] Relevant tests are added or updated.
- [x] Existing tests continue to pass.
- [x] Lint/typecheck/build pass where applicable.
- [x] Documentation is updated if behavior or usage changes.
- [x] Frozen contracts remain unchanged.
- [x] Live tests are marked and skipped unless explicitly opted in.

## Verification commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
```

## Decision log

- 2026-05-11-14-19: Keep Lane F helpers outside frozen contract modules and adapt through existing
  `ProviderResult`, `ProviderWarning`, and `ProviderHealth` models so other lanes can reuse the
  behavior without creating a contract revision.
- 2026-05-11-14-27: Live smoke scaffolding uses stdlib `urllib` only. The SEC official API smoke
  requires both `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS=1` and a configured live user agent; the
  public scraping smoke requires both live opt-in and an explicit scrape URL.

## Progress log

- 2026-05-11-14-19: Read AGENTS, roadmap, testing plan, frozen contracts, and worktree runbook.
  Confirmed branch `codex/lane-f-reliability-ci` starts at `phase-1-contract-gate`.
- 2026-05-11-14-27: Added reliability retry/degradation helpers, provider guardrails, live smoke
  tests, and CI workflow. Verified focused Lane F tests, default pytest suite, ruff check, ruff
  format check, and mypy with the shared Phase 2 venv.
