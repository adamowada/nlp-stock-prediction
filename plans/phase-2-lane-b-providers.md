# Phase 2 Lane B Providers

Status: complete. This is a historical Phase 2 lane plan.

## Goal

Implement deterministic, fixture-testable provider adapters for X/social, public news, market data,
fundamentals, SEC EDGAR supplemental data, FRED macro series, and provider response caching while
preserving the frozen Phase 1 contract models.

## Non-goals

- Do not modify `src/nlp_stock_prediction/contracts/`.
- Do not implement analysis, scoring, report rendering, or CLI orchestration owned by other lanes.
- Do not add paid-provider assumptions.
- Do not require network access or credentials for the default test suite.
- Do not add new third-party runtime dependencies.

## Context

Phase 2 Lane B starts from `phase-1-contract-gate` (`5b3d6b5`) on branch
`codex/lane-b-providers`. The frozen provider protocols return `ProviderResult[T]` envelopes with
`ProviderHealth` and `ProviderWarning` semantics. Concrete adapters should live under
`src/nlp_stock_prediction/providers/` and normalize data into existing contract types only.

Provider failures such as missing credentials, unauthenticated requests, stale data, no data,
malformed responses, and upstream outages must be explicit result statuses and warnings rather than
exceptions. Tests should use fixtures or injected transports and remain deterministic.

## Milestones

### Milestone 1: Request, Result, And Cache Helpers

- Changes: Add private helpers for UTC timestamps, provider warnings/health, request cache keys,
  JSON HTTP request handling, stale checks, and date/ticker/source cache paths.
- Files likely affected: `src/nlp_stock_prediction/providers/_base.py`,
  `tests/test_lane_b_provider_helpers.py`.
- Verification: Lane B helper tests pass without network.

### Milestone 2: Social And News Adapters

- Changes: Add X recent-search query construction and mapping of social posts to `SourceEvidence`.
  Add configured public news-provider mapping for articles/headlines.
- Files likely affected: `src/nlp_stock_prediction/providers/social.py`,
  `src/nlp_stock_prediction/providers/news.py`, provider fixtures/tests.
- Verification: Fixture-backed tests cover `$TSLA lang:en`, unconfigured credentials, no data, and
  malformed payloads.

### Milestone 3: Market, Fundamentals, SEC, And FRED Adapters

- Changes: Add daily candle and company-overview mapping, SEC EDGAR company facts and filings
  supplemental fundamentals, and FRED macro-series mapping.
- Files likely affected: `src/nlp_stock_prediction/providers/market.py`,
  `src/nlp_stock_prediction/providers/sec_edgar.py`,
  `src/nlp_stock_prediction/providers/fred.py`, provider fixtures/tests.
- Verification: Fixture-backed tests cover normalized candles, overview metrics, SEC metrics and
  filings, FRED series, stale data warnings, missing credentials, and cache reuse.

## Acceptance Criteria

- [x] Behavior change is implemented.
- [x] Relevant tests are added or updated.
- [x] Existing tests continue to pass.
- [x] Lint/typecheck/build pass where applicable.
- [x] Documentation is updated if behavior or usage changes.

## Verification Commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
```

## Decision Log

- 2026-05-11-00-00: Keep provider implementations private to `providers/` and depend only on
  frozen contract models to avoid cross-lane contract churn.
- 2026-05-11-00-00: Use injected stdlib-compatible JSON transports for deterministic unit tests;
  live network checks remain opt-in and outside the default suite.

## Progress Log

- 2026-05-11-00-00: Read AGENTS, roadmap, testing, contract, and worktree runbook docs. Confirmed
  clean `codex/lane-b-providers` worktree at `phase-1-contract-gate`.
- 2026-05-11-14-31: Added provider helper/cache layer, X/social, public news, Alpha Vantage
  market/fundamentals, SEC EDGAR, and FRED adapters with fixture-backed Lane B tests.
- 2026-05-11-14-31: Verified focused Lane B tests, full pytest, deterministic non-live pytest,
  ruff check, ruff format check, and mypy using the shared Phase 2 virtual environment.
