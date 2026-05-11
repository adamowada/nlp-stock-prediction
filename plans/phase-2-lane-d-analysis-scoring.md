# Phase 2 Lane D Analysis Scoring

## Goal

Implement deterministic technical analysis, fundamental/sector/macro context synthesis, retail risk
controls, recommendation scoring, confidence inputs, and contradiction penalties using the frozen
Phase 1 contracts.

## Non-goals

- Do not modify `src/nlp_stock_prediction/contracts/`.
- Do not make network calls or add live provider integrations.
- Do not render reports or wire the full CLI orchestration.
- Do not add third-party dependencies.

## Context

Lane D owns analysis and scoring modules that consume provider snapshots, evidence-backed strategy
clusters, and social/news evidence summaries. The frozen contracts already define provider facts,
analysis outputs, score breakdowns, risk assessments, and trade candidates. This lane should add
private helpers under `src/nlp_stock_prediction/analysis/` and
`src/nlp_stock_prediction/scoring/` while keeping default tests deterministic and network-free.

## Milestones

### Milestone 1: Technical Analysis

- Changes: compute SMA, RSI, MACD, support/resistance, relative volume, ATR-style volatility, gap,
  and candlestick summaries from daily bars.
- Files likely affected: `src/nlp_stock_prediction/analysis/technical.py`,
  `tests/test_lane_d_analysis.py`.
- Verification: focused unit tests and contract model validation.

### Milestone 2: Fundamental, Sector, And Macro Context

- Changes: synthesize valuation, profitability, growth, balance sheet, earnings, filings, peer or
  ETF proxy comparisons, and macro horizon support/conflict.
- Files likely affected: `src/nlp_stock_prediction/analysis/fundamental.py`,
  `src/nlp_stock_prediction/analysis/sector.py`, `src/nlp_stock_prediction/analysis/macro.py`,
  `tests/test_lane_d_analysis.py`.
- Verification: focused unit tests for direct peer comparison and ETF fallback behavior.

### Milestone 3: Risk And Scoring

- Changes: score Reddit, social/news catalysts, technicals, fundamentals, sector, macro,
  liquidity/risk suitability, contradiction penalties, no-trade outcomes, and qualified candidates.
- Files likely affected: `src/nlp_stock_prediction/scoring/risk.py`,
  `src/nlp_stock_prediction/scoring/recommendations.py`, `tests/test_lane_d_scoring.py`.
- Verification: focused unit tests for qualified, conflicting, and no-trade scenarios.

## Acceptance criteria

- [x] Behavior change is implemented.
- [x] Relevant tests are added or updated.
- [x] Existing tests continue to pass.
- [x] Lint/typecheck/build pass where applicable.
- [x] Documentation is updated if behavior or usage changes.

## Verification commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
```

## Decision log

- 2026-05-11-14-19: Keep Lane D as deterministic pure functions that consume frozen provider and
  strategy contracts. This avoids provider and report-lane coupling while preserving auditable
  contract outputs.
- 2026-05-11-14-19: Encode risk posture as default-deny gates for margin, naked short options, and
  non-defined-risk option exposure. With account capital, max loss must fit one percent of capital;
  without capital, sizing remains percentage-only.

## Progress log

- 2026-05-11-14-19: Read AGENTS, roadmap, testing plan, frozen contracts, worktree runbook, and
  contract modules. Next step is to add lane D tests before implementation.
- 2026-05-11-14-30: Added deterministic lane D analysis and scoring tests, implemented analysis,
  risk, and scoring modules without contract edits, and verified the requested default suites with
  the shared Phase 2 virtual environment.
