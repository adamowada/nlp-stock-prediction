# nlp-stock-prediction

A TDD-first Python project for generating a daily, evidence-grounded stock opportunity report for retail traders.

The app will discover the six tickers highlighted by r/wallstreetbets' daily Devvit ticker card, collect recent public discussion and news, extract discussed trading strategies with source evidence, and combine that signal with technical, fundamental, sector, and macro analysis. The final output is intended to be a Markdown report plus structured JSON for auditability.

## Current status

Phase 0 contract settlement is implemented as a minimal Python package with frozen public contracts. Phase 1 contract test harness is next, and parallel implementation worktrees/subagents remain blocked until the full contract gate is complete. See `AGENTS.md` for project conventions, `docs/contracts.md` for the frozen Phase 0 contracts, `docs/worktree-runbook.md` for the later parallel worktree procedure, and `PLANS.md` for the execution-plan format used for larger Codex tasks.

## Intended workflow

1. Discover the daily r/wallstreetbets ticker card and extract six tickers.
2. Retrieve recent WSB posts and daily-thread comments for each ticker.
3. Use evidence-grounded NLP/LLM extraction to identify discussed strategies.
4. Gather public market, news, fundamental, sector, and macro data.
5. Generate one daily report with ticker sections and any qualified trading ideas.

## Development

Expected commands once the Python package is scaffolded:

```sh
python -m pytest
ruff check .
ruff format .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/
```

The canonical CLI invocation is the Python module form, `python -m nlp_stock_prediction`. If a console script is added later, it should remain a thin alias for that module command and the docs should be updated together.

During Phase 0, `run` validates the command contract and exits with code `3` because report generation is not implemented yet.
