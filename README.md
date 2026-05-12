# nlp-stock-prediction

A TDD-first Python project for generating a daily, evidence-grounded stock opportunity report for retail traders.

The app will discover the six tickers highlighted by r/wallstreetbets' daily Devvit ticker card, collect recent public discussion and news, always query X API recent search for the top 50 relevant stock-news/social posts for each ticker when live provider orchestration is enabled, extract discussed trading strategies with source evidence, and combine that signal with technical, fundamental, sector, and macro analysis. The final output is intended to be a Markdown report plus structured JSON and audit artifacts for traceability.

## Current status

Phase 3 is complete for the local V1 CLI on `feature/release-v1`. The current CLI generates a deterministic offline report bundle with Markdown, JSON, and audit artifacts, and it also has an experimental fixture-backed `--source-mode scrape` path that wires the Reddit public-page, AP News, Candlecharts feasibility, and X recent-search adapters into provider health and audit artifacts. Live API and scraping checks remain opt-in, and live LLM smoke remains reserved until a live adapter and credential contract exist. See `AGENTS.md` for project conventions, `docs/contracts.md` for the contract baseline, and `PLANS.md` for the execution-plan format used for larger Codex tasks.

## Intended workflow

1. Discover the daily r/wallstreetbets ticker card and extract six tickers.
2. Retrieve recent WSB posts and daily-thread comments for each ticker.
3. Use evidence-grounded NLP/LLM extraction to identify discussed strategies.
4. Gather public market, news, X top relevant ticker discussion, fundamental, sector, and macro data.
5. Generate one daily report with ticker sections, qualified trading ideas or a clear no-trade summary, provider warnings, freshness, evidence, and disclaimers.

## Development

Canonical local commands:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape --live-providers --cache-dir cache/live
```

The canonical CLI invocation is the Python module form, `python -m nlp_stock_prediction`. If a console script is added later, it should remain a thin alias for that module command and the docs should be updated together.

The offline run writes `reports/YYYY-MM-DD/report.md`, `reports/YYYY-MM-DD/report.json`, and `reports/YYYY-MM-DD/audit/` using deterministic fixture data. `--source-mode scrape` writes the same report bundle plus `audit/provider-results.json`, using deterministic provider fixtures and degraded-provider probes rather than live network calls. Add `--live-providers` to `--source-mode scrape` for explicit local live calls to Reddit public pages, AP News public HTML, Candlecharts feasibility, and X recent search. The CLI automatically loads a local `.env` file without overriding exported shell variables. The default test harness blocks network access; live tests require explicit opt-in environment variables and, for provider-specific checks, credentials or configured URLs.

Configuration details, live-smoke environment variables, and `.env` handling are documented in `docs/configuration.md`. Real credentials belong in environment variables or ignored local `.env` files, never in committed files.
