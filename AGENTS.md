# AGENTS.md

## Project Overview

This project is a TDD-first Python CLI application being rebuilt into a Codex-led prediction research
assistant.

The product is not a trading app. It must not place trades, size positions, or frame output as
instructions to buy or sell. Its job is to generate evidence-backed prediction reports. Reports may
describe bullish, bearish, neutral, volatile, uncertain, or insufficient-evidence scenarios, but every
claim must be grounded in sources, tool artifacts, assumptions, uncertainty, and baseline context.

The redesigned app centers on a Codex agent. The agent's two responsibilities are:

- call independent research tools and inspect their artifacts;
- synthesize the output into Markdown/JSON prediction reports.

The app should support any retail-accessible instrument class when data is available: stocks, ETFs,
crypto, currencies, commodities, futures context, and related proxy instruments. Tradability must be
represented with source provenance and availability constraints, not hardcoded assumptions.

Fine-tuning TimesFM is retired. Raw TimesFM may remain as one cheap technical signal inside a broader
technical package, but it must not become the product center of gravity.

See [docs/architecture.md](docs/architecture.md), [docs/contracts.md](docs/contracts.md),
[docs/roadmap.md](docs/roadmap.md), and [docs/testing-plan.md](docs/testing-plan.md).

## Common Commands

Use the repository's configured commands:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-12 --output reports/ --offline
```

The current `run` command is legacy scaffolding during the rebuild. Do not treat old TimesFM HPO or
fine-tuning commands as the preferred workflow.

Use `python -m nlp_stock_prediction` as the canonical CLI invocation until a console script is
introduced.

## Coding Rules

- Prefer simple, typed Python modules with small interfaces and Pydantic models at API boundaries.
- Keep source adapters, tool execution, artifact writing, prediction scoring, and report rendering
  separate.
- Preserve provenance for all external data: provider, query, URL/permalink, retrieved timestamp,
  published timestamp when available, raw identifier, freshness, and extraction confidence.
- Separate observed claims from the app's analysis. Reddit, X/Twitter, forums, news, filings, and web
  pages are evidence sources, not automatically true statements.
- Make internet and live-provider calls through explicit adapters or Codex browsing/search paths that
  write evidence records.
- Keep secrets out of the repo. Load API keys from environment variables or ignored `.env` files.
- Handle rate limits, partial failures, stale data, missing providers, and contradictory evidence
  explicitly.
- Store large raw payloads and generated artifacts on disk. Store planning records in the tracked
  planning SQLite database, and store runtime research metadata in the ignored research SQLite
  database.
- Avoid data visualizations unless the user explicitly asks to revisit them.
- Use clear names and concise comments only where the code's intent is not obvious.
- Perform your own code review on the code you write and fix all issues.
- ACP (add commit push) to the current branch once coding and the code review is complete.

## Testing Rules

- Follow [docs/testing-plan.md](docs/testing-plan.md).
- Write tests before or alongside behavior changes.
- Keep the default fast suite deterministic, offline, and independent of optional GPU/model packages.
- Use fixtures and mocks for provider/tool contracts.
- Mark live API and live scraping tests explicitly and require opt-in credentials/network access.
- Include negative tests for stale evidence, unavailable providers, malformed pages, unsupported
  instruments, ambiguous symbols, contradictory evidence, insufficient evidence, and source claims that
  cannot be attributed.
- Any prediction scoring change must test no-evidence, conflicting-evidence, and evidence-supported
  outcomes.
- SQLite schema changes must include migration/idempotency tests and round-trip tests.

## Planning Rules

- `AGENTS.md` and `PLANS.md` are immutable by default. Edit them only when the user specifically asks.
- Active planning state belongs in `plans/planning.sqlite3`, which is tracked in git. Use the typed
  storage interface described in `PLANS.md`.
- Use in-thread plans and task checklists for short-lived turn coordination, but do not add new
  Markdown plans unless the user explicitly asks.
- Source-of-truth docs remain Markdown: README, AGENTS, PLANS, and files under `docs/`.
- Keep docs aligned with implemented behavior. If a doc describes target behavior, label it clearly as
  target architecture or roadmap rather than available functionality.
- Record major product, architecture, provider, and risk-policy decisions in structured planning state
  once available.

## Definition Of Done

- The requested behavior is implemented and covered by focused tests.
- Existing relevant tests pass, and relevant lint/typecheck/build commands pass where applicable.
- Generated prediction reports preserve evidence, provider metadata, tool artifacts, confidence
  inputs, uncertainty, and dissenting context.
- External provider failures degrade gracefully and are visible in artifacts and reports.
- SQLite writes are deterministic, idempotent where appropriate, and covered by tests.
- Documentation is updated when commands, configuration, behavior, contracts, or report structure
  changes.
- No secrets, raw credentials, unrelated generated artifacts, or stale planning files are committed.
