# Configuration

This document describes local configuration conventions for the agentic prediction research rebuild.

## Runtime

Use Python 3.14.5 and the repository's editable install.

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m nlp_stock_prediction --help
```

Optional provider, model, and GPU dependencies must remain opt-in. The default test suite must not
require network access, live credentials, CUDA, or TimesFM packages.

TimesFM is an optional technical-signal adapter only. Raw inference artifacts may support technical
context; tuning and promotion workflows are outside the product workflow.

## Current Commands

Generate the deterministic offline report:

```sh
python -m nlp_stock_prediction research --date 2026-05-12 --symbol TSLA --output reports/ --offline
```

Run the optional Phase 4 real-Codex smoke after installing the MCP extra:

```sh
python -m pip install -e ".[dev,codex-smoke]"
NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE=1 python scripts/run_phase2_codex_smoke.py --date 2026-05-13 --output reports/phase2-codex-smoke --symbol TSLA
```

The smoke command launches `codex --search` against the local
`python -B -m nlp_stock_prediction.codex_mcp` server so the MCP process does not write bytecode
caches outside artifact roots. It drives the Phase 4 MCP tool suite and writes only ignored local
artifacts. On the current Windows Codex CLI, the runner uses `danger-full-access` because stdio MCP
tool calls are cancelled under `workspace-write`; the MCP service still enforces write roots and the
runner fails if tracked files or restricted ignored repo files change. Each smoke run uses a
date/symbol-specific ignored SQLite database under `data/` so stale evidence cannot satisfy a later
run.

## Local Storage

Use these conventions:

```text
plans/
  planning.sqlite3
data/
  prediction-research.sqlite3
  universes/
  market/
artifacts/
  tools/
  technical-package/
  providers/
reports/
cache/
```

SQLite should store metadata, relationships, hashes, statuses, and planning state. Large payloads and
reports should stay as files with paths recorded in SQLite.

Generated payloads are local working state by default. Keep `artifacts/`, `reports/`, provider
`cache/`, and `data/ml/` out of git unless a small, scrubbed file is deliberately promoted into
`tests/fixtures/` with a clear fixture purpose. TimesFM tuning weights and ad hoc report bundles
should be deleted or archived outside the repository rather than treated as source artifacts.

The SQLite foundation is implemented in `nlp_stock_prediction.storage`. The planning database is
`plans/planning.sqlite3` and is tracked in git. The research database is
`data/prediction-research.sqlite3` and is generated local state ignored by git. Create or verify both
with:

```python
from pathlib import Path

from nlp_stock_prediction.storage import (
    initialize_planning_database,
    initialize_research_database,
)

planning_store = initialize_planning_database(Path("plans/planning.sqlite3"))
research_store = initialize_research_database(Path("data/prediction-research.sqlite3"))
```

The planning schema covers plans, milestones, acceptance criteria, decisions, progress events, and
links. The research schema covers instruments, research runs, tool runs, artifacts, source queries,
evidence items, and prediction candidates.

## Environment Variables

Keep secrets out of git. Load credentials from the environment or ignored `.env` files.

Expected variable families:

```text
OPENAI_API_KEY
NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE
NLP_STOCK_PREDICTION_X_BEARER_TOKEN
NLP_STOCK_PREDICTION_LIVE_USER_AGENT
NLP_STOCK_PREDICTION_SEC_USER_AGENT
NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT
NEWS_* provider keys
MARKET_DATA_* provider keys
NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS
NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL
NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT
```

Provider-specific names should be documented when a provider is implemented.

## Internet Search

Codex may use internet search and browsing as part of research. Search results are evidence only when
recorded with:

- query;
- URL;
- retrieved timestamp;
- publication timestamp when available;
- extracted claim;
- extraction confidence;
- source reliability notes.

Repeatable provider integrations should still be implemented as tools when they become important to
regular reports.

## Instrument Universe

The implemented Phase 3 universe layer is contract and storage infrastructure. It does not introduce
a new CLI command or a live universe provider. The current command surface remains the offline
`research` command and the optional Phase 2 Codex smoke runner above.

The target universe is retail-accessible instruments, including:

- stocks;
- ETFs;
- crypto;
- currency exposure through available instruments;
- commodity exposure through available instruments;
- futures context where available;
- user watchlists;
- web/social/news-discovered instruments.

Current contracts support these asset classes directly: `stock`, `etf`, `crypto`, `currency`,
`commodity`, `futures`, `fund`, `index`, `proxy`, and `unknown`.

Instrument identity should be stored as a canonical instrument ID plus a normalized symbol, display
name, asset class, optional venue, aliases, provider IDs, related instruments, data availability, and
tradability/access evidence. Provider IDs should include the provider name, identifier, optional
namespace, optional URL, and provider metadata. Examples include a market-data symbol, an exchange
listing ID, a CIK, a FIGI, a crypto pair, or a fixture namespace.

Availability changes over time. Store tradability evidence and provider source instead of assuming a
symbol is always accessible. A tradability/access observation must be traceable to a source URL,
permalink, or raw identifier and should record the provider, status, retrieved timestamp, and any
access constraints. This is evidence for research availability, not permission or advice to trade.

Universe requests may include direct instrument queries and watchlists. Resolution results must be
explicitly marked as `resolved`, `ambiguous`, `unsupported`, or `unavailable`; ambiguous symbols must
retain their candidate matches until a caller supplies enough context to select one.

Fixture-backed Phase 3 scenarios live in `tests/fixtures/tools/universe_discovery/`. Runtime
universe artifacts created by local runs should stay under ignored `artifacts/`, `reports/`, or
`data/` paths unless deliberately promoted as small scrubbed fixtures.
