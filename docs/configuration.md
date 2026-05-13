# Configuration

This document describes local configuration conventions for the agentic prediction research rebuild.

## Runtime

Use Python 3.12 and the repository's editable install.

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
python -m nlp_stock_prediction research --date 2026-05-12 --output reports/ --offline
```

Run the optional Phase 2 real-Codex smoke after installing the MCP extra:

```sh
python -m pip install -e ".[dev,codex-smoke]"
NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE=1 python scripts/run_phase2_codex_smoke.py --date 2026-05-13 --output reports/phase2-codex-smoke --symbol TSLA
```

The smoke command launches `codex --search` against the local
`python -m nlp_stock_prediction.codex_mcp` server. It may use live web search, but it writes only
ignored local artifacts. On the current Windows Codex CLI, the runner uses `danger-full-access`
because stdio MCP tool calls are cancelled under `workspace-write`; the MCP service still enforces
write roots and the runner fails if tracked files change. Each smoke run uses a date/symbol-specific
ignored SQLite database under `data/` so stale evidence cannot satisfy a later run.

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
X_BEARER_TOKEN
REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET
NEWS_* provider keys
MARKET_DATA_* provider keys
LIVE_PROVIDER_USER_AGENT
NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS
NLP_STOCK_PREDICTION_ALLOW_LIVE_API_TESTS
NLP_STOCK_PREDICTION_ALLOW_LIVE_SCRAPING_TESTS
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

The target universe is retail-accessible instruments, including:

- stocks;
- ETFs;
- crypto;
- currency exposure through available instruments;
- commodity exposure through available instruments;
- futures context where available;
- user watchlists;
- web/social/news-discovered instruments.

Availability changes over time. Store tradability evidence and provider source instead of assuming a
symbol is always accessible.
