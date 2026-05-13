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

## Current Commands

The legacy CLI remains available while the rebuild proceeds:

```sh
python -m nlp_stock_prediction run --date 2026-05-12 --output reports/ --offline
```

The target command shape is:

```sh
python -m nlp_stock_prediction research --objective daily-prediction-report --universe retail-tradable --output reports/
```

Do not rely on the target command until it is implemented and tested.

## Local Storage

Use these conventions:

```text
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

The SQLite foundation is implemented in `nlp_stock_prediction.storage`. Initialize a local database
with:

```python
from pathlib import Path

from nlp_stock_prediction.storage import initialize_database

store = initialize_database(Path("data/prediction-research.sqlite3"))
```

The initial schema covers instruments, research runs, tool runs, artifacts, source queries, evidence
items, prediction candidates, and structured planning state.

## Environment Variables

Keep secrets out of git. Load credentials from the environment or ignored `.env` files.

Expected variable families:

```text
OPENAI_API_KEY
X_BEARER_TOKEN
REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET
NEWS_* provider keys
MARKET_DATA_* provider keys
LIVE_PROVIDER_USER_AGENT
ALLOW_LIVE_API_TESTS
ALLOW_LIVE_SCRAPING_TESTS
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
