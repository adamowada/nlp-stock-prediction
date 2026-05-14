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

Generate a guarded live-provider report:

```sh
python -m nlp_stock_prediction research --date 2026-05-12 --symbol TSLA --output reports/ --live
```

Reports are written under `<output>/<YYYY-MM-DD>/<symbol-slug>/`.

## Report Data Modes

Report assembly records a machine-checkable `report_data_mode` in run metadata, report
`command_args`, audit manifest `command_args`, tool-run inputs, and report artifact metadata.

Implemented modes:

- `offline_fixture`: the `research --offline` Phase 4 path. It uses deterministic fixture
  providers and is allowed only when the caller explicitly requests offline mode.
- `dummy_smoke`: the legacy deterministic dummy orchestration path. It is structural validation only
  and refuses non-offline configs.
- `codex_smoke`: the optional Codex smoke path that may include live Codex search evidence but still
  uses smoke-only structural tools.
- `live`: the guarded `research --live` Phase 4 path. It uses live provider adapters and public
  source adapters only, records missing credentials or upstream failures as tool/provider warnings,
  and refuses stored fixture, dummy, or smoke inputs. If a live run has no admissible stored evidence
  or candidates, the report renders structured insufficient evidence rather than falling back to
  fixtures or dummy data.

Direct non-offline pipeline calls still fail unless `source_mode="live"` or `live_providers=True` is
set, so callers cannot accidentally route live requests to fixture or dummy data.

## Report Assembly Source Of Truth

Phase 5 report assembly reads from the stored research SQLite run graph and persisted artifacts. The
renderer uses stored evidence, prediction candidates, candidate-evidence links, candidate-artifact
links, tool runs, artifact paths, and artifact hashes as the report source of truth. It does not
synthesize replacement candidates or fixture fallback data during final report rendering.

Candidate claims are emitted only when their required stored evidence and artifacts are available and
valid. Missing linked evidence, missing required artifacts, hash mismatches, or malformed typed
artifacts exclude the affected candidate and surface as structured insufficient evidence when no
candidate remains usable. Audit manifests preserve artifact ids, paths, hashes, validation status,
and whether an artifact was required for assembly.

Report source references are candidate-specific where the run graph supplies the relationship:
source evidence references point to the candidates that cite them, artifact references point to the
candidates linked to those artifacts, and provider-health references preserve partial, empty, failed,
or assembly-level failures. Material claim traces cover candidate thesis, baseline context, and
prediction-quality evaluation claims.

## Markdown Product Reports

Markdown reports are the human-facing companion to the JSON payload. They render report metadata,
provider health, universe resolution, instrument identity and availability, observed source evidence,
report-authored analysis, baseline context, prediction scenarios, uncertainty, dissent, change
triggers, prior-outcome reviews, source references, the evidence ledger, and audit artifacts.

Observed source claims are labeled separately from report-authored scenario analysis and labeled
inference. Candidate sections preserve evidence-for and evidence-against references, uncertainty
drivers, dissenting evidence, evaluation quality metadata, and prior-outcome review links without
using recommendation, position sizing, or trade-instruction framing. Structured insufficient-evidence
reports render their blocking reasons, providers, evidence, artifacts, and metadata instead of
inventing a fallback scenario.

## JSON Reports And Runtime Index

JSON reports keep the top-level `DailyReport` payload shape and are validated against
`json-report-contract.v1` before they are written. The contract maps every material Markdown product
section to stable JSON fields, including report metadata, data freshness, provider health and
warnings, instrument sections, prediction scenarios or structured insufficient evidence,
prior-outcome reviews, material claim traces, source references, the evidence ledger, and audit
artifacts.

Final Markdown, JSON, and audit-manifest files are also recorded in the research database
`report_artifact_index`. The index stores paths, hashes, artifact schema version, report schema
version, report date, instrument identity, report data mode, tool run id, and source run timestamps.
It does not store report bodies or raw provider payloads; those remain file-backed artifacts under
ignored output directories.

## Prior Outcome Review

Report rendering loads the latest indexed prior JSON report for the same instrument, or an explicit
prior report artifact recorded in run metadata. Prior report files are read from disk and checked
against the stored hash before they can source a `PriorOutcomeReview`. If no prior report exists,
the candidate receives an explicit `not_available` review. Missing, malformed, or stale prior
artifacts are represented as unavailable or stale limitations rather than synthesized history.

When a prior report is usable, the current report records follow-up evidence as the outcome context,
links the prior JSON artifact in the current audit manifest, and adds change triggers for supporting
or contradictory evidence, outcome data, baseline changes, and provider refreshes where applicable.

Run the optional Phase 4 real-Codex smoke after installing the MCP extra:

```sh
python -m pip install -e ".[dev,codex-smoke]"
NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE=1 python scripts/run_phase2_codex_smoke.py --date 2026-05-13 --output reports/phase4-codex-smoke --symbol TSLA
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
NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY
NLP_STOCK_PREDICTION_FRED_API_KEY
NLP_STOCK_PREDICTION_X_BEARER_TOKEN
NLP_STOCK_PREDICTION_LIVE_USER_AGENT
NLP_STOCK_PREDICTION_SEC_USER_AGENT
NLP_STOCK_PREDICTION_SEC_CIK_MAP
NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT
NEWS_* provider keys
MARKET_DATA_* provider keys
NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS
NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL
NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT
```

`NLP_STOCK_PREDICTION_SEC_CIK_MAP` accepts comma-separated `SYMBOL=CIK` entries for SEC EDGAR
lookups. The live path also honors `ALPHA_VANTAGE_API_KEY`, `MARKET_DATA_ALPHA_VANTAGE_API_KEY`,
`FRED_API_KEY`, and `X_BEARER_TOKEN` as fallback names. Missing optional credentials are surfaced in
the run graph and final report instead of being replaced with fixture data.

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

The implemented Phase 3 universe layer is contract and storage infrastructure. The live `research`
path materializes requested symbols as live-mode instrument identities, then relies on provider
artifacts and warnings to establish actual data availability. The legacy-named optional Phase 4
Codex smoke runner remains separate from the live-provider CLI path.

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
