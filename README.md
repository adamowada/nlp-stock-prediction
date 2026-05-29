# nlp-stock-prediction

## Short Description

`nlp-stock-prediction` is a local Python CLI for generating evidence-backed market prediction
research reports. It is built for research workflows that need provenance, reproducible artifacts,
and explicit uncertainty across retail-accessible instruments such as stocks, ETFs, crypto,
currencies, commodities, futures context, and related proxies.

This is not a trading application. It does not place trades, size positions, or tell users what to
buy or sell. Its output is a Markdown and JSON research report that separates source evidence from
analysis, preserves dissenting context, and explains what would change the prediction.

The project is in active development. The current implementation includes deterministic offline and
guarded live-provider research commands, SQLite-backed planning and research storage, Instrument-Universe Stage
instrument-universe contracts/storage, Research Stage research tools, Report Stage Markdown/JSON/audit
prediction reports, and an opt-in Codex MCP smoke path.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [User Guide](docs/user-guide.md)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)

## Installation

Requirements:

- Python 3.14.5
- Git
- A local shell capable of running Python virtual environments

Clone the repository and create a virtual environment:

```sh
git clone git@github.com:adamowada/nlp-stock-prediction.git
cd nlp-stock-prediction
python -m venv .venv
```

Activate the environment:

```sh
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the project in editable mode with development dependencies:

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For in-app Codex Agent Chat, also install the MCP extra and ensure the Codex CLI is on `PATH`:

```sh
python -m pip install -e ".[dev,codex-smoke]"
codex --version
```

Verify the CLI:

```sh
python -m nlp_stock_prediction --help
```

## Quick Start

Launch the persistent terminal app:

```sh
python main.py
```

The app opens a menu for research, report browsing, evaluation, settings, and Codex-agent chat.
Research defaults to today's date, live providers, and `reports/`; it asks for a symbol unless you
remember one in Settings. Use Settings or the Research advanced prompt when you want offline fixtures
or a date override.
Agent Chat can read selected report artifacts and use project MCP tools; web search outside those
tools is treated as unaudited chat context, not report evidence.

Generate a deterministic offline research report:

```sh
python -m nlp_stock_prediction research --date 2026-05-12 --symbol TSLA --output reports/ --offline
```

The offline command writes local report artifacts under
`reports/<YYYY-MM-DD>/<symbol-slug>/` without using network providers or live credentials.

In an interactive terminal, launch the Rich terminal UI for a guided report run:

```sh
python -m nlp_stock_prediction tui
```

You can also pass the same research options to run the terminal UI non-interactively:

```sh
python -m nlp_stock_prediction tui --date 2026-05-12 --symbol TSLA --output reports/ --offline
```

Generate a guarded live-provider report:

```sh
python -m nlp_stock_prediction research --date 2026-05-12 --symbol TSLA --output reports/ --live
```

The live command uses live providers and public-source adapters only. Stock/ETF market data uses
Alpha Vantage when configured and otherwise uses the public Yahoo Finance chart endpoint. Missing
credentials, upstream failures, stale data, or empty providers are recorded in the report instead of
falling back to fixtures or dummy data.

## Usage Examples

Show available CLI commands:

```sh
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction app
python -m nlp_stock_prediction research --help
python -m nlp_stock_prediction tui --help
```

Generate an offline report for a specific date:

```sh
python -m nlp_stock_prediction research \
  --date 2026-05-12 \
  --symbol TSLA \
  --output reports/ \
  --offline
```

Generate a live-provider report for a specific date:

```sh
python -m nlp_stock_prediction research \
  --date 2026-05-12 \
  --symbol TSLA \
  --output reports/ \
  --live
```

Run the optional Research Stage Codex smoke workflow:

```sh
python -m pip install -e ".[dev,codex-smoke]"
NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE=1 python scripts/run_codex_smoke.py \
  --date 2026-05-13 \
  --output reports/research-codex-smoke \
  --symbol TSLA
```

The legacy-named smoke script prepares a fresh ignored SQLite database, starts a local MCP server,
exposes the Research Stage research tool suite to Codex, and writes ignored artifacts under local output
directories.

## Current Instrument Universe

Instrument-Universe Stage adds the durable instrument-universe layer that reports and storage can share. The
implemented contracts and SQLite helpers represent broad retail-accessible research targets:
stocks, ETFs, crypto pairs, currency and commodity exposure, futures context, funds, indexes, and
related proxies. Instrument records carry canonical IDs, symbols, display names, asset classes,
venues, aliases, provider-specific IDs, related instruments, data availability, and
tradability/access evidence.

Resolution is explicit. A query can be `resolved`, `ambiguous`, `unsupported`, or `unavailable`;
ambiguous symbols such as `AI` must keep multiple matches instead of silently choosing one.
Watchlists are represented as named collections of instrument queries in contracts and as
instrument-linked lists in SQLite.

The offline report path remains fixture-backed. The live report path materializes requested symbols
as live-mode instrument identities and then relies on live provider artifacts to establish data
availability. Broader live universe-discovery adapters remain future hardening.

## Current Research Stage Tool Suite

The fixture-backed Research Stage suite now includes universe discovery, market data, technical packages,
social evidence, news/catalysts, fundamentals, sector/macro context, prediction-quality evaluation,
conservative candidate synthesis, and final Markdown/JSON/audit report rendering. These tools write
typed artifacts and SQLite run-graph rows while preserving provider/source provenance. Live providers
remain opt-in and incremental; deterministic fixtures are the default QA and offline path.

## Current Evaluation Stage Evaluation And Calibration

Evaluation Stage can evaluate stored prediction candidates against later outcome evidence, then persist
signal-family ablations, walk-forward folds, and calibration summaries from those point-in-time
outcome evaluations. These tools read from the research SQLite run graph and write audit artifacts
plus `calibration_runs`/`calibration_slices`; they do not use fixture or dummy fallbacks.

Rendered reports now preserve Evaluation Stage outputs when they exist for the same run. Stored outcome
evaluations become prior-outcome review entries in Markdown/JSON, and calibration artifacts remain
separate audit artifacts referenced by the report rather than being collapsed into trading-style
performance claims.

## Current Reliability Stage Evaluation Hardening

Reliability Stage now adds typed freshness and aging reviews to point-in-time evaluation targets. Target
freezing records evidence aging and artifact freshness under `reliability_freshness`, normalizes
date-only market artifact metadata deterministically, and keeps aged-out, stale, missing,
malformed, hash-mismatched, provider-replaced, and future/lookahead artifact states auditable.
Standalone `artifact_freshness_review` and `evidence_aging_summary` audit artifacts can be written
and indexed without recomputing or mutating calibration artifacts.

Live outcome materialization uses only live-mode market artifacts, rejects same-day daily closes
that were not observable at the cutoff, and marks started attempts as failed if a late validation or
persistence step raises. SQLite outcome rows reject impossible observed/non-observed shapes, report
artifact index rows must align with their tool run, and JSON metadata writes reject non-finite
numbers. Provider hardening keeps X API limits within the real provider contract, routes Reddit
public-page scraping through the shared HTML retry/cache path, and preserves timeout/cache-failure
classification without substituting fixture or dummy data.

The canonical public interface is public:

```sh
python -m nlp_stock_prediction evaluation --database data/prediction-research.sqlite3 inspect --run-id <run-id>
python -m nlp_stock_prediction evaluation --database data/prediction-research.sqlite3 outcome-summary --run-id <run-id> --artifact-root reports/<run-id>/audit
python -m nlp_stock_prediction evaluation --database data/prediction-research.sqlite3 calibration --run-id <run-id> --cohort-id <cohort-id> --as-of 2026-05-22T00:00:00+00:00 --artifact-root reports/<run-id>/audit
```

`evaluation` subcommands cover `inspect`, `materialize-outcome`, `load-outcomes`,
`outcome-summary`, `stale-artifacts`, `evidence-aging`, `source-reliability`,
`provider-playbook`, `calibration`, `walk-forward`, `ablation`, and `calibration-drift`. The
commands require an explicit database and run ID, and write commands require `--artifact-root` so
audit writes stay under the repository write policy.

## Current Report Stage Prediction Reports

The report product writes Markdown, JSON, and audit-manifest artifacts from the stored run graph.
Reports preserve evidence for and against, dissenting evidence, uncertainty, baseline context,
signal artifacts, prior-outcome review context, source references, material claim traces, provider
health, and audit hashes. When the honest result is no call, reports render structured
insufficient-evidence details instead of fabricating a scenario. Provider failures, stale data,
malformed artifacts, ambiguous instruments, unsupported instruments, and contradictory evidence
remain visible in the report and audit surfaces.

## Configuration

Local configuration is documented in [docs/configuration.md](docs/configuration.md).

Environment variables are optional for the default offline workflow. Keep secrets in exported shell
variables or ignored `.env` files.

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | No | Required only for workflows that call OpenAI-backed tooling. |
| `NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE` | No | Set to `1` to opt in to the real Codex smoke path. |
| `NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY` | No | Optional Alpha Vantage key for live market data and fundamentals; live stock/ETF market data can still use credential-free public chart data when this is absent. |
| `NLP_STOCK_PREDICTION_FRED_API_KEY` | No | Optional FRED key for live macro context. |
| `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` | No | Token for X/Twitter-backed provider experiments. |
| `NLP_STOCK_PREDICTION_LIVE_USER_AGENT` | No | Contact User-Agent for opt-in live provider smoke tests. |
| `NLP_STOCK_PREDICTION_SEC_USER_AGENT` | No | Contact User-Agent for SEC EDGAR requests. |
| `NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT` | No | User agent for public HTML scraping providers. |
| `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS` | No | Enables opt-in live test groups when combined with marked tests. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL` | No | URL used by the opt-in live scraping smoke test. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT` | No | Text expected in the opt-in live scraping smoke response. |

SEC EDGAR ticker-to-CIK resolution is automatic through SEC's public
`company_tickers_exchange.json` dataset; unresolved or malformed lookups surface as provider
failures instead of requiring local per-symbol environment mappings.

The project uses two SQLite databases:

- `plans/planning.sqlite3`: tracked planning state for plans, decisions, progress, and related
  metadata.
- `data/prediction-research.sqlite3`: ignored default research state for service/tool runs that do
  not choose a per-run database.
- `data/research-{offline|live}-runtime-{date}-{symbol_hash}-{output_hash}.sqlite3`: ignored CLI
  research databases created by `python -m nlp_stock_prediction research ...`. These isolate report
  runs by date, symbol, output path, and live/offline mode so one invocation cannot silently reuse
  another invocation's stored evidence.

Generated reports, provider cache files, research databases, and raw artifacts are local working
state by default and are ignored unless explicitly promoted as small test fixtures.

## Project Structure

```text
src/nlp_stock_prediction/     Application package
src/nlp_stock_prediction/cli.py
                              CLI entry point and command wiring
src/nlp_stock_prediction/contracts/
                              Pydantic contracts for instruments, evidence, reports, and tools
src/nlp_stock_prediction/orchestration/
                              Research runtime, Research Stage tools, MCP smoke support, and report assembly
src/nlp_stock_prediction/providers/
                              Provider adapters and provider-facing contracts
src/nlp_stock_prediction/reporting/
                              Markdown, JSON, and audit rendering
src/nlp_stock_prediction/storage/
                              SQLite schema initialization and storage helpers
tests/                        Unit, contract, provider, storage, and orchestration tests
tests/fixtures/tools/universe_discovery/
                              Small Instrument-Universe Stage instrument-universe fixture contracts
scripts/                      Maintenance and smoke-test scripts
docs/                         Architecture, contracts, configuration, roadmap, and testing docs
plans/planning.sqlite3        Tracked planning database
data/                         Ignored runtime research databases and generated local state
reports/                      Ignored generated report output
artifacts/                    Ignored generated tool and research artifacts
```

## Development

Use the repository virtual environment for local work:

```sh
. .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the common development checks:

```sh
python -m pytest -m "not live_api and not live_scraping and not codex_smoke and not llm"
ruff check .
ruff format --check .
mypy .
```

Format code with Ruff:

```sh
ruff format .
```

Use `python -m nlp_stock_prediction` as the canonical CLI invocation until the project introduces a
console script.

Source-of-truth documentation lives in `README.md`, `AGENTS.md`, `PLANS.md`, and `docs/`. Active
planning state belongs in `plans/planning.sqlite3`; do not create ad hoc Markdown plans for routine
development work.

## Testing

The default test suite is deterministic and offline:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping and not codex_smoke and not llm"
```

Run focused live groups only when credentials, network access, and explicit opt-in environment
variables are available:

```sh
python -m pytest -m live_api
python -m pytest -m live_scraping
```

The optional Codex smoke tests require the `codex-smoke` extra and
`NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE=1`:

```sh
python -m pip install -e ".[dev,codex-smoke]"
NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE=1 python -m pytest -m codex_smoke
```

See [docs/testing-plan.md](docs/testing-plan.md) for test layering, negative-case expectations, and
acceptance criteria. Instrument-Universe Stage fixture scenarios live under
`tests/fixtures/tools/universe_discovery/`; the Research Stage production gate lives in
`tests/test_research_tool_suite_e2e.py`.
