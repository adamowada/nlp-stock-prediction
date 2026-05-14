# nlp-stock-prediction

## Short Description

`nlp-stock-prediction` is a local Python CLI for generating evidence-backed market prediction
research reports. It is built for research workflows that need provenance, reproducible artifacts,
and explicit uncertainty across retail-accessible instruments such as stocks, ETFs, crypto,
currencies, commodities, futures context, and related proxies.

This is not a trading application. It does not place trades, size positions, or tell users what to
buy or sell. Its output is a Markdown and JSON research report that separates source evidence from
analysis, preserves dissenting context, and explains what would change the prediction.

The project is in active development. The current implementation includes a deterministic offline
research command, SQLite-backed planning and research storage, Phase 3 instrument-universe
contracts/storage, fixture-backed Phase 4 research tools, and an opt-in Codex MCP smoke path.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
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

Verify the CLI:

```sh
python -m nlp_stock_prediction --help
```

## Quick Start

Generate a deterministic offline research report:

```sh
python -m nlp_stock_prediction research --date 2026-05-12 --symbol TSLA --output reports/ --offline
```

The offline command writes local report artifacts under
`reports/<YYYY-MM-DD>/<symbol-slug>/` without using network providers or live credentials.

## Usage Examples

Show available CLI commands:

```sh
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction research --help
```

Generate an offline report for a specific date:

```sh
python -m nlp_stock_prediction research \
  --date 2026-05-12 \
  --symbol TSLA \
  --output reports/ \
  --offline
```

Run the optional Phase 4 Codex smoke workflow:

```sh
python -m pip install -e ".[dev,codex-smoke]"
NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE=1 python scripts/run_phase2_codex_smoke.py \
  --date 2026-05-13 \
  --output reports/phase4-codex-smoke \
  --symbol TSLA
```

The legacy-named smoke script starts a local MCP server, exposes the Phase 4 fixture-backed research
tool suite to Codex, and writes ignored artifacts under local output directories.

## Current Instrument Universe

Phase 3 adds the durable instrument-universe layer that reports and storage can share. The
implemented contracts and SQLite helpers represent broad retail-accessible research targets:
stocks, ETFs, crypto pairs, currency and commodity exposure, futures context, funds, indexes, and
related proxies. Instrument records carry canonical IDs, symbols, display names, asset classes,
venues, aliases, provider-specific IDs, related instruments, data availability, and
tradability/access evidence.

Resolution is explicit. A query can be `resolved`, `ambiguous`, `unsupported`, or `unavailable`;
ambiguous symbols such as `AI` must keep multiple matches instead of silently choosing one.
Watchlists are represented as named collections of instrument queries in contracts and as
instrument-linked lists in SQLite.

This layer is fixture-backed today. The default offline report, optional Phase 4 Codex smoke path,
and first-class Phase 4 universe discovery tool can write instrument artifacts and registry rows
without live universe providers. Broader live universe-discovery adapters remain future hardening.

## Current Phase 4 Tool Suite

The fixture-backed Phase 4 suite now includes universe discovery, market data, technical packages,
social evidence, news/catalysts, fundamentals, sector/macro context, prediction-quality evaluation,
conservative candidate synthesis, and final Markdown/JSON/audit report rendering. These tools write
typed artifacts and SQLite run-graph rows while preserving provider/source provenance. Live providers
remain opt-in and incremental; deterministic fixtures are the default QA and offline path.

## Configuration

Local configuration is documented in [docs/configuration.md](docs/configuration.md).

Environment variables are optional for the default offline workflow. Keep secrets in exported shell
variables or ignored `.env` files.

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | No | Required only for workflows that call OpenAI-backed tooling. |
| `NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE` | No | Set to `1` to opt in to the real Codex smoke path. |
| `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` | No | Token for X/Twitter-backed provider experiments. |
| `NLP_STOCK_PREDICTION_LIVE_USER_AGENT` | No | Contact User-Agent for opt-in live provider smoke tests. |
| `NLP_STOCK_PREDICTION_SEC_USER_AGENT` | No | Contact User-Agent for SEC EDGAR requests. |
| `NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT` | No | User agent for public HTML scraping providers. |
| `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS` | No | Enables opt-in live test groups when combined with marked tests. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL` | No | URL used by the opt-in live scraping smoke test. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT` | No | Text expected in the opt-in live scraping smoke response. |

The project uses two SQLite databases:

- `plans/planning.sqlite3`: tracked planning state for plans, decisions, progress, and related
  metadata.
- `data/prediction-research.sqlite3`: ignored runtime research state for tool runs, artifacts,
  evidence, and prediction candidates.

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
                              Research runtime, Phase 4 tools, MCP smoke support, and report assembly
src/nlp_stock_prediction/providers/
                              Provider adapters and provider-facing contracts
src/nlp_stock_prediction/reporting/
                              Markdown, JSON, and audit rendering
src/nlp_stock_prediction/storage/
                              SQLite schema initialization and storage helpers
tests/                        Unit, contract, provider, storage, and orchestration tests
tests/fixtures/tools/universe_discovery/
                              Small Phase 3 instrument-universe fixture contracts
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
python -m pytest -m "not live_api and not live_scraping"
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
python -m pytest -m "not live_api and not live_scraping"
python -m pytest -m "not live_api and not live_scraping and not codex_smoke"
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
acceptance criteria. Phase 3 fixture scenarios live under
`tests/fixtures/tools/universe_discovery/`; the Phase 4 production gate lives in
`tests/test_phase4_tool_suite_e2e.py`.
