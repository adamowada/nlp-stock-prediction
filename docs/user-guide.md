# User Guide

This guide assumes you have already installed the project and can run:

```sh
python -m nlp_stock_prediction --help
```

Use `python -m nlp_stock_prediction` as the canonical command. The app is a local prediction
research assistant, not a trading app. It generates evidence-backed Markdown and JSON reports,
preserves provider/source provenance, and records uncertainty. It does not place trades, size
positions, or tell you what to buy or sell.

## Mental Model

The application has two main user-facing workflows:

1. `research` creates a report bundle for one instrument on one report date.
2. `evaluation` inspects and hardens stored prediction evaluation data for an existing research run.

Research runs write local files and SQLite metadata. Evaluation commands read an existing research
SQLite database and, for write commands, add audit artifacts under a concrete artifact directory.

The app is designed to make missing, stale, contradictory, or unavailable evidence visible. A report
that says there is insufficient evidence is a valid result, not a failed run.

## Before Running Commands

Run commands from the repository root unless you have a specific reason to do otherwise.

If your virtual environment is not already active, activate it first:

```sh
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

On Windows, you can also call the virtual environment interpreter directly:

```sh
.\.venv\Scripts\python.exe -m nlp_stock_prediction --help
```

## Command Map

Launch the persistent terminal app:

```sh
python main.py
```

Show the top-level command list:

```sh
python -m nlp_stock_prediction --help
```

Show research command help:

```sh
python -m nlp_stock_prediction research --help
```

Show terminal UI help:

```sh
python -m nlp_stock_prediction tui --help
```

Show evaluation command help:

```sh
python -m nlp_stock_prediction evaluation --help
```

The top-level commands are:

| Command | Use it when you want to |
| --- | --- |
| `app` | Launch the persistent menu app from the package entrypoint. |
| `research` | Generate a Markdown report, JSON report, and audit artifacts for a symbol. |
| `tui` | Launch a Rich-styled terminal workflow for guided report generation. |
| `evaluation` | Inspect or write evaluation-hardening artifacts for an existing research database and run ID. |

`python main.py` and `python -m nlp_stock_prediction app` open the same app. The app defaults
research to today's date, live mode, and `reports/`; it asks for a symbol unless one is remembered in
Settings. Report summaries are concise by default; provider, evidence, and audit details are
available from report submenus. In an interactive terminal, menu commands clear and redraw the current
screen so tables, report summaries, and command output do not accumulate as scrollback.
Agent Chat can summarize selected reports and use project MCP tools; independent web-search chat
context is not audited report evidence unless those tools persist it. While Codex is working, Agent
Chat shows a `Thinking` animation with a sanitized activity trace of observable events such as tool
use, response drafting, and retries. It does not expose private chain-of-thought; ask for a reasoning
summary when you want the rationale behind an answer.

## Quick Start: Generate Your First Report

For normal interactive use, start with the app:

```sh
python main.py
```

Choose `Research` to run live research with the current date, enter the symbol when prompted, or use
the advanced Research prompt when you need offline fixtures or a historical report date. Choose
`Reports` to view generated reports inside the app instead of browsing for files manually.

The safest first run is an offline report. Offline mode is deterministic and does not use network
providers or live credentials:

```sh
python -m nlp_stock_prediction research --date 2026-05-12 --symbol TSLA --output reports/ --offline
```

This writes:

```text
reports/2026-05-12/tsla/report.md
reports/2026-05-12/tsla/report.json
reports/2026-05-12/tsla/audit/audit-manifest.json
```

The run ID for that command is:

```text
research-2026-05-12-tsla
```

Open `report.md` first. Use `report.json` when you need a machine-readable payload. Use
`audit/audit-manifest.json` when you need to trace which artifacts and hashes support the report.

When stdout is an interactive terminal, the `research` command renders a Rich terminal dashboard
when the run completes. Captured or redirected `research` output keeps the plain one-line report
path records used by scripts. The dashboard shows the report files, provider health, prediction
scenario summary, evidence preview, and tool-run status while preserving the same
Markdown/JSON/audit files on disk.

## Launch The Rich Terminal UI

Use `tui` when you want a more app-like terminal flow. If you run it in an interactive terminal, it
prompts for the report date, output directory, and offline/live mode:

```sh
python -m nlp_stock_prediction tui
```

You can also pass the same options as `research` for a non-interactive Rich-styled run:

```sh
python -m nlp_stock_prediction tui --date 2026-05-12 --symbol TSLA --output reports/ --offline
python -m nlp_stock_prediction tui --date 2026-05-12 --symbol TSLA --output reports/ --live
```

The terminal UI is presentation only. Report contracts, evidence provenance, audit artifacts, and
SQLite metadata are the same artifacts produced by `research`. Rich styling degrades to
no-color/plain text when terminal capabilities are limited.

## Generate A Live Report

Use live mode when you want real providers and public-source adapters:

```sh
python -m nlp_stock_prediction research --date 2026-05-12 --symbol TSLA --output reports/ --live
```

Live mode does not fall back to fixtures, dummy records, or smoke data. If a provider is missing,
rate-limited, stale, malformed, or unable to support the requested instrument, that condition is
recorded in provider health, audit artifacts, and the report. The report may conclude that there is
insufficient evidence.

Stock and ETF market data can use Alpha Vantage when configured. If Alpha Vantage is not configured,
the live stock/ETF path can use the public Yahoo Finance chart endpoint. Other providers may require
environment variables.

## Research Command Reference

The command shape is:

```sh
python -m nlp_stock_prediction research \
  --date <YYYY-MM-DD> \
  --symbol <SYMBOL> \
  --output <OUTPUT_DIR> \
  --offline
```

or:

```sh
python -m nlp_stock_prediction research \
  --date <YYYY-MM-DD> \
  --symbol <SYMBOL> \
  --output <OUTPUT_DIR> \
  --live
```

Arguments:

| Argument | Required | Meaning |
| --- | --- | --- |
| `--date` | Yes | Report date in `YYYY-MM-DD` format. This is the point-in-time research date. |
| `--output` | Yes | Base output directory. Reports are written under `<output>/<YYYY-MM-DD>/<symbol-slug>/`. |
| `--symbol` | No | Instrument symbol or pair. Defaults to `TSLA`. Examples: `TSLA`, `MSFT`, `SPY`, `BTC-USD`. |
| `--offline` | One mode required | Use deterministic offline fixtures. |
| `--live` | One mode required | Use live providers and public-source adapters without fixture fallback. |
| `--fixture-dir` | No | Offline fixture root override. Use only when fixtures are not in the default repository location. |
| `--cache-dir` | No | Optional provider cache directory for live provider/cache metadata. Prefer an ignored path such as `cache/`. |

`--offline` and `--live` are mutually exclusive. You must choose one.

## Choosing Offline Or Live Mode

Use `--offline` when:

- you are learning the CLI;
- you need deterministic output;
- you want a fast local workflow;
- you are running tests or reviewing report structure.

Use `--live` when:

- you want real current provider data;
- you have configured any needed credentials;
- you are prepared for provider warnings, rate limits, partial evidence, or insufficient-evidence
  reports.

Offline reports are useful, but they are fixture-backed. Live reports are the real provider path.
Neither mode places trades or manages position sizing; reports may discuss prediction scenarios,
price levels, and strategy context when the evidence supports it.

## Instrument Symbols

The app is built for retail-accessible research targets when data is available, including stocks,
ETFs, crypto pairs, currency and commodity exposure, futures context, funds, indexes, and related
proxy instruments.

Examples:

```sh
python -m nlp_stock_prediction research --date 2026-05-12 --symbol MSFT --output reports/ --live
python -m nlp_stock_prediction research --date 2026-05-12 --symbol SPY --output reports/ --live
python -m nlp_stock_prediction research --date 2026-05-12 --symbol BTC-USD --output reports/ --live
```

Instrument resolution is explicit. Ambiguous, unsupported, unavailable, or partially available
symbols should remain visible in the report instead of being silently coerced into a different
instrument.

## Output Layout

For:

```sh
python -m nlp_stock_prediction research --date 2026-05-12 --symbol TSLA --output reports/ --live
```

the primary files are:

| Path | Purpose |
| --- | --- |
| `reports/2026-05-12/tsla/report.md` | Human-readable report. Start here. |
| `reports/2026-05-12/tsla/report.json` | Machine-readable report contract payload. |
| `reports/2026-05-12/tsla/audit/audit-manifest.json` | Audit manifest of report and supporting artifacts. |
| `reports/2026-05-12/tsla/audit/` | Tool artifacts, source reliability notes, freshness reviews, and other audit files. |
| `data/research-live-runtime-2026-05-12-<symbol-hash>-<output-hash>.sqlite3` | Ignored runtime research database for the run. |

Offline mode uses the same report layout and an offline runtime database named:

```text
data/research-offline-runtime-<date>-<symbol-hash>-<output-hash>.sqlite3
```

Generated reports, provider caches, runtime research databases, and raw artifacts are local working
state and are ignored by git by default.

## Reading The Markdown Report

The Markdown report is the human-facing research product. Important sections include:

| Section | What to look for |
| --- | --- |
| Report metadata | Run ID, schema, report date, and objective. |
| Data freshness | Whether data was current, stale, partial, or unavailable. |
| Provider health | Provider warnings, missing credentials, empty responses, rate limits, and failures. |
| Universe resolution | How the requested symbol was resolved and whether it was ambiguous or unsupported. |
| Instrument identity | Canonical instrument ID, symbol, provider IDs, tradability/access evidence, and data availability. |
| Observed evidence | Source-backed facts, claims, metrics, URLs, timestamps, and provenance. |
| Report-authored analysis | The app's synthesis based on the evidence. |
| Prediction scenarios | Bullish, bearish, neutral, volatile, uncertain, or insufficient-evidence scenarios when supported. |
| Baseline context | The baseline or reference case the scenario is being compared against. |
| Dissent and uncertainty | Evidence against the scenario, unresolved tensions, and risk drivers. |
| Change triggers | Conditions that would materially change the report's interpretation. |
| Prior outcome reviews | Previously stored outcome evaluations when available for the instrument. |
| Source references | Source, artifact, and provider references used by claims. |
| Evidence ledger | Detailed source evidence with provenance. |
| Audit artifacts | Paths and hashes for report artifacts and supporting tool outputs. |

The report separates observed source claims from report-authored analysis. A news article, filing,
social post, or web page is evidence, not automatically truth.

## Reading The JSON Report

Use `report.json` when you want stable fields for automation, comparison, or downstream review. It
contains the same product surface as the Markdown report, including:

- report metadata;
- data freshness;
- provider health and warnings;
- instrument resolutions and instrument sections;
- prediction scenarios or structured insufficient evidence;
- prior outcome reviews;
- material claim traces;
- source references;
- evidence sources;
- audit manifest references.

The JSON report is validated before it is written. If you are comparing reports over time, compare
the JSON payload rather than scraping the Markdown.

## Using The Audit Manifest

The audit manifest is the traceability layer. Use it to answer:

- Which files were produced?
- Which tool or provider produced each artifact?
- What schema version did each artifact use?
- What SHA-256 hash was recorded?
- Which artifacts were required for final report assembly?
- Which provider warnings were visible at report time?

The app stores large raw payloads and generated artifacts on disk, while SQLite stores metadata,
relationships, hashes, statuses, and run-graph links.

## Configure Live Providers

Live runs load a local `.env` file automatically without overriding variables already exported in
your shell. Keep credentials in environment variables or ignored `.env` files. Do not commit secrets.

Common variables:

| Variable | Purpose |
| --- | --- |
| `NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY` | Optional Alpha Vantage key for live market data and fundamentals. |
| `NLP_STOCK_PREDICTION_FRED_API_KEY` | Optional FRED key for macro context. |
| `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` | Optional X/Twitter bearer token for X-backed provider experiments. |
| `NLP_STOCK_PREDICTION_SEC_USER_AGENT` | Contact user agent for SEC EDGAR requests. |
| `NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT` | User agent for public HTML scraping providers. |
| `NLP_STOCK_PREDICTION_LIVE_USER_AGENT` | Contact user agent for opt-in live provider smoke tests. |
| `OPENAI_API_KEY` | Required only for workflows that call OpenAI-backed tooling. |

SEC EDGAR ticker-to-CIK resolution is automatic through SEC's public
`company_tickers_exchange.json` dataset. If that dataset cannot be fetched, parsed, or matched to
the requested ticker, the live report records a loud SEC provider failure rather than asking for a
per-symbol local mapping.

Example `.env`:

```text
NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY=your-key
NLP_STOCK_PREDICTION_FRED_API_KEY=your-key
NLP_STOCK_PREDICTION_SEC_USER_AGENT=Your Name your.email@example.com
NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT=Your Name your.email@example.com
```

See [configuration.md](configuration.md) for the full configuration reference.

## Evaluation Workflow Overview

Evaluation commands operate on persisted research data. They do not create a new research database
for you. You need:

- an existing research SQLite database;
- a run ID;
- an artifact root for commands that write audit artifacts.

Global evaluation arguments come before the subcommand:

```sh
python -m nlp_stock_prediction evaluation --database <DATABASE> <subcommand> --run-id <RUN_ID>
```

For a report generated with:

```sh
python -m nlp_stock_prediction research --date 2026-05-12 --symbol TSLA --output reports/ --live
```

the run ID is:

```text
research-2026-05-12-tsla
```

and the usual audit artifact root is:

```text
reports/2026-05-12/tsla/audit
```

To find the newest CLI runtime database on Windows PowerShell:

```powershell
Get-ChildItem data\research-*-runtime-*.sqlite3 |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
```

On macOS/Linux:

```sh
ls -t data/research-*-runtime-*.sqlite3 | head -1
```

In the examples below, set these variables to match your run.

Windows PowerShell:

```powershell
$DATABASE = "data/research-live-runtime-2026-05-12-REPLACE_WITH_HASHES.sqlite3"
$RUN_ID = "research-2026-05-12-tsla"
$AUDIT_ROOT = "reports/2026-05-12/tsla/audit"
$CANDIDATE_ID = "candidate-id-from-report-json"
```

macOS/Linux:

```sh
DATABASE="data/research-live-runtime-2026-05-12-REPLACE_WITH_HASHES.sqlite3"
RUN_ID="research-2026-05-12-tsla"
AUDIT_ROOT="reports/2026-05-12/tsla/audit"
CANDIDATE_ID="candidate-id-from-report-json"
```

Then inspect the run:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  inspect \
  --run-id "$RUN_ID"
```

Evaluation commands print JSON to stdout. Writer commands also create audit artifacts.

## Evaluation Command Reference

All evaluation commands require `--database` and `--run-id`.

Commands that write artifacts also require `--artifact-root`.

| Subcommand | Writes artifacts | Purpose |
| --- | --- | --- |
| `inspect` | No | Show persisted evaluation counts and run state. |
| `materialize-outcome` | Yes | Materialize one candidate outcome from real post-window market data. |
| `load-outcomes` | No | Validate and load persisted outcome-evaluation artifacts for one run. |
| `outcome-summary` | Yes | Write summary artifacts for stored outcome reviews. |
| `stale-artifacts` | Yes | Write artifact freshness reviews for one run. |
| `evidence-aging` | Yes | Write evidence aging summaries for one run. |
| `source-reliability` | Yes | Write source reliability notes for stored live evidence. |
| `provider-playbook` | Yes | Write provider replacement playbooks. |
| `calibration` | Yes | Write reliability bins from stored outcome evaluations. |
| `walk-forward` | Yes | Write chronological walk-forward folds from stored outcomes. |
| `ablation` | Yes | Write signal-family ablation slices from stored outcomes. |
| `calibration-drift` | Yes | Compare two calibration summaries and write a drift check. |

### Inspect A Run

Use this first:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  inspect \
  --run-id "$RUN_ID"
```

The output is JSON. It is useful for confirming that you are pointing at the correct database and run.

### Review Stored Outcomes

Load persisted outcome-evaluation artifacts:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  load-outcomes \
  --run-id "$RUN_ID"
```

Write an outcome summary:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  outcome-summary \
  --run-id "$RUN_ID" \
  --artifact-root "$AUDIT_ROOT"
```

Use `--created-at <ISO-8601 timestamp>` only when you need a deterministic timestamp for a controlled
review workflow.

### Materialize A Candidate Outcome

Use `materialize-outcome` after the evaluation window has elapsed and real post-window market data is
available. Candidate IDs are in `report.json`; search for `candidate_id`.

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  materialize-outcome \
  --run-id "$RUN_ID" \
  --candidate-id "$CANDIDATE_ID" \
  --point-in-time-cutoff 2026-05-12T20:00:00+00:00 \
  --evaluation-window-start 2026-05-13T00:00:00+00:00 \
  --evaluation-window-end 2026-05-22T00:00:00+00:00 \
  --artifact-root "$AUDIT_ROOT"
```

Optional arguments:

| Argument | Meaning |
| --- | --- |
| `--report-date` | Override/report date metadata for the materialization. |
| `--market-artifact-id` | Existing market artifact ID to use. You can repeat this argument. |
| `--created-at` | Deterministic artifact creation timestamp. |
| `--evaluated-at` | Deterministic outcome evaluation timestamp. |

Same-day daily closes are not usable as cutoff evidence until the close would have been observable.
If a live provider cannot supply admissible outcome data, the attempt is recorded as unavailable or
failed rather than replaced with dummy data.

### Check Freshness And Evidence Aging

Write artifact freshness reviews:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  stale-artifacts \
  --run-id "$RUN_ID" \
  --artifact-root "$AUDIT_ROOT"
```

Write evidence aging summaries:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  evidence-aging \
  --run-id "$RUN_ID" \
  --artifact-root "$AUDIT_ROOT"
```

Use `--reviewed-at <ISO-8601 timestamp>` only when you need deterministic review time.

### Review Source Reliability And Provider Replacement

Write source reliability notes:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  source-reliability \
  --run-id "$RUN_ID" \
  --artifact-root "$AUDIT_ROOT"
```

Write provider replacement playbooks:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  provider-playbook \
  --run-id "$RUN_ID" \
  --artifact-root "$AUDIT_ROOT"
```

These artifacts are especially useful for live runs. They preserve source quality, retrieval method,
freshness, provider compatibility requirements, and limitations. They do not assert that a source
claim is true.

### Build Calibration, Walk-Forward, And Ablation Artifacts

Calibration needs stored outcome evaluations:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  calibration \
  --run-id "$RUN_ID" \
  --cohort-id tsla-swing \
  --as-of 2026-05-22T00:00:00+00:00 \
  --artifact-root "$AUDIT_ROOT"
```

Optional filters:

```sh
--family technical
--family news
--prediction-type directional
--horizon 10d
--bin-edge 0.0
--bin-edge 0.25
--bin-edge 0.5
--bin-edge 0.75
--bin-edge 1.0
```

Write walk-forward folds:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  walk-forward \
  --run-id "$RUN_ID" \
  --cohort-id tsla-swing \
  --point-in-time-cutoff 2026-05-22T00:00:00+00:00 \
  --minimum-train-size 20 \
  --test-size 1 \
  --step-size 1 \
  --artifact-root "$AUDIT_ROOT"
```

Write signal-family ablation slices:

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  ablation \
  --run-id "$RUN_ID" \
  --cohort-id tsla-swing \
  --point-in-time-cutoff 2026-05-22T00:00:00+00:00 \
  --family technical \
  --family news \
  --artifact-root "$AUDIT_ROOT"
```

Compare two calibration summaries:

Set `PRIOR_CALIBRATION_ID` and `CURRENT_CALIBRATION_ID` to calibration artifact IDs from earlier
calibration runs.

```sh
python -m nlp_stock_prediction evaluation \
  --database "$DATABASE" \
  calibration-drift \
  --run-id "$RUN_ID" \
  --prior-calibration-id "$PRIOR_CALIBRATION_ID" \
  --current-calibration-id "$CURRENT_CALIBRATION_ID" \
  --as-of 2026-06-01T00:00:00+00:00 \
  --artifact-root "$AUDIT_ROOT"
```

Optional drift thresholds:

```sh
--signal-family technical
--min-resolved-count 10
--watch-delta 0.05
--degraded-delta 0.10
--improved-delta 0.10
```

## A Practical End-To-End Workflow

1. Generate a live report:

   ```sh
   python -m nlp_stock_prediction research --date 2026-05-12 --symbol TSLA --output reports/ --live
   ```

2. Read `reports/2026-05-12/tsla/report.md`.

3. Inspect provider health and insufficient-evidence sections before trusting any scenario language.

4. Confirm the runtime database and run ID:

   ```text
   Database: data/research-live-runtime-2026-05-12-<symbol-hash>-<output-hash>.sqlite3
   Run ID: research-2026-05-12-tsla
   Audit root: reports/2026-05-12/tsla/audit
   ```

   Set `DATABASE`, `RUN_ID`, and `AUDIT_ROOT` to those values as shown in the evaluation workflow
   section above.

5. Inspect the run:

   ```sh
   python -m nlp_stock_prediction evaluation \
     --database "$DATABASE" \
     inspect \
     --run-id "$RUN_ID"
   ```

6. Write hardening artifacts:

   ```sh
   python -m nlp_stock_prediction evaluation \
     --database "$DATABASE" \
     stale-artifacts \
     --run-id "$RUN_ID" \
     --artifact-root "$AUDIT_ROOT"

   python -m nlp_stock_prediction evaluation \
     --database "$DATABASE" \
     evidence-aging \
     --run-id "$RUN_ID" \
     --artifact-root "$AUDIT_ROOT"

   python -m nlp_stock_prediction evaluation \
     --database "$DATABASE" \
     source-reliability \
     --run-id "$RUN_ID" \
     --artifact-root "$AUDIT_ROOT"
   ```

7. After the evaluation window has elapsed, materialize outcomes for candidate IDs from `report.json`.

8. Once enough resolved outcomes exist, run calibration, walk-forward, ablation, and calibration-drift
   checks.

## Troubleshooting

### `--database must reference an existing research SQLite database`

The evaluation CLI does not create databases. Use a database that already exists under `data/`, such
as the runtime database created by a previous `research` command.

### `Research artifact writes are limited to ...`

Use an artifact root under an allowed local artifact area. For report hardening, prefer:

```text
reports/<YYYY-MM-DD>/<symbol-slug>/audit
```

### `research run already exists`

The same date, symbol, output path, and mode can resolve to an existing runtime database and run ID.
Use a new report date, output directory, or runtime database when you want a distinct run.

### The live report has provider warnings

That is expected when credentials are missing, providers are down, data is stale, or a source does not
support the requested instrument. The app surfaces those conditions instead of hiding them.

### The report says insufficient evidence

This means the run did not have enough admissible evidence to support a scenario. Check provider
health, universe resolution, source references, and audit artifacts to see what was missing or
excluded.

### A candidate is missing from the final report

Candidates with missing required evidence, malformed artifacts, hash mismatches, or unsupported
evaluation metadata can be excluded from final report assembly. Check `report.json`,
`audit/audit-manifest.json`, and provider health warnings.

### Outcome materialization rejects same-day data

The live outcome path protects cutoff observability. A same-day daily bar is not accepted as cutoff
evidence until the close would have been observable.

### Live commands are slow or rate-limited

Use provider credentials where available, configure a polite user agent for scraping/SEC paths, and
consider an ignored `--cache-dir cache/` for provider cache metadata. Rate limits and cache failures
remain visible in artifacts.

## Where To Learn More

- [configuration.md](configuration.md): runtime configuration, environment variables, storage, and
  live-provider conventions.
- [contracts.md](contracts.md): implemented and target report, evidence, instrument, and evaluation
  contracts.
- [architecture.md](architecture.md): target architecture and component responsibilities.
- [testing-plan.md](testing-plan.md): deterministic offline testing and opt-in live test strategy.
