# CLI And Configuration

## Current CLI Modes

The V1 CLI supports deterministic offline report generation:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
```

It also supports an experimental fixture-backed scrape-source path:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape
```

To explicitly call live providers from the local machine, add `--live-providers` and preferably a
cache directory:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape --live-providers --cache-dir cache/live
```

This writes:

- `reports/YYYY-MM-DD/report.md`
- `reports/YYYY-MM-DD/report.json`
- `reports/YYYY-MM-DD/audit/`

If both `--offline` and `--source-mode` are omitted, the CLI exits with code `3` and explains the
available explicit modes. `--source-mode scrape` uses deterministic Reddit/AP/Candlecharts/X
fixtures plus provider degradation probes by default; it does not make live network calls unless
`--live-providers` is also present.

## CLI Options

- `--date`: report date in `YYYY-MM-DD` format.
- `--output`: base output directory; the app writes into `<output>/<YYYY-MM-DD>/`.
- `--capital`: optional non-negative account capital for risk gates.
- `--risk-profile`: scoring/report risk profile; defaults to `exploratory`.
- `--fixture-dir`: optional fixture root reserved for external fixtures; current offline runs record
  this path in command metadata and use built-in deterministic fixtures.
- `--cache-dir`: optional provider response cache directory; deterministic runs record this path in
  command metadata.
- `--source-mode`: explicit source mode. `scrape` runs the experimental compliance-aware provider
  path against deterministic fixtures and writes provider-result audit artifacts.
- `--offline`: uses the deterministic offline fixture-backed report path and prevents live network
  provider usage.
- `--live-providers`: opt into real provider calls for `--source-mode scrape`. This calls public
  Reddit/AP/Candlecharts HTML providers and the X recent-search API while degrading expected
  provider failures into provider-result warnings.

## Environment Variables

The default test suite, `run --offline`, and `run --source-mode scrape` do not require credentials,
`.env` files, or network access. The CLI automatically loads the nearest local `.env` file when it
starts, without overriding variables already exported in the shell. Live smoke checks are explicit
opt-in:

| Variable | Used for | Required when |
| --- | --- | --- |
| `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS` | Enables live API/scraping tests when set to `1`. | Running `pytest -m live_api` or `pytest -m live_scraping` against real services. |
| `NLP_STOCK_PREDICTION_LIVE_USER_AGENT` | Identifies SEC live API smoke requests. | Running the SEC live API smoke test. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL` | Narrow URL for the public scraping smoke test. | Running the live scraping smoke test. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT` | Literal text expected in the configured scraping smoke response. | Running the live scraping smoke test. |
| `NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT` | Optional User-Agent for future compliance-aware public HTML adapters. | Manual/live scraping adapter runs; deterministic tests use injected transports. |
| `NLP_STOCK_PREDICTION_SCRAPE_MIN_DELAY_SECONDS` | Optional non-negative crawl delay floor for future scraping adapters; defaults to `1.0`. | Manual/live scraping adapter runs that enforce polite throttling. |
| `NLP_STOCK_PREDICTION_LIVE_LLM_SMOKE` | Reserved for future live LLM smoke wiring. | Leave unset or `0` until live LLM adapter wiring is enabled. |
| `NLP_STOCK_PREDICTION_DISABLE_DOTENV` | Disables CLI `.env` auto-loading when set to `1`. | Deterministic tests, CI, or debugging an exported shell environment. |

Provider adapters already return structured `missing_credentials` warnings when keys are absent.
`--source-mode scrape --live-providers` uses the configured live provider variables below where
the corresponding provider is wired. Missing optional credentials degrade into provider warnings
instead of blocking the whole report:

| Variable | Intended provider |
| --- | --- |
| `NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY` | Alpha Vantage market data and fundamentals. |
| `NLP_STOCK_PREDICTION_FRED_API_KEY` | FRED macro data. |
| `NLP_STOCK_PREDICTION_X_API_KEY` | X App API Key, also called the Consumer Key; used to identify the app and regenerate app-only tokens when needed. |
| `NLP_STOCK_PREDICTION_X_API_SECRET` | X App API Secret, also called the Consumer Secret; keep secret and use only for token generation or OAuth flows. |
| `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` | X App-only Bearer Token for read-only recent-search requests and X live smoke. |
| `NLP_STOCK_PREDICTION_NEWS_API_KEY` | Public news provider adapters that require an API key. |

## X API Stock News

The X provider direction is official X API recent search, not browser scraping. X is treated as a
stock-news/social evidence source for every discovered ticker. The default scrape source mode uses
deterministic X API-shaped fixtures; `--source-mode scrape --live-providers` uses the configured X
Bearer Token for live recent-search calls. Live X verification is covered by opt-in smoke tests.

For each ticker, the app should request the top 50 relevant posts from the X recent-search endpoint:

- Query: `$TICKER lang:en -is:retweet`
- Result order: `sort_order=relevancy`
- Result count: `max_results=50`

The provider should authenticate with `NLP_STOCK_PREDICTION_X_BEARER_TOKEN`, request public post
fields such as `created_at`, `public_metrics`, `lang`, and `author_id`, and preserve the query,
sort order, returned post IDs, timestamps, metrics, and API response snapshot IDs in provenance and
audit artifacts. The app intentionally avoids `sort_order=recency` for production stock-news
evidence because the smoke test showed too much spam/noise. API key and secret values are kept in
`.env` for completeness and token rotation; normal read-only recent-search calls should use the
Bearer Token.

## Public HTML Scraping Guardrails

Shared adapter helpers use a source policy registry before public HTML fetches. Policies describe
allowlisted paths, disallowed paths, robots review status, login and JavaScript requirements, and
fallback behavior. Blocked, login-required, and markup-drift cases return `ProviderResult` warnings
instead of raising for expected provider conditions. `--source-mode scrape` carries these provider
warnings into report provider health and `audit/provider-results.json`.

Live scrape-mode orchestration currently calls:

- Reddit public `r/wallstreetbets` pages for ticker discovery and public discussion evidence.
- AP News public financial-markets hub and linked public article pages.
- Candlecharts public live chart page as a feasibility probe for first-party OHLCV only.
- X API v2 recent search for the six report tickers when `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` is
  configured.

The default public HTML request identity is
`nlp-stock-prediction/0.1 compliance-aware-scraper`. Override it with
`NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT` for manual/live adapter runs when a more specific contact
string is appropriate. `NLP_STOCK_PREDICTION_SCRAPE_MIN_DELAY_SECONDS` is reserved as the shared
polite crawl-delay floor for future live adapters; keep it non-negative.

## Candlecharts Data Limitation

The Candlecharts adapter is a feasibility probe for public first-party OHLCV only. Fixture-backed
HTML tests cover JSON/table candle data when it is present, but if a page only exposes an embedded
TradingView/widget chart, the provider returns a structured `no_data` warning and does not scrape
TradingView internals. Downstream ML or technical-analysis work should use approved provider data or
user-supplied OHLCV CSV fixtures when Candlecharts is widget-only.

## Local ML Technical Analysis

The ML technical-analysis lane builds local research artifacts from user-supplied OHLCV CSV files
and records model, dataset, hardware, and usage-limitation metadata. Outputs are not investment
advice. When an evaluated model is explicitly attached to an analysis bundle, the report treats it
as a conservative sidecar with model/dataset hashes, probability, calibration, freshness, and
validation metrics. Recommendation scoring applies weak, stale, unavailable, or conflicting ML
gates when that sidecar is present; ML output cannot qualify a trade by itself.

The experimental fixture-backed `--source-mode scrape` path includes a TSLA ML sidecar so Markdown,
JSON, and audit payloads exercise the integration shape without requiring a local model artifact.
The explicit live-provider scrape path suppresses fixture ML, fundamental-agent, extraction, and
scoring sidecars for now; it records live provider evidence and emits no-trade guidance until live
analysis wiring is enabled under a future plan.

Default tests use a pure-Python CPU logistic baseline and do not require CUDA, PyTorch, network
access, or local training data. CUDA/RTX metadata is detected only when available and when the local
training command is configured with `--device auto` or `--device cuda`.

Local training data and artifacts should stay outside git. The repo ignores `data/ml/`,
`artifacts/`, and `models/` for this purpose. A local CSV must include:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- optional `adjusted_close`

Use `--as-of` with `--max-latest-bar-age-days` when the training CSV is expected to be current.
That gate rejects stale local data and bars dated after the as-of value, which protects local
experiments from accidental lookahead leakage.

Example CPU training command:

```sh
python -m nlp_stock_prediction.ml.train --csv data/ml/TSLA.csv --ticker TSLA --output-dir artifacts/ml/TSLA --device cpu --as-of 2026-05-11 --max-latest-bar-age-days 5
```

Training writes `model.json`, `metrics.json`, and `metadata.json`. The metadata artifact records the
seed, hyperparameters, temporal split, purged validation gap, train/validation metrics, dataset hash,
model hash, model and metrics artifact SHA-256 hashes, Python/runtime metadata, execution backend,
selected device, CUDA availability, GPU name when detected, and the usage limitation text. The CLI
stdout also returns the metadata artifact SHA-256 hash for external run manifests.

When running from an uninstalled source checkout, set `PYTHONPATH=src` or install the package in
editable mode before using `python -m`.

Example evaluation command:

```sh
python -m nlp_stock_prediction.ml.evaluate --model artifacts/ml/TSLA/model.json --csv data/ml/TSLA.csv --ticker TSLA --output artifacts/ml/TSLA/evaluation.json
```

## Fundamental Agent Analysis

The fundamental-agent lane accepts a citation-bound `FundamentalNlpAnalysisRequest` and returns a
validated `FundamentalNlpAnalysisResponse` with claims, risks, assumptions, confidence inputs, and
audit metadata. Fixture-backed providers are used in deterministic tests. Expected malformed,
unsupported, stale, contradictory, or unavailable agent outputs become provider warnings rather than
uncaught pipeline failures.

When a validated agent result is attached, the report preserves the existing deterministic
fundamental analysis and adds an `agent_signal` sidecar. The sidecar is surfaced in Markdown, JSON,
`provider-results.json`, and `analysis-contexts.json`; generated agent interpretation remains
separate from observed source evidence.

## `.env` Files

`.env` and `.env.*` are ignored by git. `.env.example` contains placeholder keys only and is safe to
commit.

The CLI automatically searches from the current working directory upward for the nearest `.env` file
and loads it on startup. Existing shell variables win over `.env` values, so temporary PowerShell
exports can override local defaults for one run. Set `NLP_STOCK_PREDICTION_DISABLE_DOTENV=1` to
disable this behavior.

For an X smoke test, create a local `.env` shaped like this and fill in the values from your X
Developer App's "Keys and tokens" page:

```dotenv
NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS=1
NLP_STOCK_PREDICTION_X_API_KEY=your-consumer-key
NLP_STOCK_PREDICTION_X_API_SECRET=your-secret-key
NLP_STOCK_PREDICTION_X_BEARER_TOKEN=your-bearer-token
```

Do not quote the values unless your shell loader requires it. Do not commit `.env`; `.env` and
`.env.*` are ignored by git. The app should never print these values in logs, reports, audit
artifacts, or test failures.

## Configured Live Smoke Paths

The checked live API paths include SEC company tickers and X recent search. SEC does not require an
API key, but SEC requests must include a contact-oriented User-Agent. X requires
`NLP_STOCK_PREDICTION_X_BEARER_TOKEN`.

PowerShell example:

```powershell
$env:NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS = "1"
$env:NLP_STOCK_PREDICTION_LIVE_USER_AGENT = "your-name your-email@example.com"
python -m pytest -m live_api tests/test_lane_f_live_smoke.py
```

The checked live scraping paths include provider-specific Reddit, AP News, and Candlecharts smoke
coverage plus a configurable narrow URL. A known-good low-risk configurable smoke is:

```powershell
$env:NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS = "1"
$env:NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL = "https://example.com/"
$env:NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT = "Example Domain"
python -m pytest -m live_scraping tests/test_lane_f_live_smoke.py
```

After Stage 3, live LLM smoke remains a reserved gate. The V1 CLI currently validates LLM
extraction through deterministic fixture-backed schema tests; no live LLM adapter or credential
contract is enabled yet.

## Missing Credentials

Missing provider credentials are expected configuration problems, not programmer errors. Provider
adapters should return `ProviderResult` envelopes with:

- `status=unconfigured`
- a `missing_credentials` warning
- `credential_state=missing`
- `data=None`

Live tests skip cleanly when `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS` is not set. Once live tests are
explicitly requested with that opt-in flag, missing per-test configuration fails with an actionable
message naming the required environment variable. Network, quota, HTTP, and upstream availability
failures also fail with provider-specific context instead of raw socket tracebacks.
