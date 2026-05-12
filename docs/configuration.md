# CLI And Configuration

## Current CLI Mode

The V1 CLI currently supports deterministic offline report generation:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
```

This writes:

- `reports/YYYY-MM-DD/report.md`
- `reports/YYYY-MM-DD/report.json`
- `reports/YYYY-MM-DD/audit/`

`--offline` is required for report generation in the current Phase 3 CLI. If it is omitted, the
CLI exits with code `3` and explains that live-provider report orchestration is not enabled yet.

## CLI Options

- `--date`: report date in `YYYY-MM-DD` format.
- `--output`: base output directory; the app writes into `<output>/<YYYY-MM-DD>/`.
- `--capital`: optional non-negative account capital for risk gates.
- `--risk-profile`: scoring/report risk profile; defaults to `exploratory`.
- `--fixture-dir`: optional fixture root reserved for external fixtures; current offline runs record
  this path in command metadata and use built-in deterministic fixtures.
- `--cache-dir`: optional provider response cache directory reserved for future live/provider runs;
  current offline runs record this path in command metadata.
- `--offline`: required for the current deterministic report path and prevents live network provider
  usage.

## Environment Variables

The default test suite and `run --offline` do not require credentials, `.env` files, or network
access. Live smoke checks are explicit opt-in:

| Variable | Used for | Required when |
| --- | --- | --- |
| `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS` | Enables live API/scraping tests when set to `1`. | Running `pytest -m live_api` or `pytest -m live_scraping` against real services. |
| `NLP_STOCK_PREDICTION_LIVE_USER_AGENT` | Identifies SEC live API smoke requests. | Running the SEC live API smoke test. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL` | Narrow URL for the public scraping smoke test. | Running the live scraping smoke test. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT` | Literal text expected in the configured scraping smoke response. | Running the live scraping smoke test. |
| `NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT` | Optional User-Agent for future compliance-aware public HTML adapters. | Manual/live scraping adapter runs; deterministic tests use injected transports. |
| `NLP_STOCK_PREDICTION_SCRAPE_MIN_DELAY_SECONDS` | Optional non-negative crawl delay floor for future scraping adapters; defaults to `1.0`. | Manual/live scraping adapter runs that enforce polite throttling. |
| `NLP_STOCK_PREDICTION_LIVE_LLM_SMOKE` | Reserved for future live LLM smoke wiring. | Leave unset or `0` until live LLM adapter wiring is enabled. |

Provider adapters already return structured `missing_credentials` warnings when keys are absent.
The current CLI does not wire live provider orchestration yet, but these names are the project
conventions for future live setup or manual adapter wiring:

| Variable | Intended provider |
| --- | --- |
| `NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY` | Alpha Vantage market data and fundamentals. |
| `NLP_STOCK_PREDICTION_FRED_API_KEY` | FRED macro data. |
| `NLP_STOCK_PREDICTION_X_API_KEY` | X App API Key, also called the Consumer Key; used to identify the app and regenerate app-only tokens when needed. |
| `NLP_STOCK_PREDICTION_X_API_SECRET` | X App API Secret, also called the Consumer Secret; keep secret and use only for token generation or OAuth flows. |
| `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` | X App-only Bearer Token for read-only recent-search requests. |
| `NLP_STOCK_PREDICTION_NEWS_API_KEY` | Public news provider adapters that require an API key. |

## X API Stock News

Live provider orchestration is not enabled in the current V1 CLI, but the live X provider direction
is official X API recent search, not browser scraping. X is always treated as a stock-news/social
evidence source for every discovered ticker once live provider orchestration is enabled.

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

Live scraping remains unwired in the V1 CLI, but shared adapter helpers now use a source policy
registry before public HTML fetches. Policies describe allowlisted paths, disallowed paths, robots
review status, login and JavaScript requirements, and fallback behavior. Blocked, login-required,
and markup-drift cases should return `ProviderResult` warnings instead of raising for expected
provider conditions.

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

## `.env` Files

`.env` and `.env.*` are ignored by git. `.env.example` contains placeholder keys only and is safe to
commit.

The app does not automatically load `.env` files yet. For now, either set environment variables in
your shell or load a local `.env` with your own shell tooling before running live smoke checks.

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

The current checked live API path is the SEC company tickers JSON endpoint. It does not require an
API key, but SEC requests must include a contact-oriented User-Agent.

PowerShell example:

```powershell
$env:NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS = "1"
$env:NLP_STOCK_PREDICTION_LIVE_USER_AGENT = "your-name your-email@example.com"
python -m pytest -m live_api tests/test_lane_f_live_smoke.py
```

The current checked live scraping path is deliberately configurable so it can point at a narrow
public page the project is allowed to fetch. A known-good low-risk smoke configuration is:

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
