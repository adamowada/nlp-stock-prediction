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

`--offline` is required for report generation during Phase 3 Stage 2. If it is omitted, the CLI
exits with code `3` and explains that live-provider report orchestration is not enabled yet.

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
| `NLP_STOCK_PREDICTION_LIVE_LLM_SMOKE` | Reserved for future live LLM smoke wiring. | Leave unset or `0` until live LLM adapter wiring is enabled. |

Provider adapters already return structured `missing_credentials` warnings when keys are absent.
The current CLI does not wire live provider orchestration yet, but these names are the project
conventions for future live setup or manual adapter wiring:

| Variable | Intended provider |
| --- | --- |
| `NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY` | Alpha Vantage market data and fundamentals. |
| `NLP_STOCK_PREDICTION_FRED_API_KEY` | FRED macro data. |
| `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` | X recent-search social evidence. |
| `NLP_STOCK_PREDICTION_NEWS_API_KEY` | Public news provider adapters that require an API key. |

## `.env` Files

`.env` and `.env.*` are ignored by git. `.env.example` contains placeholder keys only and is safe to
commit.

The app does not automatically load `.env` files yet. For now, either set environment variables in
your shell or load a local `.env` with your own shell tooling before running live smoke checks.

PowerShell example:

```powershell
$env:NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS = "1"
$env:NLP_STOCK_PREDICTION_LIVE_USER_AGENT = "your-name your-email@example.com"
python -m pytest -m live_api
```

## Missing Credentials

Missing provider credentials are expected configuration problems, not programmer errors. Provider
adapters should return `ProviderResult` envelopes with:

- `status=unconfigured`
- a `missing_credentials` warning
- `credential_state=missing`
- `data=None`

Live tests that need external configuration skip with actionable messages when their required
environment variables are absent.
