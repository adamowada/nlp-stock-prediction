# Testing Plan

## Summary

This project should test both deterministic application logic and the live external dependencies
that make the report useful. Fixture-based tests provide fast feedback and reproducibility; live
API and scraping tests verify that selected provider edges still work against real services.

Live dependency tests are part of the testing strategy, but they should be explicitly marked because they can require credentials, internet access, paid/free quota, and resilient handling of upstream changes.
The default test harness blocks network access. Live tests must use the appropriate live marker and
set `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS=1` before opening sockets.

## Test layers

### 1. Unit tests

- Validate pure functions and small modules without network access.
- Cover ticker extraction, ticker matching, evidence normalization, clustering, technical indicators, scoring rules, risk gates, and report rendering helpers.
- Cover ML technical datasets with deterministic OHLCV fixtures, including candle, volatility,
  volume, gap, and forward-return label generation.
- Cover ML technical sidecar integration, including model/evaluation conversion, report attachment,
  weak or conflicting signal gates, and scoring penalties.
- Cover fundamental-agent integration, including valid citation-bound sidecars, malformed output,
  unsupported claims, stale or contradictory evidence, and unavailable-agent fallbacks.
- Include negative cases for malformed HTML, duplicate or insufficient tickers, missing provider
  data, rate-limit and unavailable-provider results, stale market or macro data, unsupported
  recommendations, conflicting evidence, joke/sarcasm risk, no qualified strategies, and short
  ticker false positives.
- Include ML negative cases for insufficient history, duplicate bars, missing OHLCV, impossible
  prices, split-like leakage, and temporal train/validation leakage.
- Include local training command smoke coverage that verifies model, metrics, metadata, artifact
  SHA-256 hashes, split metadata, and selected-device metadata are written.

### 2. Schema and model tests

- Validate Pydantic models for all public data boundaries.
- Ensure required evidence fields are present before strategies or recommendations can be emitted.
- Check JSON serialization for `DailyReport`, `TradeCandidate`, `StrategyCluster`, and provider metadata.

### 3. Provider contract tests with fixtures

- Use recorded API and scraping fixtures to test adapters deterministically.
- Confirm each provider maps raw responses into normalized internal models.
- Keep fixtures small, representative, and refreshable.
- Store enough raw provider metadata to debug failures and regenerate fixtures safely.

### 4. Live API integration tests

- Current live API coverage verifies the SEC company tickers endpoint and X API recent search, and
  keeps a reserved LLM smoke gate explicit. Future live provider coverage should extend to market
  data, fundamentals, FRED, and any enabled LLM provider.
- X API provider coverage should verify the production stock-news/social query shape:
  `$TICKER lang:en -is:retweet`, `sort_order=relevancy`, and `max_results=50`.
- Require explicit environment variables for credentials and opt-in execution.
- Check authentication failures, quota/rate-limit responses, malformed upstream responses, and stale data behavior.
- Mark these tests separately from fast local tests, for example:

```sh
python -m pytest -m live_api
```

### 5. Live scraping tests

- Verify that permitted public HTML scraping fallbacks still locate expected page sections and fail clearly when markup changes.
- Keep scraping tests narrow: assert configured expected text, selectors, or parseable required
  structures rather than broad page content. Current coverage includes source-specific Reddit, AP
  News, and Candlecharts smoke checks plus a configured public URL and literal expected text.
- Include alerts or failure messages that explain which selector, subtree, or regex no longer matches.
- Mark scraping tests separately, for example:

```sh
python -m pytest -m live_scraping
```

### 6. LLM extraction tests

- Use frozen prompt inputs and expected schema-shaped outputs for deterministic validation.
- Test that unsupported strategies are rejected when evidence is missing.
- Test clustering behavior for near-duplicates such as "buy calls," "weekly calls," and "calls into earnings."
- For future live LLM smoke tests, validate schema conformance and evidence preservation rather than
  exact wording. The current V1 CLI does not enable a live LLM adapter; LLM validation is
  fixture-backed.

### 7. End-to-end report tests

- Run the full CLI from fixtures and verify that Markdown, JSON, and audit artifacts are generated.
- Include scenarios for:
  - A normal six-ticker day.
  - Experimental scrape source mode with provider health, warnings, and `provider-results.json`.
  - Experimental scrape source mode with ML sidecar and fundamental-agent sidecar entries in
    Markdown, JSON, provider-result, and analysis-context audit payloads.
  - No qualifying trade ideas.
  - Partial provider outages.
  - Conflicting Reddit/news/technical/fundamental signals.
  - At least one qualified stock idea.
  - At least one qualified defined-risk options idea.

## Suggested markers

- `unit`: Pure logic and small modules.
- `schema`: Pydantic models and serialization.
- `contract`: Provider adapters using recorded fixtures.
- `integration`: Multi-module behavior with controlled dependencies.
- `live_api`: Real API calls requiring credentials or internet access.
- `live_scraping`: Real public HTML scraping checks.
- `llm`: LLM extraction and schema validation.
- `e2e`: Full CLI report generation.

## Expected commands

Canonical command targets:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
python -m pytest -m live_api
python -m pytest -m live_scraping
python -m pytest -m e2e
ruff check .
ruff format --check .
mypy .
```

Fixture-backed e2e coverage now exercises offline CLI report generation and experimental
`--source-mode scrape` orchestration. Live API and live scraping checks remain opt-in; the explicit
`--source-mode scrape --live-providers` report path is local opt-in and should be validated with
credentials/configuration outside the default deterministic suite. Live LLM checks are reserved
until a live adapter and credential contract exist. See `docs/configuration.md` for the current
live-smoke environment variables and `.env` guidance.

ML lane smoke coverage is CPU-only by default:

```sh
python -m pytest tests/test_ml_dataset.py tests/test_ml_training_smoke.py -q
```

GPU/RTX training is a local opt-in command path and should not be required in PR CI.

Stage 5 failure drills are represented across parser, provider-adapter, extraction/clustering,
scoring, report-rendering, and CLI e2e tests. The matrix covers malformed Reddit ticker-card HTML,
duplicate/insufficient tickers, missing provider data, rate-limit and unavailable-provider
envelopes, stale market or macro data, unsupported recommendations, conflicting evidence,
joke/sarcasm risk, and no qualified strategies.

## CI expectations

- Pull-request CI should run fast unit, schema, contract, integration, and fixture-backed end-to-end tests.
- Live API and live scraping tests should run manually or on a scheduled workflow with required secrets and quota controls.
- Scheduled live tests should produce actionable failure messages when an upstream API, auth token, rate limit, or page structure changes.
- Failures from live dependencies should distinguish app regressions from upstream/provider availability issues.

## Acceptance criteria

- Core logic is covered by fast deterministic tests.
- Each provider adapter has fixture-backed contract tests; live opt-in tests are added as each live
  provider path is enabled.
- Scraping fallbacks have narrow live tests that detect markup drift.
- The full report can be generated from offline fixtures and from fixture-backed scrape source mode.
- Selected live provider edges and the explicit live-provider scrape report path can be
  smoke-tested when credentials/configuration and network access are available; default report
  generation remains network-free.
- Recommendation scoring has tests for both qualified opportunities and reports where nothing qualifies.
