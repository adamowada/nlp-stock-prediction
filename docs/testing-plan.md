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
- Include negative cases for malformed HTML, duplicate tickers, missing provider data, short ticker false positives, unsupported recommendations, and joke/sarcasm risk.

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

- Current Phase 3 live API coverage verifies the SEC company tickers endpoint and keeps a reserved
  LLM smoke gate explicit. Future live provider coverage should extend to Reddit, X/Twitter, news,
  market data, fundamentals, FRED, and any enabled LLM provider.
- Require explicit environment variables for credentials and opt-in execution.
- Check authentication failures, quota/rate-limit responses, malformed upstream responses, and stale data behavior.
- Mark these tests separately from fast local tests, for example:

```sh
python -m pytest -m live_api
```

### 5. Live scraping tests

- Verify that permitted public HTML scraping fallbacks still locate expected page sections and fail clearly when markup changes.
- Keep scraping tests narrow: assert configured expected text, selectors, or parseable required
  structures rather than broad page content. Current Phase 3 coverage uses a configured public URL
  and literal expected text.
- Include alerts or failure messages that explain which selector, subtree, or regex no longer matches.
- Mark scraping tests separately, for example:

```sh
python -m pytest -m live_scraping
```

### 6. LLM extraction tests

- Use frozen prompt inputs and expected schema-shaped outputs for deterministic validation.
- Test that unsupported strategies are rejected when evidence is missing.
- Test clustering behavior for near-duplicates such as "buy calls," "weekly calls," and "calls into earnings."
- For future live LLM smoke tests, validate schema compliance and evidence preservation rather than
  exact wording. The current V1 CLI does not enable a live LLM adapter; LLM validation is
  fixture-backed.

### 7. End-to-end report tests

- Run the full CLI from fixtures and verify that Markdown, JSON, and audit artifacts are generated.
- Include scenarios for:
  - A normal six-ticker day.
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

Fixture-backed e2e coverage now exercises offline CLI report generation. Live API and live scraping
checks remain opt-in, and live LLM checks are reserved until a live adapter and credential contract
exist. See `docs/configuration.md` for the current live-smoke environment variables and `.env`
guidance.

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
- The full report can be generated from fixtures.
- The full report can be smoke-tested against live dependencies when credentials and network access are available.
- Recommendation scoring has tests for both confident-trade and no-trade outcomes.
