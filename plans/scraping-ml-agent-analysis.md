# Scraping, ML Technical Analysis, And Agent Fundamentals Plan

## Goal

Add a live source mode that uses compliance-aware scraping adapters where public HTML is the right
source, official APIs where they are the safer or supported interface, a local RTX 3090 training
pipeline for ML-assisted technical analysis, and a Codex-agent-based NLP fundamental analysis lane
that produces contract-valid, evidence-grounded outputs for the daily report.

The end state is a report pipeline that can run from scraped public web inputs where allowed and
official APIs where configured, degrades visibly when a source blocks access or lacks usable data,
trains and evaluates a local technical-analysis model reproducibly, and separates observed evidence
from generated analysis and recommendations.

## Non-goals

- Do not bypass robots.txt, paywalls, login walls, anti-bot systems, CAPTCHAs, or platform access
  controls.
- Do not use real-money brokerage execution, auto-trading, or executable order payloads.
- Do not claim scraped social/news discussion as fact without attribution and provenance.
- Do not make the default deterministic test suite depend on live websites, GPUs, browser sessions,
  or local Codex agent availability.
- Do not treat the ML model as investment advice or as a replacement for evidence, risk gates,
  and disclaimers.
- Do not scrape TradingView internals embedded inside Candlecharts unless legal review and source
  terms allow it.

## Context

Current V1 behavior uses explicit local source modes. `run --offline` builds a deterministic
fixture bundle. `run --source-mode scrape` builds a fixture-backed scrape-source bundle with
provider health, normalized evidence, ML and fundamental-agent sidecars, and provider-result audit
artifacts. `run --source-mode scrape --live-providers` explicitly calls the wired live Reddit public
page, AP News public HTML, Candlecharts feasibility, and X recent-search providers, records live
provider health and normalized evidence, and emits evidence-only output without live scored
predictions or recommendations until live extraction, analysis, and scoring are enabled under a
future plan. The X provider request shape is implemented,
fixture-tested, and wired into the live scrape provider path for the six report tickers with the
production default of 50 relevant posts per ticker.

Requested scrape targets:

- Reddit ticker/discussion source: `https://www.reddit.com/r/wallstreetbets/`
- Financial news source: `https://apnews.com/hub/financial-markets`
- Candlestick source: `https://candlecharts.com/live-charts`
- X stock-news/social source: official X API recent search for each ticker's cashtag.

Initial public-source constraints checked on 2026-05-11:

- Reddit robots.txt allows broad page access but disallows API/search endpoints and several dynamic
  service paths. Plan for the public subreddit page and public post pages only.
- AP News robots.txt allows the hub path but disallows `/api/v2/feed/`, search, RSS, and several
  internal paths. Plan for hub/article HTML only.
- X browser scraping is replaced by official X API recent search. For every discovered ticker, use
  the app-only Bearer Token to request the top 50 relevant results (`sort_order=relevancy`,
  `max_results=50`) for `$TICKER lang:en -is:retweet`.
- Candlecharts live chart page is public HTML but points users to TradingView for chart tracking.
  Plan a feasibility probe before assuming usable OHLCV data can be extracted.

Relevant existing modules:

- CLI/orchestration: `src/nlp_stock_prediction/cli.py`, `src/nlp_stock_prediction/pipeline.py`
- Provider contracts: `src/nlp_stock_prediction/contracts/providers.py`
- Provenance and warnings: `src/nlp_stock_prediction/contracts/provenance.py`
- Reddit parsing/evidence: `src/nlp_stock_prediction/reddit/`
- Provider helpers/cache: `src/nlp_stock_prediction/providers/_base.py`
- Extraction and clustering: `src/nlp_stock_prediction/extraction/`
- Analysis/scoring: `src/nlp_stock_prediction/analysis/`, `src/nlp_stock_prediction/scoring/`
- Reporting/audit: `src/nlp_stock_prediction/reporting/`
- Live test gates: `tests/conftest.py`, `tests/test_lane_f_live_smoke.py`

## Parallel Development Model

Use one coordinator worktree for integration and one isolated git worktree per Codex subagent or
human workstream. Start each worktree from the current integration branch, use the `codex/` branch
prefix for new work, and give every subagent a disjoint write set. Subagents may read the full repo,
but they should only edit files in their assigned write set and should not revert edits from other
worktrees.

Recommended worktree setup:

```sh
git fetch origin
git worktree add ../nlp-stock-prediction-policy -b codex/scraping-policy feature/release-v1
git worktree add ../nlp-stock-prediction-reddit -b codex/reddit-scraper feature/release-v1
git worktree add ../nlp-stock-prediction-apnews -b codex/apnews-scraper feature/release-v1
git worktree add ../nlp-stock-prediction-candlecharts -b codex/candlecharts-feasibility feature/release-v1
git worktree add ../nlp-stock-prediction-ml -b codex/ml-technical-analysis feature/release-v1
git worktree add ../nlp-stock-prediction-agent -b codex/fundamental-agent feature/release-v1
```

Do not copy committed secrets into worktrees. If a live smoke needs credentials, use a local ignored
`.env` or shell environment variables in that worktree only, never test fixtures, logs, reports, or
commits.

Parallel workstreams:

| ID | Branch | Owner Scope | Depends On | Primary Write Set |
| --- | --- | --- | --- | --- |
| P0 | `feature/release-v1` or integration branch | Coordinator, merge sequencing, final E2E | all | `plans/`, final docs, integration conflict resolution only |
| W1 | `codex/scraping-policy` | Source policy registry, scraping warnings, shared fetch/cache contracts | none | `src/nlp_stock_prediction/compliance.py`, `src/nlp_stock_prediction/providers/_base.py`, `src/nlp_stock_prediction/contracts/`, `tests/test_lane_f_*` |
| W2 | `codex/reddit-scraper` | Reddit public-page discovery/evidence adapter and fixtures | W1 interfaces, or temporary local shim | `src/nlp_stock_prediction/reddit/`, `src/nlp_stock_prediction/providers/reddit_scrape.py`, `tests/test_lane_a_reddit_*`, Reddit fixtures |
| W3 | `codex/apnews-scraper` | AP hub/article scraper and news evidence normalization | W1 interfaces, or temporary local shim | `src/nlp_stock_prediction/providers/apnews.py`, narrow additions to `providers/news.py`, AP tests/fixtures |
| W4 | `codex/candlecharts-feasibility` | Candlecharts feasibility probe and unavailable/widget-only warning path | W1 interfaces, market contracts read-only unless needed | `src/nlp_stock_prediction/providers/candlecharts.py`, Candlecharts tests/fixtures, docs for data limitations |
| W5 | `codex/x-orchestration` | Bind the existing X provider into live-source provider slots and smoke coverage | current X provider, W6 provider hook shape | narrow additions to X orchestration registration and X-specific tests |
| W6 | `codex/live-source-orchestration` | CLI mode, provider hook shape, provider composition, degraded-provider reporting, audit manifest | W1 for policy types; adapter PRs for final E2E | `src/nlp_stock_prediction/cli.py`, `pipeline.py`, `reporting/`, E2E tests |
| W7 | `codex/ml-technical-analysis` | Dataset schema, leakage checks, CPU/GPU training/evaluation commands | existing market contracts | `src/nlp_stock_prediction/ml/`, ML tests, ignored artifact paths, ML docs |
| W8 | `codex/ml-signal-integration` | Conservative ML signal integration into analysis/scoring/reporting | W7 metrics/artifacts schema, W6 report hooks | `contracts/analysis.py`, `analysis/technical.py`, `scoring/`, report tests |
| W9 | `codex/fundamental-agent` | Codex-agent request/response schema, fixture-backed provider, audit artifacts | existing evidence contracts; W6 integration later | `src/nlp_stock_prediction/agents/`, `analysis/fundamentals.py`, agent tests |
| W10 | `codex/docs-ci-rollout` | Docs, CI, rollout checklist after interfaces stabilize | W1-W9 | README, docs, CI workflow |

Merge order:

1. Land W1 first because it defines the shared policy and scraping primitives.
2. Land a W6 skeleton after W1 if adapters need a shared provider hook shape; keep final E2E in W6
   open until source adapters are ready.
3. Land W2, W3, and W4 independently once they compile against W1 and any W6 hook shape they need.
   They should not depend on each other.
4. Land W5 after the existing X provider and W6 provider hook shape are stable.
5. Finish W6 after at least one scraper adapter and the X provider can run through fixture-backed
   E2E, then add an explicit `--live-providers` local path for bounded live evidence collection.
6. Develop W7 in parallel with W1-W6 because it is mostly isolated. Land W8 only after W7 and W6.
7. Develop W9 schema/fixtures in parallel. Land its pipeline integration after W6.
8. Land W10 last, after commands and behavior settle.

Codex subagent packet template:

- Goal: one workstream ID and milestone name.
- Branch/worktree: exact worktree path and branch.
- Allowed writes: copy the workstream's primary write set.
- Read-only context: full repo, current plan, AGENTS.md, relevant tests.
- Forbidden writes: `.env`, unrelated workstream files, generated reports/artifacts outside ignored
  paths, and broad refactors outside the assigned scope.
- Expected output: changed file list, tests run, remaining blockers, and any contract changes other
  workstreams must consume.
- Verification: run the narrow tests for the workstream plus `ruff check`, `ruff format --check`,
  and `mypy .` when shared contracts are touched.

Integration discipline:

- Shared contracts and base helpers should change in W1 or the coordinator branch, not separately in
  every adapter branch.
- If a subagent needs a shared contract that does not exist yet, it should add a minimal local shim
  only inside its adapter and flag the desired shared shape in its final report.
- Provider adapters must return contract-valid degraded results for unavailable, blocked, stale,
  malformed, and rate-limited sources. They should not raise through the pipeline for expected
  provider failures.
- Fixtures should be small, source-specific, and named by provider and scenario. Live snapshots and
  large training data stay out of git unless deliberately curated as fixtures.
- The coordinator owns cross-workstream conflict resolution, final E2E, and report/audit consistency.

## Milestones

### Milestone 1: Compliance And Scraping Policy Gate

- Changes:
  - Add a source policy registry for each target with robots status, allowed paths, disallowed paths,
    crawl delay, login requirement, JavaScript requirement, and fallback behavior.
  - Add a `ScrapingPolicyResult` or equivalent provider warning helper so blocked sources return
    structured `ProviderResult` warnings instead of silent omissions.
  - Add user-agent configuration for polite scraping and rate limiting.
  - Preserve the current network-blocked default tests.
- Files likely affected:
  - `src/nlp_stock_prediction/providers/_base.py`
  - `src/nlp_stock_prediction/compliance.py`
  - `src/nlp_stock_prediction/contracts/providers.py`
  - `src/nlp_stock_prediction/contracts/enums.py`
  - `docs/configuration.md`
  - `.env.example`
  - `tests/test_lane_f_compliance.py`
  - new tests for scraping policy behavior
- Verification:
  - Unit tests for allowed, disallowed, login-required, and drift-detected policies.
  - Live scraping tests remain opt-in and skip without `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS=1`.

### Milestone 2: Scraping Infrastructure

- Changes:
  - Add a shared HTML fetcher/parser adapter with rate limits, retries, cache keys, content hashing,
    text extraction, canonical URL capture, and raw snapshot audit persistence.
  - Prefer static HTML parsing with `html.parser` or a small dependency such as BeautifulSoup.
  - Add optional browser-rendered scraping only behind an explicit live marker for sources that
    require JavaScript and are allowed by policy.
  - Add markup drift warnings when selectors fail or required sections are missing.
- Files likely affected:
  - `src/nlp_stock_prediction/providers/_base.py`
  - new `src/nlp_stock_prediction/providers/scraping.py`
  - `src/nlp_stock_prediction/contracts/fixtures.py`
  - `src/nlp_stock_prediction/reporting/audit.py`
  - `tests/test_scraping_provider_base.py`
- Verification:
  - Fixture-backed HTML parser tests.
  - Cache-hit tests prove no network request is made after a cached raw snapshot.
  - Drift tests cover missing selectors and malformed HTML.

### Milestone 3: Reddit WSB Scraper

- Changes:
  - Replace the fixture-only Reddit provider with a policy-aware public-page scraper that can:
    discover the Devvit daily ticker card from `r/wallstreetbets`,
    fetch public post/comment context where allowed,
    normalize discussion into `SourceEvidence`,
    and retain fixture fallback for tests.
  - Keep ticker extraction compatible with the existing `ticker-container-*` parser, but add drift
    probes for alternate public markup.
  - Do not use Reddit API, JSON endpoints, private endpoints, or login-only content.
- Files likely affected:
  - `src/nlp_stock_prediction/reddit/discovery.py`
  - `src/nlp_stock_prediction/reddit/evidence.py`
  - `src/nlp_stock_prediction/reddit/provider.py`
  - new `src/nlp_stock_prediction/providers/reddit_scrape.py`
  - `tests/test_lane_a_reddit_*`
  - new recorded raw Reddit HTML fixtures
- Verification:
  - Contract tests for valid six-ticker discovery, insufficient tickers, duplicate tickers,
    malformed markup, stale snapshot, and no-discussion cases.
  - Opt-in live smoke checks the public page shape without requiring login.

### Milestone 4: AP News Financial Markets Scraper

- Changes:
  - Add a news scraper for the AP financial markets hub and linked article pages.
  - Extract headline, article URL, published timestamp when available, author/source label, body
    summary text, matched tickers, and provenance.
  - Avoid AP API/feed/search/RSS paths disallowed by robots.
  - Add AP-specific drift warnings and freshness thresholds.
- Files likely affected:
  - `src/nlp_stock_prediction/providers/news.py`
  - new `src/nlp_stock_prediction/providers/apnews.py`
  - `tests/test_lane_b_providers.py`
  - new AP raw fixtures
- Verification:
  - Fixture tests for hub page extraction, article page extraction, missing timestamp, unrelated
    article filtering, and stale news.
  - Opt-in live smoke checks only the hub page and one public article URL if discovered.

### Milestone 5: Candlecharts Candlestick Data Feasibility And Adapter

- Changes:
  - First implement a feasibility probe that records whether Candlecharts exposes usable OHLCV data
    in public HTML, JSON script tags, or permitted page content.
  - If usable public OHLCV exists, build a scraper-backed `MarketDataProvider` that emits
    `MarketSnapshot` and raw audit snapshots.
  - If Candlecharts only embeds a TradingView widget or canvas without extractable licensed data,
    return a `ProviderStatus.UNAVAILABLE` or `PARTIAL` result with a clear warning and require
    user-supplied OHLCV CSV fixtures for ML training.
  - Add no-scrape guardrails for embedded third-party widget internals unless explicitly approved.
- Files likely affected:
  - new `src/nlp_stock_prediction/providers/candlecharts.py`
  - `src/nlp_stock_prediction/providers/market.py`
  - `src/nlp_stock_prediction/analysis/technical.py`
  - `docs/configuration.md`
  - new Candlecharts fixtures and tests
- Verification:
  - Feasibility tests for public HTML with OHLCV, widget-only HTML, missing symbol, and stale data.
  - Live smoke classifies the public page without extracting protected third-party internals.

### Milestone 6: X API Relevant Search

Status: implemented and wired into the opt-in live scrape provider path; live extraction/scoring
remains a later milestone.

- Changes:
  - Use the official X API v2 recent-search endpoint instead of browser scraping X search pages.
  - Read `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` for app-only read access.
  - Document `NLP_STOCK_PREDICTION_X_API_KEY` and `NLP_STOCK_PREDICTION_X_API_SECRET` for app
    identity and Bearer Token regeneration, but avoid using them in normal read-only calls.
  - For each discovered ticker, make one bounded recent-search request:
    `$TICKER lang:en -is:retweet` with `sort_order=relevancy` and `max_results=50` for
    top relevant stock-news results.
  - Preserve query, sort order, post IDs, timestamps, public metrics, language, author ID, cache key,
    raw snapshot ID, and freshness in `SourceEvidence` provenance.
  - If credentials are missing, quota is exhausted, X is unavailable, or a ticker returns no posts,
    emit structured provider warnings and continue the report.
- Files likely affected:
  - `src/nlp_stock_prediction/providers/social.py`
  - `src/nlp_stock_prediction/compliance.py`
  - `tests/test_lane_b_providers.py`
  - `docs/configuration.md`
- Verification:
  - Tests for missing credentials, relevancy request shape, default 50-post limit,
    six-ticker bounded orchestration, no ticker matches, duplicate posts, malformed API responses,
    stale social evidence, rate limits, and upstream failures.
  - Current provider-level verification passed with `tests/test_lane_b_providers.py`, full pytest,
    Ruff, Mypy, and `git diff --check` after the X provider update.

### Milestone 7: Live Scrape Orchestration

- Changes:
  - Add explicit scrape modes while keeping `--offline` deterministic: fixture-backed
    `run --source-mode scrape` and bounded live-provider collection through
    `run --source-mode scrape --live-providers`.
  - Wire Reddit, AP News, Candlecharts, and X API recent-search provider results into one
    degraded-provider-aware reporting and audit path.
  - Keep live extraction, analysis, scoring, ML sidecars, and fundamental-agent sidecars disabled
    for live provider runs until a future plan enables them against real evidence. Live provider
    reports remain evidence-only.
  - Ensure all missing, blocked, stale, and drifted sources appear in provider health and report
    warnings.
  - Keep raw snapshots and normalized artifacts out of git unless they are curated fixtures.
- Files likely affected:
  - `src/nlp_stock_prediction/cli.py`
  - `src/nlp_stock_prediction/pipeline.py`
  - `src/nlp_stock_prediction/reporting/fixtures.py`
  - `src/nlp_stock_prediction/reporting/markdown.py`
  - `src/nlp_stock_prediction/reporting/json.py`
  - end-to-end tests
- Verification:
  - Fixture-backed scrape-mode e2e writes Markdown, JSON, and audit artifacts.
  - Opt-in live provider scrape writes Markdown, JSON, normalized evidence, provider health, and
    `audit/provider-results.json` while surfacing X credential/quota failures and widget-only
    Candlecharts as warnings.

### Milestone 8: Local ML Technical Analysis Dataset

- Changes:
  - Define a training dataset schema from OHLCV bars, derived candle pattern features, volatility,
    volume, gap, and forward-return labels.
  - Add a data validation stage that rejects insufficient history, duplicate bars, missing OHLCV,
    stale data, and impossible prices.
  - If Candlecharts cannot legally provide enough historical OHLCV, require local CSV/raw snapshot
    imports rather than scraping unapproved sources.
  - Create deterministic sample fixtures for CPU tests and separate local training data paths that
    stay ignored by git.
- Files likely affected:
  - new `src/nlp_stock_prediction/ml/`
  - new `tests/test_ml_dataset.py`
  - `.gitignore`
  - `docs/configuration.md`
  - `docs/testing-plan.md`
- Verification:
  - Unit tests for feature generation, label generation, train/validation split integrity, no
    lookahead leakage, and insufficient-data failures.

### Milestone 9: RTX 3090 Training Pipeline

- Changes:
  - Add a PyTorch training pipeline that auto-detects CUDA and records GPU name, CUDA version,
    seed, dataset hash, hyperparameters, metrics, and model artifact hash.
  - Start with a compact temporal model such as a 1D CNN/TCN or small transformer over OHLCV-derived
    windows, plus a logistic/gradient baseline for sanity checks.
  - Add reproducible train/evaluate commands and store model artifacts outside git by default.
  - Add calibration metrics and qualification thresholds so weak predictions do not become recommendations.
- Files likely affected:
  - `pyproject.toml`
  - new `src/nlp_stock_prediction/ml/train.py`
  - new `src/nlp_stock_prediction/ml/model.py`
  - new `src/nlp_stock_prediction/ml/evaluate.py`
  - new `tests/test_ml_training_smoke.py`
  - `docs/configuration.md`
- Verification:
  - CPU smoke test trains on tiny fixtures.
  - Local GPU command trains on RTX 3090 and writes metrics/artifacts.
  - Evaluation proves no data leakage and reports confidence/calibration.

### Milestone 10: ML Technical Analysis Integration

- Changes:
  - Extend `TechnicalAnalysis` or add an ML sidecar component that includes model version, input
    feature references, prediction horizon, probability/calibration, and limitations.
  - Combine deterministic technical indicators and ML signal conservatively.
  - Block ML-driven recommendations unless model metrics and data freshness pass configured gates.
- Files likely affected:
  - `src/nlp_stock_prediction/contracts/analysis.py`
  - `src/nlp_stock_prediction/analysis/technical.py`
  - `src/nlp_stock_prediction/scoring/recommendations.py`
  - `src/nlp_stock_prediction/reporting/markdown.py`
  - schema/report tests
- Verification:
  - Tests for no model, stale model, weak confidence, conflicting deterministic/ML signal, and
    qualified ML-assisted but evidence-grounded setup.

### Milestone 11: Codex Agent Fundamental Analysis Lane

- Changes:
  - Define a `FundamentalNlpAnalysisRequest` and response schema for a Codex agent that reads scraped
    company/news/fundamental evidence and returns structured analysis with citations, assumptions,
    risks, and confidence inputs.
  - Add a fixture-backed agent provider for tests and a local/manual Codex-agent runner for real use.
  - Require the agent to separate observed scraped content from its own interpretation.
  - Persist agent prompts, response JSON, validation warnings, and source evidence IDs in audit
    artifacts.
  - If no supported programmatic Codex agent runner is available in the CLI environment, keep this
    as a human-supervised local step with importable agent-output JSON.
- Files likely affected:
  - new `src/nlp_stock_prediction/agents/fundamental.py`
  - `src/nlp_stock_prediction/contracts/analysis.py`
  - `src/nlp_stock_prediction/analysis/fundamentals.py`
  - `src/nlp_stock_prediction/pipeline.py`
  - `src/nlp_stock_prediction/reporting/audit.py`
  - new `tests/test_fundamental_agent.py`
- Verification:
  - Tests for valid agent output, missing citations, unsupported claims, contradictory evidence,
    stale evidence, malformed JSON, and no-agent-available fallback.

### Milestone 12: Documentation, CI, And Rollout

- Changes:
  - Update README, configuration, testing plan, and roadmap with scrape-mode commands, env vars,
    live-source caveats, ML training commands, and Codex agent workflow.
  - Add CI jobs for deterministic scrape fixtures and CPU ML smoke only.
  - Keep live scraping and GPU training out of PR CI unless explicitly scheduled/configured.
  - Add a staged rollout checklist before enabling scrape mode as non-experimental.
- Files likely affected:
  - `README.md`
  - `docs/configuration.md`
  - `docs/testing-plan.md`
  - `docs/multi-milestone-plan.md`
  - `.github/workflows/ci.yml`
- Verification:
  - Docs examples run locally for offline and fixture-backed scrape modes.
  - CI remains deterministic without network, GPU, or local Codex agent.

## Acceptance criteria

- [x] Scraping source policy gate exists and prevents disallowed scraping by default.
- [x] Reddit scraper discovers the daily WSB ticker card from allowed public pages or reports a
      drift/blocked warning.
- [x] AP News scraper returns attributed financial-market evidence from public hub/article HTML.
- [x] Candlecharts adapter either returns contract-valid OHLCV from allowed public data or a clear
      unavailable/widget-only warning.
- [x] X provider uses official API recent search for every discovered ticker and requests the top
      50 relevant results with `sort_order=relevancy` and `max_results=50`.
- [x] `run --source-mode scrape` or equivalent produces Markdown, JSON, and audit artifacts from
      fixture-backed scrape inputs.
- [x] `run --source-mode scrape --live-providers` explicitly calls wired live providers, records
      provider health and normalized evidence, writes `audit/provider-results.json`, and emits
      evidence-only output until live extraction/scoring is enabled.
- [x] Live scraping tests are opt-in, rate-limited, and source-specific.
- [x] ML dataset generation has leakage tests and data-quality gates.
- [x] RTX 3090 training command records reproducible metrics and model artifact metadata.
- [x] ML signal integrates conservatively into technical analysis and scoring with stale/weak-model
      gates.
- [x] Codex fundamental analysis agent lane has a strict request/response schema and fixture-backed
      tests.
- [x] Reports preserve source evidence, provider metadata, confidence inputs, disclaimers, and
      warnings for blocked/stale/drifted providers.
- [x] Existing offline behavior remains deterministic and green.
- [x] Documentation covers configuration, compliance limits, training, and agent workflow.

## Verification commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
python -m pytest tests/test_lane_a_reddit_discovery.py tests/test_lane_a_reddit_evidence.py
python -m pytest tests/test_lane_b_providers.py
python -m pytest tests/test_lane_c_extraction.py
python -m pytest tests/test_lane_d_analysis.py tests/test_lane_d_scoring.py
python -m pytest tests/test_lane_e_cli_e2e.py
python -m pytest tests/test_lane_f_compliance.py tests/test_lane_f_reliability.py
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape
```

Opt-in live checks after source configuration:

```sh
python -m pytest -m live_scraping
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/live-aapl --source-mode scrape --live-providers --cache-dir cache/live
```

Local GPU training smoke after ML implementation:

```sh
python -m nlp_stock_prediction.ml.train --csv data/ml/TSLA.csv --ticker TSLA --output-dir artifacts/ml/TSLA --device auto --as-of 2026-05-11 --max-latest-bar-age-days 5
python -m nlp_stock_prediction.ml.evaluate --model artifacts/ml/TSLA/model.json --csv data/ml/TSLA.csv --ticker TSLA --output artifacts/ml/TSLA/evaluation.json
```

## Decision log

- 2026-05-11-00-00: Keep the default test suite deterministic and network-free; scraping and GPU
  training stay opt-in because source markup, robots policies, and local hardware availability are
  unstable.
- 2026-05-11-00-00: Use official X API recent search for X stock-news/social evidence instead of
  browser scraping. Read-only recent-search calls use the app-only Bearer Token; API Key and API
  Secret are documented for app identity and token regeneration.
- 2026-05-12-00-00: Use `https://api.x.com/2/tweets/search/recent` for X recent search and default
  production evidence requests to `$TICKER lang:en -is:retweet`, `sort_order=relevancy`, and
  `max_results=50`. Avoid `sort_order=recency` in production because smoke-test results were too
  noisy for stock-prediction evidence.
- 2026-05-11-00-00: Treat Candlecharts as a feasibility-gated source because the public live chart
  page appears to rely on TradingView for chart tracking; do not scrape embedded third-party widget
  internals without approval.
- 2026-05-11-00-00: Use local ML only as an analysis input with model provenance and confidence
  gates, not as an autonomous recommendation engine.
- 2026-05-11-00-00: Keep the Codex fundamental analysis lane schema-first and fixture-backed, with
  live/local agent execution optional until a supported programmatic runner is confirmed.
- 2026-05-12-00-00: Use one git worktree per parallel workstream and assign disjoint write sets to
  Codex subagents. Shared contracts and base helpers land before adapter work, and the coordinator
  owns integration conflicts, final E2E, and cross-workstream report consistency.
- 2026-05-12-00-00: Per user direction, follow-up implementation work after the closed PR is being
  done directly on `feature/release-v1` without git worktrees or subagents. Keep edits linear,
  reviewed locally, and verified before any ACP.
- 2026-05-12-00-00: Keep live-provider scrape orchestration explicit behind
  `--source-mode scrape --live-providers`. The live path records provider evidence, health, and
  audit artifacts, but suppresses fixture sidecars and emits evidence-only output until live
  extraction, analysis, and scoring are enabled intentionally.

## Progress log

- 2026-05-11-00-00: Created plan from requested scrape-only provider direction, RTX 3090 local ML
  training requirement, and Codex-agent fundamental-analysis requirement. No implementation had
  started at the time this plan was created.
- 2026-05-11-00-00: Superseded the earlier X local-capture idea after the user provided X Developer
  App credentials and selected the official API path.
- 2026-05-11-00-00: Initially updated X milestone to official API recent search using both
  relevance and newest-first result orders for each ticker, backed by `.env` X App credentials.
  This was superseded by the next entry.
- 2026-05-11-00-00: Superseded recency for production after smoke-test result quality was too noisy.
  Production X evidence now uses `$TICKER lang:en -is:retweet`, `sort_order=relevancy`, and
  `max_results=50`.
- 2026-05-12-00-00: Implemented and committed the X provider-level defaults in `168141d`
  (`Configure X API relevancy provider path`): provider endpoint, query defaults, provenance,
  `.env.example`, docs, roadmap/testing-plan updates, and contract tests. Verification passed with
  `257 passed, 3 skipped`, Ruff clean, Mypy clean, and `git diff --check` clean. At that point,
  live report orchestration and the broader scraping/ML/Codex-agent milestones were still planned
  work.
- 2026-05-12-00-00: Reworked this plan for parallel development with git worktrees, Codex subagent
  packet templates, workstream IDs, branch names, ownership boundaries, and merge order.
- 2026-05-12-00-00: Tagged and pushed `parallel-worktree-start-2026-05-12` at `81a9bd1`, then
  launched six Codex worker worktrees from that exact baseline:
  `codex/scraping-foundation`, `codex/reddit-scraper`, `codex/apnews-scraper`,
  `codex/candlecharts-feasibility`, `codex/ml-technical-analysis`, and
  `codex/fundamental-agent`.
- 2026-05-12-00-00: Integrated the six pushed workstreams into coordinator branch
  `codex/parallel-integration`, including scraping policy/shared HTML helpers, Reddit public-page
  scraping, AP News public HTML scraping, Candlecharts feasibility probing, isolated ML dataset and
  training foundations, and fixture-backed fundamental-agent schemas. Coordinator review adjusted
  Reddit blocked-policy warnings to match the shared W1 semantics.
- 2026-05-12-00-00: Final integration verification passed with `299 passed, 3 skipped`,
  `ruff check src tests`, `ruff format --check src tests`, `mypy .`, and `git diff --check`.
- 2026-05-12-00-00: Started the six-step direct implementation on `feature/release-v1` without
  worktrees/subagents. Added shared HTML response compatibility for scraper cleanup, experimental
  `run --source-mode scrape`, fixture-backed provider orchestration for Reddit public pages, AP
  News, Candlecharts feasibility, and X API-shaped relevant search, degraded provider reporting in
  report health plus `audit/provider-results.json`, fixture-backed e2e coverage, and
  source-specific opt-in live smoke tests. Narrow verification passed for CLI/e2e/AP/Candlecharts/
  Reddit/scraping tests with `48 passed`.
- 2026-05-12-00-00: Completed the direct ML/fundamental integration slice. Added report-facing
  `TechnicalMlSignal` and `FundamentalAgentSignal` sidecars, conservative ML scoring penalties and
  gates, fixture-backed TSLA sidecars in experimental scrape mode, Markdown rendering, provider
  result audit coverage, refreshed analysis-context audit records, and focused tests for ML signal
  conversion, ML score gates, fundamental-agent report integration, and scrape-mode E2E sidecars.
- 2026-05-12-00-00: Completed the remaining ML acceptance items. Added optional as-of/freshness
  gates for local OHLCV CSV datasets, explicit lookahead/stale/partial-adjustment tests, future-bar
  feature-isolation coverage, richer training runtime/split/metric metadata, artifact SHA-256
  recording, and a CLI training command smoke test that verifies model, metrics, metadata, device,
  and usage-limitation outputs.
- 2026-05-12-00-00: Addressed review findings in the ML acceptance slice: datetime as-of checks now
  compare aware instants instead of calendar dates, stale-age config requires `as_of`, training
  metadata is rebuilt through validated contracts, CUDA detection is recorded separately from the
  CPU execution backend, artifact hashes are recomputed in tests, subprocess ML tests use a minimal
  environment, and the GPU/evaluation smoke commands now match the actual CLI.
- 2026-05-12-00-00: Merged the opt-in live-provider orchestration PR into `feature/release-v1`.
  The branch now has the live scrape path for Reddit, AP News, Candlecharts, and X recent search,
  with live normalized evidence, provider health, provider-result audit artifacts, and evidence-only
  output until live extraction/scoring is implemented.
- 2026-05-12-00-00: Cross-checked the merged branch against active plans, source docs, and CLI
  behavior. Updated stale plan/docs language that still described live orchestration as unwired, and
  narrowed the no-mode CLI error to point users at the three explicit local modes.
- 2026-05-12-00-00: Verification after the cross-check passed with targeted CLI tests
  (`31 passed`), full deterministic pytest (`327 passed, 7 skipped`), Ruff check, Ruff format
  check, Mypy, `git diff --check`, and a temp-directory `--source-mode scrape` CLI smoke verifying
  `report.md`, `report.json`, and `audit/provider-results.json`.
