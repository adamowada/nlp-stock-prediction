# Contracts-First Parallel V1 Plan

## Summary

Build a TDD-first Python CLI that generates one daily, evidence-grounded stock opportunity report for a retail trader with a small account.

The product goal is an app that discovers the six tickers surfaced by the r/wallstreetbets Devvit daily ticker card, gathers recent public discussion and news, always uses X API recent search for the top 50 relevant stock-news/social posts for each discovered ticker when live provider orchestration is enabled, extracts discussed trading strategies with evidence, combines that signal with technical, fundamental, sector, and macro analysis, then writes Markdown and JSON reports.

The v1 posture is exploratory but auditable. The app may surface speculative stock/options ideas, but each recommendation must include confidence, source evidence, risks, and invalidation criteria.

This roadmap is organized for git worktrees and Codex subagents. Contracts were settled single-threaded first; implementation then happened in parallel lanes from the same frozen contract commit; Phase 3 now returns to a coordinated integration and hardening flow.

Current state: Phase 0 contract settlement, Phase 1 contract test harness, Phase 2 parallel implementation, Phase 3 local V1 acceptance, the follow-on scrape/ML/agent integration slices, and Phase 4 local TimesFM technical analysis are complete on `feature/release-v1` and the TimesFM work branch. The CLI generates deterministic offline Markdown, JSON, and audit artifacts; `--source-mode scrape` generates fixture-backed provider/audit artifacts; and `--source-mode scrape --live-providers` explicitly calls live Reddit/AP/Candlecharts/X providers for evidence collection without generating live recommendations. Live API and scraping checks remain opt-in, and live LLM smoke remains reserved until a live adapter and credential contract exists. The completed Phase 4 acceptance record is `plans/timesfm-technical-analysis.md`; `docs/configuration.md` is the user workflow for clean Windows setup, training, evaluation, report attachment, and troubleshooting. See `plans/phase-3-integration-live-smoke.md` for the Phase 3 acceptance record and `plans/scraping-ml-agent-analysis.md` for the follow-on scrape/ML/agent record.

## Phase Status

- **Phase 0:** Complete. Public contracts are implemented under `src/nlp_stock_prediction/contracts/` and documented in `docs/contracts.md`.
- **Phase 1:** Complete. Comprehensive schema, import, CLI, provider-contract, fixture, and report-shape tests are implemented.
- **Phase 2:** Complete. Lanes A-F are merged and verified with deterministic tests, lint, format check, typecheck, CLI help, and offline report smoke coverage.
- **Phase 3:** Complete. Integration hardening, CLI/configuration polish, live smoke readiness, report QA, failure drills, and final V1 acceptance are complete for the local CLI.
- **Phase 4:** Complete. Built, fine-tuned, evaluated, score-guarded, documented, and accepted a local Google TimesFM 2.5 technical-analysis model path for a given stock, using native Windows RTX 3090 training as the primary path.

## Delivery Strategy

1. **Phase 0: Contract settlement, single-threaded.**
   - Create the Python package scaffold only as needed to define public contracts.
   - Freeze shared models, provider protocols, report shapes, CLI behavior, fixture conventions, provider health/error semantics, and test markers.
   - Record decisions in the active plan's decision log before parallel implementation begins.
2. **Phase 1: Contract test harness, single-threaded.**
   - Add schema, import, CLI, provider-contract, and report-shape tests that define the expected public behavior.
   - These tests may fail for unimplemented provider behavior, but shared contracts must import, serialize, and typecheck.
3. **Phase 2: Parallel implementation lanes.**
   - Create one git worktree per lane from the frozen contract commit.
   - Assign each Codex subagent exactly one lane and an explicit ownership boundary.
   - Lane agents may add private helpers freely but must not change frozen shared contracts directly.
4. **Phase 3: Integration and live smoke.**
   - Harden the integrated CLI and report pipeline from one coordinator thread.
   - Keep deterministic fixture-backed report generation green before marked live API or live scraping checks.
   - Resolve contract gaps single-threaded and record decisions in the active Phase 3 plan.
5. **Phase 4: Local TimesFM technical analysis.**
   - Use `plans/timesfm-technical-analysis.md` as the completed phase acceptance record.
   - Use `docs/configuration.md` as the detailed Windows workflow and troubleshooting guide.
   - Keep TimesFM dependencies optional so deterministic tests remain lightweight and network-free.
   - Run local Windows RTX 3090 smoke, LoRA fine-tuning, rolling evaluation, and report integration.
   - Treat TimesFM as a technical-analysis sidecar that works through the existing evidence, risk, and scoring gates.

## Contract Gate

Parallel work starts only after the contract gate is complete. The Phase 1 contract gate is now complete on this branch.

The contract gate requires:

- Canonical CLI invocation documented and testable:

```sh
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --capital 1000 --risk-profile exploratory --offline
```

- Public models defined at API boundaries with Pydantic or typed dataclasses where appropriate.
- Provider protocols defined for Reddit, X/social, news, market data, fundamentals, macro data, and LLM extraction.
- Report contracts defined for `reports/YYYY-MM-DD/report.md`, `reports/YYYY-MM-DD/report.json`, and `reports/YYYY-MM-DD/audit/`. Phase 1 froze the JSON spine, audit manifest shape, and minimal Markdown section outline; Phase 2 implemented deterministic rendering and file writing for offline runs.
- Fixture shapes defined for raw provider snapshots, normalized evidence, extraction outputs, analysis contexts, scoring inputs, and expected reports.
- Provider health and failure semantics defined for stale data, rate limits, missing credentials, upstream outages, malformed responses, and scraping drift.
- Test marker taxonomy aligned with `docs/testing-plan.md`: `unit`, `schema`, `contract`, `integration`, `live_api`, `live_scraping`, `llm`, and `e2e`.
- Contract tests committed before implementation lanes begin.

If a lane discovers a missing or wrong shared contract, that lane pauses. The contract is revised single-threaded, the decision log is updated, and affected worktrees rebase onto the revised contract commit.

## Public Interfaces To Freeze

Core models:

- `TickerDiscoveryResult`
- `SourceEvidence`
- `StrategyExtraction`
- `StrategyCluster`
- `TechnicalAnalysis`
- `FundamentalAnalysis`
- `SectorContext`
- `MacroContext`
- `TradeCandidate`
- `DailyReport`
- `ProviderHealth`
- `ProviderWarning`

Provider adapters:

- `RedditProvider`
- `XProvider`
- `NewsProvider`
- `MarketDataProvider`
- `FundamentalsProvider`
- `MacroProvider`
- `LLMExtractor`

Shared behavior:

- Preserve source provenance for all external data: provider name, fetched timestamp, permalink or source URL, raw identifier, freshness, and relevant provider metadata.
- Clearly separate observed Reddit, X/Twitter, and news discussion from the app's own analysis and recommendations.
- Treat ticker discovery as valid only when the Devvit ticker-card parser produces exactly six unique tickers by first-seen order.
- Reject unsupported strategy and recommendation claims unless they cite normalized evidence records.
- Reject qualified recommendations that fail risk gates, fail score gates, or do not meet their configured score threshold.
- Keep report sections, recommendation IDs, candidate tickers, and summaries for reports where nothing qualifies internally consistent.
- Degrade gracefully when providers fail and surface failures in report warnings or logs.

## Parallel Implementation Lanes

### Lane A: Reddit Discovery And Evidence

Ownership:

- Devvit ticker-card discovery.
- Reddit post and daily-thread retrieval.
- Ticker matching and evidence normalization.
- Reddit fixtures and contract tests.

Deliverables:

- Extract ticker candidates from `ticker-container-*` identifiers while preserving raw order and identifiers.
- Normalize to exactly six unique tickers by first-seen order.
- Handle malformed HTML, duplicate-only shortages, and unexpected extra unique ticker candidates with explicit discovery validation warnings/errors.
- Normalize posts and comments into evidence records containing source type, Reddit ID, author hash, created timestamp, permalink, score, text, matched ticker, provider metadata, and freshness.
- Apply high-precision ticker matching, including short-symbol handling for examples such as `MU`, `AI`, `ON`, and `IT`.

### Lane B: Social, News, Market, Fundamentals, And Macro Providers

Ownership:

- X/social provider.
- News provider.
- Market data provider.
- Fundamentals provider.
- SEC EDGAR and FRED integrations.
- Provider caching and provider-level fixtures.

Deliverables:

- Query X API recent search for each discovered ticker as a stock-news/social/catalyst signal.
  For every ticker, request the top 50 relevant results with `$TICKER lang:en -is:retweet`,
  `sort_order=relevancy`, and `max_results=50`.
- Query a configured public news provider for recent company or ticker headlines/articles.
- Pull daily candle data and company overview/fundamental fields from free official providers where available.
- Use SEC EDGAR APIs as supplemental filing and company-facts sources.
- Use FRED for macro series such as rates, CPI, unemployment, GDP, and yield-curve indicators.
- Cache provider responses by date, ticker, and source to reduce rate-limit pressure.
- Emit explicit provider warnings when a provider is unavailable, stale, unauthenticated, or unconfigured.

### Lane C: Strategy Extraction And Clustering

Ownership:

- Schema-constrained LLM extraction.
- Evidence quote validation.
- Unsupported-claim rejection.
- Strategy clustering.
- LLM fixture and smoke-test scaffolding.

Deliverables:

- Extract discussed strategies, not direct app recommendations.
- Require fields such as ticker, label, direction, instrument, position type, time horizon, catalyst, risk/hedge, slang terms, evidence quotes, confidence, and sarcasm/joke risk.
- Ensure every extracted strategy cites one or more normalized evidence records.
- Cluster near-duplicates by ticker, direction, instrument, time horizon, and catalyst.
- Include tests for unsupported claims, missing evidence, sarcasm/joke risk, and near-duplicate examples such as "buy calls," "weekly calls," and "calls into earnings."

### Lane D: Analysis, Risk, And Recommendation Scoring

Ownership:

- Technical analysis.
- Fundamental, sector, and macro context synthesis.
- Risk gates and recommendation scoring.
- Confidence inputs and contradiction penalties.

Deliverables:

- Technical analysis covers trend, support/resistance, volume, volatility, RSI/MACD/SMA-style indicators, recent gap behavior, and candlestick summary.
- Company fundamentals cover valuation, profitability, growth, balance sheet risk, earnings timing, and notable filings/metrics.
- Sector context compares company metrics against configured sector peers or sector ETF proxies where direct peer data is unavailable.
- Macro context summarizes whether current macro conditions support or conflict with each strategy horizon.
- Scoring covers Reddit strength, social/news catalyst strength, technical alignment, company fundamentals, sector context, macro context, liquidity/risk suitability, and contradiction penalties.
- Default risk controls: no margin, no naked options, long shares or defined-risk options only, max 1% account risk per idea when account capital is provided, and percentage-only sizing when capital is omitted.
- Emit zero or more confident strategies; days with no qualified strategies are valid outcomes.

### Lane E: Report Rendering, Audit Artifacts, And CLI Orchestration

Ownership:

- CLI command wiring.
- Markdown report renderer.
- JSON report renderer.
- Audit artifact writing.
- Fixture-backed end-to-end report generation.

Deliverables:

- Produce one Markdown report and one JSON report per run.
- Include date, data freshness, and provider warnings in the report header.
- Include six ticker sections with Reddit strategies, social/news summary, technical analysis, company fundamentals, sector fundamentals, macro context, and per-ticker opportunity notes.
- Include a final section listing confident trading strategies or explicitly stating that none qualified.
- Ensure JSON mirrors the report and preserves raw evidence IDs, confidence scores, provider metadata, and recommendation inputs.
- Write audit artifacts for raw snapshots, normalized evidence, extracted strategies, scoring inputs, and final reports.

### Lane F: Reliability, Observability, CI, And Live Smoke

Ownership:

- Retry/backoff and rate-limit behavior.
- Provider health messages and graceful degradation.
- CI marker wiring.
- Live API and live scraping smoke checks.
- Provider reliability and failure handling.

Deliverables:

- Add retry/backoff, rate-limit handling, provider health messages, and graceful degradation when a source is unavailable.
- Clearly distinguish official API data from public scraping fallback data.
- Keep default tests deterministic and fixture-backed.
- Mark live dependency tests separately and require explicit credentials, network access, and quota controls.
- Ensure provider failures degrade cleanly and remain visible in report and audit output.

## Worktree And Subagent Protocol

- Start every implementation worktree from the same frozen contract commit.
- Use a branch naming pattern such as `codex/lane-a-reddit-evidence`, `codex/lane-b-providers`, and so on.
- Give each subagent the frozen contract commit, its lane ownership, expected verification commands, and a reminder that other agents are working in parallel.
- Subagents must not revert or overwrite changes outside their ownership boundary.
- Subagents should list changed files, tests added, verification run, known gaps, and any contract issues in their final handoff.
- Merge order should prefer lower-level dependencies first: contracts, providers/evidence, extraction, analysis/scoring, report/CLI, reliability/live smoke.
- If two lanes need the same helper, settle its public shape single-threaded before either lane depends on it.

## Verification Commands

Canonical verification commands:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
python -m pytest -m e2e
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape
```

Opt-in live smoke selections:

```sh
python -m pytest -m live_api
python -m pytest -m live_scraping
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/live-aapl --source-mode scrape --live-providers --cache-dir cache/live
```

## Phase 1 Contract-Gate Acceptance Criteria

- The contract gate is complete before parallel implementation begins.
- Each lane has clear ownership and can be assigned to a separate git worktree and Codex subagent.
- Public contracts preserve evidence, provider metadata, and confidence inputs.
- Contract and fixture-backed tests define expected behavior before implementation fills it in.
- Default tests remain deterministic and do not require live credentials or network access.
- Live API, live scraping, and e2e markers are registered; fixture-backed e2e coverage now runs while live checks remain opt-in.
- Provider failures have coherent result-envelope semantics and are visible as warnings/health records.
- Documentation is updated when commands, configuration, behavior, or report structure changes.

## Phase 3 V1 Acceptance Criteria

- Full fixture-backed report generation produces Markdown, JSON, and audit artifacts.
- Live API and live scraping tests are opt-in, marked, and implemented by their provider/reliability lanes.
- Provider failures degrade gracefully through actual provider adapters and are visible in reports or logs.
- Recommendation scoring has fixture-backed outcomes where nothing qualifies, conflicting-evidence, and qualified-strategy outcomes.
- The explicit live-provider scrape path preserves provider health, warnings, normalized evidence,
  and audit artifacts without generating live recommendations.
- CLI configuration, report quality, failure drills, and final clean-checkout verification meet
  `plans/phase-3-integration-live-smoke.md`.

## Assumptions

- The repo began as a greenfield project, so the application structure was created from scratch.
- The first complete version is a local Python CLI that runs manually.
- Reports are written as Markdown plus structured JSON.
- Recommendations may include stocks and defined-risk options.
- The risk posture is exploratory, with explicit evidence and risk notes.
- Use free official data sources first, with permitted public scraping fallback where necessary.
- Contract settlement is intentionally single-threaded; implementation after the contract gate is intentionally parallel.
