# Multi-Milestone Plan

## Summary

Build a TDD-first Python CLI that generates one daily, evidence-grounded stock opportunity report for a retail trader with a small account.

The app discovers the six tickers surfaced by the r/wallstreetbets Devvit ticker card, gathers recent public discussion and news, extracts discussed trading strategies with evidence, combines that signal with technical, fundamental, sector, and macro analysis, then writes Markdown and JSON reports.

The v1 posture is exploratory but auditable. The app may surface speculative stock/options ideas, but each recommendation must include confidence, source evidence, risks, invalidation criteria, and a clear non-advice disclaimer. No real-money brokerage execution is included.

## Milestone 1: Project foundation

- Create a Python package with a CLI entrypoint, typed modules, config loading, logging, and test scaffolding.
- Add core developer tooling: `pytest`, `ruff`, `mypy`, and deterministic fixtures.
- Define provider interfaces before implementation: Reddit, X/Twitter/social, news, market data, fundamentals, macro data, LLM extraction, and report writing.
- Establish initial CLI shape using the Python module name `nlp_stock_prediction`:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/
python -m nlp_stock_prediction run --capital 1000 --risk-profile exploratory
```

If a console script is added later, it should remain a thin alias for the module command and be documented alongside it.

## Milestone 2: Reddit WSB ticker discovery

- Fetch the public r/wallstreetbets page or current sticky post.
- Scope parsing to the Devvit ticker table subtree before extracting ticker symbols.
- Extract ticker candidates from `ticker-container-*` identifiers while preserving raw order and identifiers.
- Normalize to unique tickers by first-seen order; discovery succeeds only when exactly six unique tickers remain.
- Treat malformed HTML, duplicate-only shortages, or unexpected extra unique ticker candidates as explicit discovery validation warnings/errors rather than silently guessing.
- Store raw HTML/API snapshots for auditability and fixture regeneration.

## Milestone 3: Reddit discussion retrieval and filtering

- For each discovered ticker, retrieve recent WSB posts using subreddit search sorted by newest.
- Retrieve current daily-thread comments sorted by newest with shallow comment-tree depth.
- Normalize posts and comments into evidence records containing source type, Reddit ID, author hash, created timestamp, permalink, score, text, and matched ticker.
- Apply high-precision ticker matching, including special handling for short symbols such as `MU`, `AI`, `ON`, and `IT`.

## Milestone 4: Strategy extraction and clustering

- Use schema-constrained LLM extraction to identify discussed strategies, not direct recommendations.
- Require fields such as ticker, label, direction, instrument, position type, time horizon, catalyst, risk/hedge, slang terms, evidence quotes, confidence, and sarcasm/joke risk.
- Reject unsupported claims: every extracted strategy must cite one or more normalized evidence records.
- Cluster near-duplicates by ticker, direction, instrument, time horizon, and catalyst.

## Milestone 5: Social, news, market, fundamental, and macro data

- Query X/Twitter recent search for cashtags and ticker terms as a social/catalyst signal, prioritizing queries such as `$TSLA lang:en`.
- Query a configured public news provider for recent company or ticker headlines/articles; if no news provider is configured or available, emit an explicit provider warning instead of treating X/Twitter as news.
- Pull daily candle data and company overview/fundamental fields from free official providers where available.
- Use SEC EDGAR APIs as a supplemental source for filings and company facts.
- Use FRED for macro series such as rates, CPI, unemployment, GDP, and yield-curve indicators.
- Cache provider responses by date, ticker, and source to reduce rate-limit pressure.

## Milestone 6: Analysis engine

- Technical analysis: trend, support/resistance, volume, volatility, RSI/MACD/SMA-style indicators, recent gap behavior, and candlestick summary.
- Company fundamentals: valuation, profitability, growth, balance sheet risk, earnings timing, and notable filings/metrics.
- Sector analysis: compare company metrics against configured sector peers or sector ETF proxies where direct peer data is unavailable.
- Macro analysis: summarize whether current macro conditions support or conflict with each strategy horizon.

## Milestone 7: Recommendation scoring

- Score each candidate strategy across Reddit strength, social/news catalyst strength, technical alignment, company fundamentals, sector context, macro context, liquidity/risk suitability, and contradiction penalties.
- Use an exploratory default: surface high-upside ideas when evidence is interesting, but clearly mark speculative ideas when fundamentals, macro, or technicals conflict.
- Emit zero or more confident strategies. A strategy qualifies only if it has a clear thesis, instrument, timeframe, entry logic, invalidation level, max-loss estimate, and enough evidence to exceed the configured confidence threshold.
- Default risk controls: no margin, no naked options, long shares or defined-risk options only, max 1% account risk per idea when account capital is provided, and percentage-only sizing when capital is omitted.

## Milestone 8: Report generation

- Produce one Markdown report and one JSON report per run.
- Markdown structure:
  - Header with date, data freshness, provider warnings, and disclaimer.
  - Six ticker sections.
  - Each section includes Reddit strategies, social/news summary, technical analysis, company fundamentals, sector fundamentals, macro context, and per-ticker opportunity notes.
  - Final section lists confident trading strategies or explicitly states that none qualified.
- JSON structure mirrors the report and preserves raw evidence IDs, confidence scores, provider metadata, and recommendation inputs.

## Milestone 9: Reliability, compliance, and observability

- Add retry/backoff, rate-limit handling, provider health messages, and graceful degradation when a source is unavailable.
- Clearly distinguish official API data from public scraping fallback data.
- Add an audit trail for raw snapshots, normalized evidence, extracted strategies, scoring inputs, and final reports.
- Include legal/financial disclaimers in generated reports and prohibit auto-trading in v1.

## Public interfaces and types

- Core models:
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
- Provider adapters:
  - `RedditProvider`
  - `XProvider`
  - `NewsProvider`
  - `MarketDataProvider`
  - `FundamentalsProvider`
  - `MacroProvider`
  - `LLMExtractor`
- Report outputs:
  - `reports/YYYY-MM-DD/report.md`
  - `reports/YYYY-MM-DD/report.json`
  - `reports/YYYY-MM-DD/audit/`

## Assumptions

- The repo is greenfield, so the application structure can be created from scratch.
- The first complete version is a local Python CLI that runs manually.
- Reports are written as Markdown plus structured JSON.
- Recommendations may include stocks and defined-risk options.
- The risk posture is exploratory, with explicit evidence, risk notes, and disclaimers.
- Use free official data sources first, with permitted public scraping fallback where necessary.
