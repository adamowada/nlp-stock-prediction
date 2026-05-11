# Frozen Phase 0 Contracts

Phase 0 settles the public contract surface for later worktree lanes. Concrete providers,
analysis formulas, report rendering, and live integrations are intentionally out of scope until
later phases.

## Python Package And CLI

- Source layout: `src/nlp_stock_prediction/`.
- Canonical invocation: `python -m nlp_stock_prediction`.
- `run` accepts `--date`, `--output`, `--capital`, `--risk-profile`, `--fixture-dir`,
  `--cache-dir`, and `--offline`.
- During Phase 0, `run` validates the command shape and exits with code `3` because report
  generation is not implemented yet.
- No console script is frozen in Phase 0.

## Contract Modules

- `contracts.base`: common Pydantic base, ticker symbols, confidence, score, and JSON aliases.
- `contracts.enums`: stable enum taxonomy for providers, warnings, sources, instruments,
  horizons, risk profiles, and recommendation actions.
- `contracts.provenance`: `SourceProvenance`, `ProviderHealth`, `ProviderWarning`,
  `EvidenceReference`, and `DataReference`.
- `contracts.discovery`: Devvit ticker-card candidates and `TickerDiscoveryResult`.
- `contracts.evidence`: normalized external evidence records.
- `contracts.extraction`: evidence-backed strategy extractions and clusters.
- `contracts.analysis`: technical, fundamental, sector, macro, and combined analysis contracts.
- `contracts.recommendation`: score, risk, and `TradeCandidate` contracts.
- `contracts.report`: Markdown/JSON report spine and audit manifest contracts.
- `contracts.providers`: provider request/result envelopes and provider protocols only.
- `contracts.fixtures`: raw and normalized fixture manifests.

## Provider Semantics

Every provider method returns `ProviderResult[T]`. Expected provider or upstream failures should
be represented as `ProviderStatus`, `ProviderWarning`, and nullable `data`, not raised exceptions.
Exceptions are reserved for programmer errors or invalid contract usage.

Provider contracts return normalized facts, evidence, candles, metrics, and series only. They do
not return `TechnicalAnalysis`, `FundamentalAnalysis`, `SectorContext`, `MacroContext`, or other
Lane D analysis outputs.

`ProviderResult[T]` requires the result provider/status to match its `ProviderHealth`, requires data
for `ok` results, and requires at least one warning for non-`ok` results.

All normalized external data carries `SourceProvenance` with provider name, source kind,
retrieval method, fetched timestamp, source URL or permalink when available, raw identifier,
raw snapshot ID, freshness status, cache key, query, and provider metadata.
Metadata fields must be JSON-serializable.

Frozen warning codes are:

- `missing_credentials`
- `auth_failed`
- `rate_limited`
- `quota_exceeded`
- `timeout`
- `upstream_unavailable`
- `malformed_response`
- `schema_mismatch`
- `scraping_drift`
- `stale_data`
- `no_data`
- `partial_data`
- `llm_schema_invalid`
- `llm_evidence_mismatch`
- `unsupported_claim`

## Evidence And Recommendation Invariants

- Valid ticker discovery requires exactly six unique tickers in first-seen order.
- Invalid ticker discovery results must include warnings.
- Strategy extractions and clusters must cite normalized evidence.
- Actionable trade candidates must cite evidence and include score, risk, invalidation, and
  disclaimer references.
- Reports must include exactly six ticker sections matching ticker discovery order.
- Reports without trade candidates must include a no-trade summary.
- Valid ticker discovery must include candidate records and a raw snapshot ID.

## Fixture Shape

Fixture manifests are modeled in `contracts.fixtures`. Later tests should store fixtures under
`tests/fixtures/` with raw provider snapshots, normalized provider results, evidence,
extraction, analysis, scoring, and expected report artifacts separated by scenario.

Core scenario names to use first:

- `normal_six_ticker_day`
- `duplicate_tickers_in_devvit_card`
- `malformed_reddit_html`
- `partial_provider_outage`
- `missing_credentials`
- `rate_limited_provider`
- `stale_market_data`
- `conflicting_signals`
- `unsupported_llm_claim`
- `llm_sarcasm_or_joke_risk`
- `no_qualified_trade`
- `qualified_stock_idea`
- `qualified_defined_risk_options_idea`

## Phase Boundary

Phase 0 is complete when this contract surface exists and imports. Phase 1 remains responsible for
the comprehensive schema, provider-contract, CLI, and report-shape test harness. Parallel
implementation lanes remain blocked until Phase 1 is complete and the full contract gate is met.
