# Frozen Contract Gate

Phase 0 settled the public contract surface and Phase 1 added the deterministic test harness for
later worktree lanes. Concrete providers, analysis formulas, report rendering, and live integrations
are intentionally out of scope until later phases.

## Python Package And CLI

- Source layout: `src/nlp_stock_prediction/`.
- Canonical invocation: `python -m nlp_stock_prediction`.
- `run` accepts `--date`, `--output`, `--capital`, `--risk-profile`, `--fixture-dir`,
  `--cache-dir`, and `--offline`.
- During the contract-gate phase, `run` validates the command shape and exits with code `3` because
  report generation is not implemented yet.
- No console script is frozen in the contract gate.

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
- Evidence quote spans must be monotonic when both start and end offsets are provided.
- Score breakdowns must include at least one component.
- Actionable trade candidates must cite evidence and include score, risk, invalidation, and
  disclaimer references.
- V1 disclaimers must remain educational-only, not financial advice, and no-auto-trading.
- Reports must include exactly six ticker sections matching ticker discovery order.
- Reports without trade candidates must include a no-trade summary.
- Valid ticker discovery must include candidate records and a raw snapshot ID.

## Phase 1 Contract Harness

The Phase 1 harness covers:

- Public contract imports and `__all__` re-export stability.
- Pydantic schema invariants for provenance, evidence, extraction, analysis, scoring,
  recommendations, reports, and fixture manifests.
- CLI parser and module-entrypoint behavior for the canonical `python -m nlp_stock_prediction`
  invocation.
- Deterministic fake-provider protocol behavior for Reddit, X/social, news, market data,
  fundamentals, macro, and LLM extraction adapters.
- Daily report shape, including six ticker sections, no-trade summaries, provider health, data
  freshness, disclaimers, audit manifests, and JSON round trips.

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

Phase 0 and Phase 1 are complete on this branch. Phase 2 parallel implementation lanes should start
from the Phase 1 contract-gate commit and treat the shared contracts as frozen unless a
single-threaded contract revision is recorded.
