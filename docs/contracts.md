# Frozen Contract Gate

Phase 0 settled the public contract surface and Phase 1 added the deterministic test harness for
later worktree lanes. Phase 2 integrated concrete providers, analysis helpers, scoring, report
rendering, audit writing, reliability helpers, and opt-in live smoke scaffolding against those
contracts. Phase 3 owns hardening the integrated CLI and live smoke surface without weakening the
frozen contract invariants.

## Python Package And CLI

- Source layout: `src/nlp_stock_prediction/`.
- Canonical invocation: `python -m nlp_stock_prediction`.
- `run` accepts `--date`, `--output`, `--capital`, `--risk-profile`, `--fixture-dir`,
  `--cache-dir`, and `--offline`.
- With `--offline`, `run` writes a deterministic fixture-backed report bundle. Without `--offline`,
  it currently exits with code `3` because live-provider report orchestration is not enabled yet.
- No console script is frozen in the contract gate.
- Environment variables and optional ignored `.env` files are documented in
  `docs/configuration.md`; the default offline path does not require credentials or network access.

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
- `contracts.report`: Markdown/JSON report spine, report evidence-source map, and audit manifest
  contracts.
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
for `ok` results, and requires at least one warning for non-`ok` results. Warning provider names
must be absent or match the result provider. `empty` results must not carry data and must include a
`no_data` warning.

All normalized external data carries `SourceProvenance` with provider name, source kind,
retrieval method, fetched timestamp, source URL or permalink when available, raw identifier,
raw snapshot ID, freshness status, cache key, query, and provider metadata.
Metadata fields must be JSON-serializable.
External provenance must include a source URL or permalink, raw identifier, raw snapshot ID, explicit
freshness status, and timezone-aware timestamps. Internal/derived analysis provenance is the explicit
exception.

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
- High sarcasm/joke risk and conflicting source evidence must be surfaced as extraction/cluster
  warnings and must not qualify silently as clean trade candidates.
- Evidence quote spans must be monotonic when both start and end offsets are provided.
- Score breakdowns must include at least one component.
- Actionable trade candidates must cite evidence and include score, risk, invalidation, and
  disclaimer references.
- Qualified trade candidates must pass risk gates, have no failed risk or score gates, and meet or
  exceed the configured score threshold.
- V1 disclaimers must remain educational-only, not financial advice, and no-auto-trading.
- Reports must include exactly six ticker sections matching ticker discovery order.
- Reports may include `evidence_sources`; when present, every cited evidence ID in ticker sections,
  strategy clusters, analysis components, trade candidates, and score inputs must resolve to a
  normalized evidence source with provenance.
- Reports without trade candidates must include a no-trade summary.
- Report trade candidates must use discovered tickers, have unique candidate IDs, match the report
  disclaimer, and be referenced by exactly one matching ticker section.
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
- Minimal Markdown report outline, including required header, freshness, provider warnings,
  per-ticker subsections, final qualified-strategy or no-trade section, disclaimer, and audit
  artifact sections.

## Fixture Shape

Fixture manifests are modeled in `contracts.fixtures`. Later tests should store fixtures under
`tests/fixtures/` with raw provider snapshots, normalized provider results, evidence,
extraction, analysis, scoring, and expected report artifacts separated by scenario.
Manifest scenarios must match nested raw and normalized fixture scenarios, and raw fixture request
dates must match the manifest run date.

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

Phase 0 and Phase 1 are complete, and Phase 2 implementation lanes have been integrated. Phase 3 is
active on `feature/integration-and-hardening`; Stages 0-5 are complete, with final V1 acceptance
still pending. Shared public contracts should stay stable unless a
single-threaded contract revision is recorded in the active Phase 3 plan.
