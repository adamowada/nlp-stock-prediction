# Testing Plan

The test strategy should protect the prediction research assistant's core promises: provenance,
repeatability, conservative claims, and graceful degradation.

## Default Suite

The default suite must be deterministic and offline.

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
python -m pytest -m "not live_api and not live_scraping and not codex_smoke"
ruff check .
ruff format --check .
mypy .
```

Default tests must not require:

- live credentials;
- internet access;
- CUDA;
- optional TimesFM dependencies;
- browser automation;
- visual rendering.

## Test Layers

### Unit

Pure logic tests for:

- instrument normalization;
- alias and ambiguity resolution;
- provider ID uniqueness;
- tradability/access evidence source requirements;
- universe request and watchlist validation;
- source claim extraction helpers;
- freshness classification;
- signal normalization;
- prediction scoring;
- report language guardrails.

### Schema

Round-trip tests for Pydantic contracts and SQLite rows:

- Instrument;
- ToolRun;
- Artifact;
- EvidenceItem;
- PredictionCandidate;
- TechnicalPackage;
- Report;
- PlanningState.

### SQLite

Database tests must cover:

- idempotent initialization for planning and research databases;
- migration ordering;
- uniqueness constraints;
- foreign-key relationships;
- artifact path/hash indexing;
- evidence deduplication by stable ID;
- append-only planning progress records;
- separation of tracked planning tables from ignored research tables;
- query helpers used by Codex orchestration.

Phase 3 SQLite gates must cover:

- normalized instrument child tables for aliases, provider IDs, related instruments, data
  availability, and tradability evidence;
- idempotent instrument upsert behavior that replaces stale child rows;
- symbol/alias lookup and provider ID lookup;
- asset-class filtering;
- latest tradability evidence by provider;
- watchlist, watchlist item, and watchlist instrument round trips.

### Provider And Tool Contracts

Provider/tool tests should use fixtures and mocks by default. They should verify:

- successful artifact writes;
- partial failures;
- rate limits;
- stale data;
- malformed responses;
- missing provider fields;
- duplicated evidence;
- source query logging;
- retryable transport failures;
- semantic provider error payloads that must not be cached;
- source match spans that must align with stored evidence text.

Universe-discovery fixtures should be small and contract-shaped. Current Phase 3 fixtures live under
`tests/fixtures/tools/universe_discovery/` and should cover mixed asset classes, explicit ambiguity,
and watchlist-driven requests without relying on live provider access.

### Integration

Integration tests should cover:

- cheap broad screen into SQLite;
- deep dive on selected candidates;
- evidence ledger to PredictionCandidate synthesis;
- technical package attachment;
- report generation from stored artifacts;
- audit manifests that include rendered report artifacts;
- deterministic run IDs, duplicate-run rejection, and cleanup after failed transactional tool steps;
- neutral-only evidence staying insufficient instead of becoming supporting evidence.
- report failure products for stale, missing, malformed, unsupported, ambiguous, contradictory, and
  failed-provider inputs.

### Live API And Live Scraping

Live tests remain opt-in and explicitly marked:

```sh
python -m pytest -m live_api
python -m pytest -m live_scraping
```

They must skip unless the required credentials, user agent, network access, and explicit opt-in
environment variables are present.

### End To End

End-to-end tests should run fixture-backed research objectives:

- daily prediction report;
- user-prompted research report;
- no-evidence outcome;
- conflicting-evidence outcome;
- stale-provider outcome;
- broad-universe cheap screen;
- single-instrument deep dive.

### Phase 3 Instrument Universe Gates

Focused Phase 3 checks should run when instrument contracts, report contracts, storage schema, or
fixture shapes change:

```sh
python -m pytest tests/test_phase3_instrument_contracts.py
python -m pytest tests/test_storage_sqlite.py -k "instrument or watchlist or tradability"
python -m pytest tests/test_phase1_schema_contracts.py -k "resolution or report"
```

These gates are offline. They should not require live market-data providers, live scraping, OpenAI
credentials, optional GPU packages, or a new CLI command.

### Phase 4 QA Gates

The Phase 4 gate is deterministic and fixture-backed, and it exercises the real public tool
modules and Phase 4 service path. The production-gate E2E
starts a Phase 4 run, executes universe discovery, market data, technical package, social evidence,
news/catalysts, fundamentals, sector/macro context, candidate synthesis, prediction evaluation, and
final report rendering with fixture providers, then asserts SQLite run-graph rows, typed artifacts,
report output, and no trading-instruction language.

```sh
python -m pytest tests/test_phase4_tool_suite_e2e.py
python -m pytest tests/test_phase4_tool_suite_e2e.py tests/test_phase4_public_wiring.py tests/test_phase4_service.py
```

The gate must stay offline and should verify mixed-asset universe discovery, typed artifacts,
SQLite tool-run/evidence/candidate indexing, rendered Markdown/JSON reports, audit manifest
coverage, prediction-quality evaluation artifacts, Phase 4 artifact type alignment, and visible
recovery from partial tool failure.

### Cross-Cutting Integrity Gates

Run these when contracts, providers, orchestration, reporting, storage, or local ML behavior changes:

```sh
python -m pytest tests/test_phase0_contracts.py tests/test_phase1_schema_contracts.py
python -m pytest tests/test_research_providers.py tests/test_apnews_provider.py tests/test_candlecharts_provider.py
python -m pytest tests/test_phase2_mcp_service.py tests/test_orchestration_runtime.py
python -m pytest tests/test_ml_dataset.py tests/test_timesfm_dataset.py tests/test_ml_training_smoke.py
```

These tests pin traceable provenance, evidence-reference integrity, provider degradation and cache
semantics, report artifact manifests, neutral/contradictory synthesis, and ML leakage controls.

### Codex Smoke

The Phase 4 Codex smoke is opt-in because it launches the real Codex CLI and may use live web
search. The script is still named `scripts/run_phase2_codex_smoke.py` for compatibility, but it
drives the Phase 4 MCP tool suite. Mark tests with `codex_smoke` and skip unless
`NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE=1` and the `codex-smoke` extra is installed. The smoke path
must write only ignored artifacts and must not modify tracked source files. The runner also
fingerprints restricted ignored paths before and after the run, and launches the MCP Python process
with bytecode writes disabled.

## Negative Cases

Always include negative coverage for:

- unsupported or ambiguous instruments;
- stale market data;
- stale or unavailable source evidence;
- malformed HTML/JSON;
- social pump or joke/sarcasm risk;
- contradictory evidence;
- missing baseline comparison;
- model/tool output that is the only support for a candidate;
- report language that sounds like trading instruction.

## Acceptance

A behavior change is done when:

- focused tests cover the new behavior;
- relevant existing tests pass;
- live dependencies are either mocked or explicitly marked;
- docs are updated when behavior, configuration, contracts, or commands change;
- reports preserve provenance, uncertainty, and dissenting evidence.
