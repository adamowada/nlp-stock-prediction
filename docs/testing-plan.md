# Testing Plan

The test strategy should protect the prediction research assistant's core promises: provenance,
repeatability, conservative claims, and graceful degradation.

## Default Suite

The default suite must be deterministic and offline.

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
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

- idempotent initialization;
- migration ordering;
- uniqueness constraints;
- foreign-key relationships;
- artifact path/hash indexing;
- evidence deduplication by stable ID;
- append-only planning progress records;
- query helpers used by Codex orchestration.

### Provider And Tool Contracts

Provider/tool tests should use fixtures and mocks by default. They should verify:

- successful artifact writes;
- partial failures;
- rate limits;
- stale data;
- malformed responses;
- missing provider fields;
- duplicated evidence;
- source query logging.

### Integration

Integration tests should cover:

- cheap broad screen into SQLite;
- deep dive on selected candidates;
- evidence ledger to PredictionCandidate synthesis;
- technical package attachment;
- report generation from stored artifacts.

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
