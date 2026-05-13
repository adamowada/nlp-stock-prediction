# Roadmap

This roadmap describes the Codex-led prediction research assistant rebuild.

## Phase 0: Documentation Reset

Goal: align source-of-truth docs with the Codex-led prediction research assistant direction.

Acceptance:

- Old Markdown plans are removed.
- README, AGENTS, PLANS, architecture, contracts, configuration, and testing docs describe the new
  product boundary.
- Fine-tuned TimesFM is no longer documented as a preferred workflow.
- Visualization is explicitly out of scope.

## Phase 1: SQLite Foundation

Goal: add local SQLite as the operational memory.

Status: initial foundation implemented. The schema, repository wrappers, tracked planning database,
and ignored research database exist; future phases will connect orchestration and tools to them.

Build:

- schema management;
- migrations;
- run registry;
- artifact index;
- evidence ledger;
- source query log;
- prediction candidate tables;
- structured planning tables.

Acceptance:

- database initialization is idempotent for both databases;
- schema is tested;
- artifacts remain file-backed with hashes and paths in SQLite;
- planning state can be created, updated, queried, and closed.

## Phase 2: Instrument Universe

Goal: support a broad retail-accessible universe.

Build:

- instrument registry;
- aliases and ambiguity handling;
- asset classes;
- related instruments;
- watchlists;
- provider IDs;
- tradability evidence.

Acceptance:

- stocks, ETFs, crypto, currency/commodity proxies, and futures context can be represented;
- ambiguous symbols require explicit resolution;
- universe discovery can write instrument records.

## Phase 3: Tool Suite

Goal: convert research capabilities into first-class independent tools.

Build:

- universe discovery tool;
- market data tool;
- technical package tool;
- social evidence tool;
- news/catalyst tool;
- fundamentals tool;
- sector/macro tool;
- prediction evaluation tool;
- report tool.

Acceptance:

- every tool writes typed artifacts;
- every tool run is indexed in SQLite;
- tool failures are recoverable and visible.

## Phase 4: Codex Orchestrator

Goal: make Codex the disciplined prediction research assistant.

Build:

- objective parsing;
- tool selection;
- cheap-screen-before-deep-dive workflow;
- internet search evidence capture;
- candidate synthesis;
- contradiction handling;
- report assembly.

Acceptance:

- Codex can run a daily report workflow;
- Codex can run an on-demand prompt workflow;
- every report candidate traces to evidence and artifacts.

## Phase 5: Prediction Reports

Goal: make reports the primary product.

Build:

- Markdown report rendering;
- JSON report rendering;
- evidence ledger appendix;
- uncertainty and dissenting evidence sections;
- "what would change this prediction" sections;
- prior-outcome review.

Acceptance:

- reports never use buy/sell instruction language;
- reports include evidence for and against;
- reports preserve artifact references and source links.

## Phase 6: Evaluation And Calibration

Goal: learn which signals deserve trust.

Build:

- baseline comparisons;
- walk-forward evaluation;
- prediction outcome tracking;
- signal-family ablations;
- calibration summaries.

Acceptance:

- deterministic technicals, raw TimesFM, social evidence, news, fundamentals, and sector/macro signals
  can be evaluated separately and together;
- evaluation uses only point-in-time available evidence;
- results are reported as prediction quality, not trading performance claims.

## Phase 7: Evaluation Hardening

Goal: make the assistant easier to audit after repeated report runs.

Build:

- outcome review summaries;
- stale artifact detection;
- source reliability notes;
- provider replacement playbooks;
- calibration drift checks.

Acceptance:

- repeated reports can explain which prior evidence aged out;
- calibration artifacts remain separate from prediction reports;
- provider swaps preserve provenance and contract compatibility.
