# Roadmap

This roadmap describes the Codex-led prediction research assistant rebuild. The implementation
sequence is agentic-first: Codex orchestration comes before deeper universe and tool-suite expansion
so the new direction develops around the assistant workflow from the start.

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

## Phase 2: Codex Orchestrator

Goal: make Codex the disciplined prediction research assistant.

Status: initial orchestrator implemented. The deterministic offline `research` command runs through
dummy tools, and an opt-in real Codex smoke path exposes Phase 2 tools through local MCP, records
live-search evidence, writes ignored artifacts, and renders Markdown/JSON reports. The research tools
themselves remain dummy/fixture-backed until Phase 4.

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

## Phase 3: Instrument Universe

Goal: support a broad retail-accessible universe.

Status: contract and storage layer implemented. The app can represent broad instrument identities,
explicit query resolutions, provider IDs, watchlists, tradability/access evidence, related
instruments, and fixture-backed universe artifacts. The default CLI has not added a separate
universe command, and live universe-discovery providers are not yet implemented.

Built:

- instrument registry contracts and SQLite tables;
- aliases and ambiguity handling;
- asset classes: stocks, ETFs, crypto, currencies, commodities, futures context, funds, indexes,
  proxies, and unknowns;
- related instruments;
- watchlists;
- provider IDs;
- tradability/access evidence;
- instrument universe requests/results;
- report-level instrument resolution references;
- small fixture-backed universe discovery scenarios.

Acceptance status:

- stocks, ETFs, crypto, currency/commodity exposure, and futures context can be represented in
  contracts and registry storage;
- ambiguous symbols require explicit resolution and cannot select an instrument silently;
- fixture/dummy universe paths can write instrument records and artifacts;
- first-class live universe discovery remains Phase 4 work.

## Phase 4: Tool Suite

Goal: convert research capabilities into first-class independent tools.

Build:

- universe discovery tool over the Phase 3 contracts and registry;
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
