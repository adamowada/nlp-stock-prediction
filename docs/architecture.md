# Architecture

This document describes the target architecture for the Codex-led prediction research assistant.

## Product Boundary

The app is an agentic market and instrument prediction research assistant.

It does:

- discover instruments and prediction candidates;
- gather evidence;
- run independent tools;
- compare signals against baselines;
- preserve provenance;
- write Markdown/JSON prediction reports.

## Agent-Centered Design

Codex is the central research agent. It can:

- call local tools;
- read and write tool artifacts;
- query SQLite;
- search the internet when local tools are insufficient;
- reconcile contradictory evidence;
- choose which candidates deserve deeper investigation;
- write the final report.

Tools should be deterministic where practical. Codex synthesis is allowed to be language-heavy, but
tool outputs must be structured and auditable.

## Tool Suite

First-class tools should be independently callable and should emit typed artifacts:

- universe discovery;
- instrument registry;
- market data;
- technical package;
- social evidence;
- news and catalyst evidence;
- fundamentals;
- sector and macro context;
- prediction candidate synthesis;
- prediction evaluation and calibration;
- report generation.

Raw TimesFM belongs only in the technical package as a baseline-aware signal. TimesFM tuning is not
part of the product workflow.

## SQLite Role

SQLite is split into two local databases:

- `plans/planning.sqlite3`: tracked planning state for plans, decisions, progress, and links.
- `data/prediction-research.sqlite3`: ignored runtime research state for runs, artifacts, evidence,
  and prediction candidates.

Together they provide local operational memory:

- instrument index;
- evidence ledger;
- source query log;
- tool run registry;
- artifact index;
- prediction candidates;
- prediction scores;
- report index;
- prediction outcomes;
- active planning state.

Large raw files, model artifacts, reports, and provider payloads remain on disk. SQLite stores paths,
hashes, summaries, relationships, timestamps, and status.

## Prediction Flow

```text
user prompt or scheduled objective
-> Codex selects universe
-> cheap broad tools run first
-> SQLite indexes artifacts and evidence
-> Codex ranks candidate depth
-> deeper tools run on survivors
-> Codex builds PredictionCandidate records
-> report tool writes Markdown/JSON
-> outcomes can be evaluated later
```

The app should prefer broad cheap screens before expensive deep dives.

## Evidence Discipline

Every material report claim should trace to one of:

- a source evidence item;
- a tool artifact;
- a baseline or historical evaluation artifact;
- a clearly labeled Codex inference from those materials.

Evidence for and evidence against must both be preserved. Contradictions are useful signal, not
reporting noise.

Implemented hardening currently enforces that normalized source match spans and report evidence
reference quotes point back into the stored evidence text. External evidence provenance must remain
traceable even when a record is derived from another source, and neutral context is preserved as
context rather than counted as positive support for a prediction.

## Failure And Artifact Discipline

Provider and tool failures are represented as contract-shaped warning results where practical. Shared
HTTP/HTML fetch helpers retry retryable transport failures deterministically, avoid caching known
provider error payloads, and surface malformed provider payloads without dropping warning context.

Report bundles write Markdown, JSON, and audit-manifest artifacts. The final audit manifest includes
the rendered report artifacts and their hashes, while SQLite run-graph queries include artifacts and
source queries reachable through evidence and candidate links. Phase 2 smoke and Phase 4 tool-suite
runs use deterministic run IDs, reject duplicate starts where applicable, and preserve prior
successful outputs if a later transactional tool retry fails.

Phase 6 evaluation artifacts remain independent audit files, not inline report calculations. When a
report is rendered for a run with persisted outcome evaluations or calibration runs, the renderer
adds those artifacts to the audit manifest, surfaces outcome evaluations as prior-outcome reviews,
and references calibration artifacts as tool artifacts so downstream readers can audit prediction
quality without treating it as trading performance.

## ML Signal Discipline

Technical ML sidecars are conservative audit inputs. OHLCV timestamps are normalized across date and
timezone-aware datetime values, impossible OHLC relationships are rejected at the provider contract,
TimesFM train/validation/test splits include a purge that separates labels from later features, and
model/evaluation hash mismatches degrade to an unavailable ML signal instead of producing support.
