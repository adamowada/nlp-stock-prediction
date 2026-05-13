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
