# Contracts

This document defines target contracts and invariants for the prediction research rebuild.

## Core Invariants

- The product is a prediction research assistant, not a trading app.
- Reports may make evidence-backed predictions, not buy/sell instructions.
- Every material claim must trace to evidence, a tool artifact, a baseline, or a labeled Codex
  inference.
- Evidence for and evidence against must both be preserved.
- Social content is observed discussion, not fact.
- News and filings are source evidence, but extracted claims still require provenance.
- Raw TimesFM is a technical signal only; TimesFM tuning is not part of the product workflow.
- SQLite is split into a tracked planning database and an ignored research database; large payloads
  remain on disk.

## Instrument

An instrument record should identify what the app is researching.

Required concepts:

- canonical symbol or identifier;
- display name;
- asset class;
- venue or provider namespace;
- aliases;
- related instruments;
- tradability evidence;
- data availability;
- sector, category, or theme when applicable.

Ambiguous symbols must not resolve silently.

## Tool Run

Every tool invocation should produce a run record:

- tool name and version;
- input parameters;
- started and completed timestamps;
- status;
- warnings and errors;
- artifact paths;
- artifact hashes;
- provider metadata where applicable.

Tool failures should be reportable and should not corrupt prior artifacts.

## Evidence Item

Evidence items capture source claims or observations.

Required concepts:

- source type;
- provider;
- URL or permalink when available;
- query or request used;
- retrieved timestamp;
- published timestamp when available;
- referenced instruments;
- extracted claim;
- confidence in extraction;
- source reliability notes;
- freshness status;
- artifact link.

Evidence must support deduplication across repeated searches and providers.

## Prediction Candidate

`PredictionCandidate` is the central product object.

Required concepts:

- candidate ID;
- instrument;
- prediction horizon;
- prediction type;
- scenario or outcome;
- direction or state, if applicable;
- confidence;
- evidence for;
- evidence against;
- signal artifacts;
- baseline comparison;
- uncertainty;
- freshness;
- contradictions;
- report status.

Report status should distinguish high-confidence, moderate-confidence, watchlist, insufficient
evidence, and rejected candidates without implying trade execution.

## Technical Package

The technical package should combine cheap deterministic indicators with raw model context.

Required concepts:

- OHLCV data provenance;
- latest usable bar;
- deterministic indicators;
- baselines;
- raw TimesFM metrics when available;
- uncertainty and interval width when available;
- agreement or disagreement with the candidate scenario;
- warnings;
- artifact hash.

Technical signals can support or weaken a prediction. They must not create reportable predictions by
themselves.

## Report

Reports should include:

- objective or prompt;
- universe;
- run metadata;
- prediction candidates;
- evidence for and against;
- confidence and uncertainty;
- source links;
- artifact references;
- unavailable or stale data warnings;
- what would change the prediction.

Markdown and JSON reports should carry the same substantive information.

## Planning State

Active plans belong in the tracked planning SQLite database. Planning contracts include:

- plan;
- milestone;
- acceptance criterion;
- decision;
- progress event;
- blocker;
- open question;
- linked artifact;
- linked commit.

`AGENTS.md` and `PLANS.md` are immutable by default and should be changed only on explicit user
request.
