# Contracts

This document defines implemented and target contracts for the prediction research rebuild. When a
section describes behavior beyond the current command surface, it is labeled as target behavior.

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

An instrument record identifies what the app is researching. Phase 3 implements the public Pydantic
contracts for instruments, universe requests/results, watchlists, and resolution outcomes, plus
SQLite registry tables and query helpers for persisted instrument records.

Required concepts:

- canonical instrument ID;
- normalized symbol or identifier;
- display name;
- asset class;
- venue or provider namespace;
- aliases;
- provider IDs;
- related instruments;
- tradability evidence;
- data availability;
- sector, category, or theme when applicable.

Implemented asset classes are `stock`, `etf`, `crypto`, `currency`, `commodity`, `futures`, `fund`,
`index`, `proxy`, and `unknown`. Provider IDs are namespaced so one provider can contribute multiple
identifiers, such as ticker and CIK, without overwriting each other.

Tradability/access evidence is provenance, not a trading instruction. Each observation must include
provider, status, retrieval timestamp, and at least one traceable source field such as `source_url`,
`permalink`, or `raw_identifier`.

Resolution results use explicit statuses:

- `resolved`: exactly one selected instrument ID that appears in the matches.
- `ambiguous`: two or more matches and no silent selection.
- `unsupported`: the app cannot represent or process the requested instrument class yet.
- `unavailable`: the app can represent the class but available providers did not return access or
  data.

Ambiguous symbols must not resolve silently. `AI`, for example, can remain an ambiguous stock/token
resolution until the request supplies asset class, venue, provider namespace, or another disambiguator.

## Instrument Universe And Watchlists

Implemented Phase 3 contracts:

- `InstrumentQuery`: a direct user/provider query with optional asset class, venue, provider, and
  provider identifier hints.
- `Watchlist` and `WatchlistEntry`: named collections of instrument queries with local notes, tags,
  and optional requested instrument IDs.
- `InstrumentUniverseRequest`: a mixed input containing direct queries, watchlists, allowed asset
  classes, provider names, and a flag for related instruments.
- `InstrumentUniverse`: resolved instruments plus request-to-instrument traceability.

SQLite currently persists normalized instrument rows, aliases, provider IDs, related instruments,
data availability, tradability evidence, watchlists, and watchlist items. Helper queries cover symbol
or alias lookup, provider ID lookup, asset-class filtering, latest tradability evidence by provider,
and watchlist instrument listing.

Target behavior: first-class live universe discovery tools will turn provider/search/watchlist input
into these contracts. Today, Phase 4 fixture-backed universe discovery and smoke paths exercise the
contracts and storage; they do not claim live provider coverage for all asset classes.

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

Implemented validation requires source match spans to stay inside the stored evidence text and match
the exact referenced substring. Derived external records still need source URL/permalink, raw
identifier, raw snapshot ID, and freshness status; only internal analysis can omit external source
traceability.

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

Implemented ML dataset contracts reject invalid OHLC relationships, normalize date and aware-datetime
timestamps to one comparable key, preserve the one-bar lookback used by return features in metadata,
and enforce purged TimesFM split boundaries so labels from one split do not overlap features in the
next split.

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

Markdown and JSON reports should carry the same substantive information. The implemented report
contract includes `instruments` and `instrument_resolutions`; selected resolution IDs must reference
report instruments.

Implemented report validation also requires instrument-section and candidate symbols to match their
referenced instruments, and requires evidence-reference quotes/spans to match the cited source
evidence text. Markdown rendering includes published/created timestamps, data-quality metadata,
strategy cluster summaries when present, and audit manifest references or artifact entries.

Phase 5 report contracts add first-class structures for dissenting evidence, uncertainty drivers,
prediction change triggers, prior-outcome reviews, structured insufficient-evidence outcomes,
report-level source references, and material claim traces. Reports with candidates must include
material claim traces covering every candidate, and each cited source reference, prior review,
evidence ID, and audit artifact ID must resolve inside the report bundle. Report source references
must point to source evidence, tool artifacts, provider context, or prior outcomes rather than only
to a candidate. Candidate records must either include specific change triggers or an explicit
limitation explaining why the report cannot define them yet.

`json-report-contract.v1` validates the machine-readable report artifact without wrapping or
renaming the top-level `DailyReport` fields. It maps each material Markdown section to stable JSON
pointers and round-trips through the `DailyReport` contract. Final report artifacts are indexed in
runtime SQLite through `ReportArtifactRecord`, which stores metadata needed for point-in-time lookup
without storing large report bodies in SQLite.

Settled prediction/evaluation contracts now include an explicit `prediction_type`, typed
`SignalArtifactReference` records by signal family (`technicals`, `timesfm`, `social`, `news`,
`fundamentals`, and `sector_macro`), and per-family signal counts in prediction-quality evaluation
payloads. Flat `signal_artifact_ids` remain for compatibility with existing SQLite candidate rows,
but typed references are the stable report/export shape for new code.

Phase 6 outcome tracking begins with `PredictionOutcome` and `PredictionOutcomeEvaluation`.
`PredictionOutcome` records the evaluated candidate, instrument, prediction type, horizon,
evaluation window, observed/unavailable/stale/pending state, observed result when available,
outcome evidence, artifact IDs, and limitations. `PredictionOutcomeEvaluation` records the review
status, quality score when resolved, optional baseline comparison, evidence, artifacts, and
limitations. Resolved outcome evaluations require an observed outcome plus evidence or artifacts;
pending, stale, or not-evaluable evaluations must explain their limitations.

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
