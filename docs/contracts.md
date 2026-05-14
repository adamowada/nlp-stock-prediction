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
identifier, and raw snapshot ID; only internal analysis can omit external source traceability.
Freshness status is always carried, but `unknown` is a valid explicit state for malformed, future,
or otherwise not-point-in-time-checkable source timestamps.

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

Insufficient-evidence reports are first-class outputs. They carry blocking reasons, missing evidence
types, provider names, evidence references, artifact IDs, and metadata for excluded candidates,
missing evidence, missing artifacts, provider statuses, and instrument-resolution status counts.
Contradictory source evidence remains visible as `evidence_against` and dissenting evidence, and it
forces a contradicted report candidate rather than being converted into an evidence-supported
conclusion. Audit manifests now include provider-health snapshots so failed, stale, empty, partial,
or malformed provider/tool results remain visible in both the JSON report and the audit manifest.

`json-report-contract.v1` validates the machine-readable report artifact without wrapping or
renaming the top-level `DailyReport` fields. It maps each material Markdown section to stable JSON
pointers and round-trips through the `DailyReport` contract. Final report artifacts are indexed in
runtime SQLite through `ReportArtifactRecord`, which stores metadata needed for point-in-time lookup
without storing large report bodies in SQLite.

Settled prediction/evaluation contracts now include an explicit `prediction_type`, typed
`SignalArtifactReference` records by signal family (`technicals`, `timesfm`, `social`, `news`,
`fundamentals`, and `sector_macro`), and per-family signal counts in prediction-quality evaluation
payloads. Signal-family and artifact-type compatibility is centralized in the signal artifact
policy; report assembly, candidate conversion, and prediction evaluation must use that policy rather
than hardcoding family/type mappings. Flat `signal_artifact_ids` remain for compatibility with
existing SQLite candidate rows, but typed references are the stable report/export shape for new code.

Report inputs and outputs carry typed data-mode provenance. `report_data_mode`, `provider_mode`,
and `input_data_mode` identify whether stored records were produced by live providers, offline
fixtures, dummy smoke, or Codex smoke paths. Live report assembly treats fixture, dummy, and smoke
markers as boundary violations; string-marker scanning is a backstop for legacy or malformed
metadata, not the primary contract shape. Phase 4/5 report runs must stamp data-mode metadata on the
run and tool records; generic Phase 4 run names are not enough to infer offline fixture mode.

Phase 6 outcome tracking begins with `PredictionOutcome` and `PredictionOutcomeEvaluation`.
`PredictionOutcome` records the evaluated candidate, instrument, prediction type, horizon,
evaluation window, observed/unavailable/stale/pending state, observed result when available,
outcome evidence, artifact IDs, and limitations. `PredictionOutcomeEvaluation` records the review
status, quality score when resolved, optional baseline comparison, evidence, artifacts, and
limitations. Resolved outcome evaluations require an observed outcome plus evidence or artifacts;
pending, stale, or not-evaluable evaluations must explain their limitations.

Phase 6 calibration can now persist signal-family ablations. A `signal_family_ablation` audit
artifact records the point-in-time cohort, source target/outcome-evaluation IDs, source signal
artifacts, and one `SignalFamilyAblation` per requested family. Each ablation compares resolved
prediction quality for candidates with that signal family against the cohort without it, while
unresolved cohorts remain metric-free and carry explicit limitations. The same run is indexed in
SQLite through `calibration_runs` and `calibration_slices` so downstream calibration summaries can
reuse the persisted attribution data.

Walk-forward evaluation is persisted as a `walk_forward_evaluation` audit artifact. It sorts
eligible outcome evaluations chronologically, excludes rows after the point-in-time cutoff, and
emits train/test folds with held-out quality metrics, unresolved held-out status counts, provenance,
and limitations. Fold-level held-out metrics are also stored as `calibration_slices` under a
`calibration_runs` row so later calibration summaries can reuse the same chronological evaluation
without recomputing or using lookahead data.

Calibration summaries are persisted as `calibration_summary` audit artifacts. A summary records
scoreable point-in-time outcome evaluations, reliability bins, Brier score, log loss, accuracy,
expected calibration error, a no-skill baseline comparison, signal-family calibration summaries, and
source outcome/artifact provenance. Inputs after the `as_of` cutoff, outside requested
prediction/horizon filters, or missing prediction scores are excluded with explicit limitations.
Overall, bin-level, and signal-family slices are also stored in `calibration_slices`.

Rendered Markdown/JSON reports now integrate persisted Phase 6 outputs without recomputing them.
Stored `prediction_outcome_evaluations` for rendered candidates become `PriorOutcomeReview`
records, candidates reference those review IDs, and report source references include the prior
review trace. Phase 6 audit artifacts (`prediction_outcome`, `prediction_outcome_evaluation`,
`calibration_summary`, `signal_family_ablation`, and `walk_forward_evaluation`) remain separate
artifacts but are preserved in the final audit manifest and report source references where they
support calibration context.

Phase 5 report rendering now populates `PriorOutcomeReview` directly from stored prior JSON report
artifacts when available. The prior report artifact must resolve through the runtime report index,
match its stored hash, and load through the JSON report contract. First runs, missing or malformed
prior artifacts, stale report windows, and unlinked prior candidates are represented as explicit
review limitations. Current candidates link to the review ID and receive concrete
`PredictionChangeTrigger` entries for follow-up evidence, provider refreshes, baseline changes, or
outcome data. Each prior review also carries a `prediction_outcome` and
`prediction_outcome_evaluation` metadata projection using the settled Phase 6 outcome contracts so
future consumers do not need a second prior-review vocabulary.

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
