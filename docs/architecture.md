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
- `data/prediction-research.sqlite3`: ignored default research state for service/tool runs that do
  not choose a per-run database.
- `data/phase4-{offline|live}-runtime-{date}-{symbol_hash}-{output_hash}.sqlite3`: ignored CLI
  research state for isolated Phase 4/5 report invocations.

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
Freshness is explicit provenance: fresh, stale, missing, and unknown states remain visible in
provider results and report inputs instead of being coerced into support.

## Failure And Artifact Discipline

Provider and tool failures are represented as contract-shaped warning results where practical. Shared
HTTP/HTML fetch helpers retry retryable transport failures deterministically, avoid caching known
provider error payloads, and surface malformed provider payloads without dropping warning context.
Provider adapters build degradation results through a shared provider execution module so raw
snapshot IDs, cache keys, rate-limit status, malformed payloads, and partial item warnings preserve
the same point-in-time identity across live and fixture-backed tests.

Report bundles write Markdown, JSON, and audit-manifest artifacts. The final audit manifest includes
the rendered report artifacts and their hashes, while SQLite run-graph queries include artifacts and
source queries reachable through evidence and candidate links. Final report files are additionally
indexed in `report_artifact_index` with path, hash, schema version, report date, instrument, data
mode, tool run, and source run timestamps; report bodies and raw provider payloads remain on disk.
Report bundle construction uses a phase-neutral builder request, while the older Phase 2 function
name remains a compatibility wrapper. Artifact type, JSON-artifact, final-report-artifact, and
source-reference policies live in shared artifact policy modules rather than in renderer-local
literal sets.
Phase 2 smoke and Phase 4 tool-suite runs use deterministic run IDs, reject duplicate starts where
applicable, and preserve prior successful outputs if a later transactional tool retry fails.

Phase 4 live/offline behavior is selected through a run-mode adapter. The service asks the adapter
for provider choices, instrument identity, data-mode metadata, macro availability, and live
no-evidence policy instead of branching separately inside each tool runner.

Report failure modes are report products rather than exceptions when the run graph is otherwise
valid. Missing evidence, malformed artifacts or pages, failed providers, stale evidence,
unsupported or ambiguous instrument resolutions, and contradictory source evidence are rendered as
structured insufficient-evidence, provider-health, audit-manifest, or contradicted-candidate context
instead of being hidden or converted into unsupported conclusions.

Phase 6 evaluation artifacts remain independent audit files, not inline report calculations. When a
report is rendered for a run with persisted outcome evaluations or calibration runs, the renderer
adds those artifacts to the audit manifest, surfaces outcome evaluations as prior-outcome reviews,
and references calibration artifacts as tool artifacts so downstream readers can audit prediction
quality without treating it as trading performance.

The public evaluation surface is a thin phase-neutral layer over the same run graph. The CLI group
`python -m nlp_stock_prediction evaluation` and the local MCP registration both expose registry
derived evaluation tools for inspection, live outcome materialization, outcome loading and summary,
artifact freshness, evidence aging, source reliability, provider playbooks, calibration,
walk-forward evaluation, ablation, and calibration drift. The CLI and MCP server require an existing
explicit database, and writer tools require a concrete artifact root; service resolution then applies
the repository write policy before any artifact is indexed.

## ML Signal Discipline

Technical ML sidecars are conservative audit inputs. OHLCV timestamps are normalized across date and
timezone-aware datetime values, impossible OHLC relationships are rejected at the provider contract,
TimesFM train/validation/test splits include a purge that separates labels from later features, and
model/evaluation hash mismatches degrade to an unavailable ML signal instead of producing support.
