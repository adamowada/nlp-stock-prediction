# Lane C Strategy Extraction And Clustering

Status: complete. This is a historical Phase 2 lane plan.

## Goal

Implement deterministic, evidence-grounded strategy extraction helpers for Phase 2 Lane C. The lane should validate schema-shaped LLM output, reject unsupported or uncited claims, preserve quote provenance, cluster near-duplicate discussed strategies, and add fixture/live-smoke test scaffolding without changing frozen contracts.

## Non-goals

- Do not modify `src/nlp_stock_prediction/contracts/`.
- Do not implement recommendation scoring, risk gates, report rendering, provider fetching, or real brokerage behavior.
- Do not require network access or LLM credentials for the default test suite.
- Do not add third-party dependencies.

## Context

Lane C starts from tag `phase-1-contract-gate` on branch `codex/lane-c-extraction`. Public models are frozen in `contracts.extraction`, `contracts.evidence`, `contracts.providers`, and `contracts.provenance`. Strategy extraction must describe observed discussion only, not app recommendations.

## Milestones

### Milestone 1: Fixture validation tests

- Changes: Add Lane C tests for accepted schema-shaped LLM payloads, missing evidence rejection, quote mismatch rejection, unsupported claim rejection, and sarcasm/joke risk preservation.
- Files likely affected: `tests/test_lane_c_extraction.py`.
- Verification: `python -m pytest tests/test_lane_c_extraction.py`.

### Milestone 2: Extraction validation implementation

- Changes: Add extraction package helpers to validate payloads into `StrategyExtraction` and produce provider warnings for rejected records.
- Files likely affected: `src/nlp_stock_prediction/extraction/`.
- Verification: `python -m pytest tests/test_lane_c_extraction.py`.

### Milestone 3: Clustering and LLM smoke scaffolding

- Changes: Cluster near duplicates by ticker, direction, instrument, horizon, and catalyst normalization; add live LLM smoke placeholder that is skipped unless explicitly configured.
- Files likely affected: `src/nlp_stock_prediction/extraction/`, `tests/test_lane_c_extraction.py`, `tests/test_lane_c_live_llm_smoke.py`.
- Verification: lane tests plus default suite/lint/typecheck where practical.

## Acceptance criteria

- [x] Behavior change is implemented.
- [x] Relevant tests are added or updated.
- [x] Existing tests continue to pass.
- [x] Lint/typecheck/build pass where applicable.
- [x] Documentation is updated if behavior or usage changes.

## Verification commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
```

## Decision log

- 2026-05-11-00-00: Keep Lane C helpers private to the new `extraction` package and reuse frozen contract models at the boundary, avoiding contract changes while still giving later orchestration lanes a deterministic API.
- 2026-05-11-00-00: Treat direct recommendation-shaped LLM output as an unsupported claim because Lane C extracts observed discussion only; recommendation scoring belongs to Lane D.
- 2026-05-11-00-00: Quote validation requires an exact or case-insensitive contiguous match in the normalized evidence body and fills offsets when the LLM omits them.

## Progress log

- 2026-05-11-00-00: Created plan after reading AGENTS, roadmap, testing, contracts, and worktree runbook.
- 2026-05-11-00-00: Added Lane C tests for schema constants, payload validation, evidence quote validation, unsupported claim rejection, sarcasm/joke risk preservation, clustering, JSON parsing, fixture extractor behavior, and live LLM smoke scaffolding.
- 2026-05-11-00-00: Implemented the `nlp_stock_prediction.extraction` package with validation, clustering, fixture extractor, and schema constants without changing frozen contracts.
- 2026-05-11-00-00: Verified focused Lane C tests, full default pytest suite, non-live pytest selection, ruff, format check, and mypy using the shared Phase 2 venv.
