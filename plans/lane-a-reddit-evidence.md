# Lane A Reddit Discovery And Evidence

## Goal

Implement deterministic Reddit ticker-card discovery, Reddit post/comment evidence normalization,
and high-precision ticker matching for Phase 2 Lane A while preserving the frozen public contracts.

## Non-goals

- Do not modify `src/nlp_stock_prediction/contracts/`.
- Do not add live Reddit API or scraping calls to the default test suite.
- Do not implement report rendering, strategy extraction, scoring, or provider caching outside the
  minimum metadata needed by Lane A.
- Do not add third-party dependencies unless the standard library cannot satisfy a requirement.

## Context

- Worktree: `C:\Users\adams\projects\nlp-stock-prediction-lane-a`.
- Branch: `codex/lane-a-reddit-evidence`.
- Starting contract ref: `phase-1-contract-gate` (`5b3d6b5`).
- Frozen contracts used by this lane:
  - `TickerCandidate` and `TickerDiscoveryResult` in `contracts.discovery`.
  - `SourceEvidence` and `TextSpan` in `contracts.evidence`.
  - `SourceProvenance`, `ProviderWarning`, `ProviderHealth`, and `ProviderResult`.
  - `RedditProvider`, `TickerDiscoveryRequest`, and `EvidenceRequest`.
- Default tests must remain deterministic and network-free.

## Milestones

### Milestone 1: Devvit ticker-card discovery

- Changes:
  - Parse `ticker-container-*` identifiers from fixture/raw HTML.
  - Preserve raw candidate order, raw identifiers, raw text when available, and provenance.
  - Normalize exactly six unique tickers by first-seen order.
  - Emit validation warnings/errors for malformed source, duplicates, too few unique tickers, and
    too many unique tickers.
- Files likely affected:
  - `src/nlp_stock_prediction/reddit/discovery.py`
  - `tests/test_lane_a_reddit_discovery.py`
  - `tests/fixtures/reddit/`
- Verification:
  - `python -m pytest tests/test_lane_a_reddit_discovery.py`

### Milestone 2: Ticker matching

- Changes:
  - Add high-precision symbol matching with cashtag and uppercase bare-symbol support.
  - Avoid false positives for short symbols such as `MU`, `AI`, `ON`, and `IT` inside ordinary
    words or lower-case prose.
- Files likely affected:
  - `src/nlp_stock_prediction/reddit/matching.py`
  - `tests/test_lane_a_reddit_matching.py`
- Verification:
  - `python -m pytest tests/test_lane_a_reddit_matching.py`

### Milestone 3: Reddit evidence normalization

- Changes:
  - Normalize fixture-shaped Reddit posts and comments into `SourceEvidence`.
  - Preserve source kind, Reddit ID, author hash, created timestamp, permalink, score, body/title
    text, matched ticker, provider metadata, raw snapshot ID, and freshness.
- Files likely affected:
  - `src/nlp_stock_prediction/reddit/evidence.py`
  - `tests/test_lane_a_reddit_evidence.py`
- Verification:
  - `python -m pytest tests/test_lane_a_reddit_evidence.py`

### Milestone 4: Provider adapter and package exports

- Changes:
  - Provide a fixture-backed Reddit adapter that satisfies the frozen `RedditProvider` protocol.
  - Keep external calls out of the default suite.
- Files likely affected:
  - `src/nlp_stock_prediction/reddit/provider.py`
  - `src/nlp_stock_prediction/reddit/__init__.py`
  - focused Lane A tests
- Verification:
  - `python -m pytest tests/test_lane_a_reddit_provider.py`

## Acceptance criteria

- [x] Devvit card parsing preserves candidate order and identifiers.
- [x] Valid discovery returns exactly six first-seen unique tickers and candidate records.
- [x] Invalid discovery returns warnings without changing frozen contracts.
- [x] Reddit posts/comments normalize into `SourceEvidence` with full provenance and metadata.
- [x] Short tickers match only in high-precision contexts.
- [x] Relevant tests are added and pass.
- [x] Default deterministic tests, lint, format check, and typecheck pass or scoped failures are
  documented.
- [x] Lane changes are committed locally.

## Verification commands

```sh
python -m pytest tests/test_lane_a_reddit_discovery.py tests/test_lane_a_reddit_matching.py tests/test_lane_a_reddit_evidence.py tests/test_lane_a_reddit_provider.py
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
```

## Decision log

- 2026-05-11-00-00: Keep Lane A modules under `src/nlp_stock_prediction/reddit/` and rely only on
  the Python standard library plus existing contracts to avoid dependency churn across parallel
  worktrees.

## Progress log

- 2026-05-11-00-00: Read `AGENTS.md`, roadmap, testing plan, frozen contracts, worktree runbook,
  and relevant contract/test files. Confirmed branch `codex/lane-a-reddit-evidence` in the lane A
  worktree and no need for contract edits so far.
- 2026-05-11-14-27: Added focused Lane A fixtures and tests, implemented Reddit discovery,
  ticker matching, evidence normalization, and a fixture-backed provider. Verified focused tests,
  full default tests, non-live marker suite, ruff lint, ruff format check, and mypy with the shared
  Phase 2 venv.
