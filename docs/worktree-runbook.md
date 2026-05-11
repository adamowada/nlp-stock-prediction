# Git Worktree Runbook

Use this runbook after Phase 1 contract tests are committed and the full contract gate is complete.
Phase 1 is now complete on this branch; Phase 2 implementation lanes should start from the
contract-gate commit.

## Frozen Contract Ref

1. Confirm the base contract commit is clean and includes Phase 1 contract tests.
2. Tag it:

```sh
git tag phase-1-contract-gate <contract-commit-sha>
git push origin phase-1-contract-gate
```

If contracts change later, do not move the tag silently. Create a new contract revision commit,
record the decision, and rebase affected worktrees.

## Branch And Directory Layout

Use one integration branch and one worktree per lane:

```sh
git switch -c codex/integration-v1 phase-1-contract-gate
git push -u origin codex/integration-v1

git worktree add ..\nlp-stock-prediction-lane-a -b codex/lane-a-reddit-evidence phase-1-contract-gate
git worktree add ..\nlp-stock-prediction-lane-b -b codex/lane-b-providers phase-1-contract-gate
git worktree add ..\nlp-stock-prediction-lane-c -b codex/lane-c-extraction phase-1-contract-gate
git worktree add ..\nlp-stock-prediction-lane-d -b codex/lane-d-analysis-scoring phase-1-contract-gate
git worktree add ..\nlp-stock-prediction-lane-e -b codex/lane-e-report-cli phase-1-contract-gate
git worktree add ..\nlp-stock-prediction-lane-f -b codex/lane-f-reliability-ci phase-1-contract-gate
```

Each Codex subagent owns exactly one lane branch and must treat `src/nlp_stock_prediction/contracts/`
as read-only unless a single-threaded contract revision is approved.

## Lane Handoff Template

Each lane handoff must include:

- Branch name and starting contract ref.
- Files changed.
- Tests added or updated.
- Verification commands run and results.
- Known gaps and risks.
- Any contract gaps discovered.

If a lane discovers a contract gap, pause that lane. Fix the contract on a dedicated contract branch,
merge it into `codex/integration-v1`, and rebase affected lane worktrees before continuing.

## Verification Before Handoff

Each lane should run the relevant subset plus the default fast suite:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
```

Live checks stay opt-in and must not be required for default lane handoff.

## Integration Flow

1. Merge exactly one lane into `codex/integration-v1` at a time.
2. After each merge, run the default fast suite.
3. Resolve conflicts in favor of the frozen contracts unless a contract revision has been approved.
4. Preferred merge order:
   - Lane A: Reddit discovery and evidence.
   - Lane B: social, news, market, fundamentals, macro providers.
   - Lane C: strategy extraction and clustering.
   - Lane D: analysis, risk, and scoring.
   - Lane E: report rendering, audit artifacts, and CLI orchestration.
   - Lane F: reliability, observability, CI, and live smoke.
5. Open one final PR from `codex/integration-v1` to `main` after fixture-backed e2e report
   generation is green.

## PR Strategy

Use one PR per lane into `codex/integration-v1` when review bandwidth allows. Use one final
integration PR into `main`. This keeps lane review focused while preserving a single final merge
gate for the full product behavior.
