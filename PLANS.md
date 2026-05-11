# PLANS.md

This file defines how to write execution plans for complex Codex tasks.

Use an execution plan when the task involves:
- Multiple files or modules
- Significant refactoring
- Database or API changes
- Ambiguous requirements
- Risky behavior changes
- Work that may span multiple Codex sessions

Execution plans should be stored in `plans/`.

## Plan template

# [Plan title]

## Goal

Describe the intended end state in plain language.

## Non-goals

List anything that should explicitly remain out of scope.

## Context

Include relevant files, modules, APIs, constraints, and existing behavior.

## Milestones

### Milestone 1: [name]

- Changes:
- Files likely affected:
- Verification:

### Milestone 2: [name]

- Changes:
- Files likely affected:
- Verification:

## Acceptance criteria

- [ ] Behavior change is implemented.
- [ ] Relevant tests are added or updated.
- [ ] Existing tests continue to pass.
- [ ] Lint/typecheck/build pass where applicable.
- [ ] Documentation is updated if behavior or usage changes.

## Verification commands

```sh
pnpm test
pnpm lint
pnpm typecheck
pnpm build
```

## Decision log

- YYYY-MM-DD-HH-MM: Decision and rationale.

## Progress log

- YYYY-MM-DD-HH-MM: Completed work, findings, blockers, and next step.
