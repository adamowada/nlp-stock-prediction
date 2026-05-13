# PLANS.md

This file defines how planning state works in this repo.

`PLANS.md` is immutable by default. Edit it only when the user specifically asks. Active execution
plans belong in `plans/planning.sqlite3`, not in Markdown files.

## Planning Model

The planning database is operational state for Codex and is tracked in git so planning records are
part of project history. It answers:

- what are we trying to do;
- why are we doing it this way;
- what has changed since the plan started;
- what remains blocked or unverified;
- which artifacts and commits belong to the work.

Markdown remains the source of truth for stable human-facing knowledge. Do not use the planning
tables as a private replacement for README, AGENTS, contracts, architecture, or user workflow docs.

## Planning Tables

The initial planning schema is created by
`nlp_stock_prediction.storage.initialize_planning_database()`.

### `plans`

One row per active, completed, abandoned, or superseded plan.

| Column | Required | Meaning |
| --- | --- | --- |
| `plan_id` | yes | Stable unique ID, for example `plan-sqlite-layer`. |
| `slug` | yes | Human-readable unique slug, for example `sqlite-layer`. |
| `title` | yes | Short display title. |
| `goal` | yes | Plain-language intended end state. |
| `non_goals_json` | yes | JSON object describing explicit non-goals. Use `{}` if empty. |
| `context_json` | yes | JSON object with relevant files, constraints, links, and assumptions. |
| `status` | yes | Lifecycle status: `active`, `blocked`, `completed`, `abandoned`, or `superseded`. |
| `priority` | yes | Integer sort key. Higher means more urgent. |
| `owner_agent` | no | Agent/person currently coordinating the plan. |
| `superseded_by_plan_id` | no | Replacement plan ID when status is `superseded`. |
| `created_at` | yes | UTC timestamp written by the storage layer. |
| `updated_at` | yes | UTC timestamp written by the storage layer. |

### `plan_milestones`

One row per major work slice inside a plan.

| Column | Required | Meaning |
| --- | --- | --- |
| `milestone_id` | yes | Stable unique ID. |
| `plan_id` | yes | Parent plan. |
| `title` | yes | Milestone name. |
| `status` | yes | `pending`, `in_progress`, `completed`, `blocked`, or `skipped`. |
| `sort_order` | yes | Integer ordering within the plan. |
| `details_json` | yes | JSON object for notes, expected files, and verification hints. |

### `plan_acceptance_criteria`

One row per explicit completion condition.

| Column | Required | Meaning |
| --- | --- | --- |
| `criterion_id` | yes | Stable unique ID. |
| `plan_id` | yes | Parent plan. |
| `description` | yes | Testable acceptance statement. |
| `status` | yes | `pending`, `satisfied`, `failed`, or `waived`. |
| `verification_command` | no | Command or check that proves the criterion. |

### `plan_decisions`

Append-only decision log.

| Column | Required | Meaning |
| --- | --- | --- |
| `decision_id` | yes | Stable unique ID. |
| `plan_id` | yes | Parent plan. |
| `decision` | yes | The decision that was made. |
| `rationale` | yes | Why this choice was made. |
| `alternatives_considered` | no | Other paths considered. |
| `consequences` | no | Expected tradeoffs or follow-up work. |
| `decided_at` | yes | UTC decision timestamp. |

### `plan_progress_events`

Append-only progress log.

| Column | Required | Meaning |
| --- | --- | --- |
| `progress_id` | yes | Stable unique ID. |
| `plan_id` | yes | Parent plan. |
| `event_type` | yes | `started`, `completed`, `blocked`, `unblocked`, `verified`, `note`, or similar. |
| `summary` | yes | Short human-readable event summary. |
| `details` | no | Extra context. |
| `linked_artifact_id` | no | Stable artifact ID, usually stored in the ignored research database. |
| `occurred_at` | yes | UTC event timestamp. |

### `plan_artifact_links`

Many-to-many link between plans and artifact IDs. The planning database does not foreign-key into
the ignored research database; it stores stable artifact IDs so the record remains meaningful in git
history.

| Column | Required | Meaning |
| --- | --- | --- |
| `plan_id` | yes | Parent plan. |
| `artifact_id` | yes | Stable artifact ID. |
| `relationship` | yes | Link type, for example `output`, `evidence`, `verification`, or `audit`. |

### `plan_commit_links`

Many-to-many link between plans and commits.

| Column | Required | Meaning |
| --- | --- | --- |
| `plan_id` | yes | Parent plan. |
| `commit_sha` | yes | Git commit SHA. |
| `relationship` | yes | Link type, for example `implements`, `fixes`, or `verifies`. |

## Planning Interface

Use the storage interface instead of writing ad hoc SQL.

```python
from pathlib import Path

from nlp_stock_prediction.storage import (
    PlanDecisionRecord,
    PlanProgressRecord,
    PlanRecord,
    initialize_planning_database,
)

store = initialize_planning_database(Path("plans/planning.sqlite3"))

store.upsert_plan(
    PlanRecord(
        plan_id="plan-example",
        slug="example",
        title="Example Plan",
        goal="Describe the end state.",
        status="active",
        priority=10,
        owner_agent="codex",
        non_goals={"visualization": "out of scope"},
        context={"docs": ["docs/architecture.md"]},
    )
)

store.add_plan_decision(
    PlanDecisionRecord(
        decision_id="decision-example-001",
        plan_id="plan-example",
        decision="Use SQLite for active planning state.",
        rationale="Structured rows are easier for Codex to query than Markdown logs.",
    )
)

store.add_plan_progress(
    PlanProgressRecord(
        progress_id="progress-example-001",
        plan_id="plan-example",
        event_type="started",
        summary="Plan created.",
    )
)
```

Current public helpers cover plans, decisions, and progress events. If a task needs milestones,
acceptance criteria, artifact links, or commit links, add typed helper methods to the storage layer
before using those tables. Do not scatter direct SQL across feature code.

## Correct Usage

Create or update a plan when work is complex enough to span modules, change contracts, touch the
database, alter provider behavior, or create user-visible workflow changes.

For each active plan:

- create a `plans` row before implementation;
- write `non_goals_json` and `context_json` clearly enough for a future Codex session;
- append `plan_decisions` before or during meaningful architecture choices;
- append `plan_progress_events` as work starts, completes, blocks, unblocks, or verifies;
- update `plans.status` instead of deleting completed or abandoned plans;
- link artifacts and commits once the relevant helper methods exist;
- update Markdown source-of-truth docs when durable behavior changes.

Do not:

- create new Markdown files under `plans/` unless the user specifically asks;
- hide source-of-truth decisions only in SQLite;
- edit `AGENTS.md` or `PLANS.md` unless the user specifically asks;
- record vague progress like "worked on stuff";
- mark a plan completed until acceptance criteria are verified or explicitly waived.

Historical Markdown plans have been removed from the working tree. Git history remains the archive.
