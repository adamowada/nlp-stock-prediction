# nlp-stock-prediction

This repo is being rebuilt into a local, Codex-led prediction research assistant.

The important boundary: this is not a trading app. It should not place trades, size positions, or
tell anyone what to buy or sell. The output is a prediction report: what the evidence seems to imply,
how strong that evidence is, what conflicts with it, and what would change the view.

The old version was centered on a fixed stock-report pipeline and later spent too much energy on
TimesFM fine-tuning. That direction is being retired. The new direction is a set of small research
tools, a local SQLite memory layer, and a Codex agent that coordinates the work and writes the final
Markdown/JSON report.

## Current State

This branch is in the middle of the rebuild.

What exists now:

- legacy offline report CLI and tests;
- provider, evidence, scoring, reporting, and TimesFM-era modules from the previous design;
- a new SQLite storage layer under `nlp_stock_prediction.storage`;
- refreshed docs that describe the new architecture.

What is intentionally not the focus anymore:

- TimesFM fine-tuning;
- one-off HPO workflows;
- visual dashboards;
- trading-language output.

Raw TimesFM may still be useful later as a cheap technical signal, but only as one input inside a
broader technical package.

## Product Shape

The target app has three moving parts:

1. Independent tools gather evidence or produce artifacts.
2. SQLite stores planning state in git and runtime research state locally.
3. Codex decides what to investigate next and writes the prediction report.

The core object is `PredictionCandidate`:

```text
PredictionCandidate
- instrument and asset class
- prediction horizon
- scenario or outcome being predicted
- evidence for
- evidence against
- source and tool artifacts
- freshness and uncertainty
- baseline comparison
- confidence
- report status
```

The eventual universe should cover whatever a typical retail investor can reasonably research or
access through retail platforms: stocks, ETFs, crypto, currencies, commodities, futures context, and
related proxies. Availability changes, so tradability has to be stored with provenance instead of
hardcoded.

## Useful Files

- [docs/architecture.md](docs/architecture.md): where the rebuild is headed.
- [docs/contracts.md](docs/contracts.md): target contracts and invariants.
- [docs/configuration.md](docs/configuration.md): local setup and storage conventions.
- [docs/roadmap.md](docs/roadmap.md): phased rebuild plan.
- [docs/testing-plan.md](docs/testing-plan.md): test strategy.
- [AGENTS.md](AGENTS.md): operating rules for Codex.
- [PLANS.md](PLANS.md): SQLite planning schema and usage rules.

`AGENTS.md` and `PLANS.md` are read-only by default. Change them only when the user specifically asks.

## Local Setup

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m nlp_stock_prediction --help
```

Run the normal checks:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
```

The legacy report command still works while the rebuild is underway:

```sh
python -m nlp_stock_prediction run --date 2026-05-12 --output reports/ --offline
```

The target command is not implemented yet, but the direction is roughly:

```sh
python -m nlp_stock_prediction research --objective daily-prediction-report --universe retail-tradable --output reports/
```

Do not treat that target command as available until it has code and tests behind it.

## SQLite

The project uses two local SQLite databases:

- `plans/planning.sqlite3`: tracked in git so planning records are part of project history.
- `data/prediction-research.sqlite3`: ignored local runtime state for runs, tools, artifacts,
  evidence, and prediction candidates.

Create or verify them with:

```python
from pathlib import Path

from nlp_stock_prediction.storage import (
    initialize_planning_database,
    initialize_research_database,
)

planning_store = initialize_planning_database(Path("plans/planning.sqlite3"))
research_store = initialize_research_database(Path("data/prediction-research.sqlite3"))
```

Durable source-of-truth docs still stay in Markdown.
