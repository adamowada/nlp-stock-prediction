# AGENTS.md

## Project overview

This project is a TDD-first Python CLI application for generating a daily, evidence-grounded stock opportunity report for a retail trader with a small account.

The product goal is an app that discovers the six tickers surfaced by the r/wallstreetbets Devvit daily ticker card, gathers recent public discussion and news, extracts discussed trading strategies with evidence, combines that with technical, fundamental, sector, and macro analysis, then writes a Markdown and JSON report. The current V1 CLI supports deterministic `--offline`, fixture-backed `--source-mode scrape`, and explicit opt-in live provider evidence collection with `--source-mode scrape --live-providers`; live provider runs currently collect evidence and audit metadata only and do not generate live recommendations. Phase 4 local TimesFM 2.5 technical analysis is complete on the Windows RTX 3090 path; use [plans/timesfm-technical-analysis.md](plans/timesfm-technical-analysis.md) as the acceptance record and [docs/configuration.md](docs/configuration.md) for the user workflow. The system should support exploratory stock/options ideas while clearly separating observed discussion from the app's own analysis and recommendations.

The project should favor correctness, traceability, and testability over speed of adding features. Any generated recommendation must preserve its source evidence, assumptions, risks, and confidence inputs.

See [docs/multi-milestone-plan.md](docs/multi-milestone-plan.md) for the product milestone plan and [docs/testing-plan.md](docs/testing-plan.md) for the testing strategy.

## Common commands

Use the repository's configured commands. Expected commands are:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
python -m pytest tests/test_lane_d_scoring.py tests/test_timesfm_smoke.py tests/test_timesfm_dataset.py tests/test_timesfm_adapter.py tests/test_timesfm_training.py tests/test_timesfm_evaluation.py tests/test_timesfm_report_integration.py
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape
python -m nlp_stock_prediction.ml.timesfm.smoke --device cuda --steps 2
python -m nlp_stock_prediction.ml.timesfm.adapter --synthetic --ticker TSLA --device cuda --output artifacts/ml/timesfm-forecast-smoke/forecast.json
python -m nlp_stock_prediction.ml.timesfm.train --synthetic --ticker TSLA --device cuda --output-dir artifacts/ml/timesfm-train-smoke --epochs 1 --max-steps 1 --batch-size 1 --validation-batches 1
python -m nlp_stock_prediction.ml.timesfm.evaluate --synthetic --ticker TSLA --model-dir artifacts/ml/timesfm-train-smoke --device cuda --output artifacts/ml/timesfm-eval-smoke/evaluation.json --max-windows 1 --min-evaluation-windows 1
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline --ml-artifact artifacts/ml/timesfm-eval-smoke/evaluation.json
```

Use `python -m nlp_stock_prediction` as the canonical CLI invocation until a console script is introduced.

If the final project uses a task runner, keep this section updated with the canonical commands.

## Coding rules

- Prefer simple, typed Python modules with small provider interfaces and Pydantic models at API boundaries.
- Keep source adapters separate from analysis logic, recommendation scoring, and report rendering.
- Use structured parsers or typed clients where available; avoid ad hoc string parsing except for tightly scoped extraction rules covered by tests.
- Preserve source provenance for all external data, including provider name, fetched timestamp, permalink or source URL, raw identifier, and freshness.
- Do not present Reddit, X/Twitter, or news discussion as fact without attribution.
- Do not add real-money brokerage execution in this project unless explicitly planned and approved.
- Keep secrets out of the repo. Load API keys from environment variables or local `.env` files that are ignored by git.
- Make external calls through provider adapters so tests can use fixtures and mocks without network access.
- Handle rate limits, partial provider failures, and stale data explicitly.
- Use clear names and concise comments only where the code's intent is not obvious.

## Testing rules

- Follow the testing strategy in [docs/testing-plan.md](docs/testing-plan.md).
- Write tests before or alongside behavior changes.
- Unit test ticker extraction, ticker matching, evidence normalization, strategy extraction schema validation, clustering, scoring, and report rendering.
- Use recorded fixtures for provider contract tests, plus separately marked live API and live scraping tests for dependency coverage.
- Keep the default fast suite deterministic; run live dependency tests explicitly or in scheduled CI with the required credentials, network access, and quota controls.
- End-to-end tests should include fixture-backed report generation plus separate opt-in live provider smoke coverage when the needed external dependencies are configured.
- Include negative tests for malformed HTML, duplicate or insufficient tickers, missing provider
  data, rate-limit and unavailable-provider results, stale market or macro data, unsupported
  recommendations, conflicting evidence, joke/sarcasm risk, no qualified strategies, and short
  ticker false positives.
- Any recommendation logic change must include tests for days with no qualified strategies, conflicting evidence, and at least one qualified strategy.

## Planning rules

- Use [docs/multi-milestone-plan.md](docs/multi-milestone-plan.md) as the current product roadmap.
- Use `PLANS.md` for complex tasks involving multiple modules, API/provider changes, risky behavior changes, or work that may span sessions.
- Store execution plans in `plans/`.
- Make plans decision-complete before implementation: include goal, non-goals, context, milestones, acceptance criteria, verification commands, decision log, and progress log.
- Update the progress log as meaningful work is completed or new blockers are discovered.
- Record notable product, architecture, provider, and risk-policy choices in the decision log with a timestamp.
- Keep implementation scoped to the active plan unless the user explicitly expands the task.

## Definition of done

- The requested behavior is implemented and covered by focused tests.
- Existing tests pass, and relevant lint/typecheck/build commands pass where applicable.
- Generated reports preserve evidence, provider metadata, confidence inputs, and disclaimers.
- External provider failures degrade gracefully and are visible in the report or logs.
- Documentation is updated when commands, configuration, behavior, or report structure changes.
- No secrets, raw credentials, or unrelated generated artifacts are committed.
- The final response summarizes what changed, how it was verified, and any remaining risks or follow-up work.
