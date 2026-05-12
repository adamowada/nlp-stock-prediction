# TimesFM 2.5 Technical Analysis Phase

Status: complete.

## Goal

Make the current phase about building, fine-tuning, evaluating, and integrating a local Google
TimesFM 2.5 technical-analysis model for a given stock. The model should run locally on the user's
Windows RTX 3090 workstation first, use local OHLCV history as input, and produce an auditable
technical-analysis sidecar that can be included in Markdown, JSON, and audit artifacts.

The end state is a repeatable local workflow:

```sh
python -m nlp_stock_prediction.ml.timesfm.train --ticker TSLA --csv data/ml/TSLA.csv --output-dir artifacts/ml/TSLA/timesfm --device cuda
python -m nlp_stock_prediction.ml.timesfm.evaluate --ticker TSLA --csv data/ml/TSLA.csv --model-dir artifacts/ml/TSLA/timesfm --output artifacts/ml/TSLA/timesfm/evaluation.json
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline --ml-artifact artifacts/ml/TSLA/timesfm/evaluation.json
```

Exact command names may change during implementation, but the phase should preserve that shape:
train locally, evaluate locally, attach an evidence-preserving technical signal to the report.

## Non-goals

- Do not use hosted TimeGPT or other remote model APIs for the core phase.
- Do not broaden this phase to Chronos, Moirai, TinyTimeMixer, Toto, Time-MoE, or model ensembles.
- Do not let the ML sidecar override evidence, risk gates, provider warnings, or scoring gates.
- Do not require CUDA, Hugging Face downloads, or the TimesFM stack for the default deterministic
  test suite.

## Context

- The repo already has a local ML baseline under `src/nlp_stock_prediction/ml/`:
  - `dataset.py` builds leakage-checked OHLCV technical datasets.
  - `training.py` trains a pure-Python logistic baseline and writes model, metrics, and metadata.
  - `train.py` exposes the current baseline training CLI.
- The repo already has report-safe sidecar plumbing under
  `src/nlp_stock_prediction/analysis/ml_signal.py`.
- Existing tests include local ML dataset, training, freshness, artifact, and report integration
  coverage without requiring CUDA.
- Local ML data and generated model artifacts are intentionally ignored by git:
  `data/ml/`, `artifacts/`, and `models/`.
- A Windows-native TimesFM 2.5 smoke gate passed on 2026-05-12:
  - Model: `google/timesfm-2.5-200m-transformers`
  - Device: `NVIDIA GeForce RTX 3090`
  - Python: `3.12.13`
  - PyTorch: `2.11.0+cu128`
  - CUDA runtime: `12.8`
  - LoRA smoke: 10 optimizer steps completed
  - Trainable params: 1,382,912 out of 232,672,192 total, about 0.59%
  - Peak allocated VRAM: about 0.956GB for the tiny smoke
  - Local smoke result: `artifacts/ml/timesfm-windows-smoke/smoke-result.json`
- Hugging Face cache symlink warnings on Windows are acceptable for development. Developer Mode or
  admin shells can reduce cache duplication but are not required for correctness.

## Stages

### Stage 1: Environment And Dependency Gate

Status: implemented.

- Changes:
  - Add a dedicated optional TimesFM dependency path instead of bloating the default install.
  - Add a local smoke command or script that verifies Python, PyTorch CUDA, RTX 3090 detection,
    Transformers TimesFM import, one forecast, and one tiny LoRA optimization step.
  - Document Windows-first setup and Ubuntu fallback criteria.
- Files likely affected:
  - `pyproject.toml`
  - `docs/configuration.md`
  - new `src/nlp_stock_prediction/ml/timesfm/`
  - new `tests/test_timesfm_smoke.py`
- Done when:
  - `python -m nlp_stock_prediction.ml.timesfm.smoke --device cuda --steps 2` passes on Windows.
  - The smoke writes a JSON artifact with torch, CUDA, device, model ID, forecast shapes, and memory
    usage.
  - Missing CUDA, missing dependencies, missing Hugging Face access, and model-download failures
    produce actionable errors.
  - The default `python -m pytest` suite still runs without installing TimesFM dependencies.

### Stage 2: TimesFM Dataset Windows

Status: implemented.

- Changes:
  - Extend the existing OHLCV dataset layer with TimesFM-compatible context and horizon windows.
  - Keep the first implementation target simple: forecast adjusted close when available, otherwise
    close.
  - Preserve existing leakage protections: sorted timestamps, duplicate rejection, split-like move
    detection, freshness gates, and no future bars in features.
  - Do not externally normalize TimesFM inputs; TimesFM 2.5 applies internal instance normalization.
  - Support future extension to multivariate inputs, but keep the first deliverable univariate.
- Files likely affected:
  - `src/nlp_stock_prediction/ml/dataset.py`
  - new `src/nlp_stock_prediction/ml/timesfm/dataset.py`
  - `tests/test_ml_dataset.py`
  - new `tests/test_timesfm_dataset.py`
- Done when:
  - Tests verify context/horizon shapes, target selection, temporal ordering, and metadata hashes.
  - Tests prove future labels are not included in context windows.
  - Tests cover insufficient history, stale data, future-dated bars, duplicate timestamps, and split
    leakage.
  - A fixture CSV can produce deterministic TimesFM training, validation, and test windows.

### Stage 3: TimesFM Inference Adapter

Status: implemented.

- Changes:
  - Add a local adapter that loads `google/timesfm-2.5-200m-transformers` through Transformers.
  - Run inference on a given ticker's latest context window and capture point/quantile forecasts.
  - Convert forecasts into report-safe technical fields: horizon, expected return, interval width,
    directional probability proxy, uncertainty, and model metadata.
  - Keep the adapter behind dependency checks so normal imports do not require Transformers/Torch.
- Files likely affected:
  - new `src/nlp_stock_prediction/ml/timesfm/adapter.py`
  - new `src/nlp_stock_prediction/ml/timesfm/contracts.py`
  - `src/nlp_stock_prediction/ml/timesfm/__init__.py`
  - `tests/test_timesfm_adapter.py`
- Done when:
  - Unit tests pass with a fake TimesFM model and no CUDA.
  - A local CUDA smoke loads TimesFM 2.5 and writes a forecast artifact for a synthetic or fixture
    ticker.
  - Forecast artifacts include model ID, model revision when available, input hash, dataset hash,
    forecast horizon, and forecast timestamp.
  - Adapter failures degrade into structured unavailable/weak signal states instead of crashing the
    report pipeline.

### Stage 4: LoRA Fine-Tuning CLI

Status: implemented.

- Changes:
  - Add a TimesFM LoRA training command using PEFT with a conservative default configuration.
  - Record all training inputs: ticker, CSV hash, dataset hash, model ID, adapter config, seed,
    context length, horizon length, train/validation split, device, CUDA metadata, and package
    versions.
  - Save adapter weights, config, metrics, and metadata under `artifacts/ml/<TICKER>/timesfm/`.
  - Keep generated artifacts out of git.
- Files likely affected:
  - new `src/nlp_stock_prediction/ml/timesfm/train.py`
  - new `src/nlp_stock_prediction/ml/timesfm/artifacts.py`
  - `tests/test_timesfm_training.py`
  - `docs/configuration.md`
- Done when:
  - A Windows RTX 3090 command completes a short training run on local or fixture data.
  - The command writes adapter weights, `training-metadata.json`, `training-metrics.json`, and SHA-256
    hashes.
  - Tests cover deterministic metadata with a fake trainer and do not download model weights.
  - CLI errors clearly identify bad CSVs, insufficient bars, stale data, missing CUDA when requested,
    missing dependencies, and Hugging Face download problems.

### Stage 5: Rolling Evaluation And Baselines

Status: implemented.

- Changes:
  - Add a walk-forward evaluation command for a given ticker and trained adapter.
  - Compare TimesFM against simple baselines:
    - last-close persistence
    - recent mean return
    - existing local logistic technical baseline where applicable
  - Track directional accuracy, MAE/RMSE, interval coverage, calibration proxy, benchmark deltas,
    and a latest-context forward forecast from the trained adapter.
  - Mark weak, stale, or underperforming models as unsuitable for scoring support.
- Files likely affected:
  - new `src/nlp_stock_prediction/ml/timesfm/evaluate.py`
  - `src/nlp_stock_prediction/ml/training.py`
  - `tests/test_timesfm_evaluation.py`
  - `docs/configuration.md`
- Done when:
  - Evaluation writes `evaluation.json` with model hash, adapter hash, dataset hash, baseline metrics,
    pass/fail suitability flags, and the latest forward forecast used by the report sidecar.
  - Tests cover a model that beats the naive baseline, a model that does not, and a stale evaluation.
  - Evaluation can run separately from training so a user can re-score new local CSV data.
  - The report pipeline never treats an unevaluated or underperforming adapter as a strong signal.

### Stage 6: Technical Analysis Integration

Status: implemented.

- Changes:
  - Extend the existing `TechnicalMlSignal` sidecar or add a TimesFM-specific sidecar while preserving
    the current report contract style.
  - Attach evaluated TimesFM output and the latest trained-adapter forward forecast to per-ticker
    technical analysis.
  - Surface the signal in Markdown, JSON, and audit artifacts with model provenance.
  - Preserve separation between deterministic technical indicators and ML interpretation.
- Files likely affected:
  - `src/nlp_stock_prediction/cli.py`
  - `src/nlp_stock_prediction/pipeline.py`
  - `src/nlp_stock_prediction/contracts/analysis.py`
  - `src/nlp_stock_prediction/contracts/providers.py`
  - `src/nlp_stock_prediction/contracts/report.py`
  - `src/nlp_stock_prediction/analysis/ml_signal.py`
  - `src/nlp_stock_prediction/reporting/markdown.py`
  - `tests/test_timesfm_report_integration.py`
  - `tests/test_phase1_cli_contracts.py`
- Done when:
  - Markdown includes a concise TimesFM technical signal with horizon, latest forward forecast
    direction, confidence, uncertainty, and model hash.
  - JSON preserves the full model/evaluation/audit provenance needed to reproduce the signal.
  - Reports still render cleanly when the TimesFM artifact is missing, stale, weak, or unavailable.
  - Existing offline and scrape fixture reports remain deterministic unless an ML artifact is
    explicitly attached.

### Stage 7: Scoring Guardrails

Status: implemented.

- Changes:
  - Allow TimesFM output to contribute only to the technical-analysis component of scoring.
  - Add penalties or weak-signal states for stale data, poor evaluation, wide forecast intervals, and
    contradiction with deterministic technical analysis.
  - Keep TimesFM from qualifying an opportunity by itself.
- Files likely affected:
  - `src/nlp_stock_prediction/scoring/recommendations.py`
  - `src/nlp_stock_prediction/scoring/risk.py`
  - `tests/test_lane_d_scoring.py`
  - `docs/contracts.md`
- Done when:
  - Tests cover supportive, weak, stale, conflicting, and unavailable TimesFM sidecars.
  - Tests prove the TimesFM sidecar cannot overcome failed evidence gates or risk gates.
  - Tests include a report where nothing qualifies and a report where TimesFM strengthens an already
    evidence-supported opportunity.
  - Scoring output records exactly how the TimesFM sidecar affected the technical component.

### Stage 8: User Workflow And Documentation

Status: implemented.

- Changes:
  - Document the Windows-first TimesFM setup, including PyTorch CUDA install, Hugging Face cache
    behavior, optional Developer Mode note, and RTX 3090 verification.
  - Document local CSV format, training command, evaluation command, report attachment command, and
    artifact locations.
  - Add troubleshooting for CUDA unavailable, out-of-memory, HF rate limits, symlink warnings, and
    Windows path issues.
  - Update roadmap/status docs to make this phase the current active phase.
- Files likely affected:
  - `README.md`
  - `AGENTS.md`
  - `docs/configuration.md`
  - `docs/multi-milestone-plan.md`
  - this plan
- Done when:
  - A user can follow docs from a clean Windows checkout to a TimesFM evaluation artifact.
  - Docs explain the local prediction/technical-analysis workflow.
  - Verification commands and artifact paths are current.
  - The active roadmap points to this plan as the source of truth for the current phase.

### Stage 9: Phase Acceptance And Cleanup

Status: implemented.

- Changes:
  - Run the full deterministic suite and focused TimesFM tests.
  - Run one opt-in Windows CUDA TimesFM smoke with the RTX 3090.
  - Check that generated artifacts are ignored by git and no secrets or model weights are committed.
  - Record final verification in the progress log.
- Files likely affected:
  - this plan
  - docs touched by prior stages
- Done when:
  - `python -m pytest` passes.
  - `python -m pytest -m "not live_api and not live_scraping"` passes.
  - `ruff check .`, `ruff format --check .`, and `mypy .` pass.
  - The Windows TimesFM smoke/fine-tune/evaluate command passes on the RTX 3090.
  - A report generated with an explicit TimesFM evaluation artifact includes the sidecar in Markdown,
    JSON, and audit outputs.
  - The worktree has no unrelated generated artifacts staged.

## Acceptance Criteria

- [x] TimesFM 2.5 is the only foundation model used in this phase.
- [x] Windows RTX 3090 is the primary local training path, with Ubuntu retained only as fallback.
- [x] TimesFM dependencies are optional and do not affect the default deterministic test suite.
- [x] Local OHLCV input is validated for freshness, ordering, sufficient history, and leakage risks.
- [x] Training writes adapter, metadata, metrics, hashes, package versions, and hardware metadata.
- [x] Evaluation compares TimesFM to simple baselines and records suitability flags.
- [x] Evaluation records a latest-context forward forecast from the trained adapter.
- [x] Report integration preserves model provenance, dataset provenance, confidence inputs, and warnings.
- [x] TimesFM output cannot bypass evidence/risk gates.
- [x] Documentation explains setup, commands, artifacts, and troubleshooting.

## Verification Commands

Default deterministic checks:

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
```

Focused TimesFM deterministic checks:

```sh
python -m pytest tests/test_lane_d_scoring.py tests/test_timesfm_smoke.py tests/test_timesfm_dataset.py tests/test_timesfm_adapter.py tests/test_timesfm_training.py tests/test_timesfm_evaluation.py tests/test_timesfm_report_integration.py
```

Opt-in local Windows CUDA checks:

```sh
python -m nlp_stock_prediction.ml.timesfm.smoke --device cuda --steps 2 --output artifacts/ml/timesfm-smoke/smoke-result.json
python -m nlp_stock_prediction.ml.timesfm.adapter --synthetic --ticker TSLA --device cuda --output artifacts/ml/timesfm-forecast-smoke/forecast.json
python -m nlp_stock_prediction.ml.timesfm.train --synthetic --ticker TSLA --device cuda --output-dir artifacts/ml/timesfm-train-smoke --epochs 1 --max-steps 1 --batch-size 1 --validation-batches 1
python -m nlp_stock_prediction.ml.timesfm.evaluate --synthetic --ticker TSLA --model-dir artifacts/ml/timesfm-train-smoke --device cuda --output artifacts/ml/timesfm-eval-smoke/evaluation.json --max-windows 1 --min-evaluation-windows 1
python -m nlp_stock_prediction.ml.timesfm.train --ticker TSLA --csv data/ml/TSLA.csv --output-dir artifacts/ml/TSLA/timesfm --device cuda --epochs 1 --max-steps 20
python -m nlp_stock_prediction.ml.timesfm.evaluate --ticker TSLA --csv data/ml/TSLA.csv --model-dir artifacts/ml/TSLA/timesfm --output artifacts/ml/TSLA/timesfm/evaluation.json
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline --ml-artifact artifacts/ml/timesfm-eval-smoke/evaluation.json
```

Report smoke with explicit model artifact:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline --ml-artifact artifacts/ml/TSLA/timesfm/evaluation.json
```

## Decision Log

- 2026-05-12-00-00: Use Google TimesFM 2.5 as the only time-series foundation model for this phase,
  because the user wants the quickest local black-box path and has approved TimesFM over other
  local TSFM candidates.
- 2026-05-12-00-00: Treat native Windows as the primary training environment because the Codex app is
  available there and the Windows smoke gate passed on the RTX 3090. Ubuntu remains a fallback only
  if a later dependency or kernel requirement blocks native Windows.
- 2026-05-12-00-00: Start with PEFT/LoRA fine-tuning rather than full fine-tuning because the smoke
  showed a small trainable parameter set, low VRAM use, and a clean adapter artifact workflow.
- 2026-05-12-00-00: Keep TimesFM as a technical-analysis sidecar because project policy requires
  evidence, risk gates, and confidence inputs.

## Progress Log

- 2026-05-12-00-00: Confirmed native Windows feasibility with a temporary TimesFM smoke environment.
  PyTorch CUDA detected the RTX 3090, TimesFM 2.5 loaded on `cuda:0`, inference worked, and a 10-step
  LoRA optimization loop wrote local ignored artifacts under `artifacts/ml/timesfm-windows-smoke/`.
- 2026-05-12-00-00: Created this active implementation plan for the TimesFM 2.5 technical-analysis
  phase.
- 2026-05-12-00-00: Implemented Stage 1 with an optional `timesfm` dependency group, a
  dependency-light `nlp_stock_prediction.ml.timesfm.smoke` command, deterministic unit coverage for
  missing dependencies/CUDA errors/artifact writing, and Windows-first configuration docs.
- 2026-05-12-00-00: Stage 1 verification passed on native Windows after installing the optional
  TimesFM stack into the repo venv: full pytest reported 331 passed and 7 opt-in live skips;
  non-live pytest reported 331 passed and 7 deselected; `ruff check .`, `ruff format --check .`,
  and `mypy .` passed; `python -m nlp_stock_prediction.ml.timesfm.smoke --device cuda --steps 2`
  loaded TimesFM 2.5 on the RTX 3090, produced forecast shapes `[1, 128]` and `[1, 128, 10]`, ran
  two LoRA optimizer steps, and wrote the ignored smoke artifact under `artifacts/ml/timesfm-smoke/`.
- 2026-05-12-00-00: Implemented Stage 2 with dependency-light TimesFM window contracts, deterministic
  univariate context/future window generation, adjusted-close-when-complete target selection,
  train/validation/test window splits with purged window starts, dataset hashing, and tests for
  shape, target selection, temporal ordering, future-label separation, CSV determinism, stale data,
  future-dated bars, duplicate timestamps, split leakage, and partial adjusted-close history.
- 2026-05-12-00-00: Stage 2 verification passed on native Windows: full pytest reported 341 passed
  and 7 opt-in live skips; non-live pytest reported 341 passed and 7 deselected; `ruff check . --no-cache`,
  `ruff format --check . --no-cache`, and `mypy .` passed.
- 2026-05-12-00-00: Implemented Stage 3 with a dependency-gated TimesFM inference adapter, a
  report-safe forecast artifact contract, synthetic and CSV-backed adapter CLI inputs, point and
  quantile forecast capture, forecast summary fields, structured unavailable artifacts for model or
  dependency failures, and fake-model unit coverage that does not require CUDA.
- 2026-05-12-00-00: Stage 3 Windows CUDA adapter smoke passed:
  `python -m nlp_stock_prediction.ml.timesfm.adapter --synthetic --ticker TSLA --device cuda --output artifacts/ml/timesfm-forecast-smoke/forecast.json`
  loaded `google/timesfm-2.5-200m-transformers`, captured model revision
  `5a9806b9b291fad9233b5249d88263f1846304d3`, wrote a usable 16-session forecast artifact with
  dataset/input hashes, point and 10 quantile forecasts, forecast timestamp, and limitation text.
- 2026-05-12-00-00: Stage 3 review tightened full-prediction shape validation so malformed
  quantile paths degrade to a structured unavailable artifact. Final verification passed:
  full pytest reported 345 passed and 7 opt-in live skips; non-live pytest reported 345 passed
  and 7 deselected; `ruff check . --no-cache`, `ruff format --check .`, `mypy .`, and
  `git diff --check` passed.
- 2026-05-12-00-00: Implemented Stage 4 with a dependency-gated
  `nlp_stock_prediction.ml.timesfm.train` command, PEFT/LoRA training artifact contracts, adapter
  directory hashing, `training-metadata.json`, `training-metrics.json`, deterministic fake-trainer
  tests, source/data hashes, split metadata, package/runtime metadata, and actionable CLI failures.
- 2026-05-12-00-00: Stage 4 Windows CUDA LoRA smoke passed:
  `python -m nlp_stock_prediction.ml.timesfm.train --synthetic --ticker TSLA --device cuda --output-dir artifacts/ml/timesfm-train-smoke --epochs 1 --max-steps 1 --batch-size 1 --validation-batches 1`
  loaded `google/timesfm-2.5-200m-transformers` on the RTX 3090, captured model revision
  `5a9806b9b291fad9233b5249d88263f1846304d3`, wrote ignored adapter artifacts, and recorded train
  and validation loss summaries plus adapter, metrics, and metadata hashes.
- 2026-05-12-00-00: Stage 4 deterministic verification passed: focused TimesFM tests reported
  22 passed; full pytest reported 349 passed and 7 opt-in live skips; non-live pytest reported
  349 passed and 7 deselected; `ruff check . --no-cache`, `ruff format --check .`, `mypy .`, and
  `git diff --check` passed.
- 2026-05-12-00-00: Implemented Stage 5 with a dependency-gated
  `nlp_stock_prediction.ml.timesfm.evaluate` command, rolling held-out window evaluation, TimesFM
  MAE/RMSE/directional accuracy and interval metrics, last-close persistence and recent-mean-return
  baselines, benchmark deltas, adapter/model/training metadata hashes, stale and underperforming
  suitability reasons, and fake-predictor tests that do not download model weights.
- 2026-05-12-00-00: Stage 5 Windows CUDA evaluation smoke passed:
  `python -m nlp_stock_prediction.ml.timesfm.evaluate --synthetic --ticker TSLA --model-dir artifacts/ml/timesfm-train-smoke --device cuda --output artifacts/ml/timesfm-eval-smoke/evaluation.json --max-windows 1 --min-evaluation-windows 1 --min-directional-accuracy 0 --max-rmse-ratio-vs-best-baseline 10`
  loaded the Stage 4 synthetic LoRA adapter on the RTX 3090 and wrote an evaluation artifact with
  status `weak`, `suitable_for_scoring=false`, model hash, adapter hash, dataset hash, baseline
  metrics, and underperformance suitability reasons.
- 2026-05-12-00-00: Stage 5 review tightened adapter provenance handling so a mismatch between the
  recorded training metadata adapter hash and the current adapter directory hash is a blocking weak
  suitability reason, not just a non-blocking warning.
- 2026-05-12-00-00: Stage 5 deterministic verification passed: focused TimesFM tests reported
  29 passed; full pytest reported 356 passed and 7 opt-in live skips; non-live pytest reported
  356 passed and 7 deselected; `ruff check . --no-cache`, `ruff format --check .`, `mypy .`, and
  `git diff --check` passed.
- 2026-05-12-00-00: Implemented Stage 6 with explicit `--ml-artifact` report attachment for
  evaluated TimesFM artifacts, conversion into the conservative `TechnicalMlSignal` sidecar,
  TimesFM-specific Markdown rendering, JSON provenance fields, and a new `audit/ml-artifacts.json`
  payload that preserves the full evaluation artifact.
- 2026-05-12-00-00: Stage 6 report integration preserves the existing fixture posture: default
  offline and scrape reports remain unchanged without `--ml-artifact`, and explicit TimesFM
  artifacts attach only to the matching ticker's technical-analysis section. Later Stage 7 work
  added guarded re-scoring and scoring audit refreshes when `--ml-artifact` is present.
- 2026-05-12-00-00: Stage 6 verification passed: focused TimesFM tests reported 35 passed; focused
  TimesFM report integration tests reported 6 passed; full pytest reported 362 passed and 7 opt-in
  live skips; non-live pytest reported 362 passed and 7 deselected; `ruff check . --no-cache`,
  `ruff format --check .`, `mypy .`, `git diff --check`, doc-drift wording scan, and the
  `--ml-artifact` report smoke against `artifacts/ml/timesfm-eval-smoke/evaluation.json` passed.
- 2026-05-12-00-00: Implemented Stage 7 with Lane D TimesFM scoring guardrails. Supportive,
  suitable TimesFM sidecars can adjust only the technical-alignment component, while weak, stale,
  unavailable, wide-interval, underqualified, or conflicting sidecars create auditable penalties and
  failed gates. Scoring now blocks candidates that only cross the threshold because of the TimesFM
  technical adjustment, and tests cover no-qualified and already-evidence-supported outcomes.
- 2026-05-12-00-00: Stage 7 verification passed: full pytest reported 370 passed and 7 opt-in live
  skips; non-live pytest reported 370 passed and 7 deselected; `ruff check . --no-cache`,
  `ruff format --check .`, `mypy .`, `python -m nlp_stock_prediction --help`, `git diff --check`,
  and the live-mode wording drift scan passed.
- 2026-05-12-00-00: Implemented Stage 8 documentation with a clean Windows setup path, CUDA PyTorch
  install order, RTX 3090 smoke verification, local CSV requirements, training/evaluation/report
  commands, artifact locations, troubleshooting for CUDA, VRAM, Hugging Face, symlink, path, CSV,
  and weak-evaluation issues, plus README, AGENTS, and roadmap alignment back to this source of
  truth plan.
- 2026-05-12-00-00: Stage 8 verification passed: full pytest reported 370 passed and 7 opt-in live
  skips; non-live pytest reported 370 passed and 7 deselected; focused TimesFM deterministic tests
  reported 53 passed; `ruff check . --no-cache`, `ruff format --check .`, `mypy .`,
  `python -m nlp_stock_prediction --help`, `git diff --check`, and doc-drift scans passed.
- 2026-05-12-00-00: Implemented Stage 9 phase acceptance and cleanup. Full pytest reported
  370 passed and 7 opt-in live skips; non-live pytest reported 370 passed and 7 deselected;
  focused TimesFM deterministic tests reported 53 passed; `ruff check . --no-cache`,
  `ruff format --check .`, `mypy .`, `python -m nlp_stock_prediction --help`, and
  `git diff --check` passed.
- 2026-05-12-00-00: Stage 9 Windows RTX 3090 acceptance smoke passed. PyTorch `2.11.0+cu128`
  detected `NVIDIA GeForce RTX 3090`; TimesFM smoke wrote
  `artifacts/ml/timesfm-smoke/smoke-result.json`; synthetic adapter inference wrote
  `artifacts/ml/timesfm-forecast-smoke/forecast.json`; one-step LoRA training wrote ignored adapter,
  metadata, and metrics artifacts under `artifacts/ml/timesfm-train-smoke/`; rolling evaluation
  wrote `artifacts/ml/timesfm-eval-smoke/evaluation.json` with status `weak` and
  `suitable_for_scoring=false`; and `--ml-artifact` report generation wrote Markdown, JSON, and
  `audit/ml-artifacts.json` with the TimesFM sidecar preserved.
- 2026-05-12-00-00: Confirmed cleanup posture for Phase 4 acceptance: generated reports, local ML
  artifacts, adapter weights, caches, bytecode, and `.env` remain ignored by git; no secrets, model
  weights, or unrelated generated artifacts are staged.
- 2026-05-12-00-00: Post-acceptance branch review fixes closed three readiness gaps: Stage 5
  evaluation artifacts now include a latest-context forward forecast from the trained adapter;
  Stage 6 report sidecars use that forward forecast instead of the latest held-out backtest record;
  Stage 7 report generation re-scores the matching existing candidate and refreshes
  `audit/scoring-inputs.json`; and evaluation now rejects training metadata whose ticker does not
  match the evaluation dataset ticker.
