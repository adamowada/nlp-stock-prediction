# TimesFM Signal Funnel And Leaderboard Plan

## Goal

Build a developer workflow that finds useful technical-analysis signals quickly by running cheap,
baseline-aware screens first, killing weak ideas early, and spending GPU-heavy HPO time only on
survivors.

The end state is one walkaway command:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-11 --device cuda --profile walkaway
```

The command should collect or reuse focused ticker data, run a staged funnel, write a durable
leaderboard, and promote only scoring-eligible artifacts:

1. Data and dataset validation.
2. Cheap baselines and deterministic technical-analysis screens.
3. Raw base TimesFM rolling evaluation without a LoRA adapter.
4. Tiny adapter smoke only for symbols/configs with enough signal.
5. Survivor HPO only for candidates that pass smoke gates.
6. Final held-out evaluation and report-ready artifact promotion.

The workflow should make it obvious when to move on. Failed, weak, stale, or underperforming ideas
must remain auditable in the leaderboard but must not become scoring support.

## Non-Goals

- Do not relax recommendation scoring guardrails to make ML look useful.
- Do not let TimesFM qualify a strategy by itself or bypass evidence/risk gates.
- Do not use the final held-out test split for HPO selection.
- Do not rebuild the old broad S&P batch workflow.
- Do not require live Reddit/news/provider collection for this research command.
- Do not make default tests depend on CUDA, Hugging Face downloads, network access, or TimesFM
  optional dependencies.
- Do not hide failed experiments; the goal is a useful research log, not only successful artifacts.

## Context

Current TimesFM pieces:

- `src/nlp_stock_prediction/ml/timesfm/focused_hpo.py` fetches adjusted Yahoo OHLCV for the focused
  six tickers, trains many LoRA candidates, selects by validation mean loss, evaluates the selected
  run once, and writes `artifacts/ml/wsb-six-10y/focused-hpo-manifest.json`.
- `src/nlp_stock_prediction/ml/timesfm/evaluate.py` evaluates a trained adapter against
  `last_close_persistence` and `recent_mean_return` baselines, then marks underperforming artifacts
  `weak`.
- `src/nlp_stock_prediction/ml/timesfm/adapter.py` can run base TimesFM inference for a latest
  forecast, but the rolling evaluator currently expects an adapter-backed training directory.
- `src/nlp_stock_prediction/ml/timesfm/dataset.py` already provides leakage-aware train,
  validation, and test windows.
- `src/nlp_stock_prediction/analysis/ml_signal.py` and
  `src/nlp_stock_prediction/scoring/recommendations.py` already keep weak/stale/conflicting
  TimesFM sidecars from supporting scoring.

Recent learning:

- The focused HPO workflow is too expensive as a first question. It can spend hours before telling
  us the adapter does not beat simple baselines.
- The current run selected MU/SPY winners by validation loss, but final evaluation marked them
  `weak` because they underperformed best baselines on RMSE and/or directional accuracy.
- We need raw base TimesFM metrics on the exact same windows to know whether fine-tuning helped,
  hurt, or merely changed the forecast.

## Proposed Funnel

Each ticker should advance through stages only when it clears that stage's kill criteria.

| Stage | Name | Expensive? | Purpose | Outputs |
| --- | --- | --- | --- | --- |
| 0 | data_check | no | Fetch/reuse adjusted OHLCV and validate dataset windows. | data sidecar, row/window counts |
| 1 | baseline_screen | no | Score naive baselines and deterministic technical-analysis features. | baseline metrics, TA snapshot |
| 2 | raw_timesfm_screen | medium | Evaluate pretrained TimesFM without adapter on capped rolling windows. | raw-base evaluation artifact |
| 3 | adapter_smoke | medium | Train one cheap LoRA recipe only if raw/base/TA screen earns it. | smoke adapter eval |
| 4 | survivor_hpo | high | Run bounded HPO only for candidate configs with positive smoke lift. | per-trial validation leaderboard |
| 5 | final_eval | high | Evaluate selected survivor on untouched held-out windows. | promoted evaluation artifact |
| 6 | report_ready | no | Emit scoring-ready artifact only if final gates pass. | manifest, leaderboard, promoted path |

Default `--profile walkaway` should run every stage automatically, but the implementation should
also support `--stop-after baseline_screen`, `--stop-after raw_timesfm_screen`, and
`--stop-after adapter_smoke` for fast debugging.

## Kill Criteria

The first implementation should make kill criteria explicit and configurable, while using
conservative defaults.

### Data Gate

- Reject symbols with insufficient train/validation/test windows for the requested context/horizon.
- Reject stale latest bars when `--as-of` and latest-bar age gates fail.
- Reject partial adjusted-close history or split-like discontinuities.
- Mark short-history tickers as `research_only` when they can run but have limited sample windows.

### Baseline Screen Gate

- Always compute and record:
  - `last_close_persistence`
  - `recent_mean_return`
  - deterministic technical-analysis direction and confidence where available
- Continue to raw TimesFM when the dataset is valid, even if baselines are strong; this stage is the
  yardstick, not a model-kill stage.

### Raw TimesFM Gate

- Evaluate raw base TimesFM on capped validation windows by default, for example
  `--screen-max-windows 32` or `64`.
- Kill adapter work for a ticker when raw TimesFM is materially worse than baselines:
  - default `raw_rmse_ratio_vs_best_baseline > 1.15`, and
  - `raw_directional_delta_vs_best_baseline < -0.05`
- Allow `research_only` when one metric is close but not good enough:
  - `raw_rmse_ratio_vs_best_baseline <= 1.10`, or
  - `raw_directional_delta_vs_best_baseline >= -0.02`

### Adapter Smoke Gate

- Run only one or two cheap adapter recipes, for example `max_steps=100-200`, small trial count, and
  capped evaluation windows.
- Kill HPO when adapter lift versus raw TimesFM is negative or trivial:
  - `adapter_rmse_ratio_vs_raw >= 1.00`, and
  - `adapter_directional_delta_vs_raw <= 0.00`
- Promote to HPO candidate when adapter improves raw TimesFM and is near or better than best
  baseline:
  - `adapter_rmse_ratio_vs_best_baseline <= 1.05`, or
  - `adapter_directional_delta_vs_best_baseline >= 0.02`

### Survivor HPO Gate

- Select HPO candidates by validation baseline lift, not validation loss alone.
- Ranking key:
  1. `suitable_for_scoring` on validation-style screen metrics.
  2. Lower `rmse_ratio_vs_best_baseline`.
  3. Higher `directional_delta_vs_best_baseline`.
  4. Higher adapter lift versus raw TimesFM.
  5. Lower validation mean loss as a tie-breaker.
- Keep the final held-out test split untouched until stage 5.

### Final Promotion Gate

- Promote only when final evaluation is `suitable_for_scoring`.
- Keep `weak`, `borderline`, `research_only`, `failed`, and `killed` artifacts in the manifest and
  leaderboard, but do not attach them as scoring support.
- Optionally copy the best `research_only` artifact separately for audit, not recommendation support.

## Leaderboard

The command should write both JSON and CSV so it is easy to inspect from PowerShell, Python, or a
spreadsheet:

- `artifacts/ml/timesfm-funnel/leaderboard.json`
- `artifacts/ml/timesfm-funnel/leaderboard.csv`
- `artifacts/ml/timesfm-funnel/manifest.json`

Each leaderboard row should represent one method/config/ticker/stage result.

Required fields:

```text
run_id
created_at
as_of
symbol
stage
method
status
kill_reason
decision
asset_type
history_start
latest_bar
bar_count
train_windows
validation_windows
test_windows
context_length
horizon_length
max_windows
runtime_seconds
device
model_id
model_revision
adapter_sha256
dataset_hash
evaluation_artifact
training_metadata
rmse
best_baseline_rmse
rmse_ratio_vs_best_baseline
directional_accuracy
best_baseline_directional_accuracy
directional_delta_vs_best_baseline
raw_timesfm_rmse
adapter_rmse_ratio_vs_raw
adapter_directional_delta_vs_raw
validation_mean_loss
interval_coverage
mean_interval_width
calibration_proxy
selected_for_next_stage
promoted_for_scoring
notes
```

Status values:

```text
passed
killed
research_only
borderline
weak
suitable
failed
skipped
```

Decision values:

```text
continue
stop
run_adapter_smoke
run_hpo
promote_for_scoring
audit_only
```

## CLI Design

Add a new module:

- `src/nlp_stock_prediction/ml/timesfm/signal_funnel.py`

Primary command:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-11 --device cuda --profile walkaway
```

Useful options:

```text
--symbols
--as-of
--device
--profile quick|walkaway|full
--data-dir
--output-root
--refresh-data
--refresh-runs
--stop-after data_check|baseline_screen|raw_timesfm_screen|adapter_smoke|survivor_hpo|final_eval
--screen-max-windows
--smoke-max-steps
--max-hpo-trials-per-ticker
--raw-rmse-kill-threshold
--raw-directional-kill-threshold
--adapter-rmse-promote-threshold
--adapter-directional-promote-threshold
--min-final-directional-accuracy
--max-final-rmse-ratio-vs-best-baseline
```

Profile defaults:

| Profile | Intended use | Behavior |
| --- | --- | --- |
| quick | minutes | data, baselines, capped raw TimesFM only |
| walkaway | unattended default | full funnel with ruthless early kills and survivor HPO |
| full | deep research | fewer early skips, broader HPO, still no final-test leakage |

## Implementation Milestones

### Milestone 1: Funnel Contracts And Leaderboard Writer

- Changes:
  - Add typed/Pydantic contracts for funnel stages, decisions, kill reasons, and leaderboard rows.
  - Add JSON and CSV leaderboard writers.
  - Add deterministic ranking helpers for stage decisions.
- Files likely affected:
  - `src/nlp_stock_prediction/ml/timesfm/signal_funnel.py`
  - `tests/test_timesfm_signal_funnel.py`
- Verification:
  - Unit tests for row serialization, status validation, decision ranking, and CSV stability.

### Milestone 2: Reuse Focused Data Collection Safely

- Changes:
  - Reuse `focused_hpo.ensure_symbol_data`, ticker policies, and adjusted OHLCV policy.
  - Add stage 0 records with row/window counts and explicit data failures.
  - Keep cache validation rules compatible with focused HPO.
- Files likely affected:
  - `src/nlp_stock_prediction/ml/timesfm/signal_funnel.py`
  - `src/nlp_stock_prediction/ml/timesfm/focused_hpo.py` if helper extraction is needed
  - `tests/test_timesfm_signal_funnel.py`
- Verification:
  - Tests cover valid data, insufficient rows, stale data, and short-history research-only state.

### Milestone 3: Baseline And Deterministic TA Screen

- Changes:
  - Add a cheap baseline screen using existing evaluation baseline calculations or shared helpers.
  - Add deterministic technical-analysis snapshot fields where existing analysis code can provide
    them without live providers.
  - Write leaderboard rows before any TimesFM model load.
- Files likely affected:
  - `src/nlp_stock_prediction/ml/timesfm/signal_funnel.py`
  - `src/nlp_stock_prediction/ml/timesfm/evaluate.py` if baseline helpers need to be public
  - `src/nlp_stock_prediction/analysis/technical.py`
  - `tests/test_timesfm_signal_funnel.py`
- Verification:
  - Tests prove baseline rows are generated without optional TimesFM dependencies.

### Milestone 4: Raw Base TimesFM Rolling Evaluation

- Changes:
  - Extend the evaluator to support raw base TimesFM without requiring adapter metadata, or add a
    funnel-local raw evaluator that shares the same records/metrics contracts.
  - Ensure raw base TimesFM uses the exact same windows and baselines as adapter evaluation.
  - Record raw TimesFM metrics and raw-stage kill decisions.
- Files likely affected:
  - `src/nlp_stock_prediction/ml/timesfm/evaluate.py`
  - `src/nlp_stock_prediction/ml/timesfm/signal_funnel.py`
  - `tests/test_timesfm_evaluation.py`
  - `tests/test_timesfm_signal_funnel.py`
- Verification:
  - Fake-model tests compare raw TimesFM metrics against baselines.
  - Tests prove raw-stage kill criteria prevent adapter smoke.

### Milestone 5: Adapter Smoke Stage

- Changes:
  - Add one or two cheap LoRA smoke recipes that run only for raw-stage survivors.
  - Evaluate smoke adapters on capped validation windows.
  - Compare adapter metrics against raw TimesFM and best baselines.
- Files likely affected:
  - `src/nlp_stock_prediction/ml/timesfm/signal_funnel.py`
  - `src/nlp_stock_prediction/ml/timesfm/train.py`
  - `tests/test_timesfm_signal_funnel.py`
- Verification:
  - Tests cover negative adapter lift, positive adapter lift, failed training, and artifact reuse.

### Milestone 6: Survivor HPO With Baseline-Lift Selection

- Changes:
  - Run bounded HPO only for adapter-smoke survivors.
  - Evaluate HPO candidates on validation-style screen windows.
  - Select by baseline lift and adapter lift, with validation loss only as a tie-breaker.
- Files likely affected:
  - `src/nlp_stock_prediction/ml/timesfm/signal_funnel.py`
  - `src/nlp_stock_prediction/ml/timesfm/focused_hpo.py` if HPO trial helpers are shared
  - `tests/test_timesfm_signal_funnel.py`
- Verification:
  - Tests prove the selector can prefer a higher-loss model when it has better baseline lift.

### Milestone 7: Final Held-Out Evaluation And Promotion

- Changes:
  - Run final evaluation only for selected HPO survivors.
  - Promote `best/evaluation.json` only when final suitability gates pass.
  - Write `manifest.json` with per-symbol final state and promoted paths.
- Files likely affected:
  - `src/nlp_stock_prediction/ml/timesfm/signal_funnel.py`
  - `src/nlp_stock_prediction/analysis/ml_signal.py` only if artifact metadata expands
  - `tests/test_timesfm_signal_funnel.py`
- Verification:
  - Tests cover suitable promotion, weak audit-only artifact, and failed final evaluation.

### Milestone 8: Documentation And Walkaway UX

- Changes:
  - Document the command, profiles, output files, and kill criteria.
  - Add clear terminal progress lines for each stage and ticker.
  - Ensure interruption leaves a readable partial manifest and leaderboard.
- Files likely affected:
  - `README.md`
  - `docs/configuration.md`
  - `plans/timesfm-signal-funnel.md`
- Verification:
  - Dry-run command and docs examples match the parser.

## Acceptance Criteria

- [ ] A single `signal_funnel --profile walkaway` command runs the staged workflow for the focused
  ticker set.
- [ ] The command writes `manifest.json`, `leaderboard.json`, and `leaderboard.csv` incrementally so
  partial runs remain inspectable.
- [ ] Raw base TimesFM metrics are recorded on the same rolling windows as adapter evaluations.
- [ ] Adapter smoke runs only for symbols/configs that pass raw-stage gates.
- [ ] HPO runs only for adapter-smoke survivors.
- [ ] HPO winner selection uses validation baseline lift before validation loss.
- [ ] Final held-out evaluation is not used for HPO selection.
- [ ] Scoring promotion happens only for final `suitable_for_scoring` artifacts.
- [ ] Killed and failed ideas include explicit `kill_reason` and remain visible in the leaderboard.
- [ ] Default tests run without CUDA, network access, or TimesFM optional dependencies.
- [ ] Documentation explains the funnel, profiles, output files, and one-command workflow.

## Verification Commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_timesfm_signal_funnel.py
.\.venv\Scripts\python.exe -m pytest tests/test_timesfm_focused_hpo.py tests/test_timesfm_evaluation.py tests/test_timesfm_dataset.py tests/test_timesfm_training.py
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --dry-run --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-11 --device cuda --profile walkaway
ruff check .
ruff format --check .
mypy .
```

Optional CUDA verification:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-11 --device cuda --profile quick
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-11 --device cuda --profile walkaway
```

## Decision Log

- 2026-05-12: Create a new `signal_funnel` command instead of expanding `focused_hpo`, because the
  developer workflow needs explicit staged decisions, raw TimesFM screens, and a leaderboard while
  the existing focused HPO command remains a useful acceptance/reference path.
- 2026-05-12: Treat raw base TimesFM as a required benchmark before adapter work, because adapter
  metrics alone cannot show whether fine-tuning helped.
- 2026-05-12: Keep final scoring gates conservative; the funnel may label `borderline` or
  `research_only`, but only final `suitable_for_scoring` artifacts can support recommendations.
- 2026-05-12: Use leaderboard rows as the main research interface so failed experiments become
  cheap, searchable learning rather than lost terminal output.

## Progress Log

- 2026-05-12: Drafted the execution plan after reviewing the current focused HPO, TimesFM
  evaluation, and scoring guardrail workflow.
- 2026-05-12: Implemented Stage 0 `data_check` with focused data reuse, dataset validation,
  incremental manifest/leaderboard output, dry-run support, and unit coverage.
- 2026-05-12: Implemented Stage 1 `baseline_screen` with cheap last-close and recent-mean-return
  validation-window baselines plus deterministic technical-analysis snapshot rows, while keeping
  dry-run and failed data checks auditable as skipped baseline rows.
- 2026-05-12: Implemented Stage 2 `raw_timesfm_screen` with capped validation-window raw base
  TimesFM evaluation, baseline-relative kill/promote decisions, raw evaluation artifacts, and
  fake-predictor unit coverage so default tests stay offline and CUDA-free.
