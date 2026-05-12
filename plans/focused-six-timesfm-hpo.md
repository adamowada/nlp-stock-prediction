# Focused Six-Ticker TimesFM HPO Plan

## Goal

Replace the broad S&P 500 smoke-style batch with a quality-first TimesFM workflow for the current
r/wallstreetbets ticker set:

```text
MU, SPY, ASTS, SNDK, GOOG, NVDA
```

The workflow should collect up to 10 years of daily adjusted OHLCV for each symbol, train
ticker-specific LoRA adapters, run deterministic bounded HPO, choose the best candidate by validation
loss, and run a final held-out evaluation only for the selected adapter. Weak adapters remain
auditable but must not support recommendation scoring.

## Non-Goals

- Do not build one general multi-ticker stock predictor in this phase.
- Do not force 10 years of history for tickers whose current economic entity has less public
  trading history.
- Keep TimesFM routed through the existing evidence, risk, and scoring gates.
- Do not tune against the final report-scoring outcome.
- Do not use the final held-out TimesFM evaluation split for HPO candidate selection.

## Context

The prior S&P 500 batch used a shallow recipe (`max_steps=20`, `batch_size=2`) and produced many
`weak` adapters. That was useful as a feasibility sweep, but it is not the right quality target.

This plan keeps the per-ticker specialist-adapter model but narrows the scope to six names and gives
each candidate a serious HPO run. The data path converts Yahoo chart raw OHLC to adjusted OHLC using
the adjusted-close ratio, then writes TimesFM-compatible CSV rows with `adjusted_close`.

Ticker-specific data policies:

- `SPY`: ETF. TimesFM may use OHLCV normally, but fundamentals/reporting should use ETF-aware
  context rather than company revenue/profitability/earnings.
- `SNDK`: current standalone Sandisk history starts on 2025-02-24 after separation from Western
  Digital. Do not stitch pre-2016 SanDisk history into the current ticker series.
- `NVDA` and `GOOG`: use adjusted OHLCV so split history is continuous.
- `MU` and `ASTS`: use full available current-entity adjusted OHLCV; do not backfill unrelated
  pre-listing history.

Reference sources for ticker-history decisions:

- Sandisk separation/listing: https://www.nasdaq.com/press-release/sandisk-celebrates-nasdaq-listing-after-completing-separation-western-digital-2025-02
- NVIDIA 2024 split FAQ: https://investor.nvidia.com/files/doc_downloads/2024/06/nvidia-2024-stock-split_faq_investors.pdf
- SPY issuer page: https://www.ssga.com/us/en/individual/etfs/spdr-sp-500-etf-trust-spy

## One-Command Workflow

Run from the repo root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.focused_hpo --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-11 --device cuda
```

Default outputs:

- Data CSVs and data metadata: `data/ml/wsb_10y/`
- HPO runs and promoted best artifacts: `artifacts/ml/wsb-six-10y/`
- Run manifest: `artifacts/ml/wsb-six-10y/focused-hpo-manifest.json`

Use `--max-trials-per-ticker 0` only for a full grid. The default bounded grid runs the strongest
first-pass recipe and nearby HPO candidates so the command is suitable for an overnight run.

## Milestones

1. Remove generated broad-batch S&P data and artifacts.
2. Add focused data collection with adjusted OHLCV and ticker-lineage policy metadata.
3. Add deterministic HPO candidate generation with the strong first-pass recipe first.
4. Train each candidate with the existing TimesFM training CLI and select by validation loss.
5. Run the held-out TimesFM evaluator once for the selected adapter and promote that evaluated run.
6. Document the overnight PowerShell command and verification workflow.

## Acceptance Criteria

- The old generated `data/ml/sp500_5y`, `data/ml/sp500_constituents_2026-05-12.csv`, and
  `artifacts/ml/sp500-5y` outputs are removed.
- The focused workflow can be run with one PowerShell command.
- `SNDK` is clipped to current standalone history beginning 2025-02-24.
- Data metadata records adjusted OHLCV policy, source, effective start, and ticker-lineage notes.
- HPO starts with `context_length=128`, `horizon_length=16`, `max_steps=1000`, `batch_size=8`,
  `learning_rate=3e-5`, `lora_r=8`, `lora_alpha=16`, `lora_dropout=0.10`.
- The best HPO run is chosen by validation mean loss; held-out suitability, RMSE ratio, and
  directional accuracy are final quality-gate outputs rather than HPO selection inputs.
- Weak or failed adapters remain visible in the manifest but are not promoted as scoring support.

## Verification Commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_timesfm_focused_hpo.py
.\.venv\Scripts\python.exe -m pytest tests/test_timesfm_dataset.py tests/test_timesfm_training.py tests/test_timesfm_evaluation.py
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.focused_hpo --dry-run --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-11 --device cuda
ruff check .
ruff format --check .
mypy .
```

## Decision Log

- 2026-05-12: Keep source scripts/code, delete only generated broad-batch S&P data/artifacts.
- 2026-05-12: Use adjusted OHLCV for the focused TimesFM workflow so split-heavy histories remain
  continuous for univariate price forecasting.
- 2026-05-12: Treat `SPY` as ETF technical data, not an operating-company fundamentals target.
- 2026-05-12: Treat `SNDK` as current standalone history only, starting 2025-02-24.
- 2026-05-12: Use bounded deterministic HPO by default for overnight practicality, with
  `--max-trials-per-ticker 0` available for full-grid runs.
- 2026-05-12: Select HPO candidates by training validation loss, then run the final held-out
  evaluator only once for the selected adapter.
- 2026-05-12: Reuse cached data and run artifacts only when their lineage metadata matches the
  current ticker policy, CSV hash, model, hyperparameters, and `as_of` settings.

## Progress Log

- 2026-05-12: Removed generated S&P 500 5-year batch data/artifacts.
- 2026-05-12: Added `nlp_stock_prediction.ml.timesfm.focused_hpo` with data collection, HPO
  orchestration, best-run promotion, and focused ticker policy metadata.
- 2026-05-12: Removed obsolete generated TimesFM smoke artifacts and local TSLA example data so
  the next overnight run starts with only the focused six-ticker workflow outputs.
- 2026-05-12: Tightened focused HPO selection, cache validation, artifact reuse checks, and
  bounded-trial diversity after code review.
