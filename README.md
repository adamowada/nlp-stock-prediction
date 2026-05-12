# nlp-stock-prediction

A Python CLI for building evidence-grounded stock opportunity reports.

The app discovers the six tickers from the r/wallstreetbets daily Devvit ticker card, gathers public discussion/news/market context, extracts discussed strategies with citations, scores candidate ideas against risk and confidence inputs, and writes Markdown, JSON, and audit artifacts. The current branch also includes a local TimesFM 2.5 technical-analysis path for Windows + RTX 3090.

## Status

- Deterministic offline report generation is implemented.
- Fixture-backed scrape mode is implemented.
- Opt-in live provider collection is implemented for public Reddit/AP/Candlecharts/X evidence paths.
- Local TimesFM 2.5 smoke, training, rolling evaluation, report attachment, and scoring integration are implemented.
- Live provider runs currently collect evidence and audit metadata; live extraction/scoring is the next report-generation step.

## Commands

```sh
python -m pytest
python -m pytest -m "not live_api and not live_scraping"
ruff check .
ruff format --check .
mypy .
python -m nlp_stock_prediction --help
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape --live-providers --cache-dir cache/live
```

## TimesFM 2.5

Windows setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[timesfm]"
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.smoke --device cuda --steps 2
```

Preferred WSB ticker workflow:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda --profile walkaway --refresh-data --refresh-runs
```

The signal funnel is the canonical TimesFM research workflow. It writes an incremental leaderboard,
runs cheap baselines before raw TimesFM, runs a cheap adapter smoke only for raw TimesFM survivors,
runs bounded survivor HPO only for smoke winners, and emits a report-ready
`ml.timesfm.evaluation.v1` artifact only when the selected adapter clears held-out final
baseline-aware scoring gates.

Use the current report date for `--as-of`. If the market/data provider has not posted that day's
close yet, the latest usable bar may still be the prior session; the freshness gate allows this by
default. To add WSB tickers later, change only the comma-separated `--symbols` list. Omit
`--refresh-runs` when you want to reuse compatible existing artifacts, and include it when you want
a clean recompute. See [docs/timesfm-funnel-runbook.md](docs/timesfm-funnel-runbook.md) for the
full walkaway runbook.

Broad S&P 500 raw-screen scan:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --universe sp500 --as-of 2026-05-12 --device cuda --profile quick --refresh-data --refresh-runs --refresh-universe --output-root artifacts/ml/timesfm-funnel-sp500
```

For broad scans, start with `raw_candidates.csv`; it ranks raw TimesFM candidates by baseline lift
before you spend time on adapter smoke or HPO.

Single-ticker CSV workflow:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.train --csv data/ml/TSLA.csv --ticker TSLA --device cuda --output-dir artifacts/ml/TSLA/timesfm --epochs 1 --max-steps 20 --as-of 2026-05-12 --max-latest-bar-age-days 5
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.evaluate --csv data/ml/TSLA.csv --ticker TSLA --model-dir artifacts/ml/TSLA/timesfm --device cuda --output artifacts/ml/TSLA/timesfm/evaluation.json --as-of 2026-05-12 --suitability-max-latest-bar-age-days 5
.\.venv\Scripts\python.exe -m nlp_stock_prediction run --date 2026-05-12 --output reports/ --offline --ml-artifact artifacts/ml/TSLA/timesfm/evaluation.json
```

Legacy focused HPO:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.focused_hpo --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda
```

The old broad S&P 500 batch workflow has been retired. Use the signal funnel for WSB ticker-set
research, focused HPO only for reference/comparison runs, and the single-ticker train/evaluate
commands for targeted experiments.

CSV files belong under ignored paths such as `data/ml/`. Generated models, adapters, metrics, and reports belong under ignored paths such as `artifacts/` and `reports/`.

## Notes

The canonical CLI entrypoint is `python -m nlp_stock_prediction`. Keep provider credentials in environment variables or ignored `.env` files. See `docs/configuration.md` for provider settings, TimesFM artifact details, and troubleshooting.
