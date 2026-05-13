# CLI And Configuration

## Report Modes

Deterministic offline report:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
```

Fixture-backed scrape report:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape
```

Opt-in live provider collection:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape --live-providers --cache-dir cache/live
```

All report modes write:

- `reports/YYYY-MM-DD/report.md`
- `reports/YYYY-MM-DD/report.json`
- `reports/YYYY-MM-DD/audit/`

`--source-mode scrape` uses deterministic Reddit/AP/Candlecharts/X fixtures by default. Adding
`--live-providers` calls configured live providers and records provider health, normalized
evidence, and audit artifacts. Live extraction/scoring remains the next implementation phase.

## CLI Options

- `--date`: report date in `YYYY-MM-DD` format.
- `--output`: base output directory; files are written under `<output>/<YYYY-MM-DD>/`.
- `--capital`: optional non-negative account capital for risk gates.
- `--risk-profile`: scoring/report risk profile; defaults to `exploratory`.
- `--fixture-dir`: optional fixture root; current built-in fixtures still drive deterministic runs.
- `--cache-dir`: optional provider response cache directory.
- `--source-mode`: explicit mode, currently `offline` or `scrape`.
- `--offline`: use the deterministic fixture-backed path.
- `--live-providers`: call configured live providers for scrape mode.
- `--ml-artifact`: attach an evaluated local TimesFM artifact as a technical-analysis sidecar.

## Environment

The CLI loads the nearest local `.env` file unless `NLP_STOCK_PREDICTION_DISABLE_DOTENV=1` is set.
Exported shell variables win over `.env` values. The default test suite and deterministic report
modes do not require credentials or network access.

| Variable | Used for |
| --- | --- |
| `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS` | Enables marked live API/scraping tests when set to `1`. |
| `NLP_STOCK_PREDICTION_LIVE_USER_AGENT` | SEC live API smoke identity. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL` | Narrow URL for the public scraping smoke test. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT` | Literal text expected in the configured scraping smoke response. |
| `NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT` | User-Agent for public HTML adapters. |
| `NLP_STOCK_PREDICTION_SCRAPE_MIN_DELAY_SECONDS` | Optional non-negative crawl delay floor. |
| `NLP_STOCK_PREDICTION_LIVE_LLM_SMOKE` | Reserved for future live LLM adapter wiring. |
| `NLP_STOCK_PREDICTION_DISABLE_DOTENV` | Disables `.env` auto-loading. |
| `NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY` | Alpha Vantage market data and fundamentals. |
| `NLP_STOCK_PREDICTION_FRED_API_KEY` | FRED macro data. |
| `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` | X read-only recent-search requests and X smoke. |
| `NLP_STOCK_PREDICTION_X_API_KEY` | X app key for token management workflows. |
| `NLP_STOCK_PREDICTION_X_API_SECRET` | X app secret for token management workflows. |
| `NLP_STOCK_PREDICTION_NEWS_API_KEY` | News adapters that require an API key. |

Secrets belong in environment variables or ignored `.env` files.

## Public HTML Providers

The public HTML helpers preserve raw snapshots, canonical URLs, source URLs, cache keys, hashes, and
markup-drift warnings. The default request identity is:

```text
nlp-stock-prediction/0.1 public-html-adapter
```

Live scrape-mode orchestration currently calls:

- Reddit public `r/wallstreetbets` pages for ticker discovery and discussion evidence.
- AP News public financial-markets pages and linked public article pages.
- Candlecharts public chart pages as first-party OHLCV feasibility probes.
- X API v2 recent search for discovered tickers when `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` is configured.

## X Recent Search

For each ticker, the X provider requests up to 50 relevant posts:

- Query: `$TICKER lang:en -is:retweet`
- Result order: `sort_order=relevancy`
- Result count: `max_results=50`

The provider preserves query text, result order, returned post IDs, timestamps, metrics, API
snapshot IDs, and provenance metadata.

## TimesFM 2.5

Native Windows with an RTX 3090 is the primary local path. TimesFM dependencies are optional so
the default tests remain lightweight.

Setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[timesfm]"
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.smoke --device cuda --steps 2
```

CSV input belongs under an ignored path such as `data/ml/TSLA.csv` and must contain:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- optional `adjusted_close`

The dataset builder rejects duplicate timestamps, insufficient windows, future bars, stale latest
bars, partial adjusted-close history, and split-like price jumps.

Preferred TimesFM signal-funnel runbook:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda --profile walkaway --refresh-data --refresh-runs
```

The signal funnel is the preferred WSB ticker-set research workflow. It writes `manifest.json`,
`leaderboard.json`, and `leaderboard.csv` under `artifacts/ml/timesfm-funnel/`, evaluates cheap
baselines before raw base TimesFM, and runs one cheap LoRA adapter smoke only for raw TimesFM
survivors. Smoke winners then enter bounded survivor HPO, where candidates are ranked by
validation-window baseline lift before validation loss. The walkaway/full default evaluates the
selected adapter on untouched held-out test windows and emits a report-ready
`ml.timesfm.evaluation.v1` artifact only when the final baseline-aware gates pass. Survivor HPO
starts with 8 trials per ticker by default; use `--max-hpo-trials-per-ticker 24` only for a deeper
rerun of the strongest report-ready contenders.

Use the current report date for `--as-of`; if that day's close has not posted yet, the latest usable
bar may still be the prior session and the default freshness gate allows it. To run a different WSB
set later, change only the comma-separated `--symbols` list. Omit `--refresh-runs` to reuse
compatible existing artifacts, or include it for a clean recompute. See
[`docs/timesfm-funnel-runbook.md`](timesfm-funnel-runbook.md) for the full walkaway procedure.

Useful variants:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --dry-run --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda --profile walkaway
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda --profile quick --refresh-data
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda --profile walkaway --stop-after adapter_smoke
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --universe sp500 --as-of 2026-05-12 --device cuda --profile quick --refresh-data --refresh-runs --refresh-universe --output-root artifacts/ml/timesfm-funnel-sp500
```

Broad scans also write `raw_candidates.csv` and `raw_candidates.json`, ranked from the raw TimesFM
screen before adapter smoke or HPO. Use `--symbols-file` for custom watchlists and promote a short
list into walkaway mode rather than sending every raw survivor into HPO.

Train/evaluate/attach single-ticker CSV workflow:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.train --csv data/ml/TSLA.csv --ticker TSLA --device cuda --output-dir artifacts/ml/TSLA/timesfm --epochs 1 --max-steps 20 --as-of 2026-05-12 --max-latest-bar-age-days 5
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.evaluate --csv data/ml/TSLA.csv --ticker TSLA --model-dir artifacts/ml/TSLA/timesfm --device cuda --output artifacts/ml/TSLA/timesfm/evaluation.json --as-of 2026-05-12 --suitability-max-latest-bar-age-days 5
.\.venv\Scripts\python.exe -m nlp_stock_prediction run --date 2026-05-12 --output reports/ --offline --ml-artifact artifacts/ml/TSLA/timesfm/evaluation.json
```

Legacy focused HPO workflow:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.focused_hpo --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda
```

The retired broad S&P 500 batch helper workflow is no longer documented or kept in `data/ml/`.
Use the signal funnel for current WSB ticker-set research, focused HPO only for reference/comparison
runs, and the single-ticker commands above for targeted research.

Generated data and model artifacts stay out of git through `data/ml/`, `artifacts/`, `models/`,
and `reports/`.

## TimesFM Artifacts

| Path | Created by | Purpose |
| --- | --- | --- |
| `artifacts/ml/timesfm-smoke/smoke-result.json` | smoke | Python/CUDA/model-load/forecast/LoRA smoke metadata. |
| `artifacts/ml/<TICKER>/timesfm/adapter/` | train | PEFT adapter files and weights. |
| `artifacts/ml/<TICKER>/timesfm/training-metadata.json` | train | Model ID/revision, source hashes, split settings, seed, device/CUDA metadata, packages, and artifact hashes. |
| `artifacts/ml/<TICKER>/timesfm/training-metrics.json` | train | Train and validation loss summaries. |
| `artifacts/ml/<TICKER>/timesfm/evaluation.json` | evaluate | Rolling evaluation, baseline comparisons, suitability flags, and forward forecast. |
| `data/ml/wsb_10y/<TICKER>.csv` | signal_funnel/focused_hpo | Adjusted OHLCV inputs for the current focused WSB ticker set. |
| `artifacts/ml/timesfm-funnel/leaderboard.json` | signal_funnel | Incremental staged leaderboard with data, baseline, raw TimesFM, adapter-smoke, HPO, and final-eval decisions. |
| `artifacts/ml/timesfm-funnel/raw_candidates.csv` | signal_funnel | Ranked raw TimesFM broad-scan shortlist. |
| `artifacts/ml/timesfm-funnel/raw_candidates.json` | signal_funnel | JSON form of the raw TimesFM broad-scan shortlist. |
| `artifacts/ml/timesfm-funnel/raw_timesfm_screen/<TICKER>.evaluation.json` | signal_funnel | Raw base TimesFM validation-window screen artifact. |
| `artifacts/ml/timesfm-funnel/adapter_smoke/<TICKER>.evaluation.json` | signal_funnel | Cheap LoRA adapter-smoke validation-window screen artifact. |
| `artifacts/ml/timesfm-funnel/survivor_hpo/<TICKER>/<TRIAL>.evaluation.json` | signal_funnel | Per-trial survivor HPO validation-window artifact. |
| `artifacts/ml/timesfm-funnel/survivor_hpo/<TICKER>.summary.json` | signal_funnel | Survivor HPO selection summary, including the validation-selected trial. |
| `artifacts/ml/timesfm-funnel/final_eval/<TICKER>.evaluation.json` | signal_funnel | Held-out test-window final evaluation and scoring-promotion decision for the selected HPO adapter. |
| `artifacts/ml/timesfm-funnel/report_ready/<TICKER>/best/evaluation.json` | signal_funnel | Report-consumable `ml.timesfm.evaluation.v1` artifact to pass with `--ml-artifact` when final gates pass. |
| `artifacts/ml/timesfm-funnel/report_ready/<TICKER>/manifest.json` | signal_funnel | Per-ticker report-ready decision, source final-eval artifact, promoted path, and command hint. |
| `artifacts/ml/wsb-six-10y/focused-hpo-manifest.json` | focused_hpo | Per-ticker HPO status, selected run, suitability, and failure metadata. |
| `artifacts/ml/wsb-six-10y/<TICKER>/best/evaluation.json` | focused_hpo | Promoted evaluation artifact for the selected focused HPO adapter. |
| `reports/<YYYY-MM-DD>/audit/ml-artifacts.json` | report run | TimesFM evaluation payload copied into the report audit bundle. |

## Scoring Integration

TimesFM attaches to the matching ticker section as a technical-analysis sidecar. Report JSON keeps
the `TechnicalMlSignal` with model/dataset hashes, source artifact hash, status, suitability
reasons, baseline metrics, latest rolling-evaluation record, and latest forward forecast. Markdown
shows the direction, horizon, probability proxy, confidence, status, model hash, expected return,
and interval width.

Lane D scoring can use a suitable TimesFM signal as a bounded adjustment to the technical-alignment
component. Weak, stale, unavailable, underqualified, wide-interval, or contradictory TimesFM
artifacts are recorded as penalties and failed score gates. Evidence, provider-warning, and risk
gates still control qualification.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| CUDA unavailable | Confirm the NVIDIA driver, install the CUDA PyTorch wheel, then run a small `torch.cuda.is_available()` check inside `.venv`. |
| CUDA out of memory | Close other GPU workloads, lower `--batch-size`, lower `--max-steps`, or shorten context length. |
| Hugging Face download/cache errors | Check network access, retry after rate limits, or pre-populate the Hugging Face cache. |
| Windows symlink warning | Accept it for development or enable Windows Developer Mode to reduce duplicated cache files. |
| CSV rejected | Check required columns, duplicate timestamps, stale/latest bars versus `--as-of`, partial `adjusted_close`, and split-like jumps. |
| Evaluation is weak | Inspect `suitability_reasons` in `evaluation.json`; weak artifacts still attach for audit visibility, but scoring treats them as non-supportive. |
