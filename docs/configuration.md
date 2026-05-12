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

Train/evaluate/attach:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.train --csv data/ml/TSLA.csv --ticker TSLA --device cuda --output-dir artifacts/ml/TSLA/timesfm --epochs 1 --max-steps 20 --as-of 2026-05-11 --max-latest-bar-age-days 5
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.evaluate --csv data/ml/TSLA.csv --ticker TSLA --model-dir artifacts/ml/TSLA/timesfm --device cuda --output artifacts/ml/TSLA/timesfm/evaluation.json --as-of 2026-05-11 --suitability-max-latest-bar-age-days 5
.\.venv\Scripts\python.exe -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline --ml-artifact artifacts/ml/TSLA/timesfm/evaluation.json
```

Staged six-ticker signal funnel:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-11 --device cuda --profile walkaway
```

The signal funnel is the preferred research workflow while the baseline-aware gates are being
developed. It writes `manifest.json`, `leaderboard.json`, and `leaderboard.csv` under
`artifacts/ml/timesfm-funnel/`, evaluates cheap baselines before raw base TimesFM, and runs one
cheap LoRA adapter smoke only for raw TimesFM survivors. Use `--profile quick` to stop after the raw
TimesFM screen, or `--stop-after adapter_smoke` for an explicit smoke-stage run.

Focused six-ticker HPO workflow:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.focused_hpo --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-11 --device cuda
```

The retired broad S&P 500 batch helper workflow is no longer documented or kept in `data/ml/`.
Use focused HPO for the current WSB ticker set, and use the single-ticker commands above for
targeted research.

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
| `data/ml/wsb_10y/<TICKER>.csv` | focused_hpo | Adjusted OHLCV inputs for the current focused WSB ticker set. |
| `artifacts/ml/timesfm-funnel/leaderboard.json` | signal_funnel | Incremental staged leaderboard with data, baseline, raw TimesFM, and adapter-smoke decisions. |
| `artifacts/ml/timesfm-funnel/raw_timesfm_screen/<TICKER>.evaluation.json` | signal_funnel | Raw base TimesFM validation-window screen artifact. |
| `artifacts/ml/timesfm-funnel/adapter_smoke/<TICKER>.evaluation.json` | signal_funnel | Cheap LoRA adapter-smoke validation-window screen artifact. |
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
