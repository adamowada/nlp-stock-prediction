# CLI And Configuration

## Current CLI Modes

The V1 CLI supports deterministic offline report generation:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline
```

It also supports an experimental fixture-backed scrape-source path:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape
```

To explicitly call live providers from the local machine, add `--live-providers` and preferably a
cache directory:

```sh
python -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --source-mode scrape --live-providers --cache-dir cache/live
```

This writes:

- `reports/YYYY-MM-DD/report.md`
- `reports/YYYY-MM-DD/report.json`
- `reports/YYYY-MM-DD/audit/`

If both `--offline` and `--source-mode` are omitted, the CLI exits with code `3` and explains the
available explicit modes. `--source-mode scrape` uses deterministic Reddit/AP/Candlecharts/X
fixtures plus provider degradation probes by default; it does not make live network calls unless
`--live-providers` is also present.

## CLI Options

- `--date`: report date in `YYYY-MM-DD` format.
- `--output`: base output directory; the app writes into `<output>/<YYYY-MM-DD>/`.
- `--capital`: optional non-negative account capital for risk gates.
- `--risk-profile`: scoring/report risk profile; defaults to `exploratory`.
- `--fixture-dir`: optional fixture root reserved for external fixtures; current offline runs record
  this path in command metadata and use built-in deterministic fixtures.
- `--cache-dir`: optional provider response cache directory; deterministic runs record this path in
  command metadata.
- `--source-mode`: explicit source mode. `scrape` runs the experimental compliance-aware provider
  path against deterministic fixtures and writes provider-result audit artifacts.
- `--offline`: uses the deterministic offline fixture-backed report path and prevents live network
  provider usage.
- `--live-providers`: opt into real provider calls for `--source-mode scrape`. This calls public
  Reddit/AP/Candlecharts HTML providers and the X recent-search API while degrading expected
  provider failures into provider-result warnings.

## Environment Variables

The default test suite, `run --offline`, and `run --source-mode scrape` do not require credentials,
`.env` files, or network access. The CLI automatically loads the nearest local `.env` file when it
starts, without overriding variables already exported in the shell. Live smoke checks are explicit
opt-in:

| Variable | Used for | Required when |
| --- | --- | --- |
| `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS` | Enables live API/scraping tests when set to `1`. | Running `pytest -m live_api` or `pytest -m live_scraping` against real services. |
| `NLP_STOCK_PREDICTION_LIVE_USER_AGENT` | Identifies SEC live API smoke requests. | Running the SEC live API smoke test. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL` | Narrow URL for the public scraping smoke test. | Running the live scraping smoke test. |
| `NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT` | Literal text expected in the configured scraping smoke response. | Running the live scraping smoke test. |
| `NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT` | Optional User-Agent for future compliance-aware public HTML adapters. | Manual/live scraping adapter runs; deterministic tests use injected transports. |
| `NLP_STOCK_PREDICTION_SCRAPE_MIN_DELAY_SECONDS` | Optional non-negative crawl delay floor for future scraping adapters; defaults to `1.0`. | Manual/live scraping adapter runs that enforce polite throttling. |
| `NLP_STOCK_PREDICTION_LIVE_LLM_SMOKE` | Reserved for future live LLM smoke wiring. | Leave unset or `0` until live LLM adapter wiring is enabled. |
| `NLP_STOCK_PREDICTION_DISABLE_DOTENV` | Disables CLI `.env` auto-loading when set to `1`. | Deterministic tests, CI, or debugging an exported shell environment. |

Provider adapters already return structured `missing_credentials` warnings when keys are absent.
`--source-mode scrape --live-providers` uses the configured live provider variables below where
the corresponding provider is wired. Missing optional credentials degrade into provider warnings
instead of blocking the whole report:

| Variable | Intended provider |
| --- | --- |
| `NLP_STOCK_PREDICTION_ALPHA_VANTAGE_API_KEY` | Alpha Vantage market data and fundamentals. |
| `NLP_STOCK_PREDICTION_FRED_API_KEY` | FRED macro data. |
| `NLP_STOCK_PREDICTION_X_API_KEY` | X App API Key, also called the Consumer Key; used to identify the app and regenerate app-only tokens when needed. |
| `NLP_STOCK_PREDICTION_X_API_SECRET` | X App API Secret, also called the Consumer Secret; keep secret and use only for token generation or OAuth flows. |
| `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` | X App-only Bearer Token for read-only recent-search requests and X live smoke. |
| `NLP_STOCK_PREDICTION_NEWS_API_KEY` | Public news provider adapters that require an API key. |

## X API Stock News

The X provider direction is official X API recent search, not browser scraping. X is treated as a
stock-news/social evidence source for every discovered ticker. The default scrape source mode uses
deterministic X API-shaped fixtures; `--source-mode scrape --live-providers` uses the configured X
Bearer Token for live recent-search calls. Live X verification is covered by opt-in smoke tests.

For each ticker, the app should request the top 50 relevant posts from the X recent-search endpoint:

- Query: `$TICKER lang:en -is:retweet`
- Result order: `sort_order=relevancy`
- Result count: `max_results=50`

The provider should authenticate with `NLP_STOCK_PREDICTION_X_BEARER_TOKEN`, request public post
fields such as `created_at`, `public_metrics`, `lang`, and `author_id`, and preserve the query,
sort order, returned post IDs, timestamps, metrics, and API response snapshot IDs in provenance and
audit artifacts. The app intentionally avoids `sort_order=recency` for production stock-news
evidence because the smoke test showed too much spam/noise. API key and secret values are kept in
`.env` for completeness and token rotation; normal read-only recent-search calls should use the
Bearer Token.

## Public HTML Scraping Guardrails

Shared adapter helpers use a source policy registry before public HTML fetches. Policies describe
allowlisted paths, disallowed paths, robots review status, login and JavaScript requirements, and
fallback behavior. Blocked, login-required, and markup-drift cases return `ProviderResult` warnings
instead of raising for expected provider conditions. `--source-mode scrape` carries these provider
warnings into report provider health and `audit/provider-results.json`.

Live scrape-mode orchestration currently calls:

- Reddit public `r/wallstreetbets` pages for ticker discovery and public discussion evidence.
- AP News public financial-markets hub and linked public article pages.
- Candlecharts public live chart page as a feasibility probe for first-party OHLCV only.
- X API v2 recent search for the six report tickers when `NLP_STOCK_PREDICTION_X_BEARER_TOKEN` is
  configured.

The default public HTML request identity is
`nlp-stock-prediction/0.1 compliance-aware-scraper`. Override it with
`NLP_STOCK_PREDICTION_SCRAPE_USER_AGENT` for manual/live adapter runs when a more specific contact
string is appropriate. `NLP_STOCK_PREDICTION_SCRAPE_MIN_DELAY_SECONDS` is reserved as the shared
polite crawl-delay floor for future live adapters; keep it non-negative.

## Candlecharts Data Limitation

The Candlecharts adapter is a feasibility probe for public first-party OHLCV only. Fixture-backed
HTML tests cover JSON/table candle data when it is present, but if a page only exposes an embedded
TradingView/widget chart, the provider returns a structured `no_data` warning and does not scrape
TradingView internals. Downstream ML or technical-analysis work should use approved provider data or
user-supplied OHLCV CSV fixtures when Candlecharts is widget-only.

## Local ML Technical Analysis

The ML technical-analysis lane builds local research artifacts from user-supplied OHLCV CSV files
and records model, dataset, hardware, and usage-limitation metadata. Outputs are not investment
advice. When an evaluated model is explicitly attached to an analysis bundle, the report treats it
as a conservative sidecar with model/dataset hashes, probability, calibration, freshness, and
validation metrics. Recommendation scoring applies weak, stale, unavailable, wide-interval,
underqualified, or conflicting ML gates when that sidecar is present; ML output cannot qualify a
trade by itself.

### TimesFM Windows Workflow From Clean Checkout

The active TimesFM path is native Windows first. From a clean checkout, use Python 3.12, install
the lightweight dev dependencies, then install the CUDA-enabled PyTorch wheel before the optional
TimesFM extra:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[timesfm]"
```

Verify the RTX 3090 path before training:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.smoke --device cuda --steps 2 --output artifacts/ml/timesfm-smoke/smoke-result.json
```

Place local OHLCV data under an ignored path such as `data/ml/TSLA.csv`. The CSV must contain:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- optional `adjusted_close`

Use ISO-like timestamps that sort chronologically, keep one row per bar, and keep the latest bar
within the configured `--as-of` and freshness window. The TimesFM dataset builder rejects duplicate
timestamps, future bars, stale data, partial adjusted-close history, insufficient windows, and
split-like price jumps.

Train, evaluate, and attach the evaluated sidecar:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.train --csv data/ml/TSLA.csv --ticker TSLA --device cuda --output-dir artifacts/ml/TSLA/timesfm --epochs 1 --max-steps 20 --as-of 2026-05-11 --max-latest-bar-age-days 5
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.evaluate --csv data/ml/TSLA.csv --ticker TSLA --model-dir artifacts/ml/TSLA/timesfm --device cuda --output artifacts/ml/TSLA/timesfm/evaluation.json --as-of 2026-05-11 --suitability-max-latest-bar-age-days 5
.\.venv\Scripts\python.exe -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline --ml-artifact artifacts/ml/TSLA/timesfm/evaluation.json
```

Expected local artifacts:

| Path | Created by | Purpose |
| --- | --- | --- |
| `artifacts/ml/timesfm-smoke/smoke-result.json` | smoke | Verifies Python, CUDA, RTX 3090, model load, forecast shapes, memory, and tiny LoRA optimization. |
| `artifacts/ml/<TICKER>/timesfm/adapter/` | train | PEFT LoRA adapter files and weights. |
| `artifacts/ml/<TICKER>/timesfm/training-metadata.json` | train | Model ID/revision, hashes, split settings, seed, device/CUDA metadata, package versions, and usage limitations. |
| `artifacts/ml/<TICKER>/timesfm/training-metrics.json` | train | Train and validation loss summaries. |
| `artifacts/ml/<TICKER>/timesfm/evaluation.json` | evaluate | Rolling evaluation, baseline comparisons, suitability flags, and sidecar provenance. |
| `reports/<YYYY-MM-DD>/audit/ml-artifacts.json` | report run | Full TimesFM evaluation payload copied into the report audit bundle when `--ml-artifact` is used. |

Generated data and model artifacts stay out of git through `data/ml/`, `artifacts/`, and `models/`.
The workflow remains a local prediction/technical-analysis aid. It does not place orders, generate
broker payloads, or override evidence, risk, warning, scoring, or disclaimer guardrails.

### TimesFM 2.5 Windows Smoke

The active TimesFM phase uses native Windows with the RTX 3090 as the primary local environment.
TimesFM dependencies are optional so the default test suite remains lightweight and network-free.
Install the CUDA-enabled PyTorch wheel from the official PyTorch selector first, then install the
project's TimesFM extra:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[timesfm]"
```

Verify the environment with:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.smoke --device cuda --steps 2
```

The smoke command loads `google/timesfm-2.5-200m-transformers`, runs one forecast, performs a tiny
LoRA optimization loop on synthetic data, and writes
`artifacts/ml/timesfm-smoke/smoke-result.json` by default. The artifact records Python, PyTorch,
CUDA, RTX 3090 metadata, model ID, forecast shapes, losses, memory usage, and a usage limitation.
If CUDA is unavailable, dependencies are missing, or Hugging Face download/cache access fails, the
command exits with an actionable error. Hugging Face may warn about degraded symlink caching on
Windows; that is acceptable for development, though enabling Windows Developer Mode can reduce cache
duplication.

Ubuntu remains a fallback only if a later TimesFM dependency, CUDA kernel, or local training command
blocks native Windows execution.

### TimesFM Dataset Windows

Stage 2 of the active TimesFM phase adds dependency-light TimesFM window construction. The dataset
builder consumes validated local OHLCV bars, emits univariate context/future windows, and uses
`adjusted_close` only when every bar supplies it; otherwise it forecasts `close`. It does not apply
external normalization because TimesFM 2.5 performs internal instance normalization.

The window builder preserves the local ML lane's safety posture: sorted timestamps, duplicate
timestamp rejection, sufficient-history checks, stale and future-dated bar checks, split-like move
detection, partial adjusted-close rejection, stable dataset hashes, and train/validation/test splits
with purged window starts. The default test suite covers this path without requiring Torch,
Transformers, Hugging Face access, or CUDA.

The experimental fixture-backed `--source-mode scrape` path includes a TSLA ML sidecar so Markdown,
JSON, and audit payloads exercise the integration shape without requiring a local model artifact.
The explicit live-provider scrape path suppresses fixture ML, fundamental-agent, extraction, and
scoring sidecars for now; it records live provider evidence without generating live
recommendations.

### TimesFM Inference Adapter

Stage 3 of the active TimesFM phase adds a local inference adapter for
`google/timesfm-2.5-200m-transformers`. The adapter reads a TimesFM dataset window, loads the model
only at runtime, and writes a forecast artifact. Normal imports and default tests still do not
require Torch, Transformers, Hugging Face access, or CUDA.

Synthetic Windows CUDA smoke:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.adapter --synthetic --ticker TSLA --device cuda --output artifacts/ml/timesfm-forecast-smoke/forecast.json
```

CSV-backed local inference:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.adapter --csv data/ml/TSLA.csv --ticker TSLA --device cuda --as-of 2026-05-11 --max-latest-bar-age-days 5 --output artifacts/ml/TSLA/timesfm/forecast.json
```

The forecast artifact records the ticker, model ID, model revision when exposed by the loaded
model, dataset hash, input hash, forecast timestamp, context dates, horizon length, point forecast,
quantile forecasts when available, expected return, interval width, a directional probability
proxy, uncertainty, warnings, and usage limitations. Status values are `usable`, `weak`, or
`unavailable`; expected model, dependency, CUDA, or data failures become structured unavailable
artifacts instead of uncaught report-pipeline errors. The artifact is a local technical-analysis
prediction sidecar, not live trading instructions or financial advice.

### TimesFM LoRA Training

Stage 4 adds a local PEFT/LoRA training command for TimesFM 2.5. The command records all local input
identity and run metadata needed to audit a training run while keeping generated adapter weights
under ignored artifact directories.

Synthetic Windows CUDA smoke:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.train --synthetic --ticker TSLA --device cuda --output-dir artifacts/ml/timesfm-train-smoke --epochs 1 --max-steps 1 --batch-size 1 --validation-batches 1
```

CSV-backed local training:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.train --csv data/ml/TSLA.csv --ticker TSLA --device cuda --output-dir artifacts/ml/TSLA/timesfm --epochs 1 --max-steps 20 --as-of 2026-05-11 --max-latest-bar-age-days 5
```

Training writes:

- `adapter/` with PEFT adapter files such as `adapter_config.json` and adapter weights.
- `training-metadata.json` with model ID/revision, source hash, CSV hash for CSV runs, dataset hash,
  context/horizon settings, split metadata, LoRA config, seed, device/CUDA metadata, package
  versions, artifact hashes, and usage limitations.
- `training-metrics.json` with train and validation loss summaries.

Expected bad CSVs, insufficient history, stale data, future-dated bars, missing optional
dependencies, CUDA unavailability, and Hugging Face model-load failures produce actionable CLI
errors. TimesFM training output remains a local technical-analysis research artifact; it is not
financial advice, a standalone recommendation, or live trading instructions.

### TimesFM Rolling Evaluation

Stage 5 adds a local rolling evaluation command for trained TimesFM adapters. It loads a Stage 4
training directory, scores held-out TimesFM windows, compares the adapter against last-close
persistence and recent-mean-return baselines, and writes suitability flags for later report/scoring
integration.

Synthetic Windows CUDA smoke using the Stage 4 smoke adapter:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.evaluate --synthetic --ticker TSLA --model-dir artifacts/ml/timesfm-train-smoke --device cuda --output artifacts/ml/timesfm-eval-smoke/evaluation.json --max-windows 1 --min-evaluation-windows 1
```

CSV-backed local evaluation:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.evaluate --csv data/ml/TSLA.csv --ticker TSLA --model-dir artifacts/ml/TSLA/timesfm --device cuda --output artifacts/ml/TSLA/timesfm/evaluation.json --as-of 2026-05-11 --suitability-max-latest-bar-age-days 5
```

The evaluation artifact records the model ID/revision, model hash, adapter hash, training metadata
hash, evaluation dataset hash, source hash, per-window records, TimesFM MAE/RMSE/directional
accuracy, interval coverage when full predictions are available, baseline metrics, benchmark deltas,
a latest-context forward forecast from the trained adapter, and `suitable_for_scoring`.
Underperforming, insufficient, or stale evaluations are marked `weak`; later report/scoring
integration must not treat weak or unevaluated TimesFM artifacts as strong signals.

### Focused TimesFM HPO Workflow

Use the focused workflow when you want quality-first ticker-specific adapters for the current
r/wallstreetbets ticker set instead of a broad S&P 500 sweep. The command collects adjusted daily
OHLCV, trains bounded HPO candidates, chooses the best candidate by validation loss, then evaluates
that selected adapter once on held-out rolling windows. The promoted run is copied under
`artifacts/ml/wsb-six-10y/<TICKER>/best/`.

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.focused_hpo --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-11 --device cuda
```

Defaults write data under `data/ml/wsb_10y/`, artifacts under `artifacts/ml/wsb-six-10y/`, and a
manifest at `artifacts/ml/wsb-six-10y/focused-hpo-manifest.json`. The first HPO candidate is the
strong first-pass recipe: 128-session context, 16-session horizon, 1,000 steps, batch size 8,
learning rate `3e-5`, LoRA rank 8, LoRA alpha 16, and dropout 0.10. The default bounded search runs
diverse nearby candidates across context, horizon, step, batch, learning-rate, rank, and dropout
values; pass `--max-trials-per-ticker 0` only when you intentionally want the full grid.

The focused data policy is intentionally conservative. `SPY` is treated as ETF technical data,
not an operating-company fundamentals target. `SNDK` is clipped to current standalone Sandisk
history beginning 2025-02-24 rather than stitched to old pre-2016 SanDisk history. Split-heavy names
such as `NVDA` and `GOOG` use adjusted OHLCV derived from the provider's adjusted-close ratio.

### TimesFM Report Attachment

Stage 6 adds explicit report attachment for evaluated TimesFM artifacts. Use `--ml-artifact` with a
Stage 5 `evaluation.json` file to attach a TimesFM technical-analysis sidecar to the matching
ticker section:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction run --date 2026-05-11 --output reports/ --offline --ml-artifact artifacts/ml/timesfm-eval-smoke/evaluation.json
```

The report keeps deterministic technical indicators separate from TimesFM interpretation. Markdown
shows the TimesFM direction, horizon, probability proxy, confidence, status, model hash, latest
forward expected return, forward interval width, and limitations. `report.json` preserves the
`TechnicalMlSignal` sidecar, including model/dataset hashes, source artifact hash, evaluation
status, suitability reasons, baseline metrics, latest evaluation record, and latest forward
forecast. The audit bundle adds `audit/ml-artifacts.json` with the full TimesFM evaluation payload
and refreshes `audit/analysis-contexts.json` so the sidecar is reproducible from report artifacts.

Weak, stale, or unavailable TimesFM artifacts remain visible as sidecars but are not promoted to
strong signals. Existing offline and scrape fixture reports remain deterministic unless
`--ml-artifact` is provided, and the live-provider scrape path still suppresses app analysis unless
a later phase explicitly changes that policy.

### TimesFM Scoring Guardrails

Stage 7 lets Lane D scoring use an evaluated TimesFM sidecar only as a bounded adjustment to the
`technical-alignment` score component. When `--ml-artifact` is present, report generation attaches
the sidecar and re-scores the matching existing candidate so Markdown, JSON, and
`audit/scoring-inputs.json` all reflect the TimesFM guardrails. A supportive, fresh, suitable
TimesFM signal can strengthen an already evidence-supported candidate by recording a
`timesfm_adjustment` in that component's raw value and adding the source artifact ID/hash as data
references. The sidecar does not affect Reddit strength, catalyst strength, fundamentals, sector
context, macro context, liquidity, risk gates, or provider warning gates.

TimesFM sidecars that are weak, stale, unavailable, underqualified by rolling evaluation,
too uncertain because of a wide forecast interval, or contradictory with deterministic technical
analysis are recorded as score penalties and failed score gates. If a candidate only clears the
score threshold because of the TimesFM technical adjustment, Lane D adds
`timesfm-cannot-qualify-standalone` and keeps the candidate watch-only. This preserves the product
rule that TimesFM is a local prediction/technical-analysis aid, not live trading and not a
standalone recommendation engine.

### TimesFM Troubleshooting

| Symptom | What to check |
| --- | --- |
| CUDA is unavailable or the smoke says no CUDA device was found | Confirm the NVIDIA driver is installed, run the CUDA PyTorch install command above, then verify `.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"`. Use `--device cpu` only for dependency-light debugging, not the RTX 3090 acceptance path. |
| CUDA out of memory | Close other GPU workloads, lower `--batch-size`, lower `--max-steps` for smoke runs, or use a shorter `--context-length`. The default LoRA path is intentionally small, but long contexts and larger batches can still exhaust VRAM. |
| Hugging Face download, cache, or rate-limit errors | Check network access and Hugging Face availability, retry later if rate-limited, or pre-populate the Hugging Face cache. If a private or gated revision is used later, authenticate through normal Hugging Face tooling and keep tokens out of the repo. |
| Hugging Face symlink warnings on Windows | The warning is acceptable. Enabling Windows Developer Mode or running an admin shell can reduce duplicated cache files, but neither is required for correctness. |
| Windows path or quoting errors | Prefer PowerShell paths with `.\.venv\Scripts\python.exe`, quote extras as `".[dev]"` or `".[timesfm]"` only when your shell requires it, and keep generated artifacts under ignored paths such as `artifacts/ml/...`. |
| Training rejects the CSV | Check required columns, duplicate timestamps, future-dated bars, stale latest bar relative to `--as-of`, missing volume values, partial `adjusted_close`, and split-like price jumps. |
| Evaluation writes `weak` or `suitable_for_scoring=false` | Inspect `suitability_reasons` in `evaluation.json`. Weak artifacts can still be attached to reports for audit visibility, but scoring treats them as non-supportive sidecars. |

Default tests use a pure-Python CPU logistic baseline and do not require CUDA, PyTorch, network
access, or local training data. CUDA/RTX metadata is detected only when available and when the local
training command is configured with `--device auto` or `--device cuda`.

Use `--as-of` with `--max-latest-bar-age-days` when the training CSV is expected to be current.
That gate rejects stale local data and bars dated after the as-of value, which protects local
experiments from accidental lookahead leakage.

Example CPU training command:

```sh
python -m nlp_stock_prediction.ml.train --csv data/ml/TSLA.csv --ticker TSLA --output-dir artifacts/ml/TSLA --device cpu --as-of 2026-05-11 --max-latest-bar-age-days 5
```

Training writes `model.json`, `metrics.json`, and `metadata.json`. The metadata artifact records the
seed, hyperparameters, temporal split, purged validation gap, train/validation metrics, dataset hash,
model hash, model and metrics artifact SHA-256 hashes, Python/runtime metadata, execution backend,
selected device, CUDA availability, GPU name when detected, and the usage limitation text. The CLI
stdout also returns the metadata artifact SHA-256 hash for external run manifests.

When running from an uninstalled source checkout, set `PYTHONPATH=src` or install the package in
editable mode before using `python -m`.

Example evaluation command:

```sh
python -m nlp_stock_prediction.ml.evaluate --model artifacts/ml/TSLA/model.json --csv data/ml/TSLA.csv --ticker TSLA --output artifacts/ml/TSLA/evaluation.json
```

## Fundamental Agent Analysis

The fundamental-agent lane accepts a citation-bound `FundamentalNlpAnalysisRequest` and returns a
validated `FundamentalNlpAnalysisResponse` with claims, risks, assumptions, confidence inputs, and
audit metadata. Fixture-backed providers are used in deterministic tests. Expected malformed,
unsupported, stale, contradictory, or unavailable agent outputs become provider warnings rather than
uncaught pipeline failures.

When a validated agent result is attached, the report preserves the existing deterministic
fundamental analysis and adds an `agent_signal` sidecar. The sidecar is surfaced in Markdown, JSON,
`provider-results.json`, and `analysis-contexts.json`; generated agent interpretation remains
separate from observed source evidence.

## `.env` Files

`.env` and `.env.*` are ignored by git. `.env.example` contains placeholder keys only and is safe to
commit.

The CLI automatically searches from the current working directory upward for the nearest `.env` file
and loads it on startup. Existing shell variables win over `.env` values, so temporary PowerShell
exports can override local defaults for one run. Set `NLP_STOCK_PREDICTION_DISABLE_DOTENV=1` to
disable this behavior.

For an X smoke test, create a local `.env` shaped like this and fill in the values from your X
Developer App's "Keys and tokens" page:

```dotenv
NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS=1
NLP_STOCK_PREDICTION_X_API_KEY=your-consumer-key
NLP_STOCK_PREDICTION_X_API_SECRET=your-secret-key
NLP_STOCK_PREDICTION_X_BEARER_TOKEN=your-bearer-token
```

Do not quote the values unless your shell loader requires it. Do not commit `.env`; `.env` and
`.env.*` are ignored by git. The app should never print these values in logs, reports, audit
artifacts, or test failures.

## Configured Live Smoke Paths

The checked live API paths include SEC company tickers and X recent search. SEC does not require an
API key, but SEC requests must include a contact-oriented User-Agent. X requires
`NLP_STOCK_PREDICTION_X_BEARER_TOKEN`.

PowerShell example:

```powershell
$env:NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS = "1"
$env:NLP_STOCK_PREDICTION_LIVE_USER_AGENT = "your-name your-email@example.com"
python -m pytest -m live_api tests/test_lane_f_live_smoke.py
```

The checked live scraping paths include provider-specific Reddit, AP News, and Candlecharts smoke
coverage plus a configurable narrow URL. A known-good low-risk configurable smoke is:

```powershell
$env:NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS = "1"
$env:NLP_STOCK_PREDICTION_LIVE_SCRAPE_URL = "https://example.com/"
$env:NLP_STOCK_PREDICTION_LIVE_SCRAPE_EXPECT_TEXT = "Example Domain"
python -m pytest -m live_scraping tests/test_lane_f_live_smoke.py
```

After Stage 3, live LLM smoke remains a reserved gate. The V1 CLI currently validates LLM
extraction through deterministic fixture-backed schema tests; no live LLM adapter or credential
contract is enabled yet.

## Missing Credentials

Missing provider credentials are expected configuration problems, not programmer errors. Provider
adapters should return `ProviderResult` envelopes with:

- `status=unconfigured`
- a `missing_credentials` warning
- `credential_state=missing`
- `data=None`

Live tests skip cleanly when `NLP_STOCK_PREDICTION_ALLOW_LIVE_TESTS` is not set. Once live tests are
explicitly requested with that opt-in flag, missing per-test configuration fails with an actionable
message naming the required environment variable. Network, quota, HTTP, and upstream availability
failures also fail with provider-specific context instead of raw socket tracebacks.
