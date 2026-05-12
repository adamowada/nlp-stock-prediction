# TimesFM Signal Funnel Runbook

Use this runbook when you want one command that can run unattended, kill weak TimesFM ideas early,
and promote only scoring-ready technical-analysis artifacts.

## Walkaway Command

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda --profile walkaway --refresh-data --refresh-runs
```

`--profile walkaway` runs through `report_ready` by default:

1. `data_check`
2. `baseline_screen`
3. `raw_timesfm_screen`
4. `adapter_smoke`
5. `survivor_hpo`
6. `final_eval`
7. `report_ready`

## Dates And Tickers

Use the current report date for `--as-of`. For example, a run prepared on May 12, 2026 should use
`--as-of 2026-05-12`. This is the cutoff and freshness reference. If the provider has not posted
the May 12 close yet, the latest usable bar may still be May 11; the default freshness gate allows
recent prior-session data.

To run a different WSB set, change only `--symbols`:

```powershell
--symbols MU,SPY,ASTS,SNDK,GOOG,NVDA,TSLA,AMD,PLTR
```

Use comma-separated tickers with no `$` prefix. More symbols increase runtime and disk usage, but
weak tickers should stop before expensive HPO.

## Refresh Policy

Use `--refresh-data` when you want to refetch OHLCV inputs for the requested `--as-of`.

Use `--refresh-runs` when you want a clean recompute of model/evaluation artifacts. Omit
`--refresh-runs` when adding a few tickers later and you want to reuse compatible existing artifacts
for tickers already processed with the same data/configuration.

## Fast Checks

Dry-run the orchestration and docs command without model loading:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --dry-run --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda --profile walkaway
```

Run only the cheap screen through raw base TimesFM:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda --profile quick --refresh-data
```

Stop after adapter smoke:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols MU,SPY,ASTS,SNDK,GOOG,NVDA --as-of 2026-05-12 --device cuda --profile walkaway --stop-after adapter_smoke
```

## S&P 500 Broad Scan

Use `quick` for a market-wide scan so the command stops after the raw base TimesFM screen:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --universe sp500 --as-of 2026-05-12 --device cuda --profile quick --refresh-data --refresh-runs --refresh-universe --output-root artifacts/ml/timesfm-funnel-sp500
```

The command fetches the current S&P 500 constituents from Wikipedia wikitext and caches them under
`data/ml/universes/sp500-symbols.csv`. Omit `--refresh-universe` to reuse that cached universe.

For a custom universe, use:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction.ml.timesfm.signal_funnel --symbols-file data/ml/universes/my-watchlist.txt --as-of 2026-05-12 --device cuda --profile quick --refresh-data --output-root artifacts/ml/timesfm-funnel-watchlist
```

## Outputs To Inspect

The default output root is `artifacts/ml/timesfm-funnel/`.

| Path | Purpose |
| --- | --- |
| `manifest.json` | Run metadata, final states, and promoted artifact paths. |
| `leaderboard.csv` | Spreadsheet-friendly staged decisions and metrics. |
| `leaderboard.json` | Full staged leaderboard payload. |
| `raw_candidates.csv` | Ranked raw TimesFM candidates for broad screens. |
| `raw_candidates.json` | JSON form of the ranked raw-candidate summary. |
| `raw_timesfm_screen/<TICKER>.evaluation.json` | Raw base TimesFM validation-window screen. |
| `adapter_smoke/<TICKER>.evaluation.json` | Cheap adapter-smoke evaluation for raw survivors. |
| `survivor_hpo/<TICKER>/` | Per-trial survivor HPO artifacts. |
| `final_eval/<TICKER>.evaluation.json` | Held-out final evaluation for the selected HPO adapter. |
| `report_ready/<TICKER>/best/evaluation.json` | Report-consumable artifact, written only for final suitable signals. |

The most useful first read is `leaderboard.csv`. Look at `stage`, `status`, `decision`,
`kill_reason`, `rmse_ratio_vs_best_baseline`, `directional_delta_vs_best_baseline`, and
`promoted_for_scoring`.

## Using A Promoted Artifact

If a ticker reaches `report_ready`, pass its artifact into the report command:

```powershell
.\.venv\Scripts\python.exe -m nlp_stock_prediction run --date 2026-05-12 --output reports/ --offline --ml-artifact artifacts/ml/timesfm-funnel/report_ready/MU/best/evaluation.json
```

Weak, failed, stale, missing-forward-forecast, or provenance-blocked artifacts remain auditable in
the leaderboard but must not be used as scoring support.
