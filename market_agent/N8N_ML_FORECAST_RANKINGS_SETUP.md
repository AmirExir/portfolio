# Daily ML Forecast Rankings to Telegram with n8n

Use this with your existing n8n workflow:

`Schedule Trigger -> News API -> Message a model -> Code -> Stock Market`

and:

`Message a model -> Code -> Telegram`

The ML forecast step can run from the same Schedule Trigger as a parallel branch.

## Option A: n8n sends the message

1. Add a **Schedule Trigger** node.
2. Add an **Execute Command** node named **ML Forecast Rankings**:

```bash
cd /Users/amirexir/Documents/GitHub/portfolio && /usr/bin/caffeinate -i .venv/bin/python market_agent/daily_ml_forecast_report.py --run-profile overnight
```

3. Connect the Execute Command node to your existing **Telegram Send Message** node.
4. Set the Telegram message text to the Execute Command stdout, usually:

```text
{{$json.stdout}}
```

The script also writes:

- local-only `market_agent/reports/ml_forecast_rankings_latest.txt`
- local-only `market_agent/reports/ml_forecast_rankings_latest.json`
- local-only `market_agent/reports/ml_forecast_rankings_cache_*.json`
- timestamped `.json` report files in `market_agent/reports/`
- timestamped `.txt` recommendation summaries in `market_agent/reports/optimization_summaries/`

The Streamlit website reads the newest timestamped report first, so the scheduled workflow should publish timestamped files instead of overwriting `ml_forecast_rankings_latest.*`. The `latest.*` files are kept only as local convenience files because rewriting a tracked file on every scheduled run creates recurring Git conflicts.

## Add it to the existing workflow

Recommended node layout:

```text
Schedule Trigger
├─ News API -> Message a model -> Code -> Stock Market
├─ Message a model -> Code -> Telegram
└─ ML Forecast Rankings -> Validate Optimization Output
                         ├-> deterministic publication adapter -> Telegram
                         ├-> validated text GitHub payload -> GitHub
                         └-> raw JSON GitHub payload -> GitHub
```

The validation step is required. It must stop the branch when command stdout
is blank or malformed, when no ranking rows were produced, or when generated
time, horizon, universe, or report text is missing. A failed producer must not
be rewritten as a valid “no signals” market report. The checked repair utility
adds this guard, publishes deterministic validated report text instead of
trusting free-form model output, replaces the optimization language-model node
with a deterministic publication adapter, reconnects both timestamped
publishers, moves them to `main`, and reuses an existing n8n HTTP-header
credential without copying secret values:

```bash
python3 scripts/repair_n8n_market_optimization.py \
  --database /path/to/.n8n/database.sqlite \
  --backup /path/to/database.sqlite.market-optimizer-backup \
  --workflow-id YOUR_WORKFLOW_ID
```

Stop n8n before running the repair, then start it again. The utility updates
the draft plus the active/published workflow versions and refuses to overwrite
an existing backup.

For the forecast **Telegram Code** node, use:

```javascript
return [
  {
    json: {
      text: $json.stdout,
    },
  },
];
```

Then set your Telegram message body to:

```text
={{ String($json.text ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') }}
```

Set the Telegram node parse mode to `HTML`. Optimization notification nodes
should use `Continue (using error output)` so a Telegram outage cannot prevent
the validated text and JSON reports from being published.

For the forecast **Stock Market Code** node, if you are using `--json-only`, use this command:

```bash
cd /Users/amirexir/Documents/GitHub/portfolio && /usr/bin/caffeinate -i .venv/bin/python market_agent/daily_ml_forecast_report.py --run-profile overnight --json-only
```

Then parse it in a Code node:

```javascript
const report = JSON.parse($json.stdout);
return [
  {
    json: {
      path: report.paths.txt,
      content: report.telegram_text,
      contentBase64: Buffer.from(report.telegram_text, "utf8").toString("base64"),
      jsonPath: report.paths.json,
      jsonContent: Buffer.from(JSON.stringify(report, null, 2), "utf8").toString("base64"),
    },
  },
];
```

Use the same GitHub upload/update logic already in your **Stock Market** branch, but with `path` set to:

```text
market_agent/reports/optimization_summaries/ml_forecast_rankings_<timestamp>.txt
market_agent/reports/ml_forecast_rankings_<timestamp>.json
```

If the **Stock Market** node is a GitHub `PUT /contents` HTTP request, send `contentBase64` as the GitHub `content` value.

## Option B: script sends directly to Telegram

Set these environment variables in n8n if you want the script to post directly to Telegram:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Then run:

```bash
cd /Users/amirexir/Documents/GitHub/portfolio && /usr/bin/caffeinate -i .venv/bin/python market_agent/daily_ml_forecast_report.py --run-profile overnight --send-telegram
```

The nightly workflow uses the bounded `overnight` profile by default. It runs
one optimized 30-session pass with five calendar years of requested history,
uses adaptive sequence research only where independent prior evaluations
support it, and omits the duplicate one-session universe. It stops accepting
ranking work at 210 minutes, reserves 30 minutes for ledger/report
finalization, and requires publication by 240 minutes. This leaves another 30
minutes before n8n's 270-minute external ceiling. A native model call cannot
be hard-cancelled mid-fit; the runner checks immediately after it returns and
fails closed without publishing if the ranking or publication cutoff was
crossed. The fast `scheduled` profile remains available for manual fallbacks
with fixed settings, sequence models off, and no nested search.
The deadline is one absolute workflow clock shared by the main pass, every
requested short-horizon pass, and finalization. Every requested subrun must
return the literal `run_complete: true`. Report, latest, policy-state, and
append-only ledger changes are built in a staging directory and rolled back as
a unit if publication misses its cutoff; per-model/data caches are the only
artifacts retained from an incomplete ranking run. n8n independently requires
the same literal completion value before GitHub or Telegram publication.
`Best Validation` is a pre-registered live champion order: fixed non-RL
Ensemble, Ridge, Random Forest, Gradient Boosting, then XGBoost. It is not a
same-run winner chosen after inspecting the current holdout.
RL is disabled in the overnight profile by default because it is non-actionable
shadow diagnostics; it can still be requested explicitly and remains excluded
from champion selection, ensemble weights, reliability, published allocations,
and orders.

The forecast horizon is point-to-point. A positive 30-session forecast does
not claim that price should rise immediately or monotonically. The append-only
outcome ledger records maximum adverse and favorable excursion so a forecast
that first falls sharply and later recovers is evaluated as that full path, not
just by its final sign.

## Useful arguments

```bash
--horizon 30
--run-profile overnight
--history-days 1825
--primary-model "Best Validation"
--pattern-short-window 20
--pattern-long-window 50
--sequence-model adaptive
--adaptive-sequence-min-wins 2
--adaptive-sequence-min-share 0.50
--runtime-budget-minutes 240
--min-signal-return-pct 2
--max-signal-rows 0
--force-retrain
--model-cache-max-age-days 7
--max-data-lag-sessions 1
--include-rl-policy
--no-rl-policy
--portfolio-current-weights-json '{"SPY": 0.04, "MU": 0.03}'
--earnings-payload-dir /path/to/verified/earnings_payloads
--request-text "Run a full market optimization without RL"
--show-timing
--json-only
```

By default, the primary text sections include every non-RL, non-low-reliability
model buy/sell call with an absolute forecast return of at least 2%. Smart-policy
watchlists remain separate. Research forecast visibility is also separate from
execution authorization: missing broker state keeps executable allocation at
zero without erasing a qualified research forecast. Published reliability
requires at least 8 raw out-of-sample observations and 8 horizon-strided
non-overlapping windows, positive MAE skill versus a zero-return forecast,
positive Brier skill versus the training-period direction base rate, positive
direction skill above that base rate, at least a 50% realized direction hit
rate, calibration error no greater than 20%, and a Brier score no greater than
0.30. Missing or malformed validation evidence fails closed. Use
`--min-signal-return-pct` to adjust the threshold and `--max-signal-rows` only
when a report-length cap is needed.

Daily OHLCV inputs fail closed when they trail the latest completed market
session by more than `--max-data-lag-sessions`. An incomplete current-session
daily bar is removed before forecasting. This prevents a failed provider call
from silently publishing an old cache as a current ranking.

`--history-days` is an actual calendar lookback: the overnight value `1825`
means approximately five years, not a doubled provider request. If a shared
on-disk cache begins after the requested start, the loader requests the missing
history. A shorter on-demand request returns only its deterministic window but
does not truncate the longer cache needed by the next overnight run. Each
symbol snapshot and the market-context metadata record requested start,
available start, row count, and whether provider history was limited.

Adaptive LSTM/Transformer compute also fails closed. Evidence is read only from
complete reports in payload `generated_at` order and must be out-of-sample,
horizon-spaced, unique by symbol/as-of/horizon, non-overlapping, and positive on
MAE, Brier, and direction skill. Equity spacing uses the US trading calendar;
crypto spacing uses UTC calendar days. The overnight profile also rotates at
most two deterministic exploration symbols so evidence can accumulate. All
sequence outputs, including exploration, remain research-only and cannot enter
the live ensemble or published signal until an independent promotion gate is
implemented and passed. An explicit `--adaptive-sequence-symbols` override
runs both sequence families for those named symbols, still research-only.

Do not rank RL against price forecasters by forecast MAE. RL is a target-weight
policy and has a different objective. The normal overnight profile retains
XGBoost as a live direct-model component/fallback and adaptive sequence models
as research-only comparisons. The fixed champion order above remains the live
forecasting baseline.

Positive allocation targets fail closed unless the workflow supplies verified
current executed weights through `--portfolio-current-weights-json` and enough
price history exists to build the full covariance matrix. A previous
recommendation file is not broker state and cannot authorize a new allocation.
Fetch current weights from the broker immediately before this command; never
hard-code them in the workflow.

Earnings results are interpreted on every run and appear in the Telegram text.
For near-immediate alerts during an earnings window, trigger a lightweight or
quick-profile run every five minutes and retain n8n deduplication by the
reported event timestamp. A scheduled/imprecise provider timestamp is
display-only and cannot affect allocation; policy use requires a verified
publication time, comparable actual/estimate data, and the event's effective
trading session.

For richer results, have the earnings workflow write one canonical
`SYMBOL.json` file per company into `--earnings-payload-dir`. The payload may
contain EPS, revenue, and guidance:

```json
{
  "symbol": "MU",
  "reported_at": "2026-07-29T16:05:00-04:00",
  "timestamp_quality": "provider_reported",
  "eps": {"actual": 1.10, "estimate": 1.00},
  "revenue": {"actual": 9000000000, "estimate": 8700000000},
  "guidance": {"direction": "raised"},
  "source": "verified-provider"
}
```

The interpreter rejects a mismatched symbol, future publication, stale event,
missing comparable values, or a release that is not yet effective for the
decision session.

Recommended nightly profile:

```bash
--run-profile overnight
```

Fast fixed-settings fallback profile:

```bash
--run-profile scheduled
```

Manual optimized adaptive profile:

```bash
--run-profile quality
```

Manual full-universe deep research profile:

```bash
--run-profile research
```

Quick answer profile:

```bash
--run-profile quick
```

## On-demand command safety

For Telegram on-demand runs, pass the original user text into the runner:

```bash
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --run-profile quick --request-text "{{$json.message.text}}" --json-only
```

The Python runner now treats phrases like `without RL`, `no RL`, `skip RL`, and `exclude reinforcement` as a hard override. That override forces:

```text
primary_model = Best Validation
include_rl_policy = false
```

even though RL is otherwise enabled by default, and even if an upstream n8n model emits `primary_model = RL Policy` or `--include-rl-policy`.

Use `--run-profile overnight` for the normal nightly run,
`--run-profile scheduled` for a fast fixed-settings fallback, and
`--run-profile quick` for on-demand answers. Use `--run-profile quality` only
for an explicit optimized adaptive comparison that also requests the
one-session horizon. Reserve `--run-profile research` for explicit manual deep
research: it trains both LSTM and Transformer for every symbol and for each
requested horizon, so a full 80-symbol 1-day plus 30-day run can take several
hours. If you want the report text and Telegram message to stay clean, omit
`--show-timing`; timing details are still saved in the JSON file.

General market news remains a separately published point-in-time digest. It is
not injected into price-model training or signal scoring because the repository
does not yet contain a timestamped historical news panel suitable for leakage-
safe walk-forward validation. Earnings context remains a point-in-time policy
overlay. This distinction is stated in every generated report.

The report uses the same Yahoo Finance data path and the same Ridge, XGBoost,
Neural Net, optional LSTM/Transformer, and Ensemble forecast comparison used in
the Streamlit app. Live direct-model fits are deliberately cold: existing
Ridge/XGBoost/sequence artifact helpers are not reused for live predictions
until equivalent warm-start outer-holdout evidence exists. Sequence outputs
are research-only. RL is optional shadow policy diagnostics and is off in the
overnight profile unless explicitly requested. The per-symbol result cache
avoids repeating an unchanged fit for up to `--model-cache-max-age-days`; use
`--force-retrain` to bypass that result cache and recompute the comparison.
If you omit `--symbols`, the script uses the full default stock, ETF, commodity, crypto, and meme-crypto universe from the app.
The text report and JSON rows include the recognized primary pattern, validation MAE, direction hit rate, and per-model returns for each ranked ticker, so the n8n summary and website table can describe the same pick.
The JSON payload includes all model snapshots per symbol, which makes it usable by the app and by an LLM-driven n8n branch without rerunning forecasts.
