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
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --run-profile quality --short-sequence-model off
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
                         ├-> Message Model -> Telegram
                         ├-> validated text GitHub payload -> GitHub
                         └-> raw JSON GitHub payload -> GitHub
```

The validation step is required. It must stop the branch when command stdout
is blank or malformed, when no ranking rows were produced, or when generated
time, horizon, universe, or report text is missing. A failed producer must not
be rewritten as a valid “no signals” market report. The checked repair utility
adds this guard, publishes deterministic validated report text instead of
trusting free-form model output, reconnects both timestamped publishers, moves
them to `main`, and reuses an existing n8n HTTP-header credential without
copying secret values:

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
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --run-profile quality --short-sequence-model off --json-only
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
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --send-telegram
```

The scheduled workflow uses the bounded quality profile by default. It applies
adaptive sequence models only to a small prior-validation subset in the 30-day
pass, keeps the separate 1-day evaluation, and leaves 1-day sequence models off
until the adaptive selector separates evidence by horizon. `Best Validation` is
a pre-registered live champion order:
the fixed non-RL ensemble first, then Ridge if the ensemble is unavailable.
XGBoost and sequence models remain research comparisons. RL may be enabled only
for shadow diagnostics; it is excluded from champion selection, ensemble
weights, reliability, published allocations, and orders.

The forecast horizon is point-to-point. A positive 30-session forecast does
not claim that price should rise immediately or monotonically. The append-only
outcome ledger records maximum adverse and favorable excursion so a forecast
that first falls sharply and later recovers is evaluated as that full path, not
just by its final sign.

## Useful arguments

```bash
--horizon 30
--run-profile quality
--short-horizons 1
--short-sequence-model off
--history-days 913
--primary-model "Best Validation"
--pattern-short-window 20
--pattern-long-window 50
--sequence-model adaptive
--adaptive-sequence-min-wins 5
--adaptive-sequence-min-share 0.20
--min-signal-return-pct 2
--max-signal-rows 0
--no-market-context
--no-optimize
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
requires at least 8 out-of-sample observations, at least a 50% realized
direction hit rate, calibration error no greater than 20%, and a Brier score no
greater than 0.30. Missing or malformed validation evidence fails closed. Use
`--min-signal-return-pct` to adjust the threshold and `--max-signal-rows` only
when a report-length cap is needed.

Daily OHLCV inputs fail closed when they trail the latest completed market
session by more than `--max-data-lag-sessions`. An incomplete current-session
daily bar is removed before forecasting. This prevents a failed provider call
from silently publishing an old cache as a current ranking.

Do not rank RL against price forecasters by forecast MAE. RL is a target-weight
policy and has a different objective. The normal overnight profile retains
XGBoost and adaptive sequence models for research comparison, while the fixed
ensemble/Ridge champion remains the live forecasting baseline.

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

Recommended scheduled profile:

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
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --request-text "{{$json.message.text}}" --json-only
```

The Python runner now treats phrases like `without RL`, `no RL`, `skip RL`, and `exclude reinforcement` as a hard override. That override forces:

```text
primary_model = Best Validation
include_rl_policy = false
```

even though RL is otherwise enabled by default, and even if an upstream n8n model emits `primary_model = RL Policy` or `--include-rl-policy`.

Use `--run-profile quality --short-sequence-model off` for scheduled runs and
`--run-profile quick` for on-demand answers. Reserve `--run-profile research`
for explicit manual deep research: it trains both LSTM and Transformer for
every symbol and for each requested horizon, so a full 80-symbol 1-day plus
30-day run can take several hours. If you want the report text and Telegram
message to stay clean, omit `--show-timing`; timing details are still saved in
the JSON file.

The report uses the same Yahoo Finance data path and the same Ridge, XGBoost,
Neural Net, optional LSTM/Transformer, and Ensemble forecast comparison used in
the Streamlit app. RL is generated separately as shadow policy diagnostics.
Model weights are persisted under `market_agent/reports/model_weights/`: Ridge
updates saved sufficient statistics, XGBoost continues from its saved booster,
LSTM/Transformer reload saved PyTorch weights and fine-tune on new labeled
samples, and the shadow RL policy updates a saved Q-table. Use `--force-retrain`
when you want to ignore saved weights and rebuild from scratch.
If you omit `--symbols`, the script uses the full default stock, ETF, commodity, crypto, and meme-crypto universe from the app.
The text report and JSON rows include the recognized primary pattern, validation MAE, direction hit rate, and per-model returns for each ranked ticker, so the n8n summary and website table can describe the same pick.
The JSON payload includes all model snapshots per symbol, which makes it usable by the app and by an LLM-driven n8n branch without rerunning forecasts.
