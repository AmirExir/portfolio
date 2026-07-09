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
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --run-profile research
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
└─ ML Forecast Rankings -> Code -> Telegram
                      └-> Code -> Stock Market
```

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
{{$json.text}}
```

For the forecast **Stock Market Code** node, if you are using `--json-only`, use this command:

```bash
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --run-profile research --json-only
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

The scheduled workflow should use the full research profile by default when it runs overnight: `Best Validation`, RL policy enabled, XGBoost/Ridge/Ensemble available, and LSTM/Transformer trained across the universe. Use the quick profile for on-demand answers when speed matters, and the quality profile when you want a balanced scheduled run.

## Useful arguments

```bash
--horizon 30
--run-profile research
--short-horizons 1
--short-sequence-model adaptive
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
--include-rl-policy
--no-rl-policy
--request-text "Run a full market optimization without RL"
--show-timing
--json-only
```

By default, the text report includes every buy/sell signal with a directional model or smart-policy call and an absolute forecast return of at least 2%. Use `--min-signal-return-pct` to adjust that threshold, and use `--max-signal-rows` only if you need to cap the report length.

Recent saved report JSONs ranked the model families this way by validation error: RL Policy was strongest where available, followed by XGBoost, then Ridge/Ensemble. LSTM and Transformer were useful for a smaller set of symbols, so the normal overnight profile keeps them through `--sequence-model adaptive` instead of running them on every ticker.

Recommended scheduled overnight full research profile:

```bash
--run-profile research
```

Balanced scheduled quality profile:

```bash
--run-profile quality
```

Quick answer profile:

```bash
--run-profile quick
```

Deep overnight profile:

```bash
--run-profile research
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

Use `--run-profile research` for the normal overnight run if you are comfortable with the runtime. It selects `Best Validation`, keeps RL Policy/XGBoost/Ridge/Ensemble in the candidate set, and trains both LSTM and Transformer on every symbol. Use `--run-profile quality` for the balanced adaptive run, and `--run-profile quick` for on-demand answers. If you want the report text and Telegram message to stay clean, omit `--show-timing`; timing details are still saved in the JSON file.

The report uses the same Yahoo Finance data path and the same Ridge, XGBoost, Neural Net, optional LSTM/Transformer, default-on RL Policy, and Ensemble forecast comparison used in the Streamlit app.
Model weights are persisted under `market_agent/reports/model_weights/`: Ridge updates saved sufficient statistics, XGBoost continues from its saved booster, LSTM/Transformer reload saved PyTorch weights and fine-tune on new labeled samples, and RL Policy updates a saved Q-table. Use `--force-retrain` when you want to ignore saved weights and rebuild from scratch.
If you omit `--symbols`, the script uses the full default stock, ETF, commodity, crypto, and meme-crypto universe from the app.
The text report and JSON rows include the recognized primary pattern, validation MAE, direction hit rate, and per-model returns for each ranked ticker, so the n8n summary and website table can describe the same pick.
The JSON payload includes all model snapshots per symbol, which makes it usable by the app and by an LLM-driven n8n branch without rerunning forecasts.
