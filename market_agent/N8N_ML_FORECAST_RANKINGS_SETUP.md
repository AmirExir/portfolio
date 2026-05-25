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
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py
```

3. Connect the Execute Command node to your existing **Telegram Send Message** node.
4. Set the Telegram message text to the Execute Command stdout, usually:

```text
{{$json.stdout}}
```

The script also writes:

- `market_agent/reports/ml_forecast_rankings_latest.txt`
- `market_agent/reports/ml_forecast_rankings_latest.json`
- `market_agent/reports/ml_forecast_rankings_cache_*.json`
- timestamped `.txt` and `.json` report files

The Streamlit website reads the saved forecast cache first. Generated files are ignored by normal source-control commits, so add a second **Execute Command** node after report generation to publish the generated outputs automatically:

```bash
cd /Users/amirexir/Documents/GitHub/portfolio && market_agent/commit_forecast_results.sh
```

That script publishes report outputs to the `generated-output` branch from a clean temporary checkout. It does not stage or push your normal portfolio edits. To use a different output branch, set:

```bash
GENERATED_OUTPUT_BRANCH=main
```

The recommended default is still `generated-output`, because it keeps scheduled data churn separate from normal app and portfolio commits.

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
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --json-only
```

Then parse it in a Code node:

```javascript
const report = JSON.parse($json.stdout);
return [
  {
    json: {
      path: "market_agent/reports/ml_forecast_rankings_latest.txt",
      content: report.telegram_text,
      contentBase64: Buffer.from(report.telegram_text, "utf8").toString("base64"),
      jsonContent: JSON.stringify(report.rows, null, 2),
    },
  },
];
```

Use either the publisher script above or the same GitHub upload/update logic already in your **Stock Market** branch, but with `path` set to:

```text
market_agent/reports/ml_forecast_rankings_latest.txt
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

If you want the scheduled node to precompute the optimized model comparison for the app and the LLM, keep the default optimization run and just add `--send-telegram` when you want the script to deliver the forecast text directly.

## Useful arguments

```bash
--horizon 30
--short-horizons 1
--short-sequence-model both
--history-days 913
--primary-model "Best Validation"
--pattern-short-window 20
--pattern-long-window 50
--sequence-model off
--top-n 5
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

even if an upstream n8n model accidentally emits `primary_model = RL Policy` or `--include-rl-policy`.

Use `--sequence-model both` for the slower overnight run that trains both LSTM and Transformer. You can also use `--sequence-model lstm` or `--sequence-model transformer` to run only one deep sequence model. If you want the report text and Telegram message to stay clean, omit `--show-timing`; timing details are still saved in the JSON file.

The report uses the same Yahoo Finance data path and the same Ridge, XGBoost, Neural Net, optional LSTM/Transformer, optional RL Policy, and Ensemble forecast comparison used in the Streamlit app.
Model weights are persisted under `market_agent/reports/model_weights/`: Ridge updates saved sufficient statistics, XGBoost continues from its saved booster, LSTM/Transformer reload saved PyTorch weights and fine-tune on new labeled samples, and RL Policy updates a saved Q-table. Use `--force-retrain` when you want to ignore saved weights and rebuild from scratch.
If you omit `--symbols`, the script uses the full default stock, ETF, commodity, crypto, and meme-crypto universe from the app.
The text report and JSON rows include the recognized primary pattern, validation MAE, direction hit rate, and per-model returns for each ranked ticker, so the n8n summary and website table can describe the same pick.
The JSON payload includes all model snapshots per symbol, which makes it usable by the app and by an LLM-driven n8n branch without rerunning forecasts.
