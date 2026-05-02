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
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --no-optimize
```

3. Connect the Execute Command node to your existing **Telegram Send Message** node.
4. Set the Telegram message text to the Execute Command stdout, usually:

```text
{{$json.stdout}}
```

The script also writes:

- `market_agent/reports/ml_forecast_rankings_latest.txt`
- `market_agent/reports/ml_forecast_rankings_latest.json`
- timestamped `.txt` and `.json` report files

The Streamlit website now reads `market_agent/reports/ml_forecast_rankings_latest.txt`, so your existing **Stock Market** publish branch can upload that file to GitHub the same way it uploads `summary_*.txt`.

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
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --no-optimize --json-only
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

Use the same GitHub upload/update logic already in your **Stock Market** branch, but with `path` set to:

```text
market_agent/reports/ml_forecast_rankings_latest.txt
```

If the **Stock Market** node is a GitHub `PUT /contents` HTTP request, send `contentBase64` as the GitHub `content` value.

## Option B: script sends directly to Telegram

Set these environment variables in n8n:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Then run:

```bash
cd /Users/amirexir/Documents/GitHub/portfolio && .venv/bin/python market_agent/daily_ml_forecast_report.py --no-optimize --send-telegram
```

## Useful arguments

```bash
--symbols "AAPL,MSFT,NVDA,AVGO,SPY,VOO,GLD,SLV,USO"
--horizon 30
--history-days 365
--primary-model Ensemble
--top-n 5
--no-market-context
--no-optimize
--json-only
```

The report uses the same Yahoo Finance data path and the same Ridge, XGBoost, and Ensemble forecast comparison used in the Streamlit app.
The scheduled examples use `--no-optimize` because the full optimized XGBoost search can take several minutes across a larger watchlist.
