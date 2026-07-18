# n8n ERCOT monitor setup

Use this flow to poll ERCOT pages, archive authoritative documents, run the
incremental central RAG update, and optionally send Telegram alerts.

## Flow

1. `Schedule Trigger`
2. `Execute Command`
3. `Code`
4. `IF`
5. `Telegram`

## Execute Command

Use the tracked launcher rather than a user-specific Python path:

```bash
ERCOT_REPO_ROOT=/path/to/portfolio /path/to/portfolio/scripts/run_ercot_link_monitor.sh
```

The launcher resolves the checkout, sets `PYTHONPATH`, and defaults to
`ERCOTAPI/.venv/bin/python`. Set `ERCOT_MONITOR_PYTHON` when n8n uses another
environment. On macOS it can read `OPENAI_API_KEY` from the Keychain service
`openai-api-key`; override the service/account with
`ERCOT_OPENAI_KEYCHAIN_SERVICE` and `ERCOT_OPENAI_KEYCHAIN_ACCOUNT`.

After the archive run succeeds, the monitor reconciles the complete
`ERCOTAPI/NEWS/official/` root with the central generation. Unchanged hashes
reuse vectors. Set `ERCOT_RAG_AUTO_INGEST=false` only when another process owns
ingestion.

Expected stdout is one JSON object:

```json
{
  "checked_sources": 18,
  "new_items": 2,
  "has_updates": true,
  "total_changes": 2,
  "reported_changes": 2,
  "omitted_changes": 0,
  "changes": [
    {
      "source": "SSWG",
      "title": "ERCOT SSWG Schedule 2026",
      "url": "https://www.ercot.com/...",
      "summary": "Spreadsheet detected. Sheet names: ...",
      "status": "new"
    }
  ],
  "ingestion": {
    "status": "completed",
    "summary": {
      "changed": true,
      "generation": "<generation-id>"
    }
  },
  "telegram_text": "<b>ERCOT Monitor</b> ..."
}
```

## Code node

```javascript
const raw = $json.stdout || $json.data || $json.output || '';
const start = raw.indexOf('{');

if (start === -1) {
  throw new Error('ERCOT monitor did not return JSON.');
}

return [{ json: JSON.parse(raw.slice(start)) }];
```

## IF and Telegram nodes

IF condition:

```javascript
{{ $json.has_updates === true }}
```

Telegram message:

```javascript
{{ $json.telegram_text }}
```

Use `{{ $json.telegram_chat_id || '@ERCOTNEWS' }}` for the chat ID and `HTML`
parse mode.

## Useful bounds

- `ERCOT_LINK_MAX_ITEMS_PER_SOURCE=5` — successful unseen top-level downloads.
- `ERCOT_LINK_MAX_UNSEEN_ATTEMPTS_PER_SOURCE=20` — attempts allowed to get them.
- `ERCOT_LINK_MAX_KNOWN_RECHECKS_PER_SOURCE=10` — rotating known-URL checks.
- `ERCOT_LINK_MAX_NESTED_ITEMS_PER_SOURCE=20` — detail-page attachments.
- `ERCOT_LINK_REPORT_WINDOW=100` — newest numeric report candidates.
- `ERCOT_LINK_MAX_RESPONSE_BYTES=52428800` — streamed response limit.
- `ERCOT_LINK_MAX_OUTPUT_ITEMS=40` — maximum changes emitted to n8n, with omitted items counted.
- `ERCOT_LINK_MAX_TELEGRAM_CHARS=3900` — safe upper bound for direct Telegram text.
- `ERCOT_RAG_STORE=/persistent/path` — store shared with retrieval processes.

Output limits affect only the JSON/digest returned to n8n. Every successfully
downloaded document is still archived, and the complete durable archive is
still reconciled with the central RAG store.

State is stored in ignored `ERCOTAPI/.ercot_link_state.json`; archived bytes and
provenance sidecars live under ignored `ERCOTAPI/NEWS/official/`. Deleting only
the state file does not reliably replay alerts because archive provenance is an
independent seen ledger. Back up state before repair and use the monitor tests
or a separate temporary state/archive for replay experiments.

Keep Telegram/OpenAI credentials in n8n credentials, environment variables, or
the macOS Keychain—not in workflow JSON or the repository. See
`ERCOTAPI/RAG_INGESTION.md` for store recovery and manual ingestion commands.
