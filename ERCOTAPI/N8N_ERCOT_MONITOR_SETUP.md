# n8n ERCOT Monitor Setup

Use this when you want n8n to poll ERCOT pages and send Telegram alerts for newly posted files or meeting-detail links.

## Flow

1. `Schedule Trigger`
2. `Execute Command`
3. `Code`
4. `IF`
5. `Telegram`

## Execute Command

Command:

```bash
python /absolute/path/to/ERCOTAPI/ercot_link_monitor.py
```

Expected stdout is a JSON object like:

```json
{
  "checked_sources": 10,
  "new_items": 2,
  "has_updates": true,
  "changes": [
    {
      "source": "SSWG",
      "title": "ERCOT SSWG Schedule 2026",
      "url": "https://www.ercot.com/...",
      "summary": "Spreadsheet detected. Sheet names: ...",
      "status": "new"
    }
  ],
  "telegram_text": "<b>ERCOT Monitor</b> ..."
}
```

## Code Node

Use this code to parse the command output:

```javascript
const raw = $json.stdout || $json.data || $json.output || '';
const start = raw.indexOf('{');

if (start === -1) {
  throw new Error('ercot_link_monitor.py did not return JSON.');
}

const payload = JSON.parse(raw.slice(start));

return [
  {
    json: payload,
  },
];
```

## IF Node

Condition:

```javascript
{{ $json.has_updates === true }}
```

## Telegram Node

Message:

```javascript
{{ $json.telegram_text }}
```

Parse Mode:

```txt
HTML
```

Disable link previews only if you do not want ERCOT page previews in Telegram.

## Notes

- State is stored in `ERCOTAPI/.ercot_link_state.json`.
- Delete that file if you want to resend all currently discoverable items.
- PDF summaries require `pypdf` to be installed in the Python environment running the script.
- DOCX, CSV, JSON, TXT, HTML, and basic XLSX inspection work without extra configuration.
