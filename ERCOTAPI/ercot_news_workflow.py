"""ERCOT news workflow for n8n, cron, or manual runs.

This script mirrors the stock-market news pipeline pattern:
- fetch latest ERCOT, data center, NOGRR, and PGRR news files from GitHub
- fall back to local files when needed
- send a digest to Telegram
- store last-sent file names to avoid duplicates
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ercotapi import get_latest_news_by_prefix


NEWS_CATEGORIES = [
    {
        "key": "ercot_news",
        "title": "ERCOT News",
        "prefixes": ["ercot_news_", "ercot_summary_", "summary_ercot_"],
    },
    {
        "key": "data_center_news",
        "title": "Data Center News",
        "prefixes": ["datacenter_news_", "data_center_news_", "dc_news_"],
    },
    {
        "key": "nogrr_updates",
        "title": "NOGRR Updates",
        "prefixes": ["nogrr_", "nodal_operating_guide_", "nodal_guide_change_"],
    },
    {
        "key": "pgrr_updates",
        "title": "PGRR Updates",
        "prefixes": ["pgrr_", "planning_guide_change_", "planning_guide_update_"],
    },
]

STATE_FILE = Path(
    os.getenv("ERCOT_NEWS_STATE_FILE", Path(__file__).with_name(".ercot_news_state.json"))
)
BOT_TOKEN = os.getenv("ERCOT_NEWS_TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("ERCOT_NEWS_TELEGRAM_CHAT_ID", "").strip()
DRY_RUN = os.getenv("ERCOT_NEWS_DRY_RUN", "false").lower() in {"1", "true", "yes"}


def load_state() -> Dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: Dict[str, str]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def chunk_text(text: str, size: int = 3500) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def send_telegram_message(text: str) -> None:
    if DRY_RUN:
        print("[DRY RUN] Telegram message:\n")
        print(text)
        return

    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError(
            "Missing ERCOT_NEWS_TELEGRAM_BOT_TOKEN or ERCOT_NEWS_TELEGRAM_CHAT_ID"
        )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chunk in chunk_text(text):
        response = requests.post(
            url,
            json={
                "chat_id": CHAT_ID,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": False,
            },
            timeout=20,
        )
        response.raise_for_status()


def format_digest(changes: List[Dict[str, str]]) -> str:
    if not changes:
        return "<b>ERCOT News Digest</b>\n\nNo new ERCOT, data center, NOGRR, or PGRR updates were found."

    lines = ["<b>ERCOT News Digest</b>", ""]
    for item in changes:
        lines.append(f"<b>{item['title']}</b>")
        lines.append(f"File: <code>{item['name']}</code>")
        lines.append(item["content"])
        lines.append("")
    return "\n".join(lines).strip()


def collect_latest_news() -> List[Dict[str, str]]:
    state = load_state()
    changes: List[Dict[str, str]] = []

    for category in NEWS_CATEGORIES:
        latest = get_latest_news_by_prefix(category["prefixes"], repo_path="ERCOTAPI")
        if not latest:
            continue

        current_name = latest.get("name", "")
        last_name = state.get(category["key"])
        if current_name and current_name != last_name:
            changes.append(
                {
                    "key": category["key"],
                    "title": category["title"],
                    "name": current_name,
                    "content": latest.get("content", "").strip(),
                }
            )
            state[category["key"]] = current_name

    save_state(state)
    return changes


def main() -> None:
    changes = collect_latest_news()
    digest = format_digest(changes)

    if not changes:
        print("No new ERCOT news updates found.")
        return

    send_telegram_message(digest)
    print(f"Sent {len(changes)} ERCOT news update(s) to Telegram.")


if __name__ == "__main__":
    main()