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

DEFAULT_TELEGRAM_CHAT_ID = "@ERCOTNEWS"
STATE_FILE = Path(
    os.getenv("ERCOT_NEWS_STATE_FILE", Path(__file__).with_name(".ercot_news_state.json"))
)
DRY_RUN = os.getenv("ERCOT_NEWS_DRY_RUN", "false").lower() in {"1", "true", "yes"}
FORCE_SEND = os.getenv("ERCOT_NEWS_FORCE_SEND", "false").lower() in {"1", "true", "yes"}
SEND_NO_UPDATES = os.getenv("ERCOT_NEWS_SEND_NO_UPDATES", "false").lower() in {"1", "true", "yes"}


def first_env(names: List[str], default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


BOT_TOKEN = first_env(
    [
        "ERCOT_NEWS_TELEGRAM_BOT_TOKEN",
        "ERCOT_LINK_TELEGRAM_BOT_TOKEN",
        "TELEGRAM_BOT_TOKEN",
    ]
)
CHAT_ID = first_env(
    [
        "ERCOT_NEWS_TELEGRAM_CHAT_ID",
        "ERCOT_LINK_TELEGRAM_CHAT_ID",
        "TELEGRAM_CHAT_ID",
    ],
    DEFAULT_TELEGRAM_CHAT_ID,
)
GENERATED_OUTPUT_REF = first_env(
    ["ERCOT_NEWS_GENERATED_REF", "GENERATED_OUTPUT_REF", "GITHUB_GENERATED_REF"],
    "generated-output",
)


def github_content_refs() -> List[str]:
    refs: List[str] = []
    for raw_ref in str(GENERATED_OUTPUT_REF or "").replace(";", ",").split(","):
        ref = raw_ref.strip()
        if ref and ref not in refs:
            refs.append(ref)
    for fallback_ref in ("generated-output", "main"):
        if fallback_ref not in refs:
            refs.append(fallback_ref)
    return refs


def fetch_news_file_index(path: str = "ERCOTAPI", branch: str = "main"):
    contents_url = (
        "https://api.github.com/repos/AmirExir/portfolio/contents/"
        f"{path.strip('/')}?ref={branch}"
    )
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ERCOT-News-Workflow",
    }
    response = requests.get(contents_url, headers=headers, timeout=12)
    if response.status_code == 403 and "rate limit" in response.text.lower():
        return None
    response.raise_for_status()
    files = response.json()
    return files if isinstance(files, list) else []


def fetch_news_repo_tree(branch: str = "main"):
    tree_url = f"https://api.github.com/repos/AmirExir/portfolio/git/trees/{branch}?recursive=1"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ERCOT-News-Workflow",
    }
    response = requests.get(tree_url, headers=headers, timeout=12)
    if response.status_code == 403 and "rate limit" in response.text.lower():
        return []
    response.raise_for_status()
    payload = response.json()
    tree = payload.get("tree", []) if isinstance(payload, dict) else []
    return tree if isinstance(tree, list) else []


def normalize_news_text(raw_text: str) -> str:
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return str(payload.get("content") or payload.get("message") or payload)
        if isinstance(payload, list):
            return "\n".join(str(item) for item in payload)
        return str(payload)
    except json.JSONDecodeError:
        return raw_text


def read_latest_local_news(local_dir: Path, prefixes: List[str]) -> Optional[Dict[str, str]]:
    if not local_dir.is_dir():
        return None

    candidates = []
    for root, _, files in os.walk(local_dir):
        for name in files:
            lower_name = name.lower()
            if not lower_name.endswith((".txt", ".md", ".json")):
                continue
            if any(lower_name.startswith(prefix.lower()) for prefix in prefixes):
                candidates.append(Path(root) / name)

    if not candidates:
        return None

    latest_path = sorted(candidates, key=lambda path: path.name, reverse=True)[0]
    try:
        content = normalize_news_text(latest_path.read_text(encoding="utf-8").strip())
        return {"name": latest_path.name, "content": content}
    except Exception:
        return None


def get_latest_news_by_prefix(prefixes: List[str], repo_path: str = "ERCOTAPI") -> Optional[Dict[str, str]]:
    candidate_paths = [repo_path, f"{repo_path}/market_agent"]

    for branch in github_content_refs():
        for candidate_path in candidate_paths:
            try:
                files = fetch_news_file_index(candidate_path, branch)
            except Exception:
                files = None

            if not files:
                continue

            matches = [
                file_info
                for file_info in files
                if file_info.get("type") == "file"
                and file_info.get("name", "").lower().endswith((".txt", ".md", ".json"))
                and any(file_info.get("name", "").lower().startswith(prefix.lower()) for prefix in prefixes)
            ]
            if not matches:
                continue

            latest = sorted(matches, key=lambda item: item.get("name", ""), reverse=True)[0]
            download_url = latest.get("download_url")
            if not download_url:
                continue

            try:
                response = requests.get(download_url, timeout=12)
                response.raise_for_status()
                return {
                    "name": latest.get("name", ""),
                    "content": normalize_news_text(response.text.strip()),
                }
            except Exception:
                pass

    for branch in github_content_refs():
        try:
            tree_items = fetch_news_repo_tree(branch)
        except Exception:
            tree_items = []

        tree_matches = []
        for item in tree_items:
            if item.get("type") != "blob":
                continue
            rel_path = item.get("path", "")
            base_name = os.path.basename(rel_path).lower()
            if not base_name.endswith((".txt", ".md", ".json")):
                continue
            if any(base_name.startswith(prefix.lower()) for prefix in prefixes):
                tree_matches.append(rel_path)

        if tree_matches:
            latest_path = sorted(tree_matches, reverse=True)[0]
            raw_url = f"https://raw.githubusercontent.com/AmirExir/portfolio/{branch}/{latest_path}"
            try:
                response = requests.get(raw_url, timeout=12)
                response.raise_for_status()
                return {
                    "name": os.path.basename(latest_path),
                    "content": normalize_news_text(response.text.strip()),
                }
            except Exception:
                pass

    return read_latest_local_news(Path(__file__).resolve().parent, prefixes)


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
            "Missing ERCOT_NEWS_TELEGRAM_BOT_TOKEN or ERCOT_LINK_TELEGRAM_BOT_TOKEN"
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


def collect_latest_news(force_send: bool = False) -> List[Dict[str, str]]:
    state = load_state()
    changes: List[Dict[str, str]] = []

    for category in NEWS_CATEGORIES:
        latest = get_latest_news_by_prefix(
            category["prefixes"],
            repo_path="ERCOTAPI",
        )
        if not latest:
            continue

        current_name = latest.get("name", "")
        last_name = state.get(category["key"])
        if current_name and (force_send or current_name != last_name):
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
    changes = collect_latest_news(force_send=FORCE_SEND)
    digest = format_digest(changes)

    if not changes and not SEND_NO_UPDATES:
        print("No new ERCOT news updates found.")
        return

    send_telegram_message(digest)
    print(f"Sent {len(changes)} ERCOT news update(s) to Telegram chat {CHAT_ID}.")


if __name__ == "__main__":
    main()
