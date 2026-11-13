#!/usr/bin/env python3
"""Register discoverable market commands without sending a chat message."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from market_agent.chat_commands import TELEGRAM_BOT_COMMANDS  # noqa: E402

PRIVATE_SCOPE = {"type": "all_private_chats"}
LEGACY_MENU_COMMANDS = {"optimize", "optimization", "market_optimization", "marketoptimization"}


def merge_market_commands(existing: list[dict[str, str]]) -> list[dict[str, str]]:
    """Show the new commands first while preserving unrelated existing entries."""
    commands = [dict(command) for command in TELEGRAM_BOT_COMMANDS]
    names = {entry["command"] for entry in commands} | LEGACY_MENU_COMMANDS
    commands.extend(dict(entry) for entry in existing if entry["command"] not in names)
    if len(commands) > 100:
        raise ValueError("Merged Telegram menu exceeds 100 commands; no update was sent")
    return commands


def telegram_api(token: str, method: str, payload: dict[str, Any]) -> Any:
    """Call the Telegram metadata API without exposing its token-bearing URL."""
    if method not in {"getMyCommands", "setMyCommands"}:
        raise ValueError("Only command-menu metadata methods are supported")
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/{method}", json=payload, timeout=20,
        )
    except requests.RequestException:
        raise RuntimeError(f"Telegram {method} request failed") from None
    if response.status_code != 200:
        raise RuntimeError(f"Telegram {method} returned HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        raise RuntimeError(f"Telegram {method} returned invalid JSON") from None
    if not isinstance(body, dict) or body.get("ok") is not True:
        raise RuntimeError(f"Telegram {method} did not confirm success")
    return body.get("result")


def configure_commands(token: str, backup: Path) -> dict[str, Any]:
    """Back up and update only the private-chat command menu, then read it back."""
    if not token.strip():
        raise ValueError("Set TELEGRAM_BOT_TOKEN to configure the menu")
    default = telegram_api(token, "getMyCommands", {})
    private = telegram_api(token, "getMyCommands", {"scope": PRIVATE_SCOPE})
    if not isinstance(default, list) or not isinstance(private, list):
        raise RuntimeError("Telegram returned an invalid command list")
    existing = {entry["command"]: entry for entry in default}
    existing.update({entry["command"]: entry for entry in private})
    desired = merge_market_commands(list(existing.values()))
    if desired == private:
        return {"changed": False, "commands": desired}
    backup.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"scope": PRIVATE_SCOPE, "private_commands": private,
                   "default_commands": default}, handle, indent=2)
    confirmed = telegram_api(token, "setMyCommands", {"scope": PRIVATE_SCOPE, "commands": desired})
    if confirmed is not True:
        raise RuntimeError("Telegram did not confirm the command-menu update")
    actual = telegram_api(token, "getMyCommands", {"scope": PRIVATE_SCOPE})
    if actual != desired:
        raise RuntimeError("Telegram command menu readback did not match the requested update")
    return {"changed": True, "commands": actual, "backup": str(backup)}


def main() -> None:
    """Configure the menu using an environment credential and exclusive backup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(configure_commands(os.environ.get("TELEGRAM_BOT_TOKEN", ""), args.backup)))


if __name__ == "__main__":
    main()
