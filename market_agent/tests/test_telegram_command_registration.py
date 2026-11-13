"""Command registration must preserve other features and never send messages."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from scripts import configure_market_telegram_commands as menu


class TelegramCommandRegistrationTests(unittest.TestCase):
    def test_merge_preserves_unrelated_commands_and_hides_legacy_names(self) -> None:
        existing = [{"command": "ercot", "description": "Grid news"},
                    {"command": "optimization", "description": "Old name"},
                    {"command": "analysis", "description": "Old description"}]
        merged = menu.merge_market_commands(existing)
        self.assertIn(existing[0], merged)
        self.assertNotIn(existing[1], merged)
        self.assertEqual(sum(command["command"] == "analysis" for command in merged), 1)

    def test_registration_preserves_default_menu_and_reads_back_private_scope(self) -> None:
        defaults = [{"command": "ercot", "description": "Grid news"}]
        desired = menu.merge_market_commands(defaults)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            menu, "telegram_api", side_effect=[defaults, [], True, desired],
        ) as api:
            backup = Path(directory) / "before.json"
            result = menu.configure_commands("fixture-token", backup)
            self.assertTrue(result["changed"])
            self.assertEqual(json.loads(backup.read_text())["default_commands"], defaults)
            calls = api.call_args_list
            self.assertEqual([call.args[1] for call in calls],
                             ["getMyCommands", "getMyCommands", "setMyCommands", "getMyCommands"])
            self.assertEqual(calls[2].args[2]["scope"], {"type": "all_private_chats"})

    def test_existing_backup_prevents_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            menu, "telegram_api", side_effect=[[], []],
        ) as api:
            backup = Path(directory) / "before.json"
            backup.write_text("preserve")
            with self.assertRaises(FileExistsError):
                menu.configure_commands("fixture-token", backup)
            self.assertEqual(api.call_count, 2)
            self.assertEqual(backup.read_text(), "preserve")

    def test_nonempty_private_menu_keeps_unrelated_default_entries(self) -> None:
        defaults = [{"command": "ercot", "description": "Grid news"}]
        private = [{"command": "calendar", "description": "Calendar help"}]
        desired = menu.merge_market_commands(defaults + private)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            menu, "telegram_api", side_effect=[defaults, private, True, desired],
        ):
            result = menu.configure_commands("fixture-token", Path(directory) / "before.json")
            self.assertIn(defaults[0], result["commands"])
            self.assertIn(private[0], result["commands"])

    def test_idempotent_registration_does_not_write(self) -> None:
        desired = menu.merge_market_commands([])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            menu, "telegram_api", side_effect=[[], desired],
        ) as api:
            backup = Path(directory) / "unused.json"
            self.assertFalse(menu.configure_commands("fixture-token", backup)["changed"])
            self.assertFalse(backup.exists())
            self.assertEqual(api.call_count, 2)

    def test_api_error_does_not_leak_token_or_issue_a_message(self) -> None:
        with patch.object(menu.requests, "post", side_effect=requests.ConnectionError("sensitive URL")):
            with self.assertRaises(RuntimeError) as failure:
                menu.telegram_api("fixture-secret", "getMyCommands", {})
            self.assertNotIn("fixture-secret", str(failure.exception))
            self.assertNotIn("sensitive", str(failure.exception))
        with patch.object(menu.requests, "post") as post:
            with self.assertRaises(ValueError):
                menu.telegram_api("fixture-secret", "sendMessage", {})
            post.assert_not_called()

    def test_failed_readback_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            menu, "telegram_api", side_effect=[[], [], True, []],
        ):
            with self.assertRaisesRegex(RuntimeError, "readback"):
                menu.configure_commands("fixture-token", Path(directory) / "before.json")


if __name__ == "__main__":
    unittest.main()
