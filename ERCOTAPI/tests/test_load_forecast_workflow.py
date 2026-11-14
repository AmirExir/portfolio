"""Offline checks for ERCOT load routing, independent schedules and delivery."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import unittest

from ERCOTAPI.load_forecast_workflow import (
    LOAD_CHANNEL, LOAD_CONDITION, LOAD_CONTEXT, LOAD_DIRECT, LOAD_DIRECT_JS,
    LOAD_REPORT, LOAD_REPORT_JS, LOAD_RUNNER, LOAD_SCHEDULE, LOAD_SENDER,
    LOAD_SPLITTER, repair_ercot_load_forecast_workflow,
)
from market_agent.n8n_workflow import TELEGRAM_SPLIT_JS


def edge(name: str) -> dict:
    return {"node": name, "type": "main", "index": 0}


def fixture() -> dict:
    """Provide only public integration seams, with fake credential bindings."""
    names = ["ERCOT Schedule Trigger", "Is ERCOT Request", "Is ERCOT Telegram QA",
             "Telegram ERCOT News", "Send Direct Market Telegram", "Split Direct Telegram Message"]
    nodes = {name: {"name": name, "type": "n8n-nodes-base.code", "typeVersion": 2,
                    "parameters": {}} for name in names}
    nodes["ERCOT Schedule Trigger"].update(type="n8n-nodes-base.scheduleTrigger", typeVersion=1.2,
        parameters={"rule": {"interval": [{"triggerAtHour": 7}, {"triggerAtHour": 16}]}})
    nodes["Telegram ERCOT News"].update(type="n8n-nodes-base.telegram", typeVersion=1.2,
        credentials={"telegramApi": {"id": "fixture-ercot-bot"}})
    nodes["Send Direct Market Telegram"]["parameters"] = {
        "chatId": "={{ $json.chat_id }}", "text": "={{ String($json.text).replace(/&/g, '&amp;').replace(/</g, '&lt;') }}",
        "additionalFields": {"parse_mode": "HTML"}, "replyMarkup": "replyKeyboard",
        "replyKeyboard": {"rows": []}, "replyKeyboardOptions": {"resize_keyboard": True},
    }
    nodes["Split Direct Telegram Message"]["parameters"]["jsCode"] = TELEGRAM_SPLIT_JS
    return {"settings": {"timezone": "America/New_York"}, "nodes": list(nodes.values()), "connections": {
        "ERCOT Schedule Trigger": {"main": [[edge("ERCOT Request Context")]]},
        "Is ERCOT Request": {"main": [[edge("Is ERCOT Telegram QA")], [edge("Is Market Request")]]},
        "Is ERCOT Telegram QA": {"main": [[edge("ERCOT Retriever")], [edge("ERCOT Request Context")]]},
        "Split Direct Telegram Message": {"main": [[edge("Send Direct Market Telegram")]]},
    }}


def execute_js(code: str, payload: dict, context: dict | None = None) -> dict | list:
    """Run n8n-style JavaScript without browser globals, network or messaging."""
    script = r"""const vm = require('node:vm');
const data = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const env = {$json: data.payload, $: () => ({first: () => ({json: data.context})}),
  $input: {first: () => ({json: data.payload}), all: () => [{json: data.payload}]}};
process.stdout.write(JSON.stringify(vm.runInNewContext('(function(){' + data.code + '\n})()', env)));"""
    result = subprocess.run(["node", "-e", script], text=True,
                            input=json.dumps({"code": code, "payload": payload, "context": context}),
                            capture_output=True, check=True)
    return json.loads(result.stdout)


class ForecastWorkflowTests(unittest.TestCase):
    def test_idempotent_repair_preserves_news_schedule_settings_and_credential_sources(self) -> None:
        original = fixture()
        snapshot = copy.deepcopy(original)
        repaired, changes = repair_ercot_load_forecast_workflow(original)
        self.assertTrue(changes)
        self.assertEqual(original, snapshot)
        self.assertEqual(repair_ercot_load_forecast_workflow(repaired), (repaired, ()))
        by_name = {node["name"]: node for node in repaired["nodes"]}
        self.assertEqual(repaired["settings"], original["settings"])
        self.assertEqual(by_name[LOAD_SCHEDULE]["parameters"], by_name["ERCOT Schedule Trigger"]["parameters"])
        self.assertEqual(by_name[LOAD_SENDER]["credentials"], by_name["Telegram ERCOT News"]["credentials"])
        self.assertEqual(by_name[LOAD_SENDER]["parameters"]["replyMarkup"], "none")
        self.assertNotIn("replyKeyboard", by_name[LOAD_SENDER]["parameters"])
        self.assertEqual(by_name[LOAD_SENDER]["parameters"]["additionalFields"]["parse_mode"], "HTML")
        for name in ("ERCOT Schedule Trigger", "Is ERCOT Telegram QA"):
            self.assertEqual(repaired["connections"][name], original["connections"][name])

    def test_load_path_cannot_run_news_retrieval_optimization_or_calendar(self) -> None:
        repaired, _ = repair_ercot_load_forecast_workflow(fixture())
        connections = repaired["connections"]
        self.assertEqual(connections["Is ERCOT Request"]["main"][0], [edge(LOAD_CONDITION)])
        self.assertEqual(connections[LOAD_CONDITION]["main"],
                         [[edge(LOAD_CONTEXT)], [edge("Is ERCOT Telegram QA")]])
        pending, visited = [LOAD_SCHEDULE], set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(d["node"] for output in connections.get(current, {}).get("main", []) for d in output)
        self.assertEqual(visited, {LOAD_SCHEDULE, LOAD_CONTEXT, LOAD_RUNNER, LOAD_REPORT,
                                  LOAD_DIRECT, LOAD_CHANNEL, LOAD_SPLITTER, LOAD_SENDER,
                                  "Split Direct Telegram Message", "Send Direct Market Telegram"})

    def test_partial_and_changed_routes_fail_without_overwriting(self) -> None:
        minimal = {"nodes": [], "connections": {}}
        self.assertEqual(repair_ercot_load_forecast_workflow(minimal), (minimal, ()))
        changed = fixture()
        changed["connections"]["Is ERCOT Request"]["main"][0].append(edge("Custom route"))
        with self.assertRaisesRegex(ValueError, "Unexpected ERCOT request route"):
            repair_ercot_load_forecast_workflow(changed)
        changed = fixture()
        next(n for n in changed["nodes"] if n["name"] == "Telegram ERCOT News").pop("credentials")
        with self.assertRaisesRegex(ValueError, "credential binding"):
            repair_ercot_load_forecast_workflow(changed)


@unittest.skipUnless(shutil.which("node"), "Node.js is required to test n8n snippets")
class ForecastDeliveryTests(unittest.TestCase):
    def test_valid_report_and_source_failure_preserve_text_exactly(self) -> None:
        context = {"source": "telegram", "reply_chat_id": "123", "channel_chat_id": "@test"}
        for ok in (True, False):
            report = {"ok": ok, "text": "Forecast: 50,001 MW\nSource published Nov 13, 2026 at 6 AM CST"}
            actual = execute_js(LOAD_REPORT_JS, {"stdout": json.dumps(report), "exitCode": 0}, context)["json"]
            self.assertEqual(actual, {**context, "text": report["text"], "report_ok": ok})

    def test_execution_errors_are_visible_without_echoing_raw_output(self) -> None:
        for payload in ({"error": "private process details"}, {"exitCode": 1, "stdout": "private"},
                        {"stdout": "not json"}, {"stdout": json.dumps({"ok": True, "text": ""})},
                        {"stdout": "x" * 100001}):
            actual = execute_js(LOAD_REPORT_JS, payload, {"source": "schedule"})["json"]
            self.assertFalse(actual["report_ok"])
            self.assertIn("unavailable", actual["text"])
            self.assertNotIn("private", actual["text"])

    def test_direct_reply_only_for_requester_and_no_duplicate_channel_reply(self) -> None:
        base = {"text": "Load forecast", "report_ok": True, "channel_chat_id": "@test"}
        for source, chat, count in (("schedule", "", 0), ("telegram", "123", 1),
                                    ("telegram", "@test", 0), ("telegram", "", 0)):
            actual = execute_js(LOAD_DIRECT_JS, {**base, "source": source, "reply_chat_id": chat})
            self.assertEqual(len(actual), count)
            if count:
                self.assertEqual(actual[0]["json"]["chat_id"], chat)
                self.assertTrue(actual[0]["json"]["send_direct"])

    def test_long_channel_report_keeps_final_day_and_unicode(self) -> None:
        text = "Forecast ⚡\n" * 600 + "Final day: 50,001 MW"
        result = execute_js(TELEGRAM_SPLIT_JS, {"text": text, "chat_id": "@test"})
        self.assertGreater(len(result), 1)
        bodies = [item["json"]["text"].split("\n\n", 1)[1] for item in result]
        self.assertEqual("".join(bodies), text)
        self.assertTrue(all(len(item["json"]["text"].encode("utf-16-le")) // 2 < 4096 for item in result))


if __name__ == "__main__":
    unittest.main()
