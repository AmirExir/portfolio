"""Offline checks for ERCOT load routing, independent schedules and delivery."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import unittest

from ERCOTAPI.load_forecast_workflow import (
    LOAD_CHANNEL, LOAD_CHANNEL_JS, LOAD_CONDITION, LOAD_CONTEXT, LOAD_DIRECT, LOAD_DIRECT_JS,
    LOAD_REPORT, LOAD_REPORT_JS, LOAD_RUNNER, LOAD_SCHEDULE, LOAD_SENDER,
    LOAD_SPLITTER, LOAD_CHANNEL_CHART, LOAD_DIRECT_CHART, LOAD_CHANNEL_PHOTO,
    LOAD_DIRECT_PHOTO, LOAD_CHANNEL_RESTORE, LOAD_DIRECT_RESTORE,
    repair_ercot_load_forecast_workflow, restore_text_js,
)
from market_agent.n8n_workflow import TELEGRAM_SPLIT_JS

PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aVZkAAAAASUVORK5CYII="


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
    nodes["Send Direct Market Telegram"].update(type="n8n-nodes-base.telegram", typeVersion=1.2,
        credentials={"telegramApi": {"id": "fixture-direct-bot"}})
    nodes["Split Direct Telegram Message"]["parameters"]["jsCode"] = TELEGRAM_SPLIT_JS
    return {"settings": {"timezone": "America/New_York"}, "nodes": list(nodes.values()), "connections": {
        "ERCOT Schedule Trigger": {"main": [[edge("ERCOT Request Context")]]},
        "Is ERCOT Request": {"main": [[edge("Is ERCOT Telegram QA")], [edge("Is Market Request")]]},
        "Is ERCOT Telegram QA": {"main": [[edge("ERCOT Retriever")], [edge("ERCOT Request Context")]]},
        "Split Direct Telegram Message": {"main": [[edge("Send Direct Market Telegram")]]},
    }}


def execute_js(code: str, payload: dict, context: dict | None = None,
               binary: dict | None = None) -> dict | list:
    """Run n8n-style JavaScript without browser globals, network or messaging."""
    script = r"""const vm = require('node:vm');
const data = JSON.parse(require('fs').readFileSync(0, 'utf8'));
// n8n's $binary exposes metadata only: inline image bytes are deliberately omitted.
const metadata = Object.fromEntries(Object.entries(data.binary || {}).map(([name, value]) =>
  [name, Object.fromEntries(Object.entries(value).filter(([key]) => key !== 'data'))]));
const input = {json: data.payload, binary: data.binary};
const env = {Buffer, $binary: metadata, $json: data.payload, $: () => ({first: () => ({json: data.context})}),
  $input: {item: input, first: () => input, all: () => [input]}};
process.stdout.write(JSON.stringify(vm.runInNewContext('(function(){' + data.code + '\n})()', env)));"""
    result = subprocess.run(["node", "-e", script], text=True,
                            input=json.dumps({"code": code, "payload": payload, "context": context, "binary": binary}),
                            capture_output=True, check=True)
    return json.loads(result.stdout)


def legacy_public_forecast_fixture() -> dict:
    """Model the installed forecast publisher alongside news and filing routes."""
    workflow = fixture()
    news_routes = {
        "ERCOT Request Context": ["ERCOT News API", "Run ERCOT Link Monitor"],
        "ERCOT News API": ["Build ERCOT News Digest"],
        "Build ERCOT News Digest": ["Build ERCOT Channel Payload"],
        "Run ERCOT Link Monitor": ["Parse ERCOT File Updates"],
        "Parse ERCOT File Updates": ["Message Model For ERCOT"],
        "Message Model For ERCOT": ["Build ERCOT Channel Payload"],
        "Build ERCOT Channel Payload": ["Telegram ERCOT News"],
    }
    for name, destinations in news_routes.items():
        workflow["nodes"].append({"name": name, "parameters": {"fixture": name}})
        workflow["connections"][name] = {"main": [[edge(target) for target in destinations]]}
    workflow, _ = repair_ercot_load_forecast_workflow(workflow)
    nodes = {node["name"]: node for node in workflow["nodes"]}
    nodes[LOAD_CONTEXT]["parameters"]["jsCode"] = """const source = $json.source === 'telegram' ? 'telegram' : 'schedule';
return {json: {source, reply_chat_id: String($json.reply_chat_id || ''), channel_chat_id: '@ERCOTNEWS'}};"""
    nodes[LOAD_CHANNEL]["parameters"] = {
        "mode": "runOnceForEachItem",
        "jsCode": "return {json: {chat_id: $json.channel_chat_id, text: $json.text}};",
    }
    return workflow


class ForecastWorkflowTests(unittest.TestCase):
    def test_legacy_delivery_repair_preserves_news_and_revision_routes(self) -> None:
        original = legacy_public_forecast_fixture()
        repaired, _ = repair_ercot_load_forecast_workflow(original)
        original_nodes = {node["name"]: node for node in original["nodes"]}
        repaired_nodes = {node["name"]: node for node in repaired["nodes"]}
        news_nodes = (
            "ERCOT Schedule Trigger", "ERCOT Request Context", "ERCOT News API",
            "Build ERCOT News Digest",
            "Run ERCOT Link Monitor", "Parse ERCOT File Updates",
            "Message Model For ERCOT",
            "Build ERCOT Channel Payload", "Telegram ERCOT News",
        )
        for name in news_nodes:
            with self.subTest(node=name):
                self.assertEqual(repaired_nodes[name], original_nodes[name])
                self.assertEqual(
                    repaired["connections"].get(name),
                    original["connections"].get(name),
                )
        self.assertEqual(repaired["settings"], original["settings"])

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
        for name, source in ((LOAD_CHANNEL_PHOTO, "Telegram ERCOT News"),
                             (LOAD_DIRECT_PHOTO, "Send Direct Market Telegram")):
            photo = by_name[name]
            self.assertEqual(photo["credentials"], by_name[source]["credentials"])
            self.assertEqual(photo["parameters"]["operation"], "sendPhoto")
            self.assertTrue(photo["parameters"]["binaryData"])
            self.assertEqual(photo["parameters"]["binaryPropertyName"], "data")
            self.assertLess(len(photo["parameters"]["additionalFields"]["caption"]), 1024)
            self.assertEqual(photo["onError"], "continueRegularOutput")
        self.assertTrue(by_name[LOAD_RUNNER]["parameters"]["command"].endswith(" --chart --compare"))

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
                                  LOAD_CHANNEL_CHART, LOAD_DIRECT_CHART, LOAD_CHANNEL_PHOTO,
                                  LOAD_DIRECT_PHOTO, LOAD_CHANNEL_RESTORE, LOAD_DIRECT_RESTORE,
                                  "Split Direct Telegram Message", "Send Direct Market Telegram"})
        self.assertEqual(connections[LOAD_CHANNEL_PHOTO]["main"], [[edge(LOAD_CHANNEL_RESTORE)]])
        self.assertEqual(connections[LOAD_CHANNEL_RESTORE]["main"], [[edge(LOAD_SPLITTER)]])
        self.assertEqual(connections[LOAD_DIRECT_CHART]["main"][1], [edge("Split Direct Telegram Message")])

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

    def test_owner_configuration_is_explicit_and_rejects_public_destinations(self) -> None:
        for owner in ("@channel", "-123456", "0", "bad", "123; command"):
            with self.subTest(owner=owner), self.assertRaisesRegex(ValueError, "private Telegram"):
                repair_ercot_load_forecast_workflow(fixture(), owner_chat_id=owner)
        with self.assertRaisesRegex(ValueError, "boolean"):
            repair_ercot_load_forecast_workflow(fixture(), publish_scheduled="true")
        configured, _ = repair_ercot_load_forecast_workflow(fixture(), owner_chat_id="123")
        self.assertEqual(repair_ercot_load_forecast_workflow(configured, owner_chat_id="123"), (configured, ()))


@unittest.skipUnless(shutil.which("node"), "Node.js is required to test n8n snippets")
class ForecastDeliveryTests(unittest.TestCase):
    def test_repair_stops_legacy_public_forecasts_and_retains_private_requests(self) -> None:
        legacy = legacy_public_forecast_fixture()
        legacy_nodes = {node["name"]: node for node in legacy["nodes"]}
        old_context = execute_js(legacy_nodes[LOAD_CONTEXT]["parameters"]["jsCode"], {})["json"]
        old_delivery = execute_js(
            legacy_nodes[LOAD_CHANNEL]["parameters"]["jsCode"],
            {**old_context, "text": "Official ERCOT + XGBoost"},
        )
        self.assertEqual(old_delivery["json"]["chat_id"], "@ERCOTNEWS")

        workflow, _ = repair_ercot_load_forecast_workflow(legacy)
        nodes = {node["name"]: node for node in workflow["nodes"]}
        context_code = nodes[LOAD_CONTEXT]["parameters"]["jsCode"]
        channel_parameters = nodes[LOAD_CHANNEL]["parameters"]
        self.assertEqual(channel_parameters["mode"], "runOnceForAllItems")
        for source in ("schedule", "telegram"):
            with self.subTest(source=source):
                context = execute_js(
                    context_code, {"source": source, "reply_chat_id": "123"}
                )["json"]
                self.assertIs(context.get("publish_scheduled"), False)
                report = {**context, "text": "Official ERCOT + XGBoost", "report_ok": True}
                self.assertEqual(execute_js(channel_parameters["jsCode"], report), [])
                if source == "telegram":
                    direct = execute_js(nodes[LOAD_DIRECT]["parameters"]["jsCode"], report)
                    self.assertEqual(direct[0]["json"]["chat_id"], "123")
                    self.assertEqual(direct[0]["json"]["text"], report["text"])

    def test_valid_report_and_source_failure_preserve_text_exactly(self) -> None:
        context = {"source": "telegram", "reply_chat_id": "123", "channel_chat_id": "@test"}
        for ok in (True, False):
            report = {"ok": ok, "text": "Forecast: 50,001 MW\nSource published Nov 13, 2026 at 6 AM CST"}
            if ok:
                report["chart"] = {"base64": PNG_BASE64, "mime_type": "image/png", "file_name": "chart.png"}
            actual = execute_js(LOAD_REPORT_JS, {"stdout": json.dumps(report), "exitCode": 0}, context)["json"]
            self.assertEqual(actual, {**context, "text": report["text"], "report_ok": ok, "has_chart": ok,
                                      "chart_caption": "ERCOT hourly load forecast · MW · Central time"})

    def test_execution_errors_are_visible_without_echoing_raw_output(self) -> None:
        for payload in ({"error": "private process details"}, {"exitCode": 1, "stdout": "private"},
                        {"stdout": "not json"}, {"stdout": json.dumps({"ok": True, "text": ""})},
                        {"stdout": "x" * 750001}):
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

    def test_scheduled_reports_go_only_to_configured_owner_by_default(self) -> None:
        configured, _ = repair_ercot_load_forecast_workflow(fixture(), owner_chat_id="123")
        node = next(node for node in configured["nodes"] if node["name"] == LOAD_CONTEXT)
        context = execute_js(node["parameters"]["jsCode"], {})["json"]
        self.assertEqual(context["source"], "schedule")
        self.assertFalse(context["publish_scheduled"])
        direct = execute_js(LOAD_DIRECT_JS, {**context, "text": "Forecast"})
        self.assertEqual(direct[0]["json"]["chat_id"], "123")
        self.assertEqual(execute_js(LOAD_CHANNEL_JS, context), [])
        self.assertEqual(execute_js(LOAD_DIRECT_JS, {**context, "owner_chat_id": ""}), [])

    def test_commands_never_broadcast_even_with_public_schedule_enabled(self) -> None:
        base = {"channel_chat_id": "@test", "publish_scheduled": True, "text": "Forecast"}
        for source in ("telegram", "invalid", None):
            self.assertEqual(execute_js(LOAD_CHANNEL_JS, {**base, "source": source}), [])
        self.assertEqual(execute_js(LOAD_CHANNEL_JS, {**base, "source": "schedule"})[0]["json"]["chat_id"], "@test")

    def test_long_channel_report_keeps_final_day_and_unicode(self) -> None:
        text = "Forecast ⚡\n" * 600 + "Final day: 50,001 MW"
        result = execute_js(TELEGRAM_SPLIT_JS, {"text": text, "chat_id": "@test"})
        self.assertGreater(len(result), 1)
        bodies = [item["json"]["text"].split("\n\n", 1)[1] for item in result]
        self.assertEqual("".join(bodies), text)
        self.assertTrue(all(len(item["json"]["text"].encode("utf-16-le")) // 2 < 4096 for item in result))

    def test_png_bytes_survive_report_and_both_destination_builders(self) -> None:
        caption = "Official ERCOT + XGBoost\n24 matching hourly intervals · MW"
        report = {"ok": True, "text": "Forecast", "chart_caption": caption,
                  "chart": {"base64": PNG_BASE64, "mime_type": "image/png"}}
        context = {"source": "telegram", "reply_chat_id": "123", "channel_chat_id": "@test"}
        parsed = execute_js(LOAD_REPORT_JS, {"stdout": json.dumps(report)}, context)
        self.assertEqual(parsed["binary"]["data"]["data"], PNG_BASE64)
        self.assertEqual(parsed["binary"]["data"]["fileName"], "ercot-load-forecast.png")
        direct = execute_js(LOAD_DIRECT_JS, parsed["json"], binary=parsed["binary"])[0]
        channel = execute_js(LOAD_CHANNEL_JS, {**parsed["json"], "source": "schedule", "publish_scheduled": True}, binary=parsed["binary"])[0]
        self.assertEqual(direct["binary"], parsed["binary"])
        self.assertEqual(channel["binary"], parsed["binary"])
        self.assertEqual(channel["json"]["chat_id"], "@test")
        self.assertEqual(channel["json"]["chart_caption"], caption)
        self.assertEqual(direct["json"]["chart_caption"], caption)

    def test_comparison_failure_has_no_image_or_redundant_missing_chart_notice(self) -> None:
        report = {"ok": True, "text": "Official forecast. Model comparison unavailable.",
                  "chart_error": "comparison_unavailable"}
        result = execute_js(LOAD_REPORT_JS, {"stdout": json.dumps(report)}, {})
        self.assertFalse(result["json"]["has_chart"])
        self.assertNotIn("binary", result)
        self.assertEqual(result["json"]["text"], report["text"])

    def test_missing_invalid_and_oversized_chart_preserve_valid_forecast(self) -> None:
        for chart in (None, {"base64": "not a PNG", "mime_type": "image/png"},
                      {"base64": PNG_BASE64, "mime_type": "text/html"},
                      {"base64": "A" * 600001, "mime_type": "image/png"}):
            actual = execute_js(LOAD_REPORT_JS, {"stdout": json.dumps({"ok": True, "text": "Forecast 50,001 MW", "chart": chart})}, {})
            self.assertTrue(actual["json"]["report_ok"])
            self.assertFalse(actual["json"]["has_chart"])
            self.assertIn("Forecast 50,001 MW", actual["json"]["text"])
            self.assertIn("Chart unavailable", actual["json"]["text"])
            self.assertNotIn("binary", actual)

    def test_photo_delivery_failure_still_restores_text_with_visible_notice(self) -> None:
        original = {"text": "Validated forecast 50,001 MW", "chat_id": "@test"}
        for response, failed in (({"message_id": 10}, False), ({"error": "private response details"}, True)):
            actual = execute_js(restore_text_js(LOAD_CHANNEL), response, original)["json"]
            self.assertEqual(actual["chart_delivery_failed"], failed)
            self.assertTrue(actual["text"].startswith(original["text"]))
            self.assertEqual("could not be delivered" in actual["text"], failed)
            self.assertNotIn("private response details", actual["text"])


if __name__ == "__main__":
    unittest.main()
