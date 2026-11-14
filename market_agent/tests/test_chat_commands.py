"""Regression tests for safe market command routing and visible Telegram menus."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import unittest

from market_agent.chat_commands import (
    CLASSIFIER_NODE,
    CONTEXT_NODE,
    DIRECT_SENDER_NODE,
    HELP_BUILDER_NODE,
    HELP_CONDITION_NODE,
    MARKET_COMMAND_RESOLVER_JS,
    MARKET_HELP_JS,
    MARKET_HELP_TEXT,
    MARKET_REPORT_TITLE,
    MARKET_REPLY_KEYBOARD,
    PRIVATE_REPLY_MARKUP_EXPRESSION,
    RESOLVER_NODE,
    TELEGRAM_BOT_COMMANDS,
    repair_market_chat_workflow,
)


def run_js(code: str, payload: dict, *, chat_type: str | None = "private") -> dict | str:
    """Execute a Code-node snippet in VM isolation, without network or messages."""
    script = """const vm = require('node:vm');
const data = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const context = {$json: data.payload, $: () => {
  if (data.chat_type === null) throw new Error('Trigger did not execute');
  return {first: () => ({json: {message: {chat: {type: data.chat_type}}}})};
}};
const result = vm.runInNewContext('(function(){' + data.code + '\\n})()', context);
process.stdout.write(JSON.stringify(result));"""
    result = subprocess.run(
        ["node", "-e", script],
        input=json.dumps({"code": code, "payload": payload, "chat_type": chat_type}),
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def node(name: str, code: str = "") -> dict:
    """Create a minimal serializable workflow-node fixture."""
    return {"name": name, "type": "n8n-nodes-base.code", "typeVersion": 2,
            "position": [0, 0], "parameters": {"mode": "runOnceForEachItem", "jsCode": code}}


def edge(name: str) -> dict:
    return {"node": name, "type": "main", "index": 0}


def workflow_fixture() -> dict:
    """Provide the established routing seams and request fields without private data."""
    context_code = r"""const text = ($json.text || 'scheduled market update').toString();
const newsRequested = /\b(news|briefing)\b/i.test(text);
const optimizationWords = /\b(optimization|forecast|ridge)\b/i;
const newsOnly = newsRequested && !optimizationWords.test(text);
function shellQuote(value) { return JSON.stringify(value); }
const args = ['--request-text', shellQuote(text)];
return {json: {request_text: text, run_news: newsRequested,
  run_optimization: !newsOnly, args}};"""
    acknowledgement = (
        "const text = 'Got it. I am running market news and optimization now.';\n"
        "const other = 'Got it. I am running the market optimization now.';\n"
        "const completion = 'I will send the recommendation summary and add the buy/sell event to Calendar.';"
    )
    return {"nodes": [
        node("Normalize Telegram Text"),
        node(CLASSIFIER_NODE, "return {json: {...$json, intent: 'existing-classifier'}};"),
        node(CONTEXT_NODE, context_code),
        node("Build Market Acknowledgement", acknowledgement),
        node("Build ERCOT Acknowledgement", "const text = 'I will send the digest back here and post it to @ERCOTNEWS when it finishes.';"),
        node("Build Optimization Telegram Payload", "const title = 'Market Optimization';"),
        node("Build Optimization Channel Payload", "const title = 'Market Optimization';"),
        node("Build Calendar Event Payload", "const summary = `Market model signals: Buy ${buyLabel} | Sell ${sellLabel}`;"),
        node("Send Market Acknowledgement"),
        node("Send ERCOT Acknowledgement"),
        node(DIRECT_SENDER_NODE),
        node("Send Assistant Telegram"),
        node("Is ERCOT Request"), node("Is Market Request"),
        node("News API"), node("Run Optimization Scripts"), node("Create Market Calendar Event"),
    ], "connections": {
        "Normalize Telegram Text": {"main": [[edge(CLASSIFIER_NODE)]]},
        CLASSIFIER_NODE: {"main": [[edge("Is ERCOT Request")]]},
        "Is ERCOT Request": {"main": [[edge("Existing ERCOT path")], [edge("Is Market Request")]]},
        "Is Market Request": {"main": [[edge(CONTEXT_NODE)], [edge("Existing assistant path")]]},
        CONTEXT_NODE: {"main": [[edge("News API"), edge("Run Optimization Scripts")]]},
        "Run Optimization Scripts": {"main": [[edge("Create Market Calendar Event")]]},
    }}


@unittest.skipUnless(shutil.which("node"), "Node.js is required to test n8n Code nodes")
class MarketCommandExecutionTests(unittest.TestCase):
    def resolve(self, text: str, *, source: str = "telegram") -> dict:
        return run_js(MARKET_COMMAND_RESOLVER_JS, {
            "text": text, "source": source, "reply_chat_id": "123",
        })["json"]

    def test_new_commands_preserve_modifiers_and_normalize_to_existing_semantics(self) -> None:
        cases = {
            "Run market analysis for NVDA, MSFT next week using Ridge without RL":
                "Run market optimization for NVDA, MSFT next week using Ridge without RL",
            "Run full market briefing for BTC-USD tomorrow":
                "Run market news and optimization for BTC-USD tomorrow",
            "Market news": "Market news",
            "/analysis@market_bot NVDA MSFT next week": "Run market optimization NVDA MSFT next week",
            "/briefing": "Run market news and optimization",
            "/news@market_bot": "Market news",
            "Can you run market analysis for NVDA long term": "Run market optimization for NVDA long term",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                resolved = self.resolve(original)
                self.assertEqual(resolved["text"], expected)
                self.assertEqual(resolved["market_original_text"], original)
                self.assertFalse(resolved["market_help_requested"])
                self.assertFalse(resolved["ercot_load_requested"])

    def test_full_briefing_runs_both_sections_and_preserves_original_request(self) -> None:
        repaired, _ = repair_market_chat_workflow(workflow_fixture())
        context = next(n for n in repaired["nodes"] if n["name"] == CONTEXT_NODE)
        for original, flags in (
            ("Run full market briefing", (True, True)),
            ("Run market analysis", (False, True)),
            ("Market news", (True, False)),
            ("/briefing@market_bot NVDA next week", (True, True)),
        ):
            resolved = self.resolve(original)
            result = run_js(context["parameters"]["jsCode"], resolved)["json"]
            self.assertEqual((result["run_news"], result["run_optimization"]), flags)
            self.assertEqual(result["request_text"], original)
            self.assertEqual(json.loads(result["args"][1]), original)

    def test_help_aliases_disable_work_and_produce_only_help_payload(self) -> None:
        for text in ("/start", "/start@market_bot welcome", "/help", "/menu", "/market",
                     "Market help", "help", "show market options", "Show me the market commands!",
                     "what can I ask?", "what can I ask you?", "what can you do?", "How do I use this bot?"):
            with self.subTest(text=text):
                result = self.resolve(text)
                self.assertTrue(result["market_help_requested"])
                self.assertFalse(result["run_news"])
                self.assertFalse(result["run_optimization"])
                self.assertFalse(result["ercot_load_requested"])
                self.assertIsNone(result["optimization_command"])
                help_payload = run_js(MARKET_HELP_JS, result)["json"]
                self.assertEqual(help_payload["text"], MARKET_HELP_TEXT)
                self.assertEqual(help_payload["chat_id"], "123")
                self.assertLess(len(help_payload["text"]), 3900)

    def test_help_builder_rejects_an_unrelated_input_or_missing_chat(self) -> None:
        for payload in ({"source": "telegram", "market_help_requested": False, "reply_chat_id": "123"},
                        {"source": "telegram", "market_help_requested": True},
                        {"source": "schedule", "market_help_requested": True, "reply_chat_id": "123"}):
            with self.assertRaises(subprocess.CalledProcessError):
                run_js(MARKET_HELP_JS, payload)

    def test_existing_requests_and_non_telegram_runs_are_not_reinterpreted(self) -> None:
        for text in ("Run market optimization for NVDA next week", "Run ERCOT news",
                     "Explain ERCOT planning guide section 4", "What can I ask about ERCOT?",
                     "Read my latest email", "Use XGBoost for BTC-USD tomorrow"):
            result = self.resolve(text)
            self.assertEqual(result["text"], text)
            self.assertFalse(result["market_help_requested"])
            self.assertFalse(result["ercot_load_requested"])
        scheduled = self.resolve("/help", source="schedule")
        self.assertFalse(scheduled["market_help_requested"])
        self.assertFalse(scheduled["ercot_load_requested"])
        self.assertEqual(scheduled["text"], "/help")

    def test_ercot_news_aliases_keep_the_existing_news_route(self) -> None:
        for original in ("ERCOT news", "Run ERCOT news", "/ercot_news", "/ercot_news@grid_bot"):
            with self.subTest(original=original):
                result = self.resolve(original)
                self.assertEqual(result["text"], "Run ERCOT news")
                self.assertEqual(result["market_original_text"], original)
                self.assertEqual(result["market_chat_command"], "ercot_news")
                self.assertFalse(result["market_help_requested"])
                self.assertFalse(result["ercot_load_requested"])

    def test_explicit_ercot_load_requests_set_the_load_route_flag(self) -> None:
        for original in ("ERCOT load forecast", "Run ERCOT load forecast", "ERCOT demand forecast",
                         "Show me the ERCOT load forecast", "Can you please show ERCOT demand forecast?",
                         "What is the ERCOT load forecast for tomorrow?", "Load forecast for ERCOT",
                         "Forecast ERCOT demand", "/ercot_load", "/ercot_load@grid_bot"):
            with self.subTest(original=original):
                result = self.resolve(original)
                self.assertEqual(result["text"], "ERCOT load forecast")
                self.assertEqual(result["market_original_text"], original)
                self.assertEqual(result["market_chat_command"], "ercot_load")
                self.assertTrue(result["ercot_load_requested"])
                self.assertFalse(result["market_help_requested"])
        slash_with_modifier = self.resolve("/ercot_load tomorrow")
        self.assertEqual(slash_with_modifier["text"], "ERCOT load forecast tomorrow")
        self.assertTrue(slash_with_modifier["ercot_load_requested"])

    def test_load_routing_does_not_hijack_engineering_questions_or_keep_old_flags(self) -> None:
        for text in ("Explain how ERCOT load forecasting works", "What does ERCOT require for a load forecast?",
                     "Is an ERCOT demand forecast an operating limit?", "ERCOT load forecast methodology",
                     "Don't run ERCOT load forecast", "Load forecast", "Run market analysis"):
            with self.subTest(text=text):
                result = run_js(MARKET_COMMAND_RESOLVER_JS, {
                    "text": text, "source": "telegram", "reply_chat_id": "123",
                    "ercot_load_requested": True,
                })["json"]
                self.assertFalse(result["ercot_load_requested"])
        scheduled = self.resolve("ERCOT load forecast", source="schedule")
        self.assertFalse(scheduled["ercot_load_requested"])

    def test_menu_is_shown_only_in_private_chats(self) -> None:
        expression = PRIVATE_REPLY_MARKUP_EXPRESSION[3:-2].strip()
        for chat_type, expected in (("private", "replyKeyboard"), ("group", "none"),
                                    ("supergroup", "none"), ("channel", "none"), (None, "none")):
            self.assertEqual(run_js(f"return ({expression});", {}, chat_type=chat_type), expected)


class MarketChatWorkflowTests(unittest.TestCase):
    def test_repair_is_idempotent_and_preserves_original_classifier_and_routes(self) -> None:
        original = workflow_fixture()
        snapshot = copy.deepcopy(original)
        repaired, changes = repair_market_chat_workflow(original)
        self.assertTrue(changes)
        self.assertEqual(original, snapshot)
        again, changes = repair_market_chat_workflow(repaired)
        self.assertEqual(again, repaired)
        self.assertEqual(changes, ())
        originals = {n["name"]: n for n in original["nodes"]}
        nodes = {n["name"]: n for n in repaired["nodes"]}
        self.assertEqual(nodes[CLASSIFIER_NODE], originals[CLASSIFIER_NODE])
        for name in (CLASSIFIER_NODE, "Is ERCOT Request", "Is Market Request", CONTEXT_NODE):
            self.assertEqual(repaired["connections"][name], original["connections"][name])
        self.assertEqual(repaired["connections"]["Normalize Telegram Text"]["main"], [[edge(RESOLVER_NODE)]])

    def test_help_true_branch_cannot_reach_research_or_publication(self) -> None:
        repaired, _ = repair_market_chat_workflow(workflow_fixture())
        connections = repaired["connections"]
        self.assertEqual(connections[RESOLVER_NODE]["main"], [[edge(HELP_CONDITION_NODE)]])
        self.assertEqual(connections[HELP_CONDITION_NODE]["main"],
                         [[edge(HELP_BUILDER_NODE)], [edge(CLASSIFIER_NODE)]])
        pending = [HELP_BUILDER_NODE]
        reachable = set()
        while pending:
            name = pending.pop()
            if name in reachable:
                continue
            reachable.add(name)
            pending.extend(d["node"] for output in connections.get(name, {}).get("main", []) for d in output)
        self.assertEqual(reachable, {HELP_BUILDER_NODE, DIRECT_SENDER_NODE})

    def test_buttons_and_user_facing_titles_are_consistent(self) -> None:
        repaired, _ = repair_market_chat_workflow(workflow_fixture())
        nodes = {n["name"]: n for n in repaired["nodes"]}
        for name in ("Send Market Acknowledgement", "Send ERCOT Acknowledgement", DIRECT_SENDER_NODE, "Send Assistant Telegram"):
            parameters = nodes[name]["parameters"]
            self.assertEqual(parameters["replyKeyboard"], MARKET_REPLY_KEYBOARD)
            self.assertEqual(parameters["replyMarkup"], PRIVATE_REPLY_MARKUP_EXPRESSION)
            self.assertFalse(parameters["replyKeyboardOptions"]["one_time_keyboard"])
        labels = [button["text"] for row in MARKET_REPLY_KEYBOARD["rows"] for button in row["row"]["buttons"]]
        self.assertEqual(labels, ["Run market analysis", "Run full market briefing", "Market news", "Market help",
                                  "ERCOT news", "ERCOT load forecast"])
        for name in ("Build Market Acknowledgement", "Build Optimization Telegram Payload",
                     "Build Optimization Channel Payload", "Build Calendar Event Payload"):
            code = nodes[name]["parameters"]["jsCode"]
            self.assertIn(MARKET_REPORT_TITLE, code)
            self.assertNotIn("Market Optimization", code)
            self.assertNotIn("running market news and optimization", code)
        self.assertEqual([c["command"] for c in TELEGRAM_BOT_COMMANDS],
                         ["analysis", "briefing", "news", "ercot_news", "ercot_load", "market", "help"])
        self.assertIn("@ERCOTNEWS", MARKET_HELP_TEXT)
        self.assertIn("existing twice-daily ERCOT news schedule", MARKET_HELP_TEXT)
        self.assertIn("6:00 AM and 3:00 PM Central", MARKET_HELP_TEXT)
        ercot_ack = nodes["Build ERCOT Acknowledgement"]["parameters"]["jsCode"]
        self.assertIn("I will send the digest back here. Scheduled updates post new items to @ERCOTNEWS.", ercot_ack)
        self.assertNotIn("post it to @ERCOTNEWS when it finishes", ercot_ack)

    def test_optimization_only_workflow_is_unchanged_and_partial_chat_fails(self) -> None:
        fixture = {"nodes": [node(CONTEXT_NODE), node(DIRECT_SENDER_NODE)], "connections": {}}
        self.assertEqual(repair_market_chat_workflow(fixture), (fixture, ()))
        incomplete = workflow_fixture()
        incomplete["nodes"] = [n for n in incomplete["nodes"] if n["name"] != DIRECT_SENDER_NODE]
        with self.assertRaises(ValueError):
            repair_market_chat_workflow(incomplete)


if __name__ == "__main__":
    unittest.main()
