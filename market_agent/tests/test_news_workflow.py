"""Execute the n8n news and delivery JavaScript with deterministic fixtures."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import unittest

from market_agent.news_workflow import (
    NEWS_API_NODE,
    NEWS_EVIDENCE_JS,
    NEWS_EVIDENCE_NODE,
    NEWS_MODEL_NODE,
    NEWS_PROMPT,
    repair_market_news_workflow,
)
from market_agent.n8n_workflow import (
    TELEGRAM_SPLIT_JS,
    _repair_market_delivery,
    repair_market_optimization_workflow,
)
from market_agent.tests.test_n8n_workflow import _workflow_fixture


def run_javascript(code: str, payload: dict, now: str = "2026-11-11T06:00:00Z") -> dict | list:
    """Run production JS locally without provider, model or messaging calls."""
    script = """const fs = require('fs');
const vm = require('node:vm');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = {$json: data.payload, $now: {toMillis: () => Date.parse(data.now)},
  $input: {all: () => [{json: data.payload}]},
  $: () => ({first: () => ({json: {source: 'telegram', reply_chat_id: 'test'}})})};
// n8n Code nodes run in a VM; URL and URLSearchParams are not exposed globals.
const result = vm.runInNewContext('(function () {' + data.code + '\\n})()', context);
process.stdout.write(JSON.stringify(result));"""
    result = subprocess.run(
        ["node", "-e", script], input=json.dumps({"code": code, "payload": payload, "now": now}),
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def delivery_fixture() -> dict:
    """Retain the legacy truncating helper as a regression input, without an export dependency."""
    code = r"""function safeTelegramText(title, body, footer) {
  const text = `${title}\n\n${body}`;
  return text.slice(0, 3900);
}
const text = JSON.parse($json.message.content).telegram_recommendations;
return {json: {text: safeTelegramText('Market Optimization', text, '')}};"""
    names = ("Build Optimization Telegram Payload", "Build Optimization Channel Payload")
    return {"nodes": [{"name": name, "parameters": {"jsCode": code}} for name in names],
            "connections": {name: {"main": [[{"node": "Existing delivery", "type": "main", "index": 0}]]}
                            for name in names}}


def article(title: str, *, url: str = "https://publisher.example/story", published: str = "2026-11-11T05:00:00Z") -> dict:
    """Create a clearly synthetic article for freshness and selection tests."""
    return {"title": title, "url": url, "publishedAt": published,
            "source": {"name": "Fixture Publisher"}, "description": "Reported context."}


def workflow_fixture() -> dict:
    """Model both a legacy digest route and a disconnected LLM route."""
    return {"nodes": [
        {"name": NEWS_API_NODE, "parameters": {"queryParameters": {"parameters": []},
          "headerParameters": {"parameters": [{"name": "X-Api-Key", "value": "fixture-only"}]}}},
        {"name": NEWS_EVIDENCE_NODE, "parameters": {}, "type": "old"},
        {"name": NEWS_MODEL_NODE, "parameters": {}, "credentials": {"openAiApi": {"id": "fixture"}}},
        {"name": "Existing delivery", "parameters": {"unchanged": True}},
    ], "connections": {
        NEWS_API_NODE: {"main": [[{"node": NEWS_EVIDENCE_NODE, "type": "main", "index": 0}]]},
        NEWS_EVIDENCE_NODE: {"main": [[{"node": "Existing delivery", "type": "main", "index": 0}]]},
        NEWS_MODEL_NODE: {"main": [[{"node": "Existing delivery", "type": "main", "index": 0}]]},
    }}


@unittest.skipUnless(shutil.which("node"), "Node.js is required to execute n8n code")
class NewsEvidenceTests(unittest.TestCase):
    def test_freshness_duplicates_bad_dates_and_urls_are_rejected(self) -> None:
        items = [article("Nvidia earnings update", url="https://publisher.example/story?utm_source=feed"),
                 article("Different title of same stock report"),
                 article("Old inflation report", url="https://publisher.example/old", published="2026-11-09T05:00:00Z"),
                 article("Future inflation report", url="https://publisher.example/future", published="2026-11-11T06:01:00Z"),
                 article("Unknown date stocks", url="https://publisher.example/no-date", published=""),
                 article("Ambiguous time stocks", url="https://publisher.example/naive", published="2026-11-11T05:00:00"),
                 article("Bad URL stocks", url="javascript:alert(1)"),
                 article("[Removed]", url="https://publisher.example/removed"),
                 article("Cooking recipes", url="https://publisher.example/recipe")]
        result = run_javascript(NEWS_EVIDENCE_JS, {"status": "ok", "articles": items, "totalResults": 100})["json"]
        self.assertEqual(len(result["articles"]), 1)
        self.assertEqual(result["articles"][0]["url"], "https://publisher.example/story")
        self.assertEqual(result["news_coverage"]["rejected"],
                         {"stale": 1, "future": 1, "invalid": 4, "duplicate": 1, "unrelated": 1})
        self.assertTrue(result["news_coverage"]["sample_limited"])

    def test_material_governance_story_survives_newer_feed_cluster(self) -> None:
        items = [article(f"Stock market update number {i}", url=f"https://feed.example/{i}") for i in range(40)]
        items.append(article("Fixture chip company board launches CEO investigation",
                             url="https://biztoc.com/older-story", published="2026-11-10T01:00:00Z"))
        result = run_javascript(NEWS_EVIDENCE_JS, {"articles": items})["json"]
        governance = [a for a in result["articles"] if "Governance / legal" in a["topics"]]
        self.assertEqual(len(governance), 1)
        self.assertTrue(governance[0]["is_aggregator"])
        self.assertIn("not verified", governance[0]["evidence_status"])
        self.assertLessEqual(sum(a["domain"] == "feed.example" for a in result["articles"]), 4)
        self.assertGreater(result["news_coverage"]["selection_omitted_count"], 0)

    def test_chicago_dates_use_local_day_and_daylight_time(self) -> None:
        winter = run_javascript(NEWS_EVIDENCE_JS, {"articles": [article("Nvidia earnings")]})["json"]
        self.assertEqual(winter["articles"][0]["published_display"], "Nov 10, 2026, 11:00 PM CST")
        summer = run_javascript(NEWS_EVIDENCE_JS,
            {"articles": [article("Nvidia earnings", published="2026-07-15T05:00:00Z")]},
            now="2026-07-15T06:00:00Z")["json"]
        self.assertEqual(summer["articles"][0]["published_display"], "Jul 15, 2026, 12:00 AM CDT")

    def test_provider_failure_does_not_look_like_no_news(self) -> None:
        for payload in ({"status": "error", "articles": []}, {"status": "ok"}):
            with self.assertRaises(subprocess.CalledProcessError):
                run_javascript(NEWS_EVIDENCE_JS, payload)
        result = run_javascript(NEWS_EVIDENCE_JS, {"status": "ok", "articles": []})["json"]
        self.assertEqual(result["news_coverage"]["selected_count"], 0)

    def test_all_delivery_paths_preserve_the_final_sell_row(self) -> None:
        workflow = delivery_fixture()
        _repair_market_delivery(workflow)
        nodes = {node["name"]: node for node in workflow["nodes"]}
        body = "Market Optimization\n\n" + "BUY fixture details\n" * 500 + "SELL FINAL_FIXTURE -3.2%"
        publication = {"message": {"content": json.dumps({"telegram_recommendations": body})}}
        for name in ("Build Optimization Telegram Payload", "Build Optimization Channel Payload"):
            payload = run_javascript(nodes[name]["parameters"]["jsCode"], publication)["json"]
            self.assertEqual(payload["text"], body)
            parts = run_javascript(TELEGRAM_SPLIT_JS, payload)
            recovered = "".join(part["json"]["text"].split("\n\n", 1)[1] for part in parts)
            self.assertEqual(recovered, body)
            self.assertTrue(all(len(part["json"]["text"]) < 3900 for part in parts))
            self.assertIn("SELL FINAL_FIXTURE", parts[-1]["json"]["text"])


class NewsWorkflowTests(unittest.TestCase):
    def test_bound_news_credential_is_preserved_without_environment_dependency(self) -> None:
        fixture = workflow_fixture()
        api = fixture["nodes"][0]
        api["parameters"].pop("headerParameters")
        api["parameters"].update(authentication="genericCredentialType", genericAuthType="httpHeaderAuth")
        api["credentials"] = {"httpHeaderAuth": {"id": "configured-fixture"}}
        repaired, _ = repair_market_news_workflow(fixture)
        fixed_api = repaired["nodes"][0]
        self.assertEqual(fixed_api["credentials"], api["credentials"])
        self.assertEqual(fixed_api["parameters"]["authentication"], "genericCredentialType")
        self.assertNotIn("NEWS_API_KEY", json.dumps(fixed_api))

    def test_combined_repair_connects_and_authenticates_existing_news_publisher(self) -> None:
        fixture = _workflow_fixture()
        news = workflow_fixture()
        fixture["nodes"].extend(news["nodes"])
        fixture["connections"].update(news["connections"])
        fixture["nodes"].extend([
            {"name": "Build News GitHub Payload", "parameters": {
                "jsCode": "return {json: {branch: 'generated-output'}};"}},
            {"name": "Save News To GitHub", "parameters": {
                "jsonBody": "={{ {branch: $json.branch || 'generated-output'} }}"}},
        ])
        repaired, _ = repair_market_optimization_workflow(fixture)
        nodes = {node["name"]: node for node in repaired["nodes"]}
        self.assertEqual(nodes["Save News To GitHub"]["credentials"],
                         nodes["Save Optimization To GitHub"]["credentials"])
        self.assertIn("branch: 'main'", nodes["Build News GitHub Payload"]["parameters"]["jsCode"])
        model_targets = [d["node"] for d in repaired["connections"][NEWS_MODEL_NODE]["main"][0]]
        self.assertIn("Build News GitHub Payload", model_targets)
        self.assertIn("Existing delivery", model_targets)
        self.assertEqual(repair_market_optimization_workflow(repaired)[1], ())

    def test_repair_is_idempotent_and_preserves_existing_delivery(self) -> None:
        fixture = workflow_fixture()
        original = copy.deepcopy(fixture)
        repaired, changes = repair_market_news_workflow(fixture)
        self.assertTrue(changes)
        self.assertEqual(fixture, original)
        again, changes = repair_market_news_workflow(repaired)
        self.assertEqual(changes, ())
        self.assertEqual(repaired, again)
        self.assertEqual(repaired["nodes"][-1], original["nodes"][-1])
        self.assertEqual(repaired["connections"][NEWS_EVIDENCE_NODE]["main"][0][0]["node"], NEWS_MODEL_NODE)
        self.assertEqual(len(repaired["connections"][NEWS_MODEL_NODE]["main"][0]), 1)
        api = next(n for n in repaired["nodes"] if n["name"] == NEWS_API_NODE)
        params = {p["name"]: p["value"] for p in api["parameters"]["queryParameters"]["parameters"]}
        self.assertEqual(params["pageSize"], "100")
        self.assertEqual(params["sortBy"], "relevancy")
        self.assertIn("36 * 60 * 60", params["from"])
        self.assertIn("NEWS_API_KEY", api["parameters"]["headerParameters"]["parameters"][0]["value"])

    def test_model_contract_separates_materiality_verification_and_actions(self) -> None:
        self.assertIn("Event significance and source verification are separate", NEWS_PROMPT)
        self.assertIn("Attribute allegations explicitly", NEWS_PROMPT)
        self.assertIn("company/board response", NEWS_PROMPT)
        self.assertIn("not independent corroboration", NEWS_PROMPT)
        self.assertIn("cannot promote a failed quantitative model signal", NEWS_PROMPT)
        self.assertIn("published_display (verbatim)", NEWS_PROMPT)
        self.assertIn("direct article URL", NEWS_PROMPT)

    def test_missing_model_fails_instead_of_leaving_partial_news_repair(self) -> None:
        fixture = workflow_fixture()
        fixture["nodes"] = [n for n in fixture["nodes"] if n["name"] != NEWS_MODEL_NODE]
        with self.assertRaises(ValueError):
            repair_market_news_workflow(fixture)


if __name__ == "__main__":
    unittest.main()
