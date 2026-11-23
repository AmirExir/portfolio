"""Exercise the n8n publisher-reader output contract and final model evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest

from market_agent.breaking_news_sources import BREAKING_NEWS_FEEDS
from market_agent.news_workflow import (
    NEWS_API_NODE, NEWS_COMBINE_JS, NEWS_COMBINE_NODE, NEWS_EVIDENCE_JS,
    NEWS_EVIDENCE_NODE, NEWS_FEED_REQUESTS_JS, NEWS_FEED_REQUESTS_NODE,
    NEWS_MODEL_NODE, NEWS_RSS_NODE, repair_market_news_workflow,
)
from market_agent.tests.test_news_workflow import article, run_javascript, workflow_fixture


def combine(api: dict, rss: list[dict]) -> dict:
    """Use the n8n RSS node's actual per-article/per-error paired item shape."""
    script = """const fs = require('node:fs');
const vm = require('node:vm');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const run = (code, context) => vm.runInNewContext('(function(){' + code + '\\n})()', context);
const requests = run(data.requests, {});
const sources = {
  'News API': {first: () => ({json: data.api})},
  'Build Publisher Feed Requests': {all: () => requests},
  'Read Market Publisher Feeds': {all: () => data.rss},
};
const incoming = data.rss.map(({json}, index) => ({json, pairedItem: {item: index}}));
const result = run(data.combine, {$: (name) => sources[name], $input: {all: () => incoming}});
if (result.length !== 1) throw new Error('Expected exactly one news model input');
process.stdout.write(JSON.stringify(result[0].json));"""
    return json.loads(subprocess.run(
        ["node", "-e", script], check=True, capture_output=True, text=True,
        input=json.dumps({"api": api, "rss": rss, "requests": NEWS_FEED_REQUESTS_JS,
                          "combine": NEWS_COMBINE_JS}),
    ).stdout)


def rss_item(publisher: str, **fields: str) -> dict:
    """Construct an RSS output item linked to its configured publisher."""
    publishers = [name for name, _ in BREAKING_NEWS_FEEDS]
    return {"json": fields, "pairedItem": [{"item": publishers.index(publisher)}]}


@unittest.skipUnless(shutil.which("node"), "Standalone n8n JS checks require Node")
class BreakingNewsTests(unittest.TestCase):
    now = "2026-11-23T15:04:14Z"

    def test_current_issuer_decision_reaches_model_despite_delayed_newswire(self) -> None:
        old = article("Shareholders ask fixture company to reinstate former CEO",
                      published="2026-11-22T14:59:00Z")
        decision = rss_item("NVIDIA Newsroom", title="Fixture company reinstates its former CEO",
                            link="https://issuer.example/decision", pubDate="Mon, 23 Nov 2026 13:55:00 GMT",
                            contentSnippet="The board voted to reinstate the chief executive.")
        clutter = [rss_item("MarketWatch", title=f"Stock market update {i}",
                            link=f"https://finance.example/update/{i}", pubDate="Mon, 23 Nov 2026 14:59:00 GMT")
                   for i in range(40)]
        clutter.extend(rss_item("NVIDIA Newsroom", title=f"Annual CEO interview covering topic {i}",
                                link=f"https://issuer.example/interview/{i}",
                                pubDate="Mon, 23 Nov 2026 14:58:00 GMT") for i in range(10))
        # NewsAPI may also have a syndicated copy. Prefer the direct RSS source.
        duplicate = article("Fixture company reinstates its former CEO", url="https://wire.example/duplicate",
                            published=self.now)
        combined = combine({"status": "ok", "articles": [old, duplicate], "totalResults": 3},
                           [*clutter, decision])
        evidence = run_javascript(NEWS_EVIDENCE_JS, combined, now=self.now)["json"]
        source = next(item for item in evidence["articles"] if "reinstates" in item["title"])
        self.assertEqual(source["source"], "NVIDIA Newsroom")
        self.assertEqual(source["url"], "https://issuer.example/decision")
        self.assertEqual(source["published_display"], "Nov 23, 2026, 7:55 AM CST")
        self.assertEqual(source["retrieved_via"], "publisher_rss")
        self.assertEqual(evidence["news_coverage"]["newest_selected_at"], "2026-11-23T14:59:00.000Z")
        self.assertTrue(evidence["news_coverage"]["sample_limited"])
        self.assertEqual(evidence["news_coverage"]["rejected"]["duplicate"], 1)

    def test_ordinary_language_business_headlines_survive_keyword_filter(self) -> None:
        titles = ["Acme closes 50 stores nationwide", "Orion to buy Northstar for $7 billion"]
        reports = [rss_item("BBC Business", title=title, link=f"https://bbc.example/{i}",
                            pubDate="Mon, 23 Nov 2026 14:00:00 GMT") for i, title in enumerate(titles)]
        unrelated = article("Cooking recipes", published=self.now)
        combined = combine({"status": "ok", "articles": [unrelated]}, reports)
        evidence = run_javascript(NEWS_EVIDENCE_JS, combined, now=self.now)["json"]
        self.assertEqual({row["title"] for row in evidence["articles"]}, set(titles))
        self.assertEqual(evidence["news_coverage"]["rejected"]["unrelated"], 1)

    def test_partial_provider_failure_keeps_working_feed_with_attribution(self) -> None:
        inputs = [rss_item("NVIDIA Newsroom", error="unavailable"),
                  rss_item("BBC Business", title="Central bank cuts rates", link="https://finance.example/rates",
                           pubDate="Mon, 23 Nov 2026 08:33:00 -0600")]
        combined = combine({"error": "NewsAPI unavailable"}, inputs)
        evidence = run_javascript(NEWS_EVIDENCE_JS, combined, now=self.now)["json"]
        self.assertEqual([row["source"] for row in evidence["articles"]], ["BBC Business"])
        statuses = {row["source"]: row["status"] for row in evidence["news_coverage"]["sources"]}
        self.assertEqual(statuses, {"NVIDIA Newsroom": "error", "BBC Business": "ok",
                                   "MarketWatch": "empty", "NewsAPI": "error"})
        self.assertIsNone(evidence["news_coverage"]["provider_total_results"])

    def test_full_failure_is_explicit_and_delayed_only_fallback_is_unavailable(self) -> None:
        failed = [rss_item(publisher, error="fixture error") for publisher, _ in BREAKING_NEWS_FEEDS]
        with self.assertRaises(subprocess.CalledProcessError):
            combine({"status": "error", "articles": []}, failed)
        day_old = article("Company forecasts earnings", published="2026-11-22T15:04:14Z")
        combined = combine({"status": "ok", "articles": [day_old]}, failed)
        evidence = run_javascript(NEWS_EVIDENCE_JS, combined, now=self.now)["json"]
        self.assertTrue(evidence["news_coverage"]["delayed_only"])
        self.assertIsNone(evidence["news_coverage"]["newest_selected_at"])
        self.assertEqual(evidence["articles"], [])
        # A successful empty RSS response yields a placeholder if all feeds are empty.
        self.assertEqual(combine({"status": "ok", "articles": []}, [{"json": {}}])["articles"], [])

    def test_bad_publication_dates_remain_invalid_and_future_reports_stay_excluded(self) -> None:
        dates = ["Mon, 23 Nov 2026 09:00:00", "unknown", "Mon, 23 Nov 2026 16:00:00 GMT"]
        rows = [rss_item("BBC Business", title=f"Stock market {idx}", link=f"https://finance.example/{idx}",
                         pubDate=value, isoDate=self.now) for idx, value in enumerate(dates)]
        combined = combine({"status": "ok", "articles": []}, rows)
        evidence = run_javascript(NEWS_EVIDENCE_JS, combined, now=self.now)["json"]
        self.assertEqual(evidence["articles"], [])
        self.assertEqual(evidence["news_coverage"]["rejected"]["invalid"], 2)
        self.assertEqual(evidence["news_coverage"]["rejected"]["future"], 1)

    def test_existing_triggers_share_one_collection_and_delivery_path(self) -> None:
        fixture = workflow_fixture()
        target = {"node": NEWS_API_NODE, "index": 0, "type": "main"}
        for trigger in ("Scheduled news", "Telegram news"):
            fixture["connections"][trigger] = {"main": [[target]]}
        repaired, _ = repair_market_news_workflow(fixture)
        path = (NEWS_API_NODE, NEWS_FEED_REQUESTS_NODE, NEWS_RSS_NODE, NEWS_COMBINE_NODE,
                NEWS_EVIDENCE_NODE, NEWS_MODEL_NODE, "Existing delivery")
        for source, destination in zip(path, path[1:]):
            self.assertEqual(repaired["connections"][source]["main"],
                             [[{"node": destination, "index": 0, "type": "main"}]])
        for trigger in ("Scheduled news", "Telegram news"):
            self.assertEqual(repaired["connections"][trigger], fixture["connections"][trigger])
        nodes = {node["name"]: node for node in repaired["nodes"]}
        self.assertGreaterEqual(nodes[NEWS_RSS_NODE]["typeVersion"], 1.1)
        for source in (NEWS_API_NODE, NEWS_RSS_NODE):
            self.assertTrue(nodes[source]["alwaysOutputData"])
            self.assertEqual(nodes[source]["onError"], "continueRegularOutput")
        self.assertEqual(repair_market_news_workflow(repaired), (repaired, ()))
