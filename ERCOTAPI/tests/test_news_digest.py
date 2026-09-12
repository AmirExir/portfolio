"""Exercise the deployed n8n digest JavaScript without network or n8n access."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import unittest
from typing import Any


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "news_digest.js"
NOW = "2026-09-12T18:00:00.000Z"
NODE_HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(input.script, 'utf8');
const fixedNow = Date.parse(input.now);
class FixedDate extends Date {
  constructor(...args) { super(...(args.length ? args : [fixedNow])); }
  static now() { return fixedNow; }
}
const sandbox = {
  Date: FixedDate,
  $input: {all: () => input.payloads.map(json => ({json}))},
  $: (name) => {
    if (name !== 'ERCOT Request Context') throw new Error('Unexpected node');
    return {first: () => ({json: {source: input.source}})};
  },
  $getWorkflowStaticData: (scope) => {
    if (scope !== 'node') throw new Error('Expected persistent node state');
    return input.state;
  },
};
try {
  const output = vm.runInNewContext('(function () {\n' + source + '\n})()', sandbox);
  process.stdout.write(JSON.stringify({output, state: input.state}));
} catch (error) {
  process.stdout.write(JSON.stringify({error: error.message, state: input.state}));
}
"""


def article(
    name: str = "new transmission plan",
    *,
    published_at: str | None = "2026-09-12T17:00:00Z",
    url: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic provider article with explicit publication data."""
    return {
        "title": title or f"ERCOT announces {name}",
        "url": url or f"https://example.com/news/{name.replace(' ', '-')}",
        "publishedAt": published_at,
        "source": {"name": "Example News"},
        "description": "An update on Texas electricity transmission planning.",
    }


@unittest.skipUnless(shutil.which("node"), "Node.js is required to verify n8n JavaScript")
class NewsDigestTests(unittest.TestCase):
    """Check freshness, durable deduplication, and scheduled suppression."""

    def run_digest(
        self,
        articles: list[dict[str, Any]] | None = None,
        *,
        state: dict[str, Any] | None = None,
        source: str = "schedule",
        payloads: list[dict[str, Any]] | None = None,
        now: str = NOW,
    ) -> dict[str, Any]:
        """Execute in a new process, round-tripping the persistent JSON state."""
        result = subprocess.run(
            ["node", "-e", NODE_HARNESS],
            input=json.dumps({
                "script": str(SCRIPT_PATH),
                "now": now,
                "state": state if state is not None else {},
                "source": source,
                "payloads": payloads if payloads is not None else [
                    {"status": "ok", "articles": articles or []}
                ],
            }),
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        return json.loads(result.stdout)

    def initialized_state(self) -> dict[str, Any]:
        """Establish an empty successful deployment baseline."""
        result = self.run_digest([])
        self.assertEqual(result["output"], [])
        return result["state"]

    def test_first_scheduled_run_baselines_existing_feed_then_new_article_posts(self) -> None:
        existing = article("existing project")
        baseline = self.run_digest([existing])
        self.assertEqual(baseline["output"], [])
        repeated = self.run_digest([existing], state=baseline["state"])
        self.assertEqual(repeated["output"], [])
        new = self.run_digest([existing, article("new project")], state=repeated["state"])
        self.assertEqual(new["output"][0]["json"]["article_count"], 1)
        self.assertIn("new project", new["output"][0]["json"]["message"]["content"])
        self.assertNotIn("existing project", new["output"][0]["json"]["message"]["content"])

    def test_repeated_feed_is_suppressed_after_process_restart(self) -> None:
        first = self.run_digest([article()], state=self.initialized_state())
        self.assertEqual(first["output"][0]["json"]["article_count"], 1)
        self.assertTrue(first["output"][0]["json"]["publish_to_channel"])
        second = self.run_digest([article()], state=first["state"])
        self.assertEqual(second["output"], [])

    def test_tracking_url_changes_and_updated_headlines_do_not_repost(self) -> None:
        first = article(url="https://www.example.com/news/project/?id=7&utm_source=first#part")
        changed = article(title="ERCOT publishes a revised headline", url="http://example.com/news/project?fbclid=other&id=7&utm_medium=social")
        baseline = self.run_digest([first])
        result = self.run_digest([changed], state=baseline["state"])
        self.assertEqual(result["output"], [])

    def test_same_headline_on_new_url_is_suppressed_and_alias_remembered(self) -> None:
        first = self.run_digest([article(title="ERCOT: New Transmission Plan!")])
        syndicated = article(title="ercot — new transmission plan", url="https://syndication.example/99")
        second = self.run_digest([syndicated], state=first["state"])
        self.assertEqual(second["output"], [])
        renamed = {**syndicated, "title": "ERCOT changes the project headline"}
        third = self.run_digest([renamed], state=second["state"])
        self.assertEqual(third["output"], [])

    def test_query_article_ids_are_preserved(self) -> None:
        first = self.run_digest([article("first", url="https://example.com/news?id=1")])
        second = self.run_digest([article("second", url="https://example.com/news?id=2")], state=first["state"])
        self.assertEqual(second["output"][0]["json"]["article_count"], 1)

    def test_stale_missing_invalid_and_future_dates_never_post(self) -> None:
        dates = ["2026-09-11T05:59:59Z", None, "unknown", "2026-09-12", "2026-09-12T18:00:01Z"]
        result = self.run_digest(
            [article(str(index), published_at=date) for index, date in enumerate(dates)],
            state=self.initialized_state(),
        )
        self.assertEqual(result["output"], [])

    def test_freshness_boundary_is_inclusive_and_source_date_is_preserved(self) -> None:
        result = self.run_digest([article(published_at="2026-09-11T06:00:00Z")], state=self.initialized_state())
        content = result["output"][0]["json"]["message"]["content"]
        self.assertIn("2026-09-11", content)
        self.assertNotIn("2026-09-12", content)

    def test_one_relevant_article_does_not_pull_in_unrelated_fallback(self) -> None:
        unrelated = article("unrelated", title="California electricity prices fall")
        unrelated["description"] = "California grid news."
        unrelated2 = article("unrelated2", title="Texas baseball team wins")
        unrelated2["description"] = "Sports news."
        result = self.run_digest([unrelated, article(), unrelated2], state=self.initialized_state())
        self.assertEqual(result["output"][0]["json"]["article_count"], 1)
        self.assertNotIn("California", result["output"][0]["json"]["message"]["content"])

    def test_unrelated_only_feed_produces_no_scheduled_placeholder(self) -> None:
        unrelated = article(title="New York electricity news")
        unrelated["description"] = "New York transmission planning."
        result = self.run_digest([unrelated], state=self.initialized_state())
        self.assertEqual(result["output"], [])

    def test_no_backlog_when_more_than_display_limit_arrives(self) -> None:
        batch = [article(f"project {index}") for index in range(12)]
        first = self.run_digest(batch, state=self.initialized_state())
        self.assertEqual(first["output"][0]["json"]["article_count"], 8)
        second = self.run_digest(batch, state=first["state"])
        self.assertEqual(second["output"], [])

    def test_direct_request_repeats_current_news_without_publication_state_changes(self) -> None:
        baseline = self.run_digest([article()])
        direct = self.run_digest([article()], state=baseline["state"], source="telegram")
        self.assertEqual(direct["output"][0]["json"]["article_count"], 1)
        self.assertFalse(direct["output"][0]["json"]["publish_to_channel"])
        self.assertEqual(direct["state"], baseline["state"])
        fresh_direct = self.run_digest([article()], source="telegram")
        self.assertEqual(fresh_direct["state"], {})

    def test_direct_empty_response_is_truthful(self) -> None:
        result = self.run_digest([], source="telegram")
        self.assertEqual(result["output"][0]["json"]["article_count"], 0)
        self.assertIn("No current", result["output"][0]["json"]["message"]["content"])
        self.assertEqual(result["state"], {})

    def test_failed_or_malformed_response_fails_without_baselining(self) -> None:
        for payloads in [[], [{"status": "error", "message": "secret"}], [{}], [{"status": "ok", "articles": None}]]:
            with self.subTest(payloads=payloads):
                result = self.run_digest(payloads=payloads)
                self.assertIn("retrieval failed", result["error"])
                self.assertNotIn("secret", result["error"])
                self.assertEqual(result["state"], {})

    def test_duplicate_items_within_multiple_inputs_are_collapsed(self) -> None:
        result = self.run_digest(
            state=self.initialized_state(),
            payloads=[{"status": "ok", "articles": [article()]}] * 2,
        )
        self.assertEqual(result["output"][0]["json"]["article_count"], 1)

    def test_invalid_persistent_state_fails_closed(self) -> None:
        result = self.run_digest([article()], state={"ercotNewsDigest": {"version": 99}})
        self.assertIn("state is invalid", result["error"])


if __name__ == "__main__":
    unittest.main()
