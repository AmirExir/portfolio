from __future__ import annotations

import unittest
import json
import shutil
import subprocess
from datetime import datetime, timezone

from ERCOTAPI.news_pipeline import (
    ERCOT_MODEL_NODE,
    ERCOT_FILE_PARSER_JS,
    ERCOT_FILE_PARSER_NODE,
    ERCOT_NEWS_DIGEST_NODE,
    ERCOT_PAYLOAD_NODE,
    ERCOT_PUBLISH_NODE,
    ERCOT_PUBLICATION_GATE_NODE,
    ERCOT_PUBLICATION_GATE_JS,
    ERCOT_SUMMARY_URL,
    assess_news_brief,
    news_brief_timestamp,
    news_item_sort_key,
    repair_ercot_publication_workflow,
)


def _workflow_fixture() -> dict:
    return {
        "nodes": [
            {"name": ERCOT_MODEL_NODE, "parameters": {}},
            {"name": ERCOT_NEWS_DIGEST_NODE, "parameters": {}},
            {
                "name": ERCOT_PAYLOAD_NODE,
                "parameters": {
                    "jsCode": (
                        "return {json: {filename: "
                        "'ercot_news_summary_' + timestamp + '.txt', "
                        "branch: 'generated-output'}};"
                    )
                },
            },
            {
                "name": ERCOT_PUBLISH_NODE,
                "parameters": {
                    "url": (
                        "=https://api.github.com/repos/AmirExir/portfolio/"
                        "contents/ERCOTAPI/{{$json.filename}}"
                    ),
                    "bodyParameters": {
                        "parameters": [
                            {"name": "message", "value": "={{$json.message}}"},
                            {"name": "branch", "value": "generated-output"},
                        ]
                    },
                },
            },
        ],
        "connections": {
            ERCOT_NEWS_DIGEST_NODE: {
                "main": [
                    [
                        {
                            "node": "Build ERCOT Channel Payload",
                            "type": "main",
                            "index": 0,
                        }
                    ]
                ]
            },
            ERCOT_MODEL_NODE: {
                "main": [
                    [
                        {
                            "node": "Build ERCOT Channel Payload",
                            "type": "main",
                            "index": 0,
                        }
                    ]
                ]
            },
            ERCOT_PAYLOAD_NODE: {"main": [[]]},
        },
    }


class NewsPipelineTests(unittest.TestCase):
    def test_news_timestamp_and_sorting_work_across_prefixes(self) -> None:
        older = "summary_2026-07-30T20-00-00-000Z.txt"
        newer = "ercot_news_summary_2026-07-31T05-30-00-125Z.txt"

        self.assertEqual(
            news_brief_timestamp(newer),
            datetime(
                2026, 7, 31, 5, 30, 0, 125000, tzinfo=timezone.utc
            ),
        )
        self.assertEqual(max((older, newer), key=news_item_sort_key), newer)

    def test_old_summary_is_not_labeled_live(self) -> None:
        state = assess_news_brief(
            "ercot_news_summary_2026-07-18T11-00-48-795Z.txt",
            now=datetime(2026, 7, 31, 5, 30, tzinfo=timezone.utc),
        )

        self.assertFalse(state.is_fresh)
        self.assertEqual(state.label, "No recent brief")
        self.assertEqual(state.status, "warn")
        self.assertIsNotNone(state.age_hours)
        assert state.age_hours is not None
        self.assertGreater(state.age_hours, 300)

    def test_recent_summary_is_labeled_live(self) -> None:
        state = assess_news_brief(
            "ercot_news_summary_2026-07-30T11-01-49-089Z.txt",
            now=datetime(2026, 7, 31, 5, 30, tzinfo=timezone.utc),
        )

        self.assertTrue(state.is_fresh)
        self.assertEqual(state.label, "Recent brief")
        self.assertEqual(state.status, "ok")

    def test_unknown_summary_timestamp_fails_closed(self) -> None:
        state = assess_news_brief("ercot_news_summary_latest.txt")

        self.assertFalse(state.is_fresh)
        self.assertEqual(state.label, "Update age unknown")

    def test_repair_restores_branch_path_and_publisher_edge(self) -> None:
        workflow = _workflow_fixture()

        repaired, changes = repair_ercot_publication_workflow(workflow)

        self.assertTrue(changes)
        self.assertTrue(
            workflow["nodes"][2]["parameters"]["jsCode"].endswith(
                "branch: 'generated-output'}};"
            )
        )
        self.assertIn(
            "branch: 'main'", repaired["nodes"][2]["parameters"]["jsCode"]
        )
        publisher = repaired["nodes"][3]["parameters"]
        self.assertEqual(publisher["url"], ERCOT_SUMMARY_URL)
        self.assertEqual(
            next(
                item["value"]
                for item in publisher["bodyParameters"]["parameters"]
                if item["name"] == "branch"
            ),
            "main",
        )
        model_targets = repaired["connections"][ERCOT_MODEL_NODE]["main"][0]
        self.assertTrue(
            any(item["node"] == ERCOT_PUBLICATION_GATE_NODE for item in model_targets)
        )
        digest_targets = repaired["connections"][ERCOT_NEWS_DIGEST_NODE]["main"][0]
        self.assertTrue(
            any(item["node"] == ERCOT_PUBLICATION_GATE_NODE for item in digest_targets)
        )
        gate_targets = repaired["connections"][ERCOT_PUBLICATION_GATE_NODE]["main"][0]
        self.assertTrue(
            any(
                item["node"] == "Build ERCOT Channel Payload"
                for item in gate_targets
            )
        )
        self.assertTrue(
            any(item["node"] == ERCOT_PAYLOAD_NODE for item in gate_targets)
        )
        for targets in (model_targets, digest_targets):
            self.assertFalse(any(
                item["node"] in {ERCOT_PAYLOAD_NODE, "Build ERCOT Channel Payload"}
                for item in targets
            ))
        payload_targets = repaired["connections"][ERCOT_PAYLOAD_NODE]["main"][0]
        self.assertTrue(
            any(item["node"] == ERCOT_PUBLISH_NODE for item in payload_targets)
        )

    def test_repair_is_idempotent(self) -> None:
        repaired, first_changes = repair_ercot_publication_workflow(
            _workflow_fixture()
        )
        second_repair, second_changes = repair_ercot_publication_workflow(
            repaired
        )

        self.assertTrue(first_changes)
        self.assertEqual(second_changes, ())
        self.assertEqual(second_repair, repaired)

    def test_repair_preserves_direct_replies_and_other_workflow_state(self) -> None:
        workflow = _workflow_fixture()
        direct_edge = {"node": "Direct Reply", "type": "main", "index": 0}
        workflow["connections"][ERCOT_NEWS_DIGEST_NODE]["main"][0].append(direct_edge)
        workflow["staticData"] = {"market": {"unchanged": True}}
        workflow["nodes"].append({"name": "Market Model", "parameters": {"keep": 1}})
        repaired, _ = repair_ercot_publication_workflow(workflow)
        self.assertIn(direct_edge, repaired["connections"][ERCOT_NEWS_DIGEST_NODE]["main"][0])
        self.assertEqual(repaired["staticData"], workflow["staticData"])
        self.assertIn(workflow["nodes"][-1], repaired["nodes"])

    def test_repair_corrects_parser_mode_and_preserves_extra_gate_options(self) -> None:
        workflow = _workflow_fixture()
        workflow["nodes"].append({"name": ERCOT_FILE_PARSER_NODE, "parameters": {
            "jsCode": ERCOT_FILE_PARSER_JS, "mode": "runOnceForAllItems"
        }})
        repaired, _ = repair_ercot_publication_workflow(workflow)
        parser = next(node for node in repaired["nodes"] if node["name"] == ERCOT_FILE_PARSER_NODE)
        self.assertEqual(parser["parameters"]["mode"], "runOnceForEachItem")
        gate = next(node for node in repaired["nodes"] if node["name"] == ERCOT_PUBLICATION_GATE_NODE)
        gate["parameters"]["extra_option"] = "preserved"
        repeated, changes = repair_ercot_publication_workflow(repaired)
        self.assertEqual(changes, ())
        self.assertEqual(repeated, repaired)


@unittest.skipUnless(shutil.which("node"), "Node.js is required to test n8n code")
class PublicationCodeTests(unittest.TestCase):
    def run_code(self, code: str, payload: dict) -> dict:
        """Evaluate n8n code with offline inputs and no publication services."""

        runner = """const vm = require('node:vm');
const data = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
try {
  const result = vm.runInNewContext('(function(){' + data.code + '\\n})()', {
    $json: data.payload, $input: { all: () => data.payload.items || [] },
    console: { warn: () => {} }
  });
  process.stdout.write(JSON.stringify({ result }));
} catch (error) { process.stdout.write(JSON.stringify({ error: error.message })); }
"""
        result = subprocess.run(
            ["node", "-e", runner], input=json.dumps({"code": code, "payload": payload}),
            text=True, capture_output=True, check=True,
        )
        return json.loads(result.stdout)

    def test_only_new_nonempty_scheduled_content_reaches_publishers(self) -> None:
        fresh = {"json": {"message": {"content": "- New filing (ERCOT source)"}}}
        result = self.run_code(ERCOT_PUBLICATION_GATE_JS, {"items": [
            {"json": {}},
            {"json": {"message": {"content": "No ERCOT news available."}}},
            {"json": {"content": "Requested recap", "publish_to_channel": False}},
            fresh,
        ]})
        self.assertEqual(result, {"result": [fresh]})

    def test_monitor_errors_are_excluded_from_filing_evidence(self) -> None:
        change = {"status": "updated", "title": "NPRR update"}
        result = self.run_code(ERCOT_FILE_PARSER_JS, {"stdout": json.dumps({
            "has_updates": False, "changes": [change, {"status": "error"}]
        })})
        self.assertEqual(result["result"]["json"]["changes"], [change])
        self.assertTrue(result["result"]["json"]["has_updates"])

    def test_failed_collection_is_not_reported_as_no_news(self) -> None:
        for payload in (
            {"stdout": "not JSON"},
            {"stdout": "{}"},
            {"stdout": '{"changes": [{"status": "error"}]}'},
            {"exitCode": 1, "stdout": '{"changes": []}'},
        ):
            with self.subTest(payload=payload):
                self.assertIn("error", self.run_code(ERCOT_FILE_PARSER_JS, payload))


if __name__ == "__main__":
    unittest.main()
