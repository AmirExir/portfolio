from __future__ import annotations

import unittest
from datetime import datetime, timezone

from ERCOTAPI.news_pipeline import (
    ERCOT_MODEL_NODE,
    ERCOT_NEWS_DIGEST_NODE,
    ERCOT_PAYLOAD_NODE,
    ERCOT_PUBLISH_NODE,
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
        self.assertEqual(state.label, "Pipeline stale")
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
        self.assertEqual(state.label, "Live pipeline")
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
            any(item["node"] == ERCOT_PAYLOAD_NODE for item in model_targets)
        )
        digest_targets = repaired["connections"][ERCOT_NEWS_DIGEST_NODE]["main"][0]
        self.assertTrue(
            any(item["node"] == ERCOT_PAYLOAD_NODE for item in digest_targets)
        )
        self.assertTrue(
            any(
                item["node"] == "Build ERCOT Channel Payload"
                for item in digest_targets
            )
        )
        self.assertTrue(
            any(
                item["node"] == "Build ERCOT Channel Payload"
                for item in model_targets
            )
        )
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


if __name__ == "__main__":
    unittest.main()
