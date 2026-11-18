from __future__ import annotations

import errno
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ERCOTAPI.load_forecast_workflow import (
    LOAD_CHANNEL,
    LOAD_CHANNEL_JS,
    repair_ercot_load_forecast_workflow,
)
from ERCOTAPI.news_pipeline import (
    ERCOT_MODEL_NODE,
    ERCOT_NEWS_DIGEST_NODE,
    ERCOT_PAYLOAD_NODE,
    ERCOT_PUBLISH_NODE,
    repair_ercot_publication_workflow,
)
from scripts.repair_n8n_ercot_publication import _n8n_is_running, main, repair_database


WORKFLOW_ID = "ercot-test-workflow"


def _workflow_fixture(marker: str) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "name": ERCOT_MODEL_NODE,
                "parameters": {},
                "credentials": {"openAiApi": {"id": f"fixture-{marker}"}},
            },
            {"name": ERCOT_NEWS_DIGEST_NODE, "parameters": {}},
            {
                "name": ERCOT_PAYLOAD_NODE,
                "parameters": {
                    "jsCode": "return {json: {branch: 'generated-output'}};"
                },
            },
            {
                "name": ERCOT_PUBLISH_NODE,
                "parameters": {
                    "url": "https://example.invalid/old-publication-path",
                    "bodyParameters": {
                        "parameters": [{"name": "branch", "value": "old"}]
                    },
                },
            },
            {"name": "Build ERCOT Channel Payload", "parameters": {}},
            {"name": "Independent Trigger", "parameters": {"marker": marker}},
            {"name": "Unrelated Version Task", "parameters": {"marker": marker}},
        ],
        "connections": {
            ERCOT_MODEL_NODE: {
                "main": [[{
                    "node": "Build ERCOT Channel Payload", "type": "main", "index": 0
                }]]
            },
            ERCOT_NEWS_DIGEST_NODE: {
                "main": [[{
                    "node": "Build ERCOT Channel Payload", "type": "main", "index": 0
                }]]
            },
            "Independent Trigger": {
                "main": [[{
                    "node": "Unrelated Version Task", "type": "main", "index": 0
                }]]
            },
        },
    }


class NewsWorkflowDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.database = Path(self.tempdir.name) / "database.sqlite"
        self.backup = Path(self.tempdir.name) / "backup.sqlite"
        self.running_patch = patch(
            "scripts.repair_n8n_ercot_publication._n8n_is_running", return_value=False
        )
        self.running_patch.start()
        self.addCleanup(self.running_patch.stop)
        self.originals = {
            name: _workflow_fixture(name)
            for name in ("draft", "current", "active", "published", "archive")
        }
        self.static_data = json.dumps(
            {"global": {"ercotPublishedArticles": ["retained-article-id"]}}
        )
        with closing(sqlite3.connect(self.database)) as connection:
            connection.executescript(
                """
                CREATE TABLE workflow_entity (
                    id TEXT PRIMARY KEY, nodes TEXT, connections TEXT,
                    versionId TEXT, activeVersionId TEXT, versionCounter INTEGER,
                    staticData TEXT, settings TEXT
                );
                CREATE TABLE workflow_history (
                    workflowId TEXT, versionId TEXT, nodes TEXT, connections TEXT,
                    authors TEXT, PRIMARY KEY (workflowId, versionId)
                );
                CREATE TABLE workflow_published_version (
                    workflowId TEXT PRIMARY KEY, publishedVersionId TEXT
                );
                """
            )
            draft = self.originals["draft"]
            connection.execute(
                "INSERT INTO workflow_entity VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    WORKFLOW_ID,
                    json.dumps(draft["nodes"]),
                    json.dumps(draft["connections"]),
                    "current",
                    "active",
                    7,
                    self.static_data,
                    '{"executionOrder":"v1"}',
                ),
            )
            for version_id, workflow in self.originals.items():
                if version_id == "draft":
                    continue
                connection.execute(
                    "INSERT INTO workflow_history VALUES (?, ?, ?, ?, ?)",
                    (
                        WORKFLOW_ID,
                        version_id,
                        json.dumps(workflow["nodes"]),
                        json.dumps(workflow["connections"]),
                        f"author-{version_id}",
                    ),
                )
            connection.execute(
                "INSERT INTO workflow_published_version VALUES (?, ?)",
                (WORKFLOW_ID, "published"),
            )
            connection.commit()

    def _snapshot(self, database: Path | None = None) -> str:
        with closing(sqlite3.connect(database or self.database)) as connection:
            return "\n".join(connection.iterdump())

    def _read_workflow(self, version_id: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.database)) as connection:
            if version_id == "draft":
                row = connection.execute(
                    "SELECT nodes, connections FROM workflow_entity WHERE id = ?",
                    (WORKFLOW_ID,),
                ).fetchone()
            else:
                row = connection.execute(
                    """SELECT nodes, connections FROM workflow_history
                    WHERE workflowId = ? AND versionId = ?""",
                    (WORKFLOW_ID, version_id),
                ).fetchone()
        assert row is not None
        return {"nodes": json.loads(row[0]), "connections": json.loads(row[1])}

    def _repair(self) -> tuple[str, ...]:
        return repair_database(self.database, self.backup, WORKFLOW_ID)

    def test_each_referenced_version_preserves_its_own_unrelated_content(self) -> None:
        before = self._snapshot()

        changes = self._repair()

        self.assertTrue(changes)
        self.assertEqual(self._snapshot(self.backup), before)
        self.assertEqual(self.backup.stat().st_mode & 0o777, 0o600)
        for version_id in ("draft", "current", "active", "published"):
            with self.subTest(version=version_id):
                expected, _ = repair_ercot_publication_workflow(
                    self.originals[version_id]
                )
                actual = self._read_workflow(version_id)
                self.assertEqual(actual, expected)
                unrelated_node = next(
                    node for node in actual["nodes"]
                    if node["name"] == "Unrelated Version Task"
                )
                self.assertEqual(unrelated_node["parameters"]["marker"], version_id)
                self.assertEqual(
                    actual["connections"]["Independent Trigger"],
                    self.originals[version_id]["connections"]["Independent Trigger"],
                )
        self.assertEqual(self._read_workflow("archive"), self.originals["archive"])
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute(
                "SELECT versionCounter, staticData, settings FROM workflow_entity"
            ).fetchone()
            self.assertEqual(row, (8, self.static_data, '{"executionOrder":"v1"}'))
            authors = connection.execute(
                "SELECT versionId, authors FROM workflow_history ORDER BY versionId"
            ).fetchall()
        self.assertEqual(
            authors,
            [
                (name, f"author-{name}")
                for name in sorted(self.originals) if name != "draft"
            ],
        )

    def test_correct_draft_does_not_skip_stale_active_versions(self) -> None:
        draft, _ = repair_ercot_publication_workflow(self.originals["draft"])
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "UPDATE workflow_entity SET nodes = ?, connections = ?",
                (json.dumps(draft["nodes"]), json.dumps(draft["connections"])),
            )
            connection.commit()
        self.originals["draft"] = draft

        self.assertTrue(self._repair())

        self.assertEqual(self._read_workflow("draft"), draft)
        for version_id in ("current", "active", "published"):
            expected, _ = repair_ercot_publication_workflow(self.originals[version_id])
            self.assertEqual(self._read_workflow(version_id), expected)
        with closing(sqlite3.connect(self.database)) as connection:
            counter = connection.execute(
                "SELECT versionCounter FROM workflow_entity"
            ).fetchone()[0]
            self.assertEqual(counter, 7)

    def test_repair_is_idempotent_without_creating_another_backup(self) -> None:
        self._repair()
        before = self._snapshot()

        self.assertEqual(self._repair(), ())
        self.assertEqual(self._snapshot(), before)
        new_backup = self.backup.with_name("unused-backup.sqlite")
        self.assertEqual(repair_database(self.database, new_backup, WORKFLOW_ID), ())
        self.assertFalse(new_backup.exists())

    def test_forecast_only_repair_preserves_news_and_each_version(self) -> None:
        def edge(name: str) -> dict[str, Any]:
            return {"node": name, "type": "main", "index": 0}

        with closing(sqlite3.connect(self.database)) as connection:
            for version_id, workflow in self.originals.items():
                workflow["nodes"].extend([
                    {"name": "ERCOT Schedule Trigger", "type": "n8n-nodes-base.scheduleTrigger",
                     "typeVersion": 1.2, "parameters": {"rule": {"interval": [{"triggerAtHour": 7}]}}},
                    {"name": "Is ERCOT Request", "parameters": {}},
                    {"name": "Is ERCOT Telegram QA", "parameters": {}},
                    {"name": "Telegram ERCOT News", "type": "n8n-nodes-base.telegram",
                     "typeVersion": 1.2, "parameters": {},
                     "credentials": {"telegramApi": {"id": f"news-{version_id}"}}},
                    {"name": "Send Direct Market Telegram", "type": "n8n-nodes-base.telegram",
                     "typeVersion": 1.2, "parameters": {},
                     "credentials": {"telegramApi": {"id": f"direct-{version_id}"}}},
                    {"name": "Split Direct Telegram Message", "parameters": {"jsCode": "return $input.all();"}},
                    {"name": LOAD_CHANNEL, "parameters": {"mode": "runOnceForEachItem",
                     "jsCode": "return {json: {chat_id: '@ERCOTNEWS', text: $json.text}};"}},
                ])
                workflow["connections"].update({
                    "ERCOT Schedule Trigger": {"main": [[edge(ERCOT_NEWS_DIGEST_NODE)]]},
                    "Is ERCOT Request": {"main": [[edge("Is ERCOT Telegram QA")]]},
                    "Is ERCOT Telegram QA": {"main": [[], [edge(ERCOT_MODEL_NODE)]]},
                })
                values = (json.dumps(workflow["nodes"]), json.dumps(workflow["connections"]))
                if version_id == "draft":
                    connection.execute("UPDATE workflow_entity SET nodes = ?, connections = ?", values)
                else:
                    connection.execute(
                        "UPDATE workflow_history SET nodes = ?, connections = ? WHERE versionId = ?",
                        (*values, version_id),
                    )
            connection.commit()
        before = self._snapshot()

        self.assertTrue(repair_database(
            self.database, self.backup, WORKFLOW_ID, forecast_delivery_only=True,
        ))

        self.assertEqual(self._snapshot(self.backup), before)
        self.assertEqual(self.backup.stat().st_mode & 0o777, 0o600)
        for version_id in ("draft", "current", "active", "published"):
            with self.subTest(version=version_id):
                actual = self._read_workflow(version_id)
                expected, _ = repair_ercot_load_forecast_workflow(self.originals[version_id])
                self.assertEqual(actual, expected)
                actual_nodes = {node["name"]: node for node in actual["nodes"]}
                for node in self.originals[version_id]["nodes"]:
                    if node["name"] != LOAD_CHANNEL:
                        self.assertEqual(actual_nodes[node["name"]], node)
                for name, route in self.originals[version_id]["connections"].items():
                    if name != "Is ERCOT Request":
                        self.assertEqual(actual["connections"][name], route)
                self.assertEqual(actual_nodes[LOAD_CHANNEL]["parameters"], {
                    "mode": "runOnceForAllItems", "jsCode": LOAD_CHANNEL_JS,
                })
        self.assertEqual(self._read_workflow("archive"), self.originals["archive"])
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute(
                "SELECT versionCounter, staticData, settings FROM workflow_entity"
            ).fetchone()
        self.assertEqual(row, (8, self.static_data, '{"executionOrder":"v1"}'))
        after = self._snapshot()
        unused_backup = self.backup.with_name("unused-forecast-backup.sqlite")
        self.assertEqual(repair_database(
            self.database, unused_backup, WORKFLOW_ID, forecast_delivery_only=True,
        ), ())
        self.assertEqual(self._snapshot(), after)
        self.assertFalse(unused_backup.exists())

    def test_wal_database_backup_includes_uncheckpointed_committed_data(self) -> None:
        with closing(sqlite3.connect(self.database)) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute(
                "UPDATE workflow_entity SET settings = ?",
                ('{"executionOrder":"v1","saveDataSuccessExecution":"all"}',),
            )
            writer.commit()
            before = self._snapshot()

            self.assertTrue(self._repair())

            self.assertEqual(self._snapshot(self.backup), before)

    def test_invalid_history_is_rejected_before_backup_or_updates(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "UPDATE workflow_history SET nodes = '[]' WHERE versionId = 'published'"
            )
            connection.commit()
        before = self._snapshot()

        with self.assertRaisesRegex(ValueError, "missing required node"):
            self._repair()

        self.assertFalse(self.backup.exists())
        self.assertEqual(self._snapshot(), before)

    def test_missing_history_is_rejected_before_backup_or_updates(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "DELETE FROM workflow_history WHERE versionId = 'published'"
            )
            connection.commit()
        before = self._snapshot()

        with self.assertRaisesRegex(RuntimeError, "history row for published"):
            self._repair()

        self.assertFalse(self.backup.exists())
        self.assertEqual(self._snapshot(), before)

    def test_sql_failure_rolls_back_all_updates_and_keeps_original_backup(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.executescript(
                """
                CREATE TRIGGER fail_history_update BEFORE UPDATE ON workflow_history
                WHEN OLD.versionId = 'published'
                BEGIN SELECT RAISE(ABORT, 'simulated history failure'); END;
                """
            )
        before = self._snapshot()

        with self.assertRaisesRegex(
            sqlite3.IntegrityError, "simulated history failure"
        ):
            self._repair()

        self.assertEqual(self._snapshot(), before)
        self.assertEqual(self._snapshot(self.backup), before)

    def test_unfinished_executions_block_repair_when_listener_is_stopped(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "CREATE TABLE execution_entity (workflowId TEXT, status TEXT)"
            )
            connection.commit()
        for status in ("new", "running", "waiting"):
            with self.subTest(status=status):
                with closing(sqlite3.connect(self.database)) as connection:
                    connection.execute("DELETE FROM execution_entity")
                    connection.execute(
                        "INSERT INTO execution_entity VALUES (?, ?)",
                        (WORKFLOW_ID, status),
                    )
                    connection.commit()
                before = self._snapshot()

                with self.assertRaisesRegex(RuntimeError, f"found {status}"):
                    self._repair()

                self.assertFalse(self.backup.exists())
                self.assertEqual(self._snapshot(), before)

    def test_finished_and_other_workflow_executions_do_not_block_repair(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "CREATE TABLE execution_entity (workflowId TEXT, status TEXT)"
            )
            connection.executemany(
                "INSERT INTO execution_entity VALUES (?, ?)",
                [
                    (WORKFLOW_ID, "success"),
                    (WORKFLOW_ID, "error"),
                    ("other", "waiting"),
                ],
            )
            connection.commit()

        self.assertTrue(self._repair())

    def test_running_listener_blocks_repair(self) -> None:
        before = self._snapshot()
        with patch(
            "scripts.repair_n8n_ercot_publication._n8n_is_running", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "Stop n8n"):
                self._repair()

        self.assertFalse(self.backup.exists())
        self.assertEqual(self._snapshot(), before)

    def test_existing_backup_is_never_overwritten(self) -> None:
        self.backup.write_text("retain existing backup", encoding="utf-8")
        before = self._snapshot()

        with self.assertRaises(FileExistsError):
            self._repair()

        self.assertEqual(self._snapshot(), before)
        self.assertEqual(
            self.backup.read_text(encoding="utf-8"), "retain existing backup"
        )

    def test_missing_optional_published_table_is_supported(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("DROP TABLE workflow_published_version")
            connection.commit()

        self.assertTrue(self._repair())
        self.assertEqual(self._read_workflow("published"), self.originals["published"])


class PublicationRepairPreflightTests(unittest.TestCase):
    def test_cli_selects_forecast_delivery_only_when_requested(self) -> None:
        arguments = ["repair", "--database", "fixture.sqlite", "--backup", "backup.sqlite"]
        for extra, expected in (([], False), (["--forecast-delivery-only"], True)):
            with self.subTest(forecast_only=expected), patch("sys.argv", arguments + extra), \
                    patch("scripts.repair_n8n_ercot_publication.repair_database", return_value=()) as repair, \
                    redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(repair.call_args.kwargs.get("forecast_delivery_only", False), expected)

    def test_only_connection_refused_establishes_stopped_service(self) -> None:
        with patch("scripts.repair_n8n_ercot_publication.socket.socket") as socket:
            client = socket.return_value.__enter__.return_value
            client.connect_ex.return_value = 0
            self.assertTrue(_n8n_is_running())
            client.connect_ex.return_value = errno.ECONNREFUSED
            self.assertFalse(_n8n_is_running())
            for status in (errno.EPERM, errno.EACCES, errno.ETIMEDOUT):
                client.connect_ex.return_value = status
                with self.subTest(status=status), self.assertRaisesRegex(
                    RuntimeError, "Cannot verify whether n8n is stopped",
                ):
                    _n8n_is_running()


if __name__ == "__main__":
    unittest.main()
