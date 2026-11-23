"""Exercise the news-only CLI transform against independently versioned n8n graphs."""

import json
import sqlite3
import stat
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from market_agent.news_workflow import repair_market_news_workflow
from market_agent.tests.test_news_workflow import workflow_fixture
from scripts.repair_n8n_market_optimization import repair_database


class MarketNewsDatabaseRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "workflow.sqlite"
        self.backup = Path(directory.name) / "backup.sqlite"
        self.originals = {}
        with closing(sqlite3.connect(self.database)) as db:
            db.executescript("""
                CREATE TABLE workflow_entity
                    (id, nodes, connections, versionId, activeVersionId, versionCounter);
                CREATE TABLE workflow_history (workflowId, versionId, nodes, connections);
                CREATE TABLE workflow_published_version (workflowId, publishedVersionId);
                CREATE TABLE execution_entity (workflowId, status);
                INSERT INTO workflow_published_version VALUES ('target','published');
            """)
            for version in ("draft", "draft-history", "active", "published", "archived"):
                workflow = workflow_fixture()
                workflow["nodes"].append({"name": "Private forecast settings", "parameters": {"keep": version}})
                self.originals[version] = workflow
                graph = (json.dumps(workflow["nodes"]), json.dumps(workflow["connections"]))
                if version == "draft":
                    db.execute(
                        "INSERT INTO workflow_entity VALUES ('target',?,?,'draft-history','active',1)", graph
                    )
                else:
                    db.execute("INSERT INTO workflow_history VALUES ('target',?,?,?)", (version, *graph))
            db.commit()
        stopped = patch("scripts.repair_n8n_market_optimization._n8n_is_running", return_value=False)
        stopped.start()
        self.addCleanup(stopped.stop)

    def _graphs(self, path: Path) -> dict:
        with closing(sqlite3.connect(path)) as db:
            rows = db.execute("""
                SELECT 'draft', nodes, connections FROM workflow_entity
                UNION ALL SELECT versionId, nodes, connections FROM workflow_history
            """).fetchall()
        return {
            version: {"nodes": json.loads(nodes), "connections": json.loads(connections)}
            for version, nodes, connections in rows
        }

    def _repair(self) -> tuple[str, ...]:
        return repair_database(self.database, self.backup, "target", news_only=True)

    def test_only_news_changes_in_each_current_version_with_owner_only_backup(self) -> None:
        self.assertTrue(self._repair())
        actual = self._graphs(self.database)
        for version, before in self.originals.items():
            expected = before if version == "archived" else repair_market_news_workflow(before)[0]
            self.assertEqual(actual[version], expected)
            private_settings = next(node for node in actual[version]["nodes"] if node["name"] == "Private forecast settings")
            self.assertEqual(private_settings["parameters"], {"keep": version})
        self.assertEqual(self._graphs(self.backup), self.originals)
        self.assertEqual(stat.S_IMODE(self.backup.stat().st_mode), 0o600)
        self.assertEqual(self._repair(), ())

    def test_history_error_rolls_back_draft_and_preserves_backup(self) -> None:
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("""
                CREATE TRIGGER fail_publish BEFORE UPDATE ON workflow_history
                WHEN OLD.versionId = 'published' BEGIN SELECT RAISE(ABORT, 'test failure'); END
            """)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "test failure"):
            self._repair()
        self.assertEqual(self._graphs(self.database), self.originals)
        self.assertEqual(self._graphs(self.backup), self.originals)

    def test_waiting_execution_prevents_backup_and_edit(self) -> None:
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("INSERT INTO execution_entity VALUES ('target','waiting')")
            db.commit()
        with self.assertRaisesRegex(RuntimeError, "found waiting"):
            self._repair()
        self.assertEqual(self._graphs(self.database), self.originals)
        self.assertFalse(self.backup.exists())
