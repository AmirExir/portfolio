#!/usr/bin/env python3
"""Repair the active ERCOT news publisher in an offline n8n SQLite store."""

from __future__ import annotations

import argparse
import json
import socket
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ERCOTAPI.news_pipeline import repair_ercot_publication_workflow


DEFAULT_WORKFLOW_ID = "v5QCvpvtHqKS4cWD"


def _n8n_is_running(host: str = "127.0.0.1", port: int = 5678) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.25)
        return client.connect_ex((host, port)) == 0


def _backup_database(connection: sqlite3.Connection, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing backup: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as backup:
        connection.backup(backup)


def repair_database(
    database: Path,
    backup: Path,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
) -> tuple[str, ...]:
    """Patch the draft and every active/published version in one transaction."""

    if _n8n_is_running():
        raise RuntimeError("Stop n8n before repairing its SQLite workflow store")
    if not database.is_file():
        raise FileNotFoundError(f"n8n database not found: {database}")

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=5000")
        row = connection.execute(
            """
            SELECT id, nodes, connections, versionId, activeVersionId
            FROM workflow_entity
            WHERE id = ?
            """,
            (workflow_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"n8n workflow not found: {workflow_id}")

        repaired, changes = repair_ercot_publication_workflow(
            {
                "nodes": json.loads(row["nodes"]),
                "connections": json.loads(row["connections"]),
            }
        )
        if not changes:
            return ()

        _backup_database(connection, backup)
        nodes_json = json.dumps(repaired["nodes"], separators=(",", ":"))
        connections_json = json.dumps(
            repaired["connections"], separators=(",", ":")
        )

        version_ids = {
            value
            for value in (row["versionId"], row["activeVersionId"])
            if value
        }
        published_table = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'workflow_published_version'
            """
        ).fetchone()
        if published_table:
            published_row = connection.execute(
                """
                SELECT publishedVersionId
                FROM workflow_published_version
                WHERE workflowId = ?
                """,
                (workflow_id,),
            ).fetchone()
            if published_row and published_row["publishedVersionId"]:
                version_ids.add(published_row["publishedVersionId"])

        with connection:
            connection.execute(
                """
                UPDATE workflow_entity
                SET nodes = ?, connections = ?, versionCounter = versionCounter + 1
                WHERE id = ?
                """,
                (nodes_json, connections_json, workflow_id),
            )
            for version_id in version_ids:
                cursor = connection.execute(
                    """
                    UPDATE workflow_history
                    SET nodes = ?, connections = ?
                    WHERE workflowId = ? AND versionId = ?
                    """,
                    (nodes_json, connections_json, workflow_id, version_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"Expected one workflow history row for {version_id}"
                    )
        return changes
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--workflow-id", default=DEFAULT_WORKFLOW_ID)
    args = parser.parse_args()

    changes = repair_database(args.database, args.backup, args.workflow_id)
    if not changes:
        print("ERCOT publication workflow already satisfies the contract.")
        return
    print("Repaired ERCOT publication workflow:")
    for change in changes:
        print(f"- {change}")
    print(f"Backup: {args.backup}")


if __name__ == "__main__":
    main()
