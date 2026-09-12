#!/usr/bin/env python3
"""Repair the active ERCOT news publisher in an offline n8n SQLite store."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import sys
from contextlib import closing
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path atomically so an existing backup can never be overwritten.
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    with closing(sqlite3.connect(destination)) as backup:
        connection.backup(backup)


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone() is not None


def _require_idle_workflow(connection: sqlite3.Connection, workflow_id: str) -> None:
    if not _table_exists(connection, "execution_entity"):
        return
    unfinished = connection.execute(
        """
        SELECT status FROM execution_entity
        WHERE workflowId = ? AND status IN ('new', 'running', 'waiting')
        LIMIT 1
        """,
        (workflow_id,),
    ).fetchone()
    if unfinished is not None:
        raise RuntimeError(
            "Finish or cancel outstanding ERCOT workflow executions before "
            f"repairing its SQLite store (found {unfinished['status']})"
        )


def _prepare_repair(row: sqlite3.Row) -> tuple[tuple[str, str], tuple[str, ...]]:
    repaired, changes = repair_ercot_publication_workflow(
        {
            "nodes": json.loads(row["nodes"]),
            "connections": json.loads(row["connections"]),
        }
    )
    return (
        json.dumps(repaired["nodes"], separators=(",", ":")),
        json.dumps(repaired["connections"], separators=(",", ":")),
    ), changes


def repair_database(
    database: Path,
    backup: Path,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
) -> tuple[str, ...]:
    """Repair each current version independently in a stopped n8n store.

    Validate the draft and all referenced history rows before creating a backup
    or changing workflow data. Only publication nodes/connections are updated;
    version-specific nodes and persistent workflow staticData are preserved.
    """

    if _n8n_is_running():
        raise RuntimeError("Stop n8n before repairing its SQLite workflow store")
    if not database.is_file():
        raise FileNotFoundError(f"n8n database not found: {database}")

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=5000")
        with connection:
            # Keep the validated versions stable until every update commits.
            connection.execute("BEGIN IMMEDIATE")
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

            _require_idle_workflow(connection, workflow_id)
            draft_values, draft_changes = _prepare_repair(row)
            changes = [f"draft: {change}" for change in draft_changes]
            version_ids = {
                value
                for value in (row["versionId"], row["activeVersionId"])
                if value
            }
            if _table_exists(connection, "workflow_published_version"):
                published_rows = connection.execute(
                    """
                    SELECT publishedVersionId FROM workflow_published_version
                    WHERE workflowId = ?
                    """,
                    (workflow_id,),
                ).fetchall()
                version_ids.update(
                    published_row["publishedVersionId"]
                    for published_row in published_rows
                    if published_row["publishedVersionId"]
                )

            history_updates: dict[str, tuple[str, str]] = {}
            for version_id in sorted(version_ids):
                history_rows = connection.execute(
                    """
                    SELECT nodes, connections FROM workflow_history
                    WHERE workflowId = ? AND versionId = ?
                    """,
                    (workflow_id, version_id),
                ).fetchall()
                if len(history_rows) != 1:
                    raise RuntimeError(
                        f"Expected one workflow history row for {version_id}"
                    )
                values, version_changes = _prepare_repair(history_rows[0])
                if version_changes:
                    history_updates[version_id] = values
                    changes.extend(
                        f"version {version_id}: {change}" for change in version_changes
                    )

            if not changes:
                return ()

            # A backup on the connection holding BEGIN IMMEDIATE would block.
            # A separate reader sees the same committed state; the reserved
            # write lock prevents another writer from changing it meanwhile.
            with closing(
                sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
            ) as backup_source:
                _backup_database(backup_source, backup)

            if draft_changes:
                connection.execute(
                    """
                    UPDATE workflow_entity
                    SET nodes = ?, connections = ?, versionCounter = versionCounter + 1
                    WHERE id = ?
                    """,
                    (*draft_values, workflow_id),
                )
            for version_id, values in history_updates.items():
                cursor = connection.execute(
                    """
                    UPDATE workflow_history
                    SET nodes = ?, connections = ?
                    WHERE workflowId = ? AND versionId = ?
                    """,
                    (*values, workflow_id, version_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"Expected one workflow history row for {version_id}"
                    )
        return tuple(changes)
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
