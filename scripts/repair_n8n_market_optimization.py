#!/usr/bin/env python3
"""Repair the market optimizer branch in an offline n8n SQLite store."""

from __future__ import annotations

import argparse
import errno
import json
import os
import socket
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from market_agent.n8n_workflow import (  # noqa: E402
    DEFAULT_CREDENTIAL_SOURCE_NODE,
    repair_market_optimization_workflow,
)
from market_agent.news_workflow import repair_market_news_workflow  # noqa: E402
from scripts.n8n_workflow_store import repair_workflow_versions  # noqa: E402


def _n8n_is_running(host: str = "127.0.0.1", port: int = 5678) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.25)
        status = client.connect_ex((host, port))
    if status == 0:
        return True
    if status == errno.ECONNREFUSED:
        return False
    raise RuntimeError(
        f"Cannot verify whether n8n is stopped (socket error {status}); "
        "refusing to edit its workflow store"
    )


def _backup_database(connection: sqlite3.Connection, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing backup: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as backup:
        connection.backup(backup)


def repair_database(
    database: Path,
    backup: Path,
    workflow_id: str,
    *,
    news_only: bool = False,
    credential_source_node: str = DEFAULT_CREDENTIAL_SOURCE_NODE,
    ercot_forecast_owner_chat_id: str = "",
    ercot_forecast_publish_scheduled: bool = False,
) -> tuple[str, ...]:
    """Patch the draft and every active/published version in one transaction."""

    if _n8n_is_running():
        raise RuntimeError("Stop n8n before repairing its SQLite workflow store")
    if news_only:
        return repair_workflow_versions(database, backup, workflow_id, repair_market_news_workflow)
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

        repaired, changes = repair_market_optimization_workflow(
            {
                "nodes": json.loads(row["nodes"]),
                "connections": json.loads(row["connections"]),
            },
            credential_source_node=credential_source_node,
            ercot_forecast_owner_chat_id=ercot_forecast_owner_chat_id,
            ercot_forecast_publish_scheduled=ercot_forecast_publish_scheduled,
        )
        if not changes:
            return ()

        _backup_database(connection, backup)
        nodes_json = json.dumps(repaired["nodes"], separators=(",", ":"))
        connections_json = json.dumps(
            repaired["connections"], separators=(",", ":")
        )

        version_ids = {
            value for value in (row["versionId"], row["activeVersionId"]) if value
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
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument(
        "--news-only", action="store_true", help="Update only the market news branch in each current version.",
    )
    parser.add_argument(
        "--credential-source-node",
        default=DEFAULT_CREDENTIAL_SOURCE_NODE,
    )
    parser.add_argument(
        "--ercot-forecast-owner-chat-id",
        default=os.getenv("ERCOT_FORECAST_OWNER_CHAT_ID", ""),
        help="Private Telegram chat id for scheduled reports; empty archives without private delivery.",
    )
    parser.add_argument(
        "--publish-scheduled-ercot-forecasts", action="store_true",
        help="Explicitly enable public forecasts on the existing schedule (private commands never broadcast).",
    )
    args = parser.parse_args()

    changes = repair_database(
        args.database,
        args.backup,
        args.workflow_id,
        news_only=args.news_only,
        credential_source_node=args.credential_source_node,
        ercot_forecast_owner_chat_id=args.ercot_forecast_owner_chat_id,
        ercot_forecast_publish_scheduled=args.publish_scheduled_ercot_forecasts,
    )
    branch = "news" if args.news_only else "optimization"
    if not changes:
        print(f"Market {branch} workflow already satisfies the contract.")
        return
    print(f"Repaired market {branch} workflow:")
    for change in changes:
        print(f"- {change}")
    print(f"Backup: {args.backup}")


if __name__ == "__main__":
    main()
