#!/usr/bin/env python3
"""Repair the active ERCOT news publisher in an offline n8n SQLite store."""

from __future__ import annotations

import argparse
import errno
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ERCOTAPI.load_forecast_workflow import repair_ercot_load_forecast_workflow
from ERCOTAPI.news_pipeline import repair_ercot_publication_workflow
from scripts.n8n_workflow_store import repair_workflow_versions


DEFAULT_WORKFLOW_ID = "v5QCvpvtHqKS4cWD"


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


def repair_database(
    database: Path,
    backup: Path,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    *,
    forecast_delivery_only: bool = False,
) -> tuple[str, ...]:
    """Repair each current version independently in a stopped n8n store.

    Validate the draft and all referenced history rows before creating a backup
    or changing workflow data. Only publication nodes/connections are updated;
    version-specific nodes and persistent workflow staticData are preserved.
    Forecast-delivery mode installs the private-only forecast defaults without
    changing news, revision-request, or market processing. Scheduled comparisons
    remain archived; explicit load commands reply privately to their requester.
    """

    if _n8n_is_running():
        raise RuntimeError("Stop n8n before repairing its SQLite workflow store")
    transform = (
        repair_ercot_load_forecast_workflow
        if forecast_delivery_only else repair_ercot_publication_workflow
    )
    return repair_workflow_versions(database, backup, workflow_id, transform)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--workflow-id", default=DEFAULT_WORKFLOW_ID)
    parser.add_argument(
        "--forecast-delivery-only", action="store_true",
        help="Keep the news channel free of forecasts; preserve news and revision-request processing.",
    )
    args = parser.parse_args()

    changes = repair_database(
        args.database, args.backup, args.workflow_id,
        forecast_delivery_only=args.forecast_delivery_only,
    )
    if not changes:
        print("ERCOT publication workflow already satisfies the contract.")
        return
    print("Repaired ERCOT publication workflow:")
    for change in changes:
        print(f"- {change}")
    print(f"Backup: {args.backup}")


if __name__ == "__main__":
    main()
