"""Shared freshness and n8n publication helpers for ERCOT news briefs."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


DEFAULT_FRESHNESS_HOURS = 36.0
_NEWS_TIMESTAMP = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:-\d{1,6})?Z)",
    re.IGNORECASE,
)
_MIN_UTC = datetime.min.replace(tzinfo=timezone.utc)

ERCOT_MODEL_NODE = "Message Model For ERCOT"
ERCOT_PAYLOAD_NODE = "Build ERCOT GitHub Payload"
ERCOT_PUBLISH_NODE = "Save ERCOT To GitHub"
ERCOT_SUMMARY_URL = (
    "=https://api.github.com/repos/AmirExir/portfolio/contents/"
    "ERCOTAPI/news_summaries/{{$json.filename}}"
)


@dataclass(frozen=True)
class NewsBriefState:
    """Freshness information used by the dashboard status panel."""

    published_at: datetime | None
    age_hours: float | None
    is_fresh: bool
    label: str
    status: str


def news_brief_timestamp(filename: str) -> datetime | None:
    """Read the UTC timestamp embedded in an n8n summary filename."""

    match = _NEWS_TIMESTAMP.search(str(filename or ""))
    if not match:
        return None

    raw_timestamp = match.group("timestamp")
    formats = (
        "%Y-%m-%dT%H-%M-%S-%fZ",
        "%Y-%m-%dT%H-%M-%SZ",
    )
    for timestamp_format in formats:
        try:
            return datetime.strptime(raw_timestamp, timestamp_format).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
    return None


def news_item_sort_key(filename: str) -> tuple[datetime, str]:
    """Sort mixed summary prefixes by their timestamps, then by name."""

    return (news_brief_timestamp(filename) or _MIN_UTC, str(filename or ""))


def assess_news_brief(
    filename: str,
    *,
    now: datetime | None = None,
    freshness_hours: float = DEFAULT_FRESHNESS_HOURS,
) -> NewsBriefState:
    """Classify a published brief without calling its producer."""

    published_at = news_brief_timestamp(filename)
    if published_at is None:
        return NewsBriefState(
            published_at=None,
            age_hours=None,
            is_fresh=False,
            label="Update age unknown",
            status="warn",
        )

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    age_hours = max(
        (
            current_time.astimezone(timezone.utc) - published_at
        ).total_seconds()
        / 3600.0,
        0.0,
    )
    is_fresh = age_hours <= float(freshness_hours)
    return NewsBriefState(
        published_at=published_at,
        age_hours=age_hours,
        is_fresh=is_fresh,
        label="Live pipeline" if is_fresh else "Pipeline stale",
        status="ok" if is_fresh else "warn",
    )


def format_brief_age(age_hours: float | None) -> str:
    """Return a compact human-readable artifact age."""

    if age_hours is None:
        return "age unknown"
    if age_hours < 1:
        return "less than 1 hour old"
    if age_hours < 48:
        return f"{age_hours:.0f} hours old"
    return f"{age_hours / 24.0:.1f} days old"


def _named_node(nodes: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for node in nodes:
        if node.get("name") == name:
            return node
    raise ValueError(f"Workflow is missing required node: {name}")


def _ensure_connection(
    connections: dict[str, Any],
    source: str,
    target: str,
) -> bool:
    source_connections = connections.setdefault(source, {})
    main_outputs = source_connections.setdefault("main", [[]])
    if not main_outputs:
        main_outputs.append([])
    first_output = main_outputs[0]
    if any(item.get("node") == target for item in first_output):
        return False
    first_output.append({"node": target, "type": "main", "index": 0})
    return True


def repair_ercot_publication_workflow(
    workflow: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Repair the producer/consumer contract in an exported n8n workflow.

    The function intentionally leaves credentials and unrelated nodes untouched.
    """

    repaired = copy.deepcopy(dict(workflow))
    nodes = repaired.get("nodes")
    connections = repaired.get("connections")
    if not isinstance(nodes, list) or not isinstance(connections, dict):
        raise ValueError("Workflow must contain node and connection collections")

    changes: list[str] = []
    payload_node = _named_node(nodes, ERCOT_PAYLOAD_NODE)
    publish_node = _named_node(nodes, ERCOT_PUBLISH_NODE)
    _named_node(nodes, ERCOT_MODEL_NODE)

    payload_parameters = payload_node.setdefault("parameters", {})
    javascript = str(payload_parameters.get("jsCode", ""))
    updated_javascript, branch_replacements = re.subn(
        r"(branch\s*:\s*)['\"][^'\"]+['\"]",
        r"\1'main'",
        javascript,
    )
    if branch_replacements == 0:
        raise ValueError(
            f"{ERCOT_PAYLOAD_NODE} does not define a branch in its JavaScript"
        )
    if updated_javascript != javascript:
        payload_parameters["jsCode"] = updated_javascript
        changes.append("set payload branch to main")

    publish_parameters = publish_node.setdefault("parameters", {})
    if publish_parameters.get("url") != ERCOT_SUMMARY_URL:
        publish_parameters["url"] = ERCOT_SUMMARY_URL
        changes.append("publish into ERCOTAPI/news_summaries")

    body_parameters = (
        publish_parameters.setdefault("bodyParameters", {})
        .setdefault("parameters", [])
    )
    branch_parameter = next(
        (
            parameter
            for parameter in body_parameters
            if parameter.get("name") == "branch"
        ),
        None,
    )
    if branch_parameter is None:
        body_parameters.append({"name": "branch", "value": "main"})
        changes.append("add main branch request parameter")
    elif branch_parameter.get("value") != "main":
        branch_parameter["value"] = "main"
        changes.append("set publisher branch to main")

    if _ensure_connection(connections, ERCOT_MODEL_NODE, ERCOT_PAYLOAD_NODE):
        changes.append("connect ERCOT model to GitHub payload")
    if _ensure_connection(connections, ERCOT_PAYLOAD_NODE, ERCOT_PUBLISH_NODE):
        changes.append("connect ERCOT payload to GitHub publisher")

    return repaired, tuple(changes)

