"""Shared freshness and n8n publication helpers for ERCOT news briefs."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_FRESHNESS_HOURS = 36.0
_NEWS_TIMESTAMP = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:-\d{1,6})?Z)",
    re.IGNORECASE,
)
_MIN_UTC = datetime.min.replace(tzinfo=timezone.utc)

ERCOT_MODEL_NODE = "Message Model For ERCOT"
ERCOT_NEWS_DIGEST_NODE = "Build ERCOT News Digest"
ERCOT_PAYLOAD_NODE = "Build ERCOT GitHub Payload"
ERCOT_PUBLISH_NODE = "Save ERCOT To GitHub"
ERCOT_PUBLICATION_GATE_NODE = "Filter ERCOT Publications"
ERCOT_CHANNEL_PAYLOAD_NODE = "Build ERCOT Channel Payload"
ERCOT_FILE_PARSER_NODE = "Parse ERCOT File Updates"
ERCOT_SUMMARY_URL = (
    "=https://api.github.com/repos/AmirExir/portfolio/contents/"
    "ERCOTAPI/news_summaries/{{$json.filename}}"
)

ERCOT_FILE_PROMPT = """=Summarize only the supplied newly detected ERCOT changes.
Return concise plain bullet points, at most one bullet per distinct filing or event.
Merge duplicate evidence; do not manufacture extra angles to fill a digest.
For each bullet identify what actually changed and cite the supplied title and URL.
Distinguish new documents from updates to existing pages. A detected page change
does not establish that a proposal was newly filed, approved, or made effective.
Preserve proposal identifiers, revision/status information, dates and units.
Separate explicit evidence from engineering interpretation. Do not infer approval,
effective dates, requirements, or likely impacts from a title alone. If document
content is unavailable, report that limitation and only the observed posting.
Omit website navigation, library indexes, tracking code, and collection errors.
If there are no substantive changes, respond exactly: No ERCOT news available.

ERCOT changes:
{{ JSON.stringify($json.changes || []) }}
"""

ERCOT_FILE_PARSER_JS = """const raw = String($json.stdout || '');
if (Number($json.exitCode || 0) !== 0) throw new Error('ERCOT file monitor failed');
const start = raw.indexOf('{');
if (start === -1) throw new Error('ERCOT file monitor returned no JSON');
const payload = JSON.parse(raw.slice(start));
if (!Array.isArray(payload.changes)) throw new Error('Invalid ERCOT monitor changes');
const changes = payload.changes.filter(item => ['new', 'updated'].includes(item.status));
const errors = payload.changes.filter(item => !['new', 'updated'].includes(item.status));
if (errors.length) console.warn('ERCOT monitor reported ' + errors.length + ' collection errors');
if (errors.length && !changes.length) throw new Error('ERCOT collection failed; see monitor output');
return { json: { articles: [], changes, has_updates: changes.length > 0,
  telegram_text: payload.telegram_text || '' } };
"""

ERCOT_PUBLICATION_GATE_JS = """return $input.all().filter(item => {
  if (item.json.publish_to_channel === false) return false;
  const summary = String(item.json.message?.content || item.json.content || '').trim();
  return summary && !/^(No ERCOT news available\\.?|No (?:new|current) ERCOT.*)$/i.test(summary);
});
"""


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
        label="Recent brief" if is_fresh else "No recent brief",
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
    model_node = _named_node(nodes, ERCOT_MODEL_NODE)
    digest_node = _named_node(nodes, ERCOT_NEWS_DIGEST_NODE)

    digest_parameters = digest_node.setdefault("parameters", {})
    digest_javascript = Path(__file__).with_name("news_digest.js").read_text(
        encoding="utf-8"
    )
    if (
        digest_parameters.get("jsCode") != digest_javascript
        or digest_parameters.get("mode") != "runOnceForAllItems"
    ):
        digest_parameters.update(
            jsCode=digest_javascript, mode="runOnceForAllItems"
        )
        changes.append("remember articles across scheduled runs and skip stale repeats")

    model_messages = model_node.setdefault("parameters", {}).setdefault("messages", {})
    if model_messages.get("values") != [{"content": ERCOT_FILE_PROMPT}]:
        model_messages["values"] = [{"content": ERCOT_FILE_PROMPT}]
        changes.append("summarize only evidenced filing changes without repeated angles")

    parser_node = next(
        (node for node in nodes if node.get("name") == ERCOT_FILE_PARSER_NODE), None
    )
    if parser_node is not None:
        parser_parameters = parser_node.setdefault("parameters", {})
        if (
            parser_parameters.get("jsCode") != ERCOT_FILE_PARSER_JS
            or parser_parameters.get("mode") != "runOnceForEachItem"
        ):
            parser_parameters.update(
                jsCode=ERCOT_FILE_PARSER_JS, mode="runOnceForEachItem"
            )
            changes.append("keep collection errors out of filing summaries")

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

    gate_node = next(
        (node for node in nodes if node.get("name") == ERCOT_PUBLICATION_GATE_NODE),
        None,
    )
    if gate_node is None:
        position = payload_node.get("position", [0, 0])
        gate_node = {
            "id": "ercot-publication-gate",
            "name": ERCOT_PUBLICATION_GATE_NODE,
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [position[0] - 220, position[1]],
            "parameters": {},
        }
        nodes.append(gate_node)
        changes.append("add shared ERCOT publication gate")
    gate_parameters = gate_node.setdefault("parameters", {})
    if (
        gate_parameters.get("mode") != "runOnceForAllItems"
        or gate_parameters.get("jsCode") != ERCOT_PUBLICATION_GATE_JS
    ):
        gate_parameters.update(
            mode="runOnceForAllItems", jsCode=ERCOT_PUBLICATION_GATE_JS
        )
        changes.append("skip empty digests and keep requested recaps in direct replies")

    for source in (ERCOT_MODEL_NODE, ERCOT_NEWS_DIGEST_NODE):
        outputs = connections.get(source, {}).get("main", [])
        for output in outputs:
            for edge in list(output):
                target = edge.get("node")
                if target in {ERCOT_CHANNEL_PAYLOAD_NODE, ERCOT_PAYLOAD_NODE}:
                    output.remove(edge)
                    _ensure_connection(connections, ERCOT_PUBLICATION_GATE_NODE, target)
                    changes.append(f"route {source} publication through the gate")
        if _ensure_connection(connections, source, ERCOT_PUBLICATION_GATE_NODE):
            changes.append(f"connect {source} to publication gate")
    if _ensure_connection(connections, ERCOT_PUBLICATION_GATE_NODE, ERCOT_PAYLOAD_NODE):
        changes.append("connect ERCOT publication gate to GitHub payload")
    if _ensure_connection(connections, ERCOT_PAYLOAD_NODE, ERCOT_PUBLISH_NODE):
        changes.append("connect ERCOT payload to GitHub publisher")

    return repaired, tuple(changes)
