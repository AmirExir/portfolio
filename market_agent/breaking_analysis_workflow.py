"""Install the independent breaking-news poll in the existing n8n market workflow."""

from __future__ import annotations

import copy
import shlex
from pathlib import Path
from typing import Any


POLL_NODE = "Check for Breaking Market News"
ANALYSIS_NODE = "Run Breaking News Analysis"
POLL_MINUTES = 5


def repair_breaking_analysis_workflow(
    workflow: dict[str, Any], *, project_root: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Wire a trusted local command to a five-minute timer, retaining existing branches."""
    fixed = copy.deepcopy(workflow)
    nodes, connections = fixed.get("nodes"), fixed.get("connections")
    if not isinstance(nodes, list) or not isinstance(connections, dict):
        raise ValueError("Expected a workflow with nodes and connections")
    if not project_root.is_absolute():
        raise ValueError("The market agent project root must be absolute")
    by_name = {node["name"]: node for node in nodes}
    if len(by_name) != len(nodes) or not {"News API", "Run Optimization Scripts"} <= by_name.keys():
        raise ValueError("Expected a market workflow with distinct news and optimization nodes")
    command = f"cd {shlex.quote(str(project_root))} && ./.venv/bin/python -m market_agent.breaking_analysis"
    specifications = (
        (POLL_NODE, "3f21ab2c-cab2-4435-a2e4-eee612944325", "scheduleTrigger", 1.2,
         {"rule": {"interval": [{"field": "minutes", "minutesInterval": POLL_MINUTES}]}}, [-2880, 1100]),
        (ANALYSIS_NODE, "58da02bd-38c0-4487-a7e1-73018e18cae3", "executeCommand", 1,
         {"command": command}, [-2620, 1100]),
    )
    changes = []
    for name, node_id, kind, version, params, position in specifications:
        config = {"type": f"n8n-nodes-base.{kind}", "typeVersion": version, "parameters": params}
        existing = by_name.get(name)
        if existing is None:
            if any(node.get("id") == node_id for node in nodes):
                raise ValueError(f"The node ID for {name} is already in use")
            nodes.append({"name": name, "id": node_id, "position": position, **config})
            changes.append(f"add {name}")
        elif existing.get("id") != node_id:
            raise ValueError(f"Another workflow node already uses the name {name}")
        elif any(existing.get(key) != value for key, value in config.items()):
            existing.update(config)
            changes.append(f"update {name}")
    edge = {"main": [[{"node": ANALYSIS_NODE, "type": "main", "index": 0}]]}
    if POLL_NODE in connections and connections[POLL_NODE] != edge:
        raise ValueError("The breaking-news timer has unexpected destinations")
    if connections.get(ANALYSIS_NODE, {}).get("main") and any(connections[ANALYSIS_NODE]["main"]):
        raise ValueError("The breaking-news analysis command has unexpected destinations")
    if POLL_NODE not in connections:
        connections[POLL_NODE] = edge
        changes.append("connect the breaking-news timer to the analysis command")
    return fixed, tuple(changes)
