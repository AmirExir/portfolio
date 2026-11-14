"""Install isolated, deterministic ERCOT forecast delivery in the combined agent."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any


LOAD_CONDITION = "Is ERCOT Load Forecast"
LOAD_SCHEDULE = "ERCOT Load Forecast Schedule"
LOAD_CONTEXT = "ERCOT Load Forecast Context"
LOAD_RUNNER = "Run ERCOT Load Forecast"
LOAD_REPORT = "Build ERCOT Load Forecast Report"
LOAD_DIRECT = "Build ERCOT Load Forecast Direct Payload"
LOAD_CHANNEL = "Build ERCOT Load Forecast Channel Payload"
LOAD_SPLITTER = "Split ERCOT Load Forecast Channel Message"
LOAD_SENDER = "Send ERCOT Load Forecast Channel"

LOAD_COMMAND = 'exec "${ERCOT_REPO_ROOT:-$HOME/Documents/GitHub/portfolio}/scripts/run_ercot_load_forecast.sh"'

LOAD_REPORT_JS = r"""const context = $('ERCOT Load Forecast Context').first().json;
let report;
try {
  if ($json.error || ($json.exitCode !== undefined && Number($json.exitCode) !== 0)) {
    throw new Error('Forecast command failed');
  }
  const raw = String($json.stdout ?? '');
  if (!raw || raw.length > 100000) throw new Error('Invalid report size');
  report = JSON.parse(raw);
  if (!report || typeof report.ok !== 'boolean' || typeof report.text !== 'string' ||
      !report.text.trim() || report.text.length > 20000) throw new Error('Invalid report schema');
} catch {
  report = {ok: false, text: 'ERCOT Load Forecast — unavailable\n\nThe forecast job failed or returned an invalid report. No load values were published. Please try again; if this repeats, check the n8n execution.'};
}
return {json: {...context, text: report.text, report_ok: report.ok}};"""

LOAD_DIRECT_JS = r"""const item = $input.first().json;
if (item.source !== 'telegram' || !item.reply_chat_id || item.reply_chat_id === item.channel_chat_id) return [];
return [{json: {chat_id: item.reply_chat_id, text: item.text, report_ok: item.report_ok,
  disable_web_page_preview: true, send_direct: true}}];"""

LOAD_CHANNEL_JS = r"""return {json: {chat_id: $json.channel_chat_id, text: $json.text,
  report_ok: $json.report_ok, disable_web_page_preview: true}};"""


def _edge(name: str) -> dict[str, Any]:
    return {"node": name, "type": "main", "index": 0}


def repair_ercot_load_forecast_workflow(
    workflow: Mapping[str, Any], *, channel_chat_id: str = "@ERCOTNEWS",
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Reuse the ERCOT schedule and Telegram destinations without changing models.

    A separate trigger creates an independent execution at the news schedule's
    times, so a news/retriever failure cannot block the forecast. The workflow's
    existing timezone is preserved. All commands are fixed; chat text never
    becomes a shell argument. This function only edits an in-memory copy.
    """
    repaired = copy.deepcopy(dict(workflow))
    nodes = repaired.get("nodes")
    connections = repaired.get("connections")
    if not isinstance(nodes, list) or not isinstance(connections, dict):
        raise ValueError("Workflow must contain node and connection collections")
    by_name = {node.get("name"): node for node in nodes}
    if "ERCOT Schedule Trigger" not in by_name:
        return repaired, ()
    required = ("Is ERCOT Request", "Is ERCOT Telegram QA", "Telegram ERCOT News",
                "Send Direct Market Telegram", "Split Direct Telegram Message")
    for name in required:
        if name not in by_name:
            raise ValueError(f"Missing required ERCOT forecast integration node: {name}")
    if not channel_chat_id.strip():
        raise ValueError("ERCOT forecast channel must be configured")
    channel_sender = by_name["Telegram ERCOT News"]
    if not channel_sender.get("credentials", {}).get("telegramApi"):
        raise ValueError("ERCOT channel sender must have a Telegram credential binding")
    changes: list[str] = []

    def install(name: str, node_type: str, version: float, parameters: dict[str, Any],
                position: list[int], **extra: Any) -> None:
        expected = {"name": name, "type": node_type, "typeVersion": version,
                    "parameters": parameters, **extra}
        if name not in by_name:
            node = {"id": name.lower().replace(" ", "-"), "position": position, **expected}
            nodes.append(node)
            by_name[name] = node
            changes.append(f"add {name}")
        else:
            node = by_name[name]
            if any(node.get(key) != value for key, value in expected.items()):
                node.update(copy.deepcopy(expected))
                changes.append(f"update {name}")

    schedule = by_name["ERCOT Schedule Trigger"]
    install(LOAD_SCHEDULE, schedule["type"], schedule["typeVersion"],
            copy.deepcopy(schedule["parameters"]), [-3104, 1200])
    install(LOAD_CONDITION, "n8n-nodes-base.if", 2.2, {
        "conditions": {"options": {"version": 2, "leftValue": "", "caseSensitive": True,
                                   "typeValidation": "strict"}, "combinator": "and",
                       "conditions": [{"id": "ercot-load-requested",
                                       "leftValue": "={{ String($json.ercot_load_requested) }}",
                                       "rightValue": "true",
                                       "operator": {"type": "string", "operation": "equals"}}]},
        "options": {},
    }, [-3216, 0])
    context_js = (
        "const source = $json.source === 'telegram' ? 'telegram' : 'schedule';\n"
        "const replyId = String($json.reply_chat_id ?? '').trim();\n"
        "if (source === 'telegram' && !replyId) throw new Error('Forecast request has no reply chat');\n"
        "return {json: {source, reply_chat_id: replyId, channel_chat_id: "
        + json.dumps(channel_chat_id) + "}};"
    )
    install(LOAD_CONTEXT, "n8n-nodes-base.code", 2,
            {"mode": "runOnceForEachItem", "jsCode": context_js}, [-2800, 1200])
    install(LOAD_RUNNER, "n8n-nodes-base.executeCommand", 1,
            {"command": LOAD_COMMAND}, [-2580, 1200], onError="continueRegularOutput")
    for name, code, mode, position in (
        (LOAD_REPORT, LOAD_REPORT_JS, "runOnceForEachItem", [-2360, 1200]),
        (LOAD_DIRECT, LOAD_DIRECT_JS, "runOnceForAllItems", [-2140, 1400]),
        (LOAD_CHANNEL, LOAD_CHANNEL_JS, "runOnceForEachItem", [-2140, 1100]),
        (LOAD_SPLITTER, by_name["Split Direct Telegram Message"]["parameters"]["jsCode"],
         "runOnceForAllItems", [-1920, 1100]),
    ):
        install(name, "n8n-nodes-base.code", 2, {"mode": mode, "jsCode": code}, position)
    sender_parameters = copy.deepcopy(by_name["Send Direct Market Telegram"]["parameters"])
    for key in ("replyKeyboard", "replyKeyboardOptions", "replyMarkup"):
        sender_parameters.pop(key, None)
    sender_parameters["replyMarkup"] = "none"
    install(LOAD_SENDER, channel_sender["type"], channel_sender["typeVersion"],
            sender_parameters, [-1700, 1100], credentials=copy.deepcopy(channel_sender["credentials"]))

    ercot_route = connections.get("Is ERCOT Request", {}).get("main")
    if not ercot_route or ercot_route[0] not in ([_edge("Is ERCOT Telegram QA")], [_edge(LOAD_CONDITION)]):
        raise ValueError("Unexpected ERCOT request route; refusing to replace it")
    if ercot_route[0] != [_edge(LOAD_CONDITION)]:
        ercot_route[0] = [_edge(LOAD_CONDITION)]
        changes.append("route explicit load requests before ERCOT QA and news")
    routes = {
        LOAD_CONDITION: [[_edge(LOAD_CONTEXT)], [_edge("Is ERCOT Telegram QA")]],
        LOAD_SCHEDULE: [[_edge(LOAD_CONTEXT)]],
        LOAD_CONTEXT: [[_edge(LOAD_RUNNER)]],
        LOAD_RUNNER: [[_edge(LOAD_REPORT)]],
        LOAD_REPORT: [[_edge(LOAD_CHANNEL), _edge(LOAD_DIRECT)]],
        LOAD_DIRECT: [[_edge("Split Direct Telegram Message")]],
        LOAD_CHANNEL: [[_edge(LOAD_SPLITTER)]],
        LOAD_SPLITTER: [[_edge(LOAD_SENDER)]],
    }
    for name, outputs in routes.items():
        entry = connections.setdefault(name, {})
        if entry.get("main") != outputs:
            entry["main"] = outputs
            changes.append(f"connect {name}")
    return repaired, tuple(changes)
