"""Deterministic Telegram market/ERCOT commands, help routing and presentation.

The internal optimization schema remains unchanged. New command names normalize
to the existing parser, and help bypasses every research/publication branch.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any


MARKET_REPORT_TITLE = "Market Outlook & Trade Watchlist"
RESOLVER_NODE = "Resolve Market Chat Command"
HELP_CONDITION_NODE = "Is Market Help"
HELP_BUILDER_NODE = "Build Market Help"
CLASSIFIER_NODE = "Classify Request"
CONTEXT_NODE = "Market Request Context"
DIRECT_SENDER_NODE = "Send Direct Market Telegram"

TELEGRAM_BOT_COMMANDS = [
    {"command": "analysis", "description": "Price forecasts and trade watchlist"},
    {"command": "briefing", "description": "Market news plus forecasts and watchlist"},
    {"command": "news", "description": "Market news and event watchlist"},
    {"command": "ercot_news", "description": "ERCOT and Texas grid news"},
    {"command": "ercot_load", "description": "ERCOT load forecast report"},
    {"command": "market", "description": "Show market commands and examples"},
    {"command": "help", "description": "Show market and ERCOT options"},
]

MARKET_HELP_TEXT = """Market & ERCOT agent options

Tap a button below or type a command:

Run market analysis — price forecasts, qualified buy/sell signals and research watchlists.
Run full market briefing — market news plus the same analysis.
Market news — headline brief and event watchlist.
ERCOT news — Texas grid news and ERCOT updates.
ERCOT load forecast — the load/demand forecast report.
Market help — these options only; no analysis runs.

Examples:
Run market analysis for NVDA, MSFT next week
/analysis NVDA MSFT next week
/briefing
/news
/ercot_news
/ercot_load

For market analysis, the default horizon is 30 sessions. Add “next week” for 5, “tomorrow” for 1, or “long term” for 60. Stocks count trading days; crypto counts calendar days.

ERCOT load forecasts also post to @ERCOTNEWS and accompany the existing twice-daily ERCOT news schedule (6:00 AM and 3:00 PM Central).

“Run market optimization” and other existing requests still work. Reports are forecasts and watchlists; they do not optimize your portfolio or place trades.

Use /market, /help, /start or /menu to reopen these options."""

MARKET_COMMAND_RESOLVER_JS = r"""const input = $json;
if (input.source !== 'telegram') {
  return {json: {...input, market_help_requested: false, ercot_load_requested: false}};
}
const original = String(input.market_original_text ?? input.text ?? '').trim();
const slash = original.match(/^\/(start|help|menu|market|analysis|briefing|news|ercot_news|ercot_load)(?:@[a-z0-9_]+)?(?:\s+([\s\S]*))?$/i);
const helpText = original.replace(/[?!.]+$/, '').trim();
const naturalHelp = /^(?:please\s+)?(?:help|menu|options|commands|market\s+(?:help|menu|options|commands)|(?:show|list)(?:\s+me)?(?:\s+the)?\s+(?:market\s+)?(?:options|commands|menu)|what\s+can\s+(?:i\s+ask(?:\s+you)?|you\s+do)|how\s+(?:do|can)\s+i\s+use\s+(?:this\s+)?(?:market\s+)?(?:agent|bot)|help(?:\s+me)?\s+(?:with\s+)?(?:the\s+)?market)(?:\s+please)?$/i.test(helpText);
const slashName = slash ? slash[1].toLowerCase() : '';
if (naturalHelp || ['start', 'help', 'menu', 'market'].includes(slashName)) {
  return {json: {...input, text: original, market_original_text: original,
    market_chat_command: 'help', market_help_requested: true,
    ercot_load_requested: false,
    run_news: false, run_optimization: false, optimization_command: null}};
}
let command = '';
let suffix = '';
if (slash) {
  command = slashName;
  suffix = slash[2] || '';
} else {
  const prefix = /^(?:(?:can|could)\s+you\s+)?(?:please\s+)?(?:run\s+)?(?:the\s+)?/i;
  const candidate = original.replace(prefix, '');
  const analysis = candidate.match(/^(?:stock\s+)?market\s+analysis\b([\s\S]*)$/i);
  const briefing = candidate.match(/^full\s+market\s+briefing\b([\s\S]*)$/i);
  const news = candidate.match(/^(?:stock\s+)?market\s+news\b([\s\S]*)$/i);
  const ercotNews = candidate.match(/^ercot\s+news\b([\s\S]*)$/i);
  const ercotCandidate = original.replace(/^(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?:(?:run|get|fetch|show|generate|give)(?:\s+me)?\s+|what(?:'s|\s+is)\s+)?(?:the\s+)?/i, '').replace(/[?!.]+$/, '').trim();
  // Match a request for a forecast, not an engineering question about how one is made.
  const ercotLoad = /^(?:ercot\s+(?:load|demand)\s+forecasts?|(?:load|demand)\s+forecasts?\s+(?:for\s+)?ercot|forecast\s+ercot\s+(?:load|demand))(?:\s+(?:(?:for\s+)?(?:today|tomorrow)|now|please))?$/i.test(ercotCandidate);
  if (analysis) { command = 'analysis'; suffix = analysis[1]; }
  else if (briefing) { command = 'briefing'; suffix = briefing[1]; }
  else if (news) { command = 'news'; suffix = news[1]; }
  else if (ercotNews) { command = 'ercot_news'; suffix = ercotNews[1]; }
  else if (ercotLoad) { command = 'ercot_load'; }
}
const legacyPrefix = {
  analysis: 'Run market optimization',
  briefing: 'Run market news and optimization',
  news: 'Market news',
  ercot_news: 'Run ERCOT news',
  ercot_load: 'ERCOT load forecast',
}[command];
return {json: {...input,
  text: legacyPrefix ? `${legacyPrefix}${suffix.trim() ? ' ' + suffix.trim() : ''}` : original,
  market_original_text: original, market_chat_command: command || null,
  market_help_requested: false,
  ercot_load_requested: command === 'ercot_load',
}};"""

MARKET_HELP_JS = (
    "if ($json.market_help_requested !== true || $json.source !== 'telegram') {\n"
    "  throw new Error('Market help requires an explicit Telegram help request.');\n"
    "}\n"
    "const chatId = String($json.reply_chat_id ?? '').trim();\n"
    "if (!chatId) throw new Error('Market help has no reply chat.');\n"
    "return {json: {chat_id: chatId, text: " + json.dumps(MARKET_HELP_TEXT)
    + ", disable_web_page_preview: true, send_direct: true}};"
)

PRIVATE_REPLY_MARKUP_EXPRESSION = (
    "={{ (() => { try { const update = $('Listen for Telegram').first().json; "
    "const chat = update.message?.chat || update.channel_post?.chat; "
    "return chat?.type === 'private' ? 'replyKeyboard' : 'none'; "
    "} catch { return 'none'; } })() }}"
)

MARKET_REPLY_KEYBOARD = {
    "rows": [
        {"row": {"buttons": [{"text": "Run market analysis", "additionalFields": {}}]}},
        {"row": {"buttons": [{"text": "Run full market briefing", "additionalFields": {}}]}},
        {"row": {"buttons": [
            {"text": "Market news", "additionalFields": {}},
            {"text": "Market help", "additionalFields": {}},
        ]}},
        {"row": {"buttons": [
            {"text": "ERCOT news", "additionalFields": {}},
            {"text": "ERCOT load forecast", "additionalFields": {}},
        ]}},
    ]
}
MARKET_REPLY_KEYBOARD_OPTIONS = {
    "resize_keyboard": True,
    "one_time_keyboard": False,
    "selective": False,
}


def _node(nodes: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [node for node in nodes if node.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one market chat node {name!r}")
    return matches[0]


def _edge(name: str) -> dict[str, Any]:
    return {"node": name, "type": "main", "index": 0}


def _preserve_original_request(parameters: dict[str, Any]) -> bool:
    code = parameters.get("jsCode")
    if not isinstance(code, str):
        raise ValueError("Market Request Context must define JavaScript")
    marker = "const marketOriginalRequestText = String($json.market_original_text ?? $json.text ?? 'scheduled market update');"
    if code.startswith(marker + "\n"):
        return False
    updated, field_count = re.subn(r"(\brequest_text:\s*)text\b", r"\1marketOriginalRequestText", code)
    updated, argument_count = re.subn(
        r"('--request-text',\s*)shellQuote\(text\)",
        r"\1shellQuote(marketOriginalRequestText)", updated,
    )
    if (field_count, argument_count) != (1, 1):
        raise ValueError("Market Request Context does not expose the expected original-request fields")
    parameters["jsCode"] = marker + "\n" + updated
    return True


def repair_market_chat_workflow(
    workflow: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Add command/help routing and private-chat menus without changing models.

    Workflows without a classifier retain their graph. Internal node names,
    optimization fields and the established ticker/horizon/model parser remain
    compatible. This function edits an in-memory copy and performs no messaging.
    """

    repaired = copy.deepcopy(dict(workflow))
    nodes = repaired.get("nodes")
    connections = repaired.get("connections")
    if not isinstance(nodes, list) or not isinstance(connections, dict):
        raise ValueError("Workflow must contain node and connection collections")
    changes: list[str] = []
    present = {node.get("name") for node in nodes}
    if CLASSIFIER_NODE not in present:
        return repaired, ()

    for name in ("Send Market Acknowledgement", "Send ERCOT Acknowledgement", DIRECT_SENDER_NODE, "Send Assistant Telegram"):
        if name not in present:
            continue
        parameters = _node(nodes, name).setdefault("parameters", {})
        expected = {
            "replyMarkup": PRIVATE_REPLY_MARKUP_EXPRESSION,
            "replyKeyboard": MARKET_REPLY_KEYBOARD,
            "replyKeyboardOptions": MARKET_REPLY_KEYBOARD_OPTIONS,
        }
        if any(parameters.get(key) != value for key, value in expected.items()):
            parameters.update(copy.deepcopy(expected))
            changes.append(f"show market command buttons in private chats via {name}")

    replacements = (
        ("Market Optimization", MARKET_REPORT_TITLE),
        ("Got it. I am running market news and optimization now.",
         f"I am preparing your full market briefing: market news plus {MARKET_REPORT_TITLE}."),
        ("Got it. I am running the market optimization now.",
         f"I am preparing your {MARKET_REPORT_TITLE}."),
        ("recommendation summary", "market outlook"),
        ("add the buy/sell event to Calendar", "add the forecast review to Calendar"),
        ("No optimized recommendation generated.", "No market outlook generated."),
        ("Market model signals: Buy", f"{MARKET_REPORT_TITLE} | Buy"),
        ("I will send the digest back here and post it to @ERCOTNEWS when it finishes.",
         "I will send the digest back here. Scheduled updates post new items to @ERCOTNEWS."),
    )
    for name in ("Build Market Acknowledgement", "Build ERCOT Acknowledgement", "Build Optimization Telegram Payload",
                 "Build Optimization Channel Payload", "Build Calendar Event Payload"):
        if name not in present:
            continue
        parameters = _node(nodes, name).setdefault("parameters", {})
        code = parameters.get("jsCode")
        if not isinstance(code, str):
            raise ValueError(f"{name} must define JavaScript")
        updated = code
        for before, after in replacements:
            updated = updated.replace(before, after)
        if updated != code:
            parameters["jsCode"] = updated
            changes.append(f"use accurate market outlook wording in {name}")

    classifier = _node(nodes, CLASSIFIER_NODE)
    _node(nodes, DIRECT_SENDER_NODE)
    context = _node(nodes, CONTEXT_NODE)
    if _preserve_original_request(context.setdefault("parameters", {})):
        changes.append("preserve original command text while retaining the established request parser")

    position = classifier.get("position", [0, 0])
    expected_nodes = (
        (RESOLVER_NODE, "market-chat-command-resolver-v1", "n8n-nodes-base.code", 2,
         {"mode": "runOnceForEachItem", "jsCode": MARKET_COMMAND_RESOLVER_JS}),
        (HELP_CONDITION_NODE, "market-chat-help-condition-v1", "n8n-nodes-base.if", 2.2,
         {"conditions": {"options": {"version": 2, "leftValue": "", "caseSensitive": True,
                                    "typeValidation": "strict"},
                         "combinator": "and", "conditions": [
                             {"id": "market-help-requested", "leftValue": "={{ String($json.market_help_requested) }}",
                              "rightValue": "true", "operator": {"type": "string", "operation": "equals"}},
                         ]}, "options": {}}),
        (HELP_BUILDER_NODE, "market-chat-help-builder-v1", "n8n-nodes-base.code", 2,
         {"mode": "runOnceForEachItem", "jsCode": MARKET_HELP_JS}),
    )
    for name, identifier, node_type, version, parameters in expected_nodes:
        if name not in present:
            nodes.append({"name": name, "id": identifier, "type": node_type,
                          "typeVersion": version, "position": list(position), "parameters": parameters})
            present.add(name)
            changes.append(f"add {name}")
        else:
            node = _node(nodes, name)
            if (node.get("parameters"), node.get("type"), node.get("typeVersion")) != (parameters, node_type, version):
                node.update(parameters=parameters, type=node_type, typeVersion=version)
                changes.append(f"update {name}")

    for source, entry in connections.items():
        if source in {RESOLVER_NODE, HELP_CONDITION_NODE, HELP_BUILDER_NODE}:
            continue
        for output in entry.get("main", []):
            for descriptor in output:
                if descriptor.get("node") == CLASSIFIER_NODE:
                    descriptor["node"] = RESOLVER_NODE
                    changes.append(f"resolve market commands before classification from {source}")
    routes = {
        RESOLVER_NODE: [[_edge(HELP_CONDITION_NODE)]],
        HELP_CONDITION_NODE: [[_edge(HELP_BUILDER_NODE)], [_edge(CLASSIFIER_NODE)]],
        HELP_BUILDER_NODE: [[_edge(DIRECT_SENDER_NODE)]],
    }
    for name, outputs in routes.items():
        entry = connections.setdefault(name, {})
        if entry.get("main") != outputs:
            entry["main"] = outputs
            changes.append(f"route {name} so help bypasses analysis and publishing")
    return repaired, tuple(changes)
