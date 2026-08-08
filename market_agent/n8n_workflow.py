"""Repair and audit the market-optimization branch of an n8n workflow."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any


RUN_NODE = "Run Optimization Scripts"
REQUEST_CONTEXT_NODE = "Market Request Context"
VALIDATION_NODE = "Validate Optimization Output"
MODEL_NODE = "Message Model For Optimization"
RAW_PAYLOAD_NODE = "Build Optimization Raw JSON GitHub Payload"
TEXT_PAYLOAD_NODE = "Build Optimization GitHub Payload"
PUBLISH_NODE = "Save Optimization To GitHub"
OPTIMIZATION_TELEGRAM_NODE = "Send Optimization Telegram"
DIRECT_OPTIMIZATION_TELEGRAM_NODE = "Send Direct Market Telegram"
DEFAULT_CREDENTIAL_SOURCE_NODE = "Save ERCOT To GitHub"

_VALIDATION_NODE_ID = "b966fa49-4e35-51f0-a3eb-f2c9a8df4c32"
_PUBLISH_BRANCH_FALLBACK = re.compile(
    r"(?P<prefix>\bbranch\s*:\s*\$json\.branch\s*\|\|\s*)"
    r"(?P<quote>['\"])(?P<value>[^'\"]*)(?P=quote)"
)
_ENV_GITHUB_TOKEN = "$env.GITHUB_TOKEN"
_SCHEDULED_PROFILE = re.compile(
    r"let runProfile = source === 'telegram' \? 'quick' : "
    r"'(?:research|quality|scheduled)';"
)
_SCHEDULED_SEQUENCE_MODEL = re.compile(
    r"(?:let sequenceModel = source === 'telegram' \? 'off' : "
    r"'(?:both|adaptive|off)'|let sequenceModel = 'off');"
)
_SCHEDULED_SHORT_SEQUENCE_MODEL = re.compile(
    r"let shortSequenceModel = (?:"
    r"sequenceModel === 'adaptive' \? 'adaptive' : "
    r"\(sequenceModel === 'both' \? 'both' : 'off'\)"
    r"|source === 'telegram' \? "
    r"\(sequenceModel === 'adaptive' \? 'adaptive' : "
    r"\(sequenceModel === 'both' \? 'both' : 'off'\)\) : 'off'"
    r");"
)
_DEFAULT_RL_SHADOW = re.compile(r"let includeRlPolicy = (?:false|true);")
_RL_COMMAND_FLAG = re.compile(
    r"(?:if \(noRlRequested\) args\.push\('--no-rl-policy'\);\n"
    r"if \(includeRlPolicy\) args\.push\('--include-rl-policy'\);"
    r"|args\.push\(includeRlPolicy \? '--include-rl-policy' : "
    r"'--no-rl-policy'\);)"
)
_SCHEDULED_SHORT_HORIZONS = re.compile(
    r"const shortHorizons = runProfile === 'quick' \|\| "
    r"(?:runProfile === 'scheduled' \|\| )?mainHorizon === 1 \|\| "
    r"noShortHorizon \? '' : '1';"
)
_SCHEDULED_NO_OPTIMIZE = re.compile(
    r"if \((?:runProfile === 'quick'|\(runProfile === 'quick' \|\| "
    r"runProfile === 'scheduled'\)) && !allModels && !retrainRequested\) "
    r"args\.push\('--no-optimize'\);"
)
_CAFFEINATED_RUNNER = re.compile(
    r"&& (?:/usr/bin/caffeinate -i )?\./\.venv/bin/python "
    r"market_agent/daily_ml_forecast_report\.py"
)

SCHEDULED_PROFILE_JS = (
    "let runProfile = source === 'telegram' ? 'quick' : 'scheduled';"
)
SCHEDULED_SEQUENCE_JS = "let sequenceModel = 'off';"
SCHEDULED_SHORT_SEQUENCE_JS = (
    "let shortSequenceModel = source === 'telegram' ? "
    "(sequenceModel === 'adaptive' ? 'adaptive' : "
    "(sequenceModel === 'both' ? 'both' : 'off')) : 'off';"
)
DEFAULT_RL_SHADOW_JS = "let includeRlPolicy = true;"
EXPLICIT_RL_COMMAND_FLAG_JS = (
    "args.push(includeRlPolicy ? '--include-rl-policy' : '--no-rl-policy');"
)
SCHEDULED_SHORT_HORIZONS_JS = (
    "const shortHorizons = runProfile === 'quick' || "
    "runProfile === 'scheduled' || mainHorizon === 1 || "
    "noShortHorizon ? '' : '1';"
)
SCHEDULED_NO_OPTIMIZE_JS = (
    "if ((runProfile === 'quick' || runProfile === 'scheduled') && "
    "!allModels && !retrainRequested) args.push('--no-optimize');"
)
CAFFEINATED_RUNNER_JS = (
    "&& /usr/bin/caffeinate -i ./.venv/bin/python "
    "market_agent/daily_ml_forecast_report.py"
)
TELEGRAM_HTML_TEXT_EXPRESSION = (
    "={{ String($json.text ?? '')"
    ".replace(/&/g, '&amp;')"
    ".replace(/</g, '&lt;')"
    ".replace(/>/g, '&gt;') }}"
)


VALIDATION_JS = r"""const input = $input.first()?.json ?? {};
const rawStdout = typeof input.stdout === 'string' ? input.stdout : '';
const serialized = rawStdout.trim();
const exitCode = input.exitCode ?? input.exit_code ?? null;
if (exitCode !== null && Number(exitCode) !== 0) {
  throw new Error(`Market optimizer contract violation: producer exited with code ${exitCode}.`);
}
if (!serialized) {
  throw new Error('Market optimizer contract violation: stdout is blank.');
}

let report;
try {
  report = JSON.parse(serialized);
} catch (error) {
  throw new Error(`Market optimizer contract violation: stdout is not valid JSON (${error.message}).`);
}
if (!report || typeof report !== 'object' || Array.isArray(report)) {
  throw new Error('Market optimizer contract violation: the root value must be an object.');
}
if (!Array.isArray(report.rows) || report.rows.length === 0) {
  throw new Error('Market optimizer contract violation: rows must be a non-empty array.');
}
if (typeof report.telegram_text !== 'string' || !report.telegram_text.trim()) {
  throw new Error('Market optimizer contract violation: telegram_text must be non-empty.');
}

const textMetadata = (label) => {
  const match = report.telegram_text.match(new RegExp(`^${label}:\\s*(.+)$`, 'mi'));
  return match ? match[1].trim() : null;
};
const metadata = {
  generated: report.generated_at ?? textMetadata('Generated'),
  horizon: report.horizon_days ?? report.horizon ?? textMetadata('Horizon'),
  universe_count: report.universe_count ?? report.universe_size ?? report.rows.length,
  universe: report.universe ?? textMetadata('Universe'),
};
for (const [field, value] of Object.entries(metadata)) {
  if (value === null || value === undefined || String(value).trim() === '') {
    throw new Error(`Market optimizer contract violation: ${field} metadata is missing.`);
  }
}

return [{
  json: {
    optimizer_payload: report,
    optimizer_metadata: metadata,
    execution_metadata: {
      exit_code: input.exitCode ?? input.exit_code ?? null,
    },
  },
}];"""


RAW_PAYLOAD_JS = r"""const data = $json.optimizer_payload;
const metadata = $json.optimizer_metadata ?? {};
if (!data || typeof data !== 'object' || Array.isArray(data)) {
  throw new Error('Validated optimizer_payload is unavailable.');
}
data.short_horizon_reports = data.short_horizon_reports || [];
const timestamp = String(data.generated_at || metadata.generated || new Date().toISOString()).replace(/[:.]/g, '-');
const jsonText = JSON.stringify(data, null, 2);
return {
  json: {
    filename: `market_agent/reports/ml_forecast_rankings_${timestamp}.json`,
    content: Buffer.from(jsonText, 'utf8').toString('base64'),
    message: `Raw ML Forecast Rankings JSON - ${timestamp}`,
    branch: 'main',
    short_horizon_count: data.short_horizon_reports.length
  }
};"""


TEXT_PAYLOAD_JS = r"""const data = $json.optimizer_payload;
if (!data || typeof data !== 'object' || Array.isArray(data)) {
  throw new Error('Validated optimizer_payload is unavailable.');
}
const websiteText = typeof data.telegram_text === 'string' ? data.telegram_text.trim() : '';
if (!websiteText) {
  throw new Error('Validated optimizer report text is blank.');
}
const timestamp = String(data.generated_at || new Date().toISOString()).replace(/[:.]/g, '-');
return {
  json: {
    filename: `market_agent/reports/optimization_summaries/ml_forecast_rankings_${timestamp}.txt`,
    content: Buffer.from(websiteText, 'utf8').toString('base64'),
    message: `Validated Market Optimization Report - ${timestamp}`,
    branch: 'main'
  }
};"""


DETERMINISTIC_PUBLICATION_JS = r"""const data = $json.optimizer_payload;
if (!data || typeof data !== 'object' || Array.isArray(data)) {
  throw new Error('Validated optimizer_payload is unavailable.');
}
const telegramText = typeof data.telegram_text === 'string'
  ? data.telegram_text.trim()
  : '';
if (!telegramText) {
  throw new Error('Validated optimizer telegram_text is blank.');
}
const cleanList = (value) => Array.isArray(value)
  ? [...new Set(value.map((item) => String(item).trim()).filter(Boolean))]
  : [];
const publication = {
  website_recommendations: telegramText,
  telegram_recommendations: telegramText,
  telegram_text: telegramText,
  top_buys: cleanList(data.top_buys),
  top_sells: cleanList(data.top_sells),
};
return {
  json: {
    message: {
      content: JSON.stringify(publication),
    },
  },
};"""


def _named_node(nodes: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [node for node in nodes if node.get("name") == name]
    if len(matches) != 1:
        raise ValueError(
            f"Workflow must contain exactly one node named {name!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _first_main_output(
    connections: dict[str, Any], source: str
) -> list[dict[str, Any]]:
    source_connections = connections.setdefault(source, {})
    if not isinstance(source_connections, dict):
        raise ValueError(f"Connections for {source!r} must be an object")
    outputs = source_connections.setdefault("main", [[]])
    if not isinstance(outputs, list):
        raise ValueError(f"Main outputs for {source!r} must be an array")
    if not outputs:
        outputs.append([])
    if not isinstance(outputs[0], list):
        raise ValueError(f"First main output for {source!r} must be an array")
    return outputs[0]


def _connection(target: str) -> dict[str, Any]:
    return {"node": target, "type": "main", "index": 0}


def _append_connection(
    output: list[dict[str, Any]], descriptor: Mapping[str, Any]
) -> bool:
    target = descriptor.get("node")
    if any(item.get("node") == target for item in output):
        return False
    output.append(copy.deepcopy(dict(descriptor)))
    return True


def _contains_env_github_token(value: Any) -> bool:
    if isinstance(value, str):
        return _ENV_GITHUB_TOKEN in value
    if isinstance(value, Mapping):
        return any(_contains_env_github_token(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_env_github_token(item) for item in value)
    return False


def _repair_scheduled_profile(node: dict[str, Any]) -> tuple[str, ...]:
    parameters = node.setdefault("parameters", {})
    javascript = parameters.get("jsCode")
    if not isinstance(javascript, str):
        raise ValueError(f"{REQUEST_CONTEXT_NODE} must define JavaScript")

    changes: list[str] = []
    updated, profile_count = _SCHEDULED_PROFILE.subn(
        SCHEDULED_PROFILE_JS,
        javascript,
    )
    if updated != javascript:
        changes.append("use the bounded single-pass profile for scheduled optimization")
    previous = updated
    updated, sequence_count = _SCHEDULED_SEQUENCE_MODEL.subn(
        SCHEDULED_SEQUENCE_JS,
        updated,
    )
    if updated != previous:
        changes.append("disable sequence models for routine scheduled optimization")
    previous = updated
    updated, short_sequence_count = _SCHEDULED_SHORT_SEQUENCE_MODEL.subn(
        SCHEDULED_SHORT_SEQUENCE_JS,
        updated,
    )
    if updated != previous:
        changes.append(
            "disable cross-horizon sequence selection in scheduled 1-day optimization"
        )
    previous = updated
    updated, rl_count = _DEFAULT_RL_SHADOW.subn(
        DEFAULT_RL_SHADOW_JS,
        updated,
    )
    if updated != previous:
        changes.append("keep RL diagnostics shadow-enabled by default")
    previous = updated
    updated, rl_flag_count = _RL_COMMAND_FLAG.subn(
        EXPLICIT_RL_COMMAND_FLAG_JS,
        updated,
    )
    if updated != previous:
        changes.append("make the optimizer RL command flag match workflow metadata")
    previous = updated
    updated, short_horizons_count = _SCHEDULED_SHORT_HORIZONS.subn(
        SCHEDULED_SHORT_HORIZONS_JS,
        updated,
    )
    if updated != previous:
        changes.append("disable the duplicate 1-day pass for routine scheduled runs")
    previous = updated
    updated, no_optimize_count = _SCHEDULED_NO_OPTIMIZE.subn(
        SCHEDULED_NO_OPTIMIZE_JS,
        updated,
    )
    if updated != previous:
        changes.append("disable nested hyperparameter search for routine scheduled runs")
    previous = updated
    updated, caffeinate_count = _CAFFEINATED_RUNNER.subn(
        CAFFEINATED_RUNNER_JS,
        updated,
    )
    if updated != previous:
        changes.append("keep the optimizer awake while its process is running")
    if (
        profile_count,
        sequence_count,
        short_sequence_count,
        rl_count,
        rl_flag_count,
        short_horizons_count,
        no_optimize_count,
        caffeinate_count,
    ) != (1, 1, 1, 1, 1, 1, 1, 1):
        raise ValueError(
            f"{REQUEST_CONTEXT_NODE} does not expose the expected profile controls"
        )
    if not changes:
        return ()
    parameters["jsCode"] = updated
    return tuple(changes)


def _configure_telegram_delivery(node: dict[str, Any]) -> tuple[str, ...]:
    """Make generated plain text safe and non-blocking for Telegram."""

    changes: list[str] = []
    parameters = node.setdefault("parameters", {})
    if parameters.get("text") != TELEGRAM_HTML_TEXT_EXPRESSION:
        parameters["text"] = TELEGRAM_HTML_TEXT_EXPRESSION
        changes.append(f"HTML-escape Telegram text in {node.get('name')}")
    additional_fields = parameters.setdefault("additionalFields", {})
    if not isinstance(additional_fields, dict):
        raise ValueError(f"{node.get('name')} additionalFields must be an object")
    if additional_fields.get("parse_mode") != "HTML":
        additional_fields["parse_mode"] = "HTML"
        changes.append(f"set HTML parse mode in {node.get('name')}")
    if node.get("onError") != "continueErrorOutput":
        node["onError"] = "continueErrorOutput"
        changes.append(f"make {node.get('name')} non-blocking")
    return tuple(changes)


def _configure_publisher_credential(
    publisher: dict[str, Any], credential_source: dict[str, Any]
) -> tuple[str, ...]:
    source_credentials = credential_source.get("credentials")
    if not isinstance(source_credentials, Mapping):
        raise ValueError("Credential source node has no credential binding")
    header_binding = source_credentials.get("httpHeaderAuth")
    if not isinstance(header_binding, Mapping) or not header_binding:
        raise ValueError(
            "Credential source node has no httpHeaderAuth credential binding"
        )

    changes: list[str] = []
    target_credentials = publisher.setdefault("credentials", {})
    if not isinstance(target_credentials, dict):
        raise ValueError("Optimization publisher credentials must be an object")
    if target_credentials.get("httpHeaderAuth") != header_binding:
        target_credentials["httpHeaderAuth"] = copy.deepcopy(dict(header_binding))
        changes.append("bind optimization publisher HTTP header credential")

    parameters = publisher.setdefault("parameters", {})
    if parameters.get("authentication") != "genericCredentialType":
        parameters["authentication"] = "genericCredentialType"
        changes.append("enable generic credential authentication")
    if parameters.get("genericAuthType") != "httpHeaderAuth":
        parameters["genericAuthType"] = "httpHeaderAuth"
        changes.append("select HTTP header authentication")

    json_body = parameters.get("jsonBody")
    if not isinstance(json_body, str):
        raise ValueError("Optimization publisher must define a JSON body expression")
    updated_json_body, fallback_replacements = _PUBLISH_BRANCH_FALLBACK.subn(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}main"
            f"{match.group('quote')}"
        ),
        json_body,
    )
    if fallback_replacements == 0:
        raise ValueError("Optimization publisher JSON body has no branch fallback")
    if updated_json_body != json_body:
        parameters["jsonBody"] = updated_json_body
        changes.append("set optimization publisher fallback branch to main")

    header_parameters = parameters.get("headerParameters")
    if isinstance(header_parameters, dict):
        headers = header_parameters.get("parameters")
        if isinstance(headers, list):
            filtered = [
                header
                for header in headers
                if str(header.get("name", "")).casefold() != "authorization"
            ]
            if filtered != headers:
                header_parameters["parameters"] = filtered
                changes.append("remove environment Authorization header")

    if _contains_env_github_token(parameters):
        raise ValueError(
            "Optimization publisher still depends on the GitHub token environment variable"
        )
    return tuple(changes)


def _configure_deterministic_publication_adapter(
    model_node: dict[str, Any],
) -> tuple[str, ...]:
    """Replace optimization prose generation with validated producer output."""

    changes: list[str] = []
    expected_parameters = {
        "mode": "runOnceForEachItem",
        "jsCode": DETERMINISTIC_PUBLICATION_JS,
    }
    if model_node.get("parameters") != expected_parameters:
        model_node["parameters"] = expected_parameters
        changes.append("publish the deterministic validated optimizer text")
    if model_node.get("type") != "n8n-nodes-base.code":
        model_node["type"] = "n8n-nodes-base.code"
        changes.append("replace the optimization language model with a Code adapter")
    if model_node.get("typeVersion") != 2:
        model_node["typeVersion"] = 2
        changes.append("set the deterministic publication adapter version")
    if model_node.pop("credentials", None) is not None:
        changes.append("remove unused language-model credentials")
    retry_fields = ("retryOnFail", "maxTries", "waitBetweenTries")
    if any(field in model_node for field in retry_fields):
        for field in retry_fields:
            model_node.pop(field, None)
        changes.append("remove obsolete language-model retry settings")
    return tuple(changes)


def repair_market_optimization_workflow(
    workflow: Mapping[str, Any],
    *,
    credential_source_node: str = DEFAULT_CREDENTIAL_SOURCE_NODE,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return an idempotently repaired workflow and a change description.

    Only named optimization nodes, their graph edges, and the target publisher's
    authentication binding are changed. Credential values are never included in
    errors or change descriptions.
    """

    repaired = copy.deepcopy(dict(workflow))
    nodes = repaired.get("nodes")
    connections = repaired.get("connections")
    if not isinstance(nodes, list) or not isinstance(connections, dict):
        raise ValueError("Workflow must contain node and connection collections")

    changes: list[str] = []
    run_node = _named_node(nodes, RUN_NODE)
    request_context = _named_node(nodes, REQUEST_CONTEXT_NODE)
    model_node = _named_node(nodes, MODEL_NODE)
    raw_builder = _named_node(nodes, RAW_PAYLOAD_NODE)
    text_builder = _named_node(nodes, TEXT_PAYLOAD_NODE)
    publisher = _named_node(nodes, PUBLISH_NODE)
    credential_source = _named_node(nodes, credential_source_node)
    optimization_telegram = _named_node(nodes, OPTIMIZATION_TELEGRAM_NODE)
    direct_optimization_telegram = _named_node(
        nodes,
        DIRECT_OPTIMIZATION_TELEGRAM_NODE,
    )

    changes.extend(_repair_scheduled_profile(request_context))
    changes.extend(_configure_telegram_delivery(optimization_telegram))
    changes.extend(_configure_telegram_delivery(direct_optimization_telegram))

    validation_matches = [
        node for node in nodes if node.get("name") == VALIDATION_NODE
    ]
    if len(validation_matches) > 1:
        raise ValueError(f"Workflow contains duplicate {VALIDATION_NODE!r} nodes")
    if validation_matches:
        validation_node = validation_matches[0]
    else:
        position = run_node.get("position")
        if (
            isinstance(position, list)
            and len(position) == 2
            and all(isinstance(value, (int, float)) for value in position)
        ):
            validator_position = [position[0] + 220, position[1]]
        else:
            validator_position = [0, 0]
        validation_node = {
            "parameters": {"mode": "runOnceForAllItems", "jsCode": VALIDATION_JS},
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": validator_position,
            "id": _VALIDATION_NODE_ID,
            "name": VALIDATION_NODE,
        }
        nodes.append(validation_node)
        changes.append("add fail-closed optimizer output validator")

    validation_parameters = validation_node.setdefault("parameters", {})
    expected_validation_parameters = {
        "mode": "runOnceForAllItems",
        "jsCode": VALIDATION_JS,
    }
    if any(
        validation_parameters.get(key) != value
        for key, value in expected_validation_parameters.items()
    ):
        validation_parameters.update(expected_validation_parameters)
        changes.append("update optimizer validation contract")
    if validation_node.get("type") != "n8n-nodes-base.code":
        validation_node["type"] = "n8n-nodes-base.code"
        changes.append("set optimizer validator node type")
    if validation_node.get("typeVersion") != 2:
        validation_node["typeVersion"] = 2
        changes.append("set optimizer validator node version")

    run_output = _first_main_output(connections, RUN_NODE)
    validator_output = _first_main_output(connections, VALIDATION_NODE)
    for descriptor in list(run_output):
        if descriptor.get("node") != VALIDATION_NODE:
            if _append_connection(validator_output, descriptor):
                changes.append("move optimizer consumer behind validation")
    canonical_run_output = [_connection(VALIDATION_NODE)]
    if run_output != canonical_run_output:
        run_output[:] = canonical_run_output
        changes.append("route optimizer command through validation")

    if _append_connection(validator_output, _connection(MODEL_NODE)):
        changes.append("connect validated output to publication adapter")
    if _append_connection(validator_output, _connection(RAW_PAYLOAD_NODE)):
        changes.append("connect validated output to raw JSON publisher")
    if _append_connection(validator_output, _connection(TEXT_PAYLOAD_NODE)):
        changes.append("connect validated output to text publisher")

    model_output = _first_main_output(connections, MODEL_NODE)
    filtered_model_output = [
        descriptor
        for descriptor in model_output
        if descriptor.get("node") != TEXT_PAYLOAD_NODE
    ]
    if filtered_model_output != model_output:
        model_output[:] = filtered_model_output
        changes.append("disconnect publication adapter from duplicate text publisher")
    raw_payload_output = _first_main_output(connections, RAW_PAYLOAD_NODE)
    if _append_connection(raw_payload_output, _connection(PUBLISH_NODE)):
        changes.append("connect raw optimization payload to GitHub publisher")
    text_payload_output = _first_main_output(connections, TEXT_PAYLOAD_NODE)
    if _append_connection(text_payload_output, _connection(PUBLISH_NODE)):
        changes.append("connect text optimization payload to GitHub publisher")

    if raw_builder.setdefault("parameters", {}).get("jsCode") != RAW_PAYLOAD_JS:
        raw_builder["parameters"]["jsCode"] = RAW_PAYLOAD_JS
        changes.append("consume validated optimizer object in raw JSON publisher")
    if text_builder.setdefault("parameters", {}).get("jsCode") != TEXT_PAYLOAD_JS:
        text_builder["parameters"]["jsCode"] = TEXT_PAYLOAD_JS
        changes.append("publish deterministic validated optimization text")

    changes.extend(_configure_deterministic_publication_adapter(model_node))
    changes.extend(_configure_publisher_credential(publisher, credential_source))

    issues = audit_market_optimization_workflow(
        repaired, credential_source_node=credential_source_node
    )
    if issues:
        raise ValueError("Workflow repair did not satisfy contract: " + "; ".join(issues))
    return repaired, tuple(changes)


def audit_market_optimization_workflow(
    workflow: Mapping[str, Any],
    *,
    credential_source_node: str = DEFAULT_CREDENTIAL_SOURCE_NODE,
) -> tuple[str, ...]:
    """Return contract violations without exposing workflow parameter values."""

    nodes = workflow.get("nodes")
    connections = workflow.get("connections")
    if not isinstance(nodes, list) or not isinstance(connections, dict):
        return ("workflow node or connection collection is malformed",)

    issues: list[str] = []
    required_names = (
        RUN_NODE,
        REQUEST_CONTEXT_NODE,
        VALIDATION_NODE,
        MODEL_NODE,
        RAW_PAYLOAD_NODE,
        TEXT_PAYLOAD_NODE,
        PUBLISH_NODE,
        OPTIMIZATION_TELEGRAM_NODE,
        DIRECT_OPTIMIZATION_TELEGRAM_NODE,
        credential_source_node,
    )
    named: dict[str, dict[str, Any]] = {}
    for name in required_names:
        matches = [node for node in nodes if node.get("name") == name]
        if len(matches) != 1:
            issues.append(f"expected exactly one {name!r} node")
        else:
            named[name] = matches[0]
    if issues:
        return tuple(issues)

    validator = named[VALIDATION_NODE]
    if validator.get("type") != "n8n-nodes-base.code":
        issues.append("optimizer validator is not a Code node")
    if validator.get("parameters", {}).get("jsCode") != VALIDATION_JS:
        issues.append("optimizer validator does not implement the output contract")

    request_code = named[REQUEST_CONTEXT_NODE].get("parameters", {}).get("jsCode")
    if not isinstance(request_code, str):
        issues.append("market request context has no JavaScript")
    else:
        for expected in (
            SCHEDULED_PROFILE_JS,
            SCHEDULED_SEQUENCE_JS,
            SCHEDULED_SHORT_SEQUENCE_JS,
            DEFAULT_RL_SHADOW_JS,
            EXPLICIT_RL_COMMAND_FLAG_JS,
            SCHEDULED_SHORT_HORIZONS_JS,
            SCHEDULED_NO_OPTIMIZE_JS,
            CAFFEINATED_RUNNER_JS,
        ):
            if expected not in request_code:
                issues.append("scheduled optimization profile is not bounded")
                break

    for telegram_node_name in (
        OPTIMIZATION_TELEGRAM_NODE,
        DIRECT_OPTIMIZATION_TELEGRAM_NODE,
    ):
        telegram_node = named[telegram_node_name]
        telegram_parameters = telegram_node.get("parameters", {})
        additional_fields = telegram_parameters.get("additionalFields", {})
        if telegram_parameters.get("text") != TELEGRAM_HTML_TEXT_EXPRESSION:
            issues.append(f"{telegram_node_name} does not HTML-escape text")
        if not isinstance(additional_fields, Mapping) or (
            additional_fields.get("parse_mode") != "HTML"
        ):
            issues.append(f"{telegram_node_name} does not use HTML parse mode")
        if telegram_node.get("onError") != "continueErrorOutput":
            issues.append(f"{telegram_node_name} can block report publication")

    def target_names(source: str) -> list[str]:
        source_connections = connections.get(source)
        outputs = (
            source_connections.get("main")
            if isinstance(source_connections, Mapping)
            else None
        )
        if (
            not isinstance(outputs, list)
            or not outputs
            or not isinstance(outputs[0], list)
        ):
            issues.append(f"connection output for {source!r} is malformed")
            return []
        return [str(item.get("node", "")) for item in outputs[0]]

    if target_names(RUN_NODE) != [VALIDATION_NODE]:
        issues.append("optimizer command does not route exclusively through validation")
    validator_targets = target_names(VALIDATION_NODE)
    for required_target in (MODEL_NODE, RAW_PAYLOAD_NODE):
        if required_target not in validator_targets:
            issues.append(f"validated output is not connected to {required_target!r}")
    if TEXT_PAYLOAD_NODE in target_names(MODEL_NODE):
        issues.append("optimization model remains connected to the text publisher")
    if TEXT_PAYLOAD_NODE not in validator_targets:
        issues.append("validated output is not connected to the text publisher")
    if PUBLISH_NODE not in target_names(RAW_PAYLOAD_NODE):
        issues.append("raw optimization payload is not connected to its publisher")
    if PUBLISH_NODE not in target_names(TEXT_PAYLOAD_NODE):
        issues.append("text optimization payload is not connected to its publisher")

    raw_code = named[RAW_PAYLOAD_NODE].get("parameters", {}).get("jsCode")
    if raw_code != RAW_PAYLOAD_JS:
        issues.append("raw JSON publisher does not consume optimizer_payload")
    text_code = named[TEXT_PAYLOAD_NODE].get("parameters", {}).get("jsCode")
    if text_code != TEXT_PAYLOAD_JS:
        issues.append("text publisher does not consume validated optimizer text")

    publication_adapter = named[MODEL_NODE]
    if publication_adapter.get("type") != "n8n-nodes-base.code":
        issues.append("optimization publication adapter is not a Code node")
    if publication_adapter.get("typeVersion") != 2:
        issues.append("optimization publication adapter has the wrong version")
    expected_publication_parameters = {
        "mode": "runOnceForEachItem",
        "jsCode": DETERMINISTIC_PUBLICATION_JS,
    }
    if publication_adapter.get("parameters") != expected_publication_parameters:
        issues.append("optimization publication does not use validated producer text")
    if publication_adapter.get("credentials"):
        issues.append("optimization publication adapter retains unused credentials")
    if any(
        field in publication_adapter
        for field in ("retryOnFail", "maxTries", "waitBetweenTries")
    ):
        issues.append("optimization publication adapter retains obsolete retries")

    publisher = named[PUBLISH_NODE]
    source = named[credential_source_node]
    publisher_parameters = publisher.get("parameters", {})
    if _contains_env_github_token(publisher_parameters):
        issues.append("optimization publisher depends on an environment token")
    source_credentials = source.get("credentials", {})
    expected_binding = (
        source_credentials.get("httpHeaderAuth")
        if isinstance(source_credentials, Mapping)
        else None
    )
    publisher_credentials = publisher.get("credentials", {})
    actual_binding = (
        publisher_credentials.get("httpHeaderAuth")
        if isinstance(publisher_credentials, Mapping)
        else None
    )
    if not isinstance(expected_binding, Mapping) or not expected_binding:
        issues.append("credential source has no HTTP header credential binding")
    elif actual_binding != expected_binding:
        issues.append("optimization publisher credential binding is missing or stale")
    if publisher_parameters.get("authentication") != "genericCredentialType":
        issues.append("optimization publisher does not use generic credentials")
    if publisher_parameters.get("genericAuthType") != "httpHeaderAuth":
        issues.append("optimization publisher does not use HTTP header authentication")
    json_body = publisher_parameters.get("jsonBody")
    fallback = (
        _PUBLISH_BRANCH_FALLBACK.search(json_body)
        if isinstance(json_body, str)
        else None
    )
    if fallback is None or fallback.group("value") != "main":
        issues.append("optimization publisher fallback branch is not main")
    if isinstance(json_body, str) and "generated-output" in json_body:
        issues.append("optimization publisher retains generated-output routing")
    headers = publisher_parameters.get("headerParameters", {})
    header_items = headers.get("parameters") if isinstance(headers, dict) else None
    if isinstance(header_items, list) and any(
        str(item.get("name", "")).casefold() == "authorization"
        for item in header_items
    ):
        issues.append("optimization publisher retains a manual Authorization header")

    return tuple(issues)
