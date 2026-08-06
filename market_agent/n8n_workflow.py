"""Repair and audit the market-optimization branch of an n8n workflow."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any


RUN_NODE = "Run Optimization Scripts"
VALIDATION_NODE = "Validate Optimization Output"
MODEL_NODE = "Message Model For Optimization"
RAW_PAYLOAD_NODE = "Build Optimization Raw JSON GitHub Payload"
TEXT_PAYLOAD_NODE = "Build Optimization GitHub Payload"
PUBLISH_NODE = "Save Optimization To GitHub"
DEFAULT_CREDENTIAL_SOURCE_NODE = "Save ERCOT To GitHub"

_VALIDATION_NODE_ID = "b966fa49-4e35-51f0-a3eb-f2c9a8df4c32"
_PUBLISH_BRANCH_FALLBACK = re.compile(
    r"(?P<prefix>\bbranch\s*:\s*\$json\.branch\s*\|\|\s*)"
    r"(?P<quote>['\"])(?P<value>[^'\"]*)(?P=quote)"
)
_ENV_GITHUB_TOKEN = "$env.GITHUB_TOKEN"


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


MODEL_USER_EXPRESSION = r"""={{ (() => {
  const data = $json.optimizer_payload;
  const threshold = Number(data.signal_threshold?.min_forecast_return_pct ?? 2);
  const value = (row, ...keys) => {
    for (const key of keys) if (row[key] !== undefined && row[key] !== null) return row[key];
    return null;
  };
  const qualified = (rows) => (Array.isArray(rows) ? rows : [])
    .filter((row) => String(value(row, 'Signal Tier') ?? '').startsWith('Model-Confirmed'))
    .filter((row) => String(value(row, 'Reliability') ?? '') !== 'Low')
    .filter((row) => Math.abs(Number(value(row, 'Forecast Return %', 'forecast_return_pct') ?? 0)) >= threshold)
    .filter((row) => String(value(row, 'Selected Model', 'selected_model') ?? '') !== 'RL Policy')
    .map((row) => ({
      symbol: value(row, 'Symbol', 'symbol'),
      model_call: value(row, 'Model Call', 'model_call'),
      signal_tier: value(row, 'Signal Tier'),
      forecast_return_pct: value(row, 'Forecast Return %', 'forecast_return_pct'),
      probability_up_pct: value(row, 'Probability Up %'),
      as_of_session: value(row, 'As Of Session'),
      target_session: value(row, 'Target Session'),
      reliability: value(row, 'Reliability'),
      uncertainty_pct: value(row, 'Expected Error %', 'expected_error_pct'),
      selected_model: value(row, 'Selected Model', 'selected_model'),
      indicative_target_pct: value(row, 'Pre-Portfolio Target %'),
      executable_target_pct: value(row, 'Policy Target %'),
      allocation_blocked: Boolean(value(row, 'Portfolio Allocation Blocked')),
    }));
  const qualifiedRows = qualified(data.rows);
  const shortHorizonReports = (Array.isArray(data.short_horizon_reports) ? data.short_horizon_reports : [])
    .map((report) => ({
      horizon_days: report.horizon_days,
      sequence_model: report.sequence_model,
      qualified_rows: qualified(report.rows),
    }));
  return JSON.stringify({
    generated: $json.optimizer_metadata.generated,
    horizon: $json.optimizer_metadata.horizon,
    universe_count: $json.optimizer_metadata.universe_count,
    universe: $json.optimizer_metadata.universe,
    horizon_meaning: 'Point-to-point through target_session; not an immediate or monotonic price path.',
    execution_meaning: 'Indicative targets are research outputs; executable targets require portfolio authorization.',
    threshold_pct: threshold,
    qualified_rows: qualifiedRows,
    short_horizon_reports: shortHorizonReports,
  });
})() }}"""


SYSTEM_SAFETY_SUFFIX = (
    "\n\nOptimization safety contract: Treat every forecast as point-to-point "
    "through target_session, not as an immediate or monotonic path. Summarize "
    "only the supplied qualified_rows and short-horizon qualified_rows; RL "
    "Policy is shadow-only. Clearly "
    "separate indicative_target_pct from executable_target_pct, and never "
    "describe a blocked allocation as authorized. Preserve as_of_session so "
    "the generation timestamp is not mistaken for the market-data cutoff."
)
_SYSTEM_SAFETY_MARKER = "\n\nOptimization safety contract:"


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


def _repair_model_messages(model_node: dict[str, Any]) -> tuple[str, ...]:
    parameters = model_node.setdefault("parameters", {})
    messages = parameters.get("messages")
    values = messages.get("values") if isinstance(messages, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"{MODEL_NODE} must define messages.values")

    system_message = next(
        (message for message in values if message.get("role") == "system"),
        None,
    )
    user_message = next(
        (message for message in values if message.get("role") != "system"),
        None,
    )
    if system_message is None or user_message is None:
        raise ValueError(f"{MODEL_NODE} must contain system and user messages")

    changes: list[str] = []
    system_content = str(system_message.get("content", ""))
    base_system_content = system_content.split(
        _SYSTEM_SAFETY_MARKER,
        1,
    )[0].rstrip()
    canonical_system_content = base_system_content + SYSTEM_SAFETY_SUFFIX
    if system_content != canonical_system_content:
        system_message["content"] = canonical_system_content
        changes.append("set optimization model safety contract")
    if user_message.get("content") != MODEL_USER_EXPRESSION:
        user_message["content"] = MODEL_USER_EXPRESSION
        changes.append("use validated qualified rows for optimization model")
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
    model_node = _named_node(nodes, MODEL_NODE)
    raw_builder = _named_node(nodes, RAW_PAYLOAD_NODE)
    text_builder = _named_node(nodes, TEXT_PAYLOAD_NODE)
    publisher = _named_node(nodes, PUBLISH_NODE)
    credential_source = _named_node(nodes, credential_source_node)

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
        changes.append("connect validated output to optimization model")
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
        changes.append("disconnect LLM output from deterministic text publisher")
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

    changes.extend(_repair_model_messages(model_node))
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
        VALIDATION_NODE,
        MODEL_NODE,
        RAW_PAYLOAD_NODE,
        TEXT_PAYLOAD_NODE,
        PUBLISH_NODE,
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

    model_parameters = named[MODEL_NODE].get("parameters", {})
    messages = model_parameters.get("messages", {})
    values = messages.get("values") if isinstance(messages, dict) else None
    if not isinstance(values, list):
        issues.append("optimization model messages are malformed")
    else:
        systems = [item for item in values if item.get("role") == "system"]
        users = [item for item in values if item.get("role") != "system"]
        if not systems or SYSTEM_SAFETY_SUFFIX.strip() not in str(
            systems[0].get("content", "")
        ):
            issues.append("optimization model safety contract is missing")
        if not users or users[0].get("content") != MODEL_USER_EXPRESSION:
            issues.append("optimization model does not consume qualified optimizer rows")

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
