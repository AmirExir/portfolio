from __future__ import annotations

import copy
import unittest

from market_agent.n8n_workflow import (
    DEFAULT_CREDENTIAL_SOURCE_NODE,
    DEFAULT_RL_SHADOW_JS,
    DIRECT_OPTIMIZATION_TELEGRAM_NODE,
    EXPLICIT_RL_COMMAND_FLAG_JS,
    MODEL_NODE,
    MODEL_USER_EXPRESSION,
    OPTIMIZATION_TELEGRAM_NODE,
    PUBLISH_NODE,
    RAW_PAYLOAD_JS,
    RAW_PAYLOAD_NODE,
    REQUEST_CONTEXT_NODE,
    RUN_NODE,
    SCHEDULED_PROFILE_JS,
    SCHEDULED_SHORT_SEQUENCE_JS,
    SCHEDULED_SEQUENCE_JS,
    SYSTEM_SAFETY_SUFFIX,
    TEXT_PAYLOAD_JS,
    TEXT_PAYLOAD_NODE,
    TELEGRAM_HTML_TEXT_EXPRESSION,
    VALIDATION_JS,
    VALIDATION_NODE,
    audit_market_optimization_workflow,
    repair_market_optimization_workflow,
)


def _node(name: str, *, parameters: dict | None = None) -> dict:
    return {
        "name": name,
        "parameters": parameters or {},
        "type": "test",
        "typeVersion": 1,
        "position": [100, 200],
    }


def _workflow_fixture() -> dict:
    system_prompt = "Summarize the supplied market-optimization report."
    return {
        "nodes": [
            _node(
                REQUEST_CONTEXT_NODE,
                parameters={
                    "jsCode": (
                        "let runProfile = source === 'telegram' ? 'quick' : 'research';\n"
                        "let sequenceModel = source === 'telegram' ? 'off' : 'both';\n"
                        "let includeRlPolicy = false;\n"
                        "let shortSequenceModel = sequenceModel === 'adaptive' ? "
                        "'adaptive' : (sequenceModel === 'both' ? 'both' : 'off');\n"
                        "if (noRlRequested) {\n"
                        "  includeRlPolicy = false;\n"
                        "}\n"
                        "if (noRlRequested) args.push('--no-rl-policy');\n"
                        "if (includeRlPolicy) args.push('--include-rl-policy');"
                    )
                },
            ),
            _node(RUN_NODE),
            _node(
                MODEL_NODE,
                parameters={
                    "messages": {
                        "values": [
                            {"role": "system", "content": system_prompt},
                            {"content": "={{ JSON.parse($json.stdout) }}"},
                        ]
                    }
                },
            ),
            _node(
                RAW_PAYLOAD_NODE,
                parameters={
                    "jsCode": (
                        "const data = JSON.parse($json.stdout); "
                        "return {json: {branch: 'generated-output'}};"
                    )
                },
            ),
            _node(
                TEXT_PAYLOAD_NODE,
                parameters={
                    "jsCode": (
                        "return {json: {branch: \"generated-output\", "
                        "content: $json.content}};"
                    )
                },
            ),
            {
                **_node(
                    PUBLISH_NODE,
                    parameters={
                        "sendHeaders": True,
                        "jsonBody": (
                            "={{ {message: $json.message, content: $json.content, "
                            "branch: $json.branch || 'generated-output'} }}"
                        ),
                        "headerParameters": {
                            "parameters": [
                                {
                                    "name": "Authorization",
                                    "value": "={{ 'Bearer ' + $env.GITHUB_TOKEN }}",
                                },
                                {"name": "Accept", "value": "application/json"},
                            ]
                        },
                    },
                ),
            },
            {
                **_node(DEFAULT_CREDENTIAL_SOURCE_NODE),
                "credentials": {
                    "httpHeaderAuth": {
                        "id": "test-header-credential",
                        "name": "Test GitHub Header",
                    }
                },
            },
            _node(
                OPTIMIZATION_TELEGRAM_NODE,
                parameters={
                    "text": "={{ $json.text }}",
                    "additionalFields": {
                        "appendAttribution": False,
                    },
                },
            ),
            _node(
                DIRECT_OPTIMIZATION_TELEGRAM_NODE,
                parameters={
                    "text": "={{ $json.text }}",
                    "additionalFields": {
                        "appendAttribution": False,
                    },
                },
            ),
        ],
        "connections": {
            RUN_NODE: {
                "main": [
                    [
                        {"node": MODEL_NODE, "type": "main", "index": 0},
                        {"node": "Existing Audit", "type": "main", "index": 0},
                    ]
                ]
            },
            MODEL_NODE: {
                "main": [
                    [
                        {"node": "Telegram", "type": "main", "index": 0},
                        {"node": "Calendar", "type": "main", "index": 0},
                    ]
                ]
            },
            RAW_PAYLOAD_NODE: {"main": [[]]},
            TEXT_PAYLOAD_NODE: {"main": [[]]},
        },
    }


def _targets(workflow: dict, source: str) -> list[str]:
    return [
        item["node"]
        for item in workflow["connections"][source]["main"][0]
    ]


class MarketOptimizationWorkflowTests(unittest.TestCase):
    def test_repair_bounds_schedule_and_hardens_telegram_delivery(self) -> None:
        repaired, changes = repair_market_optimization_workflow(
            _workflow_fixture()
        )

        self.assertTrue(changes)
        request_context = next(
            node
            for node in repaired["nodes"]
            if node["name"] == REQUEST_CONTEXT_NODE
        )
        request_code = request_context["parameters"]["jsCode"]
        self.assertIn(SCHEDULED_PROFILE_JS, request_code)
        self.assertIn(SCHEDULED_SEQUENCE_JS, request_code)
        self.assertIn(SCHEDULED_SHORT_SEQUENCE_JS, request_code)
        self.assertIn(DEFAULT_RL_SHADOW_JS, request_code)
        self.assertIn(EXPLICIT_RL_COMMAND_FLAG_JS, request_code)
        self.assertIn(
            "if (noRlRequested) {\n  includeRlPolicy = false;\n}",
            request_code,
        )
        self.assertNotIn("if (noRlRequested) args.push", request_code)
        self.assertNotIn("? 'quick' : 'research'", request_code)

        for name in (
            OPTIMIZATION_TELEGRAM_NODE,
            DIRECT_OPTIMIZATION_TELEGRAM_NODE,
        ):
            node = next(
                item for item in repaired["nodes"] if item["name"] == name
            )
            self.assertEqual(
                node["parameters"]["text"],
                TELEGRAM_HTML_TEXT_EXPRESSION,
            )
            self.assertEqual(
                node["parameters"]["additionalFields"]["parse_mode"],
                "HTML",
            )
            self.assertEqual(node["onError"], "continueErrorOutput")

    def test_repair_adds_fail_closed_guard_and_restores_publishers(self) -> None:
        original = _workflow_fixture()

        repaired, changes = repair_market_optimization_workflow(original)

        self.assertTrue(changes)
        self.assertNotIn(VALIDATION_NODE, [node["name"] for node in original["nodes"]])
        self.assertEqual(_targets(repaired, RUN_NODE), [VALIDATION_NODE])
        self.assertCountEqual(
            _targets(repaired, VALIDATION_NODE),
            [
                MODEL_NODE,
                "Existing Audit",
                RAW_PAYLOAD_NODE,
                TEXT_PAYLOAD_NODE,
            ],
        )
        self.assertCountEqual(
            _targets(repaired, MODEL_NODE),
            ["Telegram", "Calendar"],
        )
        self.assertEqual(_targets(repaired, RAW_PAYLOAD_NODE), [PUBLISH_NODE])
        self.assertEqual(_targets(repaired, TEXT_PAYLOAD_NODE), [PUBLISH_NODE])

        validator = next(
            node for node in repaired["nodes"] if node["name"] == VALIDATION_NODE
        )
        self.assertEqual(validator["type"], "n8n-nodes-base.code")
        self.assertEqual(validator["parameters"]["jsCode"], VALIDATION_JS)
        self.assertIn("rows.length === 0", VALIDATION_JS)
        self.assertIn("producer exited with code", VALIDATION_JS)
        self.assertIn("JSON.parse(serialized)", VALIDATION_JS)
        self.assertIn("telegram_text must be non-empty", VALIDATION_JS)
        self.assertIn("textMetadata('Generated')", VALIDATION_JS)
        self.assertIn("textMetadata('Horizon')", VALIDATION_JS)
        self.assertIn("textMetadata('Universe')", VALIDATION_JS)
        self.assertIn("universe_count", VALIDATION_JS)
        self.assertIn("optimizer_payload: report", VALIDATION_JS)
        self.assertNotIn("stdout: rawStdout", VALIDATION_JS)

        raw_builder = next(
            node for node in repaired["nodes"] if node["name"] == RAW_PAYLOAD_NODE
        )
        self.assertEqual(raw_builder["parameters"]["jsCode"], RAW_PAYLOAD_JS)
        self.assertIn("$json.optimizer_payload", RAW_PAYLOAD_JS)
        self.assertNotIn("$json.stdout", RAW_PAYLOAD_JS)
        self.assertIn("branch: 'main'", RAW_PAYLOAD_JS)

        text_builder = next(
            node for node in repaired["nodes"] if node["name"] == TEXT_PAYLOAD_NODE
        )
        self.assertEqual(text_builder["parameters"]["jsCode"], TEXT_PAYLOAD_JS)
        self.assertIn("$json.optimizer_payload", TEXT_PAYLOAD_JS)
        self.assertIn("data.telegram_text", TEXT_PAYLOAD_JS)
        self.assertNotIn("$json.message", TEXT_PAYLOAD_JS)
        self.assertIn("branch: 'main'", TEXT_PAYLOAD_JS)

    def test_model_uses_only_qualified_non_rl_rows_and_preserves_system_prompt(self) -> None:
        fixture = _workflow_fixture()
        model = next(
            node for node in fixture["nodes"] if node["name"] == MODEL_NODE
        )
        model["parameters"]["messages"]["values"][0]["content"] += (
            "\n\nOptimization safety contract: obsolete wording"
        )

        repaired, _ = repair_market_optimization_workflow(fixture)
        model = next(node for node in repaired["nodes"] if node["name"] == MODEL_NODE)
        messages = model["parameters"]["messages"]["values"]

        self.assertTrue(messages[0]["content"].startswith("Summarize the supplied"))
        self.assertIn(SYSTEM_SAFETY_SUFFIX.strip(), messages[0]["content"])
        self.assertNotIn("obsolete wording", messages[0]["content"])
        self.assertEqual(
            messages[0]["content"].count("Optimization safety contract:"),
            1,
        )
        self.assertEqual(messages[1]["content"], MODEL_USER_EXPRESSION)
        for contract_text in (
            "Signal Tier",
            "Model-Confirmed",
            "Reliability",
            "!== 'Low'",
            "Forecast Return %",
            "Selected Model",
            "!== 'RL Policy'",
            "target_session",
            "as_of_session",
            "universe_count",
            "uncertainty_pct",
            "indicative_target_pct",
            "executable_target_pct",
            "allocation_blocked",
            "short_horizon_reports",
        ):
            self.assertIn(contract_text, MODEL_USER_EXPRESSION)
        self.assertNotIn("JSON.parse", MODEL_USER_EXPRESSION)

    def test_publisher_uses_bound_credential_without_environment_token(self) -> None:
        fixture = _workflow_fixture()
        repaired, _ = repair_market_optimization_workflow(fixture)
        publisher = next(
            node for node in repaired["nodes"] if node["name"] == PUBLISH_NODE
        )
        source = next(
            node
            for node in repaired["nodes"]
            if node["name"] == DEFAULT_CREDENTIAL_SOURCE_NODE
        )

        self.assertEqual(
            publisher["credentials"]["httpHeaderAuth"],
            source["credentials"]["httpHeaderAuth"],
        )
        self.assertEqual(
            publisher["parameters"]["authentication"],
            "genericCredentialType",
        )
        self.assertEqual(
            publisher["parameters"]["genericAuthType"], "httpHeaderAuth"
        )
        self.assertIn(
            "branch: $json.branch || 'main'",
            publisher["parameters"]["jsonBody"],
        )
        self.assertNotIn(
            "generated-output", publisher["parameters"]["jsonBody"]
        )
        headers = publisher["parameters"]["headerParameters"]["parameters"]
        self.assertFalse(
            any(item["name"].casefold() == "authorization" for item in headers)
        )
        self.assertNotIn("$env.GITHUB_TOKEN", repr(publisher["parameters"]))

    def test_repair_is_idempotent_and_audit_passes(self) -> None:
        repaired, first_changes = repair_market_optimization_workflow(
            _workflow_fixture()
        )
        second, second_changes = repair_market_optimization_workflow(repaired)
        audit_input = copy.deepcopy(second)

        self.assertTrue(first_changes)
        self.assertEqual(second_changes, ())
        self.assertEqual(second, repaired)
        self.assertEqual(audit_market_optimization_workflow(second), ())
        self.assertEqual(second, audit_input)

    def test_audit_rejects_unrepaired_environment_auth(self) -> None:
        workflow, _ = repair_market_optimization_workflow(_workflow_fixture())
        publisher = next(
            node for node in workflow["nodes"] if node["name"] == PUBLISH_NODE
        )
        publisher.pop("credentials")
        publisher["parameters"].pop("authentication")
        publisher["parameters"].pop("genericAuthType")
        publisher["parameters"]["headerParameters"]["parameters"].append(
            {
                "name": "Authorization",
                "value": "={{ 'Bearer ' + $env.GITHUB_TOKEN }}",
            }
        )

        issues = audit_market_optimization_workflow(workflow)

        self.assertTrue(issues)
        self.assertTrue(any("environment token" in issue for issue in issues))
        self.assertTrue(any("credential binding" in issue for issue in issues))

    def test_repair_requires_trusted_http_header_credential(self) -> None:
        fixture = copy.deepcopy(_workflow_fixture())
        source = next(
            node
            for node in fixture["nodes"]
            if node["name"] == DEFAULT_CREDENTIAL_SOURCE_NODE
        )
        source.pop("credentials")

        with self.assertRaisesRegex(ValueError, "credential binding"):
            repair_market_optimization_workflow(fixture)


if __name__ == "__main__":
    unittest.main()
