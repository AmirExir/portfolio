"""Regression checks against an installed n8n binary proxy and Code sandbox.

These optional local integration tests use no network, credentials or workflow
database. A plain JavaScript mock of ``$binary`` missed its metadata-only
behavior, which removed the chart bytes before the Telegram channel upload.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import unittest

from ERCOTAPI.load_forecast_workflow import LOAD_CHANNEL_JS


def installed_n8n_root() -> Path | None:
    """Find the installed n8n package through its CLI, without fixed host paths."""

    executable = shutil.which("n8n")
    if not executable:
        return None
    for parent in Path(executable).resolve().parents:
        manifest = parent / "package.json"
        if not manifest.is_file():
            continue
        try:
            package = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if package.get("name") == "n8n":
            return parent
    return None


N8N_ROOT = installed_n8n_root()
PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aVZkAAAAASUVORK5CYII="

RUNTIME_SCRIPT = r"""const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const fromN8n = createRequire(path.join(input.n8n_root, 'package.json'));
const {Workflow, WorkflowDataProxy} = fromN8n('n8n-workflow');
const nodesRoot = path.dirname(fromN8n.resolve('n8n-nodes-base/package.json'));
const {JavaScriptSandbox} = require(path.join(nodesRoot, 'dist/nodes/Code/JavaScriptSandbox.js'));
const coreDist = path.dirname(fromN8n.resolve('n8n-core'));
const {normalizeItems} = require(path.join(coreDist,
  'execution-engine/node-execution-context/utils/normalize-items.js'));
const node = {id: 'binary-regression', name: 'Channel payload',
  type: 'n8n-nodes-base.code', typeVersion: 2, parameters: {}, position: [0, 0]};
const workflow = new Workflow({nodes: [node], connections: {}, active: false,
  nodeTypes: {getByNameAndVersion: () => undefined},
  settings: {binaryMode: 'separate', timezone: 'America/Chicago'}});
const proxy = new WorkflowDataProxy(workflow, null, 0, input.item_index,
  node.name, input.items, {}, 'manual', {}).getDataProxy();
const sandbox = new JavaScriptSandbox({$json: proxy.$json, $binary: proxy.$binary,
  $input: proxy.$input}, input.code, {normalizeItems});
(async () => {
  const output = await sandbox.runCodeEachItem(input.item_index);
  process.stdout.write(JSON.stringify({output,
    metadata_keys: Object.keys(proxy.$binary.data ?? {}),
    input_keys: Object.keys(proxy.$input.item.binary?.data ?? {})}));
})().catch(error => {
  process.stderr.write(error.message);
  process.exitCode = 1;
});"""


@unittest.skipUnless(N8N_ROOT and shutil.which("node"), "Installed n8n and Node.js are required")
class LoadForecastBinaryRuntimeTests(unittest.TestCase):
    """Exercise actual n8n getters and each-item Code validation, not mocks."""

    def execute(self, code: str, items: list[dict], *, item_index: int = 0) -> dict:
        """Run one Code item using the locally installed n8n implementation."""

        result = subprocess.run(
            ["node", "-e", RUNTIME_SCRIPT],
            input=json.dumps({"n8n_root": str(N8N_ROOT), "code": code,
                              "items": items, "item_index": item_index}),
            text=True, capture_output=True, check=True, timeout=30,
        )
        return json.loads(result.stdout)

    @staticmethod
    def item(chat_id: str = "@fixture", *, binary_data: dict | None = None) -> dict:
        """Create a public synthetic payload without contacting Telegram."""

        return {
            "json": {"channel_chat_id": chat_id, "text": "Forecast 50,001 MW",
                     "report_ok": True, "has_chart": True},
            "binary": {"data": binary_data or {
                "data": PNG_BASE64, "mimeType": "image/png", "fileName": "chart.png",
            }},
        }

    def test_actual_metadata_proxy_loses_bytes_but_channel_code_preserves_them(self) -> None:
        # Negative control reproduces the production failure with the old code.
        legacy = self.execute("return {json: $json, binary: $binary};", [self.item()])
        self.assertNotIn("data", legacy["output"]["binary"]["data"])
        self.assertNotIn("data", legacy["metadata_keys"])
        self.assertIn("data", legacy["input_keys"])

        actual = self.execute(LOAD_CHANNEL_JS, [self.item()])["output"]
        self.assertEqual(actual["binary"]["data"]["data"], PNG_BASE64)
        self.assertEqual(actual["json"]["chat_id"], "@fixture")

    def test_each_item_preserves_its_own_binary_including_storage_reference(self) -> None:
        stored = {"id": "filesystem-v2:fixture", "data": "filesystem-v2",
                  "mimeType": "image/png", "fileName": "second.png"}
        items = [self.item(), self.item("@second", binary_data=stored)]
        actual = self.execute(LOAD_CHANNEL_JS, items, item_index=1)["output"]
        self.assertEqual(actual["json"]["chat_id"], "@second")
        self.assertEqual(actual["binary"]["data"], stored)

    def test_text_only_failure_does_not_require_binary(self) -> None:
        item = {"json": {"channel_chat_id": "@fixture", "text": "Forecast unavailable",
                         "report_ok": False, "has_chart": False}}
        actual = self.execute(LOAD_CHANNEL_JS, [item])["output"]
        self.assertNotIn("binary", actual)
        self.assertEqual(actual["json"]["text"], "Forecast unavailable")


if __name__ == "__main__":
    unittest.main()
