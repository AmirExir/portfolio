#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point for n8n/cron.
# The publisher uses a clean temporary checkout so scheduled outputs can go to
# GitHub automatically without staging or pushing normal portfolio edits.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec "$REPO_DIR/scripts/publish_generated_outputs.sh"
