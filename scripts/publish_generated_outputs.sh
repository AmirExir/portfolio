#!/usr/bin/env bash
set -euo pipefail

# Publish scheduled output files without touching the developer working tree.
#
# Default behavior:
# - copies generated text/json outputs from this repo
# - commits them from a clean temporary clone
# - pushes them to the generated-output branch
#
# Useful env overrides:
#   GENERATED_OUTPUT_BRANCH=generated-output
#   GENERATED_OUTPUT_BASE_BRANCH=main
#   GENERATED_OUTPUT_REMOTE=git@github.com:AmirExir/portfolio.git
#   PUBLISH_GENERATED_DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="${SOURCE_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGET_BRANCH="${GENERATED_OUTPUT_BRANCH:-generated-output}"
BASE_BRANCH="${GENERATED_OUTPUT_BASE_BRANCH:-main}"
DRY_RUN="${PUBLISH_GENERATED_DRY_RUN:-0}"

REMOTE_URL="${GENERATED_OUTPUT_REMOTE:-}"
if [[ -z "$REMOTE_URL" ]]; then
  REMOTE_URL="$(git -C "$SOURCE_REPO" remote get-url origin)"
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/portfolio-generated.XXXXXX")"
PATH_LIST="$WORK_DIR/.generated-output-paths"
COPIED_COUNT=0

cleanup() {
  if [[ "${KEEP_GENERATED_OUTPUT_WORK_DIR:-0}" == "1" ]]; then
    echo "Kept temporary checkout: $WORK_DIR"
  else
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT

copy_matching_files() {
  local root="$1"
  shift
  local source_root="$SOURCE_REPO/$root"

  [[ -d "$source_root" ]] || return 0

  while IFS= read -r -d '' file_path; do
    local rel_path="${file_path#$SOURCE_REPO/}"
    mkdir -p "$WORK_DIR/$(dirname "$rel_path")"
    cp "$file_path" "$WORK_DIR/$rel_path"
    printf '%s\0' "$rel_path" >> "$PATH_LIST"
    COPIED_COUNT=$((COPIED_COUNT + 1))
  done < <(find "$source_root" "$@" -print0)
}

clone_target_branch() {
  if git clone --depth 1 --branch "$TARGET_BRANCH" "$REMOTE_URL" "$WORK_DIR" >/dev/null 2>&1; then
    return 0
  fi

  git clone --depth 1 --branch "$BASE_BRANCH" "$REMOTE_URL" "$WORK_DIR"
  git -C "$WORK_DIR" switch -c "$TARGET_BRANCH"
}

clone_target_branch

git -C "$WORK_DIR" config user.name "${GIT_AUTHOR_NAME:-n8n Automation}"
git -C "$WORK_DIR" config user.email "${GIT_AUTHOR_EMAIL:-noreply@portfolio.local}"

copy_matching_files "market_agent/reports" \
  -maxdepth 1 -type f \( \
    -name "ml_forecast_rankings_*.txt" \
    -o -name "ml_forecast_rankings_*.json" \
  \)

copy_matching_files "market_agent/reports/news_summaries" \
  -maxdepth 1 -type f \( \
    -name "summary_*.txt" \
    -o -name "news_summary_*.txt" \
  \)

copy_matching_files "market_agent" \
  -maxdepth 1 -type f -name "summary*.txt"

copy_matching_files "ERCOTAPI" \
  -maxdepth 1 -type f \( \
    -name "ercot_news_summary_*.txt" \
    -o -name "*latest*.json" \
    -o -name "*summary*.txt" \
  \)

copy_matching_files "ERCOTAPI/market_agent" \
  -maxdepth 1 -type f -name "summary*.txt"

if [[ ! -s "$PATH_LIST" ]]; then
  echo "No generated output files found to publish."
  exit 0
fi

xargs -0 git -C "$WORK_DIR" add -f -- < "$PATH_LIST"

if git -C "$WORK_DIR" diff --cached --quiet; then
  echo "No generated output changes to publish."
  exit 0
fi

commit_message="${GENERATED_OUTPUT_COMMIT_MESSAGE:-chore: publish generated outputs $(date -u +'%Y-%m-%dT%H:%M:%SZ') [skip ci]}"

git -C "$WORK_DIR" commit -m "$commit_message"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run: skipped push to $TARGET_BRANCH after copying $COPIED_COUNT files."
else
  git -C "$WORK_DIR" push origin "HEAD:$TARGET_BRANCH"
  echo "Published $COPIED_COUNT generated files to $TARGET_BRANCH."
fi
