#!/bin/bash
# Commit and push ML forecast results to GitHub

REPO_DIR="/Users/amirexir/Documents/GitHub/portfolio"
cd "$REPO_DIR" || exit 1

# Configure git (if needed)
git config user.email "noreply@portfolio.local" 2>/dev/null
git config user.name "n8n Automation" 2>/dev/null

# Add immutable scheduled artifacts only. Do not commit ml_forecast_rankings_latest.*
# because every run rewrites those files and they create recurring merge conflicts.
git add -f market_agent/reports/ml_forecast_rankings_20*.json \
           market_agent/reports/optimization_summaries/ml_forecast_rankings_20*.txt \
           market_agent/reports/news_summaries/news_summary_20*.txt 2>/dev/null

# Commit and push
if ! git diff --cached --quiet; then
    git commit -m "chore: update ML forecast rankings [skip ci]"
    git push origin main
    echo "✓ Forecast results committed and pushed"
else
    echo "✓ No changes to commit"
fi
