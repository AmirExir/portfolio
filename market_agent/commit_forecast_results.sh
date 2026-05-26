#!/bin/bash
# Commit and push ML forecast results to GitHub

REPO_DIR="/Users/amirexir/Documents/GitHub/portfolio"
cd "$REPO_DIR" || exit 1

# Configure git (if needed)
git config user.email "noreply@portfolio.local" 2>/dev/null
git config user.name "n8n Automation" 2>/dev/null

# Add the forecast files
git add market_agent/reports/ml_forecast_rankings_latest.txt \
        market_agent/reports/ml_forecast_rankings_latest.json \
        market_agent/reports/ml_forecast_rankings_cache_*.json 2>/dev/null

# Commit and push
if ! git diff --cached --quiet; then
    git commit -m "chore: update ML forecast rankings [skip ci]"
    git push origin main
    echo "✓ Forecast results committed and pushed"
else
    echo "✓ No changes to commit"
fi
