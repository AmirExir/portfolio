#!/bin/bash
# Quick setup script for ERCOT Dashboard credentials

echo "🔐 ERCOT Dashboard Credential Setup"
echo "===================================="
echo ""

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Check if secrets.toml already exists
if [ -f ".streamlit/secrets.toml" ]; then
    echo "⚠️  secrets.toml already exists!"
    read -p "Do you want to overwrite it? (y/n): " overwrite
    if [ "$overwrite" != "y" ]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

# Prompt for credentials
echo ""
read -p "Enter your ERCOT username/email: " username
read -sp "Enter your ERCOT password: " password
echo ""
read -sp "Enter your ERCOT subscription key: " subscription_key
echo ""
echo ""

# Create secrets.toml
cat > .streamlit/secrets.toml << EOF
# Streamlit Secrets Configuration
# This file stores your ERCOT API credentials securely
# NEVER commit this file to Git! (already in .gitignore)

ERCOT_USERNAME = "$username"
ERCOT_PASSWORD = "$password"
ERCOT_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
ERCOT_SUBSCRIPTION_KEY = "$subscription_key"
EOF

echo "✅ Credentials saved to .streamlit/secrets.toml"
echo ""
echo "🚀 You can now run the dashboard with: streamlit run ercotapi.py"
echo ""
echo "⚠️  Remember: NEVER commit secrets.toml to Git!"
