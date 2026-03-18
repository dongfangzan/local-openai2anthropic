#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Claude Code Docker Entrypoint - Auto-generate or use local config

set -e

# Create config directories
mkdir -p ~/.claude

# Check if settings.json exists in mounted volume, copy or generate
if [ -f ~/.claude-config/settings.json ]; then
    echo "Using existing settings.json from local..."
    cp ~/.claude-config/settings.json ~/.claude/settings.json
else
    echo "Generating default settings.json with environment variables..."
    cat > ~/.claude-config/settings.json << CONFIGEOF
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://oa2a:8080",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "API_TIMEOUT_MS": "${API_TIMEOUT_MS:-120000}",
    "ANTHROPIC_MODEL": "${CLAUDE_MODEL:-kimi-k2.5}",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "${CLAUDE_OPUS_MODEL:-kimi-k2.5}",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "${CLAUDE_SONNET_MODEL:-kimi-k2.5}",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "${CLAUDE_HAIKU_MODEL:-kimi-k2.5}",
    "ANTHROPIC_REASONING_MODEL": "${CLAUDE_REASONING_MODEL:-kimi-k2.5}"
  },
  "skipWebFetchPreflight": true,
  "skipDangerousModePermissionPrompt": true,
  "permissions": { "defaultMode": "bypassPermissions" },
  "autoUpdates": false
}
CONFIGEOF
    cp ~/.claude-config/settings.json ~/.claude/settings.json
fi

# Check if claude.json exists in mounted volume, copy or generate
if [ -f ~/.claude-config/claude.json ]; then
    echo "Using existing claude.json from local..."
    cp ~/.claude-config/claude.json ~/.claude.json
else
    echo "Generating default claude.json..."
    cat > ~/.claude-config/claude.json << 'EOF'
{"hasCompletedOnboarding": true, "installMethod": "docker", "autoUpdates": false, "userID": "docker-claude-user", "numStartups": 1}
EOF
    cp ~/.claude-config/claude.json ~/.claude.json
fi

echo "Claude Code config ready."
echo "Local config path: ./claude-code/config/"

# Execute the provided command
exec "$@"
