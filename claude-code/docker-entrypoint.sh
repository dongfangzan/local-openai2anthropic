#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Claude Code Docker Entrypoint - Auto-generate or use local config

set -e

# Create config directory if not exists
mkdir -p ~/.claude

# Check if settings.json exists, generate default if not
if [ ! -f ~/.claude/settings.json ]; then
    echo "Generating default settings.json with environment variables..."
    cat > ~/.claude/settings.json << CONFIGEOF
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
fi

# Check if claude.json exists, generate default if not
if [ ! -f ~/.claude.json ]; then
    echo "Generating default claude.json..."
    cat > ~/.claude.json << 'EOF'
{"hasCompletedOnboarding": true, "installMethod": "docker", "autoUpdates": false, "userID": "docker-claude-user", "numStartups": 1}
EOF
fi

echo "Claude Code config ready."
echo "Local config path: ./claude-code/.claude/"

# Execute the provided command
exec "$@"
