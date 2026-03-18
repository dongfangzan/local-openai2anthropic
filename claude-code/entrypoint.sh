#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Entrypoint script to generate Claude Code configuration from environment variables

set -e

# Use the correct home directory (non-root user)
HOME_DIR="${HOME:-/home/claude}"

# Create directories
mkdir -p "$HOME_DIR/.claude" || true

# Generate settings.json from environment variables
cat > "$HOME_DIR/.claude/settings.json" << EOF
{
  "env": {
    "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL:-http://oa2a:8080}",
    "ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_AUTH_TOKEN:-local}",
    "API_TIMEOUT_MS": "${API_TIMEOUT_MS:-120000}",
    "ANTHROPIC_MODEL": "${CLAUDE_MODEL:-kimi-k2.5}",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "${CLAUDE_OPUS_MODEL:-kimi-k2.5}",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "${CLAUDE_SONNET_MODEL:-kimi-k2.5}",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "${CLAUDE_HAIKU_MODEL:-kimi-k2.5}",
    "ANTHROPIC_REASONING_MODEL": "${CLAUDE_REASONING_MODEL:-kimi-k2.5}",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "skipWebFetchPreflight": true,
  "skipDangerousModePermissionPrompt": true,
  "permissions": {
    "defaultMode": "bypassPermissions"
  },
  "autoUpdates": false,
  "includeCoAuthoredBy": false,
  "sandbox": {
    "enabled": true,
    "autoAllowBashIfSandboxed": true,
    "filesystem": {
      "denyRead": ["~/", "~/.claude/settings.json", "~/.claude.json"],
      "denyWrite": ["~/", "~/.claude/settings.json", "~/.claude.json", "~/.claude/config.json"],
      "allowRead": ["/workspace", "~/.claude/"],
      "allowWrite": ["/workspace"]
    }
  }
}
EOF

# Ensure config.json exists
if [ ! -f "$HOME_DIR/.claude/config.json" ]; then
    cat > "$HOME_DIR/.claude/config.json" << EOF
{
  "primaryApiKey": "local",
  "skipWebFetchPreflight": true
}
EOF
fi

# Generate claude.json (login state)
cat > "$HOME_DIR/.claude.json" << EOF
{
  "hasCompletedOnboarding": true,
  "installMethod": "docker",
  "autoUpdates": false,
  "userID": "docker-claude-$(hostname)",
  "numStartups": 1
}
EOF

echo "Claude Code configuration initialized:"
echo "  ANTHROPIC_BASE_URL: ${ANTHROPIC_BASE_URL:-http://oa2a:8080}"
echo "  ANTHROPIC_MODEL: ${CLAUDE_MODEL:-kimi-k2.5}"
echo "  HOME: $HOME_DIR"
echo ""

# Execute the provided command
exec "$@"
