#!/bin/bash
set -e

mkdir -p /root/.oa2a

# Only generate config if it doesn't exist (e.g., when using volume mount)
if [ ! -f /root/.oa2a/config.toml ]; then
    cat > /root/.oa2a/config.toml << CONFIG
# Auto-generated from environment variables
openai_api_key = "${OA2A_OPENAI_API_KEY:-}"
openai_base_url = "${OA2A_OPENAI_BASE_URL:-https://api.openai.com/v1}"
host = "${OA2A_HOST:-0.0.0.0}"
port = ${OA2A_PORT:-8080}
api_key = "${OA2A_API_KEY:-}"
log_level = "${OA2A_LOG_LEVEL:-INFO}"
tavily_api_key = "${OA2A_TAVILY_API_KEY:-}"
tavily_timeout = ${OA2A_TAVILY_TIMEOUT:-30.0}
tavily_max_results = ${OA2A_TAVILY_MAX_RESULTS:-5}
websearch_max_uses = ${OA2A_WEBSEARCH_MAX_USES:-5}
cors_origins = ["*"]
cors_credentials = true
cors_methods = ["*"]
cors_headers = ["*"]
CONFIG
fi

# Execute the command passed to container, or default to oa2a (foreground mode)
exec "${@:-oa2a}"
