# SPDX-License-Identifier: Apache-2.0
# Claude Code Docker Image with Python and Node.js environments

FROM node:20-slim

# Install Python and other development tools (including bubblewrap for sandbox)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    vim \
    nano \
    build-essential \
    sudo \
    bubblewrap \
    socat \
    ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Create a non-root user for Claude Code
RUN useradd -m -s /bin/bash -u 1001 claude && \
    echo "claude ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/claude && \
    chmod 0440 /etc/sudoers.d/claude

# Install Claude Code CLI and Sandbox Runtime globally
RUN npm install -g @anthropic-ai/claude-code @anthropic-ai/sandbox-runtime && \
    # Fix permissions for claude user
    mkdir -p /home/claude/.claude && \
    chown -R claude:claude /home/claude

# Verify bubblewrap installation and create sandbox-runtime alias
RUN which bwrap && bwrap --version && \
    ln -sf /usr/local/bin/srt /usr/local/bin/sandbox-runtime && \
    ls -la /usr/local/bin/sandbox-runtime

# Create workspace directory and set permissions
RUN mkdir -p /workspace && chown -R claude:claude /workspace

# Copy entrypoint scripts
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER claude
WORKDIR /workspace

# Set environment defaults
ENV ANTHROPIC_BASE_URL=http://oa2a:8080 \
    ANTHROPIC_AUTH_TOKEN=local \
    CLAUDE_MODEL=kimi-k2.5 \
    CLAUDE_OPUS_MODEL=kimi-k2.5 \
    CLAUDE_SONNET_MODEL=kimi-k2.5 \
    CLAUDE_HAIKU_MODEL=kimi-k2.5 \
    CLAUDE_REASONING_MODEL=kimi-k2.5 \
    API_TIMEOUT_MS=120000 \
    HOME=/home/claude \
    CLAUDE_SANDBOX_RUNTIME=/usr/local/bin/sandbox-runtime

# Default to bash if no command provided
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
