# Claude Code Docker Integration

This directory contains the Docker configuration for running Claude Code CLI with
OA2A proxy integration.

## Features

- **Pre-configured Claude Code**: No login required, works out of the box
- **Dual Environment**: Includes both Node.js and Python for development
- **Custom API Support**: Connects to any OpenAI-compatible API through OA2A proxy
- **Model Customization**: Configure model names via environment variables

## Quick Start

```bash
# Start both services
docker-compose up -d

# Enter Claude Code
docker-compose exec claude-code claude

# Or enter bash first
docker-compose exec claude-code bash
$ claude /workspace
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_MODEL` | kimi-k2.5 | Default model for all requests |
| `CLAUDE_OPUS_MODEL` | kimi-k2.5 | Model for Opus tier requests |
| `CLAUDE_SONNET_MODEL` | kimi-k2.5 | Model for Sonnet tier requests |
| `CLAUDE_HAIKU_MODEL` | kimi-k2.5 | Model for Haiku tier requests |
| `CLAUDE_REASONING_MODEL` | kimi-k2.5 | Model for reasoning/thinking requests |
| `API_TIMEOUT_MS` | 120000 | API request timeout in milliseconds |

## Included Tools

- Node.js 20 (for Claude Code CLI)
- Python 3.11 with pip
- Git
- curl
- Vim/Nano

## Directory Structure

```
/workspace    # mounted from ./workspace for development
/root/.claude # persisted Claude Code configuration
```
