# local-openai2anthropic

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/local-openai2anthropic.svg)](https://pypi.org/project/local-openai2anthropic/)

**English | [中文](README_zh.md)**

A lightweight proxy that lets applications built with [Claude SDK](https://github.com/anthropics/anthropic-sdk-python) talk to locally-hosted OpenAI-compatible LLMs.

---

## What Problem This Solves

Many local LLM tools (vLLM, SGLang, etc.) provide an OpenAI-compatible API. But if you've built your app using Anthropic's Claude SDK, you can't use them directly.

This proxy translates Claude SDK calls to OpenAI API format in real-time, enabling:

- **Local LLM inference** with Claude-based apps
- **Offline development** without cloud API costs
- **Privacy-first AI** - data never leaves your machine
- **Seamless model switching** between cloud and local
- **Web Search tool** - built-in Tavily web search for local models

---

## Supported Local Backends

Currently tested and supported:

| Backend | Description | Status |
|---------|-------------|--------|
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput LLM inference | ✅ Fully supported |
| [SGLang](https://github.com/sgl-project/sglang) | Fast structured language model serving | ✅ Fully supported |

Other OpenAI-compatible backends may work but are not fully tested.

---

## Quick Start

### 1. Install

```bash
pip install local-openai2anthropic
```

### 2. Configure Your LLM Backend (Optional)

**Option A: Start a local LLM server**

If you don't have an LLM server running, you can start one locally:

Example with vLLM:
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf
# vLLM starts OpenAI-compatible API at http://localhost:8000/v1
```

Or with SGLang:
```bash
sglang launch --model-path meta-llama/Llama-2-7b-chat-hf --port 8000
# SGLang starts at http://localhost:8000/v1
```

**Option B: Use an existing OpenAI-compatible API**

If you already have a deployed OpenAI-compatible API (local or remote), you can use it directly. Just note the base URL for the next step.

Examples:
- Local vLLM/SGLang: `http://localhost:8000/v1`
- Remote API: `https://api.example.com/v1`

> **Note:** If you're using [Ollama](https://ollama.com), it natively supports the Anthropic API format, so you don't need this proxy. Just point your Claude SDK directly to `http://localhost:11434/v1`.

### 3. Start the Proxy (Recommended)

Run the following command to start the proxy in background mode:

```bash
oa2a start
```

**First-time setup**: If `~/.oa2a/config.toml` doesn't exist, an interactive setup wizard will guide you through:
- Enter your OpenAI API Key (for the local LLM backend)
- Enter the base URL of your local LLM (e.g., `http://localhost:8000/v1`)
- Configure server host and port (optional)
- Set server API key for authentication (optional)

After configuration, the server starts at `http://localhost:8080`.

**Daemon management commands:**

```bash
oa2a logs               # Show last 50 lines of logs
oa2a logs -f            # Follow logs in real-time (Ctrl+C to exit)
oa2a status             # Check if server is running
oa2a stop               # Stop background server
oa2a restart            # Restart with same settings
```

**Manual Configuration**

You can also manually create/edit the config file at `~/.oa2a/config.toml`:

```toml
# OA2A Configuration File
openai_api_key = "dummy"
openai_base_url = "http://localhost:8000/v1"
host = "0.0.0.0"
port = 8080
```

**Option B: Run in foreground**

```bash
oa2a                    # Run server in foreground (blocking)
# Press Ctrl+C to stop
```

### 4. Use in Your App

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080",  # Point to proxy
    api_key="dummy-key",  # Not used
)

message = client.messages.create(
    model="meta-llama/Llama-2-7b-chat-hf",  # Your local model name
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)

print(message.content[0].text)
```

---

## Using with Claude Code

You can configure [Claude Code](https://github.com/anthropics/claude-code) to use your local LLM through this proxy.

### Configuration Steps

1. **Edit Claude Code config file** at `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8080",
    "ANTHROPIC_API_KEY": "dummy-key",
    "ANTHROPIC_MODEL": "meta-llama/Llama-2-7b-chat-hf",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "meta-llama/Llama-2-7b-chat-hf",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "meta-llama/Llama-2-7b-chat-hf",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "meta-llama/Llama-2-7b-chat-hf",
    "ANTHROPIC_REASONING_MODEL": "meta-llama/Llama-2-7b-chat-hf"
  }
}
```

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_MODEL` | General model setting |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Default model for Sonnet mode (Claude Code default) |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Default model for Opus mode |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Default model for Haiku mode |
| `ANTHROPIC_REASONING_MODEL` | Default model for reasoning tasks |

### Complete Workflow Example

Make sure `~/.claude/settings.json` is configured as described above.

Terminal 1 - Start your local LLM:
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf
```

Terminal 2 - Start the proxy (background mode):
```bash
# First run: interactive setup wizard will guide you
oa2a start
```

Terminal 3 - Launch Claude Code:
```bash
claude
```

Now Claude Code will use your local LLM instead of the cloud API.

To stop the proxy:
```bash
oa2a stop
```

---

## Features

- ✅ **Streaming responses** - Real-time token streaming via SSE
- ✅ **Tool calling** - Local LLM function calling support
- ✅ **Vision models** - Multi-modal input for vision-capable models
- ✅ **Web Search** - Built-in Tavily web search for local models
- ✅ **Thinking mode** - Supports reasoning/thinking model outputs

---

## Web Search 🔍

Enable web search for your local LLM using [Tavily](https://tavily.com).

**Setup:**

1. Get a free API key at [tavily.com](https://tavily.com)

2. Add to your config (`~/.oa2a/config.toml`):
```toml
tavily_api_key = "tvly-your-api-key"
```

3. Use `web_search_20250305` tool in your app - the proxy handles search automatically.

**Options:** `tavily_max_results` (default: 5), `tavily_timeout` (default: 30), `websearch_max_uses` (default: 5)

---

## Configuration

Config file: `~/.oa2a/config.toml` (auto-created on first run)

| Option | Required | Default | Description |
|----------|----------|---------|-------------|
| `openai_base_url` | ✅ | - | Local LLM endpoint (e.g., `http://localhost:8000/v1`) |
| `openai_api_key` | ✅ | - | API key for local LLM |
| `port` | ❌ | 8080 | Proxy port |
| `host` | ❌ | 0.0.0.0 | Proxy host |
| `api_key` | ❌ | - | Auth key for this proxy |
| `tavily_api_key` | ❌ | - | Enable web search |
| `log_level` | ❌ | INFO | DEBUG, INFO, WARNING, ERROR |

---

## Architecture

```
Your App (Claude SDK)
         │
         ▼
┌─────────────────────┐
│  local-openai2anthropic  │  ← This proxy
│  (Port 8080)        │
└─────────────────────┘
         │
         ▼
Your Local LLM Server
(vLLM / SGLang)
(OpenAI-compatible API)
```

---

## Development

```bash
git clone https://github.com/dongfangzan/local-openai2anthropic.git
cd local-openai2anthropic
pip install -e ".[dev]"

pytest
```

## License

Apache License 2.0
