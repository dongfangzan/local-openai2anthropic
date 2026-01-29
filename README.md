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

### 2. Start Your Local LLM Server

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

### 3. Start the Proxy

**Option A: Run in background (recommended)**

```bash
export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1  # Your local LLM endpoint
export OA2A_OPENAI_API_KEY=dummy  # Any value, not used by local backends

oa2a start              # Start server in background
# Server starts at http://localhost:8080

# View logs
oa2a logs               # Show last 50 lines of logs
oa2a logs -f            # Follow logs in real-time (Ctrl+C to exit)

# Check status
oa2a status             # Check if server is running

# Stop server
oa2a stop               # Stop background server

# Restart server
oa2a restart            # Restart with same settings
```

**Option B: Run in foreground**

```bash
export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1
export OA2A_OPENAI_API_KEY=dummy

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

1. **Create or edit Claude Code config file** at `~/.claude/CLAUDE.md`:

```markdown
# Claude Code Configuration

## API Settings

- Claude API Base URL: http://localhost:8080
- Claude API Key: dummy-key

## Model Settings

Use model: meta-llama/Llama-2-7b-chat-hf  # Your local model name
```

2. **Alternatively, set environment variables** before running Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-key

claude
```

3. **Or use the `--api-key` and `--base-url` flags**:

```bash
claude --api-key dummy-key --base-url http://localhost:8080
```

### Complete Workflow Example

Terminal 1 - Start your local LLM:
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf
```

Terminal 2 - Start the proxy:
```bash
export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1
export OA2A_OPENAI_API_KEY=dummy
export OA2A_TAVILY_API_KEY="tvly-your-tavily-api-key"  # Optional: enable web search

oa2a
```

Terminal 3 - Launch Claude Code with local LLM:
```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-key

claude
```

Now Claude Code will use your local LLM instead of the cloud API.

---

## Features

- ✅ **Streaming responses** - Real-time token streaming via SSE
- ✅ **Tool calling** - Local LLM function calling support
- ✅ **Vision models** - Multi-modal input for vision-capable models
- ✅ **Web Search** - Give your local LLM internet access (see below)
- ✅ **Thinking mode** - Supports reasoning/thinking model outputs

---

## Web Search Capability 🔍

**Bridge the gap: Give your local LLM the web search power that Claude Code users enjoy!**

When using locally-hosted models with Claude Code, you lose access to the built-in web search tool. This proxy fills that gap by providing a server-side web search implementation powered by [Tavily](https://tavily.com).

### The Problem

| Scenario | Web Search Available? |
|----------|----------------------|
| Using Claude (cloud) in Claude Code | ✅ Built-in |
| Using local vLLM/SGLang in Claude Code | ❌ Not available |
| **Using this proxy + local LLM** | ✅ **Enabled via Tavily** |

### How It Works

```
Claude Code → Anthropic SDK → This Proxy → Local LLM
                                      ↓
                                 Tavily API (Web Search)
```

The proxy intercepts `web_search_20250305` tool calls and handles them directly, regardless of whether your local model supports web search natively.

### Setup Tavily Search

1. **Get a free API key** at [tavily.com](https://tavily.com) - generous free tier available

2. **Configure the proxy:**
```bash
export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1
export OA2A_OPENAI_API_KEY=dummy
export OA2A_TAVILY_API_KEY="tvly-your-tavily-api-key"  # Enable web search

oa2a
```

3. **Use in your app:**
```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="dummy-key",
)

message = client.messages.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    max_tokens=1024,
    tools=[
        {
            "name": "web_search_20250305",
            "description": "Search the web for current information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        }
    ],
    messages=[{"role": "user", "content": "What happened in AI today?"}],
)

if message.stop_reason == "tool_use":
    tool_use = message.content[-1]
    print(f"Searching: {tool_use.input}")
    # The proxy automatically calls Tavily and returns results
```

### Tavily Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OA2A_TAVILY_API_KEY` | - | Your Tavily API key ([get free at tavily.com](https://tavily.com)) |
| `OA2A_TAVILY_MAX_RESULTS` | 5 | Number of search results to return |
| `OA2A_TAVILY_TIMEOUT` | 30 | Search timeout in seconds |
| `OA2A_WEBSEARCH_MAX_USES` | 5 | Max search calls per request |

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OA2A_OPENAI_BASE_URL` | ✅ | - | Your local LLM's OpenAI-compatible endpoint |
| `OA2A_OPENAI_API_KEY` | ✅ | - | Any value (local backends usually ignore this) |
| `OA2A_PORT` | ❌ | 8080 | Proxy server port |
| `OA2A_HOST` | ❌ | 0.0.0.0 | Proxy server host |
| `OA2A_TAVILY_API_KEY` | ❌ | - | Enable web search ([tavily.com](https://tavily.com)) |

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
