# local-openai2anthropic

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A lightweight proxy server that converts **Anthropic Messages API** requests to **OpenAI API** calls.

This allows you to use applications built for Claude's API with any OpenAI-compatible backend (OpenAI, Azure, local vLLM, etc.).

## Features

- ✅ **Full Messages API compatibility** - All Anthropic Messages API features
- ✅ **Streaming support** - Server-sent events (SSE) for real-time responses
- ✅ **Tool/Function calling** - Convert between Anthropic tools and OpenAI functions
- ✅ **Vision/Multimodal** - Image input support
- ✅ **Official SDK types** - Uses official `anthropic` and `openai` Python SDKs for type safety
- ✅ **Easy configuration** - Environment variables or `.env` file
- ✅ **CORS support** - Ready for browser-based applications
- ✅ **Self-hosted** - Run locally or deploy to your infrastructure

## Quick Start

### Installation

```bash
# Install from source (recommended for now)
git clone https://github.com/yourusername/local-openai2anthropic.git
cd local-openai2anthropic
pip install -e ".[dev]"

# Or install directly
pip install local-openai2anthropic
```

### Configuration

Set your OpenAI API key:

```bash
export OA2A_OPENAI_API_KEY="sk-..."
```

Or create a `.env` file:

```env
OA2A_OPENAI_API_KEY=sk-...
OA2A_OPENAI_BASE_URL=https://api.openai.com/v1
OA2A_HOST=0.0.0.0
OA2A_PORT=8080
```

### Run the Server

```bash
# Using the CLI command
local-openai2anthropic

# Or using the short alias
oa2a

# Or using Python module
python -m local_openai2anthropic
```

## Usage Examples

### Using with Anthropic Python SDK

```python
import anthropic

# Point to your local proxy instead of Anthropic's API
client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="dummy-key",  # Not used but required by SDK
)

# Use the Messages API normally
message = client.messages.create(
    model="gpt-4o",  # This will be passed to OpenAI
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

print(message.content[0].text)
```

### Using with Streaming

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="dummy-key",
)

stream = client.messages.create(
    model="gpt-4o",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True,
)

for event in stream:
    if event.type == "content_block_delta":
        print(event.delta.text, end="", flush=True)
```

### Using with Tool Calling

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="dummy-key",
)

message = client.messages.create(
    model="gpt-4o",
    max_tokens=1024,
    tools=[
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ],
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
)

if message.stop_reason == "tool_use":
    tool_use = message.content[-1]
    print(f"Tool called: {tool_use.name}")
    print(f"Input: {tool_use.input}")
```

### Using with Vision (Images)

```python
import anthropic
import base64

client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="dummy-key",
)

# Read image and encode as base64
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

message = client.messages.create(
    model="gpt-4o",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
            ],
        }
    ],
)

print(message.content[0].text)
```

## Configuration Options

All configuration is done via environment variables (with `OA2A_` prefix) or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OA2A_OPENAI_API_KEY` | *Required* | Your OpenAI API key |
| `OA2A_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL |
| `OA2A_OPENAI_ORG_ID` | `None` | OpenAI organization ID |
| `OA2A_OPENAI_PROJECT_ID` | `None` | OpenAI project ID |
| `OA2A_HOST` | `0.0.0.0` | Server host to bind |
| `OA2A_PORT` | `8080` | Server port |
| `OA2A_REQUEST_TIMEOUT` | `300.0` | Request timeout in seconds |
| `OA2A_API_KEY` | `None` | Optional API key to protect your proxy |
| `OA2A_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `OA2A_LOG_LEVEL` | `INFO` | Logging level |

## Use Cases

### 1. Using Claude-based Apps with OpenAI Models

Many applications are built specifically for Claude's API. This proxy lets you use them with GPT-4, GPT-3.5, or any OpenAI-compatible model.

### 2. Local Development with vLLM

Run a local vLLM server with OpenAI-compatible API, then use this proxy to test Claude-integrated apps:

```bash
# Terminal 1: Start vLLM
vllm serve meta-llama/Llama-2-7b-chat-hf --api-key dummy

# Terminal 2: Start proxy pointing to vLLM
export OA2A_OPENAI_API_KEY=dummy
export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1
local-openai2anthropic

# Terminal 3: Use Claude SDK with local model
python my_claude_app.py  # Point to http://localhost:8080
```

### 3. Azure OpenAI Service

```bash
export OA2A_OPENAI_API_KEY="your-azure-key"
export OA2A_OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
local-openai2anthropic
```

### 4. Other OpenAI-Compatible APIs

- **Groq**
- **Together AI**
- **Fireworks**
- **Anyscale**
- **LocalAI**
- **llama.cpp server**

## API Coverage

### Supported Anthropic Features

| Feature | Status | Notes |
|---------|--------|-------|
| `messages.create()` | ✅ Full | All parameters supported |
| Streaming | ✅ Full | SSE with all event types |
| Tool use | ✅ Full | Converted to OpenAI functions |
| Vision | ✅ Full | Images converted to base64 data URLs |
| System prompts | ✅ Full | String or array format |
| Stop sequences | ✅ Full | Passed through |
| Temperature | ✅ Full | |
| Top P | ✅ Full | |
| Top K | ✅ Full | |
| Max tokens | ✅ Full | |
| Thinking | ⚠️ Partial | Mapped to reasoning_effort where supported |

### Not Supported

- **Prompt caching** (`cache_control`) - OpenAI doesn't have equivalent
- **Computer use (beta)** - Requires native Claude capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your Application                             │
│              (uses Anthropic Python SDK)                        │
└───────────────────────┬─────────────────────────────────────────┘
                        │ Anthropic Messages API format
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              local-openai2anthropic Proxy                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Anthropic  │───▶│   Converter  │───▶│    OpenAI    │      │
│  │   Request    │    │              │    │   Request    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Anthropic  │◀───│   Converter  │◀───│    OpenAI    │      │
│  │   Response   │    │              │    │   Response   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└───────────────────────┬─────────────────────────────────────────┘
                        │ OpenAI API format
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              OpenAI-compatible Backend                          │
│        (OpenAI, Azure, vLLM, Groq, etc.)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/local-openai2anthropic.git
cd local-openai2anthropic

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/

# Type check
mypy src/
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

This project is based on the Anthropic API implementation from [vLLM](https://github.com/vllm-project/vllm), adapted to work as a standalone proxy service.
