# local-openai2anthropic

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**[English](#english) | [中文](#中文)**

---

<a name="english"></a>
## English

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

---

<a name="中文"></a>
## 中文

一个轻量级代理服务器，将 **Anthropic Messages API** 请求转换为 **OpenAI API** 调用。

让你可以使用任何 OpenAI 兼容的后端（OpenAI、Azure、本地 vLLM 等）来运行为 Claude API 构建的应用程序。

## 功能特性

- ✅ **完整的 Messages API 兼容性** - 支持所有 Anthropic Messages API 功能
- ✅ **流式响应支持** - 使用服务器推送事件（SSE）实现实时响应
- ✅ **工具/函数调用** - Anthropic 工具与 OpenAI 函数之间的双向转换
- ✅ **视觉/多模态** - 支持图片输入
- ✅ **官方 SDK 类型** - 使用官方 `anthropic` 和 `openai` Python SDK 确保类型安全
- ✅ **简单配置** - 支持环境变量或 `.env` 文件
- ✅ **CORS 支持** - 开箱即用的浏览器应用支持
- ✅ **自托管** - 可在本地运行或部署到自己的基础设施

## 快速开始

### 安装

```bash
# 从源码安装（当前推荐）
git clone https://github.com/yourusername/local-openai2anthropic.git
cd local-openai2anthropic
pip install -e ".[dev]"

# 或直接安装
pip install local-openai2anthropic
```

### 配置

设置你的 OpenAI API 密钥：

```bash
export OA2A_OPENAI_API_KEY="sk-..."
```

或创建 `.env` 文件：

```env
OA2A_OPENAI_API_KEY=sk-...
OA2A_OPENAI_BASE_URL=https://api.openai.com/v1
OA2A_HOST=0.0.0.0
OA2A_PORT=8080
```

### 运行服务器

```bash
# 使用 CLI 命令
local-openai2anthropic

# 或使用短别名
oa2a

# 或使用 Python 模块
python -m local_openai2anthropic
```

## 使用示例

### 使用 Anthropic Python SDK

```python
import anthropic

# 指向本地代理而非 Anthropic 官方 API
client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="dummy-key",  # 需要填写但不会被使用
)

# 像平常一样使用 Messages API
message = client.messages.create(
    model="gpt-4o",  # 会传递给 OpenAI
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "你好，Claude！"}
    ]
)

print(message.content[0].text)
```

### 使用流式响应

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="dummy-key",
)

stream = client.messages.create(
    model="gpt-4o",
    max_tokens=1024,
    messages=[{"role": "user", "content": "数到 10"}],
    stream=True,
)

for event in stream:
    if event.type == "content_block_delta":
        print(event.delta.text, end="", flush=True)
```

### 使用工具调用

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
            "description": "获取某地天气",
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
    messages=[{"role": "user", "content": "东京天气如何？"}],
)

if message.stop_reason == "tool_use":
    tool_use = message.content[-1]
    print(f"调用工具: {tool_use.name}")
    print(f"输入参数: {tool_use.input}")
```

### 使用视觉（图片）

```python
import anthropic
import base64

client = anthropic.Anthropic(
    base_url="http://localhost:8080",
    api_key="dummy-key",
)

# 读取图片并编码为 base64
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

message = client.messages.create(
    model="gpt-4o",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么？"},
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

## 配置选项

所有配置都可以通过环境变量（前缀为 `OA2A_`）或 `.env` 文件设置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OA2A_OPENAI_API_KEY` | *必填* | 你的 OpenAI API 密钥 |
| `OA2A_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API 基础地址 |
| `OA2A_OPENAI_ORG_ID` | `None` | OpenAI 组织 ID |
| `OA2A_OPENAI_PROJECT_ID` | `None` | OpenAI 项目 ID |
| `OA2A_HOST` | `0.0.0.0` | 服务器绑定的主机地址 |
| `OA2A_PORT` | `8080` | 服务器端口 |
| `OA2A_REQUEST_TIMEOUT` | `300.0` | 请求超时时间（秒） |
| `OA2A_API_KEY` | `None` | 保护代理的可选 API 密钥 |
| `OA2A_CORS_ORIGINS` | `["*"]` | 允许的 CORS 来源 |
| `OA2A_LOG_LEVEL` | `INFO` | 日志级别 |

## 使用场景

### 1. 使用 Claude 应用配合 OpenAI 模型

许多应用是专门为 Claude API 构建的。此代理允许你将它们与 GPT-4、GPT-3.5 或任何 OpenAI 兼容模型一起使用。

### 2. 配合 vLLM 进行本地开发

运行本地 vLLM 服务器，使用 OpenAI 兼容 API，然后通过此代理测试集成 Claude 的应用：

```bash
# 终端 1：启动 vLLM
vllm serve meta-llama/Llama-2-7b-chat-hf --api-key dummy

# 终端 2：启动代理指向 vLLM
export OA2A_OPENAI_API_KEY=dummy
export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1
local-openai2anthropic

# 终端 3：使用 Claude SDK 配合本地模型
python my_claude_app.py  # 指向 http://localhost:8080
```

### 3. Azure OpenAI 服务

```bash
export OA2A_OPENAI_API_KEY="your-azure-key"
export OA2A_OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
local-openai2anthropic
```

### 4. 其他 OpenAI 兼容 API

- **Groq**
- **Together AI**
- **Fireworks**
- **Anyscale**
- **LocalAI**
- **llama.cpp server**

## API 覆盖情况

### 支持的 Anthropic 功能

| 功能 | 状态 | 备注 |
|------|------|------|
| `messages.create()` | ✅ 完整 | 支持所有参数 |
| 流式响应 | ✅ 完整 | SSE 支持所有事件类型 |
| 工具调用 | ✅ 完整 | 转换为 OpenAI 函数 |
| 视觉 | ✅ 完整 | 图片转换为 base64 data URLs |
| 系统提示词 | ✅ 完整 | 支持字符串或数组格式 |
| 停止序列 | ✅ 完整 | 透传 |
| 温度 | ✅ 完整 | |
| Top P | ✅ 完整 | |
| Top K | ✅ 完整 | |
| 最大 token | ✅ 完整 | |
| 思考模式 | ⚠️ 部分 | 在支持的模型上映射到 reasoning_effort |

### 不支持的功能

- **提示词缓存** (`cache_control`) - OpenAI 没有等效功能
- **计算机使用（测试版）** - 需要原生 Claude 能力

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    你的应用                                       │
│              (使用 Anthropic Python SDK)                          │
└───────────────────────┬─────────────────────────────────────────┘
                        │ Anthropic Messages API 格式
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              local-openai2anthropic 代理                          │
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
                        │ OpenAI API 格式
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              OpenAI 兼容后端                                      │
│        (OpenAI、Azure、vLLM、Groq 等)                            │
└─────────────────────────────────────────────────────────────────┘
```

## 开发

```bash
# 克隆仓库
git clone https://github.com/yourusername/local-openai2anthropic.git
cd local-openai2anthropic

# 以开发模式安装
pip install -e ".[dev]"

# 运行测试
pytest

# 格式化代码
black src/
ruff check src/

# 类型检查
mypy src/
```

## 许可证

Apache License 2.0 - 详情请参阅 [LICENSE](LICENSE)。

## 致谢

本项目基于 [vLLM](https://github.com/vllm-project/vllm) 的 Anthropic API 实现，适配为独立的代理服务。
