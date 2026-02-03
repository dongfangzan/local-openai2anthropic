# local-openai2anthropic

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/local-openai2anthropic.svg)](https://pypi.org/project/local-openai2anthropic/)

**[English](README.md) | 中文**

一个轻量级代理，让使用 [Claude SDK](https://github.com/anthropics/anthropic-sdk-python) 开发的应用无缝接入本地部署的大模型。

---

## 解决的问题

很多本地大模型工具（vLLM、SGLang 等）提供 OpenAI 兼容的 API。但如果你用 Anthropic 的 Claude SDK 开发了应用，无法直接调用它们。

这个代理实时将 Claude SDK 调用转换为 OpenAI API 格式，让你可以：

- **本地推理** - 用 Claude SDK 调用本地模型
- **离线开发** - 无需支付云 API 费用
- **隐私优先** - 数据不出本机
- **灵活切换** - 云端和本地模型无缝切换
- **网络搜索** - 内置 Tavily 网页搜索工具，为本地模型提供联网能力

---

## 支持的本地后端

目前已测试并完全支持：

| 后端 | 说明 | 状态 |
|---------|-------------|--------|
| [vLLM](https://github.com/vllm-project/vllm) | 高吞吐 LLM 推理引擎 | ✅ 完全支持 |
| [SGLang](https://github.com/sgl-project/sglang) | 高性能结构化语言模型服务 | ✅ 完全支持 |

其他 OpenAI 兼容后端可能可以使用，但未完整测试。

---

## 快速开始

### 1. 安装

```bash
pip install local-openai2anthropic
```

### 2. 配置你的 LLM 后端（可选）

**选项 A：启动本地模型服务**

如果你还没有运行 LLM 服务，可以在本地启动一个：

使用 vLLM 示例：
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf
# vLLM 在 http://localhost:8000/v1 提供 OpenAI 兼容 API
```

或使用 SGLang：
```bash
sglang launch --model-path meta-llama/Llama-2-7b-chat-hf --port 8000
# SGLang 在 http://localhost:8000/v1 启动
```

**选项 B：使用已有的 OpenAI 兼容 API**

如果你已经部署了 OpenAI 兼容的 API（本地或远程），可以直接使用。记下 base URL 用于下一步。

示例：
- 本地 vLLM/SGLang：`http://localhost:8000/v1`
- 远程 API：`https://api.example.com/v1`

> **注意：** 如果你使用 [Ollama](https://ollama.com)，它原生支持 Anthropic API 格式，无需使用本代理工具。直接将 Claude SDK 指向 `http://localhost:11434/v1` 即可。

### 3. 配置代理

首次运行时，交互式配置向导会引导你创建配置文件 `~/.oa2a/config.toml`：

```bash
oa2a
# 交互式配置向导启动：
# - 输入 OpenAI API Key（用于本地 LLM 后端）
# - 输入本地 LLM 的 base URL（如 http://localhost:8000/v1）
# - 配置服务器 host 和 port（可选）
# - 设置代理 API 认证密钥（可选）
```

**手动配置**

你也可以直接编辑配置文件 `~/.oa2a/config.toml`：

```toml
# OA2A 配置文件
openai_api_key = "dummy"
openai_base_url = "http://localhost:8000/v1"
host = "0.0.0.0"
port = 8080
```

### 4. 启动代理

**方式 A: 后台运行（推荐）**

```bash
oa2a start              # 后台启动服务
# 代理在 http://localhost:8080 启动

# 查看日志
oa2a logs               # 显示最后 50 行日志
oa2a logs -f            # 实时跟踪日志 (Ctrl+C 退出)

# 检查状态
oa2a status             # 检查服务是否运行

# 停止服务
oa2a stop               # 停止后台服务

# 重启服务
oa2a restart            # 使用相同配置重启
```

**方式 B: 前台运行**

```bash
oa2a                    # 前台运行服务（阻塞模式）
# 按 Ctrl+C 停止
```

### 4. 在应用中使用

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080",  # 指向代理
    api_key="dummy-key",  # 不使用
)

message = client.messages.create(
    model="meta-llama/Llama-2-7b-chat-hf",  # 你的本地模型名称
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好！"}],
)

print(message.content[0].text)
```

---

## 配合 Claude Code 使用

你可以配置 [Claude Code](https://github.com/anthropics/claude-code) 通过本代理使用本地大模型。

### 配置步骤

1. **编辑 Claude Code 配置文件** `~/.claude/settings.json`：

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

| 变量 | 说明 |
|------|------|
| `ANTHROPIC_MODEL` | 通用模型配置 |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Sonnet 模式默认模型（Claude Code 默认使用） |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Opus 模式默认模型 |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Haiku 模式默认模型 |
| `ANTHROPIC_REASONING_MODEL` | 推理任务默认模型 |

2. **或者在运行 Claude Code 前设置环境变量**：

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-key

claude
```

### 完整工作流示例

确保 `~/.claude/settings.json` 已按上述步骤配置好。

终端 1 - 启动本地模型：
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf
```

终端 2 - 配置并启动代理（后台运行）：
```bash
# 首次运行：交互式配置（创建 ~/.oa2a/config.toml）
# 或手动编辑：~/.oa2a/config.toml

oa2a start
```

**配置文件示例** (`~/.oa2a/config.toml`)：
```toml
openai_api_key = "dummy"
openai_base_url = "http://localhost:8000/v1"
host = "0.0.0.0"
port = 8080

# 可选：启用网页搜索
tavily_api_key = "tvly-your-tavily-api-key"
```

终端 3 - 启动 Claude Code：
```bash
claude
```

现在 Claude Code 将使用你的本地大模型，而不是云端 API。

如需停止代理：
```bash
oa2a stop
```

---

## 功能特性

- ✅ **流式响应** - 通过 SSE 实时流式输出
- ✅ **工具调用** - 本地模型函数调用支持
- ✅ **视觉模型** - 支持多模态视觉模型输入
- ✅ **网页搜索** - 给本地模型联网能力（见下文）
- ✅ **思考模式** - 支持推理/思考模型输出

---

## 网页搜索能力 🔍

**弥补差距：让你的本地大模型也能享受 Claude Code 的网页搜索功能！**

在 Claude Code 中使用本地部署的模型时，你会失去内置的网页搜索工具。本代理通过 [Tavily](https://tavily.com) 提供的服务端搜索实现来弥补这一差距。

### 问题所在

| 场景 | 网页搜索可用？ |
|----------|----------------------|
| 在 Claude Code 中使用 Claude（云端） | ✅ 内置支持 |
| 在 Claude Code 中使用本地 vLLM/SGLang | ❌ 不可用 |
| **使用本代理 + 本地模型** | ✅ **通过 Tavily 启用** |

### 工作原理

```
Claude Code → Anthropic SDK → 本代理 → 本地模型
                                      ↓
                                 Tavily API (网页搜索)
```

代理直接拦截 `web_search_20250305` 工具调用并处理，无论本地模型是否原生支持网页搜索。

### 配置 Tavily 搜索

1. **免费获取 API Key**：[tavily.com](https://tavily.com) 注册即可，有 generous 的免费额度

2. **配置代理：**
```bash
export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1
export OA2A_OPENAI_API_KEY=dummy
export OA2A_TAVILY_API_KEY="tvly-your-tavily-api-key"  # 启用网页搜索

oa2a
```

3. **在应用中使用：**
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
            "description": "搜索网页获取实时信息",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        }
    ],
    messages=[{"role": "user", "content": "今天 AI 圈发生了什么？"}],
)

if message.stop_reason == "tool_use":
    tool_use = message.content[-1]
    print(f"正在搜索: {tool_use.input}")
    # 代理自动调用 Tavily 并返回结果
```

### Tavily 配置选项

| 变量 | 默认值 | 说明 |
|----------|---------|-------------|
| `OA2A_TAVILY_API_KEY` | - | Tavily API Key（[tavily.com 免费获取](https://tavily.com)） |
| `OA2A_TAVILY_MAX_RESULTS` | 5 | 返回搜索结果数量 |
| `OA2A_TAVILY_TIMEOUT` | 30 | 搜索超时时间（秒） |
| `OA2A_WEBSEARCH_MAX_USES` | 5 | 每次请求最大搜索次数 |

---

## 配置选项

配置存储在 `~/.oa2a/config.toml`（首次运行时自动创建）。

| 选项 | 必需 | 默认值 | 说明 |
|----------|----------|---------|-------------|
| `openai_base_url` | ✅ | - | 本地模型的 OpenAI 兼容端点 |
| `openai_api_key` | ✅ | - | 本地 LLM 后端的 API 密钥 |
| `port` | ❌ | 8080 | 代理服务器端口 |
| `host` | ❌ | 0.0.0.0 | 代理服务器主机 |
| `api_key` | ❌ | - | 访问本代理的认证密钥 |
| `tavily_api_key` | ❌ | - | 启用网页搜索（[tavily.com](https://tavily.com)） |
| `log_level` | ❌ | INFO | 日志级别（DEBUG、INFO、WARNING、ERROR） |

**首次配置**：在没有配置文件的情况下运行 `oa2a` 可启动交互式配置向导。

---

## 架构

```
你的应用 (Claude SDK)
         │
         ▼
┌─────────────────────┐
│  local-openai2anthropic  │  ← 本代理
│  (端口 8080)        │
└─────────────────────┘
         │
         ▼
你的本地模型服务
(vLLM / SGLang)
(OpenAI 兼容 API)
```

---

## 开发

```bash
git clone https://github.com/dongfangzan/local-openai2anthropic.git
cd local-openai2anthropic
pip install -e ".[dev]"

pytest
```

## 许可证

Apache License 2.0
