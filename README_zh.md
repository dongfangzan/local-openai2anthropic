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

### 2. 启动本地模型服务

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

### 3. 启动代理

**方式 A: 后台运行（推荐）**

```bash
export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1  # 你的本地模型端点
export OA2A_OPENAI_API_KEY=dummy  # 任意值，本地后端通常忽略

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
export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1
export OA2A_OPENAI_API_KEY=dummy

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

1. **创建或编辑 Claude Code 配置文件** `~/.claude/CLAUDE.md`：

```markdown
# Claude Code 配置

## API 设置

- Claude API Base URL: http://localhost:8080
- Claude API Key: dummy-key

## 模型设置

Use model: meta-llama/Llama-2-7b-chat-hf  # 你的本地模型名称
```

2. **或者在运行 Claude Code 前设置环境变量**：

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-key

claude
```

3. **也可以使用 `--api-key` 和 `--base-url` 参数**：

```bash
claude --api-key dummy-key --base-url http://localhost:8080
```

### 完整工作流示例

终端 1 - 启动本地模型：
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf
```

终端 2 - 启动代理：
```bash
export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1
export OA2A_OPENAI_API_KEY=dummy
export OA2A_TAVILY_API_KEY="tvly-your-tavily-api-key"  # 可选：启用网页搜索

oa2a
```

终端 3 - 启动 Claude Code 并使用本地模型：
```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-key

claude
```

现在 Claude Code 将使用你的本地大模型，而不是云端 API。

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

| 变量 | 必需 | 默认值 | 说明 |
|----------|----------|---------|-------------|
| `OA2A_OPENAI_BASE_URL` | ✅ | - | 本地模型的 OpenAI 兼容端点 |
| `OA2A_OPENAI_API_KEY` | ✅ | - | 任意值（本地后端通常忽略） |
| `OA2A_PORT` | ❌ | 8080 | 代理服务器端口 |
| `OA2A_HOST` | ❌ | 0.0.0.0 | 代理服务器主机 |
| `OA2A_TAVILY_API_KEY` | ❌ | - | 启用网页搜索（[tavily.com](https://tavily.com)） |

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
