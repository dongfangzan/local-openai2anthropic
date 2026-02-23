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
- **交错思考** - 支持多轮对话中的推理内容，使用 `<think>` 标记
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

### 方式一：Docker 部署（生产环境推荐）

#### 使用 Docker Hub 镜像（无需构建）

直接拉取并运行官方镜像：

```bash
# 创建 .env 文件
cat > .env << 'EOF'
OA2A_OPENAI_API_KEY=your-openai-api-key
OA2A_OPENAI_BASE_URL=http://host.docker.internal:8000/v1
OA2A_PORT=8080
EOF

# 使用 docker run 运行
docker run -d \
  --name oa2a \
  --env-file .env \
  -p 8080:8080 \
  --restart unless-stopped \
  dongfangzan/local-openai2anthropic:latest

# 或使用 docker-compose（参考仓库中的 docker-compose.yml）
docker-compose up -d
```

可用标签：
- `latest` - 最新稳定版本
- `0.4.0`, `0.4`, `0` - 特定版本标签
- `main` - 最新开发版本

#### 从源码构建

如果你想自己构建镜像：

```bash
# 克隆仓库
git clone https://github.com/dongfangzan/local-openai2anthropic.git
cd local-openai2anthropic

# 创建 .env 配置文件
cat > .env << 'EOF'
OA2A_OPENAI_API_KEY=your-openai-api-key
OA2A_OPENAI_BASE_URL=http://host.docker.internal:8000/v1
OA2A_PORT=8080
EOF

# 使用 Docker Compose 构建并启动
docker-compose up -d --build
```

**Docker 环境变量：**

| 变量 | 必需 | 默认值 | 说明 |
|----------|----------|---------|-------------|
| `OA2A_OPENAI_API_KEY` | ✅ | - | OpenAI API 密钥 |
| `OA2A_OPENAI_BASE_URL` | ✅ | - | 本地 LLM 端点 |
| `OA2A_HOST` | ❌ | 0.0.0.0 | 服务器主机 |
| `OA2A_PORT` | ❌ | 8080 | 服务器端口 |
| `OA2A_API_KEY` | ❌ | - | 代理认证密钥 |
| `OA2A_LOG_LEVEL` | ❌ | INFO | DEBUG、INFO、WARNING、ERROR |
| `OA2A_TAVILY_API_KEY` | ❌ | - | 启用网页搜索 |
| `OA2A_CORS_ORIGINS` | ❌ | * | 允许的 CORS 来源 |

#### Docker Compose 部署

使用 Docker Compose 进行完整配置部署：

```bash
# 克隆仓库
git clone https://github.com/dongfangzan/local-openai2anthropic.git
cd local-openai2anthropic

# 使用环境变量启动
OA2A_OPENAI_API_KEY=your-api-key \
OA2A_OPENAI_BASE_URL=http://host.docker.internal:8000/v1 \
docker-compose up -d
```

**配置方式**（选择一种）：

1. **直接编辑 docker-compose.yml**（最简单，无需环境变量）：
   ```yaml
   environment:
     - OA2A_OPENAI_API_KEY=你的实际API密钥
     - OA2A_OPENAI_BASE_URL=http://localhost:8000/v1
     - OA2A_TAVILY_API_KEY=tvly-your-key  # 可选
   ```

2. **Shell 环境变量**：
   ```bash
   export OA2A_OPENAI_API_KEY=your-api-key
   export OA2A_OPENAI_BASE_URL=http://localhost:8000/v1
   docker-compose up -d
   ```

3. **.env 文件**：
   ```bash
   cp .env.example .env
   # 编辑 .env，然后：docker-compose up -d
   ```

4. **配置文件挂载**：
   ```bash
   mkdir -p config && cp ~/.oa2a/config.toml config/
   # 取消注释 docker-compose.yml 中的 volumes 部分
   docker-compose up -d
   ```

**Docker 常用命令：**

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down

# 代码变更后重新构建
docker-compose up -d --build
```

### 方式二：pip 安装

#### 1. 安装

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

### 3. 启动代理（推荐方式）

运行以下命令以后台模式启动代理：

```bash
oa2a start
```

**首次配置**：如果 `~/.oa2a/config.toml` 不存在，交互式配置向导会引导你完成：
- 输入 OpenAI API Key（用于本地 LLM 后端）
- 输入本地 LLM 的 base URL（如 `http://localhost:8000/v1`）
- 配置服务器 host 和 port（可选）
- 设置代理 API 认证密钥（可选）

配置完成后，服务将在 `http://localhost:8080` 启动。

**守护进程管理命令：**

```bash
oa2a logs               # 显示最后 50 行日志
oa2a logs -f            # 实时跟踪日志 (Ctrl+C 退出)
oa2a status             # 检查服务是否运行
oa2a stop               # 停止后台服务
oa2a restart            # 使用相同配置重启
```

**手动配置**

你也可以直接创建/编辑配置文件 `~/.oa2a/config.toml`：

```toml
# OA2A 配置文件
openai_api_key = "dummy"
openai_base_url = "http://localhost:8000/v1"
host = "0.0.0.0"
port = 8080
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

### 完整工作流示例

确保 `~/.claude/settings.json` 已按上述步骤配置好。

终端 1 - 启动本地模型：
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf
```

终端 2 - 启动代理（后台运行）：
```bash
# 首次运行：交互式配置向导会引导你完成配置
oa2a start
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

- ✅ **流式响应** - SSE 实时流式输出
- ✅ **工具调用** - 本地模型函数调用
- ✅ **视觉模型** - 多模态视觉输入
- ✅ **网页搜索** - 内置 Tavily 搜索
- ✅ **交错思考** - 支持多轮对话中的推理内容，使用 `<think>` 标记

---

## 网页搜索 🔍

使用 [Tavily](https://tavily.com) 为本地模型添加联网能力。

**配置：**

1. 在 [tavily.com](https://tavily.com) 获取免费 API Key

2. 添加到配置 (`~/.oa2a/config.toml`)：
```toml
tavily_api_key = "tvly-your-api-key"
```

3. 在应用中使用 `web_search_20250305` 工具 - 代理自动处理搜索。

**选项：** `tavily_max_results` (默认: 5), `tavily_timeout` (默认: 30), `websearch_max_uses` (默认: 5)

---

## 配置选项

配置文件：`~/.oa2a/config.toml`（首次运行自动创建）

| 选项 | 必需 | 默认值 | 说明 |
|----------|----------|---------|-------------|
| `openai_base_url` | ✅ | - | 本地 LLM 端点（如 `http://localhost:8000/v1`） |
| `openai_api_key` | ✅ | - | 本地 LLM 的 API 密钥 |
| `port` | ❌ | 8080 | 代理端口 |
| `host` | ❌ | 0.0.0.0 | 代理主机 |
| `api_key` | ❌ | - | 访问代理的认证密钥 |
| `tavily_api_key` | ❌ | - | 启用网页搜索 |
| `log_level` | ❌ | INFO | DEBUG、INFO、WARNING、ERROR |

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
