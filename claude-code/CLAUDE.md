# Claude Code Docker Image

预构建的 Claude Code 镜像，开箱即用。

## 镜像地址

```
dongfangzan/claude-code:latest
```

## 包含内容

- **Node.js**: v20 + npm
- **Python**: 3.11 + pip
- **Claude Code**: 最新版
- **沙箱工具**: bubblewrap, socat, ripgrep
- **开发工具**: git, curl, vim, nano, build-essential

## 快速使用

### 方式1: 使用 docker-compose (推荐)

```bash
# 在项目根目录
docker-compose up -d
docker-compose exec claude-code claude --dangerously-skip-permissions
```

### 方式2: 直接使用 docker run

```bash
docker run -it --rm \
  -e ANTHROPIC_BASE_URL=http://your-oa2a-server:8080 \
  -e CLAUDE_MODEL=your-model \
  -v $(pwd):/workspace \
  dongfangzan/claude-code:latest \
  claude --dangerously-skip-permissions
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ANTHROPIC_BASE_URL` | http://oa2a:8080 | OA2A 代理地址 |
| `CLAUDE_MODEL` | kimi-k2.5 | 默认模型 |
| `CLAUDE_OPUS_MODEL` | kimi-k2.5 | Opus 模型 |
| `CLAUDE_SONNET_MODEL` | kimi-k2.5 | Sonnet 模型 |
| `CLAUDE_HAIKU_MODEL` | kimi-k2.5 | Haiku 模型 |
| `CLAUDE_REASONING_MODEL` | kimi-k2.5 | 推理模型 |
| `API_TIMEOUT_MS` | 120000 | API 超时 |

## 本地构建

```bash
docker build -t claude-code:latest .
```
