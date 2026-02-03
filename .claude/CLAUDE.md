# Claude 开发指南

## Configuration File

### Config File Location

- **Linux/macOS**: `~/.oa2a/config.toml`
- **Windows**: `%USERPROFILE%\.oa2a\config.toml`

### First Run

A default config file will be created automatically on first startup. Edit this file to add your API keys.

### Configuration Example

```toml
# OA2A Configuration File
# Place this file at ~/.oa2a/config.toml

# OpenAI API Configuration
openai_api_key = "your-openai-api-key"
openai_base_url = "https://api.openai.com/v1"
openai_org_id = ""
openai_project_id = ""

# Server Configuration
host = "0.0.0.0"
port = 8080
request_timeout = 300.0

# API Key for authenticating requests to this server (optional)
api_key = ""

# CORS settings
cors_origins = ["*"]
cors_credentials = true
cors_methods = ["*"]
cors_headers = ["*"]

# Logging
log_level = "DEBUG"
log_dir = ""  # Empty uses platform-specific default

# Tavily Web Search Configuration
tavily_api_key = ""
tavily_timeout = 30.0
tavily_max_results = 5
websearch_max_uses = 5
```

## 版本号更新清单

当修改项目版本号时，需要同步更新以下所有位置：

### 1. pyproject.toml
- **位置**: 第3行
- **格式**: `version = "x.y.z"`
- **示例**: `version = "0.3.5"`

### 2. src/local_openai2anthropic/__init__.py
- **位置**: 第6行
- **格式**: `__version__ = "x.y.z"`
- **示例**: `__version__ = "0.3.5"`

### 3. src/local_openai2anthropic/main.py (FastAPI 应用版本)
- **位置**: 第104行
- **格式**: `version="x.y.z"`
- **示例**: `version="0.3.5"`

### 4. src/local_openai2anthropic/main.py (命令行 --version)
- **位置**: 第254行
- **格式**: `version="%(prog)s x.y.z"`
- **示例**: `version="%(prog)s 0.3.5"`

### 5. tests/test_main.py (测试断言)
- **位置**: 第26行
- **格式**: `assert app.version == "x.y.z"`
- **示例**: `assert app.version == "0.3.5"`

### 6. Git Tag (重要：必须推送才能触发发布)
- **格式**: `vx.y.z`
- **示例**: `v0.3.5`
- **命令**:
  ```bash
  git tag v0.3.5
  git push origin v0.3.5
  ```
- **注意**: 推送 tag 会触发 GitHub Actions 自动发布到 PyPI，不要忘记推送 tag！

## 版本号格式

使用语义化版本控制 (Semantic Versioning)：
- **MAJOR**: 不兼容的 API 修改
- **MINOR**: 向下兼容的功能新增
- **PATCH**: 向下兼容的问题修复

## 发布流程

1. 更新上述所有文件中的版本号
2. 运行测试确保通过: `pytest`
3. 提交更改: `git commit -m "chore(release): bump version to x.y.z"`
4. 推送代码: `git push`
5. **创建并推送标签** (触发 GitHub Actions 发布):
   ```bash
   git tag vx.y.z
   git push origin vx.y.z
   ```
6. GitHub Actions 将自动发布到 PyPI

**重要**: 第 5 步推送 tag 是触发自动发布的关键步骤，不要忘记！

## 代码提交规范

### 测试覆盖率要求

在提交任何新代码之前，必须满足以下测试要求：

#### 1. 新代码单元测试
- **必须**为所有未提交的新代码编写单元测试
- 新代码的测试覆盖率**必须 > 90%**
- 测试文件命名规范: `test_<module_name>.py`
- 测试函数命名规范: `test_<function_name>_<scenario>`

#### 2. 总体覆盖率检查
- 运行 `/everything-claude-code:test-coverage` 检查总体覆盖率
- **总体覆盖率必须 ≥ 80%**
- 如果总体覆盖率低于 80%，需要补充测试或优化现有代码

#### 3. 代码审查

在提交代码前，必须进行代码审查：

```bash
# 运行代码审查（检查安全漏洞、代码质量、最佳实践）
/code-review:code-review

# 根据审查结果修复问题
# - CRITICAL/HIGH 级别问题必须修复
# - MEDIUM/LOW 级别问题根据情况处理
```

代码审查将检查：
- **安全问题**: 硬编码凭证、SQL注入、XSS、路径遍历等
- **代码质量**: 函数长度、嵌套深度、错误处理、console.log等
- **最佳实践**: 不可变模式、emoji使用、测试覆盖、可访问性等

#### 4. 提交流程
```bash
# 1. 编写新代码
# 2. 编写对应的单元测试
# 3. 运行测试并检查覆盖率
pytest --cov=src/local_openai2anthropic --cov-report=term-missing

# 4. 运行 Claude Code 覆盖率检查
/everything-claude-code:test-coverage

# 5. 运行代码审查
/code-review:code-review

# 6. 确认新代码覆盖率 > 90% 且总体覆盖率 ≥ 80%，且无 CRITICAL/HIGH 问题
# 7. 提交代码
git add .
git commit -m "feat: your commit message"
```

#### 4. 测试质量标准
- 测试用例应覆盖正常路径、边界条件和异常情况
- 使用 `pytest` 作为测试框架
- 异步代码使用 `pytest-asyncio`
- 适当的测试夹具 (fixtures) 和参数化测试
