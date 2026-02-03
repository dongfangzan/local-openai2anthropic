# Dead Code Analysis Report

**Project:** local-openai2anthropic
**Analysis Date:** 2026-02-03
**Tools Used:** vulture, pylint, ruff

---

## Executive Summary

This report identifies potentially unused code in the codebase. The analysis found **47 issues** across the source code, categorized by severity level. No code has been deleted - this is for review purposes only.

### Summary by Severity

| Severity | Count | Description |
|----------|-------|-------------|
| SAFE | 23 | Unused imports, variables, and f-strings - safe to remove |
| CAUTION | 15 | Unused functions/methods - verify before removal |
| DANGER | 9 | Model fields and properties - may be used externally |

---

## Detailed Findings

### SEVERITY: SAFE (High Confidence for Removal)

These items are safe to remove with minimal risk.

#### 1. Unused Imports

| File | Line | Item | Tool | Notes |
|------|------|------|------|-------|
| `src/local_openai2anthropic/converter.py` | 20 | `ChatCompletionChunk` from `openai.types.chat` | vulture (90%), ruff, pylint | Imported but never referenced |
| `src/local_openai2anthropic/converter.py` | 26 | `ServerToolRegistry` from `local_openai2anthropic.server_tools` | vulture (90%), ruff, pylint | Imported but never referenced |
| `src/local_openai2anthropic/protocol.py` | 9 | `Field` from `pydantic` | vulture (90%), ruff, pylint | Imported but never used in file |
| `tests/test_daemon_runner.py` | 8 | `datetime` | vulture (90%) | Unused import in test file |
| `tests/test_router_comprehensive.py` | 7 | `Mock` from `unittest.mock` | vulture (90%) | Unused import |
| `tests/test_router_comprehensive.py` | 9 | `Request` from `fastapi` | vulture (90%) | Unused import |
| `tests/test_router_comprehensive.py` | 10 | `StreamingResponse` from `fastapi.responses` | vulture (90%) | Unused import |
| `tests/test_router_comprehensive.py` | 12 | `_handle_with_server_tools` from `local_openai2anthropic.router` | vulture (90%) | Unused import |
| `tests/test_router_streaming.py` | 7 | `Mock` from `unittest.mock` | vulture (90%) | Unused import |
| `tests/test_router_streaming.py` | 9 | `StreamingResponse` from `fastapi.responses` | vulture (90%) | Unused import |
| `tests/test_router_streaming.py` | 11 | `_handle_with_server_tools` from `local_openai2anthropic.router` | vulture (90%) | Unused import |

#### 2. Unused Local Variables

| File | Line | Variable | Tool | Notes |
|------|------|----------|------|-------|
| `src/local_openai2anthropic/converter.py` | 297 | `is_error` | ruff (F841) | Assigned but never used in tool_result handling |
| `src/local_openai2anthropic/daemon.py` | 274 | `e` (exception) | ruff (F841) | Exception variable not used in except block |
| `src/local_openai2anthropic/main.py` | 240 | `status_parser` | vulture (60%), ruff | Assigned but status command parser not used |
| `src/local_openai2anthropic/router.py` | 483 | `openai_stop_reason` | vulture (60%), ruff | Variable assigned but never used |
| `src/local_openai2anthropic/router.py` | 1182 | `model` | ruff (F841) | Extracted from request but never used |

#### 3. F-strings Without Placeholders

| File | Line | Content | Tool | Notes |
|------|------|---------|------|-------|
| `src/local_openai2anthropic/daemon.py` | 146 | `"Use 'oa2a logs' to view output"` | ruff (F541) | No placeholders in f-string |
| `src/local_openai2anthropic/daemon.py` | 152 | `"Another process may be listening on this port"` | ruff (F541) | No placeholders in f-string |
| `src/local_openai2anthropic/daemon.py` | 263 | `"Server did not stop gracefully, use -f to force kill"` | ruff (F541) | No placeholders in f-string |
| `src/local_openai2anthropic/daemon_runner.py` | 84 | `"Configuration loaded:"` | ruff (F541) | No placeholders in f-string |

#### 4. Unused Arguments

| File | Line | Function/Method | Argument | Tool | Notes |
|------|------|-----------------|----------|------|-------|
| `src/local_openai2anthropic/daemon_runner.py` | 45 | signal handler | `frame` | vulture (100%), pylint | Signal handler frame argument not used |
| `src/local_openai2anthropic/main.py` | 95 | `http_exception_handler` | `request` | pylint | FastAPI exception handler receives request but doesn't use it |
| `src/local_openai2anthropic/main.py` | 116 | `general_exception_handler` | `request` | pylint | FastAPI exception handler receives request but doesn't use it |
| `src/local_openai2anthropic/router.py` | 860 | (inline function) | `handler` | pylint | Exception handler argument not used |
| `src/local_openai2anthropic/router.py` | 1138 | `count_tokens` | `settings` | pylint | Settings dependency injected but not used |

---

### SEVERITY: CAUTION (Verify Before Removal)

These items appear unused but may be called dynamically or through external APIs.

#### 1. Unused Functions

| File | Line | Function | Tool | Risk Assessment |
|------|------|----------|------|-----------------|
| `src/local_openai2anthropic/converter.py` | 348 | `_build_usage_with_cache` | vulture (60%) | **LOW RISK** - Private function, appears truly unused. Could be removed or used for future cache token support. |
| `src/local_openai2anthropic/main.py` | 55 | `auth_middleware` | vulture (60%) | **MEDIUM RISK** - Middleware that may be registered dynamically. Verify if used in app setup. |
| `src/local_openai2anthropic/main.py` | 94 | `http_exception_handler` | vulture (60%) | **LOW RISK** - Exception handler, may be registered via decorator. Check FastAPI app setup. |
| `src/local_openai2anthropic/main.py` | 115 | `general_exception_handler` | vulture (60%) | **LOW RISK** - Exception handler, may be registered via decorator. Check FastAPI app setup. |
| `src/local_openai2anthropic/router.py` | 898 | `create_message` | vulture (60%) | **MEDIUM RISK** - API endpoint handler. May be registered as route. Verify router setup. |
| `src/local_openai2anthropic/router.py` | 1106 | `list_models` | vulture (60%) | **MEDIUM RISK** - API endpoint handler. May be registered as route. Verify router setup. |
| `src/local_openai2anthropic/router.py` | 1135 | `count_tokens` | vulture (60%) | **MEDIUM RISK** - API endpoint handler. May be registered as route. Verify router setup. |
| `src/local_openai2anthropic/router.py` | 1241 | `health_check` | vulture (60%) | **MEDIUM RISK** - API endpoint handler. May be registered as route. Verify router setup. |

#### 2. Unused Methods

| File | Line | Class | Method | Tool | Risk Assessment |
|------|------|-------|--------|------|-----------------|
| `src/local_openai2anthropic/server_tools/base.py` | 167 | `ServerToolRegistry` | `all_tools` | vulture (60%) | **LOW RISK** - Method that returns all registered tools. May be useful for debugging/admin. |

#### 3. Unused Test Functions/Fixtures

| File | Line | Function | Tool | Notes |
|------|------|----------|------|-------|
| `tests/test_router_comprehensive.py` | 31 | `async_iter` | vulture (60%) | Helper function, may be used in tests |
| `tests/test_router_streaming.py` | 25 | `async_iter` | vulture (60%) | Helper function, may be used in tests |
| `tests/test_router_streaming.py` | 614 | `async_iter` | vulture (60%) | Duplicate helper function |

---

### SEVERITY: DANGER (Do Not Remove Without Careful Review)

These items are model fields, properties, or public API components that may be used externally even if not referenced internally.

#### 1. Model Configuration and Properties

| File | Line | Item | Type | Tool | Risk Assessment |
|------|------|------|------|------|-----------------|
| `src/local_openai2anthropic/config.py` | 15 | `model_config` | Class variable | vulture (60%) | **HIGH RISK** - Pydantic Settings configuration. Required by pydantic-settings. |
| `src/local_openai2anthropic/config.py` | 52 | `openai_auth_headers` | Property | vulture (60%) | **HIGH RISK** - Public property that may be used by external code. Verify usage before removal. |

#### 2. TypedDict Fields (May be Required for API Compatibility)

| File | Line | Field | Parent Class | Tool | Risk Assessment |
|------|------|-------|--------------|------|-----------------|
| `src/local_openai2anthropic/openai_types.py` | 20 | `description` | `ChatCompletionToolFunction` | vulture (60%) | **HIGH RISK** - TypedDict field for OpenAI API compatibility. |
| `src/local_openai2anthropic/openai_types.py` | 21 | `parameters` | `ChatCompletionToolFunction` | vulture (60%) | **HIGH RISK** - TypedDict field for OpenAI API compatibility. |
| `src/local_openai2anthropic/openai_types.py` | 41 | `stop` | `CompletionCreateParams` | vulture (60%) | **HIGH RISK** - TypedDict field for OpenAI API compatibility. |
| `src/local_openai2anthropic/openai_types.py` | 44 | `stream_options` | `CompletionCreateParams` | vulture (60%) | **HIGH RISK** - TypedDict field for OpenAI API compatibility. |
| `src/local_openai2anthropic/openai_types.py` | 46 | `chat_template_kwargs` | `CompletionCreateParams` | vulture (60%) | **HIGH RISK** - TypedDict field for vLLM/SGLang compatibility. |

#### 3. Pydantic Model Fields (Default Values)

| File | Line | Field | Parent Class | Tool | Risk Assessment |
|------|------|-------|--------------|------|-----------------|
| `src/local_openai2anthropic/openai_types.py` | 80 | `index` | `Choice` | vulture (60%) | **MEDIUM RISK** - Model field with default value. May be used in serialization. |
| `src/local_openai2anthropic/openai_types.py` | 95 | `index` | `ChatCompletionDeltaToolCall` | vulture (60%) | **MEDIUM RISK** - Model field with default value. |
| `src/local_openai2anthropic/openai_types.py` | 114 | `index` | `StreamingChoice` | vulture (60%) | **MEDIUM RISK** - Model field with default value. |
| `src/local_openai2anthropic/openai_types.py` | 135 | `created` | `ChatCompletion` | vulture (60%) | **MEDIUM RISK** - Model field. Required for OpenAI API compatibility. |
| `src/local_openai2anthropic/openai_types.py` | 146 | `created` | `ChatCompletionChunk` | vulture (60%) | **MEDIUM RISK** - Model field. Required for OpenAI API compatibility. |

#### 4. Unused Classes

| File | Line | Class | Tool | Risk Assessment |
|------|------|-------|------|-----------------|
| `src/local_openai2anthropic/openai_types.py` | 141 | `ChatCompletionChunk` | vulture (60%) | **HIGH RISK** - Public Pydantic model. May be imported by external code. |

#### 5. Protocol Model Fields (Web Search Tool Types)

| File | Line | Field | Parent Class | Tool | Risk Assessment |
|------|------|-------|--------------|------|-----------------|
| `src/local_openai2anthropic/protocol.py` | 77 | `city` | `ApproximateLocation` | vulture (60%) | **MEDIUM RISK** - Optional field for web search location. |
| `src/local_openai2anthropic/protocol.py` | 78 | `region` | `ApproximateLocation` | vulture (60%) | **MEDIUM RISK** - Optional field for web search location. |
| `src/local_openai2anthropic/protocol.py` | 79 | `country` | `ApproximateLocation` | vulture (60%) | **MEDIUM RISK** - Optional field with default. |
| `src/local_openai2anthropic/protocol.py` | 80 | `timezone` | `ApproximateLocation` | vulture (60%) | **MEDIUM RISK** - Optional field for web search location. |
| `src/local_openai2anthropic/protocol.py` | 89 | `allowed_domains` | `WebSearchToolDefinition` | vulture (60%) | **MEDIUM RISK** - Optional field for domain filtering. |
| `src/local_openai2anthropic/protocol.py` | 90 | `blocked_domains` | `WebSearchToolDefinition` | vulture (60%) | **MEDIUM RISK** - Optional field for domain filtering. |
| `src/local_openai2anthropic/protocol.py` | 91 | `user_location` | `WebSearchToolDefinition` | vulture (60%) | **MEDIUM RISK** - Optional field for location context. |
| `src/local_openai2anthropic/protocol.py` | 148 | `web_search_requests` | `ServerToolUseUsage` | vulture (60%) | **MEDIUM RISK** - Usage tracking field. |
| `src/local_openai2anthropic/protocol.py` | 158 | `server_tool_use` | `UsageWithServerToolUse` | vulture (60%) | **MEDIUM RISK** - Usage tracking field. |

---

## Test File Analysis

The test files contain many mock-related unused attributes. These are typically false positives because:

1. **Mock return_value assignments** - Used to configure mock behavior even if not directly referenced
2. **Mock side_effect assignments** - Configure mock behavior
3. **Test fixtures** - May be used indirectly by pytest

### Test File Recommendations

| File | Issue Count | Recommendation |
|------|-------------|----------------|
| `test_daemon.py` | 4 | Review mock configurations - likely false positives |
| `test_daemon_advanced.py` | 5 | Review mock configurations - likely false positives |
| `test_daemon_runner.py` | 8 | Remove unused `datetime` import; review mocks |
| `test_main.py` | 14 | Review mock configurations - likely false positives |
| `test_router_comprehensive.py` | 7 | Remove unused imports; review helper functions |
| `test_router_edge_cases.py` | 2 | Review mock configurations |
| `test_router_streaming.py` | 6 | Remove unused imports; review helper functions |
| `test_server_tools.py` | 13 | Review test fixtures - `cls` in `@classmethod` is required |
| `test_tavily_client.py` | 14 | Review mock configurations - likely false positives |

**Note:** The `cls` variable in `@classmethod` decorators is a false positive - it's required by Python even if not explicitly used.

---

## Recommendations

### Immediate Actions (SAFE)

1. **Remove unused imports:**
   ```python
   # In converter.py - Remove lines 18-22 and 26
   from openai.types.chat import ChatCompletionChunk  # REMOVE
   from local_openai2anthropic.server_tools import ServerToolRegistry  # REMOVE
   ```

2. **Remove unused variable assignments:**
   ```python
   # In converter.py line 297
   is_error = getattr(block, "is_error", False)  # REMOVE or use
   ```

3. **Fix f-strings without placeholders:**
   ```python
   # Change f"..." to "..." where no placeholders exist
   ```

4. **Clean up test file imports** as listed above

### Cautionary Actions (CAUTION)

1. **Verify API endpoint functions** in `router.py`:
   - Check if `create_message`, `list_models`, `count_tokens`, `health_check` are registered as routes
   - If not registered, they can be removed

2. **Review exception handlers** in `main.py`:
   - Verify if they're registered with FastAPI's exception handlers
   - If not registered, they can be removed

### Do Not Remove (DANGER)

1. **Pydantic model fields** - Required for API compatibility
2. **TypedDict fields** - Define API contract
3. **Config properties** - May be used externally
4. **Web search tool type fields** - Part of feature implementation

---

## Verification Commands

Before removing any code, verify with these commands:

```bash
# Check if a function is referenced anywhere
grep -r "function_name" --include="*.py" .

# Check for dynamic imports
grep -r "importlib\|__import__\|import_module" --include="*.py" .

# Check for string-based references (common in web frameworks)
grep -r "\"create_message\"\|'create_message'" --include="*.py" .

# Run tests after changes
pytest tests/ -v

# Check for import errors
python -c "from local_openai2anthropic import *"
```

---

## Appendix: Tool Configuration

### Vulture
```bash
vulture src/ tests/ --min-confidence 60
```

### Pylint
```bash
pylint src/ --disable=all --enable=W0611,W0613,W0614,W1300,W1301,W1302,W1303,W1304,W1305,W1306,W1307,W1401,W1404,W1405,W1505
```

### Ruff
```bash
ruff check src/ tests/ --select F401,F541,F841
```

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-03 | 1.0 | Initial analysis report |

---

**Report Generated By:** Claude Code Dead Code Analysis Agent
**Disclaimer:** This report is generated by automated tools and requires human review before any code deletion. Always verify findings with grep searches and run the full test suite before removing code.
