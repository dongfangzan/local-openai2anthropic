# Thinking Content Extraction Design

## Overview

Extract `<think>...</think>` tags from OpenAI API responses and convert them to Anthropic's native thinking block format.

## Problem Statement

Some local deployed models (e.g., GLM-4.7) return thinking content embedded in the `content` field as `<think>...</think>` tags instead of using the dedicated `reasoning_content` field. When forwarding to Anthropic API format, these thinking tags should be properly extracted and converted to Anthropic's thinking block format.

## Input Format

### Non-streaming Response

```json
{
  "choices": [{
    "message": {
      "content": "<think>用户用中文说"你好"，这是一个简单的问题。我应该用中文友好地回应。\n</think>\n\n你好！很高兴见到你。有什么我可以帮助你的吗？",
      "reasoning_content": null
    }
  }]
}
```

### Streaming Response

```json
{"choices":[{"delta":{"content":"<think>用户","reasoning_content":null}}]}
{"choices":[{"delta":{"content":"用中文说","reasoning_content":null}}]}
...
{"choices":[{"delta":{"content":"</think>","reasoning_content":null}}]}
{"choices":[{"delta":{"content":"\n\n","reasoning_content":null}}]}
{"choices":[{"delta":{"content":"你好","reasoning_content":null}}]}
```

## Output Format (Anthropic API)

### Non-streaming

```json
{
  "content": [
    {"type": "thinking", "thinking": "用户用中文说"你好"，这是一个简单的问题。我应该用中文友好地回应。", "signature": ""},
    {"type": "text", "text": "\n\n你好！很高兴见到你。有什么我可以帮助你的吗？"}
  ]
}
```

### Streaming

```json
event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "用户用中文说"你好"..."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "\n\n你好！很高兴见到你。"}}
```

## Implementation Plan

### 1. Add Regex Pattern for Thinking Tags

In `converter.py`, add a constant:

```python
# Pattern to match thinking tags in content
THINKING_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
```

### 2. Modify `convert_openai_to_anthropic` (Non-streaming)

Update the function to:

1. Check if `reasoning_content` is null/empty
2. If so, scan `content` for `<think>...</think>` tags
3. Extract thinking content and replace with empty string in text
4. Build content blocks with thinking block first, then text block

### 3. Modify Streaming Handler

Update `_stream_response` in `streaming/handler.py` to:

1. Buffer incoming content chunks
2. Detect `<think>` start tag
3. Switch to thinking block mode when inside tags
4. Emit proper Anthropic SSE events for thinking block
5. Continue with text block after `</think>` closes

### 4. Add Unit Tests

Create tests in `tests/test_converter.py`:

- Test thinking extraction from non-streaming response
- Test thinking extraction from streaming response
- Test edge cases: multiple thinking blocks, empty thinking, no thinking tags
