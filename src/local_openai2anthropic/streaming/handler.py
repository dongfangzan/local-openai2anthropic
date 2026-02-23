# SPDX-License-Identifier: Apache-2.0
"""Streaming response handlers."""

import json
import logging
import time
from typing import Any, AsyncGenerator

import httpx
from fastapi.responses import JSONResponse

from local_openai2anthropic.protocol import AnthropicError, AnthropicErrorResponse
from local_openai2anthropic.utils.tokens import (
    _chunk_text,
    _count_tokens,
    _estimate_input_tokens,
)

logger = logging.getLogger(__name__)


async def _stream_response(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    json_data: dict,
    model: str,
) -> AsyncGenerator[str, None]:
    """
    Stream response from OpenAI and convert to Anthropic format.
    """
    try:
        async with client.stream(
            "POST", url, headers=headers, json=json_data
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="replace").strip()
                try:
                    error_json = json.loads(error_text) if error_text else {}
                    error_msg = error_json.get("error", {}).get("message") or error_text
                except json.JSONDecodeError:
                    error_msg = error_text
                if not error_msg:
                    error_msg = (
                        response.reason_phrase
                        or f"Upstream API error ({response.status_code})"
                    )

                error_event = AnthropicErrorResponse(
                    error=AnthropicError(type="api_error", message=error_msg)
                )
                yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Process SSE stream
            first_chunk = True
            content_block_started = False
            content_block_index = 0
            current_block_type = None  # 'thinking', 'text', or 'tool_use'
            current_tool_call_index = None
            tool_call_buffers: dict[int, str] = {}
            finish_reason = None
            input_tokens = _estimate_input_tokens(json_data)
            output_tokens = 0
            message_id = None
            sent_message_delta = False
            pending_text_prefix = ""

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
                    if not sent_message_delta:
                        stop_reason_map = {
                            "stop": "end_turn",
                            "length": "max_tokens",
                            "tool_calls": "tool_use",
                        }
                        delta_event = {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": stop_reason_map.get(
                                    finish_reason or "stop", "end_turn"
                                )
                            },
                            "usage": {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "cache_creation_input_tokens": None,
                                "cache_read_input_tokens": None,
                            },
                        }
                        logger.debug(
                            f"[Anthropic Stream Event] message_delta: {json.dumps(delta_event, ensure_ascii=False)}"
                        )
                        yield f"event: message_delta\ndata: {json.dumps(delta_event)}\n\n"
                    break

                try:
                    chunk = json.loads(data)
                    logger.debug(
                        f"[OpenAI Stream Chunk] {json.dumps(chunk, ensure_ascii=False)}"
                    )
                except json.JSONDecodeError:
                    continue

                # First chunk: message_start
                if first_chunk:
                    message_id = chunk.get("id", "")
                    usage = chunk.get("usage") or {}
                    input_tokens = usage.get("prompt_tokens", input_tokens)

                    start_event = {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": input_tokens,
                                "output_tokens": 0,
                                "cache_creation_input_tokens": None,
                                "cache_read_input_tokens": None,
                            },
                        },
                    }
                    logger.debug(
                        f"[Anthropic Stream Event] message_start: {json.dumps(start_event, ensure_ascii=False)}"
                    )
                    yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"
                    first_chunk = False
                    continue

                # Handle usage-only chunks
                if not chunk.get("choices"):
                    usage = chunk.get("usage") or {}
                    if usage:
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)
                        if content_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                            content_block_started = False

                        stop_reason_map = {
                            "stop": "end_turn",
                            "length": "max_tokens",
                            "tool_calls": "tool_use",
                        }
                        delta_event = {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": stop_reason_map.get(
                                    finish_reason or "stop", "end_turn"
                                )
                            },
                            "usage": {
                                "input_tokens": usage.get(
                                    "prompt_tokens", input_tokens
                                ),
                                "output_tokens": usage.get("completion_tokens", 0),
                                "cache_creation_input_tokens": None,
                                "cache_read_input_tokens": None,
                            },
                        }
                        logger.debug(
                            f"[Anthropic Stream Event] message_delta: {json.dumps(delta_event, ensure_ascii=False)}"
                        )
                        yield f"event: message_delta\ndata: {json.dumps(delta_event)}\n\n"
                        sent_message_delta = True
                    continue

                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                # Track finish reason (but don't skip - content may also be present)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                # Handle reasoning content (thinking)
                if delta.get("reasoning_content"):
                    reasoning = delta["reasoning_content"]
                    pending_text_prefix = ""
                    # Start thinking content block if not already started
                    if not content_block_started or current_block_type != "thinking":
                        # Close previous block if exists
                        if content_block_started:
                            stop_block = {
                                "type": "content_block_stop",
                                "index": content_block_index,
                            }
                            logger.debug(
                                f"[Anthropic Stream Event] content_block_stop ({current_block_type}): {json.dumps(stop_block, ensure_ascii=False)}"
                            )
                            yield f"event: content_block_stop\ndata: {json.dumps(stop_block)}\n\n"
                            content_block_index += 1
                        start_block = {
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "",
                            },
                        }
                        logger.debug(
                            f"[Anthropic Stream Event] content_block_start (thinking): {json.dumps(start_block, ensure_ascii=False)}"
                        )
                        yield f"event: content_block_start\ndata: {json.dumps(start_block)}\n\n"
                        content_block_started = True
                        current_block_type = "thinking"

                    for chunk in _chunk_text(reasoning):
                        delta_block = {
                            "type": "content_block_delta",
                            "index": content_block_index,
                            "delta": {"type": "thinking_delta", "thinking": chunk},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(delta_block)}\n\n"
                    continue

                # Handle content
                if isinstance(delta.get("content"), str):
                    content_text = delta.get("content", "")
                    if not content_text:
                        continue
                    if content_text.strip() == "(no content)":
                        continue

                    # Buffer content to detect thinking tags
                    content_buffer = pending_text_prefix + content_text
                    pending_text_prefix = ""

                    # Process content to extract thinking tags
                    # Some models (e.g., GLM-4.7) return thinking as标签 in content
                    while content_buffer:
                        # Check if we have a complete thinking tag
                        thinking_start = content_buffer.find("<think>")
                        thinking_end = content_buffer.find("</think>")

                        logger.debug(
                            f"[Thinking Extraction] content_buffer={repr(content_buffer)}, "
                            f"thinking_start={thinking_start}, thinking_end={thinking_end}"
                        )

                        if thinking_start == -1 and thinking_end == -1:
                            # No thinking tags at all, emit all as text
                            if content_buffer:
                                if not content_block_started or current_block_type != "text":
                                    # Close previous block if exists
                                    if content_block_started:
                                        stop_block = {
                                            "type": "content_block_stop",
                                            "index": content_block_index,
                                        }
                                        logger.debug(
                                            f"[Anthropic Stream Event] content_block_stop ({current_block_type}): {json.dumps(stop_block, ensure_ascii=False)}"
                                        )
                                        yield f"event: content_block_stop\ndata: {json.dumps(stop_block)}\n\n"
                                        content_block_index += 1
                                    start_block = {
                                        "type": "content_block_start",
                                        "index": content_block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    }
                                    logger.debug(
                                        f"[Anthropic Stream Event] content_block_start (text): {json.dumps(start_block, ensure_ascii=False)}"
                                    )
                                    yield f"event: content_block_start\ndata: {json.dumps(start_block)}\n\n"
                                    content_block_started = True
                                    current_block_type = "text"

                                output_tokens += _count_tokens(content_buffer)
                                delta_block = {
                                    "type": "content_block_delta",
                                    "index": content_block_index,
                                    "delta": {"type": "text_delta", "text": content_buffer},
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(delta_block)}\n\n"
                            break

                        # If we have thinking_start but no thinking_end, buffer until we get complete tag
                        if thinking_start != -1 and thinking_end == -1:
                            # Buffer the content and wait for more
                            pending_text_prefix = content_buffer
                            break

                        # If thinking_end comes before thinking_start (or no thinking_start), handle it
                        if thinking_end != -1 and (thinking_start == -1 or thinking_end < thinking_start):
                            # There's text before the thinking end tag (orphan end tag)
                            # This shouldn't happen normally, but handle gracefully
                            if thinking_start == -1:
                                # No start tag but have end tag - treat as text
                                if content_buffer[:thinking_end]:
                                    if not content_block_started or current_block_type != "text":
                                        if content_block_started:
                                            stop_block = {
                                                "type": "content_block_stop",
                                                "index": content_block_index,
                                            }
                                            yield f"event: content_block_stop\ndata: {json.dumps(stop_block)}\n\n"
                                            content_block_index += 1
                                        start_block = {
                                            "type": "content_block_start",
                                            "index": content_block_index,
                                            "content_block": {"type": "text", "text": ""},
                                        }
                                        yield f"event: content_block_start\ndata: {json.dumps(start_block)}\n\n"
                                        content_block_started = True
                                        current_block_type = "text"

                                    delta_block = {
                                        "type": "content_block_delta",
                                        "index": content_block_index,
                                        "delta": {"type": "text_delta", "text": content_buffer[:thinking_end]},
                                    }
                                    yield f"event: content_block_delta\ndata: {json.dumps(delta_block)}\n\n"

                                content_buffer = content_buffer[thinking_end + len("</think>"):]
                                continue

                        # We have both start and end tags
                        # Emit text before thinking tag as text block
                        if thinking_start > 0:
                            text_before = content_buffer[:thinking_start]
                            if text_before:
                                if not content_block_started or current_block_type != "text":
                                    if content_block_started:
                                        stop_block = {
                                            "type": "content_block_stop",
                                            "index": content_block_index,
                                        }
                                        logger.debug(
                                            f"[Anthropic Stream Event] content_block_stop ({current_block_type}): {json.dumps(stop_block, ensure_ascii=False)}"
                                        )
                                        yield f"event: content_block_stop\ndata: {json.dumps(stop_block)}\n\n"
                                        content_block_index += 1
                                    start_block = {
                                        "type": "content_block_start",
                                        "index": content_block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    }
                                    logger.debug(
                                        f"[Anthropic Stream Event] content_block_start (text): {json.dumps(start_block, ensure_ascii=False)}"
                                    )
                                    yield f"event: content_block_start\ndata: {json.dumps(start_block)}\n\n"
                                    content_block_started = True
                                    current_block_type = "text"

                                output_tokens += _count_tokens(text_before)
                                delta_block = {
                                    "type": "content_block_delta",
                                    "index": content_block_index,
                                    "delta": {"type": "text_delta", "text": text_before},
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(delta_block)}\n\n"

                        # Extract thinking content (between标签)
                        thinking_content = content_buffer[
                            thinking_start + len("<think>") : thinking_end
                        ]
                        content_buffer = content_buffer[thinking_end + len("</think>") :]

                        # Close previous text block if exists
                        if content_block_started and current_block_type == "text":
                            stop_block = {
                                "type": "content_block_stop",
                                "index": content_block_index,
                            }
                            logger.debug(
                                f"[Anthropic Stream Event] content_block_stop (text): {json.dumps(stop_block, ensure_ascii=False)}"
                            )
                            yield f"event: content_block_stop\ndata: {json.dumps(stop_block)}\n\n"
                            content_block_index += 1
                            content_block_started = False

                        # Start thinking block
                        start_block = {
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "",
                            },
                        }
                        logger.debug(
                            f"[Anthropic Stream Event] content_block_start (thinking): {json.dumps(start_block, ensure_ascii=False)}"
                        )
                        yield f"event: content_block_start\ndata: {json.dumps(start_block)}\n\n"
                        content_block_started = True
                        current_block_type = "thinking"

                        # Emit thinking content
                        if thinking_content:
                            for chunk in _chunk_text(thinking_content):
                                delta_block = {
                                    "type": "content_block_delta",
                                    "index": content_block_index,
                                    "delta": {"type": "thinking_delta", "thinking": chunk},
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(delta_block)}\n\n"

                        # Close thinking block
                        stop_block = {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        }
                        logger.debug(
                            f"[Anthropic Stream Event] content_block_stop (thinking): {json.dumps(stop_block, ensure_ascii=False)}"
                        )
                        yield f"event: content_block_stop\ndata: {json.dumps(stop_block)}\n\n"
                        content_block_index += 1
                        content_block_started = False
                        current_block_type = None

                # Handle tool calls
                if delta.get("tool_calls"):
                    pending_text_prefix = ""
                    for tool_call in delta["tool_calls"]:
                        tool_call_idx = tool_call.get("index", 0)

                        if tool_call.get("id"):
                            if content_block_started and (
                                current_block_type != "tool_use"
                                or current_tool_call_index != tool_call_idx
                            ):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                                content_block_started = False
                                content_block_index += 1

                            if not content_block_started:
                                func = tool_call.get("function") or {}
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'tool_use', 'id': tool_call['id'], 'name': func.get('name', ''), 'input': {}}})}\n\n"
                                content_block_started = True
                                current_block_type = "tool_use"
                                current_tool_call_index = tool_call_idx
                                tool_call_buffers.setdefault(tool_call_idx, "")

                        if (tool_call.get("function") or {}).get("arguments"):
                            args = (tool_call.get("function") or {}).get(
                                "arguments", ""
                            )
                            if (
                                not content_block_started
                                or current_block_type != "tool_use"
                                or current_tool_call_index != tool_call_idx
                            ):
                                if content_block_started:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                                    content_block_index += 1
                                func = tool_call.get("function") or {}
                                tool_id = tool_call.get("id", "")
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': func.get('name', ''), 'input': {}}})}\n\n"
                                content_block_started = True
                                current_block_type = "tool_use"
                                current_tool_call_index = tool_call_idx
                                tool_call_buffers.setdefault(tool_call_idx, "")
                            tool_call_buffers[tool_call_idx] = (
                                tool_call_buffers.get(tool_call_idx, "") + args
                            )
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n\n"

            # Close final content block
            if content_block_started:
                stop_block = {
                    "type": "content_block_stop",
                    "index": content_block_index,
                }
                logger.debug(
                    f"[Anthropic Stream Event] content_block_stop (final): {json.dumps(stop_block, ensure_ascii=False)}"
                )
                yield f"event: content_block_stop\ndata: {json.dumps(stop_block)}\n\n"

            # Message stop
            stop_event = {"type": "message_stop"}
            logger.debug(
                f"[Anthropic Stream Event] message_stop: {json.dumps(stop_event, ensure_ascii=False)}"
            )
            yield f"event: message_stop\ndata: {json.dumps(stop_event)}\n\n"

    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Stream error: {error_msg}")
        error_event = AnthropicErrorResponse(
            error=AnthropicError(type="internal_error", message=str(e))
        )
        yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"


async def _convert_result_to_stream(
    result: JSONResponse,
    model: str,
) -> AsyncGenerator[str, None]:
    """Convert a JSONResponse to streaming SSE format."""
    body = json.loads(bytes(result.body).decode("utf-8"))
    message_id = body.get("id", f"msg_{int(time.time() * 1000)}")
    content = body.get("content", [])
    usage = body.get("usage", {})
    stop_reason = body.get("stop_reason", "end_turn")

    # Map stop_reason
    stop_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    openai_stop_reason = stop_reason_map.get(stop_reason, "stop")

    # 1. message_start event
    start_event = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": 0,
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            },
        },
    }
    yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"

    # 2. Process content blocks
    for i, block in enumerate(content):
        block_type = block.get("type")

        if block_type == "text":
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            text = block.get("text", "")
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

        elif block_type == "tool_use":
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': 'tool_use', 'id': block.get('id', ''), 'name': block.get('name', ''), 'input': block.get('input', {})}})}\n\n"
            tool_input = block.get("input", {})
            if tool_input:
                input_json = json.dumps(tool_input, ensure_ascii=False)
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'input_json_delta', 'partial_json': input_json}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

        elif block_type == "server_tool_use":
            # Preserve official Anthropic block type so clients can count server tool uses.
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': 'server_tool_use', 'id': block.get('id', ''), 'name': block.get('name', ''), 'input': block.get('input', {})}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

        elif block_type == "web_search_tool_result":
            # Stream the tool result as its own content block.
            tool_result_block = dict(block)
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': tool_result_block})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

        elif block_type == "thinking":
            # Handle thinking blocks (BetaThinkingBlock)
            signature = block.get("signature", "")
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': 'thinking', 'thinking': '', 'signature': signature}})}\n\n"
            thinking_text = block.get("thinking", "")
            if thinking_text:
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'thinking_delta', 'thinking': thinking_text}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

    # 3. message_delta with final usage
    delta_event = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason},
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "server_tool_use": usage.get("server_tool_use"),
        },
    }
    yield f"event: message_delta\ndata: {json.dumps(delta_event)}\n\n"

    # 4. message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
