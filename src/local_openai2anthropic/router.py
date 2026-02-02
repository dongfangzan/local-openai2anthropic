# SPDX-License-Identifier: Apache-2.0
"""
FastAPI router for Anthropic-compatible Messages API.
"""

import json
import logging
import secrets
import string
from http import HTTPStatus
from typing import Any, AsyncGenerator

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from local_openai2anthropic.config import Settings, get_settings
from local_openai2anthropic.converter import (
    convert_anthropic_to_openai,
    convert_openai_to_anthropic,
)
from local_openai2anthropic.protocol import (
    AnthropicError,
    AnthropicErrorResponse,
    Message,
    MessageCreateParams,
)
from local_openai2anthropic.server_tools import ServerToolRegistry

logger = logging.getLogger(__name__)
router = APIRouter()


def get_request_settings(request: Request) -> Settings:
    """Resolve Settings from the running app when available.

    This allows tests (and embedders) to pass an explicit Settings instance via
    `create_app(settings=...)` without requiring environment variables.
    """
    settings = getattr(getattr(request, "app", None), "state", None)
    if settings is not None and hasattr(settings, "settings"):
        return settings.settings  # type: ignore[return-value]
    return get_settings()


def _generate_server_tool_id() -> str:
    """Generate Anthropic-style server tool use ID (srvtoolu_...)."""
    # Generate 24 random alphanumeric characters
    chars = string.ascii_lowercase + string.digits
    random_part = "".join(secrets.choice(chars) for _ in range(24))
    return f"srvtoolu_{random_part}"


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
                try:
                    error_json = json.loads(error_body.decode())
                    error_msg = error_json.get("error", {}).get(
                        "message", error_body.decode()
                    )
                except json.JSONDecodeError:
                    error_msg = error_body.decode()

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
            finish_reason = None
            input_tokens = 0
            output_tokens = 0
            message_id = None

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
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
                    input_tokens = usage.get("prompt_tokens", 0)

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
                                "input_tokens": usage.get("prompt_tokens", 0),
                                "output_tokens": usage.get("completion_tokens", 0),
                                "cache_creation_input_tokens": None,
                                "cache_read_input_tokens": None,
                            },
                        }
                        logger.debug(
                            f"[Anthropic Stream Event] message_delta: {json.dumps(delta_event, ensure_ascii=False)}"
                        )
                        yield f"event: message_delta\ndata: {json.dumps(delta_event)}\n\n"
                    continue

                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                # Track finish reason (but don't skip - content may also be present)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                # Handle reasoning content (thinking)
                if delta.get("reasoning_content"):
                    reasoning = delta["reasoning_content"]
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
                            "content_block": {"type": "thinking", "thinking": ""},
                        }
                        logger.debug(
                            f"[Anthropic Stream Event] content_block_start (thinking): {json.dumps(start_block, ensure_ascii=False)}"
                        )
                        yield f"event: content_block_start\ndata: {json.dumps(start_block)}\n\n"
                        content_block_started = True
                        current_block_type = "thinking"

                    delta_block = {
                        "type": "content_block_delta",
                        "index": content_block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(delta_block)}\n\n"
                    continue

                # Handle content
                if delta.get("content"):
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

                    delta_block = {
                        "type": "content_block_delta",
                        "index": content_block_index,
                        "delta": {"type": "text_delta", "text": delta["content"]},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(delta_block)}\n\n"

                # Handle tool calls
                tool_calls = delta.get("tool_calls", [])
                if tool_calls:
                    tool_call = tool_calls[0]

                    if tool_call.get("id"):
                        if content_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                            content_block_started = False
                            content_block_index += 1

                        func = tool_call.get("function") or {}
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'tool_use', 'id': tool_call['id'], 'name': func.get('name', ''), 'input': {}}})}\n\n"
                        content_block_started = True
                        current_block_type = "tool_use"

                    elif (tool_call.get("function") or {}).get("arguments"):
                        args = (tool_call.get("function") or {}).get("arguments", "")
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
    import time

    body = json.loads(result.body)
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
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

        elif block_type == "server_tool_use":
            # Preserve official Anthropic block type so clients can count server tool uses.
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': 'server_tool_use', 'id': block.get('id', ''), 'name': block.get('name', ''), 'input': block.get('input', {})}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

        elif block_type == "web_search_tool_result":
            # Stream the tool result as its own content block.
            # Some clients expect `results`, others expect `content`; include both when possible.
            tool_result_block = dict(block)
            if "content" not in tool_result_block and "results" in tool_result_block:
                tool_result_block["content"] = tool_result_block["results"]

            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': tool_result_block})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

        elif block_type == "thinking":
            # Handle thinking blocks (BetaThinkingBlock)
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
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


class ServerToolHandler:
    """Handles server tool execution for non-streaming requests."""

    def __init__(
        self,
        server_tools: list[type],
        configs: dict[str, dict[str, Any]],
        settings: Settings,
    ):
        self.server_tools = {t.tool_name: t for t in server_tools}
        self.configs = configs
        self.settings = settings
        self.usage: dict[str, int] = {}

    def is_server_tool_call(self, tool_call: dict[str, Any]) -> bool:
        """Check if a tool call is for a server tool."""
        func_name = tool_call.get("function", {}).get("name")
        return func_name in self.server_tools

    async def execute_tool(
        self,
        tool_call: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Execute a server tool and return content blocks + tool result message.
        Returns: (content_blocks, tool_result_message)
        """
        func_name = tool_call.get("function", {}).get("name")
        call_id = tool_call.get("id", "")

        tool_class = self.server_tools[func_name]
        config = self.configs.get(tool_class.tool_type, {})

        # Extract call arguments
        args = tool_class.extract_call_args(tool_call)
        if args is None:
            args = {}

        # Execute the tool
        result = await tool_class.execute(call_id, args, config, self.settings)

        # Update usage
        for key, value in result.usage_increment.items():
            self.usage[key] = self.usage.get(key, 0) + value

        # Build content blocks
        content_blocks = tool_class.build_content_blocks(call_id, args, result)

        # Build tool result message for OpenAI
        tool_result_msg = tool_class.build_tool_result_message(call_id, args, result)

        return content_blocks, tool_result_msg


async def _handle_with_server_tools(
    openai_params: dict[str, Any],
    url: str,
    headers: dict[str, str],
    settings: Settings,
    server_tools: list[type],
    model: str,
) -> JSONResponse:
    """Handle request with server tool execution loop."""
    params = dict(openai_params)
    configs = params.pop("_server_tools_config", {})

    handler = ServerToolHandler(server_tools, configs, settings)
    accumulated_content: list[dict[str, Any]] = []

    # Get max_uses from configs (default to settings or 5)
    max_uses = settings.websearch_max_uses
    for config in configs.values():
        if config.get("max_uses"):
            max_uses = config["max_uses"]
            break

    total_tool_calls = 0

    while True:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            try:
                # Log full request for debugging
                logger.debug(
                    f"Request body: {json.dumps(params, indent=2, default=str)[:3000]}"
                )

                response = await client.post(url, headers=headers, json=params)

                if response.status_code != 200:
                    logger.error(
                        f"OpenAI API error: {response.status_code} - {response.text}"
                    )
                    error_response = AnthropicErrorResponse(
                        error=AnthropicError(type="api_error", message=response.text)
                    )
                    return JSONResponse(
                        status_code=response.status_code,
                        content=error_response.model_dump(),
                    )

                completion_data = response.json()
                logger.debug(
                    f"OpenAI response: {json.dumps(completion_data, indent=2)[:500]}..."
                )
                from openai.types.chat import ChatCompletion

                completion = ChatCompletion.model_validate(completion_data)

                # Check for server tool calls
                server_tool_calls = []
                other_tool_calls = []

                tool_calls = completion.choices[0].message.tool_calls
                logger.info(
                    f"Model returned tool_calls: {len(tool_calls) if tool_calls else 0}"
                )

                if tool_calls:
                    for tc in tool_calls:
                        func_name = tc.function.name if tc.function else ""
                        logger.info(f"  Tool call: {func_name}")

                        # Generate Anthropic-style ID for server tools
                        is_server = handler.is_server_tool_call(
                            {
                                "id": tc.id,
                                "function": {"name": func_name, "arguments": ""},
                            }
                        )

                        # Use Anthropic-style ID for server tools, original ID otherwise
                        tool_id = _generate_server_tool_id() if is_server else tc.id

                        tc_dict = {
                            "id": tool_id,
                            "function": {
                                "name": func_name,
                                "arguments": tc.function.arguments
                                if tc.function
                                else "{}",
                            },
                        }
                        logger.info(f"    Is server tool: {is_server}, ID: {tool_id}")
                        if is_server:
                            server_tool_calls.append(tc_dict)
                        else:
                            other_tool_calls.append(tc)

                # No server tool calls - we're done
                logger.info(
                    f"Server tool calls: {len(server_tool_calls)}, Other: {len(other_tool_calls)}"
                )
                if not server_tool_calls:
                    message = convert_openai_to_anthropic(completion, model)

                    if accumulated_content:
                        message_dict = message.model_dump()
                        message_dict["content"] = (
                            accumulated_content + message_dict.get("content", [])
                        )

                        if message_dict.get("usage"):
                            message_dict["usage"]["server_tool_use"] = handler.usage

                        # Log full response for debugging
                        logger.info(
                            f"Response content blocks: {json.dumps(message_dict.get('content', []), ensure_ascii=False)[:1000]}"
                        )
                        logger.info(f"Response usage: {message_dict.get('usage')}")
                        logger.info(f"Server tool use count: {handler.usage}")

                        return JSONResponse(content=message_dict)

                    return JSONResponse(content=message.model_dump())

                # Check max_uses limit
                if total_tool_calls >= max_uses:
                    logger.warning(f"Server tool max_uses ({max_uses}) exceeded")
                    # Return error for each call
                    for call in server_tool_calls:
                        func_name = call.get("function", {}).get("name", "")
                        tool_class = handler.server_tools.get(func_name)
                        if tool_class:
                            from local_openai2anthropic.server_tools import ToolResult

                            error_result = ToolResult(
                                success=False,
                                content=[],
                                error_code="max_uses_exceeded",
                            )
                            error_blocks = tool_class.build_content_blocks(
                                call["id"],
                                {},
                                error_result,
                            )
                            accumulated_content.extend(error_blocks)

                    # Continue with modified messages
                    messages = params.get("messages", [])
                    messages = _add_tool_results_to_messages(
                        messages, server_tool_calls, handler, is_error=True
                    )
                    params["messages"] = messages
                    continue

                # Execute server tools
                messages = params.get("messages", [])
                assistant_tool_calls = []
                tool_results = []

                for call in server_tool_calls:
                    total_tool_calls += 1
                    content_blocks, tool_result = await handler.execute_tool(call)
                    accumulated_content.extend(content_blocks)

                    # Track for assistant message
                    assistant_tool_calls.append(
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                    )
                    tool_results.append(tool_result)

                # Add to messages for next iteration
                messages = _add_tool_results_to_messages(
                    messages, assistant_tool_calls, handler, tool_results=tool_results
                )
                params["messages"] = messages

            except httpx.TimeoutException:
                error_response = AnthropicErrorResponse(
                    error=AnthropicError(
                        type="timeout_error", message="Request timed out"
                    )
                )
                raise HTTPException(
                    status_code=HTTPStatus.GATEWAY_TIMEOUT,
                    detail=error_response.model_dump(),
                )
            except httpx.RequestError as e:
                error_response = AnthropicErrorResponse(
                    error=AnthropicError(type="connection_error", message=str(e))
                )
                raise HTTPException(
                    status_code=HTTPStatus.BAD_GATEWAY,
                    detail=error_response.model_dump(),
                )


def _add_tool_results_to_messages(
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    handler: ServerToolHandler,
    tool_results: list[dict[str, Any]] | None = None,
    is_error: bool = False,
) -> list[dict[str, Any]]:
    """Add assistant tool call and results to messages."""
    messages = list(messages)

    # Add assistant message with tool calls
    # SGLang requires content to be a string, not None
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": "",  # Empty string instead of None for SGLang compatibility
        "tool_calls": tool_calls,
    }
    messages.append(assistant_msg)

    # Add tool results
    if is_error:
        for call in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(
                        {
                            "error": "max_uses_exceeded",
                            "message": "Maximum tool uses exceeded.",
                        }
                    ),
                }
            )
    elif tool_results:
        messages.extend(tool_results)

    return messages


@router.post(
    "/v1/messages",
    response_model=Message,
    responses={
        HTTPStatus.OK.value: {"model": Message},
        HTTPStatus.BAD_REQUEST.value: {"model": AnthropicErrorResponse},
        HTTPStatus.UNAUTHORIZED.value: {"model": AnthropicErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": AnthropicErrorResponse},
    },
)
async def create_message(
    request: Request,
    settings: Settings = Depends(get_request_settings),
) -> JSONResponse | StreamingResponse:
    """
    Create a message using Anthropic-compatible API.
    """
    # Read and parse the request body
    try:
        body_bytes = await request.body()
        body_json = json.loads(body_bytes.decode("utf-8"))
        logger.debug(
            f"[Anthropic Request] {json.dumps(body_json, ensure_ascii=False, indent=2)}"
        )
        anthropic_params = body_json
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {e}")
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error", message=f"Invalid JSON: {e}"
            )
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        error_response = AnthropicErrorResponse(
            error=AnthropicError(type="invalid_request_error", message=str(e))
        )
        return JSONResponse(status_code=400, content=error_response.model_dump())

    # Validate request shape early (avoid making upstream calls for obviously invalid requests)
    if not isinstance(anthropic_params, dict):
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error",
                message="Request body must be a JSON object",
            )
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())

    model_value = anthropic_params.get("model")
    if not isinstance(model_value, str) or not model_value.strip():
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error", message="Model must be a non-empty string"
            )
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())

    messages_value = anthropic_params.get("messages")
    if not isinstance(messages_value, list) or len(messages_value) == 0:
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error",
                message="Messages must be a non-empty list",
            )
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())

    max_tokens_value = anthropic_params.get("max_tokens")
    if not isinstance(max_tokens_value, int):
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error", message="max_tokens is required"
            )
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())

    # Check for server tools
    tools = anthropic_params.get("tools", [])
    enabled_server_tools = ServerToolRegistry.get_enabled_tools(settings)
    server_tool_configs = ServerToolRegistry.extract_server_tools(
        [t if isinstance(t, dict) else t.model_dump() for t in tools]
    )
    has_server_tools = len(server_tool_configs) > 0

    # Convert Anthropic params to OpenAI params
    openai_params_obj = convert_anthropic_to_openai(
        anthropic_params,
        enabled_server_tools=enabled_server_tools if has_server_tools else None,
    )
    openai_params: dict[str, Any] = dict(openai_params_obj)  # type: ignore

    # Log converted OpenAI request (remove internal fields)
    log_params = {k: v for k, v in openai_params.items() if not k.startswith("_")}
    logger.debug(
        f"[OpenAI Request] {json.dumps(log_params, ensure_ascii=False, indent=2)}"
    )

    stream = openai_params.get("stream", False)
    model = openai_params.get("model", "")

    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    if settings.openai_org_id:
        headers["OpenAI-Organization"] = settings.openai_org_id
    if settings.openai_project_id:
        headers["OpenAI-Project"] = settings.openai_project_id

    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"

    # Handle server tools (works in both streaming and non-streaming modes)
    if has_server_tools:
        tool_classes = [t[0] for t in server_tool_configs]
        # Server tools require non-streaming execution internally
        # Force non-streaming for the OpenAI call, then stream the result if needed
        openai_params["stream"] = False
        # Remove stream_options if present (not allowed when stream=False)
        openai_params.pop("stream_options", None)
        result = await _handle_with_server_tools(
            openai_params, url, headers, settings, tool_classes, model
        )

        # If original request was streaming, convert result to streaming format
        if stream:
            return StreamingResponse(
                _convert_result_to_stream(result, model),
                media_type="text/event-stream",
            )
        return result

    if stream:
        client = httpx.AsyncClient(timeout=settings.request_timeout)
        return StreamingResponse(
            _stream_response(client, url, headers, openai_params, model),
            media_type="text/event-stream",
        )
    else:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=openai_params)

                if response.status_code != 200:
                    error_response = AnthropicErrorResponse(
                        error=AnthropicError(type="api_error", message=response.text)
                    )
                    return JSONResponse(
                        status_code=response.status_code,
                        content=error_response.model_dump(),
                    )

                openai_completion = response.json()
                logger.debug(
                    f"[OpenAI Response] {json.dumps(openai_completion, ensure_ascii=False, indent=2)}"
                )

                from openai.types.chat import ChatCompletion

                completion = ChatCompletion.model_validate(openai_completion)
                anthropic_message = convert_openai_to_anthropic(completion, model)

                anthropic_response = anthropic_message.model_dump()
                logger.debug(
                    f"[Anthropic Response] {json.dumps(anthropic_response, ensure_ascii=False, indent=2)}"
                )

                return JSONResponse(content=anthropic_response)

            except httpx.TimeoutException:
                error_response = AnthropicErrorResponse(
                    error=AnthropicError(
                        type="timeout_error", message="Request timed out"
                    )
                )
                raise HTTPException(
                    status_code=HTTPStatus.GATEWAY_TIMEOUT,
                    detail=error_response.model_dump(),
                )
            except httpx.RequestError as e:
                error_response = AnthropicErrorResponse(
                    error=AnthropicError(type="connection_error", message=str(e))
                )
                raise HTTPException(
                    status_code=HTTPStatus.BAD_GATEWAY,
                    detail=error_response.model_dump(),
                )


@router.get("/v1/models")
async def list_models(
    settings: Settings = Depends(get_request_settings),
) -> JSONResponse:
    """List available models (proxied to OpenAI)."""
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    if settings.openai_org_id:
        headers["OpenAI-Organization"] = settings.openai_org_id

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{settings.openai_base_url.rstrip('/')}/models",
                headers=headers,
            )
            return JSONResponse(
                status_code=response.status_code,
                content=response.json(),
            )
        except httpx.RequestError as e:
            error_response = AnthropicErrorResponse(
                error=AnthropicError(type="connection_error", message=str(e))
            )
            raise HTTPException(
                status_code=HTTPStatus.BAD_GATEWAY,
                detail=error_response.model_dump(),
            )


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    settings: Settings = Depends(get_request_settings),
) -> JSONResponse:
    """
    Count tokens in messages without creating a message.
    Uses tiktoken for local token counting.
    """
    try:
        body_bytes = await request.body()
        body_json = json.loads(body_bytes.decode("utf-8"))
        logger.debug(
            f"[Count Tokens Request] {json.dumps(body_json, ensure_ascii=False, indent=2)}"
        )
    except json.JSONDecodeError as e:
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error", message=f"Invalid JSON: {e}"
            )
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())
    except Exception as e:
        error_response = AnthropicErrorResponse(
            error=AnthropicError(type="invalid_request_error", message=str(e))
        )
        return JSONResponse(status_code=400, content=error_response.model_dump())

    # Validate required fields
    if not isinstance(body_json, dict):
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error",
                message="Request body must be a JSON object",
            )
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())

    messages = body_json.get("messages", [])
    if not isinstance(messages, list):
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error", message="messages must be a list"
            )
        )
        return JSONResponse(status_code=422, content=error_response.model_dump())

    model = body_json.get("model", "")
    system = body_json.get("system")
    tools = body_json.get("tools", [])

    try:
        # Use tiktoken for token counting
        import tiktoken

        # Map model names to tiktoken encoding
        # Claude models don't have direct tiktoken encodings, so we use cl100k_base as approximation
        encoding = tiktoken.get_encoding("cl100k_base")

        total_tokens = 0

        # Count system prompt tokens if present
        if system:
            if isinstance(system, str):
                total_tokens += len(encoding.encode(system))
            elif isinstance(system, list):
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total_tokens += len(encoding.encode(block.get("text", "")))

        # Count message tokens
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_tokens += len(encoding.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total_tokens += len(encoding.encode(block.get("text", "")))
                        elif block.get("type") == "image":
                            # Images are typically counted as a fixed number of tokens
                            # This is an approximation
                            total_tokens += 85  # Standard approximation for images

        # Count tool definitions tokens
        if tools:
            for tool in tools:
                tool_def = tool if isinstance(tool, dict) else tool.model_dump()
                # Rough approximation for tool definitions
                total_tokens += len(encoding.encode(json.dumps(tool_def)))

        logger.debug(f"[Count Tokens Response] input_tokens: {total_tokens}")

        return JSONResponse(content={"input_tokens": total_tokens})

    except Exception as e:
        logger.error(f"Token counting error: {e}")
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="internal_error", message=f"Failed to count tokens: {str(e)}"
            )
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
