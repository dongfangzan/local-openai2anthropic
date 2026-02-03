# SPDX-License-Identifier: Apache-2.0
"""Server tool handling."""

import json
import logging
from http import HTTPStatus
from typing import Any

import httpx
from fastapi.responses import JSONResponse

from local_openai2anthropic.config import Settings
from local_openai2anthropic.converter import convert_openai_to_anthropic
from local_openai2anthropic.protocol import AnthropicError, AnthropicErrorResponse
from local_openai2anthropic.server_tools import ServerToolRegistry
from local_openai2anthropic.utils.tokens import (
    _generate_server_tool_id,
    _normalize_usage,
)

logger = logging.getLogger(__name__)


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
        openai_call_id = tool_call.get("openai_id", call_id)

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
        tool_result_msg = tool_class.build_tool_result_message(
            openai_call_id, args, result
        )

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
                    raw_text = response.text
                    try:
                        if not raw_text:
                            raw_text = response.content.decode(
                                "utf-8", errors="replace"
                            )
                    except Exception:
                        raw_text = ""
                    if not raw_text:
                        raw_text = response.reason_phrase or ""
                    error_message = (raw_text or "").strip()
                    error_response = AnthropicErrorResponse(
                        error=AnthropicError(
                            type="api_error",
                            message=error_message
                            or f"Upstream API error ({response.status_code})",
                        )
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
                        func = getattr(tc, "function", None)
                        func_name = func.name if func else ""
                        logger.info(f"  Tool call: {func_name}")

                        # Generate Anthropic-style ID for server tools
                        is_server = handler.is_server_tool_call(
                            {
                                "id": tc.id,
                                "function": {"name": func_name, "arguments": ""},
                            }
                        )

                        # Use Anthropic-style ID for server tools, original ID otherwise
                        client_tool_id = (
                            _generate_server_tool_id() if is_server else tc.id
                        )

                        tc_dict = {
                            "id": client_tool_id,
                            "openai_id": tc.id,
                            "function": {
                                "name": func_name,
                                "arguments": func.arguments if func else "{}",
                            },
                        }
                        logger.info(
                            f"    Is server tool: {is_server}, ID: {client_tool_id}"
                        )
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
                        message_dict["usage"] = _normalize_usage(
                            message_dict.get("usage")
                        )

                        # Log full response for debugging
                        logger.info(
                            f"Response content blocks: {json.dumps(message_dict.get('content', []), ensure_ascii=False)[:1000]}"
                        )
                        logger.info(f"Response usage: {message_dict.get('usage')}")
                        logger.info(f"Server tool use count: {handler.usage}")

                        return JSONResponse(content=message_dict)

                    message_dict = message.model_dump()
                    message_dict["usage"] = _normalize_usage(message_dict.get("usage"))
                    return JSONResponse(content=message_dict)

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
                    assistant_tool_calls = []
                    for call in server_tool_calls:
                        assistant_tool_calls.append(
                            {
                                "id": call.get("openai_id", call.get("id", "")),
                                "type": "function",
                                "function": {
                                    "name": call.get("function", {}).get("name", ""),
                                    "arguments": call.get("function", {}).get(
                                        "arguments", "{}"
                                    ),
                                },
                            }
                        )
                    messages = params.get("messages", [])
                    messages = _add_tool_results_to_messages(
                        messages, assistant_tool_calls, handler, is_error=True
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
                            "id": call.get("openai_id", call.get("id", "")),
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
                return JSONResponse(
                    status_code=HTTPStatus.GATEWAY_TIMEOUT,
                    content=error_response.model_dump(),
                )
            except httpx.RequestError as e:
                error_response = AnthropicErrorResponse(
                    error=AnthropicError(type="connection_error", message=str(e))
                )
                return JSONResponse(
                    status_code=HTTPStatus.BAD_GATEWAY,
                    content=error_response.model_dump(),
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
            tool_call_id = call.get("openai_id", call.get("id", ""))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
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
