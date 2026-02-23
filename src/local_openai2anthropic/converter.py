# SPDX-License-Identifier: Apache-2.0
"""
Core conversion logic between Anthropic and OpenAI formats.
"""

import json
import logging
from typing import Any, Optional

from anthropic.types import (
    ContentBlock,
    Message,
    MessageParam,
    TextBlock,
    ToolUseBlock,
)
from anthropic.types.message_create_params import MessageCreateParams
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import CompletionCreateParams

from local_openai2anthropic.protocol import UsageWithCache

logger = logging.getLogger(__name__)

# Pattern to match Claude Code billing header in system prompts
# Format: x-anthropic-billing-header:cc version=...;cc_entrypoint=...;cch=...;
# The cch value changes on each request, breaking cache hits
CLAUDE_BILLING_HEADER_PATTERN = "x-anthropic-billing-header"


def _strip_claude_billing_header(text: str) -> str:
    """Strip Claude Code billing header from system prompt text.

    Claude Code 2.1.37+ adds a billing header like:
    'x-anthropic-billing-header:cc version=2.1.37.3a3;cc_entrypoint=claude-vscode;cch=xxxxx;'

    The 'cch' value changes on each request, which breaks upstream API caching.
    We remove this header to restore cache hit rates.
    """
    if not text:
        return text

    # Check if the billing header is present
    if CLAUDE_BILLING_HEADER_PATTERN not in text.lower():
        return text

    # Use regex to remove the billing header
    # The billing header format is:
    # x-anthropic-billing-header:cc version=X.X.X;cc_entrypoint=XXX;cch=XXXX;
    # It always starts with "x-anthropic-billing-header:" and ends with a semicolon
    import re

    # Match the entire billing header starting from "x-anthropic-billing-header:"
    # The header ends with the last semicolon before a newline or end of string
    # We need to match everything from "x-anthropic-billing-header:" to the semicolon
    # that follows "cch=..." (the last parameter)
    #
    # The pattern is:
    # - Start with "x-anthropic-billing-header:"
    # - Match everything until we find "cch=..." followed by a semicolon
    pattern = r"x-anthropic-billing-header:.*?cch=[a-zA-Z0-9]+;?"

    result = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Clean up any double newlines or leading/trailing whitespace
    result = re.sub(r"\n\n+", "\n", result)
    result = result.strip()

    logger.debug(
        "Stripped Claude billing header: original length=%d, cleaned length=%d",
        len(text),
        len(result),
    )
    return result


def convert_anthropic_to_openai(
    anthropic_params: MessageCreateParams,
    enabled_server_tools: list[type] | None = None,
) -> CompletionCreateParams:
    """
    Convert Anthropic MessageCreateParams to OpenAI CompletionCreateParams.

    Args:
        anthropic_params: Anthropic message creation parameters
        enabled_server_tools: List of enabled server tool classes

    Returns:
        OpenAI completion create parameters
    """
    # Extract parameters
    model = anthropic_params.get("model")
    messages = anthropic_params.get("messages", [])
    max_tokens = anthropic_params.get("max_tokens", 4096)
    system = anthropic_params.get("system")
    stop_sequences = anthropic_params.get("stop_sequences")
    stream = anthropic_params.get("stream", False)
    temperature = anthropic_params.get("temperature", 0.6)
    tool_choice = anthropic_params.get("tool_choice")
    tools = anthropic_params.get("tools")
    top_k = anthropic_params.get("top_k")
    top_p = anthropic_params.get("top_p", 0.95)
    repetition_penalty = anthropic_params.get("repetition_penalty", 1.1)
    thinking = anthropic_params.get("thinking")
    # metadata is accepted but not forwarded to OpenAI

    # Extract server tool configurations using registry
    server_tools_config: dict[str, dict[str, Any]] = {}
    if enabled_server_tools and tools:
        for tool_class in enabled_server_tools:
            for tool in tools:
                tool_def = tool if isinstance(tool, dict) else tool.model_dump()
                config = tool_class.extract_config(tool_def)
                if config is not None:
                    server_tools_config[tool_class.tool_type] = config
                    break

    # Convert messages
    openai_messages: list[dict[str, Any]] = []

    # Add system message if provided
    if system:
        if isinstance(system, str):
            # Strip Claude billing header to restore cache hit rates
            cleaned_system = _strip_claude_billing_header(system)
            if cleaned_system:
                openai_messages.append({"role": "system", "content": cleaned_system})
        else:
            # Handle list of system blocks
            system_text = ""
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "")
            if system_text:
                # Strip Claude billing header from the combined system text
                cleaned_system = _strip_claude_billing_header(system_text)
                if cleaned_system:
                    openai_messages.append({"role": "system", "content": cleaned_system})

    # Convert conversation messages
    # Handle ValidatorIterator from Pydantic by iterating directly
    msg_count = 0
    if messages:
        for msg in messages:
            converted_messages = _convert_anthropic_message_to_openai(msg)
            openai_messages.extend(converted_messages)
            msg_count += 1
    logger.debug(
        f"Converted {msg_count} messages, total OpenAI messages: {len(openai_messages)}"
    )

    # Build OpenAI params
    params: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "repetition_penalty": repetition_penalty,
    }

    # Always include usage in stream for accurate token counting
    if stream:
        params["stream_options"] = {"include_usage": True}

    if stop_sequences:
        params["stop"] = stop_sequences
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if top_k is not None:
        params["top_k"] = top_k

    # Convert tools
    if tools:
        openai_tools: list[ChatCompletionToolParam] = []
        server_tool_types = set(server_tools_config.keys())

        for tool in tools:
            tool_def = tool if isinstance(tool, dict) else tool.model_dump()
            tool_type = tool_def.get("type")

            # Skip server tools - they are handled separately
            if tool_type in server_tool_types:
                continue

            openai_tool: ChatCompletionToolParam = {
                "type": "function",
                "function": {
                    "name": tool_def.get("name", ""),
                    "description": tool_def.get("description", ""),
                    "parameters": tool_def.get("input_schema", {}),
                },
            }
            openai_tools.append(openai_tool)

        # Add server tools as OpenAI function tools
        for tool_class in enabled_server_tools or []:
            if tool_class.tool_type in server_tools_config:
                config = server_tools_config[tool_class.tool_type]
                openai_tools.append(tool_class.to_openai_tool(config))

        if openai_tools:
            params["tools"] = openai_tools

        # Convert tool_choice
        if tool_choice:
            tc = (
                tool_choice
                if isinstance(tool_choice, dict)
                else tool_choice.model_dump()
            )
            tc_type = tc.get("type")
            if tc_type == "auto":
                params["tool_choice"] = "auto"
            elif tc_type == "any":
                params["tool_choice"] = "required"
            elif tc_type == "tool":
                params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc.get("name", "")},
                }
        else:
            params["tool_choice"] = "auto"

    # Handle thinking parameter
    # vLLM/SGLang use chat_template_kwargs.thinking to toggle thinking mode
    # Some models use "thinking", others use "enable_thinking", so we include both
    # clear_thinking: false keeps the thinking content in the conversation history
    if thinking and isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type == "enabled":
            # Enable thinking mode - include both variants for compatibility
            params["chat_template_kwargs"] = {
                "thinking": True,
                "enable_thinking": True,
                "clear_thinking": False,
            }

            # Log if budget_tokens was provided but will be ignored
            budget_tokens = thinking.get("budget_tokens")
            if budget_tokens is not None:
                logger.debug(
                    "thinking.budget_tokens (%s) is accepted but not supported by "
                    "vLLM/SGLang. Using default thinking configuration.",
                    budget_tokens,
                )
        elif thinking_type == "adaptive":
            # Adaptive thinking mode - let the model decide
            params["chat_template_kwargs"] = {
                "thinking": True,
                "enable_thinking": True,
                "clear_thinking": False,
            }
        else:
            # Default to disabled thinking mode if not explicitly enabled
            params["chat_template_kwargs"] = {
                "thinking": False,
                "enable_thinking": False,
            }
    else:
        # Default to disabled thinking mode when thinking is not provided
        params["chat_template_kwargs"] = {
            "thinking": False,
            "enable_thinking": False,
        }

    # Store server tool configs for later use by router
    if server_tools_config:
        params["_server_tools_config"] = server_tools_config

    return params  # type: ignore


def _convert_anthropic_message_to_openai(
    msg: MessageParam,
) -> list[dict[str, Any]]:
    """
    Convert a single Anthropic message to OpenAI format.

    Returns a list of messages because tool_results need to be
    separate tool messages in OpenAI format.
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if isinstance(content, str):
        return [{"role": role, "content": content}]

    # Handle list of content blocks
    openai_content: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_call_results: list[dict[str, Any]] = []
    reasoning_content: str = ""

    for block in content:
        if isinstance(block, str):
            openai_content.append({"type": "text", "text": block})
            continue

        block_type = block.get("type") if isinstance(block, dict) else block.type

        if block_type == "text":
            text = block.get("text") if isinstance(block, dict) else block.text
            openai_content.append({"type": "text", "text": text})

        elif block_type == "thinking":
            # Extract thinking content to pass as reasoning_content in OpenAI format
            thinking = block.get("thinking") if isinstance(block, dict) else block.thinking
            if thinking:
                reasoning_content += thinking

        elif block_type == "image":
            # Convert image to image_url format
            source = block.get("source") if isinstance(block, dict) else block.source
            if source:
                if isinstance(source, dict):
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                else:
                    media_type = source.media_type
                    data = source.data
                # Build data URL
                url = f"data:{media_type};base64,{data}"
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )

        elif block_type == "tool_use":
            # Convert to function call
            if isinstance(block, dict):
                tool_id = block.get("id", "")
                name = block.get("name", "")
                input_data = block.get("input", {})
            else:
                tool_id = block.id
                name = block.name
                input_data = block.input

            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(input_data)
                        if isinstance(input_data, dict)
                        else str(input_data),
                    },
                }
            )

        elif block_type == "tool_result":
            # Tool results need to be separate tool messages
            if isinstance(block, dict):
                tool_use_id = block.get("tool_use_id", "")
                result_content = block.get("content", "")
                # Note: is_error is not directly supported in OpenAI API
            else:
                tool_use_id = block.tool_use_id
                result_content = block.content
                # Note: is_error is not directly supported in OpenAI API

            # Handle content that might be a list or string
            if isinstance(result_content, list):
                # Extract text from content blocks
                text_parts = []
                for item in result_content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image":
                            # Images in tool results - convert to text representation
                            text_parts.append("[Image content]")
                    else:
                        text_parts.append(str(item))
                result_text = "\n".join(text_parts)
            else:
                result_text = str(result_content)

            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": result_text,
            }
            # Note: is_error is not directly supported in OpenAI API
            # but we could add it to content if needed

            tool_call_results.append(tool_msg)

    # Build primary message
    messages: list[dict[str, Any]] = []
    # SGLang requires content field to be present, default to empty string
    primary_msg: dict[str, Any] = {"role": role, "content": ""}

    if openai_content:
        if len(openai_content) == 1 and openai_content[0]["type"] == "text":
            primary_msg["content"] = openai_content[0]["text"]
        else:
            primary_msg["content"] = openai_content

    if tool_calls:
        primary_msg["tool_calls"] = tool_calls

    # Add reasoning_content if thinking block was present
    if reasoning_content:
        primary_msg["reasoning_content"] = reasoning_content

    messages.append(primary_msg)

    # Add tool result messages separately
    messages.extend(tool_call_results)

    return messages


def _build_usage_with_cache(
    prompt_tokens: int,
    completion_tokens: int,
    # These would come from OpenAI API if supported
    cache_creation_input_tokens: Optional[int] = None,
    cache_read_input_tokens: Optional[int] = None,
) -> UsageWithCache:
    """Build usage object with optional cache token counts."""
    return UsageWithCache(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


def convert_openai_to_anthropic(
    completion: ChatCompletion,
    model: str,
) -> Message:
    """
    Convert OpenAI ChatCompletion to Anthropic Message.

    Args:
        completion: OpenAI chat completion response
        model: Model name

    Returns:
        Anthropic Message response
    """
    from anthropic.types.beta import BetaThinkingBlock

    choice = completion.choices[0]
    message = choice.message

    # Convert content blocks
    content: list[ContentBlock] = []

    # Add reasoning content (thinking) first if present
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        content.append(
            BetaThinkingBlock(
                type="thinking",
                thinking=reasoning_content,
                signature="",  # Signature not available from OpenAI format
            )
        )

    # Add text content if present
    if message.content:
        if isinstance(message.content, str):
            content.append(TextBlock(type="text", text=message.content))
        else:
            for part in message.content:
                if part.type == "text":
                    content.append(TextBlock(type="text", text=part.text))

    # Convert tool calls
    if message.tool_calls:
        for tc in message.tool_calls:
            # Handle case where function might be None
            if not tc.function:
                continue

            tool_input: dict[str, Any] = {}
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {"raw": tc.function.arguments}

            content.append(
                ToolUseBlock(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=tool_input,
                )
            )

    # Determine stop reason
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    anthropic_stop_reason = stop_reason_map.get(
        choice.finish_reason or "stop", "end_turn"
    )

    # Build usage dict with cache support (if available from upstream)
    usage_dict = None
    if completion.usage:
        usage_dict = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
            "cache_creation_input_tokens": getattr(
                completion.usage, "cache_creation_input_tokens", None
            ),
            "cache_read_input_tokens": getattr(
                completion.usage, "cache_read_input_tokens", None
            ),
        }

    # Build message dict to avoid Pydantic validation issues
    message_dict = {
        "id": completion.id,
        "type": "message",
        "role": "assistant",
        "content": [block.model_dump() for block in content],
        "model": model,
        "stop_reason": anthropic_stop_reason,
        "stop_sequence": None,
        "usage": usage_dict,
    }

    return Message.model_validate(message_dict)
