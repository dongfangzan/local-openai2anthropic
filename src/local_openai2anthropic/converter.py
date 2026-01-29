# SPDX-License-Identifier: Apache-2.0
"""
Core conversion logic between Anthropic and OpenAI formats.
"""

import json
import logging
import time
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)

from anthropic.types import (
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    Message,
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    MessageStopEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from anthropic.types.message_create_params import MessageCreateParams
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import CompletionCreateParams

from local_openai2anthropic.protocol import UsageWithCache
from local_openai2anthropic.server_tools import ServerToolRegistry

logger = logging.getLogger(__name__)


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
    temperature = anthropic_params.get("temperature")
    tool_choice = anthropic_params.get("tool_choice")
    tools = anthropic_params.get("tools")
    top_k = anthropic_params.get("top_k")
    top_p = anthropic_params.get("top_p")
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
            openai_messages.append({"role": "system", "content": system})
        else:
            # Handle list of system blocks
            system_text = ""
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "")
            if system_text:
                openai_messages.append({"role": "system", "content": system_text})

    # Convert conversation messages
    # Handle ValidatorIterator from Pydantic by iterating directly
    msg_count = 0
    if messages:
        for msg in messages:
            converted_messages = _convert_anthropic_message_to_openai(msg)
            openai_messages.extend(converted_messages)
            msg_count += 1
    logger.debug(f"Converted {msg_count} messages, total OpenAI messages: {len(openai_messages)}")

    # Build OpenAI params
    params: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": max_tokens,
        "stream": stream,
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
        for tool_class in (enabled_server_tools or []):
            if tool_class.tool_type in server_tools_config:
                config = server_tools_config[tool_class.tool_type]
                openai_tools.append(tool_class.to_openai_tool(config))

        if openai_tools:
            params["tools"] = openai_tools
        
        # Convert tool_choice
        if tool_choice:
            tc = tool_choice if isinstance(tool_choice, dict) else tool_choice.model_dump()
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
    if thinking and isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type == "enabled":
            # Enable thinking mode for vLLM/SGLang
            params["chat_template_kwargs"] = {"thinking": True}

            # Log if budget_tokens was provided but will be ignored
            budget_tokens = thinking.get("budget_tokens")
            if budget_tokens is not None:
                logger.debug(
                    "thinking.budget_tokens (%s) is accepted but not supported by "
                    "vLLM/SGLang. Using default thinking configuration.",
                    budget_tokens
                )
        else:
            # Default to disabled thinking mode if not explicitly enabled
            params["chat_template_kwargs"] = {"thinking": False}
    else:
        # Default to disabled thinking mode when thinking is not provided
        params["chat_template_kwargs"] = {"thinking": False}

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
    
    for block in content:
        if isinstance(block, str):
            openai_content.append({"type": "text", "text": block})
            continue
            
        block_type = block.get("type") if isinstance(block, dict) else block.type
        
        if block_type == "text":
            text = block.get("text") if isinstance(block, dict) else block.text
            openai_content.append({"type": "text", "text": text})
            
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
                openai_content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
                
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
                
            tool_calls.append({
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(input_data) if isinstance(input_data, dict) else str(input_data),
                },
            })
            
        elif block_type == "tool_result":
            # Tool results need to be separate tool messages
            if isinstance(block, dict):
                tool_use_id = block.get("tool_use_id", "")
                result_content = block.get("content", "")
                is_error = block.get("is_error", False)
            else:
                tool_use_id = block.tool_use_id
                result_content = block.content
                is_error = getattr(block, "is_error", False)
                
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
    choice = completion.choices[0]
    message = choice.message
    
    # Convert content blocks
    content: list[ContentBlock] = []
    
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
    anthropic_stop_reason = stop_reason_map.get(choice.finish_reason or "stop", "end_turn")
    
    # Build usage dict with cache support (if available from upstream)
    usage_dict = None
    if completion.usage:
        usage_dict = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
            "cache_creation_input_tokens": getattr(completion.usage, "cache_creation_input_tokens", None),
            "cache_read_input_tokens": getattr(completion.usage, "cache_read_input_tokens", None),
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


async def convert_openai_stream_to_anthropic(
    stream: AsyncGenerator[ChatCompletionChunk, None],
    model: str,
    enable_ping: bool = False,
    ping_interval: float = 15.0,
) -> AsyncGenerator[dict, None]:
    """
    Convert OpenAI streaming response to Anthropic streaming events.
    
    Args:
        stream: OpenAI chat completion stream
        model: Model name
        enable_ping: Whether to send periodic ping events
        ping_interval: Interval between ping events in seconds
        
    Yields:
        Anthropic MessageStreamEvent objects as dicts
    """
    message_id = f"msg_{int(time.time() * 1000)}"
    first_chunk = True
    content_block_started = False
    content_block_index = 0
    current_tool_call: Optional[dict[str, Any]] = None
    finish_reason: Optional[str] = None
    
    # Track usage for final message_delta
    input_tokens = 0
    output_tokens = 0
    
    last_ping_time = time.time()
    
    async for chunk in stream:
        # Send ping events if enabled and interval has passed
        if enable_ping:
            current_time = time.time()
            if current_time - last_ping_time >= ping_interval:
                yield {"type": "ping"}
                last_ping_time = current_time
        
        # First chunk: message_start event
        if first_chunk:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
            
            yield {
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
            first_chunk = False
            continue
        
        # Handle usage-only chunks (last chunk)
        if not chunk.choices:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
                
                # Close any open content block
                if content_block_started:
                    yield {
                        "type": "content_block_stop",
                        "index": content_block_index,
                    }
                
                # Message delta with final usage
                stop_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                }
                yield {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason_map.get(finish_reason or "stop", "end_turn"),
                    },
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_creation_input_tokens": getattr(chunk.usage, "cache_creation_input_tokens", None),
                        "cache_read_input_tokens": getattr(chunk.usage, "cache_read_input_tokens", None),
                    },
                }
            continue
        
        choice = chunk.choices[0]
        delta = choice.delta

        # Track finish reason
        if choice.finish_reason:
            finish_reason = choice.finish_reason
            continue

        # Handle reasoning content (thinking)
        if delta.reasoning_content:
            reasoning = delta.reasoning_content
            # Start thinking content block if not already started
            if not content_block_started or content_block_index == 0:
                # We need a separate index for thinking block
                if content_block_started:
                    # Close previous block
                    yield {
                        "type": "content_block_stop",
                        "index": content_block_index,
                    }
                    content_block_index += 1
                yield {
                    "type": "content_block_start",
                    "index": content_block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                }
                content_block_started = True

            yield {
                "type": "content_block_delta",
                "index": content_block_index,
                "delta": {"type": "thinking_delta", "thinking": reasoning},
            }
            continue

        # Handle content
        if delta.content:
            if not content_block_started:
                # Start text content block
                yield {
                    "type": "content_block_start",
                    "index": content_block_index,
                    "content_block": {"type": "text", "text": ""},
                }
                content_block_started = True
            
            if delta.content:
                yield {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {"type": "text_delta", "text": delta.content},
                }
        
        # Handle tool calls
        if delta.tool_calls:
            tool_call = delta.tool_calls[0]
            
            if tool_call.id:
                # Close previous content block if any
                if content_block_started:
                    yield {
                        "type": "content_block_stop",
                        "index": content_block_index,
                    }
                    content_block_started = False
                    content_block_index += 1
                
                # Start new tool_use block
                current_tool_call = {
                    "id": tool_call.id,
                    "name": tool_call.function.name if tool_call.function else "",
                    "arguments": "",
                }
                yield {
                    "type": "content_block_start",
                    "index": content_block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name if tool_call.function else "",
                        "input": {},
                    },
                }
                content_block_started = True
                
            elif tool_call.function and tool_call.function.arguments:
                # Continue tool call arguments
                args = tool_call.function.arguments
                current_tool_call["arguments"] += args
                yield {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {"type": "input_json_delta", "partial_json": args},
                }
    
    # Close final content block
    if content_block_started:
        yield {
            "type": "content_block_stop",
            "index": content_block_index,
        }
    
    # Message stop event
    yield {"type": "message_stop"}
