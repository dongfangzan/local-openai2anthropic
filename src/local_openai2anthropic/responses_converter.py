# SPDX-License-Identifier: Apache-2.0
"""
Conversion between OpenAI Responses API and OpenAI Chat Completions API.

The Responses API is what OpenAI's SDK calls ``client.responses.create`` — a
stateful, item-based interface. The Chat Completions API is the older, simpler
``/v1/chat/completions`` interface that most self-hosted backends (vLLM,
SGLang, …) implement.

This module provides two directions:

  * ``convert_responses_to_chat_completion`` — turn a Responses create request
    into a Chat Completions create request.
  * ``convert_chat_completion_to_responses`` — turn a non-streaming Chat
    Completion response back into a Responses ``Response`` object.

Streaming is handled separately in ``_stream_responses`` (see router) because
the Responses SSE event sequence has its own shape.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterable

from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

# Reasoning effort tiers accepted by the Responses API (``reasoning.effort``)
_RESPONSES_EFFORT_LEVELS = {"minimal", "low", "medium", "high"}

# Tool types that mean "web search" in the Responses API. The official SDK
# sends ``web_search_preview`` / ``web_search`` / their dated variants.
_WEB_SEARCH_TOOL_TYPES = {
    "web_search",
    "web_search_preview",
    "web_search_2025_08_26",
    "web_search_preview_2025_03_11",
    "web_search_20250305",  # Anthropic-style alias used by some clients
}

# The OpenAI function name we register for server-side web search. The
# WebSearchServerTool class uses this exact name, so the two sides agree.
WEB_SEARCH_FUNCTION_NAME = "web_search"


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _coerce_effort_to_reasoning_effort(effort: Any) -> str | None:
    """Map a Responses ``reasoning.effort`` value onto a chat-template effort.

    The Responses API tiers (``minimal``/``low``/``medium``/``high``) are
    forwarded verbatim — vLLM/SGLang chat templates accept these same strings
    via ``chat_template_kwargs.reasoning_effort``. ``minimal`` is an alias of
    ``low`` for backends that don't know it.
    """
    if not isinstance(effort, str):
        return None
    effort = effort.lower()
    if effort in _RESPONSES_EFFORT_LEVELS:
        return effort
    return None


def _is_function_tool(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    return tool.get("type") in (None, "function")


def _is_web_search_tool(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    return tool.get("type") in _WEB_SEARCH_TOOL_TYPES


def _extract_web_search_config(tool: Any) -> dict[str, Any]:
    """Extract config from a Responses web_search tool definition."""
    if not isinstance(tool, dict):
        return {}
    return {
        "max_uses": tool.get("max_uses"),
        "allowed_domains": tool.get("allowed_domains"),
        "blocked_domains": tool.get("blocked_domains"),
        "user_location": tool.get("user_location"),
        "search_context_size": tool.get("search_context_size"),
    }


def _convert_responses_tools(tools: Any) -> tuple[list[dict[str, Any]] | None, dict[str, dict[str, Any]]]:
    """Convert Responses tools into Chat Completions tools.

    Returns ``(chat_tools, web_search_configs)`` where ``chat_tools`` is the
    list of OpenAI function tools to forward upstream (web search is NOT
    included — it is handled server-side), and ``web_search_configs`` maps
    ``WebSearchServerTool.tool_type`` to its extracted config (empty when no
    web search tool was present).
    """
    web_search_configs: dict[str, dict[str, Any]] = {}
    if not tools:
        return None, web_search_configs

    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if _is_web_search_tool(tool):
            # Stash config for the server-side handler. The key matches
            # WebSearchServerTool.tool_type so ServerToolRegistry can find it.
            web_search_configs["web_search_20250305"] = _extract_web_search_config(tool)
            continue

        if not _is_function_tool(tool):
            # File search / MCP / computer use / etc. are not supported by a
            # bare chat/completions backend. Drop rather than fail.
            logger.debug("Dropping unsupported Responses tool type: %s", tool.get("type"))
            continue

        # Prefer nested form if present, otherwise read flat fields.
        fn = tool.get("function")
        if isinstance(fn, dict):
            name = fn.get("name", "")
            description = fn.get("description", "")
            parameters = fn.get("parameters", {}) or {}
        else:
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {}) or {}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return out or None, web_search_configs


def web_search_function_tool() -> dict[str, Any]:
    """The OpenAI function-tool definition for server-side web search.

    Registered alongside any client-supplied function tools so the upstream
    model can decide to call ``web_search``.
    """
    return {
        "type": "function",
        "function": {
            "name": WEB_SEARCH_FUNCTION_NAME,
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    }


def build_web_search_output_items(
    call_id: str,
    query: str,
    results: list[dict[str, Any]],
    *,
    error_code: str | None = None,
) -> list[dict[str, Any]]:
    """Build Responses-format output items for a completed web search call.

    Mirrors the OpenAI Responses ``web_search_call`` item shape: one item with
    ``type: "web_search_call"`` carrying the query, status, and (on success)
    a list of search results. On failure, ``error`` is populated instead.
    """
    if error_code:
        return [
            {
                "id": call_id,
                "type": "web_search_call",
                "call_id": call_id,
                "status": "failed",
                "query": query,
                "error": {"code": error_code, "message": f"web search failed: {error_code}"},
            }
        ]
    return [
        {
            "id": call_id,
            "type": "web_search_call",
            "call_id": call_id,
            "status": "completed",
            "query": query,
            "results": results,
        }
    ]


def _convert_tool_choice(tool_choice: Any) -> Any:
    """Map Responses ``tool_choice`` to Chat Completions ``tool_choice``."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            return tool_choice
        return None
    if isinstance(tool_choice, dict):
        t = tool_choice.get("type")
        if t == "function":
            fn = tool_choice.get("function") or {}
            name = fn.get("name", "")
            if name:
                return {"type": "function", "function": {"name": name}}
    return None


def _input_item_to_openai_messages(
    item: Any,
) -> list[dict[str, Any]]:
    """Convert one Responses ``input`` item to one or more chat messages.

    Responses input items can be:
      * ``EasyInputMessage`` — ``{role, content (str|list)}``
      * ``message`` typed input param — same shape with ``type: "message"``
      * ``function_call`` — assistant tool call (already issued)
      * ``function_call_output`` — tool result for a previous call
      * ``reasoning`` — reasoning item (carried forward as assistant reasoning)
      * raw string — treated as a user message
    """
    if isinstance(item, str):
        return [{"role": "user", "content": item}]

    if not isinstance(item, dict):
        return []

    item_type = item.get("type")

    # Message-shaped items (EasyInputMessage or {"type":"message", ...})
    if item_type in (None, "message"):
        role = item.get("role", "user")
        content = item.get("content", "")
        return [_content_to_openai_message(role, content)]

    if item_type == "function_call":
        # An assistant-issued function call in conversation history.
        name = item.get("name", "")
        arguments = item.get("arguments", "")
        try:
            args_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
        except (TypeError, ValueError):
            args_str = str(arguments)
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": item.get("call_id") or item.get("id", ""),
                        "type": "function",
                        "function": {"name": name, "arguments": args_str},
                    }
                ],
            }
        ]

    if item_type == "function_call_output":
        return [
            {
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": _stringify_tool_output(item.get("output")),
            }
        ]

    if item_type == "reasoning":
        # Carry reasoning content forward as assistant reasoning_content so
        # multi-turn reasoning models keep their chain.
        summary = item.get("summary") or []
        reasoning_text = ""
        if isinstance(summary, list):
            for part in summary:
                if isinstance(part, dict):
                    reasoning_text += part.get("text", "") or ""
                elif isinstance(part, str):
                    reasoning_text += part
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "reasoning_text",
                    "input_text",
                    "output_text",
                ):
                    reasoning_text += part.get("text", "") or ""
        if not reasoning_text:
            return []
        return [
            {
                "role": "assistant",
                "content": "",
                "reasoning": reasoning_text,
                "reasoning_content": reasoning_text,
            }
        ]

    # Unknown item types are dropped to keep the conversation shape valid.
    logger.debug("Dropping unsupported Responses input item type: %s", item_type)
    return []


_ORPHAN_TOOL_RESULT_PLACEHOLDER = "[tool result unavailable]"


def _ensure_tool_results_for_tool_calls(
    openai_messages: list[dict[str, Any]],
) -> None:
    """Backfill placeholder ``tool`` messages for any orphaned tool_call_id.

    Mutates ``openai_messages`` in place. vLLM/SGLang reject an assistant
    message carrying ``tool_calls`` if no later ``tool`` message answers each
    ``tool_call_id``. When the client only sent the call (or truncated history),
    insert a neutral placeholder tool message right after the assistant message.
    """
    if not openai_messages:
        return

    answered_ids: set[str] = set()
    for msg in openai_messages:
        if msg.get("role") == "tool":
            tid = msg.get("tool_call_id")
            if tid:
                answered_ids.add(tid)

    insertions: list[tuple[int, dict[str, Any]]] = []
    for idx, msg in enumerate(openai_messages):
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue
        missing: list[str] = []
        for tc in tool_calls:
            tc_id = tc.get("id") if isinstance(tc, dict) else None
            if tc_id and tc_id not in answered_ids:
                missing.append(tc_id)
        if not missing:
            continue
        offset = 1
        for tid in missing:
            insertions.append(
                (
                    idx + offset,
                    {
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": _ORPHAN_TOOL_RESULT_PLACEHOLDER,
                    },
                )
            )
            offset += 1
            answered_ids.add(tid)

    if not insertions:
        return

    for pos, tool_msg in sorted(insertions, key=lambda x: x[0], reverse=True):
        openai_messages.insert(pos, tool_msg)
    logger.debug(
        "Backfilled %d placeholder tool result(s) for orphaned tool_call_id(s)",
        len(insertions),
    )


def _content_to_openai_message(role: str, content: Any) -> dict[str, Any]:
    """Build a single chat message from a Responses message content value."""
    if content is None:
        return {"role": role, "content": ""}
    if isinstance(content, str):
        return {"role": role, "content": content}

    if isinstance(content, list):
        text_parts: list[str] = []
        image_parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("input_text", "text", "output_text"):
                text_parts.append(part.get("text", "") or "")
            elif ptype == "input_image":
                url = part.get("image_url")
                if not url and part.get("file_id"):
                    url = part.get("file_id")
                if url:
                    image_parts.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
        if not image_parts:
            return {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        # Mixed content — use the OpenAI multi-part content format.
        parts: list[dict[str, Any]] = []
        if text_parts:
            parts.append({"type": "text", "text": "\n".join(text_parts)})
        parts.extend(image_parts)
        return {"role": role, "content": parts}

    return {"role": role, "content": str(content)}


def _stringify_tool_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(output)


def convert_responses_to_chat_completion(
    params: dict[str, Any],
) -> dict[str, Any]:
    """Convert a Responses API create request to a Chat Completions request.

    ``params`` is the parsed JSON body of ``POST /v1/responses``.
    Returns a dict suitable for ``POST /v1/chat/completions``.
    """
    model = params.get("model", "")
    input_value = params.get("input", [])
    instructions = params.get("instructions")
    stream = bool(params.get("stream", False))

    openai_messages: list[dict[str, Any]] = []

    # ``instructions`` becomes a system message (Responses-native convention).
    if instructions:
        if isinstance(instructions, list):
            inst_text = "\n".join(
                (p.get("text", "") if isinstance(p, dict) else str(p))
                for p in instructions
            )
        else:
            inst_text = str(instructions)
        if inst_text.strip():
            openai_messages.append({"role": "system", "content": inst_text})

    # ``input`` can be a plain string or a list of items.
    if isinstance(input_value, str):
        openai_messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        for item in input_value:
            openai_messages.extend(_input_item_to_openai_messages(item))

    # Ensure every assistant tool_call has a matching tool result downstream.
    # Same reason as the Anthropic path: vLLM/SGLang reject assistant messages
    # with tool_calls when the corresponding tool messages are missing.
    _ensure_tool_results_for_tool_calls(openai_messages)

    chat_params: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "stream": stream,
    }

    if stream:
        chat_params["stream_options"] = {"include_usage": True}

    max_tokens = params.get("max_output_tokens")
    if isinstance(max_tokens, int):
        chat_params["max_tokens"] = max_tokens

    temperature = params.get("temperature")
    if isinstance(temperature, (int, float)) and not _is_bool(temperature):
        chat_params["temperature"] = temperature

    top_p = params.get("top_p")
    if isinstance(top_p, (int, float)) and not _is_bool(top_p):
        chat_params["top_p"] = top_p

    stop = params.get("stop")
    if stop:
        chat_params["stop"] = stop

    tools, web_search_configs = _convert_responses_tools(params.get("tools"))
    if tools:
        chat_params["tools"] = tools
        tc = _convert_tool_choice(params.get("tool_choice"))
        if tc is not None:
            chat_params["tool_choice"] = tc

    # Stash web search config for the router to pick up. Router adds the
    # web_search function tool to ``tools`` and runs the server-side loop.
    if web_search_configs:
        chat_params["_web_search_configs"] = web_search_configs

    # ``reasoning`` carries the effort tier. We forward it as
    # ``chat_template_kwargs.reasoning_effort`` so vLLM/SGLang chat templates
    # pick it up — same convention as the Anthropic path.
    reasoning = params.get("reasoning")
    if isinstance(reasoning, dict):
        effort = _coerce_effort_to_reasoning_effort(reasoning.get("effort"))
        if effort:
            chat_params["chat_template_kwargs"] = {"reasoning_effort": effort}

    return chat_params


def _now() -> float:
    return float(time.time())


def _gen_response_id() -> str:
    import secrets
    import string

    chars = string.ascii_lowercase + string.digits
    return "resp_" + "".join(secrets.choice(chars) for _ in range(24))


def _gen_message_id() -> str:
    import secrets
    import string

    chars = string.ascii_lowercase + string.digits
    return "msg_" + "".join(secrets.choice(chars) for _ in range(24))


def _gen_tool_call_id() -> str:
    import secrets
    import string

    chars = string.ascii_lowercase + string.digits
    return "fc_" + "".join(secrets.choice(chars) for _ in range(24))


def _gen_reasoning_id() -> str:
    import secrets
    import string

    chars = string.ascii_lowercase + string.digits
    return "rs_" + "".join(secrets.choice(chars) for _ in range(24))


def _finish_to_responses_status(finish_reason: str | None) -> str:
    if finish_reason == "length":
        return "incomplete"
    return "completed"


def convert_chat_completion_to_responses(
    completion: ChatCompletion,
    *,
    model: str,
    instructions: str | None = None,
    created_at: float | None = None,
    response_id: str | None = None,
) -> dict[str, Any]:
    """Convert a Chat Completions response to a Responses ``Response`` dict."""
    choice = completion.choices[0] if completion.choices else None
    message = getattr(choice, "message", None) if choice else None

    output_items: list[dict[str, Any]] = []

    # Reasoning content (vLLM ``reasoning`` / SGLang ``reasoning_content``).
    reasoning_text = ""
    if message is not None:
        reasoning_text = (
            getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None)
            or ""
        )
    if reasoning_text:
        output_items.append(
            {
                "id": _gen_reasoning_id(),
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
                "content": [
                    {"type": "reasoning_text", "text": reasoning_text}
                ],
                "status": "completed",
            }
        )

    # Assistant message: text + tool calls.
    text_content = ""
    if message is not None and message.content:
        if isinstance(message.content, str):
            text_content = message.content
        else:
            parts: list[str] = []
            for part in message.content:
                if getattr(part, "type", None) == "text":
                    parts.append(getattr(part, "text", ""))
            text_content = "".join(parts)

    tool_calls = getattr(message, "tool_calls", None) if message else None
    finish_reason = getattr(choice, "finish_reason", None) if choice else None

    if text_content or not tool_calls:
        output_items.append(
            {
                "id": _gen_message_id(),
                "type": "message",
                "role": "assistant",
                "status": _finish_to_responses_status(finish_reason),
                "content": [
                    {
                        "type": "output_text",
                        "text": text_content or "",
                        "annotations": [],
                    }
                ],
            }
        )

    if tool_calls:
        for tc in tool_calls:
            if not tc.function:
                continue
            output_items.append(
                {
                    "id": _gen_tool_call_id(),
                    "type": "function_call",
                    "call_id": tc.id or _gen_tool_call_id(),
                    "name": tc.function.name or "",
                    "arguments": tc.function.arguments or "",
                    "status": "completed",
                }
            )

    # Usage
    usage_dict: dict[str, Any] | None = None
    if completion.usage:
        prompt = completion.usage.prompt_tokens or 0
        completion_tokens = completion.usage.completion_tokens or 0
        usage_dict = {
            "input_tokens": prompt,
            "output_tokens": completion_tokens,
            "total_tokens": prompt + completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

    created = created_at if created_at is not None else _now()
    rid = response_id or completion.id or _gen_response_id()

    response: dict[str, Any] = {
        "id": rid,
        "object": "response",
        "created_at": created,
        "model": model,
        "status": "completed",
        "output": output_items,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
    }
    if instructions is not None:
        response["instructions"] = instructions
    if usage_dict is not None:
        response["usage"] = usage_dict
    return response


# ── Streaming helpers ────────────────────────────────────────────────────────
#
# Responses streaming uses a different event sequence than Chat Completions.
# We consume the upstream chat/completions SSE stream and emit Responses-style
# events:
#
#   response.created
#   response.in_progress
#   response.output_item.added      (reasoning / message / function_call)
#   response.output_text.delta      (or response.reasoning_text.delta)
#   response.output_text.done
#   response.output_item.done
#   response.function_call_arguments.delta
#   response.function_call_arguments.done
#   response.completed


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_responses_from_chat_completion(
    upstream_lines: AsyncIterable[str],
    *,
    model: str,
    instructions: str | None = None,
    response_id: str | None = None,
):
    """Convert a Chat Completions SSE stream to Responses SSE events.

    ``upstream_lines`` is an async iterable of raw SSE lines (``data: ...``)
    from the upstream chat/completions stream. Yields Responses-format SSE
    strings.
    """
    rid = response_id or _gen_response_id()
    created_at = _now()
    msg_id = _gen_message_id()
    reasoning_id: str | None = None
    text_started = False
    reasoning_started = False
    # Per-index tool call state from chat completions deltas.
    tool_calls_state: dict[int, dict[str, Any]] = {}
    output_index = 0
    finish_reason: str | None = None
    usage_dict: dict[str, Any] | None = None
    completed = False

    # response.created
    created_payload = {
        "type": "response.created",
        "response": _build_response_shell(
            rid, model, created_at, instructions, status="in_progress"
        ),
    }
    yield _sse("response.created", created_payload)
    yield _sse(
        "response.in_progress",
        {"type": "response.in_progress", "response": created_payload["response"]},
    )

    async for line in upstream_lines:
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        if chunk.get("usage"):
            u = chunk["usage"]
            prompt = u.get("prompt_tokens", 0) or 0
            comp = u.get("completion_tokens", 0) or 0
            usage_dict = {
                "input_tokens": prompt,
                "output_tokens": comp,
                "total_tokens": prompt + comp,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {}) or {}
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        # Reasoning content
        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        if reasoning:
            if not reasoning_started:
                reasoning_id = _gen_reasoning_id()
                yield _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "id": reasoning_id,
                            "type": "reasoning",
                            "summary": [],
                            "content": [],
                            "status": "in_progress",
                        },
                    },
                )
                output_index += 1
                reasoning_started = True
            yield _sse(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": reasoning_id,
                    "output_index": output_index - 1,
                    "delta": reasoning,
                },
            )

        # Text content
        content_text = delta.get("content")
        if isinstance(content_text, str) and content_text:
            if not text_started:
                yield _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    },
                )
                output_index += 1
                text_started = True
            yield _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": output_index - 1,
                    "delta": content_text,
                },
            )

        # Tool calls
        tcs = delta.get("tool_calls")
        if tcs:
            for tc in tcs:
                idx = tc.get("index", 0)
                state = tool_calls_state.setdefault(
                    idx,
                    {
                        "id": tc.get("id") or _gen_tool_call_id(),
                        "name": "",
                        "arguments": "",
                        "output_index": None,
                        "emitted_added": False,
                    },
                )
                if tc.get("id"):
                    state["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    state["name"] = fn["name"]
                if not state["emitted_added"]:
                    state["output_index"] = output_index
                    output_index += 1
                    state["emitted_added"] = True
                    yield _sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": state["output_index"],
                            "item": {
                                "id": state["id"],
                                "type": "function_call",
                                "call_id": state["id"],
                                "name": state["name"],
                                "arguments": "",
                                "status": "in_progress",
                            },
                        },
                    )
                if fn.get("arguments"):
                    state["arguments"] += fn["arguments"]
                    yield _sse(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": state["id"],
                            "output_index": state["output_index"],
                            "delta": fn["arguments"],
                        },
                    )

    # Close out open items.
    if reasoning_started and reasoning_id:
        yield _sse(
            "response.reasoning_text.done",
            {
                "type": "response.reasoning_text.done",
                "item_id": reasoning_id,
                "output_index": 0,
                "text": "",
            },
        )
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": reasoning_id,
                    "type": "reasoning",
                    "summary": [],
                    "content": [],
                    "status": "completed",
                },
            },
        )

    if text_started:
        yield _sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": 0 if not reasoning_started else 1,
                "text": "",
            },
        )
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0 if not reasoning_started else 1,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": _finish_to_responses_status(finish_reason),
                    "content": [
                        {"type": "output_text", "text": "", "annotations": []}
                    ],
                },
            },
        )

    for idx in sorted(tool_calls_state.keys()):
        state = tool_calls_state[idx]
        if not state["emitted_added"]:
            continue
        yield _sse(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": state["id"],
                "output_index": state["output_index"],
                "arguments": state["arguments"],
            },
        )
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": state["output_index"],
                "item": {
                    "id": state["id"],
                    "type": "function_call",
                    "call_id": state["id"],
                    "name": state["name"],
                    "arguments": state["arguments"],
                    "status": "completed",
                },
            },
        )

    if not completed:
        status = _finish_to_responses_status(finish_reason)
        final_response = _build_response_shell(
            rid, model, created_at, instructions, status=status, usage=usage_dict
        )
        yield _sse(
            "response.completed",
            {"type": "response.completed", "response": final_response},
        )


def _build_response_shell(
    rid: str,
    model: str,
    created_at: float,
    instructions: str | None,
    *,
    status: str = "completed",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "id": rid,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": status,
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
    }
    if instructions is not None:
        resp["instructions"] = instructions
    if usage is not None:
        resp["usage"] = usage
    return resp
