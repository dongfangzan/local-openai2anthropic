# SPDX-License-Identifier: Apache-2.0
"""
Server-side web search loop for the /v1/responses endpoint.

When a Responses request carries a ``web_search`` / ``web_search_preview``
tool, the proxy cannot forward it to a bare chat/completions backend
(vLLM/SGLang don't implement OpenAI's web search tool). Instead we:

  1. Register ``web_search`` as an OpenAI function tool alongside any
     client-supplied function tools.
  2. Forward the request to upstream ``/v1/chat/completions``.
  3. When the model calls ``web_search``, execute Tavily/TongXiao locally,
     append the result as a ``tool`` message, and re-prompt.
  4. Repeat until the model stops calling ``web_search`` (or ``max_uses``
     is hit).
  5. Convert the final chat completion to a Responses ``Response``, prepending
     one ``web_search_call`` output item per executed search.

Both non-streaming and streaming variants live here. Streaming is more
involved: while the model is still deciding whether to search we cannot emit
text deltas (they may be revoked by a subsequent tool call), so we buffer
until we know the turn's shape. To keep complexity bounded we run the whole
loop non-streaming upstream and then synthesize the Responses SSE sequence
from the final result — same trick the Anthropic path uses for server tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx
from fastapi.responses import JSONResponse

from local_openai2anthropic.config import Settings
from local_openai2anthropic.responses_converter import (
    WEB_SEARCH_FUNCTION_NAME,
    build_web_search_output_items,
    convert_chat_completion_to_responses,
    web_search_function_tool,
)
from local_openai2anthropic.server_tools import ServerToolRegistry, ToolResult
from local_openai2anthropic.server_tools.web_search import WebSearchServerTool
from local_openai2anthropic.tools.handler import ServerToolHandler
from local_openai2anthropic.utils.tokens import _generate_server_tool_id

logger = logging.getLogger(__name__)


def _is_web_search_enabled(settings: Settings) -> bool:
    return WebSearchServerTool.is_enabled(settings)


def _prepare_chat_params_for_loop(
    chat_params: dict[str, Any],
    web_search_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return a copy of ``chat_params`` with the web_search function tool added."""
    params = dict(chat_params)
    tools = list(params.get("tools") or [])
    tools.append(web_search_function_tool())
    params["tools"] = tools
    # Server-side loop needs non-streaming upstream calls; we synthesize SSE
    # afterward if the client asked for streaming.
    params["stream"] = False
    params.pop("stream_options", None)
    params["_server_tools_config"] = web_search_configs
    return params


def _resolve_max_uses(
    web_search_configs: dict[str, dict[str, Any]],
    settings: Settings,
) -> int:
    max_uses = settings.websearch_max_uses
    for cfg in web_search_configs.values():
        if cfg.get("max_uses"):
            max_uses = cfg["max_uses"]
            break
    return max_uses


async def _run_web_search_loop(
    params: dict[str, Any],
    url: str,
    headers: dict[str, str],
    settings: Settings,
    model: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], JSONResponse | None]:
    """Run the upstream ↔ web_search loop.

    Returns ``(final_completion_json, web_search_output_items, error_response)``.
    On success, ``error_response`` is ``None``. On upstream/network error,
    ``final_completion_json`` is ``None`` and ``error_response`` carries the
    failure as a Responses-friendly error payload.
    """
    configs = params.pop("_server_tools_config", {})
    handler = ServerToolHandler([WebSearchServerTool], configs, settings)
    accumulated_web_search_items: list[dict[str, Any]] = []
    max_uses = _resolve_max_uses(configs, settings)
    total_tool_calls = 0

    from openai.types.chat import ChatCompletion

    while True:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=params)
            except httpx.TimeoutException:
                return None, [], _error_json(504, "timeout_error", "Request timed out")
            except httpx.RequestError as e:
                return None, [], _error_json(502, "connection_error", str(e))

            if response.status_code != 200:
                raw = (response.text or "").strip() or response.reason_phrase or ""
                return None, [], _error_json(
                    response.status_code, "api_error", raw or f"Upstream error ({response.status_code})"
                )

            try:
                completion_data = response.json()
            except json.JSONDecodeError as e:
                return None, [], _error_json(502, "api_error", f"Failed to parse upstream response: {e}")

            completion = ChatCompletion.model_validate(completion_data)
            choice = completion.choices[0] if completion.choices else None
            tool_calls = getattr(getattr(choice, "message", None), "tool_calls", None) if choice else None

            server_calls: list[dict[str, Any]] = []
            other_calls: list[dict[str, Any]] = []
            if tool_calls:
                for tc in tool_calls:
                    func = getattr(tc, "function", None)
                    func_name = func.name if func else ""
                    if func_name == WEB_SEARCH_FUNCTION_NAME:
                        server_calls.append(
                            {
                                "id": _generate_server_tool_id(),
                                "openai_id": tc.id,
                                "function": {
                                    "name": func_name,
                                    "arguments": func.arguments if func else "{}",
                                },
                            }
                        )
                    else:
                        other_calls.append(tc)

            if not server_calls:
                return completion_data, accumulated_web_search_items, None

            # Execute each web_search call, respecting max_uses per-call.
            assistant_calls: list[dict[str, Any]] = []
            tool_result_msgs: list[dict[str, Any]] = []
            for call in server_calls:
                if total_tool_calls >= max_uses:
                    # Quota exhausted: emit a failed web_search_call and feed
                    # an error result back so the model can wrap up.
                    items = build_web_search_output_items(
                        call["id"],
                        _extract_query(call),
                        [],
                        error_code="max_uses_exceeded",
                    )
                    accumulated_web_search_items.extend(items)
                    assistant_calls.append(
                        {
                            "id": call.get("openai_id", call.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                    )
                    tool_result_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("openai_id", call.get("id", "")),
                            "content": json.dumps(
                                {
                                    "error": "max_uses_exceeded",
                                    "message": "Maximum tool uses exceeded.",
                                }
                            ),
                        }
                    )
                    continue

                total_tool_calls += 1
                content_blocks, tool_result_msg = await handler.execute_tool(call)
                query = _extract_query(call)
                ws_items = _content_blocks_to_web_search_items(
                    call["id"], query, content_blocks
                )
                accumulated_web_search_items.extend(ws_items)
                assistant_calls.append(
                    {
                        "id": call.get("openai_id", call.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"],
                        },
                    }
                )
                tool_result_msgs.append(tool_result_msg)

            params["messages"] = _add_tool_results_to_messages(
                params.get("messages", []),
                assistant_calls,
                handler,
                tool_results=tool_result_msgs,
            )


def _extract_query(call: dict[str, Any]) -> str:
    args = call.get("function", {}).get("arguments", "{}")
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        return ""
    if isinstance(parsed, dict):
        return str(parsed.get("query", "") or "")
    return ""


def _content_blocks_to_web_search_items(
    call_id: str,
    query: str,
    content_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate the handler's Anthropic-style content blocks into Responses items."""
    # Find the web_search_tool_result block (success) or error block.
    for block in content_blocks:
        btype = block.get("type", "")
        if btype == "web_search_tool_result":
            results = block.get("results")
            if isinstance(results, dict) and "error_code" in results:
                return build_web_search_output_items(
                    call_id, query, [], error_code=str(results.get("error_code") or "unavailable")
                )
            if isinstance(results, list):
                return build_web_search_output_items(call_id, query, results)
            # Fallback: content field
            content = block.get("content")
            if isinstance(content, list):
                return build_web_search_output_items(call_id, query, content)
            return build_web_search_output_items(call_id, query, [], error_code="unavailable")
    # No result block found — treat as failure.
    return build_web_search_output_items(call_id, query, [], error_code="unavailable")


def _add_tool_results_to_messages(
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    handler: ServerToolHandler,
    tool_results: list[dict[str, Any]] | None = None,
    is_error: bool = False,
) -> list[dict[str, Any]]:
    """Add assistant tool_call + tool result messages (mirrors tools/handler.py)."""
    messages = list(messages)
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls,
    }
    messages.append(assistant_msg)
    if is_error:
        for call in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(
                        {"error": "max_uses_exceeded", "message": "Maximum tool uses exceeded."}
                    ),
                }
            )
    elif tool_results:
        messages.extend(tool_results)
    return messages


def _error_json(status: int, err_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


async def handle_responses_with_web_search(
    chat_params: dict[str, Any],
    web_search_configs: dict[str, dict[str, Any]],
    url: str,
    headers: dict[str, str],
    settings: Settings,
    model: str,
    instructions: str | None,
    stream: bool,
) -> JSONResponse | AsyncGenerator[str, None]:
    """Entry point: run the web_search loop and return a Responses response.

    Returns either a ``JSONResponse`` (non-streaming) or an async generator
    of SSE strings (streaming). Caller is responsible for picking the right
    branch.
    """
    loop_params = _prepare_chat_params_for_loop(chat_params, web_search_configs)

    final_completion, web_search_items, err = await _run_web_search_loop(
        loop_params, url, headers, settings, model
    )
    if err is not None:
        return err

    from openai.types.chat import ChatCompletion

    completion = ChatCompletion.model_validate(final_completion)
    responses_payload = convert_chat_completion_to_responses(
        completion,
        model=model,
        instructions=instructions if isinstance(instructions, str) else None,
    )
    # Prepend web_search_call items so clients see searches before the answer.
    if web_search_items:
        responses_payload["output"] = web_search_items + responses_payload["output"]

    if stream:
        return _stream_responses_with_web_search(responses_payload, model)
    return JSONResponse(content=responses_payload)


async def _stream_responses_with_web_search(
    responses_payload: dict[str, Any],
    model: str,
) -> AsyncGenerator[str, None]:
    """Synthesize a Responses SSE stream from a fully-built Response payload."""
    import time

    from local_openai2anthropic.responses_converter import _sse

    rid = responses_payload.get("id", "")
    created_at = responses_payload.get("created_at", float(time.time()))
    status = responses_payload.get("status", "completed")

    # response.created / in_progress
    shell = {
        "id": rid,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": "in_progress",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
    }
    if "instructions" in responses_payload:
        shell["instructions"] = responses_payload["instructions"]
    yield _sse("response.created", {"type": "response.created", "response": shell})
    yield _sse("response.in_progress", {"type": "response.in_progress", "response": shell})

    output_index = 0
    for item in responses_payload.get("output", []):
        itype = item.get("type")
        if itype == "web_search_call":
            yield _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {**item, "status": "in_progress"},
                },
            )
            yield _sse(
                "response.web_search_call.completed",
                {
                    "type": "response.web_search_call.completed",
                    "output_index": output_index,
                    "item_id": item.get("id", ""),
                },
            )
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": item,
                },
            )
            output_index += 1
        elif itype == "reasoning":
            yield _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {**item, "status": "in_progress", "summary": [], "content": []},
                },
            )
            # Emit reasoning text deltas from content parts if present.
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("text"):
                    yield _sse(
                        "response.reasoning_text.delta",
                        {
                            "type": "response.reasoning_text.delta",
                            "item_id": item.get("id", ""),
                            "output_index": output_index,
                            "delta": part.get("text", ""),
                        },
                    )
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": item,
                },
            )
            output_index += 1
        elif itype == "message":
            msg_id = item.get("id", "")
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
            text = ""
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text += part.get("text", "")
            if text:
                yield _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": output_index,
                        "delta": text,
                    },
                )
                yield _sse(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": msg_id,
                        "output_index": output_index,
                        "text": text,
                    },
                )
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": item,
                },
            )
            output_index += 1
        elif itype == "function_call":
            yield _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {**item, "status": "in_progress", "arguments": ""},
                },
            )
            args = item.get("arguments", "") or ""
            if args:
                yield _sse(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": item.get("id", ""),
                        "output_index": output_index,
                        "delta": args,
                    },
                )
                yield _sse(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": item.get("id", ""),
                        "output_index": output_index,
                        "arguments": args,
                    },
                )
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": item,
                },
            )
            output_index += 1

    final = {**shell, "status": status, "output": responses_payload.get("output", [])}
    if "usage" in responses_payload:
        final["usage"] = responses_payload["usage"]
    yield _sse("response.completed", {"type": "response.completed", "response": final})
