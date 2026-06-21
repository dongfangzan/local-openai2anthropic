# SPDX-License-Identifier: Apache-2.0
"""
FastAPI router for Anthropic-compatible Messages API.
"""

import json
import logging
from http import HTTPStatus
from typing import Any, AsyncGenerator, cast

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

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
from local_openai2anthropic.streaming import _convert_result_to_stream, _stream_response
from local_openai2anthropic.tools import (
    ServerToolHandler,
    _add_tool_results_to_messages,
    _handle_with_server_tools,
)
from local_openai2anthropic.utils import (
    _chunk_text,
    _count_tokens,
    _estimate_input_tokens,
    _generate_server_tool_id,
    _normalize_usage,
)

logger = logging.getLogger(__name__)
api_logger = logging.getLogger("api")
router = APIRouter()
openai_router = APIRouter(prefix="/v1")


# Backward compatibility: re-export functions used by tests
__all__ = [
    "router",
    "openai_router",
    "get_request_settings",
    "create_message",
    "list_models",
    "count_tokens",
    "health_check",
    # Backward compatibility exports
    "_stream_response",
    "_convert_result_to_stream",
    "ServerToolHandler",
    "_handle_with_server_tools",
    "_add_tool_results_to_messages",
    "_generate_server_tool_id",
    "_normalize_usage",
    "_count_tokens",
    "_chunk_text",
    "_estimate_input_tokens",
]


def get_request_settings(request: Request) -> Settings:
    """Resolve Settings from the running app when available.

    This allows tests (and embedders) to pass an explicit Settings instance via
    `create_app(settings=...)` without requiring environment variables.
    """
    settings = getattr(getattr(request, "app", None), "state", None)
    if settings is not None and hasattr(settings, "settings"):
        return settings.settings  # type: ignore[return-value]
    return get_settings()


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
        # Log to dedicated API log file (clean format, no timestamp prefix)
        api_logger.debug(f"[Anthropic Request] {json.dumps(body_json, ensure_ascii=False)}")
        anthropic_params = body_json
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {e}")
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error", message=f"Invalid JSON: {e}"
            )
        )
        return JSONResponse(status_code=400, content=error_response.model_dump())
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
        return JSONResponse(status_code=400, content=error_response.model_dump())

    model_value = anthropic_params.get("model")
    if not isinstance(model_value, str) or not model_value.strip():
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error", message="Model must be a non-empty string"
            )
        )
        return JSONResponse(status_code=400, content=error_response.model_dump())

    # Resolve model name using mapping rules
    original_model = anthropic_params.get("model", "")
    resolved_model = settings.resolve_model(original_model)
    if resolved_model != original_model:
        logger.debug("Model name mapped: %s -> %s", original_model, resolved_model)
        anthropic_params["model"] = resolved_model

    messages_value = anthropic_params.get("messages")
    if not isinstance(messages_value, list) or len(messages_value) == 0:
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error",
                message="Messages must be a non-empty list",
            )
        )
        return JSONResponse(status_code=400, content=error_response.model_dump())

    max_tokens_value = anthropic_params.get("max_tokens")
    if not isinstance(max_tokens_value, int):
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error", message="max_tokens is required"
            )
        )
        return JSONResponse(status_code=400, content=error_response.model_dump())

    # Check for server tools
    tools = anthropic_params.get("tools", [])
    enabled_server_tools = ServerToolRegistry.get_enabled_tools(settings)
    server_tool_configs = ServerToolRegistry.extract_server_tools(
        [t if isinstance(t, dict) else t.model_dump() for t in tools]
    )
    has_server_tools = len(server_tool_configs) > 0

    # Convert Anthropic params to OpenAI params
    openai_params_obj = convert_anthropic_to_openai(
        cast(MessageCreateParams, anthropic_params),
        enabled_server_tools=enabled_server_tools if has_server_tools else None,
    )
    openai_params: dict[str, Any] = dict(openai_params_obj)  # type: ignore

    # Log converted OpenAI request (remove internal fields)
    log_params = {k: v for k, v in openai_params.items() if not k.startswith("_")}
    logger.debug(
        f"[OpenAI Request] {json.dumps(log_params, ensure_ascii=False, indent=2)}"
    )
    # Log to dedicated API log file (clean format)
    api_logger.debug(f"[OpenAI Request] {json.dumps(log_params, ensure_ascii=False)}")

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

                if not response.content:
                    error_response = AnthropicErrorResponse(
                        error=AnthropicError(
                            type="api_error",
                            message="Upstream API returned empty response",
                        )
                    )
                    return JSONResponse(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        content=error_response.model_dump(),
                    )

                try:
                    openai_completion = response.json()
                except json.JSONDecodeError as e:
                    error_response = AnthropicErrorResponse(
                        error=AnthropicError(
                            type="api_error",
                            message=f"Failed to parse upstream response: {str(e)}",
                        )
                    )
                    return JSONResponse(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        content=error_response.model_dump(),
                    )

                logger.debug(
                    f"[OpenAI Response] {json.dumps(openai_completion, ensure_ascii=False, indent=2)}"
                )

                from openai.types.chat import ChatCompletion

                completion = ChatCompletion.model_validate(openai_completion)
                anthropic_message = convert_openai_to_anthropic(completion, model)

                anthropic_response = anthropic_message.model_dump()
                anthropic_response["usage"] = _normalize_usage(
                    anthropic_response.get("usage")
                )
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
        return JSONResponse(status_code=400, content=error_response.model_dump())
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
        return JSONResponse(status_code=400, content=error_response.model_dump())

    messages = body_json.get("messages", [])
    if not isinstance(messages, list):
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="invalid_request_error", message="messages must be a list"
            )
        )
        return JSONResponse(status_code=400, content=error_response.model_dump())

    model = body_json.get("model", "")
    system = body_json.get("system")
    tools = body_json.get("tools", [])

    try:
        # Use tiktoken for token counting
        import tiktoken  # type: ignore[import-not-found]

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


@router.post("/api/event_logging/batch")
async def event_logging_batch(request: Request) -> Response:
    """Event logging endpoint placeholder."""
    try:
        body_bytes = await request.body()
        body_json = json.loads(body_bytes.decode("utf-8"))
        logger.info(f"[Event Logging] {json.dumps(body_json, ensure_ascii=False, indent=2)}")
    except Exception as e:
        logger.info(f"[Event Logging] Failed to parse body: {e}")

    return Response(status_code=204)


# ── OpenAI-native passthrough routes ──────────────────────────────────────────
# These routes accept OpenAI-format requests and proxy them directly to the
# upstream API without any conversion. Useful when clients already speak the
# OpenAI protocol and want to avoid the Anthropic↔OpenAI conversion overhead.
#
# Endpoints:
#   POST /v1/chat/completions   — non-streaming
#   POST /v1/chat/completions   — streaming (when stream=True in body)


@openai_router.post("/chat/completions", response_model=None)
async def openai_chat_completions(
    request: Request,
    settings: Settings = Depends(get_request_settings),
):
    """Proxy an OpenAI-format /v1/chat/completions request directly upstream."""
    try:
        body_bytes = await request.body()
        body_json = json.loads(body_bytes.decode("utf-8"))
        logger.debug(
            f"[OpenAI Passthrough Request] {json.dumps(body_json, ensure_ascii=False, indent=2)}"
        )
        api_logger.debug(
            f"[OpenAI Passthrough Request] {json.dumps(body_json, ensure_ascii=False)}"
        )
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in passthrough request: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}},
        )
    except Exception as e:
        logger.error(f"Failed to parse passthrough request body: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "invalid_request_error"}},
        )

    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {**settings.openai_auth_headers, "Content-Type": "application/json"}

    stream = body_json.get("stream", False)

    if stream:
        client = httpx.AsyncClient(timeout=settings.request_timeout)
        return StreamingResponse(
            _openai_stream_proxy(client, url, headers, body_json),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        try:
            response = await client.post(url, headers=headers, json=body_json)

            if not response.content:
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "Upstream API returned empty response", "type": "api_error"}},
                )

            try:
                response_json = response.json()
            except json.JSONDecodeError as e:
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": f"Failed to parse upstream response: {e}", "type": "api_error"}},
                )

            logger.debug(
                f"[OpenAI Passthrough Response] {json.dumps(response_json, ensure_ascii=False, indent=2)}"
            )
            api_logger.debug(
                f"[OpenAI Passthrough Response] {json.dumps(response_json, ensure_ascii=False)}"
            )

            return JSONResponse(status_code=response.status_code, content=response_json)

        except httpx.TimeoutException:
            return JSONResponse(
                status_code=504,
                content={"error": {"message": "Request timed out", "type": "timeout_error"}},
            )
        except httpx.RequestError as e:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": str(e), "type": "connection_error"}},
            )


async def _openai_stream_proxy(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    body_json: dict,
) -> AsyncGenerator[str, None]:
    """Stream-proxy an OpenAI request — pass chunks through unchanged."""
    try:
        async with client.stream("POST", url, headers=headers, json=body_json) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="replace").strip()
                yield f"data: {json.dumps({'error': {'message': error_text or f'Upstream error ({response.status_code})', 'type': 'api_error'}})}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield f"{line}\n\n"
                elif line.strip() == "":
                    yield "\n"
                else:
                    yield f"{line}\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'internal_error'}})}\n\n"
        yield "data: [DONE]\n\n"


# ── OpenAI Responses API bridge ──────────────────────────────────────────────
#
# POST /v1/responses accepts OpenAI Responses-format requests and translates
# them to /v1/chat/completions for upstream backends that only implement the
# chat-completions surface (vLLM, SGLang, …). The upstream response is then
# translated back to the Responses format.


@openai_router.post("/responses", response_model=None)
async def openai_responses(
    request: Request,
    settings: Settings = Depends(get_request_settings),
):
    """Accept a Responses-format request and proxy it as chat/completions."""
    from local_openai2anthropic.responses_converter import (
        convert_chat_completion_to_responses,
        convert_responses_to_chat_completion,
        stream_responses_from_chat_completion,
    )

    try:
        body_bytes = await request.body()
        body_json = json.loads(body_bytes.decode("utf-8"))
        logger.debug(
            f"[Responses Request] {json.dumps(body_json, ensure_ascii=False, indent=2)}"
        )
        api_logger.debug(
            f"[Responses Request] {json.dumps(body_json, ensure_ascii=False)}"
        )
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in responses request: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}},
        )
    except Exception as e:
        logger.error(f"Failed to parse responses request body: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "invalid_request_error"}},
        )

    if not isinstance(body_json, dict):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Request body must be a JSON object", "type": "invalid_request_error"}},
        )

    model_value = body_json.get("model")
    if not isinstance(model_value, str) or not model_value.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Model must be a non-empty string", "type": "invalid_request_error"}},
        )

    # Resolve model via configured mapping rules
    original_model = body_json.get("model", "")
    resolved_model = settings.resolve_model(original_model)
    if resolved_model != original_model:
        logger.debug("Model name mapped: %s -> %s", original_model, resolved_model)
        body_json["model"] = resolved_model

    instructions = body_json.get("instructions")
    if isinstance(instructions, list):
        # Reduce to a single string for forwarding/shell
        instructions = "\n".join(
            (p.get("text", "") if isinstance(p, dict) else str(p))
            for p in instructions
        ) or None

    try:
        chat_params = convert_responses_to_chat_completion(body_json)
    except Exception as e:
        logger.exception("Failed to convert Responses request to chat/completions")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Conversion failed: {e}", "type": "invalid_request_error"}},
        )

    log_params = {k: v for k, v in chat_params.items() if not k.startswith("_")}
    logger.debug(
        f"[Responses->ChatCompletions Request] {json.dumps(log_params, ensure_ascii=False, indent=2)}"
    )
    api_logger.debug(
        f"[Responses->ChatCompletions Request] {json.dumps(log_params, ensure_ascii=False)}"
    )

    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {**settings.openai_auth_headers, "Content-Type": "application/json"}

    stream = bool(body_json.get("stream", False))
    model = chat_params.get("model", resolved_model)

    # Server-side web search loop: when the request carries a web_search tool
    # AND a search provider is configured, run the tool loop locally instead
    # of forwarding the (unsupported) tool to the chat/completions backend.
    web_search_configs = chat_params.pop("_web_search_configs", None)
    if web_search_configs:
        from local_openai2anthropic.responses_web_search import (
            handle_responses_with_web_search,
        )
        from local_openai2anthropic.server_tools.web_search import (
            WebSearchServerTool,
        )

        if WebSearchServerTool.is_enabled(settings):
            result = await handle_responses_with_web_search(
                chat_params,
                web_search_configs,
                url,
                headers,
                settings,
                model,
                instructions if isinstance(instructions, str) else None,
                stream=stream,
            )
            if stream:
                return StreamingResponse(result, media_type="text/event-stream")
            return result
        # Search not configured — fall through and treat web_search as a
        # regular function tool would be treated (i.e. drop it, since the
        # upstream backend doesn't implement it either). The conversion has
        # already excluded web_search from chat_params["tools"].
        logger.warning(
            "Responses request carried a web_search tool but no search provider "
            "is configured; proceeding without web search capability."
        )

    if stream:
        client = httpx.AsyncClient(timeout=settings.request_timeout)
        return StreamingResponse(
            _stream_responses(client, url, headers, chat_params, model, instructions),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        try:
            response = await client.post(url, headers=headers, json=chat_params)

            if response.status_code != 200:
                raw_text = (response.text or "").strip()
                if not raw_text:
                    raw_text = response.reason_phrase or f"Upstream API error ({response.status_code})"
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": {"message": raw_text, "type": "api_error"}},
                )

            if not response.content:
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "Upstream API returned empty response", "type": "api_error"}},
                )

            try:
                openai_completion = response.json()
            except json.JSONDecodeError as e:
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": f"Failed to parse upstream response: {e}", "type": "api_error"}},
                )

            logger.debug(
                f"[Responses->ChatCompletions Response] {json.dumps(openai_completion, ensure_ascii=False, indent=2)}"
            )

            from openai.types.chat import ChatCompletion

            completion = ChatCompletion.model_validate(openai_completion)
            responses_payload = convert_chat_completion_to_responses(
                completion,
                model=model,
                instructions=instructions if isinstance(instructions, str) else None,
            )

            logger.debug(
                f"[Responses Response] {json.dumps(responses_payload, ensure_ascii=False, indent=2)}"
            )
            api_logger.debug(
                f"[Responses Response] {json.dumps(responses_payload, ensure_ascii=False)}"
            )

            return JSONResponse(content=responses_payload)

        except httpx.TimeoutException:
            return JSONResponse(
                status_code=504,
                content={"error": {"message": "Request timed out", "type": "timeout_error"}},
            )
        except httpx.RequestError as e:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": str(e), "type": "connection_error"}},
            )


async def _stream_responses(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    chat_params: dict,
    model: str,
    instructions: str | None,
) -> AsyncGenerator[str, None]:
    """Stream chat/completions upstream and emit Responses-format SSE events."""
    from local_openai2anthropic.responses_converter import (
        stream_responses_from_chat_completion,
    )

    try:
        async with client.stream(
            "POST", url, headers=headers, json=chat_params
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="replace").strip()
                err_payload = {
                    "type": "response.failed",
                    "response": {
                        "id": "",
                        "object": "response",
                        "created_at": 0,
                        "model": model,
                        "status": "failed",
                        "output": [],
                        "error": {"code": "upstream_error", "message": error_text or f"Upstream error ({response.status_code})"},
                    },
                }
                yield f"event: response.failed\ndata: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
                return

            async def _line_iter():
                async for line in response.aiter_lines():
                    yield line

            async for sse_chunk in stream_responses_from_chat_completion(
                _line_iter(),
                model=model,
                instructions=instructions if isinstance(instructions, str) else None,
            ):
                yield sse_chunk

    except Exception as e:
        import traceback

        logger.error(f"Responses stream error: {e}\n{traceback.format_exc()}")
        err_payload = {
            "type": "response.failed",
            "response": {
                "id": "",
                "object": "response",
                "created_at": 0,
                "model": model,
                "status": "failed",
                "output": [],
                "error": {"code": "internal_error", "message": str(e)},
            },
        }
        yield f"event: response.failed\ndata: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
