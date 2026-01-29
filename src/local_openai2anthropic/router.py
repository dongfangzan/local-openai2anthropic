# SPDX-License-Identifier: Apache-2.0
"""
FastAPI router for Anthropic-compatible Messages API.
"""

import json
import logging
from http import HTTPStatus
from typing import AsyncGenerator

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

logger = logging.getLogger(__name__)
router = APIRouter()


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
        async with client.stream("POST", url, headers=headers, json=json_data) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                try:
                    error_json = json.loads(error_body.decode())
                    error_msg = error_json.get("error", {}).get("message", error_body.decode())
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
                except json.JSONDecodeError:
                    continue
                
                # First chunk: message_start
                if first_chunk:
                    message_id = chunk.get("id", "")
                    usage = chunk.get("usage", {})
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
                    yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"
                    first_chunk = False
                    continue
                
                # Handle usage-only chunks
                if not chunk.get("choices"):
                    usage = chunk.get("usage", {})
                    if usage:
                        if content_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                            content_block_started = False
                        
                        stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason_map.get(finish_reason or 'stop', 'end_turn')}, 'usage': {'input_tokens': usage.get('prompt_tokens', 0), 'output_tokens': usage.get('completion_tokens', 0), 'cache_creation_input_tokens': None, 'cache_read_input_tokens': None}})}\n\n"
                    continue
                
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                
                # Track finish reason
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                    continue
                
                # Handle content
                if delta.get("content"):
                    if not content_block_started:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        content_block_started = True
                    
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"
                
                # Handle tool calls
                if delta.get("tool_calls"):
                    tool_call = delta["tool_calls"][0]
                    
                    if tool_call.get("id"):
                        if content_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                            content_block_started = False
                            content_block_index += 1
                        
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'tool_use', 'id': tool_call['id'], 'name': tool_call.get('function', {}).get('name', ''), 'input': {}}})}\n\n"
                        content_block_started = True
                        
                    elif tool_call.get("function", {}).get("arguments"):
                        args = tool_call["function"]["arguments"]
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n\n"
            
            # Close final content block
            if content_block_started:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
            
            # Message stop
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "data: [DONE]\n\n"
            
    except Exception as e:
        error_event = AnthropicErrorResponse(
            error=AnthropicError(type="internal_error", message=str(e))
        )
        yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


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
    settings: Settings = Depends(get_settings),
) -> JSONResponse | StreamingResponse:
    """
    Create a message using Anthropic-compatible API.
    """
    # Read and parse the request body
    try:
        body_bytes = await request.body()
        body_json = json.loads(body_bytes.decode('utf-8'))
        anthropic_params = body_json
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {e}")
        error_response = AnthropicErrorResponse(
            error=AnthropicError(type="invalid_request_error", message=f"Invalid JSON: {e}")
        )
        return JSONResponse(status_code=400, content=error_response.model_dump())
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        error_response = AnthropicErrorResponse(
            error=AnthropicError(type="invalid_request_error", message=str(e))
        )
        return JSONResponse(status_code=400, content=error_response.model_dump())
    
    # Convert Anthropic params to OpenAI params
    openai_params = convert_anthropic_to_openai(anthropic_params)
    
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
    
    if stream:
        # Create client that won't be closed until streaming is done
        client = httpx.AsyncClient(timeout=settings.request_timeout)
        
        return StreamingResponse(
            _stream_response(client, url, headers, openai_params, model),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response
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
                from openai.types.chat import ChatCompletion
                completion = ChatCompletion.model_validate(openai_completion)
                anthropic_message = convert_openai_to_anthropic(completion, model)
                
                return JSONResponse(content=anthropic_message.model_dump())
                
            except httpx.TimeoutException:
                error_response = AnthropicErrorResponse(
                    error=AnthropicError(type="timeout_error", message="Request timed out")
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
    settings: Settings = Depends(get_settings),
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


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
