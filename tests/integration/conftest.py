# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration and fixtures for integration tests.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport

from local_openai2anthropic.config import Settings
from local_openai2anthropic.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with mock API keys."""
    return Settings(
        openai_api_key="test-openai-api-key",
        openai_base_url="https://api.openai.com/v1",
        request_timeout=30.0,
        api_key=None,  # No server API key for basic tests
        tavily_api_key="test-tavily-api-key",
        tavily_timeout=30.0,
        tavily_max_results=5,
        websearch_max_uses=5,
    )


@pytest.fixture
def test_settings_with_auth() -> Settings:
    """Create test settings with server API key authentication."""
    return Settings(
        openai_api_key="test-openai-api-key",
        openai_base_url="https://api.openai.com/v1",
        request_timeout=30.0,
        api_key="test-server-api-key",
        tavily_api_key="test-tavily-api-key",
        tavily_timeout=30.0,
        tavily_max_results=5,
        websearch_max_uses=5,
    )


@pytest_asyncio.fixture
async def app(test_settings: Settings) -> FastAPI:
    """Create FastAPI application with test settings."""
    return create_app(test_settings)


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        yield client


@pytest_asyncio.fixture
async def auth_app(test_settings_with_auth: Settings) -> FastAPI:
    """Create FastAPI application with authentication enabled."""
    return create_app(test_settings_with_auth)


@pytest_asyncio.fixture
async def auth_client(auth_app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create async HTTP client for testing with auth."""
    transport = ASGITransport(app=auth_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def mock_openai_chat_completion() -> dict[str, Any]:
    """Mock OpenAI chat completion response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1704067200,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_openai_chat_completion_with_tool_calls() -> dict[str, Any]:
    """Mock OpenAI chat completion response with tool calls."""
    return {
        "id": "chatcmpl-test456",
        "object": "chat.completion",
        "created": 1704067200,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_test123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Tokyo"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        },
    }


@pytest.fixture
def mock_openai_streaming_chunks() -> list[bytes]:
    """Mock OpenAI streaming response chunks."""
    chunks = [
        b'data: {"id":"chatcmpl-stream123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-stream123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-stream123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-stream123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" How"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-stream123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" can"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-stream123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" I"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-stream123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" help"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-stream123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"?"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-stream123","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    return chunks


@pytest.fixture
def mock_openai_models_response() -> dict[str, Any]:
    """Mock OpenAI models list response."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1687882411,
                "owned_by": "openai",
            },
            {
                "id": "gpt-4-turbo",
                "object": "model",
                "created": 1687882411,
                "owned_by": "openai",
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677649963,
                "owned_by": "openai",
            },
        ],
    }


@pytest.fixture
def mock_tavily_search_response() -> dict[str, Any]:
    """Mock Tavily search response."""
    return {
        "query": "test query",
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "content": "This is test content 1",
                "score": 0.95,
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/2",
                "content": "This is test content 2",
                "score": 0.85,
            },
        ],
    }


@pytest.fixture
def mock_httpx_client(
    mock_openai_chat_completion: dict[str, Any],
) -> MagicMock:
    """Create a mock httpx.AsyncClient for OpenAI API calls."""
    mock_client = MagicMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_openai_chat_completion
    mock_response.text = json.dumps(mock_openai_chat_completion)
    mock_response.reason_phrase = "OK"

    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.get = AsyncMock(return_value=mock_response)

    return mock_client


@pytest.fixture
def mock_httpx_client_with_streaming(
    mock_openai_streaming_chunks: list[bytes],
) -> MagicMock:
    """Create a mock httpx.AsyncClient for streaming OpenAI API calls."""
    mock_client = MagicMock()

    # Create async iterator for streaming response
    async def mock_aiter_raw():
        for chunk in mock_openai_streaming_chunks:
            yield chunk

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.aiter_raw = mock_aiter_raw
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.stream = MagicMock(return_value=mock_response)

    return mock_client


@pytest.fixture(autouse=True)
def clear_tavily_client_singleton():
    """Clear Tavily client singleton before each test."""
    from local_openai2anthropic.server_tools.web_search import WebSearchServerTool

    WebSearchServerTool._client = None
    yield
    WebSearchServerTool._client = None
