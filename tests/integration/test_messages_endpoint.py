# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for /v1/messages endpoint - non-streaming requests.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from local_openai2anthropic.config import Settings


@pytest.mark.asyncio
class TestMessagesEndpointNonStreaming:
    """Test /v1/messages endpoint with non-streaming requests."""

    async def test_simple_chat_completion(
        self,
        client: httpx.AsyncClient,
        mock_openai_chat_completion: dict[str, Any],
    ) -> None:
        """Test basic chat completion request."""
        with patch("local_openai2anthropic.router.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_chat_completion
            mock_response.text = json.dumps(mock_openai_chat_completion)
            mock_response.reason_phrase = "OK"

            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await client.post(
                "/v1/messages",
                json={
                    "model": "gpt-4",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert len(data["content"]) > 0
            assert data["content"][0]["type"] == "text"
            assert "Hello" in data["content"][0]["text"]
            assert data["model"] == "gpt-4"
            assert "usage" in data
            assert data["usage"]["input_tokens"] == 10
            assert data["usage"]["output_tokens"] == 20

    async def test_chat_with_system_message(
        self,
        client: httpx.AsyncClient,
        mock_openai_chat_completion: dict[str, Any],
    ) -> None:
        """Test chat completion with system message."""
        with patch("local_openai2anthropic.router.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_chat_completion
            mock_response.text = json.dumps(mock_openai_chat_completion)
            mock_response.reason_phrase = "OK"

            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await client.post(
                "/v1/messages",
                json={
                    "model": "gpt-4",
                    "max_tokens": 1024,
                    "system": "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "message"

    async def test_chat_with_temperature_and_top_p(
        self,
        client: httpx.AsyncClient,
        mock_openai_chat_completion: dict[str, Any],
    ) -> None:
        """Test chat completion with temperature and top_p parameters."""
        with patch("local_openai2anthropic.router.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_chat_completion
            mock_response.text = json.dumps(mock_openai_chat_completion)
            mock_response.reason_phrase = "OK"

            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await client.post(
                "/v1/messages",
                json={
                    "model": "gpt-4",
                    "max_tokens": 1024,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "messages": [{"role": "user", "content": "Hello!"}],
                },
            )

            assert response.status_code == 200

    async def test_chat_with_stop_sequences(
        self,
        client: httpx.AsyncClient,
        mock_openai_chat_completion: dict[str, Any],
    ) -> None:
        """Test chat completion with stop sequences."""
        with patch("local_openai2anthropic.router.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_chat_completion
            mock_response.text = json.dumps(mock_openai_chat_completion)
            mock_response.reason_phrase = "OK"

            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await client.post(
                "/v1/messages",
                json={
                    "model": "gpt-4",
                    "max_tokens": 1024,
                    "stop_sequences": ["STOP", "END"],
                    "messages": [{"role": "user", "content": "Count to 10"}],
                },
            )

            assert response.status_code == 200

    async def test_chat_with_thinking_enabled(
        self,
        client: httpx.AsyncClient,
        mock_openai_chat_completion: dict[str, Any],
    ) -> None:
        """Test chat completion with thinking enabled."""
        with patch("local_openai2anthropic.router.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_chat_completion
            mock_response.text = json.dumps(mock_openai_chat_completion)
            mock_response.reason_phrase = "OK"

            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await client.post(
                "/v1/messages",
                json={
                    "model": "gpt-4",
                    "max_tokens": 1024,
                    "thinking": {"type": "enabled"},
                    "messages": [{"role": "user", "content": "Solve this problem"}],
                },
            )

            assert response.status_code == 200

    async def test_chat_with_tool_use(
        self,
        client: httpx.AsyncClient,
        mock_openai_chat_completion_with_tool_calls: dict[str, Any],
    ) -> None:
        """Test chat completion that returns tool calls."""
        with patch("local_openai2anthropic.router.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_chat_completion_with_tool_calls
            mock_response.text = json.dumps(mock_openai_chat_completion_with_tool_calls)
            mock_response.reason_phrase = "OK"

            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await client.post(
                "/v1/messages",
                json={
                    "model": "gpt-4",
                    "max_tokens": 1024,
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Get weather information",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"}
                                },
                                "required": ["location"],
                            },
                        }
                    ],
                    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "message"
            assert data["stop_reason"] == "tool_use"
            assert len(data["content"]) > 0
            # Check for tool_use block
            tool_use_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
            assert len(tool_use_blocks) == 1
            assert tool_use_blocks[0]["name"] == "get_weather"
            assert tool_use_blocks[0]["input"]["location"] == "Tokyo"


@pytest.mark.asyncio
class TestMessagesEndpointValidation:
    """Test request validation for /v1/messages endpoint."""

    async def test_missing_model(self, client: httpx.AsyncClient) -> None:
        """Test error when model is missing."""
        response = await client.post(
            "/v1/messages",
            json={
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
        assert "error" in data
        assert "model" in data["error"]["message"].lower()

    async def test_empty_model(self, client: httpx.AsyncClient) -> None:
        """Test error when model is empty string."""
        response = await client.post(
            "/v1/messages",
            json={
                "model": "",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"

    async def test_missing_messages(self, client: httpx.AsyncClient) -> None:
        """Test error when messages are missing."""
        response = await client.post(
            "/v1/messages",
            json={
                "model": "gpt-4",
                "max_tokens": 1024,
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
        assert "messages" in data["error"]["message"].lower()

    async def test_empty_messages(self, client: httpx.AsyncClient) -> None:
        """Test error when messages list is empty."""
        response = await client.post(
            "/v1/messages",
            json={
                "model": "gpt-4",
                "max_tokens": 1024,
                "messages": [],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"

    async def test_missing_max_tokens(self, client: httpx.AsyncClient) -> None:
        """Test error when max_tokens is missing."""
        response = await client.post(
            "/v1/messages",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
        assert "max_tokens" in data["error"]["message"].lower()

    async def test_invalid_json(self, client: httpx.AsyncClient) -> None:
        """Test error when request body is invalid JSON."""
        response = await client.post(
            "/v1/messages",
            content="invalid json {",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
        assert "json" in data["error"]["message"].lower()

    async def test_non_object_body(self, client: httpx.AsyncClient) -> None:
        """Test error when request body is not a JSON object."""
        response = await client.post(
            "/v1/messages",
            json="not an object",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
