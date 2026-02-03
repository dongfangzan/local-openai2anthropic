"""
Tests for router streaming and API endpoints.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from fastapi.responses import JSONResponse

from local_openai2anthropic.router import (
    _stream_response,
    _convert_result_to_stream,
    list_models,
    count_tokens,
    ServerToolHandler,
    _add_tool_results_to_messages,
)
from local_openai2anthropic.config import Settings


# Helper functions for async testing
async def async_iter(items):
    """Create an async iterator from a list."""
    for item in items:
        yield item


class async_context_manager:
    """Async context manager helper."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        pass


class TestStreamResponse:
    """Tests for _stream_response function."""

    @pytest.mark.asyncio
    async def test_stream_successful_response(self):
        """Test successful streaming response."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        # Use a real async generator function for aiter_lines
        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123", "choices": [{"delta": {"content": "Hello"}}]}',
                "data: [DONE]",
            ]:
                yield line

        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(
            return_value=async_context_manager(mock_response)
        )

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {"Authorization": "Bearer test"},
            {"model": "test"},
            "test-model",
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert any("message_start" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_error_response(self):
        """Test streaming with error response."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.aread = AsyncMock(
            return_value=b'{"error": {"message": "Bad request"}}'
        )
        mock_response.reason_phrase = "Bad Request"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(
            return_value=async_context_manager(mock_response)
        )

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        assert any("error" in chunk for chunk in chunks)
        assert any("[DONE]" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_with_reasoning_content(self):
        """Test streaming with reasoning/thinking content."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123", "choices": [{"delta": {"reasoning_content": "Let me think..."}}]}',
                'data: {"id": "msg_123", "choices": [{"delta": {"content": "Answer"}}]}',
                "data: [DONE]",
            ]:
                yield line

        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(
            return_value=async_context_manager(mock_response)
        )

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = "".join(chunks)
        # The reasoning_content should produce "thinking" blocks
        # or the regular content should be present
        assert "thinking" in stream_text or "Answer" in stream_text

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self):
        """Test streaming with tool calls."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123", "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_123", "function": {"name": "get_weather"}}]}}]}',
                'data: {"id": "msg_123", "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}]}',
                "data: [DONE]",
            ]:
                yield line

        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(
            return_value=async_context_manager(mock_response)
        )

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = "".join(chunks)
        assert "tool_use" in stream_text

    @pytest.mark.asyncio
    async def test_stream_with_usage_only_chunk(self):
        """Test streaming with usage-only chunk."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}',
                "data: [DONE]",
            ]:
                yield line

        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(
            return_value=async_context_manager(mock_response)
        )

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = "".join(chunks)
        assert "message_delta" in stream_text
        assert "input_tokens" in stream_text

    @pytest.mark.asyncio
    async def test_stream_exception_handling(self):
        """Test exception handling in stream."""
        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=Exception("Stream error"))

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        assert any("internal_error" in chunk for chunk in chunks)


class TestConvertResultToStream:
    """Tests for _convert_result_to_stream function."""

    @pytest.mark.asyncio
    async def test_convert_text_only(self):
        """Test converting result with text only."""
        result = JSONResponse(
            content={
                "id": "msg_test",
                "model": "test-model",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Hello world"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        chunks = []
        async for chunk in _convert_result_to_stream(result, "test-model"):
            chunks.append(chunk)

        stream_text = "".join(chunks)
        assert "message_start" in stream_text
        assert "Hello world" in stream_text
        assert "message_stop" in stream_text

    @pytest.mark.asyncio
    async def test_convert_with_thinking(self):
        """Test converting result with thinking block."""
        result = JSONResponse(
            content={
                "id": "msg_test",
                "model": "test-model",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me analyze...",
                        "signature": "sig123",
                    },
                    {"type": "text", "text": "Result"},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        chunks = []
        async for chunk in _convert_result_to_stream(result, "test-model"):
            chunks.append(chunk)

        stream_text = "".join(chunks)
        assert "thinking" in stream_text
        assert "Let me analyze" in stream_text

    @pytest.mark.asyncio
    async def test_convert_with_tool_use(self):
        """Test converting result with tool_use block."""
        result = JSONResponse(
            content={
                "id": "msg_test",
                "model": "test-model",
                "role": "assistant",
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        chunks = []
        async for chunk in _convert_result_to_stream(result, "test-model"):
            chunks.append(chunk)

        stream_text = "".join(chunks)
        assert "tool_use" in stream_text
        assert "get_weather" in stream_text

    @pytest.mark.asyncio
    async def test_convert_with_server_tool_use(self):
        """Test converting result with server_tool_use block."""
        result = JSONResponse(
            content={
                "id": "msg_test",
                "model": "test-model",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_123",
                        "name": "web_search",
                        "input": {"query": "test"},
                    },
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "server_tool_use": {"web_search_requests": 1},
                },
            }
        )

        chunks = []
        async for chunk in _convert_result_to_stream(result, "test-model"):
            chunks.append(chunk)

        stream_text = "".join(chunks)
        assert "server_tool_use" in stream_text

    @pytest.mark.asyncio
    async def test_convert_with_web_search_result(self):
        """Test converting result with web_search_tool_result block."""
        result = JSONResponse(
            content={
                "id": "msg_test",
                "model": "test-model",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_123",
                        "results": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com",
                                "title": "Example",
                            }
                        ],
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com",
                                "title": "Example",
                            }
                        ],
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        chunks = []
        async for chunk in _convert_result_to_stream(result, "test-model"):
            chunks.append(chunk)

        stream_text = "".join(chunks)
        assert "web_search_tool_result" in stream_text

    @pytest.mark.asyncio
    async def test_convert_with_web_search_error_result(self):
        """Test converting result with web_search_tool_result error block."""
        result = JSONResponse(
            content={
                "id": "msg_test",
                "model": "test-model",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_123",
                        "results": {
                            "type": "web_search_tool_result_error",
                            "error_code": "max_uses_exceeded",
                        },
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "max_uses_exceeded",
                        },
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        chunks = []
        async for chunk in _convert_result_to_stream(result, "test-model"):
            chunks.append(chunk)

        stream_text = "".join(chunks)
        assert "web_search_tool_result" in stream_text
        assert "web_search_tool_result_error" in stream_text


class TestListModels:
    """Tests for list_models endpoint."""

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test successful models listing."""
        settings = Settings(
            openai_api_key="test-key",
            openai_base_url="https://api.openai.com/v1",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"data": [{"id": "gpt-4"}]})

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            result = await list_models(settings)
            assert result.status_code == 200
            assert "gpt-4" in str(result.body)

    @pytest.mark.asyncio
    async def test_list_models_with_org_and_project(self):
        """Test models listing with org and project headers."""
        settings = Settings(
            openai_api_key="test-key",
            openai_base_url="https://api.openai.com/v1",
            openai_org_id="org-123",
            openai_project_id="proj-456",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"data": []})

        # Create a proper async context manager mock
        class AsyncClientMock:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, headers=None):
                return mock_response

        with patch("httpx.AsyncClient", return_value=AsyncClientMock()):
            result = await list_models(settings)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_models_connection_error(self):
        """Test models listing with connection error."""
        settings = Settings(
            openai_api_key="test-key",
            openai_base_url="https://api.openai.com/v1",
        )

        with patch(
            "httpx.AsyncClient.get", side_effect=httpx.RequestError("Connection failed")
        ):
            with pytest.raises(Exception) as exc_info:
                await list_models(settings)
            assert getattr(exc_info.value, "status_code", None) == 502


class TestCountTokens:
    """Tests for count_tokens endpoint."""

    @pytest.mark.asyncio
    async def test_count_tokens_success(self):
        """Test successful token counting."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello world"}],
                }
            ).encode()
        )

        settings = Settings(openai_api_key="test-key")

        # Patch builtins.__import__ since tiktoken is imported inside the function
        mock_tiktoken = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        mock_tiktoken.get_encoding = MagicMock(return_value=mock_encoding)

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                return mock_tiktoken
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await count_tokens(mock_request, settings)
            assert result.status_code == 200
            content = json.loads(bytes(result.body).decode("utf-8"))
            assert content["input_tokens"] > 0

    @pytest.mark.asyncio
    async def test_count_tokens_invalid_json(self):
        """Test token counting with invalid JSON."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=b"invalid json")

        settings = Settings(openai_api_key="test-key")

        result = await count_tokens(mock_request, settings)
        assert result.status_code == 400
        content = json.loads(bytes(result.body).decode("utf-8"))
        assert content["type"] == "error"

    @pytest.mark.asyncio
    async def test_count_tokens_not_dict(self):
        """Test token counting with non-dict body."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=b'"not a dict"')

        settings = Settings(openai_api_key="test-key")

        result = await count_tokens(mock_request, settings)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_count_tokens_invalid_messages(self):
        """Test token counting with invalid messages."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "model": "test-model",
                    "messages": "not a list",
                }
            ).encode()
        )

        settings = Settings(openai_api_key="test-key")

        result = await count_tokens(mock_request, settings)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_count_tokens_with_system(self):
        """Test token counting with system prompt."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "model": "test-model",
                    "system": "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            ).encode()
        )

        settings = Settings(openai_api_key="test-key")

        # Patch builtins.__import__ since tiktoken is imported inside the function
        mock_tiktoken = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.encode = MagicMock(return_value=[1, 2, 3])
        mock_tiktoken.get_encoding = MagicMock(return_value=mock_encoding)

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                return mock_tiktoken
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await count_tokens(mock_request, settings)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_count_tokens_with_system_blocks(self):
        """Test token counting with system as list of blocks."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "model": "test-model",
                    "system": [{"type": "text", "text": "You are helpful."}],
                    "messages": [],
                }
            ).encode()
        )

        settings = Settings(openai_api_key="test-key")

        # Patch builtins.__import__ since tiktoken is imported inside the function
        mock_tiktoken = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.encode = MagicMock(return_value=[1, 2])
        mock_tiktoken.get_encoding = MagicMock(return_value=mock_encoding)

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                return mock_tiktoken
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await count_tokens(mock_request, settings)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_count_tokens_with_tools(self):
        """Test token counting with tools."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "model": "test-model",
                    "messages": [],
                    "tools": [{"type": "function", "function": {"name": "test"}}],
                }
            ).encode()
        )

        settings = Settings(openai_api_key="test-key")

        # Patch builtins.__import__ since tiktoken is imported inside the function
        mock_tiktoken = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.encode = MagicMock(return_value=[1, 2, 3, 4])
        mock_tiktoken.get_encoding = MagicMock(return_value=mock_encoding)

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                return mock_tiktoken
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await count_tokens(mock_request, settings)
            assert result.status_code == 200


class TestServerToolHandler:
    """Tests for ServerToolHandler class."""

    def test_is_server_tool_call(self):
        """Test checking if tool call is for server tool."""
        settings = Settings(openai_api_key="test-key")

        # Mock server tool class
        mock_tool_class = MagicMock()
        mock_tool_class.tool_name = "web_search"

        handler = ServerToolHandler([mock_tool_class], {}, settings)

        tool_call = {
            "function": {"name": "web_search"},
        }

        assert handler.is_server_tool_call(tool_call) is True

    def test_is_not_server_tool_call(self):
        """Test checking if tool call is not for server tool."""
        settings = Settings(openai_api_key="test-key")

        mock_tool_class = MagicMock()
        mock_tool_class.tool_name = "web_search"

        handler = ServerToolHandler([mock_tool_class], {}, settings)

        tool_call = {
            "function": {"name": "other_tool"},
        }

        assert handler.is_server_tool_call(tool_call) is False


class TestAddToolResultsToMessages:
    """Tests for _add_tool_results_to_messages function."""

    def test_add_tool_results(self):
        """Test adding tool results to messages."""
        messages = [{"role": "user", "content": "Hello"}]
        tool_calls = [{"id": "call_123", "function": {"name": "test"}}]

        mock_handler = MagicMock()

        tool_results = [
            {"role": "tool", "tool_call_id": "call_123", "content": "result"}
        ]

        result = _add_tool_results_to_messages(
            messages, tool_calls, mock_handler, tool_results=tool_results
        )

        assert len(result) == 3  # original + assistant + tool
        assert result[1]["role"] == "assistant"
        assert result[1]["tool_calls"] == tool_calls
        assert result[2]["role"] == "tool"

    def test_add_error_results(self):
        """Test adding error results to messages."""
        messages = []
        tool_calls = [
            {"id": "call_123", "openai_id": "call_123", "function": {"name": "test"}}
        ]

        mock_handler = MagicMock()

        result = _add_tool_results_to_messages(
            messages, tool_calls, mock_handler, is_error=True
        )

        assert len(result) == 2  # assistant + error tool
        assert result[1]["role"] == "tool"
        assert "error" in result[1]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
