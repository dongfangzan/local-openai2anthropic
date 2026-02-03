"""
Comprehensive tests for router module to improve coverage.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.responses import JSONResponse

from local_openai2anthropic.router import (
    _count_tokens,
    _estimate_input_tokens,
    _stream_response,
    _convert_result_to_stream,
    _normalize_usage,
    create_message,
    count_tokens,
    ServerToolHandler,
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


class TestCountTokens:
    """Tests for _count_tokens function."""

    def test_count_tokens_with_tiktoken(self):
        """Test token counting when tiktoken is available."""
        # tiktoken is imported locally in the function, so we need to patch builtins.__import__
        mock_tiktoken = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        mock_tiktoken.get_encoding = MagicMock(return_value=mock_encoding)

        with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: mock_tiktoken if name == "tiktoken" else __import__(name, *args, **kwargs)):
            result = _count_tokens("Hello world")

        assert result == 5
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")

    def test_count_tokens_without_tiktoken(self):
        """Test token counting when tiktoken is not available."""
        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("No tiktoken")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _count_tokens("Hello world")

        assert result == 0


class TestEstimateInputTokensWithoutTiktoken:
    """Tests for _estimate_input_tokens when tiktoken is not available."""

    def test_estimate_without_tiktoken(self):
        """Test estimation returns 0 when tiktoken is not available."""
        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("No tiktoken")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            params = {
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "Hello"}],
            }

            result = _estimate_input_tokens(params)

            assert result == 0


class TestStreamResponseComprehensive:
    """Comprehensive tests for _stream_response function."""

    @pytest.mark.asyncio
    async def test_stream_with_content_filter_stop(self):
        """Test streaming with content_filter stop reason."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123", "choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
                'data: {"id": "msg_123", "choices": [{"delta": {}, "finish_reason": "content_filter"}]}',
                'data: [DONE]',
            ]:
                yield line
        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=async_context_manager(mock_response))

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'message_start' in stream_text
        assert 'message_stop' in stream_text

    @pytest.mark.asyncio
    async def test_stream_with_length_stop(self):
        """Test streaming with length stop reason."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123", "choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
                'data: {"id": "msg_123", "choices": [{"delta": {}, "finish_reason": "length"}]}',
                'data: [DONE]',
            ]:
                yield line
        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=async_context_manager(mock_response))

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'max_tokens' in stream_text

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls_stop(self):
        """Test streaming with tool_calls stop reason."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123", "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_123", "function": {"name": "get_weather"}}]}, "finish_reason": null}]}',
                'data: {"id": "msg_123", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]}',
                'data: [DONE]',
            ]:
                yield line
        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=async_context_manager(mock_response))

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'tool_use' in stream_text or 'message_stop' in stream_text

    @pytest.mark.asyncio
    async def test_stream_non_json_data(self):
        """Test streaming with non-JSON data lines."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123", "choices": [{"delta": {"content": "Hello"}}]}',
                'data: not valid json',
                'data: [DONE]',
            ]:
                yield line
        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=async_context_manager(mock_response))

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        # Should skip invalid JSON but continue processing
        stream_text = ''.join(chunks)
        assert 'message_start' in stream_text

    @pytest.mark.asyncio
    async def test_stream_empty_content(self):
        """Test streaming with empty content delta."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123", "choices": [{"delta": {"content": ""}}]}',
                'data: [DONE]',
            ]:
                yield line
        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=async_context_manager(mock_response))

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'message_start' in stream_text
        assert 'message_stop' in stream_text

    @pytest.mark.asyncio
    async def test_stream_no_content_field(self):
        """Test streaming with no content field in delta."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123", "choices": [{"delta": {}}]}',
                'data: [DONE]',
            ]:
                yield line
        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=async_context_manager(mock_response))

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'message_start' in stream_text

    @pytest.mark.asyncio
    async def test_stream_error_non_json(self):
        """Test streaming error response with non-JSON error body."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")
        mock_response.reason_phrase = "Internal Server Error"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=async_context_manager(mock_response))

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'error' in stream_text
        assert 'Internal Server Error' in stream_text

    @pytest.mark.asyncio
    async def test_stream_error_empty_body(self):
        """Test streaming error response with empty body."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.aread = AsyncMock(return_value=b"")
        mock_response.reason_phrase = "Bad Request"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=async_context_manager(mock_response))

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'error' in stream_text

    @pytest.mark.asyncio
    async def test_stream_no_choices_field(self):
        """Test streaming with chunks that have no choices field."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def aiter_lines():
            for line in [
                'data: {"id": "msg_123"}',
                'data: [DONE]',
            ]:
                yield line
        mock_response.aiter_lines = aiter_lines

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=async_context_manager(mock_response))

        chunks = []
        async for chunk in _stream_response(
            mock_client,
            "http://test.com",
            {},
            {},
            "test-model",
        ):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'message_start' in stream_text


class TestConvertResultToStreamComprehensive:
    """Additional tests for _convert_result_to_stream."""

    @pytest.mark.asyncio
    async def test_convert_with_max_tokens_stop(self):
        """Test converting result with max_tokens stop reason."""
        result = JSONResponse(content={
            "id": "msg_test",
            "model": "test-model",
            "role": "assistant",
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

        chunks = []
        async for chunk in _convert_result_to_stream(result, "test-model"):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'max_tokens' in stream_text

    @pytest.mark.asyncio
    async def test_convert_with_tool_use_stop(self):
        """Test converting result with tool_use stop reason."""
        result = JSONResponse(content={
            "id": "msg_test",
            "model": "test-model",
            "role": "assistant",
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "tool_123", "name": "get_weather", "input": {}}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

        chunks = []
        async for chunk in _convert_result_to_stream(result, "test-model"):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'tool_use' in stream_text

    @pytest.mark.asyncio
    async def test_convert_with_cache_tokens(self):
        """Test converting result with cache token counts."""
        result = JSONResponse(content={
            "id": "msg_test",
            "model": "test-model",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 50,
            },
        })

        chunks = []
        async for chunk in _convert_result_to_stream(result, "test-model"):
            chunks.append(chunk)

        stream_text = ''.join(chunks)
        assert 'cache_creation_input_tokens' in stream_text
        assert 'cache_read_input_tokens' in stream_text


class TestCreateMessageValidation:
    """Tests for create_message endpoint validation."""

    @pytest.mark.asyncio
    async def test_create_message_non_dict_body(self):
        """Test create_message with non-dict body."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=b'"not a dict"')

        settings = Settings(openai_api_key="test-key")

        result = await create_message(mock_request, settings)

        assert result.status_code == 400
        content = json.loads(result.body)
        assert content["type"] == "error"

    @pytest.mark.asyncio
    async def test_create_message_empty_model(self):
        """Test create_message with empty model string."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=json.dumps({
            "model": "   ",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        }).encode())

        settings = Settings(openai_api_key="test-key")

        result = await create_message(mock_request, settings)

        assert result.status_code == 400
        content = json.loads(result.body)
        assert "Model must be a non-empty string" in content["error"]["message"]

    @pytest.mark.asyncio
    async def test_create_message_empty_messages(self):
        """Test create_message with empty messages list."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=json.dumps({
            "model": "gpt-4",
            "messages": [],
            "max_tokens": 1024,
        }).encode())

        settings = Settings(openai_api_key="test-key")

        result = await create_message(mock_request, settings)

        assert result.status_code == 400
        content = json.loads(result.body)
        assert "Messages must be a non-empty list" in content["error"]["message"]

    @pytest.mark.asyncio
    async def test_create_message_non_list_messages(self):
        """Test create_message with non-list messages."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=json.dumps({
            "model": "gpt-4",
            "messages": "not a list",
            "max_tokens": 1024,
        }).encode())

        settings = Settings(openai_api_key="test-key")

        result = await create_message(mock_request, settings)

        assert result.status_code == 400
        content = json.loads(result.body)
        assert "Messages must be a non-empty list" in content["error"]["message"]

    @pytest.mark.asyncio
    async def test_create_message_missing_max_tokens(self):
        """Test create_message without max_tokens."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=json.dumps({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode())

        settings = Settings(openai_api_key="test-key")

        result = await create_message(mock_request, settings)

        assert result.status_code == 400
        content = json.loads(result.body)
        assert "max_tokens is required" in content["error"]["message"]

    @pytest.mark.asyncio
    async def test_create_message_non_int_max_tokens(self):
        """Test create_message with non-int max_tokens."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=json.dumps({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": "1024",
        }).encode())

        settings = Settings(openai_api_key="test-key")

        result = await create_message(mock_request, settings)

        assert result.status_code == 400
        content = json.loads(result.body)
        assert "max_tokens is required" in content["error"]["message"]


class TestCountTokensEndpoint:
    """Tests for count_tokens endpoint."""

    @pytest.mark.asyncio
    async def test_count_tokens_exception(self):
        """Test count_tokens with general exception."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode())

        settings = Settings(openai_api_key="test-key")

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise Exception("Unexpected error")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await count_tokens(mock_request, settings)

            assert result.status_code == 500
            content = json.loads(result.body)
            assert content["type"] == "error"
            assert "Failed to count tokens" in content["error"]["message"]

    @pytest.mark.asyncio
    async def test_count_tokens_with_image_in_messages(self):
        """Test count_tokens with image in messages."""
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=json.dumps({
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at this:"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                    ],
                }
            ],
        }).encode())

        settings = Settings(openai_api_key="test-key")

        mock_tiktoken = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.encode = MagicMock(return_value=[1, 2, 3])
        mock_tiktoken.get_encoding = MagicMock(return_value=mock_encoding)

        with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: mock_tiktoken if name == "tiktoken" else __import__(name, *args, **kwargs)):
            result = await count_tokens(mock_request, settings)

            assert result.status_code == 200
            content = json.loads(result.body)
            # Should include 85 tokens for the image
            assert content["input_tokens"] >= 85


class TestServerToolHandlerComprehensive:
    """Comprehensive tests for ServerToolHandler."""

    @pytest.mark.asyncio
    async def test_execute_tool_with_none_args(self):
        """Test execute_tool when extract_call_args returns None."""
        settings = Settings(openai_api_key="test-key")

        mock_tool_class = MagicMock()
        mock_tool_class.tool_name = "test_tool"
        mock_tool_class.extract_call_args = MagicMock(return_value=None)

        mock_result = MagicMock()
        mock_result.usage_increment = {"test": 1}
        mock_result.success = True
        mock_tool_class.execute = AsyncMock(return_value=mock_result)
        mock_tool_class.build_content_blocks = MagicMock(return_value=[])
        mock_tool_class.build_tool_result_message = MagicMock(return_value={})

        handler = ServerToolHandler([mock_tool_class], {}, settings)

        tool_call = {
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": "{}"},
        }

        await handler.execute_tool(tool_call)

        # Should use empty dict when args is None
        mock_tool_class.execute.assert_called_once()
        call_args = mock_tool_class.execute.call_args[0]
        assert call_args[2] == {}  # args should be empty dict


class TestNormalizeUsageEdgeCases:
    """Edge case tests for _normalize_usage."""

    def test_normalize_usage_with_all_allowed_keys(self):
        """Test that all allowed keys are preserved."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 300,
            "server_tool_use": {"web_search": 1},
        }

        result = _normalize_usage(usage)

        assert result == usage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
