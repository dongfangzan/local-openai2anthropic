"""
Additional tests for the router module covering edge cases.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from local_openai2anthropic.router import (
    _generate_server_tool_id,
    _normalize_usage,
    _chunk_text,
    _estimate_input_tokens,
    get_request_settings,
)
from local_openai2anthropic.config import Settings


class TestGenerateServerToolId:
    """Tests for _generate_server_tool_id function."""

    def test_id_format(self):
        """Test that generated ID has correct format."""
        tool_id = _generate_server_tool_id()

        assert tool_id.startswith("srvtoolu_")
        assert len(tool_id) == 33  # "srvtoolu_" + 24 random chars

    def test_id_uniqueness(self):
        """Test that generated IDs are unique."""
        ids = [_generate_server_tool_id() for _ in range(100)]

        assert len(set(ids)) == 100

    def test_id_characters(self):
        """Test that ID contains only valid characters."""
        tool_id = _generate_server_tool_id()

        # After prefix, should only contain lowercase letters and digits
        random_part = tool_id[9:]
        assert all(c.islower() or c.isdigit() for c in random_part)


class TestNormalizeUsage:
    """Tests for _normalize_usage function."""

    def test_none_input(self):
        """Test with None input."""
        result = _normalize_usage(None)

        assert result is None

    def test_empty_dict(self):
        """Test with empty dict."""
        result = _normalize_usage({})

        assert result is None

    def test_allowed_keys(self):
        """Test that only allowed keys are kept."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 300,
            "server_tool_use": {"web_search_requests": 1},
        }

        result = _normalize_usage(usage)

        assert result == usage

    def test_disallowed_keys_removed(self):
        """Test that disallowed keys are removed."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "extra_key": "should_be_removed",
            "another_extra": 123,
        }

        result = _normalize_usage(usage)

        assert "extra_key" not in result
        assert "another_extra" not in result
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_non_dict_input(self):
        """Test with non-dict input."""
        result = _normalize_usage("not a dict")

        assert result == "not a dict"


class TestChunkText:
    """Tests for _chunk_text function."""

    def test_empty_text(self):
        """Test with empty text."""
        result = _chunk_text("")

        assert result == []

    def test_short_text(self):
        """Test with text shorter than chunk size."""
        result = _chunk_text("Hello", chunk_size=10)

        assert result == ["Hello"]

    def test_exact_chunk_size(self):
        """Test with text exactly matching chunk size."""
        result = _chunk_text("Hello World", chunk_size=11)

        assert result == ["Hello World"]

    def test_multiple_chunks(self):
        """Test with text requiring multiple chunks."""
        result = _chunk_text("Hello World", chunk_size=5)

        assert result == ["Hello", " Worl", "d"]

    def test_custom_chunk_size(self):
        """Test with custom chunk size."""
        result = _chunk_text("ABCDEFGHIJ", chunk_size=3)

        assert result == ["ABC", "DEF", "GHI", "J"]


class TestEstimateInputTokens:
    """Tests for _estimate_input_tokens function."""

    def test_empty_params(self):
        """Test with empty params."""
        result = _estimate_input_tokens({})

        assert result == 0

    def test_system_string(self):
        """Test with system as string."""
        params = {"system": "You are a helpful assistant."}

        result = _estimate_input_tokens(params)

        assert result > 0

    def test_simple_message(self):
        """Test with simple message."""
        params = {
            "messages": [{"role": "user", "content": "Hello"}]
        }

        result = _estimate_input_tokens(params)

        assert result > 0

    def test_message_with_list_content(self):
        """Test with message containing list content."""
        params = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                }
            ]
        }

        result = _estimate_input_tokens(params)

        assert result > 0

    def test_message_with_image(self):
        """Test with message containing image."""
        params = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at this:"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                }
            ]
        }

        result = _estimate_input_tokens(params)

        # Should add 85 tokens for the image
        assert result >= 85

    def test_message_with_tool_calls(self):
        """Test with message containing tool calls."""
        params = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                }
            ]
        }

        result = _estimate_input_tokens(params)

        assert result > 0

    def test_with_tools(self):
        """Test with tools parameter."""
        params = {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "description": "Get weather"},
                }
            ],
        }

        result = _estimate_input_tokens(params)

        assert result > 0

    def test_with_tool_choice(self):
        """Test with tool_choice parameter."""
        params = {
            "messages": [],
            "tool_choice": "auto",
        }

        result = _estimate_input_tokens(params)

        assert result > 0

    def test_with_response_format(self):
        """Test with response_format parameter."""
        params = {
            "messages": [],
            "response_format": {"type": "json_object"},
        }

        result = _estimate_input_tokens(params)

        assert result > 0

    def test_invalid_message_content(self):
        """Test with invalid message content."""
        params = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "unknown_type"}],
                }
            ]
        }

        result = _estimate_input_tokens(params)

        # Should not crash
        assert result >= 0

    def test_non_dict_message(self):
        """Test with non-dict message."""
        params = {
            "messages": ["not a dict"]
        }

        result = _estimate_input_tokens(params)

        assert result == 0


class TestGetRequestSettings:
    """Tests for get_request_settings function."""

    def test_from_app_state(self):
        """Test getting settings from app state."""
        mock_settings = Settings(openai_api_key="test-key")
        mock_state = MagicMock()
        mock_state.settings = mock_settings
        mock_app = MagicMock()
        mock_app.state = mock_state
        mock_request = MagicMock()
        mock_request.app = mock_app

        result = get_request_settings(mock_request)

        assert result.openai_api_key == "test-key"

    def test_fallback_to_get_settings(self):
        """Test fallback to get_settings when app state not available."""
        mock_request = MagicMock()
        mock_request.app = None

        with patch("local_openai2anthropic.router.get_settings") as mock_get_settings:
            mock_get_settings.return_value = Settings(openai_api_key="fallback-key")
            result = get_request_settings(mock_request)

        assert result.openai_api_key == "fallback-key"

    def test_no_app(self):
        """Test when request has no app."""
        mock_request = MagicMock()
        mock_request.app = None

        with patch("local_openai2anthropic.router.get_settings") as mock_get_settings:
            mock_get_settings.return_value = Settings(openai_api_key="test")
            result = get_request_settings(mock_request)

        assert isinstance(result, Settings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
