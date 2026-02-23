"""
Additional tests for the converter module covering edge cases.
"""

import json
import pytest
from anthropic.types.message_create_params import MessageCreateParams
from anthropic.types import ToolResultBlockParam
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from local_openai2anthropic.converter import (
    convert_anthropic_to_openai,
    convert_openai_to_anthropic,
    _convert_anthropic_message_to_openai,
    _build_usage_with_cache,
)
from local_openai2anthropic.protocol import UsageWithCache


class TestConvertAnthropicToOpenAIEdgeCases:
    """Tests for convert_anthropic_to_openai edge cases."""

    def test_empty_messages(self):
        """Test conversion with empty messages list."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [],
        }

        result = convert_anthropic_to_openai(params)

        assert result["messages"] == []

    def test_system_as_list(self):
        """Test conversion with system as list of blocks."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = convert_anthropic_to_openai(params)

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful.Be concise."

    def test_system_list_with_non_text(self):
        """Test system list with non-text blocks."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "image", "source": {}},  # Should be ignored
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = convert_anthropic_to_openai(params)

        assert result["messages"][0]["content"] == "You are helpful."

    def test_repetition_penalty(self):
        """Test repetition_penalty parameter."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "repetition_penalty": 1.5,
        }

        result = convert_anthropic_to_openai(params)

        assert result["repetition_penalty"] == 1.5

    def test_default_repetition_penalty(self):
        """Test default repetition_penalty value."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = convert_anthropic_to_openai(params)

        assert result["repetition_penalty"] == 1.1


class TestConvertAnthropicMessageEdgeCases:
    """Tests for _convert_anthropic_message_to_openai edge cases."""

    def test_image_block(self):
        """Test conversion of image block."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "media_type": "image/png",
                        "data": "base64data",
                    },
                }
            ],
        }

        result, _ = _convert_anthropic_message_to_openai(msg)

        assert len(result) == 1
        assert result[0]["content"][0]["type"] == "image_url"
        assert "data:image/png;base64,base64data" in result[0]["content"][0]["image_url"]["url"]

    def test_image_block_default_media_type(self):
        """Test image block with default media type."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "data": "base64data",
                    },
                }
            ],
        }

        result, _ = _convert_anthropic_message_to_openai(msg)

        assert "data:image/jpeg;base64,base64data" in result[0]["content"][0]["image_url"]["url"]

    def test_tool_use_block(self):
        """Test conversion of tool_use block."""
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {"location": "Tokyo"},
                }
            ],
        }

        result, _ = _convert_anthropic_message_to_openai(msg)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["id"] == "tool_123"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"location": "Tokyo"}

    def test_tool_result_block(self):
        """Test conversion of tool_result block."""
        msg_param: ToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": "tool_123",
            "content": "The weather is sunny.",
        }

        result, _ = _convert_anthropic_message_to_openai({"role": "user", "content": [msg_param]})

        # Tool results create a separate message, so we have 2 messages
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "tool_123"
        assert result[1]["content"] == "The weather is sunny."

    def test_tool_result_with_list_content(self):
        """Test tool_result with list content."""
        msg_param: ToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": "tool_123",
            "content": [
                {"type": "text", "text": "Result 1"},
                {"type": "text", "text": "Result 2"},
            ],
        }

        result, _ = _convert_anthropic_message_to_openai({"role": "user", "content": [msg_param]})

        # Tool result is in the second message
        assert result[1]["content"] == "Result 1\nResult 2"

    def test_tool_result_with_image(self):
        """Test tool_result with image in content."""
        msg_param: ToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": "tool_123",
            "content": [
                {"type": "text", "text": "Here is the image:"},
                {"type": "image", "source": {}},
            ],
        }

        result, _ = _convert_anthropic_message_to_openai({"role": "user", "content": [msg_param]})

        # Tool result is in the second message
        assert "[Image content]" in result[1]["content"]

    def test_mixed_content(self):
        """Test message with mixed content types."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }

        result, _ = _convert_anthropic_message_to_openai(msg)

        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["text"] == "Hello"
        assert result[0]["content"][1]["text"] == "World"

    def test_string_block_in_list(self):
        """Test string block in content list."""
        msg = {
            "role": "user",
            "content": ["Hello", "World"],
        }

        result, _ = _convert_anthropic_message_to_openai(msg)

        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello"

    def test_empty_content(self):
        """Test message with empty content."""
        msg = {
            "role": "user",
            "content": [],
        }

        result, _ = _convert_anthropic_message_to_openai(msg)

        assert result[0]["content"] == ""

    def test_tool_use_and_result_combined(self):
        """Test assistant message with tool_use followed by user tool_result."""
        # This tests separate messages, not combined
        assistant_msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {"location": "Tokyo"},
                }
            ],
        }

        result, _ = _convert_anthropic_message_to_openai(assistant_msg)

        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]


class TestConvertOpenAIToAnthropicEdgeCases:
    """Tests for convert_openai_to_anthropic edge cases."""

    def test_content_filter_stop(self):
        """Test content_filter finish reason."""
        completion = ChatCompletion(
            id="test-id",
            model="gpt-4o",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Content filtered",
                    ),
                    finish_reason="content_filter",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )

        result = convert_openai_to_anthropic(completion, "gpt-4o")

        assert result.stop_reason == "end_turn"


class TestBuildUsageWithCache:
    """Tests for _build_usage_with_cache function."""

    def test_basic_usage(self):
        """Test basic usage without cache."""
        result = _build_usage_with_cache(
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert isinstance(result, UsageWithCache)
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cache_creation_input_tokens is None
        assert result.cache_read_input_tokens is None

    def test_usage_with_cache(self):
        """Test usage with cache tokens."""
        result = _build_usage_with_cache(
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=300,
        )

        assert result.cache_creation_input_tokens == 200
        assert result.cache_read_input_tokens == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
