"""
Tests for the converter module.
"""

import json
import pytest
from anthropic.types.message_create_params import MessageCreateParams
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from local_openai2anthropic.converter import (
    _strip_claude_billing_header,
    convert_anthropic_to_openai,
    convert_openai_to_anthropic,
    _convert_anthropic_message_to_openai,
)
from local_openai2anthropic.protocol import UsageWithCache


class TestAnthropicToOpenAI:
    """Tests for Anthropic to OpenAI conversion."""

    def test_simple_message(self):
        """Test simple text message conversion."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
        }
        
        result = convert_anthropic_to_openai(params)
        
        assert result["model"] == "gpt-4o"
        assert result["max_tokens"] == 1024
        assert result["messages"] == [{"role": "user", "content": "Hello!"}]
        assert result["stream"] is False

    def test_system_message(self):
        """Test system message conversion."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
        }
        
        result = convert_anthropic_to_openai(params)
        
        assert result["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
        assert result["messages"][1] == {"role": "user", "content": "Hello!"}

    def test_streaming(self):
        """Test streaming parameter conversion."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        
        result = convert_anthropic_to_openai(params)
        
        assert result["stream"] is True
        assert result["stream_options"] == {"include_usage": True}

    def test_tools(self):
        """Test tool conversion."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                }
            ],
            "tool_choice": {"type": "auto"},
        }
        
        result = convert_anthropic_to_openai(params)
        
        assert "tools" in result
        assert result["tool_choice"] == "auto"
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"

    def test_temperature_and_top_p(self):
        """Test temperature and top_p conversion."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        result = convert_anthropic_to_openai(params)
        
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

    def test_top_k(self):
        """Test top_k parameter conversion."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "top_k": 50,
        }
        
        result = convert_anthropic_to_openai(params)
        
        assert result["top_k"] == 50

    def test_stop_sequences(self):
        """Test stop sequences conversion."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["STOP", "END"],
        }
        
        result = convert_anthropic_to_openai(params)
        
        assert result["stop"] == ["STOP", "END"]

    def test_thinking_enabled(self):
        """Test thinking enabled conversion."""
        params: MessageCreateParams = {
            "model": "deepseek-r1",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "enabled"},
        }
        
        result = convert_anthropic_to_openai(params)
        
        assert result["chat_template_kwargs"] == {"thinking": True, "enable_thinking": True}

    def test_thinking_disabled(self):
        """Test thinking disabled conversion."""
        params: MessageCreateParams = {
            "model": "deepseek-r1",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "disabled"},
        }
        
        result = convert_anthropic_to_openai(params)
        
        assert result["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}

    def test_thinking_with_budget(self):
        """Test thinking with budget_tokens (accepted but ignored)."""
        params: MessageCreateParams = {
            "model": "deepseek-r1",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }

        result = convert_anthropic_to_openai(params)

        assert result["chat_template_kwargs"] == {"thinking": True, "enable_thinking": True}

    def test_thinking_adaptive(self):
        """Test adaptive thinking mode."""
        params: MessageCreateParams = {
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "adaptive"},
        }

        result = convert_anthropic_to_openai(params)

        assert result["chat_template_kwargs"] == {"thinking": True, "enable_thinking": True}


class TestStripClaudeBillingHeader:
    """Tests for Claude billing header stripping."""

    def test_strip_billing_header_from_system(self):
        """Test that Claude billing header is stripped from system prompts."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": """You are a helpful assistant.
x-anthropic-billing-header:cc version=2.1.37.3a3;cc_entrypoint=claude-vscode;cch=694d6;
Be helpful.""",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
        }

        result = convert_anthropic_to_openai(params)

        # The billing header should be stripped
        system_content = result["messages"][0]["content"]
        assert "x-anthropic-billing-header" not in system_content
        assert "cch=" not in system_content
        assert "You are a helpful assistant." in system_content
        assert "Be helpful." in system_content

    def test_strip_billing_header_with_different_cch(self):
        """Test that different cch values are all stripped."""
        params1: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": "Test content\nx-anthropic-billing-header:cc version=2.1.37;cch=aaaa;",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        params2: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": "Test content\nx-anthropic-billing-header:cc version=2.1.37;cch=bbbb;",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        result1 = convert_anthropic_to_openai(params1)
        result2 = convert_anthropic_to_openai(params2)

        # Both should produce the same cleaned content
        assert result1["messages"][0]["content"] == result2["messages"][0]["content"]
        assert result1["messages"][0]["content"] == "Test content"

    def test_strip_billing_header_from_list_system(self):
        """Test that billing header is stripped from list-style system prompts."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": "You are helpful.\nx-anthropic-billing-header:cc version=2.1.37;cch=xxx;"
                },
                {
                    "type": "text",
                    "text": "Be polite."
                }
            ],
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
        }

        result = convert_anthropic_to_openai(params)

        system_content = result["messages"][0]["content"]
        assert "x-anthropic-billing-header" not in system_content
        assert "cch=" not in system_content
        assert "You are helpful." in system_content
        assert "Be polite." in system_content

    def test_no_billing_header_unchanged(self):
        """Test that prompts without billing header are unchanged."""
        params: MessageCreateParams = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "system": "You are a helpful assistant. Be polite.",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
        }

        result = convert_anthropic_to_openai(params)

        assert result["messages"][0]["content"] == "You are a helpful assistant. Be polite."

    def test_strip_claude_billing_header_function(self):
        """Test the helper function directly."""
        # Test with billing header
        text = "Some text\nx-anthropic-billing-header:cc version=2.1.37;cch=abc123;\nMore text"
        result = _strip_claude_billing_header(text)
        assert result == "Some text\nMore text"

        # Test without billing header
        text2 = "Some text\nMore text"
        result2 = _strip_claude_billing_header(text2)
        assert result2 == "Some text\nMore text"

        # Test empty
        assert _strip_claude_billing_header("") == ""
        assert _strip_claude_billing_header(None) is None


class TestOpenAIToAnthropic:
    """Tests for OpenAI to Anthropic conversion."""

    def test_simple_response(self):
        """Test simple text response conversion."""
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
                        content="Hello! How can I help?",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )
        
        result = convert_openai_to_anthropic(completion, "gpt-4o")
        
        assert result.id == "test-id"
        assert result.model == "gpt-4o"
        assert result.role == "assistant"
        assert result.stop_reason == "end_turn"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello! How can I help?"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    def test_tool_call_response(self):
        """Test tool call response conversion."""
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
                        content=None,
                        tool_calls=[
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": json.dumps({"location": "Tokyo"}),
                                },
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=20,
                completion_tokens=30,
                total_tokens=50,
            ),
        )
        
        result = convert_openai_to_anthropic(completion, "gpt-4o")
        
        assert result.stop_reason == "tool_use"
        assert len(result.content) == 1
        assert result.content[0].type == "tool_use"
        assert result.content[0].id == "call_123"
        assert result.content[0].name == "get_weather"
        assert result.content[0].input == {"location": "Tokyo"}

    def test_max_tokens_stop(self):
        """Test max tokens stop reason conversion."""
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
                        content="Truncated...",
                    ),
                    finish_reason="length",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=100,
                total_tokens=110,
            ),
        )
        
        result = convert_openai_to_anthropic(completion, "gpt-4o")
        
        assert result.stop_reason == "max_tokens"

    def test_usage_with_cache_fields(self):
        """Test that usage object supports cache token fields."""
        usage = UsageWithCache(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=300,
        )
        
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_creation_input_tokens == 200
        assert usage.cache_read_input_tokens == 300


class TestThinkingBlockConversion:
    """Tests for thinking block conversion to OpenAI format."""

    def test_thinking_block_extraction(self):
        """Test that thinking block is extracted correctly."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "User greeted me in Chinese.", "signature": ""},
                {"type": "text", "text": "Hello! I'm Claude Code."},
            ],
        }

        result, has_thinking = _convert_anthropic_message_to_openai(msg)

        assert has_thinking is True
        assert result[0]["role"] == "assistant"
        assert result[0]["reasoning"] == "User greeted me in Chinese."
        assert result[0]["reasoning_content"] == "User greeted me in Chinese."
        assert result[0]["content"] == "Hello! I'm Claude Code."

    def test_thinking_block_with_tool_use_block(self):
        """Test thinking block combined with tool_use block in content."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I need to search for weather.", "signature": ""},
                {"type": "text", "text": "Let me check the weather for you."},
                {"type": "tool_use", "id": "tool_1", "name": "web_search", "input": {"query": "weather"}},
            ],
        }

        result, has_thinking = _convert_anthropic_message_to_openai(msg)

        assert has_thinking is True
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "web_search"
        assert result[0]["reasoning"] == "I need to search for weather."
        assert result[0]["reasoning_content"] == "I need to search for weather."
        assert result[0]["content"] == "Let me check the weather for you."

    def test_no_thinking_block(self):
        """Test message without thinking block."""
        msg = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello!"}],
        }

        result, has_thinking = _convert_anthropic_message_to_openai(msg)

        assert has_thinking is False
        assert result[0]["content"] == "Hello!"

    def test_multiple_text_blocks_with_thinking(self):
        """Test multiple text blocks with thinking block."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "This is my thinking.", "signature": ""},
                {"type": "text", "text": "First part."},
                {"type": "text", "text": "Second part."},
            ],
        }

        result, has_thinking = _convert_anthropic_message_to_openai(msg)

        assert has_thinking is True
        assert result[0]["reasoning"] == "This is my thinking."
        assert result[0]["reasoning_content"] == "This is my thinking."
        assert result[0]["content"] == [
            {"type": "text", "text": "First part."},
            {"type": "text", "text": "Second part."},
        ]

    def test_thinking_block_string_content(self):
        """Test thinking block with string content (edge case)."""
        msg = {
            "role": "assistant",
            "content": "Just a simple text response",
        }

        result, has_thinking = _convert_anthropic_message_to_openai(msg)

        assert has_thinking is False
        assert result[0]["content"] == "Just a simple text response"

    def test_thinking_block_empty_content(self):
        """Test thinking block with empty content list."""
        msg = {
            "role": "assistant",
            "content": [],
        }

        result, has_thinking = _convert_anthropic_message_to_openai(msg)

        assert has_thinking is False
        assert result[0]["content"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
