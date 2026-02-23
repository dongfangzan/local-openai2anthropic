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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
