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
    extract_thinking_from_content,
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


class TestExtractThinkingFromContent:
    """Tests for thinking content extraction from content string."""

    def test_extract_thinking_from_content_basic(self):
        """Test basic thinking tag extraction."""
        content = "<think>用户用中文说'你好'，这是一个简单的问题。我应该用中文友好地回应。\n</think>\n\n你好！很高兴见到你。"
        thinking_blocks, clean_content = extract_thinking_from_content(content)

        assert len(thinking_blocks) == 1
        assert "用户用中文说'你好'" in thinking_blocks[0]
        assert clean_content == "\n\n你好！很高兴见到你。"

    def test_extract_thinking_from_content_no_thinking(self):
        """Test content without thinking tags."""
        content = "Hello! How can I help you?"
        thinking_blocks, clean_content = extract_thinking_from_content(content)

        assert len(thinking_blocks) == 0
        assert clean_content == "Hello! How can I help you?"

    def test_extract_thinking_from_content_multiple_blocks(self):
        """Test multiple thinking blocks."""
        content = "<think>第一步思考</think>中间文本<think>第二步思考</think>"
        thinking_blocks, clean_content = extract_thinking_from_content(content)

        assert len(thinking_blocks) == 2
        assert "第一步思考" in thinking_blocks[0]
        assert "第二步思考" in thinking_blocks[1]
        assert clean_content == "中间文本"

    def test_extract_thinking_from_content_empty_thinking(self):
        """Test empty thinking tags."""
        content = "<think></think>中间文本"
        thinking_blocks, clean_content = extract_thinking_from_content(content)

        assert len(thinking_blocks) == 1
        assert thinking_blocks[0] == ""
        assert clean_content == "中间文本"

    def test_extract_thinking_from_content_empty(self):
        """Test empty content."""
        thinking_blocks, clean_content = extract_thinking_from_content("")
        assert len(thinking_blocks) == 0
        assert clean_content == ""

    def test_extract_thinking_from_content_none(self):
        """Test None content."""
        thinking_blocks, clean_content = extract_thinking_from_content(None)
        assert len(thinking_blocks) == 0
        assert clean_content == ""


class TestOpenAIToAnthropicWithThinkingTags:
    """Tests for OpenAI to Anthropic conversion with thinking tags in content."""

    def test_thinking_tags_extracted_from_content(self):
        """Test that thinking tags in content are extracted to thinking blocks."""
        completion = ChatCompletion(
            id="test-id",
            model="glm-4.7",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="<think>用户说'你好'，我应该用中文回应。\n</think>\n\n你好！很高兴见到你。",
                        reasoning_content=None,
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

        result = convert_openai_to_anthropic(completion, "glm-4.7")

        # Should have 2 content blocks: thinking and text
        assert len(result.content) == 2

        # First block should be thinking
        assert result.content[0].type == "thinking"
        assert "用户说'你好'" in result.content[0].thinking

        # Second block should be text (with thinking tags removed)
        assert result.content[1].type == "text"
        assert "你好！很高兴见到你。" in result.content[1].text
        assert "<think>" not in result.content[1].text

    def test_no_thinking_tags_unchanged(self):
        """Test that content without thinking tags is unchanged."""
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

        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello! How can I help?"

    def test_reasoning_content_takes_precedence(self):
        """Test that reasoning_content field takes precedence over content tags."""
        completion = ChatCompletion(
            id="test-id",
            model="glm-4.7",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="<think>这应该在reasoning_content中</think>",
                        reasoning_content="这是reasoning_content字段的内容",
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

        result = convert_openai_to_anthropic(completion, "glm-4.7")

        # Should use reasoning_content, not content tags
        assert len(result.content) == 1
        assert result.content[0].type == "thinking"
        assert result.content[0].thinking == "这是reasoning_content字段的内容"

    def test_multiple_thinking_blocks(self):
        """Test multiple thinking blocks in content."""
        completion = ChatCompletion(
            id="test-id",
            model="glm-4.7",
            object="chat.completion",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="<think>第一步思考</think>中间内容<think>第二步思考</think>",
                        reasoning_content=None,
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

        result = convert_openai_to_anthropic(completion, "glm-4.7")

        # Should have 3 content blocks: thinking, thinking, text
        # (all thinking blocks are grouped first, then text)
        assert len(result.content) == 3
        assert result.content[0].type == "thinking"
        assert "第一步思考" in result.content[0].thinking
        assert result.content[1].type == "thinking"
        assert "第二步思考" in result.content[1].thinking
        assert result.content[2].type == "text"
        assert result.content[2].text == "中间内容"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
