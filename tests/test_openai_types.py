"""
Tests for the openai_types module.
"""

import pytest
from local_openai2anthropic.openai_types import (
    ChatCompletionToolFunction,
    ChatCompletionToolParam,
    CompletionCreateParams,
    Function,
    ChatCompletionMessageToolCall,
    ChatCompletionMessage,
    Choice,
    FunctionDelta,
    ChatCompletionDeltaToolCall,
    ChoiceDelta,
    StreamingChoice,
    CompletionUsage,
    ChatCompletion,
    ChatCompletionChunk,
)


class TestFunction:
    """Tests for Function model."""

    def test_function_creation(self):
        """Test function creation."""
        func = Function(
            name="get_weather",
            arguments='{"location": "Tokyo"}',
        )

        assert func.name == "get_weather"
        assert func.arguments == '{"location": "Tokyo"}'


class TestChatCompletionMessageToolCall:
    """Tests for ChatCompletionMessageToolCall model."""

    def test_tool_call_creation(self):
        """Test tool call creation."""
        func = Function(name="get_weather", arguments='{"location": "Tokyo"}')
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            function=func,
        )

        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"


class TestChatCompletionMessage:
    """Tests for ChatCompletionMessage model."""

    def test_basic_message(self):
        """Test basic message creation."""
        message = ChatCompletionMessage(
            role="assistant",
            content="Hello!",
        )

        assert message.role == "assistant"
        assert message.content == "Hello!"
        assert message.tool_calls is None
        assert message.reasoning_content is None

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        func = Function(name="get_weather", arguments='{"location": "Tokyo"}')
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            function=func,
        )
        message = ChatCompletionMessage(
            role="assistant",
            tool_calls=[tool_call],
        )

        assert message.content is None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].id == "call_123"

    def test_message_with_reasoning(self):
        """Test message with reasoning content."""
        message = ChatCompletionMessage(
            role="assistant",
            content="The answer is 42.",
            reasoning_content="Let me think about this... 6 * 7 = 42",
        )

        assert message.content == "The answer is 42."
        assert message.reasoning_content == "Let me think about this... 6 * 7 = 42"


class TestChoice:
    """Tests for Choice model."""

    def test_choice_creation(self):
        """Test choice creation."""
        message = ChatCompletionMessage(role="assistant", content="Hello!")
        choice = Choice(
            message=message,
            finish_reason="stop",
        )

        assert choice.index == 0
        assert choice.message.content == "Hello!"
        assert choice.finish_reason == "stop"

    def test_choice_with_index(self):
        """Test choice with explicit index."""
        message = ChatCompletionMessage(role="assistant", content="Hello!")
        choice = Choice(
            index=1,
            message=message,
        )

        assert choice.index == 1


class TestFunctionDelta:
    """Tests for FunctionDelta model."""

    def test_empty_delta(self):
        """Test empty function delta."""
        delta = FunctionDelta()

        assert delta.name is None
        assert delta.arguments is None

    def test_partial_delta(self):
        """Test partial function delta."""
        delta = FunctionDelta(
            name="get_weather",
            arguments=None,
        )

        assert delta.name == "get_weather"
        assert delta.arguments is None


class TestChatCompletionDeltaToolCall:
    """Tests for ChatCompletionDeltaToolCall model."""

    def test_empty_tool_call(self):
        """Test empty tool call delta."""
        tool_call = ChatCompletionDeltaToolCall()

        assert tool_call.index == 0
        assert tool_call.id is None
        assert tool_call.type is None
        assert tool_call.function is None

    def test_tool_call_with_function(self):
        """Test tool call delta with function."""
        func_delta = FunctionDelta(name="get_weather")
        tool_call = ChatCompletionDeltaToolCall(
            id="call_123",
            type="function",
            function=func_delta,
        )

        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"


class TestChoiceDelta:
    """Tests for ChoiceDelta model."""

    def test_empty_delta(self):
        """Test empty choice delta."""
        delta = ChoiceDelta()

        assert delta.role is None
        assert delta.content is None
        assert delta.tool_calls is None
        assert delta.reasoning_content is None

    def test_delta_with_content(self):
        """Test choice delta with content."""
        delta = ChoiceDelta(
            role="assistant",
            content="Hello",
        )

        assert delta.role == "assistant"
        assert delta.content == "Hello"

    def test_delta_with_reasoning(self):
        """Test choice delta with reasoning content."""
        delta = ChoiceDelta(
            content="The answer is 42.",
            reasoning_content="Thinking...",
        )

        assert delta.content == "The answer is 42."
        assert delta.reasoning_content == "Thinking..."


class TestStreamingChoice:
    """Tests for StreamingChoice model."""

    def test_streaming_choice(self):
        """Test streaming choice creation."""
        delta = ChoiceDelta(content="Hello")
        choice = StreamingChoice(
            delta=delta,
            finish_reason=None,
        )

        assert choice.index == 0
        assert choice.delta.content == "Hello"
        assert choice.finish_reason is None


class TestCompletionUsage:
    """Tests for CompletionUsage model."""

    def test_basic_usage(self):
        """Test basic usage creation."""
        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    def test_usage_with_cache(self):
        """Test usage with cache tokens."""
        usage = CompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=300,
        )

        assert usage.cache_creation_input_tokens == 200
        assert usage.cache_read_input_tokens == 300


class TestChatCompletion:
    """Tests for ChatCompletion model."""

    def test_basic_completion(self):
        """Test basic completion creation."""
        message = ChatCompletionMessage(role="assistant", content="Hello!")
        choice = Choice(message=message, finish_reason="stop")
        completion = ChatCompletion(
            id="comp_123",
            created=1234567890,
            model="gpt-4",
            choices=[choice],
        )

        assert completion.id == "comp_123"
        assert completion.object == "chat.completion"
        assert completion.created == 1234567890
        assert completion.model == "gpt-4"
        assert len(completion.choices) == 1
        assert completion.usage is None

    def test_completion_with_usage(self):
        """Test completion with usage."""
        message = ChatCompletionMessage(role="assistant", content="Hello!")
        choice = Choice(message=message, finish_reason="stop")
        usage = CompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        completion = ChatCompletion(
            id="comp_123",
            created=1234567890,
            model="gpt-4",
            choices=[choice],
            usage=usage,
        )

        assert completion.usage.prompt_tokens == 10
        assert completion.usage.completion_tokens == 5


class TestChatCompletionChunk:
    """Tests for ChatCompletionChunk model."""

    def test_basic_chunk(self):
        """Test basic chunk creation."""
        delta = ChoiceDelta(content="Hello")
        choice = StreamingChoice(delta=delta)
        chunk = ChatCompletionChunk(
            id="chunk_123",
            created=1234567890,
            model="gpt-4",
            choices=[choice],
        )

        assert chunk.id == "chunk_123"
        assert chunk.object == "chat.completion.chunk"
        assert chunk.created == 1234567890
        assert chunk.model == "gpt-4"
        assert len(chunk.choices) == 1


class TestTypedDictTypes:
    """Tests for TypedDict types."""

    def test_chat_completion_tool_function(self):
        """Test ChatCompletionToolFunction TypedDict."""
        func: ChatCompletionToolFunction = {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
            },
        }

        assert func["name"] == "get_weather"
        assert func["description"] == "Get weather information"

    def test_chat_completion_tool_param(self):
        """Test ChatCompletionToolParam TypedDict."""
        tool: ChatCompletionToolParam = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {"type": "object"},
            },
        }

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"

    def test_completion_create_params(self):
        """Test CompletionCreateParams TypedDict."""
        params: CompletionCreateParams = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False,
        }

        assert params["model"] == "gpt-4"
        assert params["temperature"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
