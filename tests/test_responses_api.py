# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI Responses API bridge (/v1/responses)."""

from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from local_openai2anthropic.config import Settings
from local_openai2anthropic.main import create_app
from local_openai2anthropic.responses_converter import (
    convert_chat_completion_to_responses,
    convert_responses_to_chat_completion,
    stream_responses_from_chat_completion,
)


@pytest.fixture
def settings():
    return Settings(
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        request_timeout=30.0,
        api_key=None,
    )


@pytest.fixture
def client(settings):
    app = create_app(settings)
    return TestClient(app)


def _upstream_chat_completion(content="Hi!", reasoning=None, tool_calls=None, finish_reason="stop"):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason},
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


# ── convert_responses_to_chat_completion ────────────────────────────────────


class TestResponsesToChatConversion:
    def test_string_input(self):
        out = convert_responses_to_chat_completion(
            {"model": "m", "input": "hello"}
        )
        assert out["messages"] == [{"role": "user", "content": "hello"}]
        assert out["model"] == "m"
        assert out["stream"] is False

    def test_instructions_become_system(self):
        out = convert_responses_to_chat_completion(
            {"model": "m", "input": "hi", "instructions": "Be nice"}
        )
        assert out["messages"][0] == {"role": "system", "content": "Be nice"}

    def test_instructions_list_form(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "hi",
                "instructions": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
            }
        )
        assert out["messages"][0] == {"role": "system", "content": "A\nB"}

    def test_list_message_input(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        )
        assert out["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_multipart_content_text_and_image(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "see image"},
                            {"type": "input_image", "image_url": "data:image/png;base64,xxx"},
                        ],
                    }
                ],
            }
        )
        msg = out["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert any(p["type"] == "image_url" for p in msg["content"])

    def test_function_call_input_history(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": "weather?"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"sf"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "sunny",
                    },
                ],
            }
        )
        # user, assistant(tool_calls), tool(result)
        assert out["messages"][0]["role"] == "user"
        assert out["messages"][1]["role"] == "assistant"
        assert out["messages"][1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert out["messages"][2]["role"] == "tool"

    def test_orphan_function_call_gets_placeholder_result(self):
        """Regression for GitHub issue #3: orphan function_call (no output)
        must get a backfilled placeholder tool message so vLLM/SGLang accept it.
        """
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": "weather?"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"sf"}',
                    },
                    {"role": "user", "content": "thanks"},
                ],
            }
        )
        msgs = out["messages"]
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "call_1"
        assert msgs[2]["content"]

    def test_tools_and_tool_choice(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "description": "d",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )
        assert out["tools"][0]["function"]["name"] == "f"
        assert out["tool_choice"] == "auto"

    def test_tools_flat_form(self):
        # The OpenAI SDK sends function tools in the flat shape.
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "tools": [
                    {
                        "type": "function",
                        "name": "flat_fn",
                        "description": "a flat tool",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        )
        assert out["tools"][0]["function"]["name"] == "flat_fn"
        assert out["tools"][0]["function"]["description"] == "a flat tool"
        assert out["tools"][0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_tool_choice_function(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "tools": [
                    {"type": "function", "function": {"name": "f", "parameters": {}}}
                ],
                "tool_choice": {"type": "function", "function": {"name": "f"}},
            }
        )
        assert out["tool_choice"] == {"type": "function", "function": {"name": "f"}}

    def test_reasoning_effort_forwarded(self):
        out = convert_responses_to_chat_completion(
            {"model": "m", "input": "q", "reasoning": {"effort": "high"}}
        )
        assert out["chat_template_kwargs"]["reasoning_effort"] == "high"

    def test_reasoning_effort_invalid_dropped(self):
        out = convert_responses_to_chat_completion(
            {"model": "m", "input": "q", "reasoning": {"effort": "extreme"}}
        )
        assert "chat_template_kwargs" not in out

    def test_max_tokens_and_sampling_params(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "max_output_tokens": 50,
                "temperature": 0.3,
                "top_p": 0.8,
            }
        )
        assert out["max_tokens"] == 50
        assert out["temperature"] == 0.3
        assert out["top_p"] == 0.8

    def test_stream_adds_usage_option(self):
        out = convert_responses_to_chat_completion(
            {"model": "m", "input": "q", "stream": True}
        )
        assert out["stream"] is True
        assert out["stream_options"] == {"include_usage": True}

    def test_unsupported_tool_types_dropped(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "tools": [
                    {"type": "web_search_preview"},
                    {"type": "function", "function": {"name": "f", "parameters": {}}},
                ],
            }
        )
        assert len(out["tools"]) == 1
        assert out["tools"][0]["function"]["name"] == "f"

    def test_reasoning_input_item_carried_forward(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": "q"},
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "thought A"}],
                        "content": [{"type": "reasoning_text", "text": " thought B"}],
                    },
                ],
            }
        )
        reasoning_msg = out["messages"][-1]
        assert reasoning_msg["role"] == "assistant"
        assert "thought A" in reasoning_msg["reasoning_content"]
        assert "thought B" in reasoning_msg["reasoning_content"]

    def test_reasoning_input_item_empty_dropped(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": "q"},
                    {"type": "reasoning", "summary": [], "content": []},
                ],
            }
        )
        assert len(out["messages"]) == 1

    def test_unknown_input_item_dropped(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": "q"},
                    {"type": "file_search_call", "id": "x"},
                ],
            }
        )
        assert len(out["messages"]) == 1

    def test_function_call_output_dict_stringified(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "c1",
                        "output": {"key": "value"},
                    }
                ],
            }
        )
        msg = out["messages"][0]
        assert msg["role"] == "tool"
        assert "value" in msg["content"]

    def test_function_call_non_string_arguments(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "f",
                        "arguments": {"a": 1},
                    }
                ],
            }
        )
        args = out["messages"][0]["tool_calls"][0]["function"]["arguments"]
        assert "a" in args

    def test_content_with_string_parts(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": ["hello", {"type": "input_text", "text": " world"}]},
                ],
            }
        )
        assert out["messages"][0]["content"] == "hello\n world"

    def test_image_with_file_id(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": [{"type": "input_image", "file_id": "file-1"}]},
                ],
            }
        )
        msg = out["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "image_url"
        assert msg["content"][0]["image_url"]["url"] == "file-1"

    def test_empty_content(self):
        out = convert_responses_to_chat_completion(
            {"model": "m", "input": [{"role": "user", "content": None}]}
        )
        assert out["messages"][0]["content"] == ""

    def test_tool_choice_invalid_string_dropped(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
                "tool_choice": "bogus",
            }
        )
        assert "tool_choice" not in out

    def test_stop_forwarded(self):
        out = convert_responses_to_chat_completion({"model": "m", "input": "q", "stop": ["END"]})
        assert out["stop"] == ["END"]

    def test_bool_temperature_ignored(self):
        out = convert_responses_to_chat_completion({"model": "m", "input": "q", "temperature": True})
        assert "temperature" not in out


# ── convert_chat_completion_to_responses ─────────────────────────────────────


class TestChatToResponsesConversion:
    def test_basic_text(self):
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        cc = ChatCompletion(
            id="cmpl-1",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        )
        out = convert_chat_completion_to_responses(cc, model="gpt-4o")
        assert out["object"] == "response"
        assert out["status"] == "completed"
        assert out["model"] == "gpt-4o"
        assert out["output"][0]["type"] == "message"
        assert out["output"][0]["content"][0]["text"] == "Hello!"
        assert out["usage"]["input_tokens"] == 5
        assert out["usage"]["output_tokens"] == 2
        assert out["usage"]["total_tokens"] == 7

    def test_reasoning_item(self):
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        cc = ChatCompletion(
            id="cmpl-1",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content="answer", reasoning="let me think"
                    ),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        out = convert_chat_completion_to_responses(cc, model="gpt-4o")
        types = [item["type"] for item in out["output"]]
        assert "reasoning" in types
        assert "message" in types
        # Reasoning comes first
        assert types[0] == "reasoning"

    def test_tool_call(self):
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        cc = ChatCompletion(
            id="cmpl-1",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        out = convert_chat_completion_to_responses(cc, model="gpt-4o")
        # With tool calls and no text, no message item should be emitted.
        assert all(item["type"] == "function_call" for item in out["output"])
        assert out["output"][0]["name"] == "f"
        assert out["output"][0]["call_id"] == "call_1"

    def test_finish_length_status(self):
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        cc = ChatCompletion(
            id="cmpl-1",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="..."),
                    finish_reason="length",
                )
            ],
            usage=None,
        )
        out = convert_chat_completion_to_responses(cc, model="gpt-4o")
        assert out["output"][0]["status"] == "incomplete"
        assert "usage" not in out

    def test_instructions_passed_through(self):
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        cc = ChatCompletion(
            id="cmpl-1",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        out = convert_chat_completion_to_responses(cc, model="gpt-4o", instructions="be nice")
        assert out["instructions"] == "be nice"

    def test_no_choices_returns_empty_output(self):
        from openai.types.chat import ChatCompletion

        cc = ChatCompletion.model_construct(
            id="cmpl-1",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o",
            choices=[],
            usage=None,
        )
        out = convert_chat_completion_to_responses(cc, model="gpt-4o")
        # No choices -> message is None, so a placeholder empty-text message is emitted.
        assert len(out["output"]) == 1
        assert out["output"][0]["type"] == "message"
        assert out["output"][0]["content"][0]["text"] == ""

    def test_tool_call_without_function_skipped(self):
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        msg = ChatCompletionMessage.model_construct(
            role="assistant", content=None,
            tool_calls=[{"id": "c1", "type": "function", "function": None}],
        )
        cc = ChatCompletion.model_construct(
            id="cmpl-1",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o",
            choices=[Choice.model_construct(index=0, message=msg, finish_reason="tool_calls")],
            usage=None,
        )
        out = convert_chat_completion_to_responses(cc, model="gpt-4o")
        # function=None means the tool call is skipped; only a placeholder message is emitted.
        assert all(item["type"] == "message" for item in out["output"])


# ── Streaming ────────────────────────────────────────────────────────────────


class TestStreamingConversion:
    @pytest.mark.asyncio
    async def test_text_stream_emits_full_sequence(self):
        async def gen():
            for line in [
                'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"id":"1","choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
                'data: {"id":"1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
                "data: [DONE]",
            ]:
                yield line

        events = []
        async for chunk in stream_responses_from_chat_completion(gen(), model="gpt-4o"):
            events.append(chunk)

        joined = "".join(events)
        assert "event: response.created" in joined
        assert "event: response.in_progress" in joined
        assert "event: response.output_item.added" in joined
        assert "event: response.output_text.delta" in joined
        assert "event: response.output_text.done" in joined
        assert "event: response.output_item.done" in joined
        assert "event: response.completed" in joined

    @pytest.mark.asyncio
    async def test_reasoning_stream(self):
        async def gen():
            for line in [
                'data: {"id":"1","choices":[{"delta":{"reasoning":"thinking..."}}]}',
                'data: {"id":"1","choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]:
                yield line

        joined = ""
        async for chunk in stream_responses_from_chat_completion(gen(), model="gpt-4o"):
            joined += chunk

        assert "event: response.reasoning_text.delta" in joined
        assert "event: response.output_text.delta" in joined

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        async def gen():
            for line in [
                'data: {"id":"1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":""}}]}}]}',
                'data: {"id":"1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":1}"}}]}}]}',
                'data: {"id":"1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]:
                yield line

        joined = ""
        async for chunk in stream_responses_from_chat_completion(gen(), model="gpt-4o"):
            joined += chunk

        assert "event: response.function_call_arguments.delta" in joined
        assert "event: response.function_call_arguments.done" in joined

    @pytest.mark.asyncio
    async def test_invalid_json_line_skipped(self):
        async def gen():
            for line in [
                "data: not-json",
                'data: {"id":"1","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]:
                yield line

        joined = ""
        async for chunk in stream_responses_from_chat_completion(gen(), model="gpt-4o"):
            joined += chunk
        assert "event: response.completed" in joined


# ── End-to-end route tests ──────────────────────────────────────────────────


class TestResponsesRoute:
    def test_non_streaming_text(self, client):
        upstream = _upstream_chat_completion(content="Hello!")
        mock_response = httpx.Response(
            status_code=200,
            json=upstream,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            response = client.post(
                "/v1/responses",
                json={"model": "gpt-4o", "input": "Hi"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "response"
        assert data["output"][0]["type"] == "message"
        assert data["output"][0]["content"][0]["text"] == "Hello!"

    def test_model_mapping_applied(self, client):
        # Build a settings with mapping
        settings_with_mapping = Settings(
            openai_api_key="test-key",
            openai_base_url="https://api.openai.com/v1",
            request_timeout=30.0,
            api_key=None,
            default_model="mapped-model",
        )
        app = create_app(settings_with_mapping)
        c = TestClient(app)

        captured = {}
        mock_response = httpx.Response(
            status_code=200,
            json=_upstream_chat_completion(),
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        class _CaptureClient(httpx.AsyncClient):
            async def post(self, url, headers=None, json=None, **kwargs):
                captured["json"] = json
                return mock_response

        with patch("httpx.AsyncClient", _CaptureClient):
            response = c.post(
                "/v1/responses",
                json={"model": "anything", "input": "Hi"},
            )
        assert response.status_code == 200
        assert captured["json"]["model"] == "mapped-model"

    def test_missing_model_returns_400(self, client):
        response = client.post("/v1/responses", json={"input": "Hi"})
        assert response.status_code == 400
        assert "error" in response.json()

    def test_invalid_json_returns_400(self, client):
        response = client.post(
            "/v1/responses",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_upstream_error_propagated(self, client):
        mock_response = httpx.Response(
            status_code=500,
            json={"error": {"message": "boom", "type": "server_error"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            response = client.post(
                "/v1/responses",
                json={"model": "gpt-4o", "input": "Hi"},
            )
        assert response.status_code == 500
        assert "error" in response.json()

    def test_streaming_response(self, client):
        async def mock_stream_responses(client_arg, url, headers, chat_params, model, instructions):
            yield "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-4o\",\"status\":\"in_progress\",\"output\":[],\"parallel_tool_calls\":true,\"tool_choice\":\"auto\",\"tools\":[],\"text\":{\"format\":{\"type\":\"text\"}}}}\n\n"
            yield "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"m\",\"output_index\":0,\"delta\":\"Hi\"}\n\n"
            yield "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-4o\",\"status\":\"completed\",\"output\":[],\"parallel_tool_calls\":true,\"tool_choice\":\"auto\",\"tools\":[],\"text\":{\"format\":{\"type\":\"text\"}}}}\n\n"

        with patch(
            "local_openai2anthropic.router._stream_responses",
            side_effect=mock_stream_responses,
        ):
            with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "gpt-4o", "input": "Hi", "stream": True},
            ) as response:
                assert response.status_code == 200
                collected = b""
                for chunk in response.iter_bytes():
                    collected += chunk

        text = collected.decode("utf-8")
        assert "event: response.created" in text
        assert "event: response.output_text.delta" in text
        assert "event: response.completed" in text

    def test_instructions_forwarded_to_response(self, client):
        upstream = _upstream_chat_completion(content="ok")
        mock_response = httpx.Response(
            status_code=200,
            json=upstream,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            response = client.post(
                "/v1/responses",
                json={"model": "gpt-4o", "input": "Hi", "instructions": "be nice"},
            )
        data = response.json()
        assert data["instructions"] == "be nice"
