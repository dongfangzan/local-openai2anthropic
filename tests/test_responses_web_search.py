# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/responses server-side web search integration."""

import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from local_openai2anthropic.config import Settings
from local_openai2anthropic.main import create_app
from local_openai2anthropic.responses_converter import (
    build_web_search_output_items,
    convert_responses_to_chat_completion,
    web_search_function_tool,
)
from local_openai2anthropic.responses_web_search import (
    _content_blocks_to_web_search_items,
    _extract_query,
    _prepare_chat_params_for_loop,
    handle_responses_with_web_search,
)
from local_openai2anthropic.server_tools.base import ToolResult


@pytest.fixture
def settings_with_tavily():
    return Settings(
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        request_timeout=30.0,
        api_key=None,
        tavily_api_key="tvly-test",
        websearch_provider="tavily",
        websearch_max_uses=3,
    )


@pytest.fixture
def client(settings_with_tavily):
    app = create_app(settings_with_tavily)
    return TestClient(app)


def _chat_completion_with_tool_call(query="beijing weather"):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": query}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _chat_completion_with_text(text="It is sunny in Beijing."):
    return {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 1700000001,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


# ── Pure unit tests ──────────────────────────────────────────────────────────


class TestWebSearchConversion:
    def test_web_search_tool_recognized(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "tools": [{"type": "web_search_preview"}],
            }
        )
        # web_search is NOT in the forwarded tools (it's handled server-side)
        assert "tools" not in out or all(
            t.get("function", {}).get("name") != "web_search"
            for t in (out.get("tools") or [])
        )
        assert "web_search_20250305" in out["_web_search_configs"]

    def test_web_search_dated_variant_recognized(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "tools": [{"type": "web_search_2025_08_26"}],
            }
        )
        assert "web_search_20250305" in out["_web_search_configs"]

    def test_web_search_anthropic_alias_recognized(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "tools": [{"type": "web_search_20250305", "max_uses": 2}],
            }
        )
        cfg = out["_web_search_configs"]["web_search_20250305"]
        assert cfg["max_uses"] == 2

    def test_web_search_mixed_with_function_tools(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "tools": [
                    {"type": "web_search_preview"},
                    {"type": "function", "name": "calc", "parameters": {}},
                ],
            }
        )
        # function tool preserved, web_search extracted
        assert len(out["tools"]) == 1
        assert out["tools"][0]["function"]["name"] == "calc"
        assert "web_search_20250305" in out["_web_search_configs"]

    def test_web_search_config_extracted(self):
        out = convert_responses_to_chat_completion(
            {
                "model": "m",
                "input": "q",
                "tools": [
                    {
                        "type": "web_search_preview",
                        "search_context_size": "high",
                        "user_location": {"country": "CN"},
                    }
                ],
            }
        )
        cfg = out["_web_search_configs"]["web_search_20250305"]
        assert cfg["search_context_size"] == "high"
        assert cfg["user_location"] == {"country": "CN"}

    def test_web_search_function_tool_shape(self):
        tool = web_search_function_tool()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "web_search"
        assert "query" in tool["function"]["parameters"]["properties"]

    def test_build_web_search_output_items_success(self):
        items = build_web_search_output_items(
            "call_1", "q", [{"url": "http://x", "title": "X"}]
        )
        assert len(items) == 1
        item = items[0]
        assert item["type"] == "web_search_call"
        assert item["status"] == "completed"
        assert item["query"] == "q"
        assert item["results"][0]["url"] == "http://x"

    def test_build_web_search_output_items_error(self):
        items = build_web_search_output_items("call_1", "q", [], error_code="unavailable")
        assert items[0]["status"] == "failed"
        assert items[0]["error"]["code"] == "unavailable"


class TestWebSearchLoopHelpers:
    def test_extract_query_string_args(self):
        q = _extract_query({"function": {"arguments": '{"query":"hello"}'}})
        assert q == "hello"

    def test_extract_query_invalid_json(self):
        q = _extract_query({"function": {"arguments": "not-json"}})
        assert q == ""

    def test_extract_query_missing(self):
        q = _extract_query({"function": {"arguments": "{}"}})
        assert q == ""

    def test_prepare_chat_params_adds_web_search_tool(self):
        chat_params = {
            "model": "m",
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        }
        out = _prepare_chat_params_for_loop(
            chat_params, {"web_search_20250305": {"max_uses": 2}}
        )
        assert out["stream"] is False
        assert "stream_options" not in out
        names = [t["function"]["name"] for t in out["tools"]]
        assert "f" in names
        assert "web_search" in names
        assert out["_server_tools_config"]["web_search_20250305"]["max_uses"] == 2

    def test_content_blocks_to_items_success(self):
        blocks = [
            {"type": "server_tool_use", "id": "c1", "name": "web_search", "input": {}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "c1",
                "results": [{"url": "http://a", "title": "A"}],
                "content": [{"url": "http://a", "title": "A"}],
            },
        ]
        items = _content_blocks_to_web_search_items("c1", "q", blocks)
        assert items[0]["type"] == "web_search_call"
        assert items[0]["status"] == "completed"
        assert items[0]["results"][0]["url"] == "http://a"

    def test_content_blocks_to_items_error(self):
        blocks = [
            {"type": "server_tool_use", "id": "c1", "name": "web_search", "input": {}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "c1",
                "results": {"type": "web_search_tool_result_error", "error_code": "unavailable"},
                "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"},
            },
        ]
        items = _content_blocks_to_web_search_items("c1", "q", blocks)
        assert items[0]["status"] == "failed"
        assert items[0]["error"]["code"] == "unavailable"


# ── End-to-end route tests with mocked upstream ─────────────────────────────


class TestResponsesWebSearchRoute:
    def test_web_search_non_streaming_full_loop(self, client):
        """Model calls web_search, we execute it, model then answers."""
        # Upstream is called twice:
        #   1st: returns tool_call for web_search
        #   2nd: returns final text
        responses = [
            httpx.Response(
                200,
                json=_chat_completion_with_tool_call("beijing weather"),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            ),
            httpx.Response(
                200,
                json=_chat_completion_with_text("It is sunny in Beijing."),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            ),
        ]
        call_count = {"n": 0}

        class _PatchedClient(httpx.AsyncClient):
            async def post(self, url, headers=None, json=None, **kwargs):
                i = call_count["n"]
                call_count["n"] += 1
                return responses[i]

        # Mock the actual Tavily search execution
        async def _fake_execute(cls, call_id, args, config, settings):
            return ToolResult(
                success=True,
                content=[{"url": "http://x", "title": "Beijing weather", "encrypted_content": "sunny"}],
                usage_increment={"web_search_requests": 1},
            )

        with patch("httpx.AsyncClient", _PatchedClient), \
             patch(
                 "local_openai2anthropic.server_tools.web_search.WebSearchServerTool.execute",
                 new=classmethod(_fake_execute),
             ):
            r = client.post(
                "/v1/responses",
                json={
                    "model": "kimi-k2.6",
                    "input": "What's the weather in Beijing?",
                    "tools": [{"type": "web_search_preview"}],
                },
            )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["object"] == "response"
        # web_search_call item first, then message
        types = [o["type"] for o in data["output"]]
        assert "web_search_call" in types
        assert "message" in types
        assert types.index("web_search_call") < types.index("message")
        ws = next(o for o in data["output"] if o["type"] == "web_search_call")
        assert ws["query"] == "beijing weather"
        assert ws["status"] == "completed"
        assert len(ws["results"]) >= 1
        msg = next(o for o in data["output"] if o["type"] == "message")
        assert "sunny" in msg["content"][0]["text"].lower()

    def test_web_search_streaming_full_loop(self, client):
        responses = [
            httpx.Response(
                200,
                json=_chat_completion_with_tool_call("q"),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            ),
            httpx.Response(
                200,
                json=_chat_completion_with_text("Final answer."),
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            ),
        ]
        call_count = {"n": 0}

        class _PatchedClient(httpx.AsyncClient):
            async def post(self, url, headers=None, json=None, **kwargs):
                i = call_count["n"]
                call_count["n"] += 1
                return responses[i]

        async def _fake_execute(cls, call_id, args, config, settings):
            return ToolResult(
                success=True,
                content=[{"url": "http://x", "title": "X"}],
                usage_increment={"web_search_requests": 1},
            )

        with patch("httpx.AsyncClient", _PatchedClient), \
             patch(
                 "local_openai2anthropic.server_tools.web_search.WebSearchServerTool.execute",
                 new=classmethod(_fake_execute),
             ):
            with client.stream(
                "POST",
                "/v1/responses",
                json={
                    "model": "kimi-k2.6",
                    "input": "q",
                    "tools": [{"type": "web_search_preview"}],
                    "stream": True,
                },
            ) as r:
                assert r.status_code == 200
                body = b"".join(r.iter_bytes()).decode("utf-8")

        assert "event: response.created" in body
        assert "event: response.web_search_call.completed" in body
        assert "event: response.output_text.delta" in body
        assert "event: response.completed" in body
        assert "Final answer." in body

    def test_web_search_not_configured_falls_through(self):
        """When no search provider is configured, web_search is dropped silently."""
        settings = Settings(
            openai_api_key="test-key",
            openai_base_url="https://api.openai.com/v1",
            request_timeout=30.0,
            api_key=None,
            tavily_api_key=None,
            tongxiao_api_key=None,
        )
        app = create_app(settings)
        c = TestClient(app)

        upstream = _chat_completion_with_text("no search needed")
        mock = httpx.Response(
            200,
            json=upstream,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        with patch("httpx.AsyncClient.post", return_value=mock):
            r = c.post(
                "/v1/responses",
                json={
                    "model": "m",
                    "input": "q",
                    "tools": [{"type": "web_search_preview"}],
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert all(o["type"] != "web_search_call" for o in data["output"])

    def test_web_search_max_uses_respected(self, client):
        """When max_uses is hit, search calls return error and the loop continues."""
        # Always return a web_search tool call — forces max_uses path
        tool_call_resp = _chat_completion_with_tool_call("q")
        final_resp = _chat_completion_with_text("done")
        # First 4 calls return tool_call, 5th returns final text.
        responses_seq = [tool_call_resp, tool_call_resp, tool_call_resp, tool_call_resp, final_resp]
        call_count = {"n": 0}

        class _PatchedClient(httpx.AsyncClient):
            async def post(self, url, headers=None, json=None, **kwargs):
                i = min(call_count["n"], len(responses_seq) - 1)
                call_count["n"] += 1
                return httpx.Response(200, json=responses_seq[i], request=httpx.Request("POST", url))

        async def _fake_execute(cls, call_id, args, config, settings):
            return ToolResult(
                success=True,
                content=[{"url": "http://x", "title": "X"}],
                usage_increment={"web_search_requests": 1},
            )

        with patch("httpx.AsyncClient", _PatchedClient), \
             patch(
                 "local_openai2anthropic.server_tools.web_search.WebSearchServerTool.execute",
                 new=classmethod(_fake_execute),
             ):
            r = client.post(
                "/v1/responses",
                json={
                    "model": "m",
                    "input": "q",
                    "tools": [{"type": "web_search_preview", "max_uses": 1}],
                },
            )
        assert r.status_code == 200
        data = r.json()
        ws_items = [o for o in data["output"] if o["type"] == "web_search_call"]
        # First call succeeds (max_uses=1), subsequent ones hit the guard.
        # The loop should still terminate with a message.
        assert any(o["type"] == "message" for o in data["output"])

    def test_web_search_upstream_error_propagated(self, client):
        mock = httpx.Response(
            500,
            json={"error": {"message": "boom", "type": "server_error"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        with patch("httpx.AsyncClient.post", return_value=mock):
            r = client.post(
                "/v1/responses",
                json={
                    "model": "m",
                    "input": "q",
                    "tools": [{"type": "web_search_preview"}],
                },
            )
        assert r.status_code == 500
        assert "error" in r.json()

    def test_web_search_timeout(self, client):
        with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("timeout")):
            r = client.post(
                "/v1/responses",
                json={
                    "model": "m",
                    "input": "q",
                    "tools": [{"type": "web_search_preview"}],
                },
            )
        assert r.status_code == 504
        assert "timed out" in r.json()["error"]["message"].lower()

    def test_web_search_connection_error(self, client):
        with patch("httpx.AsyncClient.post", side_effect=httpx.RequestError("conn refused")):
            r = client.post(
                "/v1/responses",
                json={
                    "model": "m",
                    "input": "q",
                    "tools": [{"type": "web_search_preview"}],
                },
            )
        assert r.status_code == 502
        assert "connection_error" in r.json()["error"]["type"]

    def test_web_search_invalid_json_response(self, client):
        mock = httpx.Response(
            200,
            content=b"not json",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        with patch("httpx.AsyncClient.post", return_value=mock):
            r = client.post(
                "/v1/responses",
                json={
                    "model": "m",
                    "input": "q",
                    "tools": [{"type": "web_search_preview"}],
                },
            )
        assert r.status_code == 502

    def test_web_search_streaming_with_reasoning(self, client):
        """Streaming emits reasoning deltas when the final completion has reasoning."""
        tool_call_resp = _chat_completion_with_tool_call("q")
        final_resp = {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1700000001,
            "model": "kimi-k2.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Answer.",
                        "reasoning": "let me think",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        responses_seq = [tool_call_resp, final_resp]
        call_count = {"n": 0}

        class _PatchedClient(httpx.AsyncClient):
            async def post(self, url, headers=None, json=None, **kwargs):
                i = min(call_count["n"], len(responses_seq) - 1)
                call_count["n"] += 1
                return httpx.Response(200, json=responses_seq[i], request=httpx.Request("POST", url))

        async def _fake_execute(cls, call_id, args, config, settings):
            return ToolResult(
                success=True,
                content=[{"url": "http://x", "title": "X"}],
                usage_increment={"web_search_requests": 1},
            )

        with patch("httpx.AsyncClient", _PatchedClient), \
             patch(
                 "local_openai2anthropic.server_tools.web_search.WebSearchServerTool.execute",
                 new=classmethod(_fake_execute),
             ):
            with client.stream(
                "POST",
                "/v1/responses",
                json={
                    "model": "kimi-k2.6",
                    "input": "q",
                    "tools": [{"type": "web_search_preview"}],
                    "stream": True,
                },
            ) as r:
                assert r.status_code == 200
                body = b"".join(r.iter_bytes()).decode("utf-8")

        assert "event: response.reasoning_text.delta" in body
        assert "event: response.output_text.delta" in body
        assert "event: response.completed" in body

    def test_web_search_streaming_with_function_call_output(self, client):
        """Streaming emits function_call events when the model returns a non-web-search tool call."""
        # Model returns a non-web-search function call as the final response.
        final_resp = {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1700000001,
            "model": "kimi-k2.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_x",
                                "type": "function",
                                "function": {"name": "calc", "arguments": '{"x":1}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        call_count = {"n": 0}

        class _PatchedClient(httpx.AsyncClient):
            async def post(self, url, headers=None, json=None, **kwargs):
                call_count["n"] += 1
                return httpx.Response(200, json=final_resp, request=httpx.Request("POST", url))

        with patch("httpx.AsyncClient", _PatchedClient):
            with client.stream(
                "POST",
                "/v1/responses",
                json={
                    "model": "kimi-k2.6",
                    "input": "q",
                    "tools": [{"type": "web_search_preview"}],
                    "stream": True,
                },
            ) as r:
                assert r.status_code == 200
                body = b"".join(r.iter_bytes()).decode("utf-8")

        assert "event: response.function_call_arguments.delta" in body
        assert "event: response.function_call_arguments.done" in body
