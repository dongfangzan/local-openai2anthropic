"""
Tests for the OpenAI-native passthrough routes.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from local_openai2anthropic.config import Settings
from local_openai2anthropic.main import create_app
from local_openai2anthropic.router import _openai_stream_proxy


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


class TestOpenAIChatCompletionsPassthrough:
    def test_non_streaming_passthrough(self, client):
        upstream_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response = httpx.Response(
            status_code=200,
            json=upstream_response,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-123"
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    def test_passthrough_upstream_error(self, client):
        mock_response = httpx.Response(
            status_code=500,
            json={"error": {"message": "Internal server error", "type": "server_error"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data

    def test_passthrough_upstream_empty_response(self, client):
        mock_response = httpx.Response(
            status_code=200,
            content=b"",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 502
        data = response.json()
        assert data["error"]["type"] == "api_error"

    def test_passthrough_upstream_invalid_json(self, client):
        mock_response = httpx.Response(
            status_code=200,
            content=b"not valid json",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 502
        data = response.json()
        assert data["error"]["type"] == "api_error"

    def test_passthrough_timeout(self, client):
        with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("timeout")):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 504
        data = response.json()
        assert data["error"]["type"] == "timeout_error"

    def test_passthrough_connection_error(self, client):
        with patch("httpx.AsyncClient.post", side_effect=httpx.RequestError("connection refused")):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 502
        data = response.json()
        assert data["error"]["type"] == "connection_error"

    def test_passthrough_invalid_json(self, client):
        response = client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_passthrough_preserves_headers_and_body(self, client):
        upstream_response = {
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "kimi-k2.5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response from mapped model"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_post = None

        async def mock_post_side_effect(url, headers=None, json=None, **kwargs):
            nonlocal mock_post
            mock_post = {"url": url, "headers": headers, "json": json}
            return httpx.Response(
                status_code=200,
                json=upstream_response,
                request=httpx.Request("POST", str(url)),
            )

        with patch("httpx.AsyncClient.post", side_effect=mock_post_side_effect):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-789"
        assert data["model"] == "kimi-k2.5"
        assert mock_post is not None
        assert mock_post["url"].endswith("/chat/completions")
        assert mock_post["headers"]["Authorization"] == "Bearer test-key"

    def test_streaming_passthrough_happy_path(self, client):
        async def mock_stream_proxy(client_arg, url, headers, body_json):
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
            yield "data: [DONE]\n\n"

        with patch("local_openai2anthropic.router._openai_stream_proxy", side_effect=mock_stream_proxy):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
            )
        assert response.status_code == 200
        content = response.text
        assert "chat.completion.chunk" in content
        assert "Hello" in content
        assert "[DONE]" in content

    def test_streaming_passthrough_error(self, client):
        async def mock_stream_proxy_error(client_arg, url, headers, body_json):
            yield 'data: {"error": {"message": "Upstream error (500)", "type": "api_error"}}\n\n'
            yield "data: [DONE]\n\n"

        with patch("local_openai2anthropic.router._openai_stream_proxy", side_effect=mock_stream_proxy_error):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
            )
        assert response.status_code == 200
        content = response.text
        assert '"error"' in content
        assert "[DONE]" in content


@pytest.mark.anyio
class TestOpenAIStreamProxy:
    async def test_success_path(self):
        chunks = ['data: {"id":"1","choices":[{"delta":{"content":"Hi"}}]}\n']

        class MockStreamResponse:
            status_code = 200

            async def aiter_lines(self):
                for c in chunks:
                    yield c

            async def aread(self):
                return b""

        class MockStreamContext:
            async def __aenter__(self):
                return MockStreamResponse()

            async def __aexit__(self, *args):
                pass

        mock_client = type("MockClient", (), {"stream": lambda self, *a, **kw: MockStreamContext()})()

        results = []
        async for item in _openai_stream_proxy(mock_client, "http://test/url", {}, {}):
            results.append(item)

        assert len(results) > 0
        assert 'data: {"id":"1"' in results[0]

    async def test_upstream_error_status(self):
        class MockErrorResponse:
            status_code = 500

            async def aread(self):
                return b"Internal Error"

        class MockStreamContext:
            async def __aenter__(self):
                return MockErrorResponse()

            async def __aexit__(self, *args):
                pass

        mock_client = type("MockClient", (), {"stream": lambda self, *a, **kw: MockStreamContext()})()

        results = []
        async for item in _openai_stream_proxy(mock_client, "http://test/url", {}, {}):
            results.append(item)

        assert len(results) == 2
        assert '"error"' in results[0]
        assert "api_error" in results[0]
        assert "[DONE]" in results[1]

    async def test_exception_during_stream(self):
        class FailingStreamContext:
            async def __aenter__(self):
                raise RuntimeError("stream crash")

            async def __aexit__(self, *args):
                pass

        mock_client = type("MockClient", (), {"stream": lambda self, *a, **kw: FailingStreamContext()})()

        results = []
        async for item in _openai_stream_proxy(mock_client, "http://test/url", {}, {}):
            results.append(item)

        assert len(results) == 2
        assert '"error"' in results[0]
        assert "stream crash" in results[0]
        assert "[DONE]" in results[1]

    async def test_non_data_lines_are_passthrough(self):
        chunks = [
            'data: {"id":"1","choices":[{"delta":{"content":"A"}}]}\n',
            ": keepalive\n",
            "\n",
        ]

        class MockStreamResponse:
            status_code = 200

            async def aiter_lines(self):
                for c in chunks:
                    yield c

            async def aread(self):
                return b""

        class MockStreamContext:
            async def __aenter__(self):
                return MockStreamResponse()

            async def __aexit__(self, *args):
                pass

        mock_client = type("MockClient", (), {"stream": lambda self, *a, **kw: MockStreamContext()})()

        results = []
        async for item in _openai_stream_proxy(mock_client, "http://test/url", {}, {}):
            results.append(item)

        assert any("data:" in r for r in results)
        assert any("keepalive" in r for r in results)
