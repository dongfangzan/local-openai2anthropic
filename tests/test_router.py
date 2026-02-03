"""
Integration tests for the FastAPI router.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

from local_openai2anthropic.config import Settings
from local_openai2anthropic.main import create_app
from local_openai2anthropic.router import _convert_result_to_stream


@pytest.fixture
def settings(monkeypatch):
    """Create test settings."""
    # Ensure local developer env vars don't leak into tests.
    monkeypatch.delenv("OA2A_API_KEY", raising=False)
    monkeypatch.delenv("OA2A_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OA2A_TAVILY_API_KEY", raising=False)
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        request_timeout=30.0,
        api_key=None,
        tavily_api_key=None,
    )


@pytest.fixture
def client(settings):
    """Create test client."""
    app = create_app(settings)
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_docs_endpoint(client):
    """Test that docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_create_message_validation_error(client):
    """Test validation error handling."""
    # Missing required fields - should get validation error
    response = client.post("/v1/messages", json={})

    assert response.status_code == 400
    data = response.json()
    assert data["type"] == "error"
    assert "error" in data


def test_create_message_with_empty_model(client):
    """Test validation with empty model."""
    response = client.post(
        "/v1/messages",
        json={
            "model": "",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        },
    )

    # Should get validation error
    assert response.status_code == 400
    data = response.json()
    assert data["type"] == "error"


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options(
        "/v1/messages",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


def test_error_response_format(client):
    """Test that error responses follow Anthropic format."""
    # Send invalid JSON to trigger validation error
    response = client.post(
        "/v1/messages",
        content="invalid json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    data = response.json()
    assert data["type"] == "error"
    assert "error" in data
    assert "type" in data["error"]
    assert "message" in data["error"]


@pytest.mark.asyncio
async def test_stream_conversion_includes_web_search_blocks_and_usage():
    """Ensure streaming conversion keeps server tool blocks and usage counts.

    Claude Code's web search summary relies on seeing `server_tool_use` +
    `web_search_tool_result` blocks (and/or usage.server_tool_use).
    """
    result = JSONResponse(
        content={
            "id": "msg_test",
            "model": "test-model",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_test",
                    "name": "web_search",
                    "input": {"query": "成都明天天气 2026年1月30日"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_test",
                    "results": [
                        {
                            "type": "web_search_result",
                            "url": "https://example.com",
                            "title": "Example",
                            "page_age": None,
                            "encrypted_content": "abc",
                        }
                    ],
                },
                {"type": "text", "text": "ok"},
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
                "server_tool_use": {"web_search_requests": 1},
            },
        }
    )

    chunks: list[str] = []
    async for chunk in _convert_result_to_stream(result, "test-model"):
        chunks.append(chunk)

    stream_text = "".join(chunks)
    assert '"type": "server_tool_use"' in stream_text
    assert '"type": "web_search_tool_result"' in stream_text
    assert '"server_tool_use"' in stream_text
    assert '"web_search_requests": 1' in stream_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
