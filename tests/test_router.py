"""
Integration tests for the FastAPI router.
"""

import pytest
from fastapi.testclient import TestClient

from local_openai2anthropic.config import Settings
from local_openai2anthropic.main import create_app


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        request_timeout=30.0,
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
    
    assert response.status_code == 422
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
    assert response.status_code == 422
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
        data="invalid json",
        headers={"Content-Type": "application/json"},
    )
    
    assert response.status_code == 422
    data = response.json()
    assert data["type"] == "error"
    assert "error" in data
    assert "type" in data["error"]
    assert "message" in data["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
