# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for server startup and basic connectivity.
"""

import httpx
import pytest
from fastapi import FastAPI

from local_openai2anthropic.config import Settings


@pytest.mark.asyncio
class TestServerStartup:
    """Test server startup and basic connectivity."""

    async def test_health_check(self, client: httpx.AsyncClient) -> None:
        """Test health check endpoint returns healthy status."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data == {"status": "healthy"}

    async def test_docs_endpoint(self, client: httpx.AsyncClient) -> None:
        """Test OpenAPI docs endpoint is accessible."""
        response = await client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    async def test_openapi_schema(self, client: httpx.AsyncClient) -> None:
        """Test OpenAPI schema endpoint returns valid JSON."""
        response = await client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "/v1/messages" in data["paths"]
        assert "/health" in data["paths"]

    async def test_redoc_endpoint(self, client: httpx.AsyncClient) -> None:
        """Test ReDoc endpoint is accessible."""
        response = await client.get("/redoc")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
class TestCORS:
    """Test CORS configuration."""

    async def test_cors_headers_on_get(self, client: httpx.AsyncClient) -> None:
        """Test CORS headers are present on GET requests."""
        response = await client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"

    async def test_cors_preflight_request(self, client: httpx.AsyncClient) -> None:
        """Test CORS preflight (OPTIONS) request."""
        response = await client.options(
            "/v1/messages",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,Authorization",
            },
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    async def test_cors_headers_on_post(self, client: httpx.AsyncClient) -> None:
        """Test CORS headers are present on POST requests."""
        response = await client.post(
            "/v1/messages",
            headers={
                "Origin": "http://localhost:3000",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # We expect a validation error (400) but CORS headers should still be present
        assert "access-control-allow-origin" in response.headers


@pytest.mark.asyncio
class TestServerConfiguration:
    """Test server configuration is properly applied."""

    async def test_app_version(self, app: FastAPI) -> None:
        """Test app version is set correctly."""
        assert app.version == "0.3.8"

    async def test_app_title(self, app: FastAPI) -> None:
        """Test app title is set correctly."""
        assert app.title == "local-openai2anthropic"

    async def test_settings_in_app_state(
        self, app: FastAPI, test_settings: Settings
    ) -> None:
        """Test settings are stored in app state."""
        assert hasattr(app.state, "settings")
        assert app.state.settings.openai_api_key == test_settings.openai_api_key
        assert app.state.settings.openai_base_url == test_settings.openai_base_url
