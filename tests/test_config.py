"""
Tests for the config module.
"""

import pytest
from local_openai2anthropic.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_settings_initialization(self):
        """Test that Settings can be initialized with explicit values."""
        settings = Settings(
            openai_api_key="test-key",
            openai_base_url="https://custom.api.com",
            openai_org_id="org-123",
            openai_project_id="proj-456",
            host="127.0.0.1",
            port=9000,
            request_timeout=60.0,
            api_key="server-api-key",
            tavily_api_key="tavily-key",
            tavily_timeout=15.0,
            tavily_max_results=10,
            websearch_max_uses=3,
            log_level="INFO",
            cors_origins=["https://example.com"],
            cors_credentials=False,
            cors_methods=["GET", "POST"],
            cors_headers=["Authorization", "Content-Type"],
        )

        assert settings.openai_api_key == "test-key"
        assert settings.openai_base_url == "https://custom.api.com"
        assert settings.openai_org_id == "org-123"
        assert settings.openai_project_id == "proj-456"
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.request_timeout == 60.0
        assert settings.api_key == "server-api-key"
        assert settings.tavily_api_key == "tavily-key"
        assert settings.tavily_timeout == 15.0
        assert settings.tavily_max_results == 10
        assert settings.websearch_max_uses == 3
        assert settings.log_level == "INFO"
        assert settings.cors_origins == ["https://example.com"]
        assert settings.cors_credentials is False
        assert settings.cors_methods == ["GET", "POST"]
        assert settings.cors_headers == ["Authorization", "Content-Type"]

    def test_openai_auth_headers_basic(self):
        """Test OpenAI auth headers with just API key."""
        settings = Settings(
            openai_api_key="test-key",
        )

        headers = settings.openai_auth_headers

        assert headers == {"Authorization": "Bearer test-key"}

    def test_openai_auth_headers_with_org(self):
        """Test OpenAI auth headers with org ID."""
        settings = Settings(
            openai_api_key="test-key",
            openai_org_id="org-123",
        )

        headers = settings.openai_auth_headers

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["OpenAI-Organization"] == "org-123"

    def test_openai_auth_headers_with_project(self):
        """Test OpenAI auth headers with project ID."""
        settings = Settings(
            openai_api_key="test-key",
            openai_project_id="proj-456",
        )

        headers = settings.openai_auth_headers

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["OpenAI-Project"] == "proj-456"

    def test_openai_auth_headers_full(self):
        """Test OpenAI auth headers with all options."""
        settings = Settings(
            openai_api_key="test-key",
            openai_org_id="org-123",
            openai_project_id="proj-456",
        )

        headers = settings.openai_auth_headers

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["OpenAI-Organization"] == "org-123"
        assert headers["OpenAI-Project"] == "proj-456"

    def test_openai_auth_headers_no_api_key(self):
        """Test OpenAI auth headers without API key."""
        settings = Settings()

        headers = settings.openai_auth_headers

        # Should still work but with None in the f-string
        assert "Authorization" in headers
        assert "Bearer" in headers["Authorization"]

    def test_optional_api_keys(self):
        """Test that optional API keys can be set or None."""
        # With explicit values
        settings = Settings(
            openai_api_key="test-key",
            api_key="server-key",
            tavily_api_key="tavily-key",
        )

        assert settings.openai_api_key == "test-key"
        assert settings.api_key == "server-key"
        assert settings.tavily_api_key == "tavily-key"

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored."""
        # This should not raise an error due to extra="ignore"
        settings = Settings(
            openai_api_key="test-key",
            custom_field="ignored",  # type: ignore[call-arg]
            another_extra=123,  # type: ignore[call-arg]
        )

        assert settings.openai_api_key == "test-key"

    def test_cors_settings_defaults(self):
        """Test default CORS settings."""
        settings = Settings()

        assert settings.cors_origins == ["*"]
        assert settings.cors_credentials is True
        assert settings.cors_methods == ["*"]
        assert settings.cors_headers == ["*"]

    def test_tavily_settings(self):
        """Test Tavily-specific settings."""
        settings = Settings(
            tavily_api_key="tvly-test",
            tavily_timeout=45.0,
            tavily_max_results=10,
            websearch_max_uses=8,
        )

        assert settings.tavily_api_key == "tvly-test"
        assert settings.tavily_timeout == 45.0
        assert settings.tavily_max_results == 10
        assert settings.websearch_max_uses == 8


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()

        assert isinstance(settings, Settings)

    def test_caching(self):
        """Test that get_settings caches the result."""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same object due to @lru_cache
        assert settings1 is settings2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
