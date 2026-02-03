"""
Tests for the protocol module.
"""

import pytest
from local_openai2anthropic.protocol import (
    UsageWithCache,
    AnthropicError,
    AnthropicErrorResponse,
    PingEvent,
    ApproximateLocation,
    WebSearchToolDefinition,
    ServerToolUseBlock,
    WebSearchResult,
    WebSearchToolResult,
    WebSearchToolResultError,
    WebSearchCitation,
    ServerToolUseUsage,
    UsageWithServerToolUse,
)


class TestUsageWithCache:
    """Tests for UsageWithCache model."""

    def test_basic_usage(self):
        """Test basic usage creation."""
        usage = UsageWithCache(
            input_tokens=100,
            output_tokens=50,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    def test_usage_with_cache(self):
        """Test usage with cache tokens."""
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

    def test_usage_serialization(self):
        """Test usage JSON serialization."""
        usage = UsageWithCache(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=200,
        )

        data = usage.model_dump()
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["cache_creation_input_tokens"] == 200
        assert data["cache_read_input_tokens"] is None


class TestAnthropicError:
    """Tests for AnthropicError model."""

    def test_error_creation(self):
        """Test error creation."""
        error = AnthropicError(
            type="invalid_request_error",
            message="Invalid request",
        )

        assert error.type == "invalid_request_error"
        assert error.message == "Invalid request"

    def test_error_types(self):
        """Test different error types."""
        error_types = [
            "invalid_request_error",
            "authentication_error",
            "api_error",
            "timeout_error",
            "connection_error",
            "internal_error",
        ]

        for error_type in error_types:
            error = AnthropicError(
                type=error_type,
                message=f"Test {error_type}",
            )
            assert error.type == error_type


class TestAnthropicErrorResponse:
    """Tests for AnthropicErrorResponse model."""

    def test_error_response_creation(self):
        """Test error response creation."""
        error = AnthropicError(
            type="invalid_request_error",
            message="Invalid request",
        )
        response = AnthropicErrorResponse(error=error)

        assert response.type == "error"
        assert response.error.type == "invalid_request_error"
        assert response.error.message == "Invalid request"

    def test_error_response_serialization(self):
        """Test error response serialization."""
        error = AnthropicError(
            type="api_error",
            message="API error occurred",
        )
        response = AnthropicErrorResponse(error=error)

        data = response.model_dump()
        assert data["type"] == "error"
        assert data["error"]["type"] == "api_error"
        assert data["error"]["message"] == "API error occurred"


class TestPingEvent:
    """Tests for PingEvent model."""

    def test_default_ping(self):
        """Test default ping event."""
        ping = PingEvent()

        assert ping.type == "ping"

    def test_explicit_ping(self):
        """Test explicit ping type."""
        ping = PingEvent(type="ping")

        assert ping.type == "ping"


class TestApproximateLocation:
    """Tests for ApproximateLocation model."""

    def test_default_location(self):
        """Test default location."""
        location = ApproximateLocation()

        assert location.type == "approximate"
        assert location.country == "US"
        assert location.city is None
        assert location.region is None
        assert location.timezone is None

    def test_full_location(self):
        """Test full location specification."""
        location = ApproximateLocation(
            city="San Francisco",
            region="California",
            country="US",
            timezone="America/Los_Angeles",
        )

        assert location.type == "approximate"
        assert location.city == "San Francisco"
        assert location.region == "California"
        assert location.country == "US"
        assert location.timezone == "America/Los_Angeles"


class TestWebSearchToolDefinition:
    """Tests for WebSearchToolDefinition model."""

    def test_default_definition(self):
        """Test default tool definition."""
        tool = WebSearchToolDefinition()

        assert tool.type == "web_search_20250305"
        assert tool.name == "web_search"
        assert tool.max_uses is None
        assert tool.allowed_domains is None
        assert tool.blocked_domains is None
        assert tool.user_location is None

    def test_full_definition(self):
        """Test full tool definition."""
        location = ApproximateLocation(city="Tokyo", country="JP")
        tool = WebSearchToolDefinition(
            max_uses=5,
            allowed_domains=["example.com", "test.com"],
            blocked_domains=["blocked.com"],
            user_location=location,
        )

        assert tool.type == "web_search_20250305"
        assert tool.name == "web_search"
        assert tool.max_uses == 5
        assert tool.allowed_domains == ["example.com", "test.com"]
        assert tool.blocked_domains == ["blocked.com"]
        assert tool.user_location is not None
        assert tool.user_location.city == "Tokyo"


class TestServerToolUseBlock:
    """Tests for ServerToolUseBlock model."""

    def test_tool_use_creation(self):
        """Test tool use block creation."""
        block = ServerToolUseBlock(
            id="tool_123",
            name="web_search",
            input={"query": "test query"},
        )

        assert block.type == "server_tool_use"
        assert block.id == "tool_123"
        assert block.name == "web_search"
        assert block.input == {"query": "test query"}


class TestWebSearchResult:
    """Tests for WebSearchResult model."""

    def test_basic_result(self):
        """Test basic search result."""
        result = WebSearchResult(
            url="https://example.com",
            title="Example Title",
        )

        assert result.type == "web_search_result"
        assert result.url == "https://example.com"
        assert result.title == "Example Title"
        assert result.page_age is None
        assert result.encrypted_content is None

    def test_full_result(self):
        """Test full search result."""
        result = WebSearchResult(
            url="https://example.com",
            title="Example Title",
            page_age="2024-01-01",
            encrypted_content="encrypted_data",
        )

        assert result.url == "https://example.com"
        assert result.title == "Example Title"
        assert result.page_age == "2024-01-01"
        assert result.encrypted_content == "encrypted_data"


class TestWebSearchToolResultError:
    """Tests for WebSearchToolResultError model."""

    def test_error_types(self):
        """Test different error types."""
        error_codes = [
            "invalid_input",
            "max_uses_exceeded",
            "query_too_long",
            "too_many_requests",
            "unavailable",
        ]

        for code in error_codes:
            error = WebSearchToolResultError(error_code=code)  # type: ignore[arg-type]
            assert error.type == "web_search_tool_result_error"
            assert error.error_code == code


class TestWebSearchToolResult:
    """Tests for WebSearchToolResult model."""

    def test_success_result(self):
        """Test successful tool result."""
        results = [
            WebSearchResult(url="https://example.com", title="Example"),
        ]
        result = WebSearchToolResult(
            tool_use_id="tool_123",
            results=results,
        )

        assert result.type == "web_search_tool_result"
        assert result.tool_use_id == "tool_123"
        assert isinstance(result.results, list)
        assert len(result.results) == 1
        assert result.results[0].url == "https://example.com"

    def test_error_result(self):
        """Test error tool result."""
        error = WebSearchToolResultError(error_code="max_uses_exceeded")  # type: ignore[arg-type]
        result = WebSearchToolResult(
            tool_use_id="tool_123",
            results=error,
        )

        assert result.type == "web_search_tool_result"
        assert result.tool_use_id == "tool_123"


class TestWebSearchCitation:
    """Tests for WebSearchCitation model."""

    def test_basic_citation(self):
        """Test basic citation."""
        citation = WebSearchCitation(
            url="https://example.com",
            title="Example Title",
        )

        assert citation.type == "web_search_result_location"
        assert citation.url == "https://example.com"
        assert citation.title == "Example Title"
        assert citation.page_age is None

    def test_full_citation(self):
        """Test full citation."""
        citation = WebSearchCitation(
            url="https://example.com",
            title="Example Title",
            page_age="2024-01-01",
        )

        assert citation.page_age == "2024-01-01"


class TestServerToolUseUsage:
    """Tests for ServerToolUseUsage model."""

    def test_default_usage(self):
        """Test default usage."""
        usage = ServerToolUseUsage()

        assert usage.web_search_requests == 0

    def test_custom_usage(self):
        """Test custom usage."""
        usage = ServerToolUseUsage(web_search_requests=5)

        assert usage.web_search_requests == 5


class TestUsageWithServerToolUse:
    """Tests for UsageWithServerToolUse model."""

    def test_basic_usage(self):
        """Test basic usage."""
        usage = UsageWithServerToolUse(
            input_tokens=100,
            output_tokens=50,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None
        assert usage.server_tool_use is None

    def test_usage_with_server_tools(self):
        """Test usage with server tool tracking."""
        tool_usage = ServerToolUseUsage(web_search_requests=3)
        usage = UsageWithServerToolUse(
            input_tokens=100,
            output_tokens=50,
            server_tool_use=tool_usage,
        )

        assert usage.server_tool_use is not None
        assert usage.server_tool_use.web_search_requests == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
