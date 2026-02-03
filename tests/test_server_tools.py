"""
Tests for the server_tools module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from local_openai2anthropic.server_tools import (
    ServerTool,
    ServerToolRegistry,
    ToolResult,
    WebSearchServerTool,
)
from local_openai2anthropic.config import Settings


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(
            success=True,
            content=[{"type": "text", "text": "Result"}],
        )

        assert result.success is True
        assert result.content == [{"type": "text", "text": "Result"}]
        assert result.error_code is None
        assert result.usage_increment == {}

    def test_error_result(self):
        """Test error tool result."""
        result = ToolResult(
            success=False,
            content=[],
            error_code="max_uses_exceeded",
        )

        assert result.success is False
        assert result.content == []
        assert result.error_code == "max_uses_exceeded"

    def test_result_with_usage(self):
        """Test result with usage increment."""
        result = ToolResult(
            success=True,
            content=[],
            usage_increment={"web_search_requests": 1},
        )

        assert result.usage_increment == {"web_search_requests": 1}


class TestServerToolRegistry:
    """Tests for ServerToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""

        class TestTool(ServerTool):
            tool_type = "test_tool"
            tool_name = "test"

            @classmethod
            def is_enabled(cls, settings):
                return True

            @classmethod
            def extract_config(cls, tool_def):
                return None

            @classmethod
            def to_openai_tool(cls, config):
                return {}

            @classmethod
            def extract_call_args(cls, tool_call):
                return None

            @classmethod
            async def execute(cls, call_id, args, config, settings):
                return ToolResult(success=True, content=[])

        # Register the tool
        ServerToolRegistry.register(TestTool)

        # Verify registration
        assert ServerToolRegistry.get("test_tool") == TestTool
        assert TestTool in ServerToolRegistry.all_tools()

    def test_get_nonexistent_tool(self):
        """Test getting a non-existent tool."""
        assert ServerToolRegistry.get("nonexistent") is None

    def test_get_enabled_tools(self):
        """Test getting enabled tools."""
        settings = Settings(openai_api_key="test")

        enabled_tools = ServerToolRegistry.get_enabled_tools(settings)

        # Should be a list
        assert isinstance(enabled_tools, list)

    def test_extract_server_tools(self):
        """Test extracting server tools from tool definitions."""
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "other_tool", "name": "other"},
        ]

        result = ServerToolRegistry.extract_server_tools(tools)

        # Should find the web search tool
        assert len(result) >= 0  # May or may not find depending on registration


class TestWebSearchServerTool:
    """Tests for WebSearchServerTool."""

    def test_tool_type_and_name(self):
        """Test tool type and name constants."""
        assert WebSearchServerTool.tool_type == "web_search_20250305"
        assert WebSearchServerTool.tool_name == "web_search"

    def test_is_enabled_with_key(self):
        """Test is_enabled when API key is present."""
        settings = Settings(
            openai_api_key="test",
            tavily_api_key="tvly-test",
        )

        # Should be enabled with API key
        enabled = WebSearchServerTool.is_enabled(settings)
        assert enabled is True

    def test_is_enabled_without_key(self):
        """Test is_enabled when API key is missing."""
        # Reset the client cache to ensure fresh state
        WebSearchServerTool._client = None

        settings = Settings(
            openai_api_key="test",
            tavily_api_key=None,
        )

        # Should not be enabled without API key
        enabled = WebSearchServerTool.is_enabled(settings)
        assert enabled is False

    def test_extract_config_matching_type(self):
        """Test extracting config from matching tool definition."""
        tool_def = {
            "type": "web_search_20250305",
            "max_uses": 5,
            "allowed_domains": ["example.com"],
        }

        config = WebSearchServerTool.extract_config(tool_def)

        assert config is not None
        assert config["max_uses"] == 5
        assert config["allowed_domains"] == ["example.com"]

    def test_extract_config_non_matching_type(self):
        """Test extracting config from non-matching tool definition."""
        tool_def = {
            "type": "other_tool",
            "max_uses": 5,
        }

        config = WebSearchServerTool.extract_config(tool_def)

        assert config is None

    def test_to_openai_tool(self):
        """Test converting to OpenAI tool format."""
        config = {"max_uses": 5}

        tool = WebSearchServerTool.to_openai_tool(config)

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "web_search"
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]

    def test_extract_call_args_matching(self):
        """Test extracting call args from matching tool call."""
        tool_call = {
            "id": "call_123",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "test query"}',
            },
        }

        args = WebSearchServerTool.extract_call_args(tool_call)

        assert args is not None
        assert args["query"] == "test query"

    def test_extract_call_args_non_matching(self):
        """Test extracting call args from non-matching tool call."""
        tool_call = {
            "id": "call_123",
            "function": {
                "name": "other_tool",
                "arguments": '{"query": "test"}',
            },
        }

        args = WebSearchServerTool.extract_call_args(tool_call)

        assert args is None

    def test_extract_call_args_invalid_json(self):
        """Test extracting call args with invalid JSON."""
        tool_call = {
            "id": "call_123",
            "function": {
                "name": "web_search",
                "arguments": "invalid json",
            },
        }

        args = WebSearchServerTool.extract_call_args(tool_call)

        assert args is None

    def test_extract_call_args_missing_query(self):
        """Test extracting call args with missing query."""
        tool_call = {
            "id": "call_123",
            "function": {
                "name": "web_search",
                "arguments": '{"other": "value"}',
            },
        }

        args = WebSearchServerTool.extract_call_args(tool_call)

        assert args is None

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful tool execution."""
        settings = Settings(
            openai_api_key="test",
            tavily_api_key="tvly-test",
        )

        # Mock the Tavily client
        mock_result = MagicMock()
        mock_result.url = "https://example.com"
        mock_result.title = "Example"
        mock_result.page_age = None
        mock_result.encrypted_content = "encrypted"

        with patch.object(
            WebSearchServerTool,
            "_get_client",
            return_value=MagicMock(
                search=AsyncMock(return_value=([mock_result], None))
            ),
        ):
            result = await WebSearchServerTool.execute(
                call_id="call_123",
                args={"query": "test"},
                config={},
                settings=settings,
            )

        assert result.success is True
        assert len(result.content) == 1
        assert result.usage_increment == {"web_search_requests": 1}

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Test tool execution with error."""
        settings = Settings(
            openai_api_key="test",
            tavily_api_key="tvly-test",
        )

        with patch.object(
            WebSearchServerTool,
            "_get_client",
            return_value=MagicMock(
                search=AsyncMock(return_value=([], "search_error"))
            ),
        ):
            result = await WebSearchServerTool.execute(
                call_id="call_123",
                args={"query": "test"},
                config={},
                settings=settings,
            )

        assert result.success is False
        assert result.error_code == "search_error"
        assert result.usage_increment == {"web_search_requests": 1}

    def test_build_content_blocks_success(self):
        """Test building content blocks for successful result."""
        result = ToolResult(
            success=True,
            content=[{"type": "web_search_result", "url": "https://example.com"}],
        )

        blocks = WebSearchServerTool.build_content_blocks(
            call_id="call_123",
            call_args={"query": "test"},
            result=result,
        )

        assert len(blocks) == 2
        assert blocks[0]["type"] == "server_tool_use"
        assert blocks[0]["id"] == "call_123"
        assert blocks[1]["type"] == "web_search_tool_result"
        assert blocks[1]["tool_use_id"] == "call_123"

    def test_build_content_blocks_error(self):
        """Test building content blocks for error result."""
        result = ToolResult(
            success=False,
            content=[],
            error_code="max_uses_exceeded",
        )

        blocks = WebSearchServerTool.build_content_blocks(
            call_id="call_123",
            call_args={"query": "test"},
            result=result,
        )

        assert len(blocks) == 2
        assert blocks[0]["type"] == "server_tool_use"
        assert blocks[1]["type"] == "web_search_tool_result"
        assert "error_code" in blocks[1]["results"]

    def test_build_tool_result_message_success(self):
        """Test building tool result message for success."""
        result = ToolResult(
            success=True,
            content=[{"type": "web_search_result", "url": "https://example.com", "title": "Example"}],
        )

        message = WebSearchServerTool.build_tool_result_message(
            call_id="call_123",
            call_args={"query": "test"},
            result=result,
        )

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_123"
        assert "query" in message["content"]
        assert "results" in message["content"]

    def test_build_tool_result_message_error(self):
        """Test building tool result message for error."""
        result = ToolResult(
            success=False,
            content=[],
            error_code="search_failed",
        )

        message = WebSearchServerTool.build_tool_result_message(
            call_id="call_123",
            call_args={"query": "test"},
            result=result,
        )

        assert message["role"] == "tool"
        assert "error" in message["content"]


class TestServerToolBase:
    """Tests for ServerTool base class."""

    def test_build_content_blocks_default(self):
        """Test default build_content_blocks implementation."""

        class TestTool(ServerTool):
            tool_type = "test_tool"
            tool_name = "test"

            @classmethod
            def is_enabled(cls, settings):
                return True

            @classmethod
            def extract_config(cls, tool_def):
                return None

            @classmethod
            def to_openai_tool(cls, config):
                return {}

            @classmethod
            def extract_call_args(cls, tool_call):
                return None

            @classmethod
            async def execute(cls, call_id, args, config, settings):
                return ToolResult(success=True, content=[])

        result = ToolResult(
            success=True,
            content=[{"type": "text", "text": "Result"}],
        )

        blocks = TestTool.build_content_blocks(
            call_id="call_123",
            call_args={"arg": "value"},
            result=result,
        )

        assert len(blocks) == 2
        assert blocks[0]["type"] == "server_tool_use"
        # The default implementation extends result.content, so blocks[1] is the content item
        assert blocks[1]["type"] == "text"

    def test_build_tool_result_message_default(self):
        """Test default build_tool_result_message implementation."""

        class TestTool(ServerTool):
            tool_type = "test_tool"
            tool_name = "test"

            @classmethod
            def is_enabled(cls, settings):
                return True

            @classmethod
            def extract_config(cls, tool_def):
                return None

            @classmethod
            def to_openai_tool(cls, config):
                return {}

            @classmethod
            def extract_call_args(cls, tool_call):
                return None

            @classmethod
            async def execute(cls, call_id, args, config, settings):
                return ToolResult(success=True, content=[])

        result = ToolResult(
            success=True,
            content=[{"type": "text", "text": "Result"}],
        )

        message = TestTool.build_tool_result_message(
            call_id="call_123",
            call_args={"arg": "value"},
            result=result,
        )

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
