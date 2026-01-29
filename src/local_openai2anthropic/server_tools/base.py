# SPDX-License-Identifier: Apache-2.0
"""
Base classes for server tool plugin system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from local_openai2anthropic.config import Settings


@dataclass
class ToolResult:
    """Result of a server tool execution."""

    success: bool
    content: list[dict[str, Any]]  # Content blocks to add to response
    error_code: str | None = None  # Error code if failed
    usage_increment: dict[str, int] = field(default_factory=dict)  # e.g., {"web_search_requests": 1}


class ServerTool(ABC):
    """
    Base class for server-side tools.

    Server tools are executed by the proxy server itself (like web search),
    rather than being forwarded to the OpenAI backend.
    """

    # Tool type identifier, e.g., "web_search_20250305"
    tool_type: ClassVar[str]

    # Human-readable tool name for OpenAI function calling
    tool_name: ClassVar[str]

    @classmethod
    @abstractmethod
    def is_enabled(cls, settings: Settings) -> bool:
        """Check if this tool is enabled based on configuration."""
        pass

    @classmethod
    @abstractmethod
    def extract_config(cls, tool_def: dict[str, Any]) -> dict[str, Any] | None:
        """
        Extract tool-specific configuration from the tool definition.
        Returns None if this tool doesn't handle this definition.
        """
        pass

    @classmethod
    @abstractmethod
    def to_openai_tool(cls, config: dict[str, Any]) -> dict[str, Any]:
        """
        Convert to OpenAI function tool format.
        This is what gets sent to the OpenAI backend.
        """
        pass

    @classmethod
    @abstractmethod
    def extract_call_args(cls, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        """
        Extract call arguments from an OpenAI tool call.
        Returns None if this tool doesn't handle this call.
        """
        pass

    @classmethod
    @abstractmethod
    async def execute(
        cls,
        call_id: str,
        args: dict[str, Any],
        config: dict[str, Any],
        settings: Settings,
    ) -> ToolResult:
        """Execute the tool and return results."""
        pass

    @classmethod
    def build_content_blocks(
        cls,
        call_id: str,
        call_args: dict[str, Any],
        result: ToolResult,
    ) -> list[dict[str, Any]]:
        """
        Build content blocks for the response.
        Default implementation creates server_tool_use + tool_result blocks.
        Subclasses can override for custom formats.
        """
        blocks: list[dict[str, Any]] = []

        # 1. server_tool_use block
        blocks.append({
            "type": "server_tool_use",
            "id": call_id,
            "name": cls.tool_name,
            "input": call_args,
        })

        # 2. tool_result block
        if result.success:
            blocks.extend(result.content)
        else:
            blocks.append({
                "type": f"{cls.tool_name}_tool_result",
                "tool_use_id": call_id,
                "content": {
                    "type": f"{cls.tool_name}_tool_result_error",
                    "error_code": result.error_code,
                },
            })

        return blocks

    @classmethod
    def build_tool_result_message(
        cls,
        call_id: str,
        call_args: dict[str, Any],
        result: ToolResult,
    ) -> dict[str, Any]:
        """
        Build the tool result message for OpenAI conversation continuation.
        This is what gets added back to the messages list.
        """
        if result.success:
            content = {
                "query": call_args.get("query", ""),
                "results": result.content,
            }
        else:
            content = {
                "error": result.error_code,
                "message": f"{cls.tool_name} failed: {result.error_code}",
            }

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }


T = TypeVar("T", bound=ServerTool)


class ServerToolRegistry:
    """Registry for server tools."""

    _tools: ClassVar[dict[str, type[ServerTool]]] = {}

    @classmethod
    def register(cls, tool_class: type[T]) -> type[T]:
        """Register a server tool class."""
        cls._tools[tool_class.tool_type] = tool_class
        return tool_class

    @classmethod
    def get(cls, tool_type: str) -> type[ServerTool] | None:
        """Get a server tool class by type."""
        return cls._tools.get(tool_type)

    @classmethod
    def all_tools(cls) -> list[type[ServerTool]]:
        """Get all registered server tools."""
        return list(cls._tools.values())

    @classmethod
    def get_enabled_tools(cls, settings: Settings) -> list[type[ServerTool]]:
        """Get all enabled server tools based on settings."""
        return [t for t in cls._tools.values() if t.is_enabled(settings)]

    @classmethod
    def extract_server_tools(
        cls,
        tools: list[dict[str, Any]],
    ) -> list[tuple[type[ServerTool], dict[str, Any]]]:
        """
        Extract server tool configurations from a tools list.
        Returns list of (tool_class, config) tuples.
        """
        result = []
        for tool_def in tools:
            tool_type = tool_def.get("type")
            if tool_type and (tool_class := cls.get(tool_type)):
                config = tool_class.extract_config(tool_def)
                if config is not None:
                    result.append((tool_class, config))
        return result
