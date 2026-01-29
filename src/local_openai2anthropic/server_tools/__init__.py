# SPDX-License-Identifier: Apache-2.0
"""
Server tools plugin system for handling server-side tool execution.

Server tools (like web_search) are handled by the proxy server itself,
rather than being forwarded to the OpenAI backend.
"""

from local_openai2anthropic.server_tools.base import (
    ServerTool,
    ServerToolRegistry,
    ToolResult,
)
from local_openai2anthropic.server_tools.web_search import WebSearchServerTool

# Auto-register built-in tools
ServerToolRegistry.register(WebSearchServerTool)

__all__ = [
    "ServerTool",
    "ServerToolRegistry",
    "ToolResult",
    "WebSearchServerTool",
]
