# SPDX-License-Identifier: Apache-2.0
"""
local-openai2anthropic: A proxy server that converts Anthropic Messages API to OpenAI API.
"""

__version__ = "0.2.4"

from local_openai2anthropic.protocol import (
    AnthropicError,
    AnthropicErrorResponse,
    ContentBlock,
    Message,
    MessageCreateParams,
    MessageParam,
    PingEvent,
    ServerToolUseBlock,
    TextBlock,
    ToolUseBlock,
    UsageWithCache,
    WebSearchResult,
    WebSearchToolDefinition,
    WebSearchToolResult,
)
from local_openai2anthropic.server_tools import (
    ServerTool,
    ServerToolRegistry,
    ToolResult,
)

__all__ = [
    "__version__",
    # Protocol types
    "AnthropicError",
    "AnthropicErrorResponse",
    "ContentBlock",
    "Message",
    "MessageCreateParams",
    "MessageParam",
    "PingEvent",
    "ServerToolUseBlock",
    "TextBlock",
    "ToolUseBlock",
    "UsageWithCache",
    "WebSearchResult",
    "WebSearchToolDefinition",
    "WebSearchToolResult",
    # Server tools
    "ServerTool",
    "ServerToolRegistry",
    "ToolResult",
]
