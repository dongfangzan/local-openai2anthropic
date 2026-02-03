# SPDX-License-Identifier: Apache-2.0
"""
Protocol definitions re-exported from official SDKs.
Uses Anthropic SDK types for request/response models.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel

# Re-export all Anthropic types for convenience
from anthropic.types import (
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ImageBlockParam,
    Message,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageParam,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    TextBlock,
    TextBlockParam,
    TextDelta,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from anthropic.types.beta import (
    BetaThinkingBlock,
    BetaThinkingConfigParam,
    BetaThinkingDelta,
)

# Import request types
from anthropic.types.message_create_params import MessageCreateParams


class UsageWithCache(BaseModel):
    """Extended usage with cache token support."""
    
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicError(BaseModel):
    """Error structure for Anthropic API."""
    
    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API."""
    
    type: str = "error"
    error: AnthropicError


class PingEvent(BaseModel):
    """Ping event for streaming responses."""

    type: str = "ping"


# Web Search Tool Types

class ApproximateLocation(BaseModel):
    """Approximate user location for web search."""

    type: Literal["approximate"] = "approximate"
    city: Optional[str] = None
    region: Optional[str] = None
    country: str = "US"
    timezone: Optional[str] = None


class WebSearchToolDefinition(BaseModel):
    """Web search tool definition (type: web_search_20250305)."""

    type: Literal["web_search_20250305"] = "web_search_20250305"
    name: str = "web_search"
    max_uses: Optional[int] = None
    allowed_domains: Optional[list[str]] = None
    blocked_domains: Optional[list[str]] = None
    user_location: Optional[ApproximateLocation] = None


class ServerToolUseBlock(BaseModel):
    """Server tool use block - represents a tool call request from the model."""

    type: Literal["server_tool_use"] = "server_tool_use"
    id: str
    name: str
    input: dict[str, Any]


class WebSearchResult(BaseModel):
    """Single web search result."""

    type: Literal["web_search_result"] = "web_search_result"
    url: str
    title: str
    page_age: Optional[str] = None
    encrypted_content: Optional[str] = None


class WebSearchToolResultContent(BaseModel):
    """Successful web search tool result content."""

    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    tool_use_id: str
    content: list[WebSearchResult]


class WebSearchToolResultError(BaseModel):
    """Error content for web search tool result."""

    type: Literal["web_search_tool_result_error"] = "web_search_tool_result_error"
    error_code: Literal["max_uses_exceeded", "too_many_requests", "unavailable"]


class WebSearchToolResult(BaseModel):
    """Web search tool result block (success or error)."""

    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    tool_use_id: str
    results: list[WebSearchResult] | WebSearchToolResultError  # 'results' for client


class WebSearchCitation(BaseModel):
    """Citation format for text blocks referencing search results."""

    type: Literal["web_search_result_location"] = "web_search_result_location"
    url: str
    title: str
    page_age: Optional[str] = None


class ServerToolUseUsage(BaseModel):
    """Usage tracking for server tool calls."""

    web_search_requests: int = 0


class UsageWithServerToolUse(BaseModel):
    """Extended usage with server tool use tracking."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    server_tool_use: Optional[ServerToolUseUsage] = None


__all__ = [
    # Content blocks
    "ContentBlock",
    "TextBlock",
    "ToolUseBlock",
    "BetaThinkingBlock",
    "ImageBlockParam",
    "TextBlockParam",
    "ToolUseBlockParam",
    "ToolResultBlockParam",

    # Message types
    "Message",
    "MessageParam",
    "MessageCreateParams",

    # Streaming events
    "MessageStreamEvent",
    "MessageStartEvent",
    "MessageDeltaEvent",
    "MessageStopEvent",
    "ContentBlockStartEvent",
    "ContentBlockDeltaEvent",
    "ContentBlockStopEvent",
    "PingEvent",

    # Delta types
    "TextDelta",
    "BetaThinkingDelta",

    # Usage
    "UsageWithCache",
    "UsageWithServerToolUse",
    "MessageDeltaUsage",

    # Config
    "BetaThinkingConfigParam",

    # Error
    "AnthropicError",
    "AnthropicErrorResponse",

    # Web Search Tool Types
    "ApproximateLocation",
    "WebSearchToolDefinition",
    "ServerToolUseBlock",
    "WebSearchResult",
    "WebSearchToolResult",
    "WebSearchToolResultContent",
    "WebSearchToolResultError",
    "WebSearchCitation",
    "ServerToolUseUsage",
]
