# SPDX-License-Identifier: Apache-2.0
"""
Protocol definitions re-exported from official SDKs.
Uses Anthropic SDK types for request/response models.
"""

from typing import Optional

from pydantic import BaseModel, Field

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
    "MessageDeltaUsage",
    
    # Config
    "BetaThinkingConfigParam",
    
    # Error
    "AnthropicError",
    "AnthropicErrorResponse",
]
