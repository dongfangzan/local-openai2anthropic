# SPDX-License-Identifier: Apache-2.0
"""
OpenAI API type definitions for compatibility with vLLM/SGLang responses.

This module defines Pydantic models compatible with OpenAI API responses,
these models support additional fields like `reasoning_content` that are
returned by vLLM/SGLang but not present in the official OpenAI SDK.
"""

from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel


# TypedDict types for parameters (used as dict in code)
class ChatCompletionToolFunction(TypedDict):
    """Function definition for a tool."""

    name: str
    description: str
    parameters: dict[str, Any]


class ChatCompletionToolParam(TypedDict):
    """Tool parameter for chat completion."""

    type: Literal["function"]
    function: ChatCompletionToolFunction


class CompletionCreateParams(TypedDict, total=False):
    """Parameters for creating a chat completion."""

    model: str
    messages: list[dict[str, Any]]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stream: bool
    stop: list[str]
    tools: list[ChatCompletionToolParam]
    tool_choice: str | dict[str, Any]
    stream_options: dict[str, Any]
    # Additional fields for vLLM/SGLang compatibility
    chat_template_kwargs: dict[str, Any]
    # Internal field for server tools config
    _server_tools_config: dict[str, dict[str, Any]]


# Pydantic models for API responses
class Function(BaseModel):
    """A function call."""

    name: str
    arguments: str


class ChatCompletionMessageToolCall(BaseModel):
    """A tool call in a chat completion message."""

    id: str
    type: str = "function"
    function: Function


class ChatCompletionMessage(BaseModel):
    """A chat completion message."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ChatCompletionMessageToolCall]] = None
    # Additional field for reasoning content (thinking) from vLLM/SGLang
    reasoning_content: Optional[str] = None


class Choice(BaseModel):
    """A choice in a chat completion response."""

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None


class FunctionDelta(BaseModel):
    """A function call delta."""

    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatCompletionDeltaToolCall(BaseModel):
    """A tool call delta in a streaming response."""

    index: int = 0
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[FunctionDelta] = None


class ChoiceDelta(BaseModel):
    """A delta in a streaming chat completion response."""

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[ChatCompletionDeltaToolCall]] = None
    # Additional field for reasoning content (thinking) from vLLM/SGLang
    reasoning_content: Optional[str] = None


class StreamingChoice(BaseModel):
    """A choice in a streaming chat completion response."""

    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class CompletionUsage(BaseModel):
    """Usage statistics for a completion request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Optional cache-related fields
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class ChatCompletion(BaseModel):
    """A chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Optional[CompletionUsage] = None


class ChatCompletionChunk(BaseModel):
    """A chunk in a streaming chat completion response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamingChoice]
    usage: Optional[CompletionUsage] = None
