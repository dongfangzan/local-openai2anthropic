# SPDX-License-Identifier: Apache-2.0
"""Token-related utility functions."""

import json
import secrets
import string
from typing import Any


def _generate_server_tool_id() -> str:
    """Generate Anthropic-style server tool use ID (srvtoolu_...)."""
    # Generate 24 random alphanumeric characters
    chars = string.ascii_lowercase + string.digits
    random_part = "".join(secrets.choice(chars) for _ in range(24))
    return f"srvtoolu_{random_part}"


def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return usage
    allowed_keys = {
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "server_tool_use",
    }
    normalized = {k: v for k, v in usage.items() if k in allowed_keys}
    return normalized or None


def _count_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore[import-not-found]
    except Exception:
        return 0

    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _estimate_input_tokens(openai_params: dict[str, Any]) -> int:
    try:
        import tiktoken  # type: ignore[import-not-found]
    except Exception:
        return 0

    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0

    system = openai_params.get("system")
    if isinstance(system, str):
        total_tokens += len(encoding.encode(system))

    messages = openai_params.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                total_tokens += len(encoding.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        total_tokens += len(encoding.encode(str(block)))
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        total_tokens += len(encoding.encode(block.get("text", "")))
                    elif block_type == "image_url":
                        total_tokens += 85

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                total_tokens += len(encoding.encode(json.dumps(tool_calls)))

    tools = openai_params.get("tools")
    if isinstance(tools, list) and tools:
        total_tokens += len(encoding.encode(json.dumps(tools)))

    tool_choice = openai_params.get("tool_choice")
    if tool_choice is not None:
        total_tokens += len(encoding.encode(json.dumps(tool_choice)))

    response_format = openai_params.get("response_format")
    if response_format is not None:
        total_tokens += len(encoding.encode(json.dumps(response_format)))

    return total_tokens
