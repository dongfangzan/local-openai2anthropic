# SPDX-License-Identifier: Apache-2.0
"""
Web search server tool implementation using Tavily API.
"""

import json
from typing import Any, ClassVar

from local_openai2anthropic.config import Settings
from local_openai2anthropic.server_tools.base import ServerTool, ToolResult
from local_openai2anthropic.tavily_client import TavilyClient


class WebSearchServerTool(ServerTool):
    """
    Web search server tool using Tavily API.

    Tool type: web_search_20250305
    OpenAI function name: web_search
    """

    tool_type: ClassVar[str] = "web_search_20250305"
    tool_name: ClassVar[str] = "web_search"

    _client: ClassVar[TavilyClient | None] = None

    @classmethod
    def _get_client(cls, settings: Settings) -> TavilyClient:
        """Get or create Tavily client singleton."""
        if cls._client is None:
            cls._client = TavilyClient(
                api_key=settings.tavily_api_key,
                timeout=settings.tavily_timeout,
            )
        return cls._client

    @classmethod
    def is_enabled(cls, settings: Settings) -> bool:
        """Check if Tavily is configured."""
        client = cls._get_client(settings)
        return client.is_enabled()

    @classmethod
    def extract_config(cls, tool_def: dict[str, Any]) -> dict[str, Any] | None:
        """Extract web search configuration from tool definition."""
        if tool_def.get("type") != cls.tool_type:
            return None

        return {
            "max_uses": tool_def.get("max_uses"),
            "allowed_domains": tool_def.get("allowed_domains"),
            "blocked_domains": tool_def.get("blocked_domains"),
            "user_location": tool_def.get("user_location"),
        }

    @classmethod
    def to_openai_tool(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Convert to OpenAI function tool format."""
        return {
            "type": "function",
            "function": {
                "name": cls.tool_name,
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    @classmethod
    def extract_call_args(cls, tool_call: dict[str, Any]) -> dict[str, Any] | None:
        """Extract search query from OpenAI tool call."""
        func = tool_call.get("function", {})
        if func.get("name") != cls.tool_name:
            return None

        try:
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            query = args.get("query")
            if query:
                return {"query": query}
        except json.JSONDecodeError:
            pass

        return None

    @classmethod
    async def execute(
        cls,
        call_id: str,
        args: dict[str, Any],
        config: dict[str, Any],
        settings: Settings,
    ) -> ToolResult:
        """Execute web search using Tavily."""
        query = args.get("query", "")
        client = cls._get_client(settings)
        max_results = settings.tavily_max_results

        results, error = await client.search(query, max_results=max_results)

        if error:
            return ToolResult(
                success=False,
                content=[],
                error_code=error,
                usage_increment={"web_search_requests": 1},
            )

        # Convert results to content blocks - match Anthropic API format
        content_blocks = [
            {
                "type": "web_search_result",
                "url": r.url,
                "title": r.title,
                "page_age": r.page_age,
                "encrypted_content": r.encrypted_content or "",
            }
            for r in results
        ]

        return ToolResult(
            success=True,
            content=content_blocks,
            usage_increment={"web_search_requests": 1},
        )

    @classmethod
    def build_content_blocks(
        cls,
        call_id: str,
        call_args: dict[str, Any],
        result: ToolResult,
    ) -> list[dict[str, Any]]:
        """
        Build web_search specific content blocks.
        Format: server_tool_use + web_search_tool_result (Anthropic official format)
        """
        blocks: list[dict[str, Any]] = []

        # 1. server_tool_use block - signals a server-side tool was invoked
        blocks.append({
            "type": "server_tool_use",
            "id": call_id,
            "name": cls.tool_name,
            "input": call_args,
        })

        # 2. web_search_tool_result block - contains the search results
        # Note: Claude Code client expects 'results' field (not 'content') for counting
        if result.success:
            blocks.append({
                "type": "web_search_tool_result",
                "tool_use_id": call_id,
                "results": result.content,
            })
        else:
            blocks.append({
                "type": "web_search_tool_result",
                "tool_use_id": call_id,
                "results": {
                    "type": "web_search_tool_result_error",
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
        """Build tool result message for OpenAI conversation."""
        if result.success:
            content = {
                "query": call_args.get("query", ""),
                "results": [
                    {
                        "url": item.get("url"),
                        "title": item.get("title"),
                        "snippet": item.get("snippet"),
                        "page_age": item.get("page_age"),
                    }
                    for item in result.content
                ],
            }
        else:
            content = {
                "error": result.error_code,
                "message": f"Web search failed: {result.error_code}",
            }

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(content),
        }
