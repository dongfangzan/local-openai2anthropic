# SPDX-License-Identifier: Apache-2.0
"""
Web search server tool implementation using Tavily and/or TongXiao API.
"""

import asyncio
import json
from typing import Any, ClassVar

from local_openai2anthropic.config import Settings
from local_openai2anthropic.server_tools.base import ServerTool, ToolResult
from local_openai2anthropic.tavily_client import TavilyClient
from local_openai2anthropic.tongxiao_client import TongXiaoClient


class WebSearchServerTool(ServerTool):
    """
    Web search server tool using Tavily and/or TongXiao API.

    Tool type: web_search_20250305
    OpenAI function name: web_search

    Supports multiple search providers:
    - "tavily": Use Tavily API only
    - "tongxiao": Use TongXiao (通晓) API only
    - "both": Use both and merge results
    """

    tool_type: ClassVar[str] = "web_search_20250305"
    tool_name: ClassVar[str] = "web_search"

    _tavily_client: ClassVar[TavilyClient | None] = None
    _tongxiao_client: ClassVar[TongXiaoClient | None] = None

    @classmethod
    def _get_tavily_client(cls, settings: Settings) -> TavilyClient:
        """Get or create Tavily client singleton."""
        if cls._tavily_client is None:
            cls._tavily_client = TavilyClient(
                api_key=settings.tavily_api_key,
                timeout=settings.tavily_timeout,
            )
        return cls._tavily_client

    @classmethod
    def _get_tongxiao_client(cls, settings: Settings) -> TongXiaoClient:
        """Get or create TongXiao client singleton."""
        if cls._tongxiao_client is None:
            cls._tongxiao_client = TongXiaoClient(
                api_key=settings.tongxiao_api_key,
                timeout=settings.tongxiao_timeout,
            )
        return cls._tongxiao_client

    @classmethod
    def is_enabled(cls, settings: Settings) -> bool:
        """Check if any search provider is configured."""
        tavily = cls._get_tavily_client(settings)
        tongxiao = cls._get_tongxiao_client(settings)
        return tavily.is_enabled() or tongxiao.is_enabled()

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
    def _normalize_url(cls, url: str) -> str:
        """Normalize URL for deduplication.

        Handles:
        - Case normalization
        - Protocol stripping (http://, https://)
        - www. prefix stripping
        - Trailing slash removal
        - Fragment removal (#section)
        """
        url = url.lower().strip()

        # Strip protocol
        if url.startswith("http://"):
            url = url[7:]
        elif url.startswith("https://"):
            url = url[8:]

        # Strip www. prefix
        if url.startswith("www."):
            url = url[4:]

        # Remove fragment
        if "#" in url:
            url = url.split("#")[0]

        # Remove trailing slash
        url = url.rstrip("/")

        return url

    @classmethod
    async def _deduplicate_results(
        cls,
        results: list,
        max_results: int,
    ) -> list:
        """Deduplicate results by URL and return top results."""
        seen_urls = set()
        unique_results = []

        for result in results:
            if len(unique_results) >= max_results:
                break

            normalized_url = cls._normalize_url(result.url)

            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)

        return unique_results

    @classmethod
    async def _execute_single_search(
        cls,
        client: TavilyClient | TongXiaoClient,
        query: str,
        max_results: int,
        provider_name: str,
    ) -> tuple[list, str | None]:
        """Execute search with a single provider. Returns (results, error or None)."""
        if not client.is_enabled():
            return [], None

        results, error = await client.search(query, max_results=max_results)
        if error:
            return [], f"{provider_name}:{error}"
        return results, None

    @classmethod
    async def execute(
        cls,
        call_id: str,
        args: dict[str, Any],
        config: dict[str, Any],
        settings: Settings,
    ) -> ToolResult:
        """Execute web search using configured provider(s)."""
        query = args.get("query", "")
        provider = settings.websearch_provider.lower()

        tavily_client = cls._get_tavily_client(settings)
        tongxiao_client = cls._get_tongxiao_client(settings)

        # Build list of search tasks based on provider configuration
        search_tasks = []
        task_meta = []  # Track provider name for error reporting

        if provider in ("tavily", "both"):
            search_tasks.append(
                cls._execute_single_search(
                    tavily_client, query, settings.tavily_max_results, "tavily"
                )
            )
            task_meta.append("tavily")

        if provider in ("tongxiao", "both"):
            search_tasks.append(
                cls._execute_single_search(
                    tongxiao_client, query, settings.tongxiao_max_results, "tongxiao"
                )
            )
            task_meta.append("tongxiao")

        # Execute searches concurrently (when provider="both", this runs both in parallel)
        all_results: list = []
        errors: list[str] = []

        if search_tasks:
            results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

            for i, result in enumerate(results_list):
                if isinstance(result, Exception):
                    errors.append(f"{task_meta[i]}:exception")
                else:
                    results, error = result
                    if error:
                        errors.append(error)
                    else:
                        all_results.extend(results)

        # If we have no results and there were errors, return error
        if not all_results:
            # Prefer specific error codes
            for error in errors:
                for code in ["invalid_input", "query_too_long", "too_many_requests"]:
                    if code in error:
                        return ToolResult(
                            success=False,
                            content=[],
                            error_code=code,
                            usage_increment={"web_search_requests": 1},
                        )
            return ToolResult(
                success=False,
                content=[],
                error_code="unavailable",
                usage_increment={"web_search_requests": 1},
            )

        # Deduplicate and limit results
        max_results = max(settings.tavily_max_results, settings.tongxiao_max_results)
        final_results = await cls._deduplicate_results(all_results, max_results)

        # Convert results to content blocks - match Anthropic API format
        content_blocks = [
            {
                "type": "web_search_result",
                "url": r.url,
                "title": r.title,
                "page_age": r.page_age,
                "encrypted_content": r.encrypted_content or "",
            }
            for r in final_results
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
        blocks.append(
            {
                "type": "server_tool_use",
                "id": call_id,
                "name": cls.tool_name,
                "input": call_args,
            }
        )

        # 2. web_search_tool_result block - contains the search results
        # Provide both 'results' and 'content' for client compatibility.
        if result.success:
            blocks.append(
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": call_id,
                    "results": result.content,
                    "content": result.content,
                }
            )
        else:
            error_payload = {
                "type": "web_search_tool_result_error",
                "error_code": result.error_code or "unavailable",
            }
            blocks.append(
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": call_id,
                    "results": error_payload,
                    "content": error_payload,
                }
            )

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
                        "content": item.get("encrypted_content"),
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
