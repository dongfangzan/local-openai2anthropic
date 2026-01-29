# SPDX-License-Identifier: Apache-2.0
"""
Tavily API client for web search functionality.
"""

import logging
from typing import Optional

import httpx

from local_openai2anthropic.protocol import WebSearchResult

logger = logging.getLogger(__name__)


class TavilyClient:
    """Client for Tavily Search API."""

    BASE_URL = "https://api.tavily.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Tavily client.

        Args:
            api_key: Tavily API key. If None, client is disabled.
            timeout: Request timeout in seconds.
            base_url: Optional custom base URL for Tavily API.
        """
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url or self.BASE_URL
        self._enabled = bool(api_key)

    def is_enabled(self) -> bool:
        """Check if web search is enabled (API key configured)."""
        return self._enabled

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> tuple[list[WebSearchResult], Optional[str]]:
        """
        Execute a web search using Tavily API.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            search_depth: Search depth - "basic" or "advanced".

        Returns:
            Tuple of (list of WebSearchResult, error_code or None).
            Error codes: "max_uses_exceeded", "too_many_requests", "unavailable"
        """
        if not self._enabled:
            logger.warning("Tavily search called but API key not configured")
            return [], "unavailable"

        url = f"{self.base_url}/search"
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code == 429:
                    logger.warning("Tavily rate limit exceeded")
                    return [], "too_many_requests"

                if response.status_code >= 500:
                    logger.error(f"Tavily server error: {response.status_code}")
                    return [], "unavailable"

                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("results", []):
                    result = WebSearchResult(
                        type="web_search_result",
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        page_age=item.get("published_date"),
                        encrypted_content=item.get("content", ""),
                    )
                    results.append(result)

                logger.debug(f"Tavily search returned {len(results)} results for query: {query[:50]}...")
                return results, None

        except httpx.TimeoutException:
            logger.error("Tavily search request timed out")
            return [], "unavailable"
        except httpx.RequestError as e:
            logger.error(f"Tavily search request failed: {e}")
            return [], "unavailable"
        except Exception as e:
            logger.error(f"Tavily search unexpected error: {e}")
            return [], "unavailable"
