# SPDX-License-Identifier: Apache-2.0
"""
Tavily API client for web search functionality.
"""

import logging
from typing import Optional

from local_openai2anthropic.base_search_client import BaseSearchClient
from local_openai2anthropic.protocol import WebSearchResult

logger = logging.getLogger(__name__)


class TavilyClient(BaseSearchClient):
    """Client for Tavily Search API."""

    DEFAULT_BASE_URL = "https://api.tavily.com"

    def _validate_query(self, query: str) -> Optional[str]:
        """Validate the search query."""
        if not query or not query.strip():
            logger.warning("Tavily search called with empty query")
            return "invalid_input"
        return None

    def _build_payload(self, query: str, max_results: int, search_depth: str = "basic") -> dict:
        """Build the Tavily API request payload."""
        return {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": False,
            "include_raw_content": False,
        }

    def _parse_response(self, data: dict) -> list[WebSearchResult]:
        """Parse Tavily API response into WebSearchResult objects."""
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

        logger.debug(f"Tavily search returned {len(results)} results")
        return results

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
        """
        if not self._enabled:
            logger.warning("Tavily search called but API key not configured")
            return [], "unavailable"

        if validation_error := self._validate_query(query):
            return [], validation_error

        import httpx

        url = f"{self.base_url}/search"
        headers = {"Content-Type": "application/json"}
        payload = self._build_payload(query, max_results, search_depth)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)

                if error_code := self._get_error_code(response.status_code):
                    if error_code == "invalid_input":
                        logger.warning("Tavily invalid request")
                    elif error_code == "query_too_long":
                        logger.warning("Tavily query too long")
                    elif error_code == "too_many_requests":
                        logger.warning("Tavily rate limit exceeded")
                    else:
                        logger.error(f"Tavily error: {response.status_code}")
                    return [], error_code

                response.raise_for_status()
                data = response.json()

                results = self._parse_response(data)
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
