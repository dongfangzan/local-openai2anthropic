# SPDX-License-Identifier: Apache-2.0
"""
TongXiao (通晓) API client for web search functionality.
阿里云通晓统一搜索 API 客户端
"""

import logging
from typing import Optional

from local_openai2anthropic.base_search_client import BaseSearchClient
from local_openai2anthropic.protocol import WebSearchResult

logger = logging.getLogger(__name__)


class TongXiaoClient(BaseSearchClient):
    """Client for TongXiao Search API (阿里云通晓统一搜索)."""

    DEFAULT_BASE_URL = "https://cloud-iqs.aliyuncs.com/search"

    def _validate_query(self, query: str) -> Optional[str]:
        """Validate the search query.

        TongXiao API requires query length between 2 and 100 characters.
        """
        if not query or not query.strip():
            logger.warning("TongXiao search called with empty query")
            return "invalid_input"

        query = query.strip()
        if len(query) < 2:
            logger.warning(f"TongXiao query too short: {len(query)} chars")
            return "invalid_input"

        if len(query) > 100:
            logger.warning(f"TongXiao query too long: {len(query)} chars")
            return "query_too_long"

        return None

    def _build_payload(self, query: str, max_results: int) -> dict:
        """Build the TongXiao API request payload."""
        # Limit max_results to 10 as per API spec
        num_results = min(max(max_results, 1), 10)

        return {
            "query": query.strip(),
            "numResults": num_results,
        }

    def _parse_response(self, data: dict) -> list[WebSearchResult]:
        """Parse TongXiao API response into WebSearchResult objects."""
        results = []
        for item in data.get("pageItems", []):
            # Combine summary and mainText for content
            content_parts = []
            if summary := item.get("summary"):
                content_parts.append(summary)
            if main_text := item.get("mainText"):
                content_parts.append(main_text)

            content = "\n".join(content_parts) if content_parts else ""

            # Get published date - API uses 'publishedTime' field
            page_age = item.get("publishedTime") or item.get("publishedDate")

            result = WebSearchResult(
                type="web_search_result",
                url=item.get("link", ""),
                title=item.get("title", ""),
                page_age=page_age,
                encrypted_content=content or None,
            )
            results.append(result)

        logger.debug(f"TongXiao search returned {len(results)} results")
        return results

    def _get_headers(self) -> dict:
        """Get request headers with authentication."""
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> tuple[list[WebSearchResult], Optional[str]]:
        """
        Execute a web search using TongXiao API.

        Args:
            query: Search query string (length: >=2 and <=100).
            max_results: Maximum number of results to return (max 10).

        Returns:
            Tuple of (list of WebSearchResult, error_code or None).
        """
        if not self._enabled:
            logger.warning("TongXiao search called but API key not configured")
            return [], "unavailable"

        if validation_error := self._validate_query(query):
            return [], validation_error

        import httpx

        url = f"{self.base_url}/llm"
        headers = self._get_headers()
        payload = self._build_payload(query, max_results)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)

                # TongXiao uses 401/403 for auth errors
                if response.status_code in (401, 403):
                    logger.error(f"TongXiao authentication error: {response.status_code}")
                    return [], "unavailable"

                if error_code := self._get_error_code(response.status_code):
                    if error_code == "invalid_input":
                        logger.warning("TongXiao invalid request")
                    elif error_code == "query_too_long":
                        logger.warning("TongXiao query too long")
                    elif error_code == "too_many_requests":
                        logger.warning("TongXiao rate limit exceeded")
                    else:
                        logger.error(f"TongXiao error: {response.status_code}")
                    return [], error_code

                response.raise_for_status()
                data = response.json()

                results = self._parse_response(data)
                return results, None

        except httpx.TimeoutException:
            logger.error("TongXiao search request timed out")
            return [], "unavailable"
        except httpx.RequestError as e:
            logger.error(f"TongXiao search request failed: {e}")
            return [], "unavailable"
        except Exception as e:
            logger.error(f"TongXiao search unexpected error: {e}")
            return [], "unavailable"
