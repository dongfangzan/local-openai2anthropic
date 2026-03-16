# SPDX-License-Identifier: Apache-2.0
"""
Base class for web search clients.
Provides common functionality for Tavily and TongXiao search clients.
"""

from abc import ABC, abstractmethod
from typing import Optional

import httpx

from local_openai2anthropic.protocol import WebSearchResult


class BaseSearchClient(ABC):
    """Abstract base class for web search clients.

    Implementations must provide:
    - DEFAULT_BASE_URL: The default API base URL
    - _build_payload(): Build the API request payload
    - _parse_response(): Parse the API response into WebSearchResult objects
    """

    DEFAULT_BASE_URL: str = ""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        base_url: Optional[str] = None,
    ):
        """Initialize the search client.

        Args:
            api_key: API key for authentication. If None, client is disabled.
            timeout: Request timeout in seconds.
            base_url: Optional custom base URL for the API.
        """
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._enabled = bool(api_key)

    def is_enabled(self) -> bool:
        """Check if the client is enabled (has API key)."""
        return self._enabled

    def _get_error_code(self, status_code: int) -> Optional[str]:
        """Map HTTP status code to error code.

        Returns None if status code is not an error.
        """
        error_map = {
            400: "invalid_input",
            413: "query_too_long",
            429: "too_many_requests",
        }

        if status_code in error_map:
            return error_map[status_code]
        if status_code >= 500:
            return "unavailable"
        if status_code in (401, 403):
            return "unavailable"
        if status_code >= 400:
            return "unavailable"
        return None

    @abstractmethod
    def _build_payload(self, query: str, max_results: int) -> dict:
        """Build the API request payload.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            Dictionary payload for the API request.
        """
        pass

    @abstractmethod
    def _parse_response(self, data: dict) -> list[WebSearchResult]:
        """Parse the API response into WebSearchResult objects.

        Args:
            data: JSON response data from the API.

        Returns:
            List of WebSearchResult objects.
        """
        pass

    @abstractmethod
    def _validate_query(self, query: str) -> Optional[str]:
        """Validate the search query.

        Args:
            query: Search query string.

        Returns:
            Error code if query is invalid, None otherwise.
        """
        pass

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> tuple[list[WebSearchResult], Optional[str]]:
        """Execute a web search.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            Tuple of (list of WebSearchResult, error_code or None).
        """
        if not self._enabled:
            return [], "unavailable"

        # Validate query (implementation-specific)
        if validation_error := self._validate_query(query):
            return [], validation_error

        url = f"{self.base_url}/search"
        headers = {
            "Content-Type": "application/json",
        }
        payload = self._build_payload(query, max_results)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)

                # Check for error status codes
                if error_code := self._get_error_code(response.status_code):
                    return [], error_code

                response.raise_for_status()
                data = response.json()

                results = self._parse_response(data)
                return results, None

        except httpx.TimeoutException:
            return [], "unavailable"
        except httpx.RequestError:
            return [], "unavailable"
        except Exception:
            return [], "unavailable"
