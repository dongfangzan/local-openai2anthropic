"""
Tests for the tongxiao_client module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from local_openai2anthropic.tongxiao_client import TongXiaoClient


class TestTongXiaoClient:
    """Tests for TongXiaoClient."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = TongXiaoClient(api_key="tx-test")

        assert client.api_key == "tx-test"
        assert client.timeout == 30.0
        assert client.base_url == "https://cloud-iqs.aliyuncs.com/search"
        assert client._enabled is True

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        client = TongXiaoClient(api_key=None)

        assert client.api_key is None
        assert client._enabled is False

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = TongXiaoClient(api_key="tx-test", timeout=60.0)

        assert client.timeout == 60.0

    def test_init_custom_base_url(self):
        """Test initialization with custom base URL."""
        client = TongXiaoClient(
            api_key="tx-test",
            base_url="https://custom.iqs.aliyuncs.com",
        )

        assert client.base_url == "https://custom.iqs.aliyuncs.com"

    def test_is_enabled_true(self):
        """Test is_enabled returns True when API key is set."""
        client = TongXiaoClient(api_key="tx-test")

        assert client.is_enabled() is True

    def test_is_enabled_false(self):
        """Test is_enabled returns False when API key is not set."""
        client = TongXiaoClient(api_key=None)

        assert client.is_enabled() is False

    @pytest.mark.asyncio
    async def test_search_disabled(self):
        """Test search when client is disabled."""
        client = TongXiaoClient(api_key=None)

        results, error = await client.search("test query")

        assert results == []
        assert error == "unavailable"

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        """Test search with empty query."""
        client = TongXiaoClient(api_key="tx-test")

        # Empty query should be handled locally without making HTTP request
        results, error = await client.search("   ")

        assert results == []
        assert error == "invalid_input"

    @pytest.mark.asyncio
    async def test_search_query_too_short(self):
        """Test search with query that's too short (less than 2 chars)."""
        client = TongXiaoClient(api_key="tx-test")

        results, error = await client.search("a")

        assert results == []
        assert error == "invalid_input"

    @pytest.mark.asyncio
    async def test_search_query_too_long(self):
        """Test search with query that's too long (more than 100 chars)."""
        client = TongXiaoClient(api_key="tx-test")

        results, error = await client.search("a" * 101)

        assert results == []
        assert error == "query_too_long"

    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful search."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "pageItems": [
                {
                    "link": "https://example.com",
                    "title": "Example Title",
                    "publishedTime": "2024-01-01",
                    "summary": "Example summary",
                    "mainText": "Example main text",
                },
            ],
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error is None
        assert len(results) == 1
        assert results[0].url == "https://example.com"
        assert results[0].title == "Example Title"
        assert results[0].page_age == "2024-01-01"
        # summary + mainText combined
        assert "Example summary" in results[0].encrypted_content
        assert "Example main text" in results[0].encrypted_content

    @pytest.mark.asyncio
    async def test_search_success_with_published_date(self):
        """Test successful search using publishedDate field (alternate field name)."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "pageItems": [
                {
                    "link": "https://example.com",
                    "title": "Example Title",
                    "publishedDate": "2024-06-15",  # Alternate field name
                    "summary": "Example summary",
                },
            ],
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error is None
        assert len(results) == 1
        assert results[0].page_age == "2024-06-15"

    @pytest.mark.asyncio
    async def test_search_rate_limit(self):
        """Test search with rate limit error."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error == "too_many_requests"
        assert results == []

    @pytest.mark.asyncio
    async def test_search_invalid_input(self):
        """Test search with invalid input error."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 400

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("bad query")

        assert error == "invalid_input"
        assert results == []

    @pytest.mark.asyncio
    async def test_search_query_too_long_http(self):
        """Test search with query too long error (HTTP 413)."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 413

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error == "query_too_long"
        assert results == []

    @pytest.mark.asyncio
    async def test_search_authentication_error(self):
        """Test search with authentication error (401/403)."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 403

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error == "unavailable"
        assert results == []

    @pytest.mark.asyncio
    async def test_search_server_error(self):
        """Test search with server error."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error == "unavailable"
        assert results == []

    @pytest.mark.asyncio
    async def test_search_timeout(self):
        """Test search with timeout."""
        client = TongXiaoClient(api_key="tx-test")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error == "unavailable"
        assert results == []

    @pytest.mark.asyncio
    async def test_search_request_error(self):
        """Test search with request error."""
        client = TongXiaoClient(api_key="tx-test")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error == "unavailable"
        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """Test search with empty results."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"pageItems": []}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error is None
        assert results == []

    @pytest.mark.asyncio
    async def test_search_multiple_results(self):
        """Test search with multiple results."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "pageItems": [
                {
                    "link": "https://example1.com",
                    "title": "Title 1",
                    "publishedTime": "2024-01-01",
                    "summary": "Summary 1",
                },
                {
                    "link": "https://example2.com",
                    "title": "Title 2",
                    "publishedTime": "2024-01-02",
                    "summary": "Summary 2",
                },
            ],
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query", max_results=10)

        assert error is None
        assert len(results) == 2
        assert results[0].url == "https://example1.com"
        assert results[1].url == "https://example2.com"

    @pytest.mark.asyncio
    async def test_search_missing_optional_fields(self):
        """Test search with missing optional fields in results."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "pageItems": [
                {
                    "link": "https://example.com",
                    "title": "Example",
                    # publishedTime, summary, mainText are missing
                },
            ],
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error is None
        assert len(results) == 1
        assert results[0].page_age is None
        assert results[0].encrypted_content is None

    @pytest.mark.asyncio
    async def test_search_only_summary_no_maintext(self):
        """Test search with only summary field (no mainText)."""
        client = TongXiaoClient(api_key="tx-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "pageItems": [
                {
                    "link": "https://example.com",
                    "title": "Example",
                    "summary": "Only summary",
                },
            ],
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results, error = await client.search("test query")

        assert error is None
        assert len(results) == 1
        assert results[0].encrypted_content == "Only summary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
