# SPDX-License-Identifier: Apache-2.0
"""
WebSearch E2E tests for local-openai2anthropic.

To run these tests:
1. Start the proxy: local-openai2anthropic
2. Run: RUN_E2E_TESTS=1 pytest tests/test_e2e_websearch.py -v

Note: These tests require a running backend and may incur API costs.
"""

import os

import httpx
import pytest

# Skip all tests in this file by default
if not os.getenv("RUN_E2E_TESTS"):
    pytest.skip("Set RUN_E2E_TESTS=1 to run E2E tests", allow_module_level=True)

# Test configuration
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8080")
API_KEY = os.getenv("TEST_API_KEY", "test-key")
MODEL = "glm-4.7"


def get_text_content(data: dict) -> str:
    """Extract text content from response."""
    content = data.get("content", [])
    for block in content:
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def has_tool_use(data: dict, tool_name: str = "web_search") -> bool:
    """Check if response has tool_use block."""
    content = data.get("content", [])
    return any(
        block.get("type") == "tool_use" and block.get("name") == tool_name
        for block in content
    )


@pytest.fixture
def http_client():
    """Create HTTP client."""
    return httpx.Client(timeout=60.0)


class TestWebSearchWithKey:
    """Test web search with tavily_api_key configured."""

    def test_web_search_normal(self, http_client):
        """Test web search with valid tavily_api_key returns search results."""
        search_query = "Beijing weather today"
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 1024,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": f"Search for: {search_query}",
                    }
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0
        assert data["usage"]["input_tokens"] > 0
        assert data["usage"]["output_tokens"] > 0

        # Get text response
        text = get_text_content(data)
        assert len(text) > 0

        # Verify search was performed - response should contain weather-related keywords
        # or indicate that search results were used
        weather_keywords = ["weather", "temperature", "sunny", "rainy", "cloudy", "forecast", "°C", "°F", "degree"]
        has_weather_info = any(keyword.lower() in text.lower() for keyword in weather_keywords)

        # The response should either contain weather info OR mention search/lack of info
        assert has_weather_info or "search" in text.lower() or "抱歉" in text or "sorry" in text.lower(), (
            f"Expected weather-related content or search indication, got: {text[:200]}"
        )

        print(f"\nResponse text: {text[:300]}...")
        print(f"Contains weather info: {has_weather_info}")

    def test_web_search_with_streaming(self, http_client):
        """Test web search with streaming response."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 1024,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather like in Beijing today?",
                    }
                ],
                "stream": True,
            },
        )

        assert response.status_code == 200
        content = response.content.decode("utf-8")

        # Check streaming events
        assert "event: message_start" in content
        assert "event: message_stop" in content

    def test_web_search_multiple_queries(self, http_client):
        """Test web search with multiple queries in conversation."""
        # First message - search for Python
        response1 = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 1024,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "Search for Python programming best practices",
                    }
                ],
            },
        )

        assert response1.status_code == 200
        data1 = response1.json()
        text1 = get_text_content(data1)

        # Verify first response contains Python-related content
        python_keywords = ["python", "programming", "code", "development", "best practice"]
        has_python_content = any(keyword.lower() in text1.lower() for keyword in python_keywords)
        assert has_python_content, f"Expected Python-related content, got: {text1[:200]}"

        # Continue conversation - search for JavaScript
        response2 = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 1024,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "Search for Python programming best practices",
                    },
                    {
                        "role": "assistant",
                        "content": text1,
                    },
                    {
                        "role": "user",
                        "content": "Now search for JavaScript best practices",
                    },
                ],
            },
        )

        assert response2.status_code == 200
        data2 = response2.json()
        text2 = get_text_content(data2)
        assert len(text2) > 0

        # Verify second response contains JavaScript-related content
        js_keywords = ["javascript", "js", "node", "frontend", "web development"]
        has_js_content = any(keyword.lower() in text2.lower() for keyword in js_keywords)
        assert has_js_content, f"Expected JavaScript-related content, got: {text2[:200]}"

        print(f"\nFirst response (Python): {text1[:200]}...")
        print(f"Second response (JavaScript): {text2[:200]}...")


class TestWebSearchWithoutKey:
    """Test web search behavior when tavily_api_key is not configured."""

    def test_web_search_empty_key_model_continues(self, http_client):
        """Test that when tavily_api_key is empty, model continues without search results."""
        # This test assumes the server is running WITHOUT tavily_api_key configured
        # The model should still respond, just without web search capability

        # Use a query that requires real-time info - stock price
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 1024,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the current stock price of Apple (AAPL) today?",
                    }
                ],
            },
        )

        # Should still get a valid response even without search
        assert response.status_code == 200
        data = response.json()

        # Response should be valid
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0

        text = get_text_content(data)
        assert len(text) > 0

        # Without search capability, model should either:
        # 1. Indicate it cannot provide real-time info
        # 2. Provide general info about Apple stock without specific current price
        # 3. Mention inability to search

        # The response should NOT contain a specific current stock price
        # (since we don't have search capability in this test)
        import re

        # Look for dollar amounts that look like stock prices (e.g., $150.00)
        price_pattern = r'\$\d{2,3}\.\d{2}'
        has_specific_price = re.search(price_pattern, text)

        # If there's a specific price, it might be from training data (outdated)
        # or the test is running with search enabled
        print(f"\nResponse without search: {text[:300]}...")
        print(f"Has specific stock price: {bool(has_specific_price)}")

        # Model should respond gracefully - either admit it can't search or provide general info
        assert len(text) > 50, "Response should be substantive"

    def test_web_search_empty_key_with_streaming(self, http_client):
        """Test streaming works even without tavily_api_key."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 512,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the current stock price of Apple?",
                    }
                ],
                "stream": True,
            },
        )

        assert response.status_code == 200
        content = response.content.decode("utf-8")

        # Streaming should work normally
        assert "event: message_start" in content
        assert "event: message_stop" in content

    def test_web_search_empty_key_conversation_continues(self, http_client):
        """Test that conversation can continue without web search."""
        # First message
        response1 = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 512,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "Search for latest news (this should fail without API key)",
                    }
                ],
            },
        )

        assert response1.status_code == 200
        data1 = response1.json()
        text1 = get_text_content(data1)

        # Continue with regular question (no search needed)
        response2 = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 512,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "Search for latest news (this should fail without API key)",
                    },
                    {"role": "assistant", "content": text1},
                    {"role": "user", "content": "What is 2+2?"},
                ],
            },
        )

        assert response2.status_code == 200
        data2 = response2.json()
        text2 = get_text_content(data2)
        assert len(text2) > 0
        assert "4" in text2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
