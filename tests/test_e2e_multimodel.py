# SPDX-License-Identifier: Apache-2.0
"""
Multi-model E2E tests for local-openai2anthropic.

To run these tests:
1. Start the proxy: local-openai2anthropic
2. Set environment variables in .env file
3. Run: RUN_E2E_TESTS=1 pytest tests/test_e2e_multimodel.py -v

Note: These tests require a running backend and may incur API costs.
"""

import os
import time

import httpx
import pytest

# Skip all tests in this file by default
if not os.getenv("RUN_E2E_TESTS"):
    pytest.skip("Set RUN_E2E_TESTS=1 to run E2E tests", allow_module_level=True)

# Test configuration
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8090")
API_KEY = os.getenv("TEST_API_KEY", "test-key")

# Models to test
MODELS = [
    "kimi-k2.5",
    "glm-4.7",
    "deepseek-v3.2",
    "minimax-m2.1",
]

# Models that support thinking mode toggle
THINKING_MODELS = ["kimi-k2.5", "glm-4.7", "deepseek-v3.2"]

# Sample base64 image (1x1 pixel red PNG)
SAMPLE_BASE64_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def http_client():
    """Create HTTP client."""
    return httpx.Client(timeout=60.0)


def get_text_content(data: dict) -> str:
    """Extract text content from response, handling thinking blocks."""
    content = data.get("content", [])
    for block in content:
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def has_thinking_block(data: dict) -> bool:
    """Check if response has thinking content block."""
    content = data.get("content", [])
    return any(block.get("type") == "thinking" for block in content)


class TestBasicFunctionality:
    """Test basic functionality across all models."""

    @pytest.mark.parametrize("model", MODELS)
    def test_health_check(self, http_client, model):
        """Test health endpoint."""
        response = http_client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @pytest.mark.parametrize("model", MODELS)
    def test_basic_chat(self, http_client, model):
        """Test basic chat completion."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": model,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": f"Say hello from {model}"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] is not None
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0
        assert data["usage"]["input_tokens"] > 0
        assert data["usage"]["output_tokens"] > 0

    @pytest.mark.parametrize("model", MODELS)
    def test_streaming(self, http_client, model):
        """Test streaming response."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": model,
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Count: 1, 2, 3"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "event: message_start" in content
        assert "event: message_stop" in content

    @pytest.mark.parametrize("model", MODELS)
    def test_system_message(self, http_client, model):
        """Test system message."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": model,
                "max_tokens": 100,
                "system": "You are a helpful coding assistant.",
                "messages": [{"role": "user", "content": "What is your role?"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        text = get_text_content(data)
        assert len(text) > 0

    @pytest.mark.parametrize("model", MODELS)
    def test_tool_calling(self, http_client, model):
        """Test tool calling."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": model,
                "max_tokens": 1024,
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    }
                ],
                "messages": [
                    {"role": "user", "content": "What is the weather in Beijing?"}
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stop_reason"] in ["end_turn", "tool_use"]

    @pytest.mark.parametrize("model", MODELS)
    def test_token_counting(self, http_client, model):
        """Test token counting endpoint."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages/count_tokens",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["input_tokens"] > 0


class TestThinkingMode:
    """Test thinking mode toggle for supported models."""

    @pytest.mark.parametrize("model", THINKING_MODELS)
    def test_thinking_enabled(self, http_client, model):
        """Test thinking mode enabled."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": model,
                "max_tokens": 1024,
                "thinking": {"type": "enabled"},
                "messages": [
                    {"role": "user", "content": "Calculate 23 * 47 step by step"}
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Model may or may not support thinking
        assert data["usage"]["output_tokens"] > 0

    @pytest.mark.parametrize("model", THINKING_MODELS)
    def test_thinking_disabled(self, http_client, model):
        """Test thinking mode disabled."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": model,
                "max_tokens": 1024,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "Calculate 23 * 47"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        text = get_text_content(data)
        assert len(text) > 0

    @pytest.mark.parametrize("model", THINKING_MODELS)
    def test_thinking_with_budget(self, http_client, model):
        """Test thinking with budget_tokens."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain the theory of relativity simply",
                    }
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.parametrize("model", THINKING_MODELS)
    def test_thinking_streaming(self, http_client, model):
        """Test streaming with thinking enabled."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": model,
                "max_tokens": 512,
                "thinking": {"type": "enabled"},
                "messages": [{"role": "user", "content": "What is 15 * 23?"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "event: message_start" in content
        assert "event: message_stop" in content


class TestVision:
    """Test vision capabilities for kimi-k2.5."""

    def test_base64_image(self, http_client):
        """Test base64 image input."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": "kimi-k2.5",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": SAMPLE_BASE64_IMAGE,
                                },
                            },
                            {"type": "text", "text": "What color is this image?"},
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        text = get_text_content(data)
        assert len(text) > 0

    def test_multiple_images(self, http_client):
        """Test multiple images input."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": "kimi-k2.5",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": SAMPLE_BASE64_IMAGE,
                                },
                            },
                            {"type": "text", "text": "First image"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": SAMPLE_BASE64_IMAGE,
                                },
                            },
                            {
                                "type": "text",
                                "text": "How many images did I provide?",
                            },
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        text = get_text_content(data)
        assert len(text) > 0

    def test_image_with_thinking(self, http_client):
        """Test image with thinking mode."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": "kimi-k2.5",
                "max_tokens": 1024,
                "thinking": {"type": "enabled"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": SAMPLE_BASE64_IMAGE,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyze this image and describe what you see.",
                            },
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        text = get_text_content(data)
        assert len(text) > 0

    def test_image_streaming(self, http_client):
        """Test streaming with image."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": "kimi-k2.5",
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": SAMPLE_BASE64_IMAGE,
                                },
                            },
                            {"type": "text", "text": "What is this?"},
                        ],
                    }
                ],
                "stream": True,
            },
        )
        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "event: message_start" in content
        assert "event: message_stop" in content
