"""
Integration tests for local-openai2anthropic against real backend.

To run these tests:
1. Start the proxy: local-openai2anthropic
2. Set environment variables:
   - TEST_BASE_URL=http://localhost:8080
   - TEST_UPSTREAM_URL=http://10.0.41.2:8000
   - TEST_MODEL=kimi-k2.5
3. Run: pytest tests/test_integration.py -v

Note: These tests require a running backend and may incur API costs.
Mark as skip by default to avoid accidental runs.
"""

import os

import pytest

# Skip all tests in this file by default
# Remove this to enable integration tests
pytestmark = pytest.mark.skip(
    reason="Integration tests - requires running backend. "
    "Set RUN_INTEGRATION_TESTS=1 to enable."
)

# Or use environment variable to control
if not os.getenv("RUN_INTEGRATION_TESTS"):
    pytest.skip("Set RUN_INTEGRATION_TESTS=1 to run integration tests", allow_module_level=True)

import anthropic
import httpx

# Test configuration
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8080")
UPSTREAM_URL = os.getenv("TEST_UPSTREAM_URL", "http://10.0.41.2:8000")
API_KEY = os.getenv("TEST_API_KEY", "dummy-key")
MODEL = os.getenv("TEST_MODEL", "kimi-k2.5")


@pytest.fixture
def client():
    """Create Anthropic client."""
    return anthropic.Anthropic(
        base_url=BASE_URL,
        api_key=API_KEY,
    )


@pytest.fixture
def http_client():
    """Create HTTP client."""
    return httpx.Client(timeout=60.0)


class TestBasicFunctionality:
    """Test basic chat completion."""

    def test_simple_chat(self, client):
        """Test basic chat completion."""
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        
        assert message.id is not None
        assert message.model == MODEL
        assert message.role == "assistant"
        assert len(message.content) > 0
        assert message.content[0].type == "text"
        assert message.content[0].text is not None
        assert message.usage.input_tokens > 0
        assert message.usage.output_tokens > 0

    def test_system_message(self, client):
        """Test system message handling."""
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        
        assert message.content[0].text is not None

    def test_max_tokens(self, client):
        """Test max tokens limit."""
        message = client.messages.create(
            model=MODEL,
            max_tokens=50,
            messages=[{"role": "user", "content": "Write a long story"}],
        )
        
        assert message.usage.output_tokens <= 50

    def test_temperature_and_top_p(self, client):
        """Test temperature and top_p parameters."""
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0.5,
            top_p=0.9,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        
        assert message.content[0].text is not None

    def test_stop_sequences(self, client):
        """Test stop sequences."""
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            stop_sequences=["STOP"],
            messages=[{"role": "user", "content": "Count from 1 to 100"}],
        )
        
        assert message.stop_reason in ["end_turn", "stop_sequence", "max_tokens"]


class TestStreaming:
    """Test streaming functionality."""

    def test_basic_streaming(self, client):
        """Test streaming response."""
        stream = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            stream=True,
        )
        
        chunks = []
        for event in stream:
            chunks.append(event)
            if event.type == "content_block_delta":
                assert hasattr(event.delta, 'text') or hasattr(event.delta, 'partial_json')
        
        assert len(chunks) > 0
        
        # Check final message
        final = stream.get_final_message()
        assert final.id is not None
        assert final.usage.output_tokens > 0

    def test_streaming_events(self, client):
        """Test all streaming event types are received."""
        stream = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        
        event_types = set()
        for event in stream:
            event_types.add(event.type)
        
        # Should have at least message_start and message_stop
        assert "message_start" in event_types
        assert "message_stop" in event_types


class TestToolCalling:
    """Test tool/function calling."""

    def test_tool_use(self, client):
        """Test tool calling."""
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                }
            ],
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        )
        
        # Model may or may not call the tool depending on capabilities
        if message.stop_reason == "tool_use":
            tool_use = message.content[-1]
            assert tool_use.type == "tool_use"
            assert tool_use.name == "get_weather"
            assert "location" in tool_use.input

    def test_tool_result(self, client):
        """Test tool result handling."""
        # First get a tool call
        message1 = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=[
                {
                    "name": "calculate",
                    "description": "Calculate",
                    "input_schema": {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                    },
                }
            ],
            messages=[{"role": "user", "content": "Calculate 1+1"}],
        )
        
        if message1.stop_reason == "tool_use":
            tool_use = message1.content[-1]
            
            # Now send tool result
            messages = [
                {"role": "user", "content": "Calculate 1+1"},
                {"role": "assistant", "content": message1.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": "2",
                        }
                    ],
                },
            ]
            
            message2 = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                messages=messages,
            )
            
            assert message2.content[0].text is not None


class TestThinkingMode:
    """Test thinking/reasoning mode."""

    def test_thinking_enabled(self, client):
        """Test thinking mode enabled."""
        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "enabled"},
            messages=[{"role": "user", "content": "What is 23 * 47?"}],
        )
        
        assert message.content[0].text is not None

    def test_thinking_disabled(self, client):
        """Test thinking mode disabled."""
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": "Hello!"}],
        )
        
        assert message.content[0].text is not None

    def test_thinking_with_budget(self, client):
        """Test thinking with budget_tokens (should be accepted but ignored)."""
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            thinking={"type": "enabled", "budget_tokens": 2048},
            messages=[{"role": "user", "content": "Hello!"}],
        )
        
        assert message.content[0].text is not None


class TestDirectAPI:
    """Test direct HTTP API calls."""

    def test_direct_messages_endpoint(self, http_client):
        """Test POST /v1/messages."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0

    def test_direct_models_endpoint(self, http_client):
        """Test GET /v1/models."""
        response = http_client.get(f"{BASE_URL}/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_direct_streaming(self, http_client):
        """Test streaming via direct API."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        )
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        
        # Read some events
        content = response.content.decode("utf-8")
        assert "event: message_start" in content
        assert "event: message_stop" in content

    def test_error_response_format(self, http_client):
        """Test error responses are in Anthropic format."""
        response = http_client.post(
            f"{BASE_URL}/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "non-existent-model-12345",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        
        # Should get an error response
        if response.status_code != 200:
            data = response.json()
            assert data.get("type") == "error"
            assert "error" in data
            assert "type" in data["error"]
            assert "message" in data["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
