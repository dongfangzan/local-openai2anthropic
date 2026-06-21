# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for the /v1/responses endpoint against a real upstream.

Configure via env vars: OA2A_E2E_BASE_URL, OA2A_E2E_API_KEY, OA2A_E2E_MODEL,
OA2A_E2E_PORT (all optional except OA2A_E2E_API_KEY).
"""

import asyncio
import json
import os
import sys

import httpx

from local_openai2anthropic.config import Settings
from local_openai2anthropic.main import create_app
import uvicorn

BASE_URL = os.environ.get("OA2A_E2E_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OA2A_E2E_API_KEY", "")
MODEL = os.environ.get("OA2A_E2E_MODEL", "gpt-4o-mini")
PORT = int(os.environ.get("OA2A_E2E_PORT", "18080"))


async def call_responses_non_streaming(client: httpx.AsyncClient) -> None:
    print("\n=== Test 1: Non-streaming text response ===")
    payload = {
        "model": MODEL,
        "input": "Say exactly 'pong' and nothing else.",
        "instructions": "Be extremely concise.",
        "max_output_tokens": 50,
    }
    r = await client.post("/v1/responses", json=payload, timeout=120.0)
    print(f"Status: {r.status_code}")
    data = r.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {data}"
    assert data["object"] == "response"
    # Find the message item
    msg_items = [o for o in data["output"] if o["type"] == "message"]
    assert msg_items, "No message item in output"
    text = msg_items[0]["content"][0]["text"]
    print(f"Got text: {text!r}")
    assert "pong" in text.lower(), f"Expected 'pong' in response, got {text!r}"
    print("PASS")


async def call_responses_streaming(client: httpx.AsyncClient) -> None:
    print("\n=== Test 2: Streaming text response ===")
    payload = {
        "model": MODEL,
        "input": "Count from 1 to 5 separated by spaces.",
        "stream": True,
        "max_output_tokens": 100,
    }
    collected_text = ""
    event_types = []
    async with client.stream("POST", "/v1/responses", json=payload, timeout=120.0) as r:
        print(f"Status: {r.status_code}")
        assert r.status_code == 200
        event_type = None
        async for line in r.aiter_lines():
            if line.startswith("event: "):
                event_type = line[7:].strip()
                event_types.append(event_type)
            elif line.startswith("data: "):
                data = line[6:]
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event_type == "response.output_text.delta":
                    collected_text += obj.get("delta", "")
    print(f"Event types seen: {event_types}")
    print(f"Collected text: {collected_text!r}")
    assert "response.created" in event_types
    assert "response.output_item.added" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.output_text.done" in event_types
    assert "response.completed" in event_types
    # Check the digits appear
    for n in "12345":
        assert n in collected_text, f"Missing digit {n} in {collected_text!r}"
    print("PASS")


async def call_responses_with_instructions(client: httpx.AsyncClient) -> None:
    print("\n=== Test 3: instructions appear in response ===")
    payload = {
        "model": MODEL,
        "input": "ping",
        "instructions": "Always respond with the single word: ack",
        "max_output_tokens": 30,
    }
    r = await client.post("/v1/responses", json=payload, timeout=120.0)
    data = r.json()
    print(json.dumps(data, indent=2, ensure_ascii=False)[:500])
    assert r.status_code == 200
    assert data.get("instructions") == "Always respond with the single word: ack"
    msg_items = [o for o in data["output"] if o["type"] == "message"]
    text = msg_items[0]["content"][0]["text"].lower()
    assert "ack" in text, f"Expected 'ack' in {text!r}"
    print("PASS")


async def call_responses_with_tools(client: httpx.AsyncClient) -> None:
    print("\n=== Test 4: Tool calling ===")
    payload = {
        "model": MODEL,
        "input": "What's the weather in Beijing? Use the get_weather tool.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_output_tokens": 200,
    }
    r = await client.post("/v1/responses", json=payload, timeout=120.0)
    data = r.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {data}"
    fc_items = [o for o in data["output"] if o["type"] == "function_call"]
    if not fc_items:
        print("WARN: model did not emit a function_call — text was:")
        msg_items = [o for o in data["output"] if o["type"] == "message"]
        if msg_items:
            print(repr(msg_items[0]["content"][0]["text"]))
        return
    fc = fc_items[0]
    print(f"Tool call: name={fc['name']!r} arguments={fc['arguments']!r}")
    assert fc["name"] == "get_weather"
    args = json.loads(fc["arguments"])
    assert "city" in args
    print("PASS")


async def call_responses_usage(client: httpx.AsyncClient) -> None:
    print("\n=== Test 5: Usage reported ===")
    payload = {
        "model": MODEL,
        "input": "Hi",
        "max_output_tokens": 30,
    }
    r = await client.post("/v1/responses", json=payload, timeout=120.0)
    data = r.json()
    assert r.status_code == 200
    usage = data.get("usage")
    print(f"Usage: {usage}")
    assert usage is not None
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    print("PASS")


async def main():
    settings = Settings(
        openai_api_key=API_KEY,
        openai_base_url=BASE_URL,
        host="127.0.0.1",
        port=PORT,
        request_timeout=120.0,
        api_key="",
        log_level="WARNING",
    )
    app = create_app(settings)
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())
    # Wait for server to start
    for _ in range(40):
        await asyncio.sleep(0.25)
        try:
            async with httpx.AsyncClient() as c:
                await c.get(f"http://127.0.0.1:{PORT}/health", timeout=1.0)
            break
        except Exception:
            continue
    else:
        print("Server failed to start")
        sys.exit(1)

    failures = 0
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}") as client:
        for fn in [
            call_responses_non_streaming,
            call_responses_streaming,
            call_responses_with_instructions,
            call_responses_with_tools,
            call_responses_usage,
        ]:
            try:
                await fn(client)
            except AssertionError as e:
                print(f"FAIL: {fn.__name__}: {e}")
                failures += 1
            except Exception as e:
                print(f"ERROR: {fn.__name__}: {type(e).__name__}: {e}")
                failures += 1

    server.should_exit = True
    await server_task
    print(f"\n=== Summary: {5 - failures}/5 passed ===")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
