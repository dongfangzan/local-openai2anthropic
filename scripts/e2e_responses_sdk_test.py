# SPDX-License-Identifier: Apache-2.0
"""Verify the /v1/responses endpoint works with the official openai SDK.

The openai SDK uses a synchronous httpx client, so SDK calls are run in a
worker thread while uvicorn runs on the asyncio event loop.

Configure via env vars: OA2A_E2E_BASE_URL, OA2A_E2E_API_KEY, OA2A_E2E_MODEL,
OA2A_E2E_PORT.
"""

import asyncio
import os
import threading

import httpx
import uvicorn

from local_openai2anthropic.config import Settings
from local_openai2anthropic.main import create_app
from openai import OpenAI

BASE_URL = os.environ.get("OA2A_E2E_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OA2A_E2E_API_KEY", "")
MODEL = os.environ.get("OA2A_E2E_MODEL", "gpt-4o-mini")
PORT = int(os.environ.get("OA2A_E2E_PORT", "18090"))


async def wait_for_server(port: int) -> None:
    for _ in range(40):
        await asyncio.sleep(0.25)
        try:
            async with httpx.AsyncClient() as c:
                await c.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            return
        except Exception:
            continue
    raise RuntimeError("server failed to start")


def run_in_thread(fn):
    """Run a sync function in a thread, letting the asyncio loop run."""
    result: dict = {}

    def wrapper():
        try:
            result["value"] = fn()
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=wrapper)
    t.start()

    async def waiter():
        while t.is_alive():
            await asyncio.sleep(0.05)
        t.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    return waiter()


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
    await wait_for_server(PORT)

    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="any", timeout=60.0, max_retries=0)

    print("=== SDK: non-streaming responses.create ===")

    def call_non_streaming():
        return client.responses.create(
            model=MODEL,
            input="Reply with exactly: hello sdk",
            max_output_tokens=60,
        )

    resp = await run_in_thread(call_non_streaming)
    print(f"id={resp.id}")
    print(f"output_text={resp.output_text!r}")
    assert "hello sdk" in resp.output_text.lower(), f"Got: {resp.output_text!r}"
    print(f"usage: in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
    print("PASS")

    print("\n=== SDK: streaming responses.create ===")

    def call_streaming():
        collected = ""
        event_types: set[str] = set()
        for event in client.responses.create(
            model=MODEL,
            input="Say 'streaming works' and nothing else.",
            stream=True,
            max_output_tokens=60,
        ):
            event_types.add(event.type)
            if event.type == "response.output_text.delta":
                collected += event.delta
        return collected, event_types

    collected, event_types = await run_in_thread(call_streaming)
    print(f"Events seen: {sorted(event_types)}")
    print(f"Collected: {collected!r}")
    assert "response.created" in event_types
    assert "response.completed" in event_types
    assert "streaming works" in collected.lower(), f"Got: {collected!r}"
    print("PASS")

    print("\n=== SDK: tool calling ===")

    def call_tools():
        return client.responses.create(
            model=MODEL,
            input="What's the weather in Shanghai? Use the get_weather tool.",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            max_output_tokens=200,
        )

    resp = await run_in_thread(call_tools)
    fc_items = [o for o in resp.output if o.type == "function_call"]
    print(f"function_call items: {len(fc_items)}")
    if fc_items:
        fc = fc_items[0]
        print(f"name={fc.name} arguments={fc.arguments} call_id={fc.call_id}")
        assert fc.name == "get_weather"
        assert "Shanghai" in fc.arguments
        print("PASS")
    else:
        print(f"output_text: {resp.output_text!r}")
        print("WARN: no function_call emitted")

    print("\n=== SDK: multi-turn with function_call_output ===")

    def call_multiturn():
        # First turn: ask for a tool call
        r1 = client.responses.create(
            model=MODEL,
            input="What's the weather in Beijing? Use the get_weather tool.",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            max_output_tokens=200,
        )
        fc = next((o for o in r1.output if o.type == "function_call"), None)
        if not fc:
            return r1, None, None
        # Second turn: feed the tool output back
        r2 = client.responses.create(
            model=MODEL,
            input=[
                {"role": "user", "content": "What's the weather in Beijing? Use the get_weather tool."},
                fc,  # the function_call item from the previous turn
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": "sunny, 25C",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            max_output_tokens=200,
        )
        return r1, fc, r2

    r1, fc, r2 = await run_in_thread(call_multiturn)
    if fc:
        print(f"Turn 1 tool call: name={fc.name} args={fc.arguments}")
        assert fc.name == "get_weather"
        assert "Beijing" in fc.arguments
        # Turn 2 should summarize
        print(f"Turn 2 output_text: {r2.output_text!r}")
        assert any(w in r2.output_text.lower() for w in ("sunny", "25", "beijing")), (
            f"Expected weather info in turn 2, got: {r2.output_text!r}"
        )
        print("PASS")
    else:
        print(f"Turn 1 did not emit a function_call; output: {r1.output_text!r}")
        print("WARN: skipping multi-turn assertion")

    server.should_exit = True
    await server_task


if __name__ == "__main__":
    asyncio.run(main())
