# SPDX-License-Identifier: Apache-2.0
"""Real e2e test for /v1/responses with web_search.

Configure via env vars: OA2A_E2E_BASE_URL, OA2A_E2E_API_KEY, OA2A_E2E_MODEL,
OA2A_E2E_PORT, OA2A_E2E_TAVILY_KEY, OA2A_E2E_TONGXIAO_KEY.
"""

import asyncio
import json
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
PORT = int(os.environ.get("OA2A_E2E_PORT", "18095"))
TAVILY_KEY = os.environ.get("OA2A_E2E_TAVILY_KEY", "")
TONGXIAO_KEY = os.environ.get("OA2A_E2E_TONGXIAO_KEY", "")


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
        request_timeout=180.0,
        api_key="",
        log_level="INFO",
        tavily_api_key=TAVILY_KEY,
        tongxiao_api_key=TONGXIAO_KEY,
        websearch_provider="tongxiao",
        websearch_max_uses=3,
    )
    app = create_app(settings)
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await wait_for_server(PORT)

    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="any", timeout=180.0, max_retries=0)

    print("=== Test 1: non-streaming web search ===")

    def call_non_streaming():
        return client.responses.create(
            model=MODEL,
            input="What is the latest news about OpenAI? Search the web and tell me one recent headline.",
            tools=[{"type": "web_search_preview"}],
            max_output_tokens=400,
        )

    resp = await run_in_thread(call_non_streaming)
    types = [o.type for o in resp.output]
    print(f"output item types: {types}")
    ws_items = [o for o in resp.output if o.type == "web_search_call"]
    msg_items = [o for o in resp.output if o.type == "message"]
    assert ws_items, "No web_search_call item!"
    print(f"web_search_call count: {len(ws_items)}")
    for ws in ws_items:
        print(f"  query={ws.query!r} status={ws.status} results={len(ws.results) if ws.results else 0}")
        if ws.results:
            for r in ws.results[:2]:
                print(f"    - {getattr(r, 'title', '?')}: {getattr(r, 'url', '?')}")
    assert msg_items, "No message item!"
    text = msg_items[0].content[0].text
    print(f"answer: {text[:300]!r}")
    print("PASS")

    print("\n=== Test 2: streaming web search ===")

    def call_streaming():
        events = []
        collected_text = ""
        for event in client.responses.create(
            model=MODEL,
            input="Search the web: who won the most recent FIFA World Cup?",
            tools=[{"type": "web_search_preview"}],
            stream=True,
            max_output_tokens=300,
        ):
            events.append(event.type)
            if event.type == "response.output_text.delta":
                collected_text += event.delta
        return events, collected_text

    events, collected_text = await run_in_thread(call_streaming)
    print(f"event types: {sorted(set(events))}")
    print(f"collected text: {collected_text[:300]!r}")
    assert "response.created" in events
    assert "response.completed" in events
    # web_search_call streaming events
    assert any("web_search" in e for e in events), "No web_search events in stream"
    print("PASS")

    print("\n=== Test 3: web_search with custom max_uses ===")

    def call_max_uses():
        return client.responses.create(
            model=MODEL,
            input="Compare today's weather in Beijing and Shanghai. Search the web for both.",
            tools=[{"type": "web_search_preview", "max_uses": 2}],
            max_output_tokens=400,
        )

    resp = await run_in_thread(call_max_uses)
    ws_items = [o for o in resp.output if o.type == "web_search_call"]
    print(f"web_search_call count: {len(ws_items)} (max_uses=2)")
    # The model should respect max_uses; we just verify it doesn't exceed.
    assert len(ws_items) <= 2, f"max_uses=2 but got {len(ws_items)} calls"
    msg_items = [o for o in resp.output if o.type == "message"]
    if msg_items:
        print(f"answer: {msg_items[0].content[0].text[:300]!r}")
    print("PASS")

    print("\n=== Summary: all web search e2e tests passed ===")
    server.should_exit = True
    await server_task


if __name__ == "__main__":
    asyncio.run(main())
