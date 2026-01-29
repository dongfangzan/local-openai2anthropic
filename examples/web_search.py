#!/usr/bin/env python3
"""
Example: Using Web Search Tool

This example demonstrates how to use the web_search tool with the proxy.
When OA2A_TAVILY_API_KEY is configured, web search requests are intercepted
and executed via Tavily API.

Prerequisites:
    export OA2A_OPENAI_API_KEY=sk-your-openai-key
    export OA2A_TAVILY_API_KEY=tvly-your-tavily-key  # Optional, enables web search

Run the proxy server first:
    python -m local_openai2anthropic.main

Then run this example:
    python examples/web_search.py
"""

import anthropic

# Configure client to use the local proxy
client = anthropic.Anthropic(
    base_url="http://localhost:8080/v1",
    api_key="dummy",  # The proxy handles real authentication
)

# Example with web search tool
response = client.messages.create(
    model="gpt-4o",  # Replace with your OpenAI model
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "What is the latest news about Claude?"
        }
    ],
    tools=[
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3,
        }
    ]
)

print("Response:")
for block in response.content:
    print(f"  Type: {block.type}")
    if block.type == "text":
        print(f"  Text: {block.text[:200]}...")
    elif block.type == "server_tool_use":
        print(f"  Server Tool Use: {block.name} - {block.input}")
    elif block.type == "web_search_tool_result":
        print(f"  Web Search Results: {len(block.content)} results")

print(f"\nUsage: {response.usage}")
