"""
Tool calling example using Anthropic SDK with local-openai2anthropic proxy.
"""

import json
import anthropic


def get_weather(location: str, unit: str = "celsius") -> str:
    """Simulated weather function."""
    return json.dumps({
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny",
    })


def main():
    client = anthropic.Anthropic(
        base_url="http://localhost:8080",
        api_key="dummy-key",
    )

    # Define tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature",
                    },
                },
                "required": ["location"],
            },
        }
    ]

    # First request - get tool call
    print("=== First Request (Tool Call) ===")
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    
    response = client.messages.create(
        model="gpt-4o",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )
    
    print(f"Stop reason: {response.stop_reason}")
    print(f"Content blocks: {len(response.content)}")
    
    # Handle tool use
    if response.stop_reason == "tool_use":
        tool_use = response.content[-1]  # Last content block is tool_use
        print(f"Tool called: {tool_use.name}")
        print(f"Tool input: {tool_use.input}")
        
        # Execute the tool
        tool_result = get_weather(**tool_use.input)
        print(f"Tool result: {tool_result}")
        
        # Add to conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": tool_result,
                }
            ],
        })
        
        # Second request - get final response
        print("\n=== Second Request (Final Response) ===")
        final_response = client.messages.create(
            model="gpt-4o",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )
        
        print(f"Final response: {final_response.content[0].text}")
        print(f"Stop reason: {final_response.stop_reason}")


if __name__ == "__main__":
    main()
