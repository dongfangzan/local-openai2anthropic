"""
Basic chat example using Anthropic SDK with local-openai2anthropic proxy.

Run this after starting the proxy server:
    local-openai2anthropic
"""

import anthropic


def main():
    # Connect to local proxy instead of Anthropic's API
    client = anthropic.Anthropic(
        base_url="http://localhost:8080",
        api_key="dummy-key",  # Not validated but required by SDK
    )

    # Simple chat
    print("=== Basic Chat ===")
    message = client.messages.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello! What's your name?"}],
    )
    print(f"Response: {message.content[0].text}")
    print(f"Model: {message.model}")
    print(f"Usage: {message.usage.input_tokens} input, {message.usage.output_tokens} output tokens")
    print()

    # Multi-turn conversation
    print("=== Multi-turn Conversation ===")
    messages = [
        {"role": "user", "content": "My name is Alice."},
    ]
    
    response1 = client.messages.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=messages,
    )
    print(f"Assistant: {response1.content[0].text}")
    
    messages.append({"role": "assistant", "content": response1.content[0].text})
    messages.append({"role": "user", "content": "What's my name?"})
    
    response2 = client.messages.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=messages,
    )
    print(f"Assistant: {response2.content[0].text}")


if __name__ == "__main__":
    main()
