"""
Streaming example using Anthropic SDK with local-openai2anthropic proxy.
"""

import anthropic


def main():
    client = anthropic.Anthropic(
        base_url="http://localhost:8080",
        api_key="dummy-key",
    )

    print("=== Streaming Response ===")
    print("Assistant: ", end="", flush=True)
    
    with client.messages.stream(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Count from 1 to 10 slowly"}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    
    print()  # New line after stream
    print()
    
    # Access final message
    print(f"\nFinal message id: {stream.get_final_message().id}")
    print(f"Total tokens: {stream.get_final_message().usage}")
    print()

    # Raw event stream
    print("=== Raw Event Stream ===")
    stream = client.messages.create(
        model="gpt-4o-mini",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hi"}],
        stream=True,
    )
    
    for event in stream:
        print(f"Event type: {event.type}")
        if event.type == "content_block_delta":
            print(f"  Text: {event.delta.text}")


if __name__ == "__main__":
    main()
