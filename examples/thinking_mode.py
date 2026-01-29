"""
Thinking mode example using Anthropic SDK with local-openai2anthropic proxy.

This example shows how to use the thinking parameter to control
reasoning mode in supported models (e.g., DeepSeek-R1 via vLLM/SGLang).
"""

import anthropic


def main():
    # Connect to local proxy
    client = anthropic.Anthropic(
        base_url="http://localhost:8080",
        api_key="dummy-key",
    )

    # Example 1: Enable thinking mode
    print("=== Thinking Mode Enabled ===")
    message = client.messages.create(
        model="deepseek-r1",  # A reasoning-capable model
        max_tokens=4096,
        thinking={
            "type": "enabled",
            # budget_tokens is accepted for API compatibility but not used
            # since vLLM/SGLang don't support fine-grained budget control
            "budget_tokens": 2048,
        },
        messages=[
            {"role": "user", "content": "Solve this step by step: What is 15 * 23?"}
        ],
    )
    
    # Note: The actual reasoning content depends on the upstream model
    # Some models (like DeepSeek-R1) will output thinking in <think> tags
    print(f"Response: {message.content[0].text}")
    print(f"Usage: {message.usage.input_tokens} input, {message.usage.output_tokens} output tokens")
    if hasattr(message.usage, 'cache_creation_input_tokens'):
        print(f"Cache creation: {message.usage.cache_creation_input_tokens}")
    if hasattr(message.usage, 'cache_read_input_tokens'):
        print(f"Cache read: {message.usage.cache_read_input_tokens}")
    print()

    # Example 2: Disable thinking mode (default behavior)
    print("=== Thinking Mode Disabled ===")
    message = client.messages.create(
        model="deepseek-r1",
        max_tokens=4096,
        thinking={
            "type": "disabled",
        },
        messages=[
            {"role": "user", "content": "What is 15 * 23?"}
        ],
    )
    
    print(f"Response: {message.content[0].text}")
    print(f"Usage: {message.usage.input_tokens} input, {message.usage.output_tokens} output tokens")
    print()

    # Example 3: Streaming with thinking
    print("=== Streaming with Thinking ===")
    print("Assistant: ", end="", flush=True)
    
    stream = client.messages.create(
        model="deepseek-r1",
        max_tokens=4096,
        thinking={"type": "enabled"},
        messages=[
            {"role": "user", "content": "Explain the Pythagorean theorem"}
        ],
        stream=True,
    )
    
    for event in stream:
        if event.type == "content_block_delta":
            if hasattr(event.delta, 'text'):
                print(event.delta.text, end="", flush=True)
    
    print("\n")
    print(f"Final usage: {stream.get_final_message().usage}")


if __name__ == "__main__":
    main()
