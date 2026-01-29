"""
Vision/multimodal example using Anthropic SDK with local-openai2anthropic proxy.
"""

import base64
from pathlib import Path

import anthropic


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    client = anthropic.Anthropic(
        base_url="http://localhost:8080",
        api_key="dummy-key",
    )

    # You can use any image file
    # For this example, create a simple test or use an existing image
    image_path = input("Enter path to an image (or press Enter to skip): ").strip()
    
    if not image_path or not Path(image_path).exists():
        print("No valid image provided. Exiting.")
        return
    
    # Detect media type from extension
    ext = Path(image_path).suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    
    # Encode image
    image_data = encode_image(image_path)
    
    print(f"=== Vision Request ({media_type}) ===")
    message = client.messages.create(
        model="gpt-4o",  # Vision-capable model
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                ],
            }
        ],
    )
    
    print(f"Response: {message.content[0].text}")
    print(f"Usage: {message.usage.input_tokens} input, {message.usage.output_tokens} output tokens")


if __name__ == "__main__":
    main()
