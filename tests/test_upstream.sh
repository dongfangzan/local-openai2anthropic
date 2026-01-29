#!/bin/bash
# Test upstream API with different models and thinking settings

set -a
source .env
set +a

API_URL="${OA2A_OPENAI_BASE_URL}/chat/completions"

echo "=== Test 1: deepseek-v3.2 with thinking enabled ==="
curl -s -X POST "${API_URL}" \
  -H "Authorization: Bearer ${OA2A_OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3.2",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 1000,
    "chat_template_kwargs": {"thinking": true}
  }' | jq -C '.choices[0].message | {content: .content[:50], reasoning_content: .reasoning_content}'

echo ""
echo "=== Test 2: kimi-k2.5 with thinking enabled ==="
curl -s -X POST "${API_URL}" \
  -H "Authorization: Bearer ${OA2A_OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 1000,
    "chat_template_kwargs": {"thinking": true}
  }' | jq -C '.choices[0].message | {content: .content[:50], reasoning_content: .reasoning_content[:50]}'

echo ""
echo "=== Test 3: deepseek-v3.2 without thinking ==="
curl -s -X POST "${API_URL}" \
  -H "Authorization: Bearer ${OA2A_OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3.2",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 1000
  }' | jq -C '.choices[0].message | {content: .content[:50], reasoning_content: .reasoning_content}'

echo ""
echo "=== Test 4: kimi-k2.5 without thinking ==="
curl -s -X POST "${API_URL}" \
  -H "Authorization: Bearer ${OA2A_OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 1000
  }' | jq -C '.choices[0].message | {content: .content[:50], reasoning_content: .reasoning_content}'

echo ""
echo "=== Test 5: deepseek-v3.2 streaming with thinking ==="
curl -s -X POST "${API_URL}" \
  -H "Authorization: Bearer ${OA2A_OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3.2",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    "stream": true,
    "chat_template_kwargs": {"thinking": true}
  }' | head -10

echo ""
echo "=== Test 6: kimi-k2.5 streaming with thinking ==="
curl -s -X POST "${API_URL}" \
  -H "Authorization: Bearer ${OA2A_OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    "stream": true,
    "chat_template_kwargs": {"thinking": true}
  }' | head -10
