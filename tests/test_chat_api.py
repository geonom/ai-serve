"""
Chat API test — runs against a live backend on localhost:8030.

Usage:
    python tests/test_chat_api.py [model_name]

If model_name is omitted, the first available model is used.
Each test prints PASS/FAIL and exits with a non-zero code on first failure.
"""

import asyncio
import json
import sys
import httpx

BASE_URL = "http://localhost:8030"
TIMEOUT = 180.0


def fail(msg: str):
    print(f"  FAIL: {msg}")
    sys.exit(1)


def ok(msg: str):
    print(f"  PASS: {msg}")


async def pick_model(client: httpx.AsyncClient, override: str = None) -> str:
    resp = await client.get(f"{BASE_URL}/api/models")
    assert resp.status_code == 200, f"/api/models returned {resp.status_code}"
    models = resp.json().get("models", [])
    if not models:
        fail("No GGUF models found — check LLAMA_MODEL_DIRS")
    if override:
        matches = [m for m in models if override.lower() in m["name"].lower()]
        if not matches:
            fail(f"No model matching '{override}' found. Available: {[m['name'] for m in models]}")
        return matches[0]["name"]
    return models[0]["name"]


async def test_basic_chat(client: httpx.AsyncClient, model: str):
    print("\n[1] Basic chat — /api/chat/completions")
    resp = await client.post(f"{BASE_URL}/api/chat/completions", json={
        "model_name": model,
        "messages": [{"role": "user", "content": "Reply with the single word: HELLO"}],
        "temperature": 0.0,
        "max_tokens": 32,
        "thinking": False,
    })
    if resp.status_code != 200:
        fail(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    content = data.get("content", "")
    if not content:
        fail(f"content is empty. Full response: {json.dumps(data, indent=2)}")
    ok(f"content = {repr(content[:120])}")


async def test_thinking_false_non_empty(client: httpx.AsyncClient, model: str):
    print("\n[2] thinking=False must return non-empty content")
    resp = await client.post(f"{BASE_URL}/api/chat/completions", json={
        "model_name": model,
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
        "temperature": 0.0,
        "max_tokens": 16,
        "thinking": False,
    })
    if resp.status_code != 200:
        fail(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    content = data.get("content", "").strip()
    thinking = data.get("thinking", "").strip()
    if not content:
        fail(f"content is empty (thinking field has {len(thinking)} chars). Model put everything in reasoning_content but fallback did not work.")
    ok(f"content = {repr(content)}, thinking field empty = {not thinking}")


async def test_thinking_true(client: httpx.AsyncClient, model: str):
    print("\n[3] thinking=True — content must be non-empty (reasoning field is a bonus)")
    resp = await client.post(f"{BASE_URL}/api/chat/completions", json={
        "model_name": model,
        "messages": [{"role": "user", "content": "What is 7 * 8? Answer with just the number."}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "thinking": True,
    })
    if resp.status_code != 200:
        fail(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    content = data.get("content", "").strip()
    if not content:
        fail(f"content is empty. Full response: {json.dumps(data, indent=2)}")
    ok(f"content = {repr(content[:80])}, has thinking = {bool(data.get('thinking'))}")


async def test_structured_output(client: httpx.AsyncClient, model: str):
    print("\n[4] Structured JSON output via response_format")
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
        },
        "required": ["city", "country"],
        "additionalProperties": False,
    }
    resp = await client.post(f"{BASE_URL}/api/chat/completions", json={
        "model_name": model,
        "messages": [{"role": "user", "content": "Give me the capital of France as JSON."}],
        "temperature": 0.0,
        "max_tokens": 128,
        "thinking": False,
        "response_format": schema,
        "retries": 2,
    })
    if resp.status_code != 200:
        fail(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    parsed = data.get("data")
    if not parsed or "city" not in parsed:
        fail(f"data field missing or malformed: {json.dumps(data, indent=2)}")
    ok(f"parsed JSON: {parsed}")


async def test_openai_compat_endpoint(client: httpx.AsyncClient, model: str):
    print("\n[5] OpenAI-compatible /v1/chat/completions")
    resp = await client.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with the single word: SUCCESS"},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
        "thinking": False,
    })
    if resp.status_code != 200:
        fail(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    # Must conform to OpenAI response shape
    assert "choices" in data, f"Missing 'choices': {data}"
    assert "usage" in data, f"Missing 'usage': {data}"
    content = data["choices"][0]["message"]["content"]
    if not content:
        fail(f"choices[0].message.content is empty. Full response: {json.dumps(data, indent=2)}")
    ok(f"content = {repr(content)}")


async def test_system_prompt(client: httpx.AsyncClient, model: str):
    print("\n[6] system_prompt field is injected correctly")
    resp = await client.post(f"{BASE_URL}/api/chat/completions", json={
        "model_name": model,
        "messages": [{"role": "user", "content": "What is your name?"}],
        "system_prompt": "Your name is TestBot. Always introduce yourself as TestBot.",
        "temperature": 0.0,
        "max_tokens": 64,
        "thinking": False,
    })
    if resp.status_code != 200:
        fail(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    content = data.get("content", "")
    if not content:
        fail(f"content is empty. Full response: {json.dumps(data, indent=2)}")
    ok(f"content = {repr(content[:120])}")


async def main():
    model_override = sys.argv[1] if len(sys.argv) > 1 else None

    print("=== Chat API Test Suite ===")
    print(f"Backend: {BASE_URL}")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Health check
        try:
            r = await client.get(f"{BASE_URL}/api/status")
            r.raise_for_status()
        except Exception as e:
            fail(f"Backend not reachable at {BASE_URL}: {e}")

        model = await pick_model(client, model_override)
        print(f"Model: {model}")

        await test_basic_chat(client, model)
        await test_thinking_false_non_empty(client, model)
        await test_thinking_true(client, model)
        await test_structured_output(client, model)
        await test_openai_compat_endpoint(client, model)
        await test_system_prompt(client, model)

    print("\n=== All tests PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
