"""Test the full HFF bridge pipeline for tool call handling."""
import httpx
import json

# Send a time question through the full HFF bridge pipeline (non-streaming)
payload = {
    "model": "Alivai",
    "messages": [{"role": "user", "content": "Hey what time is it right now?"}],
    "stream": False,
}

print("Sending request to HFF bridge (non-streaming)...")
try:
    r = httpx.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        timeout=180,
    )
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("STATUS:", r.status_code)
    print("RESPONSE:", content[:1000])
except Exception as e:
    print("ERROR:", e)
