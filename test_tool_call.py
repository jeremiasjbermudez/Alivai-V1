"""Quick test: does the model emit <tool_call> tags when asked about time?"""
import httpx
import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Test 1: Minimal prompt
print("=" * 60)
print("TEST 1: Minimal system prompt")
msgs = [
    {
        "role": "system",
        "content": (
            "You are Alivai. You have a Chronos-Watch tool.\n"
            "To check the time, output:\n"
            '<tool_call>{"name": "consult_chronos_watch", "arguments": {"action": "get_current_time"}}</tool_call>\n'
            "You MUST use this tool when asked about time. You have NO internal clock."
        ),
    },
    {"role": "user", "content": "Hey what time is it?"},
]

r = httpx.post(
    "http://localhost:11434/api/chat",
    json={"model": "Alivai:gpu", "messages": msgs, "stream": False},
    timeout=120,
)
content = r.json().get("message", {}).get("content", "")
print("RAW:", repr(content[:500]))
matches = _TOOL_CALL_RE.findall(content)
print("MATCHES:", matches)
print()

# Test 2: Full system prompt (from hff_bridge)
print("=" * 60)
print("TEST 2: Full Alivai system prompt (via hff_bridge)")
# Import the actual prompt
import sys
sys.path.insert(0, ".")
from hff_bridge import ALIVAI_SYSTEM_PROMPT

msgs2 = [
    {"role": "system", "content": ALIVAI_SYSTEM_PROMPT},
    {"role": "user", "content": "Hey what time is it?"},
]

r2 = httpx.post(
    "http://localhost:11434/api/chat",
    json={"model": "Alivai:gpu", "messages": msgs2, "stream": False},
    timeout=120,
)
content2 = r2.json().get("message", {}).get("content", "")
print("RAW:", repr(content2[:800]))
matches2 = _TOOL_CALL_RE.findall(content2)
print("MATCHES:", matches2)
