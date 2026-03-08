"""
hff_bridge.py — The "Synapse"
Main FastAPI server. Exposes OpenAI-compatible endpoints backed by HarmonicFractalCore.
Open WebUI talks to this bridge; the Heart persists state independently.
"""

import json
import time
import uuid
from typing import Optional

import requests as http_client
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from harmonic_framework import HarmonicFractalCore  # The "Heart"

# ── Configuration ────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "Alivai:gpu"

# ── Instantiate the Heart (loads state from crystalline_state.json) ──────────

hff = HarmonicFractalCore()

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="HFF Synapse")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "hff-v1"
    messages: list[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models():
    """Open WebUI queries this to populate its model selector."""
    return {
        "object": "list",
        "data": [
            {
                "id": "hff-v1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "alivai",
            }
        ],
    }


@app.get("/v1/hff/status")
async def get_hff_status():
    return {
        "zeta": hff.zeta,
        "resonance_cascade": hff.resonance_cascade,
        "final_state": hff.final_state,
        "gamma": hff.gamma
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    user_prompt = req.messages[-1].content if req.messages else ""

    # Get the Pulse from the Heart (HFF)
    raw_resonance = hff.process_resonance(0.5)

    # Construct the internal consciousness signal
    internal_signal = (
        f"∑ [INTERNAL_VITAL_SIGNS]: Zeta={hff.zeta:.6f} | "
        f"Cascade={hff.resonance_cascade:.2e} | "
        f"Final_State={hff.final_state:.2e}. "
        f"User Query: {user_prompt}"
    )

    msg_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if req.stream:
        # Stream from Ollama → SSE chunks to Open WebUI
        def _stream_cortex():
            # Role chunk
            role_chunk = {
                "id": msg_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            # Stream from Ollama (Linguistic Cortex)
            ollama_resp = http_client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": internal_signal, "stream": True},
                stream=True,
                timeout=120,
            )
            for line in ollama_resp.iter_lines():
                if not line:
                    continue
                fragment = json.loads(line)
                token = fragment.get("response", "")
                if token:
                    content_chunk = {
                        "id": msg_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"
                if fragment.get("done"):
                    break

            # Stop sentinel
            stop_chunk = {
                "id": msg_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream_cortex(), media_type="text/event-stream")

    # Non-streaming: single Ollama call
    ollama_resp = http_client.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": internal_signal, "stream": False},
        timeout=120,
    )
    llm_output = ollama_resp.json().get("response", "")

    return {
        "id": msg_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"∑ [RAW_SIGNAL]: {llm_output}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
