"""
bridge_server.py — The "Synapse"
Exposes an OpenAI-compatible endpoint that Open WebUI can consume.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from harmonic_framework import HarmonicFractalFramework
import time

app = FastAPI()
hff = HarmonicFractalFramework()


class ChatRequest(BaseModel):
    model: str
    messages: list


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    user_input = request.messages[-1]["content"]
    # Run the core logic
    resonance = hff.process_resonance(0.5)

    return {
        "id": "hff-01",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": f"Resonance State: {resonance:.2e} | [HFF Active]"
            }
        }]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
