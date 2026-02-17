"""
Final Optimized Streaming LLM API
- Fast first token
- High throughput
- Valid JSON streaming
- Proper error handling
- Railway compatible
"""

import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Streaming LLM API - Final")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if AIPROXY_TOKEN:
    API_KEY = AIPROXY_TOKEN
    API_BASE = "https://aipipe.org/openai/v1"
elif OPENAI_API_KEY:
    API_KEY = OPENAI_API_KEY
    API_BASE = "https://api.openai.com/v1"
else:
    API_KEY = None

# ✅ Global HTTP client (critical for speed)
client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0, connect=5.0),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    http2=False,
)

class PromptRequest(BaseModel):
    prompt: str
    stream: bool = True


async def stream_openai_response(prompt: str):
    """Efficient SSE streaming from OpenAI-compatible API."""

    if not API_KEY:
        yield f"data: {json.dumps({'error': 'API key not configured'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini-2024-07-18",  # fast pinned model
        "messages": [
            {"role": "system", "content": "You are a fast coding assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": 800,      # optimized for throughput
        "temperature": 0.2,     # faster sampling
    }

    try:
        async with client.stream(
            "POST",
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:

            if response.status_code != 200:
                error_text = await response.aread()
                yield f"data: {json.dumps({'error': error_text.decode()})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Immediately send first event
            yield f"data: {json.dumps({'status': 'started'})}\n\n"

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line[6:]

                if data.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break

                try:
                    parsed = json.loads(data)
                    delta = parsed["choices"][0]["delta"].get("content", "")
                    if delta:
                        # Proper JSON encoding
                        yield f"data: {json.dumps({'content': delta})}\n\n"
                except Exception:
                    continue

    except httpx.TimeoutException:
        yield f"data: {json.dumps({'error': 'Request timed out'})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/stream")
async def stream_llm_response(request: PromptRequest):
    return StreamingResponse(
        stream_openai_response(request.prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def root():
    return {"status": "ok", "message": "Streaming LLM API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}