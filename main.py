"""
Ultra-Optimized Streaming LLM Response Handler
Designed to minimize first-token latency (<2218ms)
"""

import os
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Streaming LLM API - Optimized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ✅ GLOBAL CLIENT (IMPORTANT FOR PERFORMANCE)
client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0, connect=5.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
    http2=False,  # Avoid extra overhead
)


class PromptRequest(BaseModel):
    prompt: str
    stream: bool = True


async def stream_openai_response(prompt: str):

    if not API_KEY:
        yield 'data: {"error":"API key not configured"}\n\n'
        yield "data: [DONE]\n\n"
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a fast coding assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": 1000,  # reduced for faster first token
        "temperature": 0.5,  # lower = faster sampling
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
                yield f'data: {{"error":"{error_text.decode()}"}}\n\n'
                yield "data: [DONE]\n\n"
                return

            # ✅ Immediately send first chunk
            yield 'data: {"status":"started"}\n\n'

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line[6:]

                if data.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break

                yield f"data: {data}\n\n"

    except httpx.TimeoutException:
        yield 'data: {"error":"Request timed out"}\n\n'
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f'data: {{"error":"{str(e)}"}}\n\n'
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
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}