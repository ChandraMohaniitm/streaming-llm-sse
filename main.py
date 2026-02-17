"""
Streaming LLM Response Handler using FastAPI with SSE
Works with AIPROXY_TOKEN (aipipe.org)
"""

import os
import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Streaming LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tokens
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


class PromptRequest(BaseModel):
    prompt: str
    stream: bool = True


async def stream_openai_response(prompt: str):
    """Stream response using SSE."""

    if not API_KEY:
        yield 'data: {"error": "API key not configured"}\n\n'
        yield "data: [DONE]\n\n"
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini",  # safer modern model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": 1500,
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{API_BASE}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:

                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f'data: {{"error": "{error_text.decode()}"}}\n\n'
                    yield "data: [DONE]\n\n"
                    return

                # Immediately flush first chunk
                yield 'data: {"status":"started"}\n\n'
                await asyncio.sleep(0)

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

    if request.stream:
        return StreamingResponse(
            stream_openai_response(request.prompt),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # If stream = false (optional support)
    return {"message": "Set stream=true to enable streaming"}


@app.get("/")
async def root():
    return {"status": "ok", "message": "Streaming LLM API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}