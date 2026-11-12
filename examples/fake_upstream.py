"""A stand-in OpenAI-compatible server for trying the gateway without API keys.

    uvicorn examples.fake_upstream:app --port 9001

Set FAIL=1 to make every request return 503, to watch the gateway fall back.
"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
NAME = os.getenv("UPSTREAM_NAME", "fake")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    if os.getenv("FAIL"):
        return JSONResponse({"error": {"message": "simulated outage"}}, status_code=503)
    body = await request.json()
    question = body["messages"][-1]["content"]
    text = f"[{NAME}/{body['model']}] You asked: {question}"
    if body.get("stream"):

        async def events():
            for word in text.split(" "):
                yield f"data: {json.dumps({'choices': [{'delta': {'content': word + ' '}}]})}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": body["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(question) // 4, "completion_tokens": len(text) // 4},
    }
