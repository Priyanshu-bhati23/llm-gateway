"""Vercel entry point for the gateway.

Serverless caveat: each function instance keeps its own cache, rate-limit buckets and
circuit-breaker state, and instances are recycled. Routing, fallback, auth, PII masking
and streaming behave exactly as documented; the stateful features are best-effort here.
For the full behaviour, run the Docker image on a single long-lived host.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gateway.app import create_app  # noqa: E402
from gateway.config import load_config  # noqa: E402

app = create_app(load_config(os.getenv("GATEWAY_CONFIG", str(ROOT / "deploy" / "vercel.yaml"))))


@app.get("/")
def index() -> dict:
    return {
        "service": "llm-gateway",
        "docs": "https://github.com/Priyanshu-bhati23/llm-gateway",
        "endpoints": ["POST /v1/chat/completions", "GET /v1/models", "GET /healthz", "GET /metrics"],
        "auth": "Authorization: Bearer <gateway client key>",
    }
