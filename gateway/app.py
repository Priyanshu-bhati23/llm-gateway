"""The gateway application: auth -> rate limit -> (redact) -> cache -> route with fallback."""

from __future__ import annotations

import hmac
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from . import __version__
from .cache import ResponseCache, cacheable
from .config import Client, Config, Target, load_config
from .limits import CircuitBreaker, RateLimiter
from .metrics import Metrics
from .redact import redact_messages

# Failures another upstream might not share. 401/403 are included: they mean *our* key for
# that upstream is bad, which is the gateway's problem, not the caller's.
FALLBACK_STATUS = frozenset({401, 403, 408, 409, 425, 429, 500, 502, 503, 504, 529})
_HOP_HEADERS = ("content-type", "retry-after")


def error(status: int, message: str, code: str, headers: dict | None = None) -> JSONResponse:
    """Errors use the OpenAI error envelope so existing client SDKs surface them properly."""
    body = {"error": {"message": message, "type": "gateway_error", "code": code}}
    return JSONResponse(body, status_code=status, headers=headers)


def create_app(
    config: Config,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    upstreams = {
        name: httpx.AsyncClient(
            base_url=up.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {up.api_key}"} if up.api_key else {},
            timeout=up.timeout_seconds,
            transport=transport,
        )
        for name, up in config.upstreams.items()
    }
    cache = ResponseCache(
        config.cache.ttl_seconds,
        config.cache.max_entries,
        config.cache.semantic,
        config.cache.semantic_threshold,
        clock,
    )
    limiter = RateLimiter(clock)
    breaker = CircuitBreaker(config.breaker.failure_threshold, config.breaker.cooldown_seconds, clock)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for client in upstreams.values():
            await client.aclose()

    app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan)
    app.state.cache, app.state.breaker, app.state.metrics = cache, breaker, metrics

    def authenticate(request: Request) -> Client | None:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            return None
        # Compare against every client in constant time so timing does not reveal which keys exist.
        match = None
        for client in config.clients:
            if client.key and hmac.compare_digest(token.encode(), client.key.encode()):
                match = client
        return match

    async def attempt(target: Target, body: dict, stream: bool) -> httpx.Response:
        payload = {**body, "model": target.model}
        request = upstreams[target.upstream].build_request("POST", "/chat/completions", json=payload)
        return await upstreams[target.upstream].send(request, stream=stream)

    async def route(alias: str, body: dict, stream: bool) -> tuple[httpx.Response | None, Target | None, list[dict]]:
        """Try each target in order. Returns the first usable response and what was tried."""
        tried: list[dict] = []
        for target in config.routes[alias]:
            if not breaker.available(target.upstream):
                tried.append({"upstream": target.upstream, "skipped": "circuit open"})
                continue
            try:
                response = await attempt(target, body, stream)
            except httpx.HTTPError as exc:
                breaker.record_failure(target.upstream)
                metrics.inc("gateway_upstream_errors_total", upstream=target.upstream)
                tried.append({"upstream": target.upstream, "error": type(exc).__name__})
                continue
            if response.status_code in FALLBACK_STATUS:
                await response.aclose()
                breaker.record_failure(target.upstream)
                metrics.inc("gateway_upstream_errors_total", upstream=target.upstream)
                tried.append({"upstream": target.upstream, "status": response.status_code})
                continue
            # Success, or a 4xx that is the caller's fault (bad request) and would fail anywhere.
            breaker.record_success(target.upstream)
            tried.append({"upstream": target.upstream, "status": response.status_code})
            return response, target, tried
        return None, None, tried

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        client = authenticate(request)
        if client is None:
            return error(401, "Missing or invalid gateway API key.", "invalid_api_key")

        allowed, wait = limiter.allow(client.name, client.rpm)
        if not allowed:
            metrics.inc("gateway_rate_limited_total", client=client.name)
            headers = {"Retry-After": str(max(1, round(wait)))}
            return error(
                429, f"Rate limit of {client.rpm} requests per minute exceeded.", "rate_limit_exceeded", headers
            )

        try:
            body = await request.json()
        except ValueError:
            return error(400, "Request body is not valid JSON.", "invalid_json")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            return error(400, "'messages' must be a non-empty list.", "invalid_request")
        alias = body.get("model")
        if alias not in config.routes:
            known = ", ".join(sorted(config.routes))
            return error(404, f"Unknown model {alias!r}. Available: {known}.", "model_not_found")

        if config.redact_pii:
            body["messages"], found = redact_messages(body["messages"])
            for kind, n in found.items():
                if n:
                    metrics.inc("gateway_pii_redactions_total", n, kind=kind)

        stream = bool(body.get("stream"))
        # Cache entries are private to a client: one tenant's answers are never served to another.
        namespace = f"{client.name}/{alias}"
        use_cache = config.cache.enabled and cacheable(body) and "x-gateway-no-cache" not in request.headers
        if use_cache and (hit := cache.get(namespace, body)) is not None:
            metrics.inc("gateway_cache_hits_total", kind=hit.kind)
            metrics.inc("gateway_requests_total", client=client.name, model=alias, outcome="cache")
            headers = {"x-gateway-cache": hit.kind, "x-gateway-similarity": f"{hit.similarity:.3f}"}
            return JSONResponse(hit.response, headers=headers)

        response, target, tried = await route(alias, body, stream)
        fallbacks = sum(1 for t in tried[:-1] if "skipped" not in t) if response is not None else 0
        if fallbacks:
            metrics.inc("gateway_fallbacks_total", fallbacks, model=alias)
        if response is None or target is None:
            metrics.inc("gateway_requests_total", client=client.name, model=alias, outcome="unavailable")
            detail = {"message": "All upstreams failed.", "type": "gateway_error", "code": "upstream_unavailable"}
            return JSONResponse({"error": {**detail, "attempts": tried}}, status_code=502)

        headers = {k: v for k, v in response.headers.items() if k.lower() in _HOP_HEADERS}
        headers |= {
            "x-gateway-upstream": target.upstream,
            "x-gateway-attempts": str(len(tried)),
            "x-gateway-cache": "miss",
        }
        outcome = "ok" if response.status_code < 400 else "upstream_4xx"
        metrics.inc("gateway_requests_total", client=client.name, model=alias, outcome=outcome)

        if stream:
            # Relay bytes as they arrive; fallback is only possible before the first byte is sent.
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=headers,
                background=BackgroundTask(response.aclose),
            )

        content = response.content
        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                data = None
            if isinstance(data, dict):
                usage = data.get("usage") or {}
                labels = {"client": client.name, "model": alias}
                metrics.inc("gateway_tokens_total", usage.get("prompt_tokens", 0), direction="input", **labels)
                metrics.inc("gateway_tokens_total", usage.get("completion_tokens", 0), direction="output", **labels)
                if use_cache:
                    cache.put(namespace, body, data)
        return Response(content, status_code=response.status_code, headers=headers)

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        if authenticate(request) is None:
            return error(401, "Missing or invalid gateway API key.", "invalid_api_key")
        data = [{"id": alias, "object": "model", "owned_by": "gateway"} for alias in sorted(config.routes)]
        return JSONResponse({"object": "list", "data": data})

    @app.get("/healthz")
    async def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "cache_entries": len(cache),
            "upstreams": {name: breaker.state(name) for name in config.upstreams},
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus() -> str:
        return metrics.render()

    return app


def app_from_env() -> FastAPI:
    """Entry point for `uvicorn gateway.app:app_from_env --factory`."""
    return create_app(load_config(os.getenv("GATEWAY_CONFIG", "gateway.yaml")))
