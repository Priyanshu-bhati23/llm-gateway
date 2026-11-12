from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Config

KEY = {"Authorization": "Bearer web-secret"}


def completion(
    text: str = "Hello!", model: str = "upstream-model", prompt_tokens: int = 12, completion_tokens: int = 3
) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class Chunks(httpx.AsyncByteStream):
    """An unread response body, delivered in pieces like a real streamed completion."""

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def sse_response(*chunks: bytes) -> httpx.Response:
    return httpx.Response(200, stream=Chunks(*chunks), headers={"content-type": "text/event-stream"})


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Upstreams:
    """Scriptable fake upstreams, keyed by host. Records every request it receives."""

    def __init__(self) -> None:
        self.behaviour: dict[str, object] = {}
        self.requests: list[tuple[str, dict, httpx.Headers]] = []

    def calls(self, host: str) -> int:
        return sum(1 for h, _, _ in self.requests if h == host)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.requests.append((host, json.loads(request.content), request.headers))
        behaviour = self.behaviour.get(host, 200)
        if isinstance(behaviour, list):
            behaviour = behaviour.pop(0) if len(behaviour) > 1 else behaviour[0]
        if callable(behaviour):
            behaviour = behaviour()
        if isinstance(behaviour, Exception):
            raise behaviour
        if isinstance(behaviour, httpx.Response):
            return behaviour
        if behaviour == 200:
            return httpx.Response(200, json=completion(f"answer from {host}"))
        return httpx.Response(behaviour, json={"error": {"message": f"upstream said {behaviour}"}})


def make_config(**overrides) -> Config:
    data = {
        "clients": [
            {"name": "web", "key_env": "TEST_KEY_WEB", "rpm": 1000},
            {"name": "batch", "key_env": "TEST_KEY_BATCH", "rpm": 2},
        ],
        "upstreams": {
            "primary": {"base_url": "https://primary.test/v1", "api_key_env": "TEST_UPSTREAM_PRIMARY"},
            "backup": {"base_url": "https://backup.test/v1", "api_key_env": "TEST_UPSTREAM_BACKUP"},
        },
        "routes": {"fast": [{"upstream": "primary", "model": "p-small"}, {"upstream": "backup", "model": "b-small"}]},
    }
    data.update(overrides)
    return Config.model_validate(data)


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("TEST_KEY_WEB", "web-secret")
    monkeypatch.setenv("TEST_KEY_BATCH", "batch-secret")
    monkeypatch.setenv("TEST_UPSTREAM_PRIMARY", "sk-primary")
    monkeypatch.setenv("TEST_UPSTREAM_BACKUP", "sk-backup")


@pytest.fixture
def upstreams() -> Upstreams:
    return Upstreams()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def gateway(upstreams, clock):
    """Factory: gateway(**config_overrides) -> TestClient."""

    def build(**overrides) -> TestClient:
        app = create_app(make_config(**overrides), transport=httpx.MockTransport(upstreams), clock=clock)
        return TestClient(app)

    return build


def ask(client: TestClient, content: str = "What is the refund window?", headers: dict | None = None, **body):
    payload = {"model": "fast", "messages": [{"role": "user", "content": content}], **body}
    return client.post("/v1/chat/completions", json=payload, headers=KEY if headers is None else headers)
