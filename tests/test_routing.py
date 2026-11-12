import httpx

from .conftest import KEY, ask, sse_response


def test_request_is_forwarded_with_the_upstream_model_and_key(gateway, upstreams):
    response = ask(gateway(), temperature=0.2)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "answer from primary.test"
    assert response.headers["x-gateway-upstream"] == "primary" and response.headers["x-gateway-attempts"] == "1"

    host, body, headers = upstreams.requests[0]
    assert host == "primary.test" and body["model"] == "p-small" and body["temperature"] == 0.2
    assert headers["authorization"] == "Bearer sk-primary"  # the upstream key, never the client's


def test_auth_is_required(gateway, upstreams):
    client = gateway()
    assert ask(client, headers={}).status_code == 401
    assert ask(client, headers={"Authorization": "Bearer nope"}).json()["error"]["code"] == "invalid_api_key"
    assert ask(client, headers={"Authorization": "Basic web-secret"}).status_code == 401
    assert client.get("/v1/models").status_code == 401
    assert upstreams.requests == []


def test_an_unset_client_key_never_matches_an_empty_token(gateway, monkeypatch):
    monkeypatch.delenv("TEST_KEY_BATCH")
    assert ask(gateway(), headers={"Authorization": "Bearer "}).status_code == 401


def test_models_endpoint_lists_aliases_not_upstream_models(gateway):
    data = gateway().get("/v1/models", headers=KEY).json()["data"]
    assert [m["id"] for m in data] == ["fast"]


def test_invalid_requests(gateway, upstreams):
    client = gateway()
    bad_json = client.post(
        "/v1/chat/completions", content=b"{nope", headers={**KEY, "content-type": "application/json"}
    )
    assert bad_json.json()["error"]["code"] == "invalid_json"
    assert client.post("/v1/chat/completions", json={"model": "fast", "messages": []}, headers=KEY).status_code == 400
    unknown = client.post(
        "/v1/chat/completions", json={"model": "gpt-9", "messages": [{"role": "user", "content": "x"}]}, headers=KEY
    )
    assert unknown.status_code == 404 and "Available: fast" in unknown.json()["error"]["message"]
    assert upstreams.requests == []


def test_falls_back_when_the_primary_errors(gateway, upstreams):
    upstreams.behaviour["primary.test"] = 503
    client = gateway()
    response = ask(client)
    assert response.status_code == 200 and response.headers["x-gateway-upstream"] == "backup"
    assert response.headers["x-gateway-attempts"] == "2"
    assert upstreams.requests[1][1]["model"] == "b-small"
    assert client.app.state.metrics.get("gateway_fallbacks_total", model="fast") == 1


def test_falls_back_on_network_errors_and_bad_upstream_credentials(gateway, upstreams):
    upstreams.behaviour["primary.test"] = [httpx.ConnectTimeout("timed out"), 401]
    client = gateway()
    assert ask(client, "first question").headers["x-gateway-upstream"] == "backup"
    assert ask(client, "second, unrelated question").headers["x-gateway-upstream"] == "backup"


def test_caller_errors_are_returned_without_fallback(gateway, upstreams):
    upstreams.behaviour["primary.test"] = 400
    response = ask(gateway())
    assert response.status_code == 400 and response.json()["error"]["message"] == "upstream said 400"
    assert upstreams.calls("backup.test") == 0  # a malformed request would fail there too


def test_all_upstreams_down_is_a_502_that_leaks_no_secrets(gateway, upstreams):
    upstreams.behaviour.update({"primary.test": 500, "backup.test": httpx.ConnectError("refused")})
    response = ask(gateway())
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "upstream_unavailable"
    assert error["attempts"] == [
        {"upstream": "primary", "status": 500},
        {"upstream": "backup", "error": "ConnectError"},
    ]
    assert "sk-" not in response.text


def test_circuit_breaker_skips_a_failing_upstream_then_probes_it(gateway, upstreams, clock):
    upstreams.behaviour["primary.test"] = 500
    client = gateway(cache={"enabled": False}, breaker={"failure_threshold": 3, "cooldown_seconds": 30})
    for _ in range(3):
        ask(client)
    assert upstreams.calls("primary.test") == 3
    assert client.get("/healthz").json()["upstreams"] == {"primary": "open", "backup": "closed"}

    ask(client)  # circuit open: primary is not even tried
    assert upstreams.calls("primary.test") == 3 and upstreams.calls("backup.test") == 4

    clock.advance(31)
    upstreams.behaviour["primary.test"] = 200
    assert client.get("/healthz").json()["upstreams"]["primary"] == "half-open"
    assert ask(client).headers["x-gateway-upstream"] == "primary"  # the probe succeeded
    assert client.get("/healthz").json()["upstreams"]["primary"] == "closed"


def test_failed_probe_reopens_the_circuit(gateway, upstreams, clock):
    upstreams.behaviour["primary.test"] = 500
    client = gateway(cache={"enabled": False}, breaker={"failure_threshold": 1, "cooldown_seconds": 10})
    ask(client)
    clock.advance(11)
    ask(client)  # probe fails
    assert upstreams.calls("primary.test") == 2
    ask(client)
    assert upstreams.calls("primary.test") == 2 and client.get("/healthz").json()["upstreams"]["primary"] == "open"


def test_streaming_is_relayed_and_never_cached(gateway, upstreams):
    chunks = (
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
        b"data: [DONE]\n\n",
    )
    upstreams.behaviour["primary.test"] = lambda: sse_response(*chunks)
    client = gateway()
    first = ask(client, stream=True)
    assert first.content == b"".join(chunks) and first.headers["content-type"].startswith("text/event-stream")
    ask(client, stream=True)
    assert upstreams.calls("primary.test") == 2 and len(client.app.state.cache) == 0


def test_streaming_falls_back_before_the_first_byte(gateway, upstreams):
    upstreams.behaviour["primary.test"] = 529
    upstreams.behaviour["backup.test"] = lambda: sse_response(b"data: [DONE]\n\n")
    response = ask(gateway(), stream=True)
    assert response.content == b"data: [DONE]\n\n" and response.headers["x-gateway-upstream"] == "backup"
