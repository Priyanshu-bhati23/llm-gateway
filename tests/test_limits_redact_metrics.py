import pytest
from pydantic import ValidationError

from gateway.config import load_config
from gateway.limits import RateLimiter
from gateway.redact import redact, redact_messages

from .conftest import Clock, ask, make_config

BATCH = {"Authorization": "Bearer batch-secret"}


def test_rate_limit_returns_429_with_retry_after_then_refills(gateway, upstreams, clock):
    client = gateway(cache={"enabled": False})  # the "batch" client is limited to 2 requests per minute
    assert [ask(client, headers=BATCH).status_code for _ in range(3)] == [200, 200, 429]
    limited = ask(client, headers=BATCH)
    assert limited.json()["error"]["code"] == "rate_limit_exceeded" and limited.headers["retry-after"] == "30"
    assert upstreams.calls("primary.test") == 2  # limited requests never reach an upstream

    clock.advance(30)  # one token refills every 30 s at 2 rpm
    assert ask(client, headers=BATCH).status_code == 200
    assert ask(client, headers=BATCH).status_code == 429


def test_rate_limits_are_per_client(gateway):
    client = gateway(cache={"enabled": False})
    for _ in range(3):
        ask(client, headers=BATCH)
    assert ask(client).status_code == 200  # "web" has its own bucket
    assert client.app.state.metrics.get("gateway_rate_limited_total", client="batch") == 1


def test_token_bucket_allows_a_burst_up_to_the_limit_only():
    clock = Clock()
    limiter = RateLimiter(clock)
    assert [limiter.allow("c", 5)[0] for _ in range(6)] == [True] * 5 + [False]
    clock.advance(3600)  # a long idle period must not bank more than one minute of tokens
    assert [limiter.allow("c", 5)[0] for _ in range(6)] == [True] * 5 + [False]


def test_pii_is_masked_before_the_request_leaves_the_gateway(gateway, upstreams):
    client = gateway(redact_pii=True)
    ask(client, "Email asha@example.com or call +91 98765 43210 about card 4242 4242 4242 4242.")
    sent = upstreams.requests[0][1]["messages"][0]["content"]
    assert sent == "Email [EMAIL] or call [PHONE] about card [CARD]."
    metrics = client.app.state.metrics
    assert [metrics.get("gateway_pii_redactions_total", kind=k) for k in ("email", "phone", "card")] == [1, 1, 1]


def test_redaction_is_off_by_default(gateway, upstreams):
    ask(gateway(), "Email asha@example.com")
    assert "asha@example.com" in upstreams.requests[0][1]["messages"][0]["content"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Order 1234567890123456 shipped", "Order 1234567890123456 shipped"),  # 16 digits but fails the Luhn check
        ("Version 3.14.159 released in 2026", "Version 3.14.159 released in 2026"),
        ("Pay with 4111-1111-1111-1111 now", "Pay with [CARD] now"),
        ("Reach me: first.last+tag@mail.example.co.in", "Reach me: [EMAIL]"),
    ],
)
def test_redaction_precision(text, expected):
    assert redact(text)[0] == expected


def test_redact_messages_handles_content_parts_and_does_not_mutate_input():
    original = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "mail a@b.io"}, {"type": "image_url", "image_url": {"url": "u"}}],
        }
    ]
    redacted, counts = redact_messages(original)
    assert redacted[0]["content"][0]["text"] == "mail [EMAIL]" and counts["email"] == 1
    assert original[0]["content"][0]["text"] == "mail a@b.io"


def test_token_usage_is_attributed_to_client_and_alias(gateway):
    client = gateway(cache={"enabled": False})
    ask(client)
    ask(client)
    ask(client, headers=BATCH)
    metrics = client.app.state.metrics
    assert metrics.get("gateway_tokens_total", client="web", model="fast", direction="input") == 24
    assert metrics.get("gateway_tokens_total", client="batch", model="fast", direction="output") == 3
    text = client.get("/metrics").text
    assert "# TYPE gateway_requests_total counter" in text
    assert 'gateway_requests_total{client="web",model="fast",outcome="ok"} 2' in text


def test_config_validation(tmp_path):
    with pytest.raises(ValidationError, match="unknown upstream 'missing'"):
        make_config(routes={"fast": [{"upstream": "missing", "model": "m"}]})
    with pytest.raises(ValidationError, match="has no targets"):
        make_config(routes={"fast": []})
    with pytest.raises(ValidationError, match="unique"):
        make_config(clients=[{"name": "a", "key_env": "X"}, {"name": "a", "key_env": "Y"}])


def test_example_config_is_valid():
    from pathlib import Path

    config = load_config(Path(__file__).parent.parent / "gateway.example.yaml")
    assert set(config.routes) == {"fast", "private"} and config.upstreams["local"].api_key == ""
