import pytest

from gateway.cache import ResponseCache, cacheable, embed

from .conftest import Clock, ask


def body(question, system="You are a support bot.", **extra):
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": question}], **extra}


ANSWER = {"choices": [{"message": {"role": "assistant", "content": "180 days"}}]}


def test_exact_hit_through_the_gateway(gateway, upstreams):
    client = gateway()
    first, second = ask(client), ask(client)
    assert first.headers["x-gateway-cache"] == "miss" and second.headers["x-gateway-cache"] == "exact"
    assert second.json() == first.json() and upstreams.calls("primary.test") == 1
    assert client.app.state.metrics.get("gateway_cache_hits_total", kind="exact") == 1


def test_cache_is_isolated_per_client(gateway, upstreams):
    client = gateway()
    ask(client)
    other = ask(client, headers={"Authorization": "Bearer batch-secret"})
    assert other.headers["x-gateway-cache"] == "miss" and upstreams.calls("primary.test") == 2


def test_semantic_hit_for_a_trivially_reworded_question(gateway, upstreams):
    client = gateway(cache={"semantic_threshold": 0.8})
    ask(client, "What is the refund window for card payments?")
    hit = ask(client, "what is the refund window for card payments")
    assert hit.headers["x-gateway-cache"] == "semantic" and float(hit.headers["x-gateway-similarity"]) > 0.8
    assert upstreams.calls("primary.test") == 1


def test_no_cache_header_bypasses_lookup_and_storage(gateway, upstreams):
    client = gateway()
    ask(client, headers={"Authorization": "Bearer web-secret", "x-gateway-no-cache": "1"})
    ask(client, headers={"Authorization": "Bearer web-secret", "x-gateway-no-cache": "1"})
    assert upstreams.calls("primary.test") == 2 and len(client.app.state.cache) == 0


def test_failed_responses_are_not_cached(gateway, upstreams):
    upstreams.behaviour["primary.test"] = [400, 200]
    client = gateway()
    assert ask(client).status_code == 400
    assert ask(client).headers["x-gateway-cache"] == "miss"


@pytest.mark.parametrize(
    ("stored", "asked"),
    [
        ("Where is order 1042?", "Where is order 1043?"),  # different entity, almost identical text
        ("Is the API available in the EU?", "Is the API not available in the EU?"),  # negation flips the answer
        ("Convert 100 USD to EUR", "Convert 1000 USD to EUR"),
    ],
)
def test_semantic_tier_refuses_lookalike_questions_with_different_meaning(stored, asked):
    cache = ResponseCache(threshold=0.5)  # deliberately permissive: the guards must still hold
    cache.put("fast", body(stored), ANSWER)
    assert float(embed(stored) @ embed(asked)) > 0.5  # similar enough to match on similarity alone
    assert cache.get("fast", body(asked)) is None


@pytest.mark.parametrize(
    "variant",
    [
        {"system": "You are a pirate."},  # different system prompt
        {"temperature": 0.9},  # different sampling
        {"response_format": {"type": "json_object"}},
    ],
)
def test_hits_require_the_same_scope(variant):
    cache = ResponseCache(threshold=0.5)
    cache.put("fast", body("What is the refund window?"), ANSWER)
    system = variant.pop("system", "You are a support bot.")
    assert cache.get("fast", body("What is the refund window?", system=system, **variant)) is None
    assert cache.get("other-alias", body("What is the refund window?")) is None


def test_earlier_conversation_turns_are_part_of_the_scope():
    cache = ResponseCache(threshold=0.5)
    history = [{"role": "user", "content": "I paid by card."}, {"role": "assistant", "content": "Noted."}]
    question = {"role": "user", "content": "How long will the refund take?"}
    cache.put("fast", {"messages": [*history, question]}, ANSWER)
    assert cache.get("fast", {"messages": [*history, question]}).kind == "exact"
    other = [{"role": "user", "content": "I paid by bank debit."}, {"role": "assistant", "content": "Noted."}]
    assert cache.get("fast", {"messages": [*other, question]}) is None


def test_entries_expire():
    clock = Clock()
    cache = ResponseCache(ttl_seconds=60, clock=clock)
    cache.put("fast", body("q one two three"), ANSWER)
    clock.advance(59)
    assert cache.get("fast", body("q one two three")) is not None
    clock.advance(2)
    assert cache.get("fast", body("q one two three")) is None and len(cache) == 0


def test_least_recently_used_entry_is_evicted():
    cache = ResponseCache(max_entries=2, semantic=False)
    cache.put("fast", body("alpha question"), ANSWER)
    cache.put("fast", body("beta question"), ANSWER)
    cache.get("fast", body("alpha question"))  # alpha is now the most recently used
    cache.put("fast", body("gamma question"), ANSWER)
    assert cache.get("fast", body("beta question")) is None
    assert cache.get("fast", body("alpha question")) and cache.get("fast", body("gamma question"))


@pytest.mark.parametrize(
    ("request_body", "expected"),
    [
        (body("hi"), True),
        (body("hi", stream=True), False),
        (body("hi", tools=[{"type": "function"}]), False),
        (body("hi", n=3), False),
        ({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "prefill"}]}, False),
        ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}, False),
        ({"messages": [{"role": "user", "content": [{"type": "text", "text": "describe"}]}]}, True),
    ],
)
def test_cacheable(request_body, expected):
    assert cacheable(request_body) is expected
