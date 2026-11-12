"""Response cache with an exact tier and a semantic tier.

Semantic caching is the feature most likely to return a *wrong* answer, so a
semantic hit has to clear four bars, not one:

  1. same scope      - same model alias, same sampling params, and every message
                       before the final user turn byte-identical
  2. similarity      - cosine similarity of the final user turn >= threshold
  3. same numbers    - the digit sequences in both prompts are identical
                       ("order 1042" must never be answered with order 1043's reply)
  4. same negation   - one prompt may not negate what the other asserts

Requests with tools, streaming or n > 1 are never cached.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_WORD = re.compile(r"[a-z0-9']+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_NEGATION = frozenset("not no never without cannot can't don't doesn't isn't won't shouldn't".split())
_PARAMS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "response_format", "seed", "stop")


def embed(text: str, dim: int = 2048) -> np.ndarray:
    """Signed feature hashing of words and bigrams. Deterministic, no model download.
    Lexical by design: with a high threshold it only matches near-identical wording."""
    words = _WORD.findall(text.lower())
    vector = np.zeros(dim, dtype="float32")
    for feature in words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        vector[int.from_bytes(digest[:4], "little") % dim] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _text(content) -> str:
    """Message content is a string or a list of typed parts; only text parts are compared."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cacheable(body: dict) -> bool:
    messages = body.get("messages") or []
    return (
        not body.get("stream")
        and not body.get("tools")
        and not body.get("functions")
        and body.get("n", 1) == 1
        and bool(messages)
        and messages[-1].get("role") == "user"
        and bool(_text(messages[-1].get("content")).strip())
    )


@dataclass
class Entry:
    response: dict
    expires_at: float
    scope: str
    question: str
    vector: np.ndarray


@dataclass
class Hit:
    response: dict
    kind: str  # "exact" | "semantic"
    similarity: float = 1.0


class ResponseCache:
    def __init__(
        self,
        ttl_seconds: float = 3600,
        max_entries: int = 5000,
        semantic: bool = True,
        threshold: float = 0.92,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._ttl, self._max, self._semantic, self._threshold, self._clock = (
            ttl_seconds,
            max_entries,
            semantic,
            threshold,
            clock,
        )
        self._entries: OrderedDict[str, Entry] = OrderedDict()  # insertion/recency order -> LRU eviction

    @staticmethod
    def _keys(alias: str, body: dict) -> tuple[str, str, str]:
        messages = body["messages"]
        params = {k: body[k] for k in _PARAMS if k in body}
        scope = _digest([alias, params, messages[:-1]])
        question = _text(messages[-1].get("content"))
        return _digest([scope, messages[-1]]), scope, question

    def _alive(self, key: str) -> Entry | None:
        entry = self._entries.get(key)
        if entry is not None and entry.expires_at <= self._clock():
            del self._entries[key]
            return None
        return entry

    def get(self, alias: str, body: dict) -> Hit | None:
        exact_key, scope, question = self._keys(alias, body)
        entry = self._alive(exact_key)
        if entry is not None:
            self._entries.move_to_end(exact_key)
            return Hit(entry.response, "exact")
        if not self._semantic:
            return None

        vector, best_key, best = embed(question), None, 0.0
        for key in list(self._entries):
            candidate = self._alive(key)
            if candidate is None or candidate.scope != scope:
                continue
            similarity = float(vector @ candidate.vector)
            if similarity > best and self._safe_to_reuse(question, candidate.question):
                best_key, best = key, similarity
        if best_key is not None and best >= self._threshold:
            self._entries.move_to_end(best_key)
            return Hit(self._entries[best_key].response, "semantic", best)
        return None

    @staticmethod
    def _safe_to_reuse(a: str, b: str) -> bool:
        if _NUMBER.findall(a) != _NUMBER.findall(b):
            return False
        words_a, words_b = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
        return bool(words_a & _NEGATION) == bool(words_b & _NEGATION)

    def put(self, alias: str, body: dict, response: dict) -> None:
        exact_key, scope, question = self._keys(alias, body)
        self._entries[exact_key] = Entry(response, self._clock() + self._ttl, scope, question, embed(question))
        self._entries.move_to_end(exact_key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
